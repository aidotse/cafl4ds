"""P1.0.2 deployment-prototype de-risker harness entry point.

Streams a configurable BDD driving diet (the :class:`~cafl4ds.data.regime.RegimeStream`) through a
config-selected backbone (``backbone=mae|je``) and logs a Phase-0-informed **multivariate** health
signal — a label-free vector plus a labelled canary channel, every signal annotated with its Phase-0
calibrated status (:mod:`cafl4ds.health_trust`) — into a self-describing deploy corpus. It reads **no
degradation**: the acceptance bar is the two-tier validation (Tier-A wiring check + Tier-B
sanity-of-readings). The corpus + the exact config are the primary hand-off; the pipeline is the means.

Examples:
    Tier-A wiring check (fresh clone, no dataset needed)::

        uv run python scripts/run_deploy_harness.py data=attr_synthetic img_size=32 eval_every=3

    The default BDD diet, joint-embedding backbone::

        uv run python scripts/run_deploy_harness.py data=bdd bdd_root=/abs/path/to/bdd backbone=je

    MAE backbone, downscaled for a low-memory box, gate arms off::

        uv run python scripts/run_deploy_harness.py data=bdd bdd_root=/abs/p backbone=mae img_size=128 gate_arms=false
"""

import copy
import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from cafl4ds import deploy, harness
from cafl4ds.health_trust import CANARY_SIGNALS, DEFAULT_LABEL_FREE_SIGNALS, Backbone, backbone_family
from cafl4ds.ssl.base import SSLMethod, apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")


def _live_method(config: DictConfig) -> SSLMethod:
    """Instantiate the selected backbone and apply its initialization."""
    method: SSLMethod = instantiate(config.ssl, encoder=instantiate(config.encoder))
    checkpoint = config.init.checkpoint
    if config.init.mode == "pretrained" and not checkpoint:
        checkpoint = str(Path(to_absolute_path(config.pretrain_dir)) / f"{method.name}.pt")
    apply_encoder_init(method.encoder, config.init.mode, checkpoint)
    return method


def _run_live(config: DictConfig, out_dir: Path, name: str) -> harness.Arm:
    """Run the live backbone over the regime diet, logging the full multivariate health vector."""
    torch.manual_seed(int(config.seed))
    method = _live_method(config)
    if backbone_family(method.name) is not Backbone(str(config.family)):
        logger.warning(
            f"config family={config.family!r} disagrees with the instantiated backbone "
            f"{method.name!r} ({backbone_family(method.name).value}) — check the backbone override."
        )
    stream = instantiate(config.stream)
    logger.info(
        f"deploy '{name}': backbone={method.name}, {stream.num_eras} eras, {len(stream)} batches, dev={config.device}"
    )
    return harness.run_stream_arm(
        name=f"{name}_live",
        role="live",
        method=method,
        stream=stream,
        optimizer=instantiate(config.optim, params=method.parameters()),
        selection_filter=instantiate(config.filter),
        monitor=instantiate(config.monitor, eval_sets=stream.eval_sets),
        out_dir=out_dir,
        eval_every=config.eval_every,
        device=config.device,
    )


def _run_gate_arms(
    config: DictConfig, out_dir: Path, name: str, grid: list[tuple[int, int]]
) -> tuple[harness.Arm, harness.Arm]:
    """Run the optional PC (manufactured collapse) and B5 (frozen floor) gate arms on the diet.

    Args:
        config: The composed deploy config.
        out_dir: Directory the arm logs are written to.
        name: The run name prefix.
        grid: The live arm's ``(step, era)`` checkpoints — B5 is stamped onto them (measured only).

    Returns:
        The ``(pc, b5)`` gate arms.
    """
    # B5: the frozen, init-matched floor — re-seeded to the live arm's init, measured only.
    torch.manual_seed(int(config.seed))
    b5_method = _live_method(config)
    b5 = harness.run_frozen_arm(
        name=f"{name}_b5",
        frozen_method=copy.deepcopy(b5_method),
        monitor=instantiate(config.monitor, eval_sets=instantiate(config.stream).eval_sets),
        grid=grid,
        device=config.device,
    )
    # PC: the manufactured-collapse control (anti-collapse-off SimSiam), from scratch, multi-epoch.
    torch.manual_seed(int(config.seed))
    pc_method = instantiate(config.pc.ssl, encoder=instantiate(config.encoder))
    apply_encoder_init(pc_method.encoder, "from_scratch")
    pc_stream = instantiate(config.stream)
    pc = harness.run_stream_arm(
        name=f"{pc_method.name}_pc",
        role="pc",
        method=pc_method,
        stream=pc_stream,
        optimizer=instantiate(config.optim, params=pc_method.parameters()),
        selection_filter=instantiate(config.filter),
        monitor=instantiate(config.monitor, eval_sets=pc_stream.eval_sets),
        out_dir=out_dir,
        eval_every=config.eval_every,
        epochs=config.pc.epochs,
        device=config.device,
    )
    return pc, b5


def _run_leak_check(config: DictConfig, out_dir: Path, name: str) -> dict[str, Any]:
    """Repeat the live arm over the diet, sampling RSS per pass — the Tier-A flat-memory check."""
    torch.manual_seed(int(config.seed))
    method = _live_method(config)
    optimizer = instantiate(config.optim, params=method.parameters())
    rss_samples: list[float] = []
    for p in range(int(config.long_horizon_passes)):
        stream = instantiate(config.stream)
        harness.run_stream_arm(
            name=f"{name}_leak_p{p}",
            role="live",
            method=method,  # one model, repeated passes = a longer horizon
            stream=stream,
            optimizer=optimizer,
            selection_filter=instantiate(config.filter),
            monitor=instantiate(config.monitor, eval_sets=stream.eval_sets),
            out_dir=out_dir,
            eval_every=config.eval_every,
            device=config.device,
        )
        rss_samples.append(harness.rss_mb())
        logger.info(f"leak-check pass {p}: RSS {rss_samples[-1]:.1f} MiB")
    return harness.leak_report(rss_samples, growth_frac_max=float(config.long_horizon_growth_frac_max))


def _build(config: DictConfig, out_dir: Path) -> dict[str, Any]:
    """Run the arms, read the multivariate signal, and assemble the deploy corpus."""
    name = config.run_name or f"{config.family}_deploy"
    live = _run_live(config, out_dir, name)
    family = Backbone(str(config.family))

    grid = harness.health_grid(live)
    pc, b5 = _run_gate_arms(config, out_dir, name, grid) if config.gate_arms else (None, None)
    leak = _run_leak_check(config, out_dir, name) if config.long_horizon else None

    log_signals = list(config.log_signals) if config.log_signals else None
    expected = log_signals or [*DEFAULT_LABEL_FREE_SIGNALS[family], *CANARY_SIGNALS]

    header = {
        "backbone": config.family,
        "I": config.init.mode,
        "diet": "regime",
        "block_size": config.stream.block_size,
        "A": _target_leaf(config.filter),
        "seed": int(config.seed),
        "img_size": int(config.img_size),
        "device": str(config.device),
        "gate_arms": bool(config.gate_arms),
    }
    report = deploy.build_deploy_report(
        config_header=header,
        family=family,
        live=live,
        expected_signals=list(expected),
        pc=pc,
        b5=b5,
        leak=leak,
        log_signals=log_signals,
    )
    _log_verdict(report)
    return report


def _target_leaf(node: DictConfig) -> str:
    """The filter's short name — the ``_target_`` class leaf (the ``A`` factor label)."""
    return str(node.get("_target_", "filter")).rsplit(".", 1)[-1]


def _log_verdict(report: dict[str, Any]) -> None:
    """Log the human-readable Tier-A validation verdict."""
    val, chan = report["validation"], report["channels"]
    logger.info(
        "P1.0.2 deploy corpus (Tier-A wiring):\n"
        f"  channels: label_free={len(chan['label_free'])} signals, canary={len(chan['canary'])} signals\n"
        f"  channel_complete={val['channel_complete']}  trust_complete={val['trust_complete']}  "
        f"finite={val['finite']}  memory_flat={val['memory_flat']}\n"
        f"  Tier-A passed: {val['passed']}"
    )


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="deploy")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run the P1.0.2 deployment harness and write the deploy corpus."""
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    report = _build(config, out_dir)

    (out_dir / config.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"wrote deploy corpus to {out_dir / config.report}")
    if not report["validation"]["passed"]:
        logger.error("Tier-A validation did NOT pass — the deployment pipeline is mis-wired; investigate before use.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
