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
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cafl4ds import branch, deploy, deploy_corpus, harness, optim_groups, warmup
from cafl4ds.data.attributes import AttributeSource
from cafl4ds.data.regime import RegimeStream
from cafl4ds.data.streams import StreamBatch
from cafl4ds.health_trust import CANARY_SIGNALS, DEFAULT_LABEL_FREE_SIGNALS, Backbone, backbone_family
from cafl4ds.leg_checkpoints import AveragingStream, CheckpointingStream
from cafl4ds.monitor import HealthMonitor, group_key
from cafl4ds.ssl.base import SSLMethod, apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")


def _live_method(config: DictConfig) -> SSLMethod:
    """Instantiate the selected backbone and apply its initialization.

    Precedence: a ``well`` (a full-method warm well from ``scripts/warm_well.py``) resumes the model
    intact — encoder *and* head — so a warm drive starts from genuine competence with no fresh-head
    transient (P0.3.9). With no well, the ``init`` factor applies (``from_scratch`` / encoder-only
    ``pretrained``) exactly as the Phase-0 harness.
    """
    method: SSLMethod = instantiate(config.ssl, encoder=instantiate(config.encoder))
    checkpoint = config.init.checkpoint
    if config.init.mode == "pretrained" and not checkpoint:
        checkpoint = str(Path(to_absolute_path(config.pretrain_dir)) / f"{method.name}.pt")
    apply_encoder_init(method.encoder, config.init.mode, checkpoint)
    if config.get("well"):
        warmup.load_well(method, to_absolute_path(str(config.well)))
    return method


def _anchor_method(config: DictConfig) -> SSLMethod:
    """Build the frozen drift-anchor model — the selected backbone resumed from the ``anchor`` well.

    It is placed on ``config.device``, the device the live model is read on, so drift measures the
    model moving and carries no CPU-vs-accelerator numerical floor.
    """
    anchor: SSLMethod = instantiate(config.ssl, encoder=instantiate(config.encoder))
    warmup.load_well(anchor, to_absolute_path(str(config.anchor)))
    anchor.to(config.device)
    return anchor


def _branch_stream(
    config: DictConfig, stream: RegimeStream, method: SSLMethod, monitor: HealthMonitor, out_dir: Path
) -> tuple[Iterable[StreamBatch], dict[str, Any] | None]:
    """The stream the live arm drives, and the P1.1.1 branch record (``None`` on a plain drive).

    A plain drive streams the walk (or its IID control). A **branch** (P1.1.1) starts from a saved
    checkpoint instead of the well (``resume.checkpoint``): it either continues the drive from
    ``resume.step`` — optionally after damaging the encoder (``damage.*``) — or, with
    ``recovery_steps``, runs the recovery pass over the well's stationary pool. A branch reads the
    canary just before and just after the damage (identical at dose 0), so the immediate dent is on
    record and every dose consumes the same RNG. ``checkpoint_legs`` saves every remaining leg
    boundary (``true``), only the end state (``final``), or before each ``checkpoint_steps`` step and at
    the end (``steps``).
    """
    resume: DictConfig = config.get("resume") or OmegaConf.create({})
    damage: DictConfig = config.get("damage") or OmegaConf.create({})
    recovery = config.get("recovery_steps")
    record: dict[str, Any] | None = None
    fed: Iterable[StreamBatch] = stream
    start = 0
    if resume.get("checkpoint"):
        path = to_absolute_path(str(resume.checkpoint).format(seed=int(config.seed)))
        warmup.load_well(method, path)
        start = 0 if recovery else int(resume.get("step") or 0)
        method.to(config.device)
        pre = monitor.measure(method, start - 1)
        dent = branch.damage_top_blocks(
            method.encoder,
            str(damage.get("mode", "none")),
            int(damage.get("k", 0)),
            scale=float(damage.get("scale", 0.0)),
            seed=int(damage.get("seed", 0)),
        )
        post = monitor.measure(method, start - 1)
        record = {
            "checkpoint": path,
            "phase": "recover" if recovery else "drive",
            "start_step": start,
            "damage": {**{k: damage.get(k) for k in ("mode", "k", "scale", "seed")}, **dent},
            "pre": pre,
            "post": post,
        }
        logger.info(f"branch from {path}: {record['phase']} at step {start}, damage={record['damage']}")
        fed = branch.recovery_batches(stream, int(recovery)) if recovery else branch.ResumedStream(stream, start)
    elif recovery:
        raise ValueError("recovery_steps needs resume.checkpoint (the state to recover from).")
    legs = config.get("checkpoint_legs")
    if legs:
        steps = [] if str(legs) == "final" or recovery else [s for s in stream.leg_boundary_steps if s > start]
        if str(legs) == "steps":  # P1.2.1: a handful of chosen steps — a long drive has too many legs to keep
            # a value in (0, 1) is a fraction of the drive's batches (its exact length is known only here)
            chosen = [round(float(s) * len(stream)) if 0 < float(s) < 1 else int(s) for s in config.checkpoint_steps]
            steps = [s for s in chosen if s > start]
        fed = CheckpointingStream(fed, method, steps, out_dir / "checkpoints" / f"s{config.seed}")
    return fed, record


def _run_live(
    config: DictConfig, out_dir: Path, name: str, source: AttributeSource
) -> tuple[harness.Arm, RegimeStream, dict[str, Any] | None]:
    """Run the live backbone over the regime diet, logging the full multivariate health vector.

    Returns the completed arm, the stream it drove (its era names, leg composition, realized
    familiarity and probe sizes describe the corpus), and the branch record (P1.1.1; ``None`` on a
    plain drive). ``source`` is the one decoded attribute source threaded through every arm, so the
    corpus is decoded once per drive (not re-decoded per arm).

    Raises:
        ValueError: If the ``well`` or ``anchor`` was reserved with a different split than this
            drive (:func:`cafl4ds.warmup.check_split`).
    """
    torch.manual_seed(int(config.seed))
    method = _live_method(config)
    if backbone_family(method.name) is not Backbone(str(config.family)):
        logger.warning(
            f"config family={config.family!r} disagrees with the instantiated backbone "
            f"{method.name!r} ({backbone_family(method.name).value}) — check the backbone override."
        )
    stream = instantiate(config.stream, source=source)
    # P1.2.4's off-domain drives resume a BDD well on another source, whose split cannot match: the one
    # deliberate exception (resume_check_split=false).
    for path in {
        str(p) for p in (config.get("well"), config.get("anchor")) if p and config.get("resume_check_split", True)
    }:
        warmup.check_split(to_absolute_path(path), stream.split_fingerprint())
    logger.info(
        f"deploy '{name}': backbone={method.name}, order={stream.order}, {stream.num_eras} eras, "
        f"{len(stream)} batches, dev={config.device}"
    )
    monitor = instantiate(config.monitor, eval_sets=stream.eval_sets)
    if config.get("anchor"):
        with torch.random.fork_rng():  # building the anchor must not shift the arm's RNG draws
            monitor.anchor_drift(_anchor_method(config))
    fed, branched = _branch_stream(config, stream, method, monitor, out_dir)
    if config.get("ema"):  # P1.2.4: a weight average beside the drive, saved as checkpoints/s<seed>/ema_final.pt
        fed = AveragingStream(
            fed, method, float(config.ema), out_dir / "checkpoints" / f"s{config.seed}" / "ema_final.pt"
        )
    optimizer = _optimizer(config, method)
    schedule = config.get("schedule")
    arm = harness.run_stream_arm(
        name=f"{name}_live",
        role="live",
        method=method,
        stream=fed,
        optimizer=optimizer,
        selection_filter=instantiate(config.filter),
        monitor=monitor,
        out_dir=out_dir,
        eval_every=config.eval_every,
        scheduler=instantiate(schedule, optimizer=optimizer, total_steps=len(stream)) if schedule else None,
        device=config.device,
    )
    return arm, stream, branched


def _optimizer(config: DictConfig, method: SSLMethod) -> torch.optim.Optimizer:
    """The live arm's optimizer: one group, or P1.2.2's protections (``param_groups``) when asked for."""
    groups = config.get("param_groups") or {}
    if groups.get("layer_decay") is None and not groups.get("no_decay"):
        return instantiate(config.optim, params=method.parameters())
    params = optim_groups.param_groups(
        method,
        lr=float(config.optim.lr),
        weight_decay=float(config.optim.weight_decay),
        layer_decay=groups.get("layer_decay"),
        no_decay=bool(groups.get("no_decay")),
    )
    return instantiate(config.optim, _partial_=True)(params)  # plain dicts: instantiate would wrap them in DictConfig


def _probe_query_counts(stream: RegimeStream) -> dict[str, int]:
    """Query images behind each canary reading: ``pooled``, plus one per probe group (its metric suffix)."""
    query = stream.eval_sets.probe_query
    counts = {"pooled": int(query.labels.shape[0])}
    if query.groups is not None:
        for gid in sorted(set(query.groups.tolist())):
            name = group_key(stream.eval_sets.group_names.get(gid, str(gid)))
            counts[name] = int((query.groups == gid).sum().item())
    return counts


def _run_gate_arms(
    config: DictConfig, out_dir: Path, name: str, grid: list[tuple[int, int]], source: AttributeSource
) -> tuple[harness.Arm, harness.Arm]:
    """Run the optional PC (manufactured collapse) and B5 (frozen floor) gate arms on the diet.

    Args:
        config: The composed deploy config.
        out_dir: Directory the arm logs are written to.
        name: The run name prefix.
        grid: The live arm's ``(step, era)`` checkpoints — B5 is stamped onto them (measured only).
        source: The one decoded attribute source (shared with the live arm — the gate streams reuse
            its cached decode rather than re-decoding the corpus).

    Returns:
        The ``(pc, b5)`` gate arms.
    """
    # B5: the frozen, init-matched floor — re-seeded to the live arm's init, measured only.
    torch.manual_seed(int(config.seed))
    b5_method = _live_method(config)
    b5 = harness.run_frozen_arm(
        name=f"{name}_b5",
        frozen_method=copy.deepcopy(b5_method),
        monitor=instantiate(config.monitor, eval_sets=instantiate(config.stream, source=source).eval_sets),
        grid=grid,
        device=config.device,
    )
    # PC: the manufactured-collapse control (anti-collapse-off SimSiam), from scratch, multi-epoch.
    torch.manual_seed(int(config.seed))
    pc_method = instantiate(config.pc.ssl, encoder=instantiate(config.encoder))
    apply_encoder_init(pc_method.encoder, "from_scratch")
    pc_stream = instantiate(config.stream, source=source)
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


def _run_leak_check(config: DictConfig, out_dir: Path, name: str, source: AttributeSource) -> dict[str, Any]:
    """Repeat the live arm over the diet, sampling RSS per pass — the Tier-A flat-memory check."""
    torch.manual_seed(int(config.seed))
    method = _live_method(config)
    optimizer = instantiate(config.optim, params=method.parameters())
    rss_samples: list[float] = []
    for p in range(int(config.long_horizon_passes)):
        stream = instantiate(config.stream, source=source)
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


def _build(
    config: DictConfig, out_dir: Path, seed: int, source: AttributeSource | None = None
) -> tuple[dict[str, Any], dict[int, str], dict[int, Any]]:
    """Run the arms for one seed, read the multivariate signal, and assemble the deploy report."""
    config.seed = seed  # drive per-seed determinism (the held-out reservation is fixed by canary_seed)
    name = f"{config.run_name or f'{config.family}_deploy'}_s{seed}"
    if source is None:
        source = instantiate(config.data)  # decode once, thread through every arm (cached — see A2)
    live, stream, branched = _run_live(config, out_dir, name, source)
    family = Backbone(str(config.family))

    grid = harness.health_grid(live)
    pc, b5 = _run_gate_arms(config, out_dir, name, grid, source) if config.gate_arms else (None, None)
    leak = _run_leak_check(config, out_dir, name, source) if config.long_horizon else None

    log_signals = list(config.log_signals) if config.log_signals else None
    expected = log_signals or [*DEFAULT_LABEL_FREE_SIGNALS[family], *CANARY_SIGNALS]

    # Tier-B sanity-of-readings inputs: chance is 1/(#canary scenes); the loose plausibility floor
    # comes from the `tier_b` config block. Computed on the live series, kept separate from the
    # Tier-A `passed` wiring verdict.
    canary_chance = (1.0 / source.num_canary_classes) if source.num_canary_classes else None
    tier_b = {k: float(v) for k, v in config.tier_b.items()} if config.get("tier_b") else None

    header = {
        "backbone": config.family,
        "I": config.init.mode,
        "warm": bool(config.get("well")),
        "diet": "regime",
        "order": str(config.stream.get("order", "walk")),
        "block_size": config.stream.block_size,
        "stratified_canary": bool(config.stream.get("stratify_canary", False)),
        "well_fraction": float(config.stream.get("well_fraction", 0.0)),
        "well_overlap": float(config.stream.get("well_overlap", 0.0)),
        "anchor": (str(config.anchor) if config.get("anchor") else None),
        "familiarity": stream.familiarity,
        "n_probe_query": _probe_query_counts(stream),
        "A": _target_leaf(config.filter),
        "seed": int(seed),
        "img_size": int(config.img_size),
        "device": str(config.device),
        "gate_arms": bool(config.gate_arms),
        "branch": branched,
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
        canary_chance=canary_chance,
        tier_b_thresholds=tier_b,
        study=str(config.study),
    )
    _log_verdict(report)
    return report, stream.era_names, stream.era_composition()


def _git_sha() -> str | None:
    """The build's git provenance from the hatch-vcs version local segment (no subprocess)."""
    try:
        local = version("cafl4ds").split("+", 1)
    except PackageNotFoundError:
        return None
    return local[1] if len(local) > 1 else None


def _target_leaf(node: DictConfig) -> str:
    """The filter's short name — the ``_target_`` class leaf (the ``A`` factor label)."""
    return str(node.get("_target_", "filter")).rsplit(".", 1)[-1]


def _log_verdict(report: dict[str, Any]) -> None:
    """Log the human-readable Tier-A wiring verdict and (when computed) the Tier-B sanity verdict."""
    val, chan = report["validation"], report["channels"]
    logger.info(
        f"{report['study']} deploy corpus (Tier-A wiring):\n"
        f"  channels: label_free={len(chan['label_free'])} signals, canary={len(chan['canary'])} signals\n"
        f"  channel_complete={val['channel_complete']}  trust_complete={val['trust_complete']}  "
        f"finite={val['finite']}  memory_flat={val['memory_flat']}\n"
        f"  Tier-A passed: {val['passed']}"
    )
    tier_b = report.get("tier_b")
    if tier_b is not None:
        rep, checks = tier_b["reported"], tier_b["checks"]
        logger.info(
            f"{report['study']} deploy corpus (Tier-B sanity-of-readings):\n"
            f"  rankme in [{rep['rankme_min']}, {rep['rankme_max']}] -> in_range={checks['rankme_in_range']}\n"
            f"  final {rep['drift_key']}={rep['final_drift']} -> accumulated={checks['drift_accumulated']}\n"
            f"  canary({rep['canary_key']}) mean={rep['canary_mean']} vs chance {rep['canary_chance']} -> "
            f"above_chance={checks['canary_above_chance']}\n"
            f"  Tier-B passed: {tier_b['passed']}"
        )


def run_deploy(config: DictConfig, out_dir: Path) -> dict[str, Path]:
    """Drive the seed ensemble, write the per-seed reports, and assemble the hand-off corpus.

    The Hydra-free core (so it is unit-testable on the synthetic source): runs :func:`_build` per
    seed, writes each seed's report, then unions them into the corpus under ``out_dir/corpus``.

    Args:
        config: The composed deploy config.
        out_dir: The run directory the ``reports/`` and ``corpus/`` land in.

    Returns:
        The written corpus paths (from :func:`cafl4ds.deploy_corpus.write_corpus`).
    """
    seeds = [int(s) for s in config.seeds] if config.get("seeds") else [int(config.seed)]
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    reports: list[tuple[int, dict[str, Any]]] = []
    era_names: dict[int, str] = {}
    composition: dict[int, Any] = {}
    # A source that does not draw from the seed (BDD — its reservation is fixed by canary_seed) is
    # decoded once for the whole ensemble, not per seed; a seeded one (the synthetic sources) is rebuilt.
    source = None if "seed" in config.data else instantiate(config.data)
    for seed in seeds:
        report, era_names, composition = _build(config, out_dir, seed, source)
        (reports_dir / f"{config.family}_deploy_s{seed}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        reports.append((seed, report))

    manifest = {
        "git_sha": _git_sha(),
        "n_seeds": len(seeds),
        "warm": bool(config.get("well")),
        "eval_every": int(config.eval_every),
    }
    paths = deploy_corpus.write_corpus(
        out_dir / "corpus", reports=reports, era_names=era_names, composition=composition, manifest=manifest
    )
    logger.info(f"wrote deploy corpus ({len(reports)} seed(s)) to {paths['manifest'].parent}")

    if not all(report["validation"]["passed"] for _, report in reports):
        logger.error("Tier-A validation did NOT pass for a seed — the pipeline is mis-wired; investigate before use.")
    return paths


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="deploy")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run the P1.0.2 deployment harness over the seed ensemble and write the hand-off corpus."""
    run_deploy(config, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
