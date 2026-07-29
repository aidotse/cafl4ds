r"""Phase-0 false-positive stress (P0.2.4) — the quiet-side calibration of the collapse suite.

P0.2/P0.2.1/P0.2.2/P0.2.3 all calibrate how a forced-collapse (PC) arm **separates** from a
healthy arm. None stresses the *quiet* side: the "healthy" arm is always a well-behaved SimSiam,
so the suite's **false-positive rate** — a detector firing on a healthy-but-atypical
representation — is untested. In the coupled active-learning loop a false fire mis-aims selection,
so this is the failure mode that most hurts.

This harness has **no PC arm**. It runs two **healthy** SimSiam arms (``anti_collapse=True``) over
the SAME session:

* **reference** — the canonical-healthy baseline (the base config; byte-identical to the positive
  control's healthy arm, so readings are directly comparable).
* **atypical** — the same healthy SimSiam pushed into a regime that *superficially resembles*
  collapse by the selected ``stressor`` (low-diversity data / heavy augmentation / undertraining),
  applied to this arm only via ``stressor.overrides`` merged over the base config.

The atypical arm is fed into the ``pc`` slot of :func:`~cafl4ds.metric_envelope.metric_envelope`,
so the existing separation machinery is reused unchanged. A **false fire** = the atypical arm
separates from the reference in the collapse direction past its bar. The verdict is **inverted**
relative to the positive control: the run PASSES iff **no** instrument fires (:func:`quiet_verdict`).

Unlike the positive control, a *fire here is a legitimate scientific outcome* (a documented
false-positive boundary of that instrument), not a wiring bug — so this script logs the fired
instruments but does **not** exit non-zero. See ``docs/experiments/phase0/P0.2.4.md``.

Examples:
    Default (STL-10, CPU), low-diversity stressor::

        uv run python scripts/false_positive_stress.py stressor=low_diversity

    Fast network-free smoke (synthetic has 100 imgs/class, so shrink the reservations + volume)::

        uv run python scripts/false_positive_stress.py data=synthetic img_size=16 epochs=4 \\
            stream.support_per_class=8 stream.query_per_class=8 stream.era_eval_per_class=5 \\
            monitor.run_knn=false monitor.run_linear=false
"""

import copy
import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cafl4ds.loop import StreamingLoop
from cafl4ds.metric_envelope import metric_envelope, quiet_verdict, render_envelope_table
from cafl4ds.run_log import RunLogger, read_run
from cafl4ds.ssl.base import apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")


def _run_healthy_arm(config: DictConfig, *, run_name: str, out_dir: Path) -> list[dict[str, Any]]:
    """Build and run one **healthy** arm (``anti_collapse=True``), returning its health series.

    Mirrors ``scripts/positive_control.py``'s ``_run_arm`` but hard-wires the healthy toggle — the
    variable here is the *config* (reference vs. stressor-perturbed), not the anti-collapse switch.
    The global seed is reset so the run is reproducible.

    Args:
        config: The (base or stressor-merged) composed config for this arm.
        run_name: Name recorded on the run log and used for its filename.
        out_dir: Directory the run log is written to.

    Returns:
        The per-checkpoint health records.
    """
    torch.manual_seed(config.seed)
    encoder = instantiate(config.encoder)
    method = instantiate(config.ssl, encoder=encoder, anti_collapse=True)
    apply_encoder_init(method.encoder, "from_scratch")

    stream = instantiate(config.stream)
    optimizer = instantiate(config.optim, params=method.parameters())
    monitor = instantiate(config.monitor, eval_sets=stream.eval_sets)

    batches_per_epoch = len(stream)
    eval_every = max(1, config.eval_every_epochs * batches_per_epoch)
    total_steps = config.epochs * batches_per_epoch
    scheduler = instantiate(config.schedule, optimizer=optimizer, total_steps=total_steps)

    run_log_path = out_dir / f"{run_name}.jsonl"
    run_logger = RunLogger(run_log_path, run_name=run_name)
    logger.info(
        f"arm '{run_name}': {stream.num_eras} eras, "
        f"{batches_per_epoch} batches x {config.epochs} epochs = {total_steps} steps"
    )

    loop = StreamingLoop(
        stream=stream,
        method=method,
        optimizer=optimizer,
        selection_filter=instantiate(config.filter),
        monitor=monitor,
        run_logger=run_logger,
        eval_every=eval_every,
        epochs=config.epochs,
        scheduler=scheduler,
        device=config.device,
    )
    loop.run()
    records = read_run(run_log_path)
    return [r for r in records if r.get("series") == "health"]


def _apply_stressor(config: DictConfig) -> DictConfig:
    """Return a copy of ``config`` with the selected stressor's overrides merged in (test arm).

    The stressor config carries an ``overrides`` mapping (e.g. ``stream.class_order``, ``ssl.*``
    augmentation knobs, ``epochs``/``optim.lr``); it is merged over a deep copy of the base config
    with struct mode off so *new* keys (a class subset, augmentation params absent from the base)
    are allowed. The reference arm is left untouched.
    """
    test = copy.deepcopy(config)
    OmegaConf.set_struct(test, False)
    overrides = config.stressor.get("overrides", {}) or {}
    return OmegaConf.merge(test, overrides)


def _evaluate_quiet_gate(config: DictConfig, envelope: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap :func:`quiet_verdict` with the run's stressor label and fire bars for the record."""
    verdict = quiet_verdict(envelope)
    verdict["stressor"] = config.stressor.name
    verdict["fire_ratio"] = config.fire_ratio
    verdict["fire_gap"] = config.fire_gap
    return verdict


def _render_quiet_summary(gate: dict[str, Any]) -> str:
    """Render the inverted (quiet) verdict — pass iff no instrument false-fired."""
    verdict = "PASS ✅ (quiet side holds)" if gate["passed"] else "FALSE FIRE ⚠️"
    lines = [
        f"FALSE-POSITIVE STRESS [stressor={gate['stressor']}, "
        f"bars ratio>={gate['fire_ratio']} / gap>={gate['fire_gap']}]: {verdict}"
    ]
    if gate["fired"]:
        for f in gate["fired"]:
            lines.append(
                f"  FALSE FIRE: {f['metric']} @ {f['surface']} — "
                f"separation {f['separation']:.2f} >= bar {f['threshold']} "
                f"(atypical-healthy tripped a collapse detector: a false-positive boundary)"
            )
    else:
        lines.append("  no instrument separated the atypical-healthy arm from the reference past its bar")
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="false_positive_stress")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run the reference + atypical healthy arms, then apply the inverted (quiet) verdict."""
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    reference = _run_healthy_arm(config, run_name="reference_healthy", out_dir=out_dir)
    test_config = _apply_stressor(config)
    atypical = _run_healthy_arm(test_config, run_name=f"atypical_{config.stressor.name}", out_dir=out_dir)

    # Feed the atypical arm into the `pc` slot: a false fire = it separates from the canonical
    # healthy reference in the collapse direction past its bar (same envelope machinery as P0.2.2).
    envelope = metric_envelope(atypical, reference, min_ratio=config.fire_ratio, min_gap=config.fire_gap)
    gate = _evaluate_quiet_gate(config, envelope)

    logger.info(_render_quiet_summary(gate))
    logger.info(
        "collapse-suite envelope (atypical-in-PC-slot vs. reference — separation + fire/quiet per "
        "instrument × surface):\n" + render_envelope_table(envelope)
    )

    (out_dir / "comparison.json").write_text(
        json.dumps({"gate": gate, "envelope": envelope, "atypical": atypical, "reference": reference}, indent=2),
        encoding="utf-8",
    )
    logger.info(f"wrote comparison + quiet gate + envelope to {out_dir / 'comparison.json'}")

    # NB: a false fire is a documented calibration outcome (a false-positive boundary), NOT a wiring
    # bug — so, unlike the positive control, this script does not exit non-zero on a fire.
    if not gate["passed"]:
        logger.warning(
            f"{gate['num_fired']} instrument(s) false-fired on the '{gate['stressor']}' regime — "
            "document as a known false-positive boundary (see docs/experiments/phase0/P0.2.4.md)."
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
