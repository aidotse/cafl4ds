"""Phase-0 positive control (P0.2) — the collapse-instrument CALIBRATION gate.

Runs, in **one session**, two arms of the SAME SimSiam over the SAME class-blocked STL-10
stream, differing only by SimSiam's ``anti_collapse`` toggle:

* **PC** — anti-collapse DISABLED (predictor removed + stop-gradient off). Its trivial global
  optimum maps every input to one constant vector, so collapse is *mathematically forced*
  (loss → −1, RankMe → floor ~1.0), independent of scale — which is why the toy regime is
  sufficient (and required) to calibrate the instruments in the regime we measure in.
* **healthy** — SimSiam intact (predictor + stop-gradient ON). Stays in its higher RankMe
  band (the ~2.5–3+ range the P0.1 SimSiam runs showed).

Both arms are **from-scratch** (NOT the pretrained checkpoint, which in P0.1 looked possibly
pre-collapsed). The toggle is the *only* difference: the seed is reset before building each
arm so the two encoders start bit-identical and the augmentation RNG stays in lockstep, so
any divergence is attributable to anti-collapse alone. The gate is the **contrast** between
the two RankMe curves (ice-water vs boiling-water for a thermometer), checked numerically —
see ``docs/experiments/phase0/positive_control.md``.

Examples:
    Default (STL-10, CPU)::

        uv run python scripts/positive_control.py

    Fast network-free smoke::

        uv run python scripts/positive_control.py data=synthetic img_size=16 eval_every=3

    On the Gaudi HPU (inside the container; see docs/developing.md)::

        DATA_MOUNT=/mnt/stl10 ./scripts/run_gaudi_dev.sh gaudi-env-cafl4ds:latest 0 \
            python scripts/positive_control.py device=hpu
"""

import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from cafl4ds.loop import StreamingLoop
from cafl4ds.run_log import RunLogger, read_run
from cafl4ds.ssl.base import apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")

_SPARK = "▁▂▃▄▅▆▇█"


def _run_arm(
    config: DictConfig, *, anti_collapse: bool, run_name: str, out_dir: Path
) -> tuple[list[dict[str, Any]], float]:
    """Build and run one arm of the positive control, returning its health series + loss floor.

    The global seed is reset here so both arms start from a bit-identical encoder init and
    draw the same augmentation sequence — the ``anti_collapse`` toggle is the only variable.

    Args:
        config: The composed ``positive_control`` config.
        anti_collapse: SimSiam anti-collapse toggle (``False`` = PC, ``True`` = healthy).
        run_name: Name recorded on the run log and used for its filename.
        out_dir: Directory the run log is written to.

    Returns:
        A ``(health_records, loss_floor)`` pair: the per-checkpoint health records, and the
        minimum SSL loss over *all* steps (robust to batch-to-batch noise — the "did the loss
        reach its −1 constant-solution floor?" signal).
    """
    torch.manual_seed(config.seed)  # identical init + augmentation RNG across arms
    encoder = instantiate(config.encoder)
    method = instantiate(config.ssl, encoder=encoder, anti_collapse=anti_collapse)
    apply_encoder_init(method.encoder, "from_scratch")  # PC must NOT load a warm start

    stream = instantiate(config.stream)  # same seed -> identical splits/order as the other arm
    optimizer = instantiate(config.optim, params=method.parameters())
    monitor = instantiate(config.monitor, eval_sets=stream.eval_sets)

    run_log_path = out_dir / f"{run_name}.jsonl"
    run_logger = RunLogger(run_log_path, run_name=run_name)
    logger.info(f"arm '{run_name}' (anti_collapse={anti_collapse}): {stream.num_eras} eras, {len(stream)} batches")

    loop = StreamingLoop(
        stream=stream,
        method=method,
        optimizer=optimizer,
        selection_filter=instantiate(config.filter),
        monitor=monitor,
        run_logger=run_logger,
        eval_every=config.eval_every,
        device=config.device,
    )
    loop.run()
    records = read_run(run_log_path)
    health = [r for r in records if r.get("series") == "health"]
    loss_floor = min(r["loss"] for r in records if r.get("series") == "loss")
    return health, loss_floor


def _spark(values: list[float], lo: float, hi: float) -> str:
    """Render values as a unicode sparkline scaled to the shared ``[lo, hi]`` range.

    Args:
        values: The series to render.
        lo: Lower bound of the shared scale (maps to the lowest block).
        hi: Upper bound of the shared scale (maps to the highest block).

    Returns:
        A one-line sparkline string.
    """
    span = hi - lo or 1.0
    return "".join(_SPARK[min(len(_SPARK) - 1, max(0, int((v - lo) / span * (len(_SPARK) - 1))))] for v in values)


def _combined_table(pc: list[dict[str, Any]], hc: list[dict[str, Any]]) -> str:
    """Render both arms' loss + RankMe side by side, one row per aligned checkpoint.

    Args:
        pc: The PC arm's health records.
        hc: The healthy arm's health records.

    Returns:
        A fixed-width table string.
    """
    cols = ("step", "era", "pc_loss", "pc_rankme", "hc_loss", "hc_rankme")
    lines = ["  ".join(f"{c:>12}" for c in cols), "  ".join("-" * 12 for _ in cols)]
    for p, h in zip(pc, hc, strict=False):
        row = [p["step"], p["era"], p["loss"], p["rankme"], h["loss"], h["rankme"]]
        lines.append("  ".join(f"{v:>12.4f}" for v in row))
    return "\n".join(lines)


def _evaluate_gate(
    config: DictConfig,
    pc: list[dict[str, Any]],
    hc: list[dict[str, Any]],
    pc_loss_floor: float,
    hc_loss_floor: float,
) -> dict[str, Any]:
    """Apply the numeric pass criterion (the gate) — a scale-free relative-separation contrast.

    Passes iff, at the bounded (P0.1) horizon: the PC's final RankMe drops to at most
    ``gate.pc_rankme_drop_frac`` of its OWN random-init RankMe (a large relative collapse); the
    PC's loss floor reaches ``<= gate.pc_loss_floor`` (rides to its −1 constant-solution floor —
    the "right reason" fingerprint); the intact control stays ``>= gate.healthy_rankme_min`` with
    a loss floor that never reaches the collapse floor; and the loss-floor gap (the DEVICE-ROBUST
    discriminator — the RankMe endpoint gap is within FP noise at this horizon) exceeds
    ``gate.min_loss_separation``.

    Args:
        config: The composed config (its ``gate`` block holds the thresholds).
        pc: The PC arm's health records.
        hc: The healthy arm's health records.
        pc_loss_floor: Minimum SSL loss the PC reached over all steps.
        hc_loss_floor: Minimum SSL loss the intact control reached over all steps.

    Returns:
        A dict of the measured numbers, per-condition booleans, and the overall ``passed``.
    """
    g = config.gate
    pc_init, pc_final = pc[0]["rankme"], pc[-1]["rankme"]
    hc_final = hc[-1]["rankme"]
    pc_drop_frac = pc_final / pc_init if pc_init else 1.0
    loss_separation = hc_loss_floor - pc_loss_floor

    checks = {
        "pc_collapses_relative": pc_drop_frac <= g.pc_rankme_drop_frac,
        "pc_right_reason": pc_loss_floor <= g.pc_loss_floor,
        "healthy_holds": hc_final >= g.healthy_rankme_min and hc_loss_floor > g.pc_loss_floor,
        "loss_separated": loss_separation >= g.min_loss_separation,
    }
    return {
        "pc_rankme_init": pc_init,
        "pc_rankme_final": pc_final,
        "pc_rankme_drop_frac": pc_drop_frac,
        "pc_loss_floor": pc_loss_floor,
        "healthy_rankme_final": hc_final,
        "healthy_loss_floor": hc_loss_floor,
        "loss_separation": loss_separation,
        "rankme_separation": hc_final - pc_final,  # reported only — device-fragile at this horizon
        "thresholds": {
            "pc_rankme_drop_frac": g.pc_rankme_drop_frac,
            "pc_loss_floor": g.pc_loss_floor,
            "healthy_rankme_min": g.healthy_rankme_min,
            "min_loss_separation": g.min_loss_separation,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="positive_control")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run both arms, render the contrast, and apply the numeric gate."""
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    hc, hc_loss_floor = _run_arm(config, anti_collapse=True, run_name="simsiam_healthy", out_dir=out_dir)
    pc, pc_loss_floor = _run_arm(config, anti_collapse=False, run_name="simsiam_pc", out_dir=out_dir)

    table = _combined_table(pc, hc)
    # Shared scale so the two RankMe sparklines are directly comparable.
    all_rankme = [r["rankme"] for r in pc] + [r["rankme"] for r in hc]
    lo, hi = min(all_rankme), max(all_rankme)
    curves = (
        f"RankMe curves (shared scale {lo:.2f}..{hi:.2f}):\n"
        f"  PC (collapse) : {_spark([r['rankme'] for r in pc], lo, hi)}\n"
        f"  healthy       : {_spark([r['rankme'] for r in hc], lo, hi)}\n"
        f"PC loss (toward -1 floor, scale -1..0):\n"
        f"  PC (collapse) : {_spark([r['loss'] for r in pc], -1.0, 0.0)}"
    )

    gate = _evaluate_gate(config, pc, hc, pc_loss_floor, hc_loss_floor)
    t = gate["thresholds"]
    verdict = "PASS ✅" if gate["passed"] else "FAIL ❌"
    summary = (
        f"POSITIVE-CONTROL GATE: {verdict}\n"
        f"  PC RankMe    {gate['pc_rankme_init']:.3f} (init) -> {gate['pc_rankme_final']:.3f} "
        f"= {gate['pc_rankme_drop_frac'] * 100:.1f}% of init "
        f"(<= {t['pc_rankme_drop_frac'] * 100:.0f}%?  {gate['checks']['pc_collapses_relative']})\n"
        f"  PC loss floor  = {gate['pc_loss_floor']:.4f} "
        f"(<= {t['pc_loss_floor']} -> rides to -1 constant-solution floor?  {gate['checks']['pc_right_reason']})\n"
        f"  healthy RankMe final = {gate['healthy_rankme_final']:.3f} (>= {t['healthy_rankme_min']}?), "
        f"loss floor = {gate['healthy_loss_floor']:.4f} (never reaches -1?)  -> holds?  "
        f"{gate['checks']['healthy_holds']}\n"
        f"  loss-floor separation (intact - PC) = {gate['loss_separation']:.3f} "
        f"(>= {t['min_loss_separation']}?  {gate['checks']['loss_separated']})\n"
        f"  [reported] RankMe endpoint gap = {gate['rankme_separation']:.3f} "
        f"(device-fragile at this horizon; not gated)"
    )

    logger.info("positive control — side-by-side (aligned checkpoints)\n" + table)
    logger.info(curves)
    logger.info(summary)

    (out_dir / "comparison.json").write_text(
        json.dumps({"gate": gate, "pc": pc, "healthy": hc}, indent=2), encoding="utf-8"
    )
    logger.info(f"wrote comparison + gate to {out_dir / 'comparison.json'}")

    if not gate["passed"]:
        logger.error(
            "Gate did NOT pass. Per P0.2: if the PC does not collapse the instruments (or not for the "
            "right reason), the instrument/wiring is suspect — investigate before trusting downstream numbers."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
