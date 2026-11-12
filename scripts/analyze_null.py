"""P1.1.0 healthy-movement null — read the walk + matched-IID corpora into envelopes, gaps and power.

For every backbone and well-overlap level, pairs the regime-walk corpus with its matched IID control
and reports, per logged signal, how far the IID null moves and how far the walk sits from it
(:mod:`cafl4ds.null_envelope`), plus the canary's probe power (audit P1.0 C2). The overlap axis is the
familiarity sweep: the drive is identical at every level, only how much of it the well already saw
varies. Writes ``null_report.json`` and prints the per-signal summary.

Example::

    uv run python scripts/analyze_null.py root=outputs/p110
"""

import json
import sys
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from cafl4ds.health_trust import canary_base
from cafl4ds.null_envelope import (
    canary_power,
    checkpoint_steps,
    gap,
    movement,
    read_corpus,
    seed_matrix,
    signal_columns,
)

logger.remove()
logger.add(sys.stdout, level="INFO")


def _corpus(root: Path, name: str) -> list[dict[str, Any]] | None:
    """Read ``<root>/<name>/corpus/health.csv`` if it exists."""
    path = root / name / "corpus" / "health.csv"
    if not path.is_file():
        logger.warning(f"missing corpus: {path}")
        return None
    return read_corpus(path)


def _query_counts(root: Path, name: str, config: DictConfig) -> dict[str, int]:
    """Query images behind each canary reading, from the run's ``corpus/manifest.json``.

    The drive records them in its report header (``n_probe_query``: ``pooled`` plus one per probe
    group). Corpora written before that field existed fall back to the config's ``n_query_pooled`` /
    ``n_query_stratum``, applied to every group.
    """
    path = root / name / "corpus" / "manifest.json"
    counts = (
        json.loads(path.read_text(encoding="utf-8")).get("config", {}).get("n_probe_query") if path.is_file() else None
    )
    if counts:
        return {k: int(v) for k, v in counts.items()}
    logger.warning(f"{name}: no n_probe_query in the manifest — using the configured query counts")
    return {"pooled": int(config.n_query_pooled), "*": int(config.n_query_stratum)}


def _n_query(signal: str, counts: dict[str, int]) -> int:
    """The query count behind one canary signal (pooled, or its group's slice)."""
    base = canary_base(signal)
    if signal == base:
        return counts["pooled"]
    group = signal[len(f"{base}_") :]
    if group in counts:
        return counts[group]
    if "*" in counts:
        return counts["*"]
    raise KeyError(f"no query count recorded for the '{group}' group of {signal}")


def pair_report(
    walk_rows: list[dict[str, Any]], iid_rows: list[dict[str, Any]], *, alpha: float, n_query: dict[str, int]
) -> dict[str, Any]:
    """Per-signal null movement + walk-vs-IID gap, and canary power, for one walk/IID pair.

    Args:
        walk_rows: The walk corpus rows.
        iid_rows: The matched IID corpus rows.
        alpha: Family-wise level for the per-checkpoint significance share.
        n_query: Query images behind each canary reading — ``pooled``, plus one per group suffix
            (``"*"`` stands for every group).

    Returns:
        ``{"n_seeds", "n_checkpoints", "signals": {...}, "canary_power": {...}}``.

    Raises:
        ValueError: If the two corpora do not share a checkpoint step grid (not a matched pair).
    """
    if checkpoint_steps(walk_rows) != checkpoint_steps(iid_rows):
        raise ValueError("walk and IID corpora checkpoint at different steps — not a matched pair.")
    signals: dict[str, Any] = {}
    power: dict[str, Any] = {}
    for signal in sorted(set(signal_columns(walk_rows)) & set(signal_columns(iid_rows))):
        walk, iid = seed_matrix(walk_rows, signal), seed_matrix(iid_rows, signal)
        signals[signal] = {
            "iid_movement": movement(iid),
            "walk_movement": movement(walk),
            "gap": gap(walk, iid, alpha=alpha),
        }
        if canary_base(signal) is not None:
            power[signal] = canary_power(iid, _n_query(signal, n_query))
    first = seed_matrix(iid_rows, "loss")
    return {"n_seeds": first.shape[0], "n_checkpoints": first.shape[1], "signals": signals, "canary_power": power}


def _table(title: str, reports: dict[str, dict[str, Any]], field: str, sub: str) -> str:
    """Render one signal × level table of ``signals[signal][field][sub]``."""
    levels = list(reports)
    signals = sorted({s for r in reports.values() for s in r["signals"]})
    lines = [title, f"{'signal':<24}" + "".join(f"{lvl:>10}" for lvl in levels)]
    for signal in signals:
        cells = []
        for lvl in levels:
            entry = reports[lvl]["signals"].get(signal)
            cells.append(f"{entry[field][sub]:>10.3f}" if entry else f"{'-':>10}")
        lines.append(f"{signal:<24}" + "".join(cells))
    return "\n".join(lines)


def analyze(config: DictConfig) -> dict[str, Any]:
    """Build the full null report over the sweep (Hydra-free core, unit-testable)."""
    root = Path(to_absolute_path(str(config.root)))
    alpha = float(config.alpha)
    report: dict[str, Any] = {"warm": {}, "cold": {}}
    for bb in config.backbones:
        per_level: dict[str, Any] = {}
        for overlap in config.overlaps:
            walk, iid = _corpus(root, f"warm/{bb}_o{overlap}_walk"), _corpus(root, f"warm/{bb}_o{overlap}_iid")
            if walk is not None and iid is not None:
                counts = _query_counts(root, f"warm/{bb}_o{overlap}_iid", config)
                per_level[f"o={overlap}"] = pair_report(walk, iid, alpha=alpha, n_query=counts)
        report["warm"][bb] = per_level
        if per_level:
            logger.info("\n" + _table(f"[{bb} warm] walk-vs-IID max |d|", per_level, "gap", "max_abs_d"))
            logger.info("\n" + _table(f"[{bb} warm] permutation p (max over checkpoints)", per_level, "gap", "p_max"))
            logger.info(
                "\n"
                + _table(f"[{bb} warm] IID null movement (max |Δ| from start)", per_level, "iid_movement", "max_abs")
            )
    for bb in config.cold_backbones:
        walk, iid = _corpus(root, f"cold/{bb}_walk"), _corpus(root, f"cold/{bb}_iid")
        if walk is not None and iid is not None:
            counts = _query_counts(root, f"cold/{bb}_iid", config)
            report["cold"][bb] = pair_report(walk, iid, alpha=alpha, n_query=counts)
    out = Path(to_absolute_path(str(config.out)))
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"wrote {out}")
    return report


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="null_envelope")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Read the P1.1.0 sweep corpora into the null report."""
    analyze(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
