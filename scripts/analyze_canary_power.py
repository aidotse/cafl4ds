"""P1.1 audit B2 — the canary's power for the walk-vs-IID test P1.1.0 actually ran.

P1.1.0 bounded the loss its canary could miss with a single-checkpoint, unadjusted *t*-test MDD
(:func:`cafl4ds.null_envelope.canary_power`). Its verdicts, though, come from the exact permutation
test of the maximum |*d*| over every checkpoint (:func:`cafl4ds.null_envelope.gap`), which pays for
its search across checkpoints with a larger critical *d*. This measures that test's power directly
(:func:`cafl4ds.null_envelope.injected_mdd`): 5 of the 10 healthy IID drives play the walk, a known
drop is subtracted from their canary readings, and the same 252-split test is run against the other
5. The drop it catches 80% of the time is the MDD, per canary channel and per drop shape:

* ``sustained`` — held over the whole drive;
* ``last_quarter`` — over the drive's last quarter only (a late loss);
* ``single`` — at the last reading only.

The readings are raw, as the null test read them. The report (``out``) holds, per backbone and
channel, the injected MDD per shape, and the *t*-test MDD on P1.1.0's 5 IID seeds beside it.

Example::

    uv run python scripts/analyze_canary_power.py
"""

import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from cafl4ds.jsonio import dumps_valid
from cafl4ds.null_envelope import canary_power, checkpoint_steps, injected_mdd, read_corpus, seed_matrix
from scripts.analyze_null import _n_query, _query_counts

logger.remove()
logger.add(sys.stdout, level="INFO")


def _pooled(dirs: list[str], bb: str) -> list[dict[str, Any]]:
    """The corpus rows of every dir in ``dirs`` (``{bb}`` filled in), pooled; the seeds must be disjoint."""
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for d in dirs:
        part = read_corpus(Path(to_absolute_path(d.format(bb=bb))) / "corpus" / "health.csv")
        seeds = {int(r["seed"]) for r in part}
        if seeds & seen:
            raise ValueError(f"seeds {sorted(seeds & seen)} appear twice in {dirs}.")
        seen |= seeds
        rows.extend(part)
    return rows


def read_backbone(bb: str, config: DictConfig) -> dict[str, Any]:
    """Every canary channel's injected MDD per shape, with P1.1.0's t-test MDD beside it."""
    rows = _pooled(list(config.seeds_dirs), bb)
    checkpoint_steps(rows)  # every seed on one grid
    t_dir = Path(to_absolute_path(str(config.t_mdd_dir).format(bb=bb)))
    t_rows = read_corpus(t_dir / "corpus" / "health.csv")
    counts = _query_counts(t_dir.parent, t_dir.name, config)
    out: dict[str, Any] = {"seeds": sorted({int(r["seed"]) for r in rows}), "channels": {}}
    for channel in config.arbiters:
        matrix = seed_matrix(rows, str(channel))
        t_mdd = canary_power(seed_matrix(t_rows, str(channel)), _n_query(str(channel), counts))["mdd"]
        injected = {
            str(shape): injected_mdd(
                matrix,
                str(shape),
                n_inject=int(config.n_inject),
                n_groups=int(config.n_groups),
                alpha=float(config.alpha),
                power=float(config.power),
                seed=int(config.seed),
            )
            for shape in config.shapes
        }
        out["channels"][str(channel)] = {
            "t_mdd_p110": t_mdd,
            "injected": injected,
            "ratio": {s: v["mdd"] / t_mdd if np.isfinite(v["mdd"]) else None for s, v in injected.items()},
        }
        logger.info(
            f"[{bb}] {channel:22s} t-MDD {t_mdd:.3f} | "
            + " | ".join(f"{s} {v['mdd']:.3f} ({v['mdd'] / t_mdd:.1f}x)" for s, v in injected.items())
        )
    return out


def analyze(config: DictConfig) -> dict[str, Any]:
    """Build the report (Hydra-free core)."""
    report = {
        "n_inject": int(config.n_inject),
        "alpha": float(config.alpha),
        "power": float(config.power),
        "backbones": {str(bb): read_backbone(str(bb), config) for bb in config.backbones},
    }
    out = Path(to_absolute_path(str(config.out)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps_valid(report) + "\n", encoding="utf-8")
    logger.info(f"wrote {out}")
    return report


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="canary_power")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Measure the canary's power for the max-over-checkpoints test by injection."""
    analyze(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
