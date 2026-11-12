"""P1.1.4 — the reading foundation for P1.3's setup: P1.1's readers, re-calibrated on P1.3's arms.

``mode=bands`` (Part 1): per arm (``arms.<name>``: a backbone, walk and IID run dirs), P1.1.0's null and
P1.1.1's single-drive rule, with the method unchanged (:mod:`cafl4ds.null_envelope`,
:mod:`cafl4ds.reading_protocol`, and P1.1.1's own calibration code in ``scripts/analyze_reading.py``):

* ``null`` — walk against IID per instrument: effect sizes and the exact permutation test, the maximum
  over checkpoints taken inside each split;
* ``rules`` / ``panels`` — the bands fitted on the IID seeds, one threshold per panel, the panel's
  leave-one-out false alarms (bands *and* threshold refitted without each seed) and, given
  ``check_seeds``, its alarms on fresh IID seeds; the drive-level count (either panel firing) and a
  Wilson interval on each count;
* ``walk`` — the healthy walk read against the bands: which instruments fire, when, and the verdict;
* ``canary_power`` — P1.1.0's single-checkpoint canary noise and MDD per channel.

``mode=power`` (Part 4): per arm (``power.ladders``: P1.2.5's per-image ladder reads at every walk
checkpoint), task, rung and past slice (``power.past``: the slice and the checkpoint ``t0`` after the
stream left it), the loss from ``t0`` to each later checkpoint, its seed and image bootstrap noise, the
max-over-checkpoints test's MDD and the pool each target needs (:mod:`cafl4ds.referee_power`).

Example::

    uv run python scripts/analyze_p114.py root=/path/to/outputs mode=power
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig
from threadpoolctl import threadpool_limits

from cafl4ds import referee_power as rp
from cafl4ds.null_envelope import canary_power, checkpoint_steps, gap, read_corpus, signal_columns
from cafl4ds.reading_protocol import call, leave_one_out_fired
from scripts.analyze_adaptation import Ladder, _slice_mask, _slice_ok
from scripts.analyze_reading import _finite, _matrix, _settings, calibrate_rules, panels, read_arm, skipped_steps

logger.remove()
logger.add(sys.stdout, level="INFO")

# Context-like columns the null skips: they are not instruments (P1.1.0 read loss and grad norm as
# mechanical on this diet, but reports them; they stay in).
_NOT_SIGNALS = {"era_index", "is_transition", "steps_since_transition", "leg_ordinal"}


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The Wilson score interval for ``k`` successes in ``n`` trials (95% by default).

    Raises:
        ValueError: If ``n`` is not positive or ``k`` is outside ``[0, n]``.
    """
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n > 0; got k={k}, n={n}.")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _load(root: Path, dirs: list[str], seeds: list[int] | None) -> list[dict[str, Any]]:
    """The live rows of every run dir's corpus, pooled, restricted to ``seeds`` (``None`` = all)."""
    rows = []
    for d in dirs:
        csv = root / d / "corpus" / "health.csv"
        if not csv.is_file():
            raise FileNotFoundError(csv)
        rows += [r for r in read_corpus(csv) if r.get("arm") == "live"]
    if seeds is not None:
        rows = [r for r in rows if int(r["seed"]) in set(seeds)]
    return rows


def _seeds(rows: list[dict[str, Any]]) -> list[int]:
    return sorted({int(r["seed"]) for r in rows})


def _count(fired: list[bool]) -> dict[str, Any]:
    k, n = int(sum(fired)), len(fired)
    lo, hi = wilson_interval(k, n)
    return {"fired": k, "of": n, "rate": k / n, "wilson95": [lo, hi]}


def read_null(walk: list[dict[str, Any]], iid: list[dict[str, Any]], skip: set[int]) -> dict[str, Any]:
    """P1.1.0's walk-vs-IID null for every numeric instrument both arms log."""
    out = {}
    for signal in sorted(set(signal_columns(walk)) & set(signal_columns(iid)) - _NOT_SIGNALS):
        w, i = _matrix(walk, signal, skip), _matrix(iid, signal, skip)
        if np.isfinite(w).sum() == 0 or np.isfinite(i).sum() == 0:
            continue
        out[signal] = gap(w, i)
    return out


def read_bands(name: str, arm: DictConfig, config: DictConfig, root: Path) -> dict[str, Any]:
    """Part 1 for one arm: the null, the calibrated rules and their false alarms, and the walk's read."""
    bb = str(arm.backbone)
    iid = _load(root, list(arm.iid), list(arm.iid_seeds) if arm.get("iid_seeds") is not None else None)
    walk = _load(root, list(arm.walk), None)
    check = _load(root, list(arm.iid), list(arm.check_seeds)) if arm.get("check_seeds") else []
    signals, arbiters = list(config.signals[bb]), list(config.arbiters)
    skip = skipped_steps(root / str(arm.walk[0]), config)
    if checkpoint_steps(walk) != checkpoint_steps(iid):
        raise ValueError(f"{name}: walk and IID checkpoint grids differ.")
    rules = calibrate_rules(iid, signals, arbiters, config, skip)
    grid = checkpoint_steps(iid)
    entry: dict[str, Any] = {
        "backbone": bb,
        "iid_seeds": _seeds(iid),
        "walk_seeds": _seeds(walk),
        "check_seeds": _seeds(check),
        "n_checkpoints": len(grid),
        "skipped_steps": sorted(skip),
        "rules": {},
        "panels": {},
    }
    loo_any: list[bool] = []
    check_any: list[bool] = []
    for panel_name, panel in panels(iid, signals, arbiters).items():
        settings = _settings(panel, arbiters, config)
        loo = leave_one_out_fired({s: _matrix(iid, s, skip) for s in panel}, settings, float(config.alpha))
        held = [
            any(bool(call(_matrix(check, s, skip)[k], rules[s])["fired"]) for s in panel)
            for k in range(len(_seeds(check)))
        ]
        loo_any = [a or b for a, b in zip(loo_any, loo, strict=True)] if loo_any else list(loo)
        check_any = [a or b for a, b in zip(check_any, held, strict=True)] if check_any else list(held)
        for s in panel:
            entry["rules"][s] = {
                "panel": panel_name,
                "sd_median": float(np.median(rules[s].sd)),
                "sd_late": float(np.median(rules[s].sd[-len(grid) // 4 :])),
                "rho": rules[s].rho,
                "window": settings[s]["window"],
                "direction": rules[s].direction,
                "transform": rules[s].transform,
            }
        entry["panels"][panel_name] = {
            "signals": panel,
            "threshold": rules[panel[0]].threshold,
            "loo": _count(loo),
            "check": _count(held) if held else None,
            "out_of_sample": _count(list(loo) + held),
        }
    entry["drive_level"] = {"loo": _count(loo_any), "check": _count(check_any) if check_any else None}
    entry["walk"] = read_arm(walk, rules, grid, arbiters, skip)
    entry["null"] = read_null(walk, iid, skip)
    entry["canary_power"] = {
        a: canary_power(
            _matrix(iid, a, skip),
            int(config.n_query_pooled if a in ("knn_acc", "linear_acc") else config.n_query_stratum),
        )
        for a in arbiters
    }
    _log(name, entry)
    return entry


def _log(name: str, entry: dict[str, Any]) -> None:
    """Print each panel's threshold and false alarms, the walk's fires and the null's significant instruments."""
    lines = [f"[{name}] ({entry['backbone']}) IID seeds {entry['iid_seeds']}, {entry['n_checkpoints']} readings"]
    for panel_name, p in entry["panels"].items():
        lines.append(
            f"  panel {panel_name}: threshold {p['threshold']:.2f}, loo {p['loo']['fired']}/{p['loo']['of']}"
            + (f", check {p['check']['fired']}/{p['check']['of']}" if p["check"] else "")
        )
    lines.append(f"  drive-level loo {entry['drive_level']['loo']['fired']}/{entry['drive_level']['loo']['of']}")
    share = entry["walk"]["fired_share"]
    lines.append("  walk fired share: " + ", ".join(f"{s} {v:.1f}" for s, v in share.items() if v > 0))
    verdicts = [e["verdict"] + ("+persisting" if e["persisting"] else "") for e in entry["walk"]["per_seed"].values()]
    lines.append(f"  walk verdicts: {verdicts}")
    sig = {s: round(g["max_abs_d"], 2) for s, g in entry["null"].items() if g["p_max"] <= 0.05}
    lines.append(f"  null, walk vs IID at p_max <= 0.05: {sig}")
    logger.info("\n".join(lines))


def read_power(name: str, ladder_dir: Path, config: DictConfig) -> dict[str, Any]:
    """Part 4 for one arm: per past slice, task and rung, the max test's MDD now and the pool a target needs."""
    cfg = config.power
    ladder = Ladder(ladder_dir)
    seeds, order = list(cfg.seeds), list(cfg.checkpoints)
    out: dict[str, Any] = {}
    for task in cfg.tasks:
        labels = ladder.labels[task]
        classes = np.unique(labels)
        masks = {
            spec: keep
            for spec in cfg.past
            if _slice_ok(labels, keep := _slice_mask(ladder, task, spec), classes, int(cfg.min_per_class))
        }
        alpha = float(cfg.alpha) / max(len(masks), 1)
        for rung in cfg.rungs:
            rows = {cp: ladder.rows([f"walk_s{i}_{cp}" for i in seeds], f"{task}|{rung}") for cp in order}
            for spec, keep in masks.items():
                t0 = str(cfg.past[spec])
                after = order[order.index(t0) + 1 :]
                start, ends = rows[t0], [rows[cp] for cp in after]
                if start is None or any(e is None for e in ends):
                    continue
                noise = rp.change_noise(
                    start[:, keep], [e[:, keep] for e in ends if e is not None], labels[keep], n_boot=int(cfg.n_boot)
                )
                combined = rp.combined_noise(noise["seed_noise"], noise["image_noise"])
                shapes = {"sustained": np.ones(len(after)), "last_only": np.eye(len(after))[-1]}
                entry: dict[str, Any] = {
                    "t0": t0,
                    "after": after,
                    "n_images": int(keep.sum()),
                    "min_class": int(min((labels[keep] == c).sum() for c in classes)),
                    "alpha": alpha,
                    "observed_loss": noise["loss"].tolist(),
                    "sd_seed": noise["seed_noise"].std(axis=0).tolist(),
                    "sd_image": noise["image_noise"].std(axis=0).tolist(),
                    "mdd": {
                        k: rp.mdd(combined, s, alpha=alpha, power=float(cfg.power))["mdd"] for k, s in shapes.items()
                    },
                    "mdd_dense": rp.dense_grid_mdd(
                        float(combined.std(axis=0).max()),
                        int(cfg.dense_checkpoints),
                        alpha=alpha,
                        power=float(cfg.power),
                    ),
                    "pool_for": {
                        f"{target}@seeds_x{m}": rp.pool_for(
                            noise["seed_noise"],
                            noise["image_noise"],
                            shapes["sustained"],
                            float(target),
                            alpha=alpha,
                            seeds=float(m),
                            power=float(cfg.power),
                        )
                        for target in cfg.targets
                        for m in cfg.seed_multipliers
                    },
                }
                out[f"{task}|{rung}|{spec}"] = entry
    _log_power(name, out)
    return out


def _log_power(name: str, out: dict[str, Any]) -> None:
    lines = [f"[{name}] slice: n, observed loss (last), MDD sustained / last-only / dense-10, pool for targets"]
    for key, e in out.items():
        pools = ", ".join(f"{k} {v:.1f}" for k, v in e["pool_for"].items())
        lines.append(
            f"  {key:<62} n {e['n_images']:4d}  loss {e['observed_loss'][-1]:+.3f}  "
            f"MDD {e['mdd']['sustained']:.3f} / {e['mdd']['last_only']:.3f} / {e['mdd_dense']:.3f}  [{pools}]"
        )
    logger.info("\n".join(lines))


def analyze(config: DictConfig) -> dict[str, Any]:
    """Build the P1.1.4 report for ``config.mode`` (Hydra-free core)."""
    root = Path(to_absolute_path(str(config.root)))
    if config.mode == "bands":
        report = {name: read_bands(name, arm, config, root) for name, arm in config.arms.items()}
    elif config.mode == "power":
        with threadpool_limits(8):
            report = {name: read_power(name, root / str(d), config) for name, d in config.power.ladders.items()}
    else:
        raise ValueError(f"unknown mode {config.mode!r}.")
    out = Path(to_absolute_path(str(config.out)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_finite(report), indent=2, default=float, allow_nan=False), encoding="utf-8")
    logger.info(f"wrote {out}")
    return report


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="analyze_p114")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run one P1.1.4 analysis."""
    analyze(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
