"""P1.1.4 — the reading foundation for P1.3's setup: P1.1's readers, re-calibrated on P1.3's arms.

``mode=bands`` (Part 1): per arm (``arms.<name>``: a backbone, walk and IID run dirs), P1.1.0's null and
P1.1.1's single-drive rule, with the method unchanged (:mod:`cafl4ds.null_envelope`,
:mod:`cafl4ds.reading_protocol`, and P1.1.1's own calibration code in ``scripts/analyze_reading.py``), and
the canary split into two panels:

* ``null`` — walk against IID per instrument: effect sizes and the exact permutation test, the maximum
  over checkpoints taken inside each split;
* ``rules`` / ``panels`` — three panels (``label_free``, ``canary_linear``, ``canary_knn``), the bands
  fitted on the IID seeds, one threshold per panel, the panel's leave-one-out false alarms (bands *and*
  threshold refitted without each seed) and, given ``check_seeds``, its alarms on fresh IID seeds; the
  drive-level count (any panel firing) and a Wilson interval on each count;
* ``walk`` — the healthy walk read against the bands: which panels fire, first when, and the split
  label (:func:`cafl4ds.reading_protocol.adjudicate_split`), a kNN fire corroborated by the fine-tune
  rungs against the IID seeds' reads where they exist. Under 4 IID seeds no threshold can be fitted,
  and the walk is read descriptively (each panel's max ``z``);
* ``canary_power`` — P1.1.0's single-checkpoint canary noise and MDD per channel;
* ``referee_check`` — the in-drive canary against the val referee, per seed at the drives' ends.

``mode=power`` (Part 4): per arm (``power.ladders``: P1.2.5's per-image ladder reads at every walk
checkpoint), task, rung and past slice (``power.past``: the slice and the checkpoint ``t0`` after the
stream left it), the loss from ``t0`` to each later checkpoint, its seed and image bootstrap noise, the
max-over-checkpoints test's MDD and the pool each target needs (:mod:`cafl4ds.referee_power`). An arm's
ladder may be several dirs (the fine-tune rungs were read apart); each rung is read from the first holding it.

``mode=breadth`` (Part 4b): per arm and panel task, the end-of-drive Δ against the floor, its seed and
image noise, the MDD of P1.2.5's one-sided test at Holm's strictest step, and the pool and seed count
that resolve each target.

Example::

    uv run python scripts/analyze_p114.py root=/path/to/outputs mode=power
"""

import itertools
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
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits

from cafl4ds import referee_power as rp
from cafl4ds.adaptation import balanced_accuracy
from cafl4ds.null_envelope import canary_power, checkpoint_steps, gap, read_corpus, signal_columns
from cafl4ds.reading_protocol import (
    adjudicate_split,
    calibrate_panel,
    call,
    finetune_drop,
    fit_bands,
    leave_one_out_fired,
    max_z,
)
from scripts.analyze_adaptation import Ladder, _slice_mask, _slice_ok
from scripts.analyze_reading import _finite, _matrix, _present, _settings, read_arm, skipped_steps

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


def three_panels(rows: list[dict[str, Any]], signals: list[str], arbiters: list[str]) -> dict[str, list[str]]:
    """P1.1.4's panels present in ``rows``: label-free, and the canary split into information and geometry."""
    present = _present(rows, signals + arbiters)
    return {
        "label_free": [s for s in signals if s in present],
        "canary_linear": [a for a in arbiters if a.startswith("linear_acc") and a in present],
        "canary_knn": [a for a in arbiters if a.startswith("knn_acc") and a in present],
    }


def _ladder_steps(names: list[str]) -> dict[str, float]:
    """Ladder checkpoint suffix → drive step (``t00423`` → 423, ``end`` → ``inf``)."""
    return {c: math.inf if c == "end" else float(c[1:]) for c in names}


class FinetuneReads:
    """Per-seed fine-tune reads (balanced accuracy) of one arm, looked up across its ladder dirs."""

    def __init__(self, root: Path, dirs: list[str], keys: list[str], checkpoints: list[str]) -> None:
        """Open every ladder dir; ``keys`` are ``task|rung`` reads, ``checkpoints`` the suffixes to look for."""
        self.ladders = [Ladder(root / d) for d in dirs]
        self.keys, self.checkpoints = keys, checkpoints

    def read(self, name: str, key: str) -> float:
        """Balanced accuracy of encoder ``name`` on ``key`` from the first ladder that holds it (``nan`` if none)."""
        task = key.split("|", 1)[0]
        for ladder in self.ladders:
            if ladder.has(name) and (rows := ladder.rows([name], key)) is not None:
                return float(balanced_accuracy(rows[0], ladder.labels[task]))
        return math.nan

    def by_checkpoint(self, prefix: str, index: int, key: str) -> dict[str, float]:
        """``{checkpoint: read}`` for ``<prefix>_s<index>_<checkpoint>``, the checkpoints it was read at."""
        out = {c: self.read(f"{prefix}_s{index}_{c}", key) for c in self.checkpoints}
        return {c: v for c, v in out.items() if np.isfinite(v)}


def _finetune_check(ft: FinetuneReads | None, seed: int, n_iid: int, after: float) -> dict[str, Any]:
    """The kNN fire's corroboration for one walk seed (drive seed ``seed``; ladder index ``seed - 1``)."""
    if ft is None:
        return {"drop": None, "per_key": {}}
    steps = _ladder_steps(ft.checkpoints)
    per_key = {}
    for key in ft.keys:
        walk = ft.by_checkpoint("walk", seed - 1, key)
        healthy: dict[str, list[float]] = {}
        for j in range(n_iid):
            for c, v in ft.by_checkpoint("iid", j, key).items():
                healthy.setdefault(c, []).append(v)
        per_key[key] = finetune_drop(walk, healthy, steps, after)
    checked = [v for v in per_key.values() if v is not None]
    return {"drop": any(checked) if checked else None, "per_key": per_key}


def adjudicate_walk(
    walk_read: dict[str, Any], groups: dict[str, list[str]], rules: dict[str, Any], ft: FinetuneReads | None, n_iid: int
) -> dict[int, dict[str, Any]]:
    """Per walk seed: which panels fired and first when, the fine-tune corroboration, and the split label."""
    out = {}
    for seed, entry in walk_read["per_seed"].items():
        calls = entry["calls"]
        fired = {p: any(calls[s]["fired"] for s in sig) for p, sig in groups.items()}
        first = {
            p: min((calls[s]["first_step"] for s in sig if calls[s]["fired"]), default=None)
            for p, sig in groups.items()
        }
        persisting = {
            p: any(calls[s]["final_z"] > rules[s].threshold for s in sig)
            for p, sig in groups.items()
            if p != "label_free"
        }
        knn_first = first["canary_knn"]
        check = (
            _finetune_check(ft, int(seed), n_iid, float(knn_first))
            if knn_first is not None
            else {"drop": None, "per_key": {}}
        )
        out[int(seed)] = {
            "fired": fired,
            "first_step": first,
            "persisting": persisting,
            "finetune_check": check,
            "label": adjudicate_split(fired, check["drop"]),
            "fired_signals": sorted(s for s, c in calls.items() if c["fired"]),
        }
    return out


def _last_finite(row: np.ndarray, window: int = 1) -> float:
    """Mean of the last ``window`` finite readings of a trajectory."""
    finite = row[np.isfinite(row)]
    return float(np.mean(finite[-window:])) if finite.size else math.nan


def _welch_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """``mean(a) − mean(b)`` and its 95% Welch interval."""
    va, vb = np.var(a, ddof=1) / a.size, np.var(b, ddof=1) / b.size
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1)) if se > 0 else float(a.size + b.size - 2)
    t = float(student_t.ppf(0.975, df))
    diff = float(a.mean() - b.mean())
    return diff, diff - t * se, diff + t * se


def _perm_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided permutation p of the difference in means (every relabelling, the identity included)."""
    pooled, n = np.concatenate([a, b]), a.size
    observed = abs(a.mean() - b.mean())
    stats = []
    for chosen in itertools.combinations(range(pooled.size), n):
        mask = np.zeros(pooled.size, dtype=bool)
        mask[list(chosen)] = True
        stats.append(abs(pooled[mask].mean() - pooled[~mask].mean()))
    return float(np.mean(np.array(stats) >= observed - 1e-12))


def referee_check(
    walk: list[dict[str, Any]], iid: list[dict[str, Any]], ladder: Ladder, config: DictConfig, skip: set[int]
) -> dict[str, Any]:
    """Does the in-drive canary see the walk-vs-IID loss the val referee sees? Per seed, at the drives' ends.

    For each rung (kNN, linear) and slice (pooled, per time of day), the canary's end read (the last
    reading, and the trailing mean of the last ``window_canary`` the rule reads) beside the referee's
    scene read on val at the drive's end checkpoint. Reports both walk-minus-IID gaps with Welch 95%
    intervals and exact permutation p, the paired gap of gaps (canary minus referee per drive, walk
    against IID), and the correlation of the two reads across drives.

    The decision rule, fixed before reading: if the canary's gap interval excludes the referee's gap,
    the canary misses the referee's loss (``reading_gap``); if it covers both zero and the referee's
    gap, the miss may be power alone (``power``); if it excludes zero and covers the referee's gap,
    both see it (``both_see``). If the referee's own interval covers zero there is nothing to miss
    (``no_referee_gap``).
    """
    rc = config.referee
    walk_seeds, iid_seeds = _seeds(walk), _seeds(iid)
    out: dict[str, Any] = {}
    for rung in rc.rungs:
        for slice_name, spec in rc.slices.items():
            channel = f"{rung}_acc" + ("" if slice_name == "pooled" else f"_{slice_name}")
            keep = _slice_mask(ladder, "scene", str(spec))
            labels = ladder.labels["scene"][keep]
            reads: dict[str, dict[str, np.ndarray]] = {}
            for arm_name, rows, seeds in (("walk", walk, walk_seeds), ("iid", iid, iid_seeds)):
                m = _matrix(rows, channel, skip)
                ref = []
                for s in seeds:
                    r = ladder.rows([f"{arm_name}_s{s - 1}_end"], f"scene|{rung}")
                    ref.append(balanced_accuracy(r[0][keep], labels) if r is not None else math.nan)
                reads[arm_name] = {
                    "canary_last": np.array([_last_finite(m[k]) for k in range(len(seeds))]),
                    "canary_smoothed": np.array(
                        [_last_finite(m[k], int(config.window_canary)) for k in range(len(seeds))]
                    ),
                    "referee": np.array(ref, dtype=float),
                }
            entry: dict[str, Any] = {
                "per_seed": {
                    a: {k: v.tolist() for k, v in r.items()} | {"seeds": s}
                    for (a, r), s in zip(reads.items(), (walk_seeds, iid_seeds), strict=True)
                },
                "n_referee_frames": int(keep.sum()),
            }
            if np.isnan(reads["walk"]["referee"]).any() or np.isnan(reads["iid"]["referee"]).any():
                entry["verdict"] = "no_referee_read"
                out[f"{rung}|{slice_name}"] = entry
                continue
            for k in ("canary_last", "canary_smoothed", "referee"):
                diff, lo, hi = _welch_ci(reads["walk"][k], reads["iid"][k])
                entry[k] = {"gap": diff, "ci95": [lo, hi], "p_perm": _perm_p(reads["walk"][k], reads["iid"][k])}
            paired = {a: reads[a]["canary_smoothed"] - reads[a]["referee"] for a in reads}
            diff, lo, hi = _welch_ci(paired["walk"], paired["iid"])
            entry["gap_of_gaps"] = {"gap": diff, "ci95": [lo, hi], "p_perm": _perm_p(paired["walk"], paired["iid"])}
            canary = np.concatenate([reads["walk"]["canary_smoothed"], reads["iid"]["canary_smoothed"]])
            referee = np.concatenate([reads["walk"]["referee"], reads["iid"]["referee"]])
            entry["r_across_drives"] = float(np.corrcoef(canary, referee)[0, 1])
            demean = {
                a: (
                    reads[a]["canary_smoothed"] - reads[a]["canary_smoothed"].mean(),
                    reads[a]["referee"] - reads[a]["referee"].mean(),
                )
                for a in reads
            }
            entry["r_within_arm"] = float(
                np.corrcoef(
                    np.concatenate([d[0] for d in demean.values()]), np.concatenate([d[1] for d in demean.values()])
                )[0, 1]
            )
            ref_lo, ref_hi = entry["referee"]["ci95"]
            c_lo, c_hi = entry["canary_smoothed"]["ci95"]
            ref_gap = entry["referee"]["gap"]
            if ref_lo <= 0 <= ref_hi:
                verdict = "no_referee_gap"
            elif not c_lo <= ref_gap <= c_hi:
                verdict = "reading_gap"
            elif c_lo <= 0 <= c_hi:
                verdict = "power"
            else:
                verdict = "both_see"
            entry["verdict"] = verdict
            out[f"{rung}|{slice_name}"] = entry
    return out


def read_bands(name: str, arm: DictConfig, config: DictConfig, root: Path) -> dict[str, Any]:
    """Part 1 for one arm: the null, the three panels' rules and false alarms, the walk's split read, the referee check.

    With fewer than 4 IID seeds no threshold can be calibrated (each leave-one-out fit needs 3): the arm
    is read descriptively — its null, its bands' widths, and each walk seed's maximum ``z`` per panel.
    """
    bb = str(arm.backbone)
    iid = _load(root, list(arm.iid), list(arm.iid_seeds) if arm.get("iid_seeds") is not None else None)
    walk = _load(root, list(arm.walk), None)
    check = _load(root, list(arm.iid), list(arm.check_seeds)) if arm.get("check_seeds") else []
    signals, arbiters = list(config.signals[bb]), list(config.arbiters)
    skip = skipped_steps(root / str(arm.walk[0]), config)
    if checkpoint_steps(walk) != checkpoint_steps(iid):
        raise ValueError(f"{name}: walk and IID checkpoint grids differ.")
    grid = checkpoint_steps(iid)
    groups = three_panels(iid, signals, arbiters)
    n_iid = len(_seeds(iid))
    descriptive = n_iid < 4
    entry: dict[str, Any] = {
        "backbone": bb,
        "iid_seeds": _seeds(iid),
        "walk_seeds": _seeds(walk),
        "check_seeds": _seeds(check),
        "n_checkpoints": len(grid),
        "skipped_steps": sorted(skip),
        "descriptive": descriptive,
        "rules": {},
        "panels": {},
    }
    rules: dict[str, Any] = {}
    loo_by_panel: dict[str, list[bool]] = {}
    check_by_panel: dict[str, list[bool]] = {}
    for panel_name, panel in groups.items():
        settings = _settings(panel, arbiters, config)
        nulls = {s: _matrix(iid, s, skip) for s in panel}
        if descriptive:
            rules.update({s: fit_bands(nulls[s], s, **settings[s]) for s in panel})
        else:
            rules.update(calibrate_panel(nulls, settings, float(config.alpha)))
            loo_by_panel[panel_name] = leave_one_out_fired(nulls, settings, float(config.alpha))
            check_by_panel[panel_name] = [
                any(bool(call(_matrix(check, s, skip)[k], rules[s])["fired"]) for s in panel)
                for k in range(len(_seeds(check)))
            ]
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
        entry["panels"][panel_name] = {"signals": panel, "threshold": rules[panel[0]].threshold}
        if not descriptive:
            loo, held = loo_by_panel[panel_name], check_by_panel[panel_name]
            entry["panels"][panel_name] |= {
                "loo": _count(loo),
                "check": _count(held) if held else None,
                "out_of_sample": _count(list(loo) + held),
            }
    if not descriptive:
        loo_any = [any(v) for v in zip(*loo_by_panel.values(), strict=True)]
        check_any = [any(v) for v in zip(*check_by_panel.values(), strict=True)] if check else []
        # The `degradation` label on the out-of-sample IID drives: the linear panel, or the kNN panel with
        # a fine-tune drop. No IID drive has fine-tune reads before 1b, so here it is the linear panel alone.
        entry["drive_level"] = {
            "loo": _count(loo_any),
            "check": _count(check_any) if check_any else None,
            "degradation_loo_linear_only": _count(loo_by_panel["canary_linear"]),
        }
    ft = (
        FinetuneReads(root, list(arm.ft_ladders), list(config.finetune_keys), list(config.ladder_checkpoints))
        if arm.get("ft_ladders")
        else None
    )
    if descriptive:
        entry["walk"] = {
            "per_seed": {
                int(seed): {
                    p: max(max_z(_matrix(walk, s, skip)[k], rules[s]) for s in sig) for p, sig in groups.items()
                }
                for k, seed in enumerate(_seeds(walk))
            },
            "max_z_per_signal": {
                s: [max_z(_matrix(walk, s, skip)[k], rules[s]) for k in range(len(_seeds(walk)))] for s in rules
            },
        }
    else:
        walk_read = read_arm(walk, rules, grid, arbiters, skip)
        entry["walk"] = {
            "fired_share": walk_read["fired_share"],
            "per_seed": adjudicate_walk(walk_read, groups, rules, ft, n_iid),
        }
    entry["null"] = read_null(walk, iid, skip)
    entry["canary_power"] = {
        a: canary_power(
            _matrix(iid, a, skip),
            int(config.n_query_pooled if a in ("knn_acc", "linear_acc") else config.n_query_stratum),
        )
        for a in arbiters
    }
    if arm.get("ladder"):
        entry["referee_check"] = referee_check(walk, iid, Ladder(root / str(arm.ladder)), config, skip)
    _log(name, entry)
    return entry


def _log(name: str, entry: dict[str, Any]) -> None:
    """Print each panel's threshold and false alarms, the walk's labels, the null and the referee check."""
    lines = [f"[{name}] ({entry['backbone']}) IID seeds {entry['iid_seeds']}, {entry['n_checkpoints']} readings"]
    for panel_name, p in entry["panels"].items():
        if entry["descriptive"]:
            lines.append(f"  panel {panel_name}: descriptive (no threshold)")
            continue
        lines.append(
            f"  panel {panel_name}: threshold {p['threshold']:.2f}, loo {p['loo']['fired']}/{p['loo']['of']}"
            + (f", check {p['check']['fired']}/{p['check']['of']}" if p["check"] else "")
        )
    if entry["descriptive"]:
        for seed, z in entry["walk"]["per_seed"].items():
            lines.append(f"  walk s{seed} max z: " + ", ".join(f"{p} {v:.2f}" for p, v in z.items()))
    else:
        lines.append(f"  drive-level loo {entry['drive_level']['loo']['fired']}/{entry['drive_level']['loo']['of']}")
        share = entry["walk"]["fired_share"]
        lines.append("  walk fired share: " + ", ".join(f"{s} {v:.1f}" for s, v in share.items() if v > 0))
        for seed, e in entry["walk"]["per_seed"].items():
            lines.append(f"  walk s{seed}: {e['label']}, first {e['first_step']}, ft {e['finetune_check']['drop']}")
    sig = {s: round(g["max_abs_d"], 2) for s, g in entry["null"].items() if g["p_max"] <= 0.05}
    lines.append(f"  null, walk vs IID at p_max <= 0.05: {sig}")
    for key, r in entry.get("referee_check", {}).items():
        if "referee" not in r:
            lines.append(f"  referee {key}: {r['verdict']}")
            continue
        lines.append(
            f"  referee {key}: canary gap {r['canary_smoothed']['gap']:+.3f} "
            f"[{r['canary_smoothed']['ci95'][0]:+.3f}, {r['canary_smoothed']['ci95'][1]:+.3f}], "
            f"referee gap {r['referee']['gap']:+.3f} [{r['referee']['ci95'][0]:+.3f}, {r['referee']['ci95'][1]:+.3f}], "
            f"gap of gaps {r['gap_of_gaps']['gap']:+.3f} p {r['gap_of_gaps']['p_perm']:.3f}, "
            f"r {r['r_across_drives']:+.2f} -> {r['verdict']}"
        )
    logger.info("\n".join(lines))


def _rows_any(ladders: list[Ladder], names: list[str], key: str) -> np.ndarray | None:
    """``Ladder.rows`` from the first ladder holding every name with ``key`` (fine-tunes may sit apart)."""
    for ladder in ladders:
        if all(ladder.has(n) for n in names) and (rows := ladder.rows(names, key)) is not None:
            return rows
    return None


def _open_ladders(root: Path, dirs: str | list[str]) -> list[Ladder]:
    """One or more ladder dirs of one arm; they must share the val query (checked)."""
    ladders = [Ladder(root / str(d)) for d in ([dirs] if isinstance(dirs, str) else dirs)]
    if any(not np.array_equal(x.query, ladders[0].query) for x in ladders[1:]):
        raise ValueError(f"ladders {dirs} do not share the val query.")
    return ladders


def read_power(name: str, ladders: list[Ladder], config: DictConfig) -> dict[str, Any]:
    """Part 4 for one arm: per past slice, task and rung, the max test's MDD now and the pool a target needs."""
    cfg = config.power
    ladder = ladders[0]
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
            rows = {cp: _rows_any(ladders, [f"walk_s{i}_{cp}" for i in seeds], f"{task}|{rung}") for cp in order}
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


def read_breadth(name: str, ladders: list[Ladder], config: DictConfig) -> dict[str, Any]:
    """Part 4b for one arm: the end-of-drive Δ against the floor per panel task, and what resolves ``target``.

    P1.2.5's breadth test: Δ = walk end − floor (the floor is one frozen read, paired on the images),
    bootstrapped over seeds and images, one-sided, Holm over the panel. Here its MDD at Holm's strictest
    step (``alpha / len(panel)``), and the pool and seed multipliers that bring it to each target. The
    gate tasks are read beside the panel for reference, outside the family.
    """
    cfg, bc = config.power, config.breadth
    ladder = ladders[0]
    seeds = list(cfg.seeds)
    alpha = float(bc.alpha) / len(bc.panel)
    out: dict[str, Any] = {}
    for task in list(bc.panel) + list(bc.reference):
        if task not in ladder.labels:
            continue
        labels = ladder.labels[task]
        for rung in bc.rungs:
            walk = _rows_any(ladders, [f"walk_s{i}_end" for i in seeds], f"{task}|{rung}")
            floor = _rows_any(ladders, [str(bc.floor)], f"{task}|{rung}")
            if walk is None or floor is None:
                continue
            noise = rp.change_noise(walk, [np.repeat(floor, walk.shape[0], axis=0)], labels, n_boot=int(cfg.n_boot))
            combined = rp.combined_noise(noise["seed_noise"], noise["image_noise"])
            shape = np.ones(1)
            out[f"{task}|{rung}"] = {
                "in_family": task in bc.panel,
                "n_images": int(labels.size),
                "min_class": int(min((labels == c).sum() for c in np.unique(labels))),
                "alpha": alpha,
                "delta": float(noise["loss"][0]),
                "sd_seed": float(noise["seed_noise"].std()),
                "sd_image": float(noise["image_noise"].std()),
                "mdd": rp.mdd(combined, shape, alpha=alpha, power=float(cfg.power))["mdd"],
                "pool_for": {
                    f"{target}@seeds_x{m}": rp.pool_for(
                        noise["seed_noise"],
                        noise["image_noise"],
                        shape,
                        float(target),
                        alpha=alpha,
                        seeds=float(m),
                        power=float(cfg.power),
                    )
                    for target in bc.targets
                    for m in cfg.seed_multipliers
                },
            }
    lines = [f"[{name}] breadth: delta vs floor, MDD at alpha {alpha:.4f}, pool for targets"]
    for key, e in out.items():
        pools = ", ".join(f"{k} {v:.1f}" for k, v in e["pool_for"].items())
        lines.append(
            f"  {key:<48} n {e['n_images']:4d}  delta {e['delta']:+.3f}  sd seed/image {e['sd_seed']:.4f}/"
            f"{e['sd_image']:.4f}  MDD {e['mdd']:.3f}  [{pools}]"
        )
    logger.info("\n".join(lines))
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
        with threadpool_limits(8):
            report = {name: read_bands(name, arm, config, root) for name, arm in config.arms.items()}
    elif config.mode == "power":
        with threadpool_limits(8):
            report = {
                name: read_power(name, _open_ladders(root, d), config) for name, d in config.power.ladders.items()
            }
    elif config.mode == "breadth":
        with threadpool_limits(8):
            report = {
                name: read_breadth(name, _open_ladders(root, d), config) for name, d in config.power.ladders.items()
            }
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
