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

``mode=branches`` (Parts 2 and 3, with ``arm_set=arms_1b``): the damage branches, read on the referee and by
every reader (:mod:`scripts.analyze_p114_branches`).

``mode=breadth`` (Part 4b): per arm and panel task, the end-of-drive Δ against the floor, its seed and
image noise, the MDD of P1.2.5's one-sided test at Holm's strictest step, and the pool and seed count
that resolve each target.

Example::

    uv run python scripts/analyze_p114.py root=/path/to/outputs mode=power
"""

import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, ListConfig
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits

from cafl4ds import referee_power as rp
from cafl4ds.adaptation import balanced_accuracy
from cafl4ds.null_envelope import (
    canary_power,
    checkpoint_steps,
    gap_fast,
    injected_mdd,
    read_corpus,
    signal_columns,
    split_masks,
)
from cafl4ds.reading_protocol import (
    adjudicate_split,
    calibrate_panel,
    call,
    finetune_drop,
    finetune_drop_consecutive,
    fit_bands,
    max_z,
    zscores,
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
        out[signal] = gap_fast(w, i)
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


class LadderSet:
    """One arm's ladder dirs, sharing the val query, searched in order for each encoder read.

    P1.2.5's ladders name an encoder by seed index (``walk_s0_end`` is drive seed 1); P1.1.4's name it
    by drive seed (``iid_s6_end``). Each dir is given as a path (index naming) or ``{dir, names: seed}``.
    """

    def __init__(self, root: Path, specs: list[Any]) -> None:
        """Open every dir; raise if they do not share the val query and its labels."""
        self.entries: list[tuple[Ladder, str]] = []
        for spec in specs:
            path, names = (spec, "index") if isinstance(spec, str) else (spec["dir"], str(spec.get("names", "index")))
            if names not in ("index", "seed"):
                raise ValueError(f"unknown naming {names!r} for {path}.")
            self.entries.append((Ladder(root / str(path)), names))
        first = self.entries[0][0]
        for ladder, _ in self.entries[1:]:
            same = np.array_equal(ladder.query, first.query) and all(
                np.array_equal(ladder.labels[t], first.labels[t]) for t in set(ladder.labels) & set(first.labels)
            )
            if not same:
                raise ValueError(f"{ladder.path} does not share {first.path}'s val query.")

    @property
    def ladder(self) -> Ladder:
        """The first dir (its query, labels and slices stand for all)."""
        return self.entries[0][0]

    def read(self, prefix: str, seed: int, checkpoint: str, key: str, keep: np.ndarray | None = None) -> float:
        """Balanced accuracy of ``<prefix>`` drive seed ``seed`` at ``checkpoint`` on ``task|rung`` (``nan`` if unread).

        ``keep`` restricts the read to a slice of the task's frames.
        """
        task = key.split("|", 1)[0]
        for ladder, names in self.entries:
            name = f"{prefix}_s{seed - 1 if names == 'index' else seed}_{checkpoint}"
            if ladder.has(name) and (rows := ladder.rows([name], key)) is not None:
                labels = ladder.labels[task]
                if keep is None:
                    return float(balanced_accuracy(rows[0], labels))
                return float(balanced_accuracy(rows[0][keep], labels[keep]))
        return math.nan

    def by_checkpoint(self, prefix: str, seed: int, key: str, checkpoints: list[str]) -> dict[str, float]:
        """``{checkpoint: read}`` over the checkpoints this drive was read at."""
        out = {c: self.read(prefix, seed, c, key) for c in checkpoints}
        return {c: v for c, v in out.items() if np.isfinite(v)}


_FT_RULES = ("min", "consecutive", "calibrated")


class FinetunePanel:
    """The kNN fire's fine-tune corroboration, three ways, on one arm's ladder reads.

    * ``min`` — the rule fixed before 1b: a read below every healthy seed's, at any checkpoint at or
      after the fire (:func:`finetune_drop`);
    * ``consecutive`` — the same on two consecutive read checkpoints (:func:`finetune_drop_consecutive`);
    * ``calibrated`` — the fine-tune reads as one more panel, by P1.1.1's rule: bands on the healthy
      seeds, a Gumbel threshold from their leave-one-out maxima at the panel's ``alpha`` over every
      read and checkpoint, the drive firing when any read's ``z`` passes it at or after the fire.

    The healthy seeds are the IID drives read at every ladder checkpoint; a held-out IID drive is read
    against the others (bands *and* threshold refitted without it).
    """

    def __init__(self, ladders: LadderSet, config: DictConfig, seeds: list[int]) -> None:
        """Collect every IID seed's fine-tune trajectory; seeds missing a checkpoint are left out."""
        self.ladders, self.config = ladders, config
        self.checkpoints = list(config.ladder_checkpoints)
        self.steps = _ladder_steps(self.checkpoints)
        self.keys = list(config.finetune_keys)
        rows = {s: {k: self._trajectory("iid", s, k) for k in self.keys} for s in seeds}
        self.seeds = [s for s, r in rows.items() if all(np.isfinite(v).all() for v in r.values())]
        self.healthy = {k: np.array([rows[s][k] for s in self.seeds]) for k in self.keys}
        self._rules: dict[int | None, dict[str, Any]] = {}

    def _trajectory(self, prefix: str, seed: int, key: str, source: LadderSet | None = None) -> np.ndarray:
        ladders = source or self.ladders
        return np.array([ladders.read(prefix, seed, c, key) for c in self.checkpoints])

    def _settings(self) -> dict[str, dict[str, Any]]:
        fc = self.config.finetune_panel
        return {
            f"finetune_acc_{k}": {"window": 1, "span": int(fc.span), "floor": float(fc.floor), "transform": "none"}
            for k in self.keys
        }

    def rules(self, exclude: int | None) -> dict[str, Any]:
        """The calibrated panel without seed ``exclude`` (cached)."""
        if exclude not in self._rules:
            keep = [i for i, s in enumerate(self.seeds) if s != exclude]
            nulls = {f"finetune_acc_{k}": self.healthy[k][keep] for k in self.keys}
            self._rules[exclude] = calibrate_panel(nulls, self._settings(), float(self.config.finetune_panel.alpha))
        return self._rules[exclude]

    def check(self, prefix: str, seed: int, after: float, source: LadderSet | None = None) -> dict[str, Any]:
        """Each rule's verdict for drive ``<prefix>`` seed ``seed``, from step ``after`` on.

        ``source`` reads the drive from other ladder dirs (a drive outside the arm, e.g. a natural failure).
        """
        if not self.seeds:
            return {"per_key": {}, "min": None, "consecutive": None, "calibrated": None, "threshold": None}
        exclude = seed if prefix == "iid" and source is None else None
        others = [i for i, s in enumerate(self.seeds) if s != exclude]
        out: dict[str, Any] = {"per_key": {}}
        z_max = -math.inf
        rules = self.rules(exclude) if len(others) >= 4 else None
        for k in self.keys:
            traj = self._trajectory(prefix, seed, k, source)
            drive = {c: float(v) for c, v in zip(self.checkpoints, traj, strict=True) if np.isfinite(v)}
            healthy = {c: self.healthy[k][others, j].tolist() for j, c in enumerate(self.checkpoints)}
            entry: dict[str, Any] = {
                "min": finetune_drop(drive, healthy, self.steps, after),
                "consecutive": finetune_drop_consecutive(drive, healthy, self.steps, after),
            }
            if rules is not None and drive:
                z = zscores(traj, rules[f"finetune_acc_{k}"])
                late = [z[j] for j, c in enumerate(self.checkpoints) if self.steps[c] >= after and np.isfinite(z[j])]
                entry["max_z_after"] = max(late) if late else None
                z_max = max(z_max, *late) if late else z_max
            out["per_key"][k] = entry
        for rule in ("min", "consecutive"):
            checked = [e[rule] for e in out["per_key"].values() if e[rule] is not None]
            out[rule] = any(checked) if checked else None
        threshold = None if rules is None else float(next(iter(rules.values())).threshold)
        out["calibrated"] = None if threshold is None or z_max == -math.inf else bool(z_max > threshold)
        out["threshold"] = threshold
        out["max_z_after"] = None if z_max == -math.inf else z_max
        return out


def _finetune_check(
    ft: FinetunePanel | None, config: DictConfig, prefix: str, seed: int, after: float
) -> dict[str, Any]:
    """The kNN fire's corroboration for one drive: every rule's verdict, and the one the label uses."""
    if ft is None:
        return {"drop": None, "rule": None, "variants": {}}
    variants = ft.check(prefix, seed, after)
    rule = str(config.finetune_rule)
    return {"drop": variants[rule], "rule": rule, "variants": variants}


def finetune_null_rates(ft: FinetunePanel, afters: list[float]) -> dict[str, Any]:
    """How often each rule corroborates on a healthy IID drive read against the others, from each start step."""
    out: dict[str, Any] = {"seeds": ft.seeds}
    for after in afters:
        checks = [ft.check("iid", s, after) for s in ft.seeds]
        out[f"from_{int(after)}"] = {
            rule: _count([bool(c[rule]) for c in checks if c[rule] is not None]) for rule in _FT_RULES
        }
    return out


def adjudicate_walk(
    walk_read: dict[str, Any],
    groups: dict[str, list[str]],
    rules: dict[str, Any],
    ft: FinetunePanel | None,
    config: DictConfig,
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
            _finetune_check(ft, config, "walk", int(seed), float(knn_first))
            if knn_first is not None
            else {"drop": None, "rule": None, "variants": {}}
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


def _perm_p_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """:func:`_perm_p` for each column of ``a`` ``[n_a, K]`` against ``b`` ``[n_b, K]``, every split at once."""
    pooled = np.vstack([a, b])
    masks = split_masks(pooled.shape[0], a.shape[0])
    stats = np.abs(masks @ pooled / a.shape[0] - (1 - masks) @ pooled / b.shape[0])
    observed = np.abs(a.mean(axis=0) - b.mean(axis=0))
    return np.mean(stats >= observed - 1e-12, axis=0)


def referee_read(ladders: LadderSet, prefix: str, seed: int, key: str, spec: str) -> float:
    """The referee's end read of one drive on ``key``, over the slice ``spec`` (see :func:`_slice_mask`).

    ``tod_balanced`` is the mean of the per-time-of-day reads: each stratum weighted equally, as the
    canary's stratified reservation weights them, where ``all`` weights them by their share of val (the
    P1.1 audit's G2).
    """
    task = key.split("|", 1)[0]
    if spec == "tod_balanced":
        strata = ladders.ladder.summary["stratum_names"].values()
        return float(np.mean([referee_read(ladders, prefix, seed, key, f"tod:{s}") for s in strata]))
    return ladders.read(prefix, seed, "end", key, _slice_mask(ladders.ladder, task, spec))


def referee_check(
    walk: list[dict[str, Any]], iid: list[dict[str, Any]], ladders: LadderSet, config: DictConfig, skip: set[int]
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
            pooled = str(spec) in ("all", "tod_balanced")  # both read against the canary's pooled channel
            channel = f"{rung}_acc" + ("" if pooled else f"_{slice_name}")
            reads: dict[str, dict[str, np.ndarray]] = {}
            for arm_name, rows, seeds in (("walk", walk, walk_seeds), ("iid", iid, iid_seeds)):
                m = _matrix(rows, channel, skip)
                ref = [referee_read(ladders, arm_name, s, f"scene|{rung}", str(spec)) for s in seeds]
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
                "n_referee_frames": int(
                    (
                        np.ones(ladders.ladder.pos["scene"].size, dtype=bool)
                        if str(spec) == "tod_balanced"
                        else _slice_mask(ladders.ladder, "scene", str(spec))
                    ).sum()
                ),
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


def _first_fire(trajectories: dict[str, np.ndarray], rules: dict[str, Any], grid: list[int]) -> dict[str, Any]:
    """One drive's call on one panel: did any instrument fire, and at which step first."""
    firsts = [grid[c["first"]] for s, x in trajectories.items() if (c := call(x, rules[s]))["first"] is not None]
    return {"fired": bool(firsts), "first_step": min(firsts) if firsts else None}


def _loo_calls(
    nulls: dict[str, np.ndarray], settings: dict[str, dict[str, Any]], alpha: float, grid: list[int]
) -> list[dict[str, Any]]:
    """:func:`leave_one_out_fired` with the first firing step: each null seed against a panel refitted without it."""
    n = next(iter(nulls.values())).shape[0]
    out = []
    for i in range(n):
        rules = calibrate_panel({s: np.delete(x, i, axis=0) for s, x in nulls.items()}, settings, alpha)
        out.append(_first_fire({s: np.asarray(x)[i] for s, x in nulls.items()}, rules, grid))
    return out


def _drive_level(
    loo: dict[str, list[dict[str, Any]]],
    held: dict[str, list[dict[str, Any]]],
    seeds: tuple[list[int], list[int]],
    ft: FinetunePanel | None,
    config: DictConfig,
) -> dict[str, Any]:
    """The out-of-sample IID drives read as a drive is: any panel firing, and the split label.

    ``seeds`` are the leave-one-out seeds and the fresh check seeds; a kNN fire on a held-out drive is
    corroborated against every *other* healthy seed's fine-tune reads.
    """
    loo_seeds, check_seeds = seeds
    drives = [(seed, {p: loo[p][k] for p in loo}) for k, seed in enumerate(loo_seeds)]
    drives += [(seed, {p: held[p][k] for p in held}) for k, seed in enumerate(check_seeds)]
    per_drive: list[dict[str, Any]] = []
    for seed, calls in drives:
        fired = {p: c["fired"] for p, c in calls.items()}
        knn_first = calls["canary_knn"]["first_step"] if "canary_knn" in calls else None
        check = (
            _finetune_check(ft, config, "iid", seed, float(knn_first))
            if knn_first is not None
            else {"drop": None, "rule": None, "variants": {}}
        )
        per_drive.append(
            {
                "seed": seed,
                "held_out_by": "loo" if seed in loo_seeds else "check",
                "fired": fired,
                "first_step": {p: c["first_step"] for p, c in calls.items()},
                "finetune_check": check,
                "label": adjudicate_split(fired, check["drop"]),
            }
        )
    n_loo = len(loo_seeds)
    any_fired = [any(d["fired"].values()) for d in per_drive]
    degraded = [d["label"] == "degradation" for d in per_drive]
    return {
        "per_drive": per_drive,
        "loo": _count(any_fired[:n_loo]),
        "check": _count(any_fired[n_loo:]) if check_seeds else None,
        "out_of_sample": _count(any_fired),
        "degradation": {
            "loo": _count(degraded[:n_loo]),
            "check": _count(degraded[n_loo:]) if check_seeds else None,
            "out_of_sample": _count(degraded),
        },
        "degradation_loo_linear_only": _count([c["fired"] for c in loo["canary_linear"]]),
    }


def _bars(entry: dict[str, Any], config: DictConfig) -> dict[str, Any]:
    """Part 1b's pass bars: out-of-sample false alarms per panel, and the `degradation` label's."""
    bars = config.bars
    out: dict[str, Any] = {}
    for panel_name, p in entry["panels"].items():
        k = p["out_of_sample"]
        out[panel_name] = {"fired": k["fired"], "of": k["of"], "pass": k["fired"] <= int(bars.panel_false_alarms)}
    d = entry["drive_level"]["degradation"]["out_of_sample"]
    out["degradation"] = {"fired": d["fired"], "of": d["of"], "pass": d["fired"] <= int(bars.degradation_false_alarms)}
    return out


def referee_null(ladders: LadderSet, walk_seeds: list[int], iid_seeds: list[int], config: DictConfig) -> dict[str, Any]:
    """The val referee's own walk-vs-IID null, and the healthy range of each rung, at every ladder checkpoint.

    Per task and rung, the drives read at every checkpoint (IID drives saved only at their end are
    left out) — P1.1.0's exact max-over-checkpoints permutation test, on balanced accuracy.
    """
    checkpoints = list(config.ladder_checkpoints)
    out: dict[str, Any] = {}
    for task in config.referee_null.tasks:
        for rung in config.referee_null.rungs:
            key = f"{task}|{rung}"
            reads = {
                prefix: {s: [ladders.read(prefix, s, c, key) for c in checkpoints] for s in seeds}
                for prefix, seeds in (("walk", walk_seeds), ("iid", iid_seeds))
            }
            full = {p: {s: v for s, v in r.items() if np.isfinite(v).all()} for p, r in reads.items()}
            entry: dict[str, Any] = {
                "checkpoints": checkpoints,
                "walk_seeds": sorted(full["walk"]),
                "iid_seeds": sorted(full["iid"]),
            }
            if len(full["walk"]) < 3 or len(full["iid"]) < 3:
                out[key] = entry | {"gap": None}
                continue
            w, i = np.array(list(full["walk"].values())), np.array(list(full["iid"].values()))
            out[key] = entry | {
                "walk_mean": w.mean(axis=0).tolist(),
                "iid_mean": i.mean(axis=0).tolist(),
                "iid_sd": i.std(axis=0, ddof=1).tolist(),
                "iid_min": i.min(axis=0).tolist(),
                "iid_max": i.max(axis=0).tolist(),
                "gap": gap_fast(w, i),
            }
            for spec in config.referee_null.get("slices", []):
                out[f"{key}|{spec}"] = _referee_slice_null(
                    ladders, (walk_seeds, iid_seeds), key, str(spec), checkpoints
                )
    return out


def _referee_slice_null(
    ladders: LadderSet, seeds: tuple[list[int], list[int]], key: str, spec: str, checkpoints: list[str]
) -> dict[str, Any]:
    """The referee's walk-vs-IID gap on one slice at every checkpoint, with each checkpoint's exact p.

    Which regime a walk loses on, read on val rather than on the canary (the P1.1 audit's G6). Beside
    the max-over-checkpoints test, each checkpoint's own two-sided exact permutation p, so a gap can be
    placed in time (before or after the stream reaches the slice's regime).
    """
    keep = _slice_mask(ladders.ladder, key.split("|", 1)[0], spec)
    reads = {
        prefix: {s: [ladders.read(prefix, s, c, key, keep) for c in checkpoints] for s in ss}
        for prefix, ss in zip(("walk", "iid"), seeds, strict=True)
    }
    full = {p: {s: v for s, v in r.items() if np.isfinite(v).all()} for p, r in reads.items()}
    entry: dict[str, Any] = {"checkpoints": checkpoints, "n_frames": int(keep.sum())}
    if len(full["walk"]) < 3 or len(full["iid"]) < 3:
        return entry | {"gap": None}
    w, i = np.array(list(full["walk"].values())), np.array(list(full["iid"].values()))
    return entry | {
        "walk_seeds": sorted(full["walk"]),
        "iid_seeds": sorted(full["iid"]),
        "walk_minus_iid": (w.mean(axis=0) - i.mean(axis=0)).tolist(),
        "p_per_checkpoint": _perm_p_columns(w, i).tolist(),
        "gap": gap_fast(w, i),
    }


def injected_canary_power(
    iid: list[dict[str, Any]], arbiters: list[str], config: DictConfig, skip: set[int]
) -> dict[str, Any]:
    """The canary MDD for the test actually run (B2): a known drop injected into healthy seeds, per shape.

    With ``inject.sustained_from_step``, a sustained drop is also injected from the first reading at or
    after that step: every drive starts from the same state, so the seeds nearly coincide at the first
    readings, and a drop held there is caught against almost no spread (the P1.1 audit's G3).
    """
    ic = config.inject
    late_step = ic.get("sustained_from_step")
    grid = checkpoint_steps(iid)
    late = next(i for i, s in enumerate(grid) if s >= int(late_step)) if late_step is not None else None
    out: dict[str, Any] = {}
    for a in arbiters:
        m = _matrix(iid, a, skip)
        common: dict[str, Any] = {
            "n_inject": int(ic.n_inject),
            "n_groups": int(ic.n_groups),
            "alpha": float(config.alpha),
            "power": float(ic.power),
            "seed": int(ic.seed),
        }
        out[a] = {shape: injected_mdd(m, str(shape), **common) for shape in ic.shapes}
        if late is not None:
            out[a][f"sustained_from_{grid[late]}"] = injected_mdd(m, "sustained", first=late, **common)
        logger.info(f"  injected MDD {a}: " + ", ".join(f"{k} {v['mdd']:.4f}" for k, v in out[a].items()))
    return out


def read_bands(name: str, arm: DictConfig, config: DictConfig, root: Path) -> dict[str, Any]:  # noqa: C901
    """Part 1 for one arm: the null, the three panels' rules and false alarms, the walk's split read, the referee.

    ``iid_seeds`` calibrate (``null`` = every IID seed), ``check_seeds`` are the fresh check; the null
    and the referee read the walk against every IID seed. With fewer than 4 calibration seeds no
    threshold can be fitted (each leave-one-out fit needs 3): the arm is read descriptively — its
    null, its bands' widths, and each walk seed's maximum ``z`` per panel.
    """
    bb = str(arm.backbone)
    iid_all = _load(root, list(arm.iid), None)
    iid = _load(root, list(arm.iid), list(arm.iid_seeds)) if arm.get("iid_seeds") is not None else iid_all
    walk = _load(root, list(arm.walk), None)
    check = _load(root, list(arm.iid), list(arm.check_seeds)) if arm.get("check_seeds") else []
    signals, arbiters = list(config.signals[bb]), list(config.arbiters)
    skip = skipped_steps(root / str(arm.walk[0]), config)
    if checkpoint_steps(walk) != checkpoint_steps(iid_all):
        raise ValueError(f"{name}: walk and IID checkpoint grids differ.")
    grid = checkpoint_steps(iid)
    groups = three_panels(iid, signals, arbiters)
    ladders = LadderSet(root, list(arm.ladders)) if arm.get("ladders") else None
    all_seeds = _seeds(iid_all)
    ft = FinetunePanel(ladders, config, all_seeds) if ladders is not None else None
    descriptive = len(_seeds(iid)) < 4
    entry: dict[str, Any] = {
        "backbone": bb,
        "iid_seeds": _seeds(iid),
        "walk_seeds": _seeds(walk),
        "check_seeds": _seeds(check),
        "null_iid_seeds": all_seeds,
        "n_checkpoints": len(grid),
        "skipped_steps": sorted(skip),
        "descriptive": descriptive,
        "rules": {},
        "panels": {},
    }
    rules: dict[str, Any] = {}
    loo_by_panel: dict[str, list[dict[str, Any]]] = {}
    check_by_panel: dict[str, list[dict[str, Any]]] = {}
    for panel_name, panel in groups.items():
        settings = _settings(panel, arbiters, config)
        nulls = {s: _matrix(iid, s, skip) for s in panel}
        if descriptive:
            rules.update({s: fit_bands(nulls[s], s, **settings[s]) for s in panel})
        else:
            rules.update(calibrate_panel(nulls, settings, float(config.alpha)))
            loo_by_panel[panel_name] = _loo_calls(nulls, settings, float(config.alpha), grid)
            check_by_panel[panel_name] = [
                _first_fire({s: _matrix(check, s, skip)[k] for s in panel}, rules, grid)
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
            loo = [c["fired"] for c in loo_by_panel[panel_name]]
            held = [c["fired"] for c in check_by_panel[panel_name]]
            entry["panels"][panel_name] |= {
                "loo": _count(loo),
                "check": _count(held) if held else None,
                "out_of_sample": _count(loo + held),
            }
    if not descriptive:
        entry["drive_level"] = _drive_level(loo_by_panel, check_by_panel, (_seeds(iid), _seeds(check)), ft, config)
        entry["bars"] = _bars(entry, config)
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
            "per_seed": adjudicate_walk(walk_read, groups, rules, ft, config),
        }
    entry["null"] = read_null(walk, iid_all, skip)
    entry["canary_power"] = {
        a: canary_power(
            _matrix(iid, a, skip),
            int(config.n_query_pooled if a in ("knn_acc", "linear_acc") else config.n_query_stratum),
        )
        for a in arbiters
    }
    if ladders is not None:
        entry["referee_check"] = referee_check(walk, iid_all, ladders, config, skip)
    if ladders is not None and arm.get("referee_null"):
        entry["referee_null"] = referee_null(ladders, _seeds(walk), all_seeds, config)
    if ft is not None and len(ft.seeds) >= 5:
        entry["finetune_null"] = finetune_null_rates(ft, [float(a) for a in config.finetune_panel.null_from])
        if arm.get("natural"):
            nat = LadderSet(root, list(arm.natural.ladders))
            entry["finetune_natural"] = {int(s): ft.check("walk", int(s), 0.0, nat) for s in arm.natural.seeds}
    if arm.get("inject_power"):
        entry["canary_power_injected"] = injected_canary_power(iid_all, arbiters, config, skip)
    _log(name, entry)
    return entry


def _drive_level_lines(entry: dict[str, Any]) -> list[str]:
    """The out-of-sample IID drives' counts, the bars, and each firing drive's label."""
    dl = entry["drive_level"]
    oos, deg = dl["out_of_sample"], dl["degradation"]["out_of_sample"]
    lines = [f"  drive-level any panel {oos['fired']}/{oos['of']}, degradation {deg['fired']}/{deg['of']}"]
    bars = [f"{k} {v['fired']}/{v['of']} {'pass' if v['pass'] else 'FAIL'}" for k, v in entry["bars"].items()]
    lines.append("  bars: " + ", ".join(bars))
    for d in dl["per_drive"]:
        if any(d["fired"].values()):
            lines.append(
                f"    iid s{d['seed']} ({d['held_out_by']}): {d['label']}, first {d['first_step']}, "
                f"ft {d['finetune_check']['drop']}"
            )
    return lines


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
        lines += _drive_level_lines(entry)
        share = entry["walk"]["fired_share"]
        lines.append("  walk fired share: " + ", ".join(f"{s} {v:.1f}" for s, v in share.items() if v > 0))
        for seed, e in entry["walk"]["per_seed"].items():
            lines.append(f"  walk s{seed}: {e['label']}, first {e['first_step']}, ft {e['finetune_check']['drop']}")
    for key, r in entry.get("referee_null", {}).items():
        if r["gap"] is not None:
            g = r["gap"]
            lines.append(
                f"  referee null {key}: walk {len(r['walk_seeds'])} vs iid {len(r['iid_seeds'])}, "
                f"end gap {g['final_gap']:+.3f}, max|d| {g['max_abs_d']:.2f} p {g['p_max']:.4f}"
            )
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


def _own_name(name: str, naming: str) -> str:
    """An index-named encoder (``walk_s0_end``) in a dir's own naming (``seed``: ``walk_s1_end``); others unchanged."""
    if naming == "index":
        return name
    return re.sub(r"_s(\d+)_", lambda m: f"_s{int(m.group(1)) + 1}_", name, count=1)


def _rows_any(ladders: list[tuple[Ladder, str]], names: list[str], key: str) -> np.ndarray | None:
    """``Ladder.rows`` from the first ladder holding every name with ``key`` (fine-tunes may sit apart).

    ``names`` are index-named (``walk_s0_end`` is drive seed 1), and renamed to each dir's own naming.
    """
    for ladder, naming in ladders:
        own = [_own_name(n, naming) for n in names]
        if all(ladder.has(n) for n in own) and (rows := ladder.rows(own, key)) is not None:
            return rows
    return None


def _open_ladders(root: Path, dirs: str | list[Any] | ListConfig) -> list[tuple[Ladder, str]]:
    """One or more ladder dirs of one arm, each a path (index naming) or ``{dir, names: seed}``.

    They must share the val query (checked).
    """
    specs = [dirs] if isinstance(dirs, str) else list(dirs)
    ladders = [
        (Ladder(root / str(s)), "index") if isinstance(s, str) else (Ladder(root / str(s["dir"])), str(s["names"]))
        for s in specs
    ]
    if any(naming not in ("index", "seed") for _, naming in ladders):
        raise ValueError(f"unknown naming in {dirs}.")
    if any(not np.array_equal(x.query, ladders[0][0].query) for x, _ in ladders[1:]):
        raise ValueError(f"ladders {dirs} do not share the val query.")
    return ladders


def read_power(name: str, ladders: list[tuple[Ladder, str]], config: DictConfig) -> dict[str, Any]:
    """Part 4 for one arm: per past slice, task and rung, the max test's MDD now and the pool a target needs."""
    cfg = config.power
    ladder = ladders[0][0]
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


def read_breadth(name: str, ladders: list[tuple[Ladder, str]], config: DictConfig) -> dict[str, Any]:
    """Part 4b for one arm: the end-of-drive Δ against the floor per panel task, and what resolves ``target``.

    P1.2.5's breadth test: Δ = walk end − floor (the floor is one frozen read, paired on the images),
    bootstrapped over seeds and images, one-sided, Holm over the panel. Here its MDD at Holm's strictest
    step (``alpha / len(panel)``), and the pool and seed multipliers that bring it to each target. The
    gate tasks are read beside the panel for reference, outside the family.
    """
    cfg, bc = config.power, config.breadth
    seeds = list(cfg.seeds)
    alpha = float(bc.alpha) / len(bc.panel)
    out: dict[str, Any] = {}
    for task in list(bc.panel) + list(bc.reference):
        # the first dir reading the task holds its labels (MAE's breadth tasks sit in a dir of their own)
        labels = next((lad.labels[task] for lad, _ in ladders if task in lad.labels), None)
        if labels is None:
            continue
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
            arms = config[str(config.get("arm_set", "arms"))]
            report = {name: read_bands(name, arm, config, root) for name, arm in arms.items()}
    elif config.mode == "power":
        with threadpool_limits(8):
            report = {
                name: read_power(name, _open_ladders(root, d), config) for name, d in config.power.ladders.items()
            }
    elif config.mode == "branches":
        # Imported here: the branch module builds on this one's readers (a top-level import would be circular).
        from scripts.analyze_p114_branches import read_branches  # noqa: PLC0415

        with threadpool_limits(8):
            arms = config[str(config.get("arm_set", "arms"))]
            only = config.get("only")
            report = {
                name: read_branches(name, arm, config, root)
                for name, arm in arms.items()
                if arm.get("branches") and (only is None or name in only)
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
