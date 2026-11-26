"""The P1.1.0 healthy-movement null — envelopes, walk-vs-IID gaps, and canary probe power.

Phase 0's load-bearing warning is that a healthy backbone *also* moves its instruments (a U-shaped
RankMe dip, rising drift) for benign reasons, so a reading off the regime walk means nothing until it
is read against how much a healthy, *uncorrelated* drive moves. This module reads two deploy corpora
of the same diet — the regime **walk** and its matched **IID** control (same images, same step count,
same per-step batch sizes; only the order differs, see :class:`~cafl4ds.data.regime.RegimeStream`) —
and reports, per signal:

* the **null envelope** — the IID arm's per-checkpoint seed mean and spread;
* the **null movement** — how far the IID arm travels from its start (what "healthy" moves by);
* the **walk-vs-IID gap** — the per-checkpoint standardized mean difference (an effect size *d*,
  walk minus IID over the pooled seed SD), tested by an exact **seed-label permutation test** whose
  statistic is the max |d| over all checkpoints — so the family-wise error over the ~hundreds of
  correlated checkpoints is controlled without a multiple-comparisons guess.

It also owns the canary **probe-power** read carried from the P1.0 audit (C2): the per-checkpoint
binomial sampling error at the query size, the seed-to-seed spread under IID, and the smallest
walk-vs-IID canary gap a 5-seed comparison can resolve.

Pure numpy over the corpus rows (no torch, no harness), so it is fully unit-testable.
"""

from __future__ import annotations

import csv
import itertools
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# The corpus context columns — everything else numeric is a signal (see cafl4ds.deploy_corpus).
_CONTEXT = frozenset(
    {
        "backbone",
        "seed",
        "arm",
        "checkpoint_index",
        "step",
        "era_index",
        "era_name",
        "is_transition",
        "steps_since_transition",
        "leg_ordinal",
    }
)

# Student-t quantiles for the minimal-detectable-difference multiplier t(0.975, df) + t(0.80, df) —
# a two-sided 5% test at 80% power. Tabulated (scipy is not a declared dependency); beyond df=30 the
# normal quantiles are within 1% and are used instead.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         12: 2.179, 14: 2.145, 16: 2.120, 18: 2.101, 20: 2.086, 25: 2.060, 30: 2.042}  # fmt: skip
_T80 = {1: 1.376, 2: 1.061, 3: 0.978, 4: 0.941, 5: 0.920, 6: 0.906, 7: 0.896, 8: 0.889, 9: 0.883, 10: 0.879,
        12: 0.873, 14: 0.868, 16: 0.865, 18: 0.862, 20: 0.860, 25: 0.856, 30: 0.854}  # fmt: skip


def read_corpus(path: str | Path) -> list[dict[str, Any]]:
    """Read a corpus ``health.csv`` into rows with numeric values parsed (empty → ``None``).

    Args:
        path: The ``health.csv`` path.

    Returns:
        One dict per row.
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = None
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def signal_columns(rows: list[dict[str, Any]]) -> list[str]:
    """The numeric signal columns of a corpus (context columns excluded), sorted."""
    keys: set[str] = set()
    for row in rows:
        keys |= {k for k, v in row.items() if k not in _CONTEXT and isinstance(v, float)}
    return sorted(keys)


def seed_matrix(rows: list[dict[str, Any]], signal: str, arm: str = "live") -> np.ndarray:
    """One signal as a ``[seeds, checkpoints]`` matrix, rows ordered by seed, columns by checkpoint.

    Args:
        rows: Corpus rows.
        signal: The signal column.
        arm: The arm to read (``live`` by default).

    Returns:
        The matrix (missing values are ``nan``).

    Raises:
        ValueError: If the seeds do not share a checkpoint count (the trajectories cannot be aligned).
    """
    by_seed: dict[int, dict[int, float]] = {}
    for row in rows:
        if row.get("arm") != arm:
            continue
        value = row.get(signal)
        by_seed.setdefault(int(row["seed"]), {})[int(row["checkpoint_index"])] = (
            float(value) if isinstance(value, float) else math.nan
        )
    lengths = {len(v) for v in by_seed.values()}
    if len(lengths) != 1:
        raise ValueError(f"seeds have differing checkpoint counts {sorted(lengths)} for {signal!r}.")
    return np.array([[series[i] for i in sorted(series)] for _, series in sorted(by_seed.items())])


def checkpoint_steps(rows: list[dict[str, Any]], arm: str = "live") -> list[int]:
    """The stream step of each checkpoint, for aligning two arms' grids.

    Raises:
        ValueError: If the seeds do not share one step grid (their readings would be misaligned).
    """
    by_seed: dict[int, list[tuple[int, int]]] = {}
    for r in rows:
        if r.get("arm") == arm:
            by_seed.setdefault(int(r["seed"]), []).append((int(r["checkpoint_index"]), int(r["step"])))
    grids = {seed: [step for _, step in sorted(pairs)] for seed, pairs in by_seed.items()}
    first = grids[min(grids)]
    differing = [seed for seed, grid in grids.items() if grid != first]
    if differing:
        raise ValueError(f"seeds {sorted(differing)} checkpoint on a different step grid than seed {min(grids)}.")
    return first


def t_multiplier(df: int) -> float:
    """``t(0.975, df) + t(0.80, df)`` — the two-sided 5% / 80%-power MDD multiplier at ``df``.

    Args:
        df: Degrees of freedom (``>= 1``); an untabulated ``df`` uses the next tabulated *smaller* one
            (conservative), and ``df > 30`` the normal quantiles.

    Returns:
        The multiplier.
    """
    if df > 30:
        return 1.959964 + 0.841621
    key = max(k for k in _T975 if k <= max(df, 1))
    return _T975[key] + _T80[key]


def envelope(matrix: np.ndarray, k: float = 2.0) -> dict[str, np.ndarray]:
    """The per-checkpoint null envelope: seed mean ± ``k`` seed standard deviations.

    Args:
        matrix: ``[seeds, checkpoints]``.
        k: Envelope half-width in seed SDs.

    Returns:
        ``{"mean", "sd", "lo", "hi"}`` per checkpoint (``sd`` uses ``ddof=1``).
    """
    mean = np.nanmean(matrix, axis=0)
    sd = np.nanstd(matrix, axis=0, ddof=1)
    return {"mean": mean, "sd": sd, "lo": mean - k * sd, "hi": mean + k * sd}


def movement(matrix: np.ndarray) -> dict[str, float]:
    """How far a trajectory travels from its start — the null's "healthy movement".

    Args:
        matrix: ``[seeds, checkpoints]``.

    Returns:
        ``final`` — the seed-mean net change start→end; ``max_abs`` — the seed-mean of the largest
        excursion from the start; ``range`` — the seed-mean max-minus-min over the trajectory.
    """
    delta = matrix - matrix[:, :1]
    return {
        "final": float(np.nanmean(delta[:, -1])),
        "max_abs": float(np.nanmean(np.nanmax(np.abs(delta), axis=1))),
        "range": float(np.nanmean(np.nanmax(matrix, axis=1) - np.nanmin(matrix, axis=1))),
    }


def effect_sizes(walk: np.ndarray, iid: np.ndarray) -> np.ndarray:
    """Per-checkpoint standardized mean difference *d* (walk minus IID, over the pooled seed SD).

    A checkpoint where both arms have zero spread reads ``0`` if the means agree and ``nan`` if they
    differ (an undefined *d* — e.g. a quantized per-stratum accuracy where every seed coincides);
    ``nan`` checkpoints are ignored downstream.

    Args:
        walk: Walk arm ``[seeds, checkpoints]``.
        iid: IID arm ``[seeds, checkpoints]``.

    Returns:
        ``d`` per checkpoint.
    """
    diff = np.nanmean(walk, axis=0) - np.nanmean(iid, axis=0)
    pooled = np.sqrt((np.nanvar(walk, axis=0, ddof=1) + np.nanvar(iid, axis=0, ddof=1)) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(pooled > 0, diff / np.where(pooled > 0, pooled, 1.0), np.where(diff == 0, 0.0, np.nan))
    return np.asarray(d, dtype=float)


def gap(walk: np.ndarray, iid: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """The walk-vs-IID gap: effect sizes, plus an exact seed-label permutation test (max over checkpoints).

    The observed statistic is ``max_t |d_t|``. Its null distribution comes from every relabelling of
    the pooled seeds into a "walk" and an "IID" group of the original sizes (5 + 5 seeds → 252 splits),
    recomputing ``max_t |d_t|`` each time. Taking the max *inside* each relabelling is what controls
    the family-wise error across the correlated checkpoints. ``p_max`` is the share of relabellings
    at least as extreme as the observed (the identity included, so it is never 0); ``frac_sig`` is the
    share of checkpoints whose own ``|d_t|`` clears the null's ``1 - alpha`` quantile of the max — the
    checkpoints individually significant under that family-wise control. Caveat: the smallest
    attainable ``p_max`` is ``2 / n_perm`` (the label swap is symmetric), so a 3 + 3 seed comparison
    (20 splits) cannot reach 5%.

    Args:
        walk: Walk arm ``[seeds, checkpoints]``.
        iid: IID arm ``[seeds, checkpoints]`` (same checkpoint grid — the matched control).
        alpha: The family-wise level for ``frac_sig``.

    Returns:
        ``max_abs_d``, ``mean_abs_d``, ``final_d``, ``final_gap`` (raw units), ``p_max``, ``frac_sig``,
        ``d_crit`` (the null's ``1 - alpha`` max-|d| quantile), ``n_perm``.

    Raises:
        ValueError: If the arms do not share a checkpoint count.
    """
    if walk.shape[1] != iid.shape[1]:
        raise ValueError(f"walk and IID checkpoint counts differ: {walk.shape[1]} vs {iid.shape[1]}.")
    d = effect_sizes(walk, iid)
    observed = float(np.nanmax(np.abs(d)))
    pooled = np.vstack([walk, iid])
    n_walk, n_total = walk.shape[0], pooled.shape[0]
    null_max = []
    for chosen in itertools.combinations(range(n_total), n_walk):
        mask = np.zeros(n_total, dtype=bool)
        mask[list(chosen)] = True
        null_max.append(float(np.nanmax(np.abs(effect_sizes(pooled[mask], pooled[~mask])))))
    null = np.array(null_max)
    d_crit = float(np.quantile(null, 1.0 - alpha))
    w_mean = np.nanmean(walk, axis=0)
    i_mean = np.nanmean(iid, axis=0)
    return {
        "max_abs_d": observed,
        "mean_abs_d": float(np.nanmean(np.abs(d))),
        "final_d": float(d[-1]),
        "final_gap": float(w_mean[-1] - i_mean[-1]),
        "p_max": float(np.mean(null >= observed - 1e-12)),
        "frac_sig": float(np.nanmean(np.abs(d) > d_crit)),
        "d_crit": d_crit,
        "n_perm": float(null.size),
    }


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks ``0..n-1``, ties sharing the mean of their positions."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size)
    ranks[order] = np.arange(x.size, dtype=float)
    for value in np.unique(x):
        tied = x == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman_exact(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> tuple[float, float]:
    """Spearman's rho of two short series, with its exact two-sided permutation p over every ordering of ``y``.

    For the few levels of a sweep (5 overlap levels: 120 orderings) the exact p is cheap, and its floor is
    explicit: ``2 / n!`` without ties (1/60 at 5 levels).

    Args:
        x: The levels (e.g. the overlap fractions).
        y: One value per level.

    Returns:
        ``(rho, p)``; ``(nan, nan)`` if either series is constant.

    Raises:
        ValueError: If the series differ in length, or hold fewer than 3 or more than 8 values.
    """
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xs.shape != ys.shape or not 3 <= xs.size <= 8:
        raise ValueError(f"need two equal-length series of 3-8 values; got {xs.size} and {ys.size}.")
    rx, ry = _average_ranks(xs), _average_ranks(ys)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return math.nan, math.nan
    rho = float(np.corrcoef(rx, ry)[0, 1])
    null = np.array([np.corrcoef(rx, ry[list(p)])[0, 1] for p in itertools.permutations(range(ys.size))])
    return rho, float(np.mean(np.abs(null) >= abs(rho) - 1e-12))


def wilson_halfwidth(p: float, n: int, z: float = 1.959964) -> float:
    """Half-width of the Wilson score interval for a binomial proportion (95% by default).

    Args:
        p: Observed accuracy.
        n: Number of query images it was scored on.
        z: Normal quantile.

    Returns:
        The interval's half-width.
    """
    denom = 1.0 + z * z / n
    return float(z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom)


def canary_power(iid: np.ndarray, n_query: int) -> dict[str, float]:
    """The canary's per-checkpoint noise and the smallest walk-vs-IID gap ``n_seeds`` can resolve.

    Two noise sources stack on a canary reading: the **sampling** error of scoring ``n_query``
    images (binomial), and the **seed** spread of the trajectory itself. The minimal detectable
    difference (two-sided 5%, 80% power) for a walk-vs-IID comparison of seed means at one
    checkpoint is ``2.80 · sd_seed · sqrt(2 / n_seeds)``; averaging over a window of checkpoints
    shrinks it further, but adjacent readings are correlated, so the single-checkpoint figure is the
    conservative one reported here. The multiplier uses Student-t quantiles at ``2·n_seeds − 2``
    degrees of freedom (the seed SD is estimated, not known), and ``sd_seed`` is the root of the mean
    per-checkpoint variance (a median of per-checkpoint SDs would be biased low at 5 seeds).

    Args:
        iid: IID canary accuracies ``[seeds, checkpoints]``.
        n_query: Query images behind each reading.

    Returns:
        ``mean`` accuracy, ``ci95_halfwidth`` (Wilson, one checkpoint), ``sd_seed`` (root mean
        across-seed variance over checkpoints), ``sd_within`` (median per-seed SD of step-to-step
        differences / √2 — the checkpoint jitter), and ``mdd`` (single-checkpoint minimal detectable
        walk-vs-IID gap of seed means at this seed count).
    """
    n_seeds = iid.shape[0]
    mean = float(np.nanmean(iid))
    sd_seed = float(np.sqrt(np.nanmean(np.nanvar(iid, axis=0, ddof=1))))
    sd_within = float(np.nanmedian(np.nanstd(np.diff(iid, axis=1), axis=1, ddof=1)) / math.sqrt(2.0))
    return {
        "mean": mean,
        "n_query": float(n_query),
        "ci95_halfwidth": wilson_halfwidth(mean, n_query),
        "sd_seed": sd_seed,
        "sd_within": sd_within,
        "mdd": float(t_multiplier(2 * n_seeds - 2) * sd_seed * math.sqrt(2.0 / n_seeds)),
    }


def split_masks(n_total: int, n_walk: int) -> np.ndarray:
    """Every split of ``n_total`` seeds into ``n_walk`` + the rest, as a ``[splits, n_total]`` 0/1 matrix."""
    combos = list(itertools.combinations(range(n_total), n_walk))
    masks = np.zeros((len(combos), n_total))
    for row, chosen in enumerate(combos):
        masks[row, list(chosen)] = 1.0
    return masks


def split_d(pooled: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """:func:`effect_sizes` for every split in ``masks`` at once, from per-split sums: ``[splits, checkpoints]``.

    ``nan`` readings are ignored; a checkpoint where both groups' readings coincide reads ``0`` when
    the means agree and ``nan`` otherwise. Unlike the loop, readings that coincide are recognized
    as such even when their mean is not exactly representable (the loop's ``nanvar`` then returns a
    rounding-level spread, and *d* a rounding ratio).

    Args:
        pooled: ``[seeds, checkpoints]``, both arms stacked.
        masks: ``[splits, seeds]``, 1 marks the "walk" group (see :func:`split_masks`).
    """
    x = np.asarray(pooled, dtype=float)
    finite = np.isfinite(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmean(x, axis=0)  # centring keeps the sums-of-squares variance exact enough
    x0 = np.where(finite, x - np.nan_to_num(centre), 0.0)
    f = finite.astype(float)
    cnt, tot, sq = f.sum(0), x0.sum(0), (x0**2).sum(0)
    c_a, s_a, q_a = masks @ f, masks @ x0, masks @ x0**2
    c_b, s_b, q_b = cnt - c_a, tot - s_a, sq - q_a
    with np.errstate(divide="ignore", invalid="ignore"):
        m_a, m_b = s_a / c_a, s_b / c_b
        # A group whose readings coincide has variance 0 exactly in the loop; from sums it comes out at
        # rounding level, relative to the group's mean square, so snap that to 0.
        v_a = np.where(c_a > 1, (q_a - s_a * m_a) / (c_a - 1), np.nan)
        v_b = np.where(c_b > 1, (q_b - s_b * m_b) / (c_b - 1), np.nan)
        v_a = np.where(v_a <= 1e-9 * q_a / c_a, 0.0, v_a)
        v_b = np.where(v_b <= 1e-9 * q_b / c_b, 0.0, v_b)
        diff = m_a - m_b
        diff = np.where(np.abs(diff) <= 1e-9 * np.maximum(np.abs(m_a), np.abs(m_b)), 0.0, diff)
        spread = np.sqrt((v_a + v_b) / 2.0)
        return np.where(spread > 0, diff / np.where(spread > 0, spread, 1.0), np.where(diff == 0, 0.0, np.nan))


def split_max_d(pooled: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """``max_t |d_t|`` per split (see :func:`split_d`); ``nan`` for a split with no defined checkpoint."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(np.abs(split_d(pooled, masks)), axis=1)


def gap_fast(walk: np.ndarray, iid: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """:func:`gap`, with the permutation null computed in one pass (5 + 15 seeds: 15,504 splits).

    The effect sizes come from :func:`split_d`, so coinciding readings (every seed's step-0 read of the
    same start, say) read as no gap rather than a rounding ratio.

    Args:
        walk: Walk arm ``[seeds, checkpoints]``.
        iid: IID arm ``[seeds, checkpoints]``.
        alpha: The family-wise level for ``frac_sig``.

    Returns:
        The same keys as :func:`gap`.

    Raises:
        ValueError: If the arms do not share a checkpoint count.
    """
    if walk.shape[1] != iid.shape[1]:
        raise ValueError(f"walk and IID checkpoint counts differ: {walk.shape[1]} vs {iid.shape[1]}.")
    pooled = np.vstack([walk, iid])
    identity = np.zeros((1, pooled.shape[0]))
    identity[0, : walk.shape[0]] = 1.0
    d = split_d(pooled, identity)[0]
    observed = float(np.nanmax(np.abs(d)))
    null = split_max_d(pooled, split_masks(pooled.shape[0], walk.shape[0]))
    d_crit = float(np.quantile(null, 1.0 - alpha))
    return {
        "max_abs_d": observed,
        "mean_abs_d": float(np.nanmean(np.abs(d))),
        "final_d": float(d[-1]),
        "final_gap": float(np.nanmean(walk, axis=0)[-1] - np.nanmean(iid, axis=0)[-1]),
        "p_max": float(np.mean(null >= observed - 1e-12)),
        "frac_sig": float(np.nanmean(np.abs(d) > d_crit)),
        "d_crit": d_crit,
        "n_perm": float(null.size),
    }


def drop_profile(n_checkpoints: int, shape: str, last: int | None = None) -> np.ndarray:
    """A unit drop's profile over a drive: ``sustained`` (every reading), ``last_quarter``, or ``single``.

    ``last`` is the index of the drive's last usable reading (default: the last); the last quarter and
    the single reading end there.
    """
    end = n_checkpoints - 1 if last is None else last
    profile = np.zeros(n_checkpoints)
    if shape == "sustained":
        profile[:] = 1.0
    elif shape == "last_quarter":
        profile[max(0, end + 1 - math.ceil((end + 1) / 4)) : end + 1] = 1.0
    elif shape == "single":
        profile[end] = 1.0
    else:
        raise ValueError(f"unknown drop shape {shape!r}.")
    return profile


def injected_power(
    iid: np.ndarray,
    profile: np.ndarray,
    delta: float,
    groups: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """The share of injections the max-over-checkpoints permutation test catches at drop ``delta``.

    Each row of ``groups`` picks the "walk" seeds; ``delta · profile`` is subtracted from them, and the
    walk-vs-rest test of :func:`gap` (every split in ``masks``) is run on the result.

    Args:
        iid: Healthy ``[seeds, checkpoints]`` (raw readings, ``nan`` = skipped).
        profile: The drop's shape over the drive (see :func:`drop_profile`).
        delta: The drop, in the reading's units.
        groups: ``[injections, seeds]`` 0/1 rows, each the injected group (the observed split).
        masks: Every split of the seeds into a group of the same size (see :func:`split_masks`).
        alpha: The test's level.

    Returns:
        The rejection rate over the injections.
    """
    caught = 0
    for group in groups:
        x = np.asarray(iid, dtype=float) - delta * np.outer(group, profile)
        null = split_max_d(x, masks)
        observed = split_max_d(x, group[None, :])[0]
        caught += int(np.mean(null >= observed - 1e-12) <= alpha)
    return caught / len(groups)


def injected_mdd(
    iid: np.ndarray,
    shape: str,
    *,
    n_inject: int = 5,
    n_groups: int = 200,
    alpha: float = 0.05,
    power: float = 0.8,
    seed: int = 0,
    hi: float = 0.05,
    tol: float = 5e-4,
) -> dict[str, float]:
    """The drop the walk-vs-IID test catches ``power`` of the time, by injection into healthy seeds (B2).

    ``n_inject`` of the healthy seeds play the walk, a known drop is injected into them, and the exact
    max-over-checkpoints permutation test is run against the rest — the test the null actually runs,
    not P1.1.0's single-checkpoint *t* figure. The same ``n_groups`` random walk groups are used at
    every drop (common random numbers), and the drop is found by bisection.

    Args:
        iid: Healthy ``[seeds, checkpoints]``.
        shape: The drop's shape (see :func:`drop_profile`).
        n_inject: Seeds playing the walk.
        n_groups: Random walk groups per drop (all of them if fewer exist).
        alpha: The test's level.
        power: The detection rate the MDD is defined at.
        seed: The group sampler's seed.
        hi: The first upper bracket (doubled until caught, up to 1.0).
        tol: The bisection's resolution.

    Returns:
        ``mdd`` (``inf`` if not reached at 1.0), ``power_at_mdd``, ``n_groups``, ``n_splits``.
    """
    x = np.asarray(iid, dtype=float)
    n = x.shape[0]
    finite_cols = np.flatnonzero(np.isfinite(x).any(axis=0))
    profile = drop_profile(x.shape[1], shape, int(finite_cols[-1]))
    masks = split_masks(n, n_inject)
    rng = np.random.default_rng(seed)
    pick = masks if masks.shape[0] <= n_groups else masks[rng.choice(masks.shape[0], n_groups, replace=False)]
    lo = 0.0
    while injected_power(x, profile, hi, pick, masks, alpha) < power:
        lo, hi = hi, hi * 2
        if hi > 1.0:
            return {
                "mdd": math.inf,
                "power_at_mdd": math.nan,
                "n_groups": float(len(pick)),
                "n_splits": float(len(masks)),
            }
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if injected_power(x, profile, mid, pick, masks, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return {
        "mdd": hi,
        "power_at_mdd": injected_power(x, profile, hi, pick, masks, alpha),
        "n_groups": float(len(pick)),
        "n_splits": float(len(masks)),
    }
