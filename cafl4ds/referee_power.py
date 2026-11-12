"""P1.1.4 Part 4 — how small a past-regime loss the labelled referee can resolve, and what pool that needs.

P1.3 reads forgetting on a regime the stream has left: the fall in the regime's balanced accuracy from
the checkpoint ``t0`` at which the stream left it to each later checkpoint ``t``,
``L(t) = bacc(t0) − bacc(t)`` (mean over seeds, each seed against its own ``t0``). It asks whether the
loss is real at *any* later checkpoint, so the test is the maximum over checkpoints, and family-wise
over the regimes read.

The noise of ``L`` has two parts, and only one shrinks with a larger held-out pool:

* the **image** part — which query frames happen to be in the slice — scales as ``1/√n``;
* the **seed** part — how differently seeds of one recipe move — does not; only more seeds shrink it.

:func:`change_noise` draws each part separately by bootstrap (seeds fixed / images fixed), jointly over
the later checkpoints so their correlation is kept. :func:`mdd` recombines them at a chosen pool and
seed multiplier and finds the loss the max test catches with the target power. Balanced accuracy is
linear in per-image correctness, so a seed's change is the balanced accuracy of its per-image
difference (as in :func:`cafl4ds.adaptation.bootstrap_combination`). Pure numpy.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def _bacc_rows(rows: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Balanced accuracy of each row of per-image values ``[S, N]`` (linear: works on differences too)."""
    classes = np.unique(labels)
    return np.mean([rows[:, labels == c].mean(axis=1) for c in classes], axis=0)


def change_noise(
    start: np.ndarray,
    later: list[np.ndarray],
    labels: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """The loss ``L(t)`` at each later checkpoint, and its seed and image bootstrap noise, centred.

    Args:
        start: Per-image correctness at ``t0`` ``[S, N]``.
        later: Per-image correctness at each later checkpoint, ``[S, N]`` each (the same seeds, in order).
        labels: Class per image ``[N]``.
        n_boot: Resamples per component.
        seed: RNG seed.

    Returns:
        ``loss`` ``[K]`` (the point estimate per later checkpoint); ``seed_noise`` and ``image_noise``
        ``[n_boot, K]`` — each resample's ``L`` minus ``loss``, the seeds resampled with the images
        fixed, and the images resampled within each class with the seeds fixed; both draws shared
        across the ``K`` checkpoints.

    Raises:
        ValueError: If the shapes disagree or ``later`` is empty.
    """
    start = np.atleast_2d(np.asarray(start, dtype=np.float64))
    if not later:
        raise ValueError("need at least one later checkpoint.")
    diffs = [start - np.atleast_2d(np.asarray(x, dtype=np.float64)) for x in later]
    if any(d.shape != start.shape for d in diffs) or start.shape[1] != labels.shape[0]:
        raise ValueError("start, later and labels must share their seed and image axes.")
    n_seeds = start.shape[0]
    per_seed = np.stack([_bacc_rows(d, labels) for d in diffs], axis=1)  # [S, K]
    loss = per_seed.mean(axis=0)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    seed_noise = per_seed[draw].mean(axis=1) - loss  # [B, K]
    classes = np.unique(labels)
    image = np.zeros((n_boot, len(diffs)))
    for c in classes:
        members = np.flatnonzero(labels == c)
        weights = rng.multinomial(members.size, np.full(members.size, 1.0 / members.size), size=n_boot)
        for k, d in enumerate(diffs):
            image[:, k] += (weights @ d[:, members].T / members.size).mean(axis=1)
    image_noise = image / classes.size - loss
    return {"loss": loss, "seed_noise": seed_noise, "image_noise": image_noise}


def combined_noise(
    seed_noise: np.ndarray, image_noise: np.ndarray, *, pool: float = 1.0, seeds: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Seed and image noise recombined at ``pool``× the images and ``seeds``× the seeds.

    The two components are independent draws, paired at random; the image part scales as
    ``1/√pool`` and the seed part as ``1/√seeds``.
    """
    order = np.random.default_rng(seed).permutation(image_noise.shape[0])
    return seed_noise / math.sqrt(seeds) + image_noise[order] / math.sqrt(pool)


def max_test_power(noise: np.ndarray, shape: np.ndarray, delta: float, critical: float) -> float:
    """Share of resamples in which ``max_t (noise_t + delta · shape_t)`` exceeds ``critical``."""
    return float(np.mean(np.max(noise + delta * np.asarray(shape, dtype=float), axis=1) > critical))


def mdd(
    noise: np.ndarray,
    shape: np.ndarray,
    *,
    alpha: float,
    power: float = 0.8,
    hi: float = 1.0,
    tol: float = 1e-4,
) -> dict[str, float]:
    """The smallest loss the max-over-checkpoints test catches with ``power``, at level ``alpha``.

    Args:
        noise: Centred resamples of ``L`` ``[B, K]`` (the null: no loss).
        shape: The loss's profile over the ``K`` checkpoints (``1`` = the full loss there; e.g.
            all ones for a sustained loss, a single one for a loss at one checkpoint).
        alpha: The one-sided level for this slice (already split over the slices tested).
        power: The target power.
        hi: Upper end of the bisection (balanced accuracy cannot fall by more than 1).
        tol: Bisection tolerance.

    Returns:
        ``mdd``, and the ``critical`` value of the max under the null.
    """
    critical = float(np.quantile(np.max(noise, axis=1), 1.0 - alpha))
    lo = 0.0
    if max_test_power(noise, shape, hi, critical) < power:
        return {"mdd": math.inf, "critical": critical}
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if max_test_power(noise, shape, mid, critical) >= power:
            hi = mid
        else:
            lo = mid
    return {"mdd": hi, "critical": critical}


def dense_grid_mdd(sd: float, n_checkpoints: int, *, alpha: float, power: float = 0.8) -> float:
    """A conservative MDD on a denser grid: Bonferroni over ``n_checkpoints``, a sustained loss, Gaussian noise.

    For P1.3's dense checkpoints, whose correlation is not yet measured. Treating them as
    independent makes the critical value too high, so this bounds the true MDD from above.
    """
    nd = NormalDist()
    return float((nd.inv_cdf(1 - alpha / n_checkpoints) + nd.inv_cdf(power)) * sd)


def pool_for(
    seed_noise: np.ndarray,
    image_noise: np.ndarray,
    shape: np.ndarray,
    target: float,
    *,
    alpha: float,
    seeds: float = 1.0,
    power: float = 0.8,
    max_pool: float = 1024.0,
) -> float:
    """The smallest pool multiplier whose MDD reaches ``target`` (``inf`` if the seed part alone forbids it).

    The pool is doubled until the target is met, then bisected on a log scale.
    """

    def ok(pool: float) -> bool:
        noise = combined_noise(seed_noise, image_noise, pool=pool, seeds=seeds)
        return mdd(noise, shape, alpha=alpha, power=power)["mdd"] <= target

    if ok(1.0):
        return 1.0
    hi = 2.0
    while not ok(hi):
        hi *= 2
        if hi > max_pool:
            return math.inf
    lo = hi / 2
    while hi / lo > 1.05:
        mid = math.sqrt(lo * hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi
