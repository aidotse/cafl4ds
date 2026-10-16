"""The P1.1.1 reading protocol — call one live trajectory against the healthy null, per instrument.

P1.1.0 measured how much a healthy (IID) drive moves each instrument. This turns that null into a
**single-trajectory decision rule**: a deployed monitor sees one drive, not a five-seed ensemble, so
each instrument is read against precomputed **null bands** — the IID seed mean per checkpoint and a
pooled seed spread — and fires when the trajectory leaves them in its harmful direction.

The rule, per instrument:

* **Transform.** Drift instruments are positive with multiplicative, spiky noise (a healthy seed's
  drift can jump 4× for a checkpoint), so they are read on a log scale; the rest are read raw.
* **Residual.** ``r_t = (x_t − μ_t) · direction``, with ``μ_t`` the null's per-checkpoint seed mean and
  ``direction`` the instrument's harmful sign (RankMe and the canary fall, drift rises), so a positive
  residual is always "worse than healthy". Two-sided instruments use ``|r_t|``.
* **Smoothing.** An optional trailing mean over ``window`` checkpoints trades detection delay for
  noise — the canary jitters reading to reading. The window is shorter (and the mean noisier) over a
  trajectory's first checkpoints, so each ``z`` is scaled by the smoothed residual's own SD at its
  window length, under the residuals' AR(1) autocorrelation.
* **Scale.** A local SD of the raw residual per checkpoint, from the null seeds' leave-one-out
  residuals — each seed against the mean of the *others*, so the spread is the one a new trajectory
  will meet. It is pooled over the seeds *and* a band of ``±span`` neighbouring checkpoints: five
  seeds alone give a useless per-checkpoint SD, while one SD for the whole drive misreads instruments
  whose healthy spread grows (drift starts at zero) or spikes (a short final batch). A floor at a
  fraction of the drive-wide SD keeps a near-zero local spread from turning noise into alarms.
* **Threshold.** ``z_t = r_t / sd`` fires once it exceeds ``c``. One ``c`` is shared by a whole
  *panel* of instruments (the label-free ones; the canary channels), set so a healthy trajectory's
  maximum ``z`` — over the drive *and* the panel's instruments — exceeds it with probability
  ``alpha``. It is read off the null itself (:func:`panel_threshold`): each null seed is scored
  against bands calibrated on the others, its panel maximum recorded, and a Gumbel law — the
  limit law of a maximum — fitted to those maxima by moments, extrapolating to the ``1 − alpha``
  quantile that a handful of seeds cannot reach directly. The threshold never sits below the
  largest healthy maximum, so no calibration seed fires. Sharing ``c`` across a panel prices in
  the instruments' correlation (drift's two forms move together) without a Bonferroni split.

An earlier version simulated the threshold from an AR(1)-plus-offset model of the residuals; drift
behaves like a random walk, and that model's threshold ran low out of sample. The threshold is
therefore **validated, not trusted**: :func:`leave_one_out_fired` replays the whole calibration
(bands *and* threshold) without one seed and calls it, and held-out IID seeds are read the same
way. Pure numpy, unit-testable.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import Any, TypedDict

import numpy as np

# The harmful direction of each instrument: -1 = a fall is harm, +1 = a rise is harm. Anything not
# listed is read two-sided. Canary slices (``knn_acc_night``, …) inherit their base's direction.
_DIRECTION = {
    "rankme": -1,
    "rankme_proj": -1,
    "knn_acc": -1,
    "linear_acc": -1,
    "cka_drift": 1,
    "cka_drift_proj": 1,
    "cosine_drift": 1,
    "cosine_drift_proj": 1,
}

# Euler's constant: a Gumbel law's mean sits this many scale units above its location.
_EULER_GAMMA = 0.5772156649015329

# Floor added before the log, so a drift of exactly 0 (the anchor itself) stays finite.
_LOG_EPS = 1e-4


def default_transform(signal: str) -> str:
    """``"log"`` for the drift instruments (positive, multiplicative noise), ``"none"`` otherwise."""
    return "log" if "drift" in signal else "none"


def apply_transform(values: np.ndarray, transform: str) -> np.ndarray:
    """Map raw readings onto the scale the rule reads them on (``"none"`` or ``"log"``).

    Raises:
        ValueError: If ``transform`` is unknown.
    """
    x = np.asarray(values, dtype=float)
    if transform == "none":
        return x
    if transform == "log":
        return np.log(np.abs(x) + _LOG_EPS)
    raise ValueError(f"transform must be 'none' or 'log'; got {transform!r}.")


def direction(signal: str) -> int:
    """The harmful sign of ``signal`` (``-1`` / ``+1``), or ``0`` for a two-sided read."""
    if signal in _DIRECTION:
        return _DIRECTION[signal]
    for base in ("knn_acc", "linear_acc"):
        if signal.startswith(f"{base}_"):
            return _DIRECTION[base]
    return 0


def trailing_mean(matrix: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over ``window`` checkpoints along the last axis (shorter at the start).

    Args:
        matrix: ``[..., checkpoints]``.
        window: Checkpoints averaged (``1`` = no smoothing).

    Returns:
        The smoothed array, same shape (``nan`` entries are skipped within a window).
    """
    if window <= 1:
        return np.asarray(matrix, dtype=float)
    x = np.asarray(matrix, dtype=float)
    out = np.empty_like(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-skipped window stays nan
        for t in range(x.shape[-1]):
            out[..., t] = np.nanmean(x[..., max(0, t - window + 1) : t + 1], axis=-1)
    return out


def _lag1_autocorrelation(residuals: np.ndarray) -> float:
    """Pooled lag-1 autocorrelation of residual trajectories ``[seeds, checkpoints]`` (clipped to [0, 1))."""
    a, b = residuals[:, :-1].ravel(), residuals[:, 1:].ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return 0.0
    rho = float(np.corrcoef(a[keep], b[keep])[0, 1])
    return min(max(rho, 0.0), 0.99)


@dataclass(frozen=True)
class Bands:
    """A calibrated single-trajectory rule for one instrument.

    Attributes:
        signal: The instrument.
        mean: Null seed mean per checkpoint.
        sd: Local SD of the raw leave-one-out residual per checkpoint (rescaled to a new trajectory's).
        threshold: The ``z`` a trajectory must exceed to fire.
        direction: Harmful sign (``0`` = two-sided).
        window: Trailing-mean window.
        transform: The scale the instrument is read on (``"none"`` / ``"log"``).
        rho: Lag-1 autocorrelation of the residuals within a seed (scales the smoothed residual).
        n_seeds: Null seeds calibrated on.
    """

    signal: str
    mean: np.ndarray
    sd: np.ndarray
    threshold: float
    direction: int
    window: int
    transform: str
    rho: float
    n_seeds: int

    def restrict(self, index: np.ndarray | list[int]) -> Bands:
        """The same rule on a subset of the checkpoint grid (e.g. a branch resumed mid-drive).

        The threshold is kept: it was set for the full grid, so on a shorter one it is conservative.
        """
        index = np.asarray(index)
        return replace(self, mean=self.mean[index], sd=self.sd[index])


def smoothing_profile(rho: float, window: int, n_checkpoints: int) -> np.ndarray:
    """SD of a trailing mean of unit AR(1) residuals at each checkpoint (the window fills from the start).

    The mean of ``m`` consecutive AR(1) values has variance ``(1/m²) Σ_ij rho^|i−j|``; at checkpoint
    ``t`` the trailing window holds ``m = min(t + 1, window)`` values.

    Args:
        rho: Lag-1 autocorrelation.
        window: Trailing-mean window.
        n_checkpoints: Trajectory length.

    Returns:
        The SD per checkpoint (all ones at ``window <= 1``).
    """
    sizes = np.minimum(np.arange(n_checkpoints) + 1, max(window, 1))
    lags = np.abs(np.subtract.outer(np.arange(max(window, 1)), np.arange(max(window, 1))))
    corr = float(rho) ** lags
    sd_of = {m: math.sqrt(float(corr[:m, :m].sum())) / m for m in set(sizes.tolist())}
    return np.array([sd_of[m] for m in sizes.tolist()])


def local_sd(residuals: np.ndarray, span: int, floor: float) -> np.ndarray:
    """RMS of ``residuals`` ``[seeds, checkpoints]`` over all seeds and checkpoints ``t ± span``, floored.

    Args:
        residuals: The null's residual trajectories.
        span: Neighbouring checkpoints pooled on each side.
        floor: Lower bound, as a fraction of the drive-wide RMS.

    Returns:
        The local SD per checkpoint.
    """
    sq = np.asarray(residuals, dtype=float) ** 2
    n_t = sq.shape[1]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-skipped band is nan, then floored
        local = np.array([np.sqrt(np.nanmean(sq[:, max(0, t - span) : t + span + 1])) for t in range(n_t)])
    return np.fmax(local, floor * float(np.sqrt(np.nanmean(sq))))


def fit_bands(
    null: np.ndarray,
    signal: str,
    *,
    window: int = 1,
    span: int = 15,
    floor: float = 0.25,
    transform: str | None = None,
    threshold: float = math.nan,
) -> Bands:
    """Fit the bands for ``signal`` on null (healthy IID) trajectories — the mean and local SD, no threshold.

    Args:
        null: ``[seeds, checkpoints]`` (at least 3 seeds).
        signal: The instrument (sets the harmful direction).
        window: Trailing-mean window.
        span: Neighbouring checkpoints (each side) pooled into the local SD.
        floor: The local SD's floor, as a fraction of the drive-wide SD.
        transform: The reading scale; ``None`` picks :func:`default_transform`.
        threshold: The threshold to carry (``nan`` until a panel sets it).

    Returns:
        The fitted :class:`Bands`.

    Raises:
        ValueError: If fewer than 3 seeds are given.
    """
    transform = transform or default_transform(signal)
    x = apply_transform(null, transform)  # nan marks a skipped checkpoint (e.g. after a short batch)
    n = x.shape[0]
    if n < 3:
        raise ValueError(f"need >= 3 null seeds to fit bands; got {n}.")
    total = np.nansum(x, axis=0)
    counts = np.sum(np.isfinite(x), axis=0)
    loo = np.stack([(total - np.nan_to_num(x[i])) / np.maximum(counts - np.isfinite(x[i]), 1) for i in range(n)])
    raw = x - loo
    # A leave-one-out residual carries the left-out mean's error too: it is (1 + 1/(n-1))× a new
    # trajectory's against the full-n mean's (1 + 1/n)×, so rescale to the latter.
    sd = local_sd(raw, span, floor) * math.sqrt((1 + 1 / n) / (1 + 1 / (n - 1)))
    unit = raw / local_sd(raw, span, floor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a skipped (all-nan) checkpoint has no mean
        within = unit - np.nanmean(unit, axis=1, keepdims=True)
        mean = np.nanmean(x, axis=0)
    return Bands(
        signal=signal,
        mean=mean,
        sd=sd,
        threshold=threshold,
        direction=direction(signal),
        window=window,
        transform=transform,
        rho=_lag1_autocorrelation(within),
        n_seeds=n,
    )


def max_z(trajectory: np.ndarray, bands: Bands) -> float:
    """A trajectory's largest ``z`` over the drive (``-inf`` if every reading is skipped)."""
    z = zscores(trajectory, bands)
    return float(np.nanmax(z)) if np.isfinite(z).any() else -math.inf


def leave_one_out_max_z(
    null: np.ndarray,
    signal: str,
    *,
    window: int = 1,
    span: int = 15,
    floor: float = 0.25,
    transform: str | None = None,
) -> np.ndarray:
    """Each null seed's max ``z`` against bands fitted on the other seeds (the settings as :func:`fit_bands`)."""
    x = np.asarray(null, dtype=float)
    return np.array(
        [
            max_z(
                x[i], fit_bands(np.delete(x, i, 0), signal, window=window, span=span, floor=floor, transform=transform)
            )
            for i in range(x.shape[0])
        ]
    )


def panel_threshold(maxima: np.ndarray, alpha: float) -> float:
    """The ``1 − alpha`` quantile of a Gumbel law fitted (by moments) to healthy max-``z`` values, floored at their max.

    A trajectory's maximum ``z`` is a maximum over many correlated readings, and maxima tend to a
    Gumbel law; its two moments are fitted to the observed maxima, so the threshold extrapolates
    past the few seeds at hand. The floor keeps every observed healthy trajectory quiet.

    Args:
        maxima: One healthy max ``z`` per seed (leave-one-out, so each was scored out of sample).
        alpha: The false-alarm level the threshold targets.

    Returns:
        The threshold.

    Raises:
        ValueError: If fewer than 2 finite maxima are given.
    """
    m = np.asarray(maxima, dtype=float)
    m = m[np.isfinite(m)]
    if m.size < 2:
        raise ValueError(f"need >= 2 finite maxima to fit a threshold; got {m.size}.")
    beta = float(np.std(m, ddof=1)) * math.sqrt(6) / math.pi
    mu = float(np.mean(m)) - _EULER_GAMMA * beta
    return max(mu - beta * math.log(-math.log(1 - alpha)), float(m.max()))


def calibrate_panel(
    nulls: dict[str, np.ndarray], settings: dict[str, dict[str, Any]], alpha: float = 0.05
) -> dict[str, Bands]:
    """Calibrate a panel of instruments: fit each one's bands, then one threshold shared by the panel.

    Each seed's panel maximum is the largest leave-one-out max ``z`` over the panel's instruments;
    :func:`panel_threshold` turns those maxima into the threshold, so "any instrument fired" holds
    ``alpha`` for the panel as a whole.

    Args:
        nulls: Instrument → its null trajectories ``[seeds, checkpoints]`` (the same seeds, in order).
        settings: Instrument → :func:`fit_bands` keyword arguments (window, span, floor, transform).
        alpha: The panel's family-wise false-alarm level over the whole trajectory.

    Returns:
        Instrument → calibrated :class:`Bands`.

    Raises:
        ValueError: If fewer than 4 seeds are given (each leave-one-out fit needs 3).
    """
    n = {np.asarray(x).shape[0] for x in nulls.values()}
    if len(n) != 1 or min(n) < 4:
        raise ValueError(f"need >= 4 null seeds, the same for every instrument, to calibrate; got {sorted(n)}.")
    maxima = np.max([leave_one_out_max_z(nulls[s], s, **settings.get(s, {})) for s in nulls], axis=0)
    threshold = panel_threshold(maxima, alpha)
    return {s: fit_bands(nulls[s], s, threshold=threshold, **settings.get(s, {})) for s in nulls}


def calibrate(
    null: np.ndarray,
    signal: str,
    *,
    alpha: float = 0.05,
    window: int = 1,
    span: int = 15,
    floor: float = 0.25,
    transform: str | None = None,
) -> Bands:
    """Calibrate one instrument on its own — a panel of one (the settings as :func:`fit_bands`)."""
    settings: dict[str, Any] = {"window": window, "span": span, "floor": floor, "transform": transform}
    return calibrate_panel({signal: null}, {signal: settings}, alpha)[signal]


def zscores(trajectory: np.ndarray, bands: Bands) -> np.ndarray:
    """The rule's ``z`` per checkpoint for one trajectory (positive = worse than healthy).

    Args:
        trajectory: ``[checkpoints]`` on the bands' grid (or ``[seeds, checkpoints]``).
        bands: The calibrated rule.

    Returns:
        ``z`` with the trajectory's shape (two-sided rules return ``|z|``).
    """
    r = trailing_mean(apply_transform(trajectory, bands.transform) - bands.mean, bands.window)
    scale = bands.sd * smoothing_profile(bands.rho, bands.window, r.shape[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(scale > 0, r / np.where(scale > 0, scale, 1.0), np.where(r == 0, 0.0, np.inf * np.sign(r)))
    return z * bands.direction if bands.direction else np.abs(z)


class Call(TypedDict):
    """One trajectory's call under one rule (see :func:`call`)."""

    fired: bool
    first: int | None
    max_z: float
    frac_fired: float
    final_z: float


def call(trajectory: np.ndarray, bands: Bands) -> Call:
    """Call one trajectory: does it leave the null bands in the harmful direction, and when?

    Args:
        trajectory: ``[checkpoints]`` on the bands' grid.
        bands: The calibrated rule.

    Returns:
        ``fired``, ``first`` (index of the first firing checkpoint, ``None`` if quiet), ``max_z``,
        ``frac_fired`` (share of checkpoints over threshold), ``final_z``.
    """
    z = zscores(trajectory, bands)
    over = np.nan_to_num(z, nan=-np.inf) > bands.threshold
    first = int(np.argmax(over)) if over.any() else None
    return {
        "fired": bool(over.any()),
        "first": first,
        "max_z": float(np.nanmax(z)),
        "frac_fired": float(over.mean()),
        "final_z": float(z[-1]),
    }


def false_alarm_rate(trajectories: np.ndarray, bands: Bands) -> float:
    """Share of healthy trajectories ``[seeds, checkpoints]`` the rule fires on (its realized FPR)."""
    return float(np.mean([call(row, bands)["fired"] for row in np.asarray(trajectories, dtype=float)]))


def leave_one_out_fired(
    nulls: dict[str, np.ndarray], settings: dict[str, dict[str, Any]], alpha: float = 0.05
) -> list[bool]:
    """Recalibrate the panel without each null seed in turn — bands *and* threshold — and call the seed left out.

    The honest out-of-sample check on the null itself: the left-out seed shaped neither the bands it
    is read against nor the threshold (whose own leave-one-out runs on the remaining seeds).

    Args:
        nulls: Instrument → null trajectories ``[seeds, checkpoints]`` (at least 5 seeds).
        settings: Instrument → :func:`fit_bands` keyword arguments.
        alpha: The panel's false-alarm level.

    Returns:
        Per seed, whether any of the panel's instruments fired on it.
    """
    n = next(iter(nulls.values())).shape[0]
    fired = []
    for i in range(n):
        rules = calibrate_panel({s: np.delete(x, i, axis=0) for s, x in nulls.items()}, settings, alpha)
        fired.append(any(call(np.asarray(nulls[s])[i], rules[s])["fired"] for s in nulls))
    return fired


def adjudicate(fired: dict[str, bool], arbiters: list[str]) -> str:
    """Name what a set of instrument calls means, with the labelled canary as the arbiter of information loss.

    * ``quiet`` — nothing fired.
    * ``degradation`` — an arbiter (a canary channel) fired: the representation lost information the
      probe can see, whatever the label-free instruments say.
    * ``movement`` — only label-free instruments fired, with every arbiter quiet: the representation
      moved beyond healthy without a visible information loss. Benign reorganization *or* a loss the
      canary cannot resolve; the recovery test and the positive controls decide which.

    Args:
        fired: Instrument → did it fire.
        arbiters: The instruments that arbitrate (the canary channels).

    Returns:
        ``"quiet"``, ``"movement"`` or ``"degradation"``.
    """
    if any(fired.get(a, False) for a in arbiters):
        return "degradation"
    if any(v for k, v in fired.items() if k not in arbiters):
        return "movement"
    return "quiet"


def trailing_diversity(eras: np.ndarray | list[int], window: int) -> np.ndarray:
    """Distinct regimes among the last ``window`` batches at each checkpoint — how mixed the recent diet is.

    On the walk this is 1 deep inside a leg and climbs through a run of short legs (BDD's dawn/dusk
    stretch); the IID control sits near ``window`` throughout.

    Args:
        eras: The era (regime walk position) of each checkpoint's batch, in order.
        window: Trailing batches counted.

    Returns:
        The distinct-regime count per checkpoint.
    """
    seq = list(eras)
    return np.array([len(set(seq[max(0, t - window + 1) : t + 1])) for t in range(len(seq))], dtype=float)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two equal-length series (average ranks for ties; ``nan`` if one is constant)."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    rx, ry = _ranks(x[keep]), _ranks(y[keep])
    if rx.size < 3 or np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean of their positions)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size)
    ranks[order] = np.arange(x.size, dtype=float)
    for value in np.unique(x):
        tied = x == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def short_batch_steps(segments: list[dict[str, int]], batch_size: int, min_batch: int) -> set[int]:
    """Steps whose batch held fewer than ``min_batch`` images — the readings the rule skips.

    Each leg of the walk streams ``n_images`` in full batches and flushes the remainder as one short
    batch at its last step, so a leg of 3 frames is a 3-image update. Such an update is noise (P1.1.0
    read its loss and grad norm as such), and the readings right after it jump in healthy drives too.
    The IID control copies the walk's batch sizes, so the same steps are short in both orders.

    Args:
        segments: The corpus leg map (``segments.json``'s ``segments``): ``start_step``, ``end_step``,
            ``n_images`` per leg.
        batch_size: The drive's batch size.
        min_batch: The smallest batch whose following reading is kept.

    Returns:
        The steps to skip.
    """
    short = set()
    for leg in segments:
        steps = int(leg["end_step"]) - int(leg["start_step"]) + 1
        last = int(leg["n_images"]) - batch_size * (steps - 1)
        if last < min_batch:
            short.add(int(leg["end_step"]))
    return short
