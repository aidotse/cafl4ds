"""Unit tests for the single-trajectory reading protocol (``cafl4ds.reading_protocol``)."""

from __future__ import annotations

import numpy as np
import pytest

from cafl4ds.reading_protocol import (
    adjudicate,
    apply_transform,
    calibrate,
    calibrate_panel,
    call,
    default_transform,
    direction,
    false_alarm_rate,
    leave_one_out_fired,
    leave_one_out_max_z,
    local_sd,
    panel_threshold,
    short_batch_steps,
    smoothing_profile,
    spearman,
    trailing_diversity,
    trailing_mean,
    zscores,
)


def _ar1(n: int, t: int = 120, rho: float = 0.7, seed: int = 0, level: float = 10.0) -> np.ndarray:
    """Autocorrelated null trajectories around a shared drifting mean (like a healthy drive's instrument)."""
    rng = np.random.default_rng(seed)
    noise = np.zeros((n, t))
    for k in range(1, t):
        noise[:, k] = rho * noise[:, k - 1] + rng.normal(size=n) * np.sqrt(1 - rho**2)
    return level + np.linspace(0.0, 1.0, t) + noise


def test_direction_reads_harm_by_instrument() -> None:
    """Direction reads harm by instrument."""
    assert direction("rankme_proj") == -1
    assert direction("cka_drift") == 1
    assert direction("linear_acc_night") == -1
    assert direction("uniformity") == 0


def test_drift_is_read_on_a_log_scale() -> None:
    """Drift is read on a log scale."""
    assert default_transform("cka_drift_proj") == "log" and default_transform("rankme") == "none"
    assert np.isfinite(apply_transform(np.array([0.0, 0.1]), "log")).all()
    bands = calibrate(np.exp(_ar1(10, seed=16) / 5), "cka_drift")
    assert bands.transform == "log"
    with pytest.raises(ValueError, match="transform"):
        apply_transform(np.ones(2), "sqrt")


def test_trailing_mean_averages_the_window_and_shortens_at_the_start() -> None:
    """Trailing mean averages the window and shortens at the start."""
    out = trailing_mean(np.array([[1.0, 3.0, 5.0, 7.0]]), 2)
    assert out.tolist() == [[1.0, 2.0, 4.0, 6.0]]
    assert trailing_mean(np.array([1.0, 2.0]), 1).tolist() == [1.0, 2.0]


def test_calibration_holds_its_false_alarm_level_on_fresh_null_trajectories() -> None:
    """Calibration holds its false alarm level on fresh null trajectories."""
    bands = calibrate(_ar1(40, seed=1), "rankme", alpha=0.05)
    fpr = false_alarm_rate(_ar1(400, seed=2), bands)
    assert 0.0 < fpr < 0.12
    assert 0.5 < bands.rho < 0.9  # the null's autocorrelation is recovered


def test_a_harmful_shift_fires_at_its_onset_and_a_harmless_one_does_not() -> None:
    """A harmful shift fires at its onset and a harmless one does not."""
    bands = calibrate(_ar1(10, seed=3), "rankme")
    fresh = _ar1(1, seed=4)[0]
    fallen, risen = fresh.copy(), fresh.copy()
    fallen[60:] -= 8.0
    risen[60:] += 8.0
    out = call(fallen, bands)
    assert out["fired"] and out["first"] == 60 and out["max_z"] > bands.threshold
    assert not call(risen, bands)["fired"]  # a RankMe rise is not harm


def test_a_two_sided_instrument_fires_either_way() -> None:
    """A two sided instrument fires either way."""
    bands = calibrate(_ar1(10, seed=5), "uniformity")
    up = _ar1(1, seed=6)[0]
    up[80:] += 8.0
    assert call(up, bands)["fired"]
    assert (zscores(up, bands) >= 0).all()


def test_smoothing_quiets_the_noise_but_still_fires_on_a_sustained_shift() -> None:
    """Smoothing quiets the noise but still fires on a sustained shift."""
    null = _ar1(10, seed=7, rho=0.0)
    smooth = calibrate(null, "knn_acc", window=8)
    profile = smoothing_profile(smooth.rho, 8, 120)
    assert profile[0] == pytest.approx(1.0) and profile[-1] < 0.5  # the window fills, the noise shrinks
    shifted = _ar1(1, seed=8, rho=0.0)[0]
    shifted[50:] -= 1.5
    out = call(shifted, smooth)
    assert out["fired"] and out["first"] is not None and 50 <= out["first"] <= 58


def test_smoothed_rule_holds_its_false_alarm_level() -> None:
    """Smoothed rule holds its false alarm level."""
    bands = calibrate(_ar1(40, seed=11, rho=0.3), "knn_acc", window=6)
    assert 0.0 < false_alarm_rate(_ar1(400, seed=12, rho=0.3), bands) < 0.12


def test_restrict_slices_the_mean_and_keeps_the_threshold() -> None:
    """Restrict slices the mean and keeps the threshold."""
    bands = calibrate(_ar1(5, seed=9), "rankme")
    sub = bands.restrict(list(range(65, 120)))
    assert sub.mean.shape == sub.sd.shape == (55,) and sub.threshold == bands.threshold
    assert np.array_equal(sub.mean, bands.mean[65:])


def test_calibration_needs_four_seeds() -> None:
    """Calibration needs four seeds (each leave-one-out fit needs three)."""
    with pytest.raises(ValueError, match="4 null seeds"):
        calibrate(_ar1(3), "rankme")


def _random_walk(n: int, seed: int, t: int = 120) -> np.ndarray:
    """Drift-like null trajectories: a per-seed offset plus a random walk, near-unit lag-1 autocorrelation."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 1)) * 0.3 + np.cumsum(rng.normal(size=(n, t)) * 0.1, axis=1)


def test_ten_seed_calibration_holds_its_level_on_random_walk_nulls() -> None:
    """Calibrated on 10 seeds of a random-walk null with per-seed offsets (drift-like), the rule holds near level."""
    fprs = [
        false_alarm_rate(_random_walk(200, seed=100 + k), calibrate(_random_walk(10, seed=k), "rankme"))
        for k in range(4)
    ]
    assert np.mean(fprs) < 0.12


def test_panel_threshold_extrapolates_and_never_undercuts_the_healthy_max() -> None:
    """The panel threshold extrapolates past the typical maximum, and never sits below the largest one."""
    maxima = np.array([2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0])
    assert panel_threshold(maxima, 0.05) > 3.0
    assert panel_threshold(np.array([2.0, 2.1, 2.0, 6.0]), 0.5) == pytest.approx(6.0)
    assert panel_threshold(np.array([1.0, 2.0, 3.0]), 0.01) > panel_threshold(np.array([1.0, 2.0, 3.0]), 0.10)
    with pytest.raises(ValueError, match="2 finite maxima"):
        panel_threshold(np.array([3.0, np.nan]), 0.05)


def test_a_panel_shares_one_threshold_and_holds_its_level_as_a_whole() -> None:
    """A panel's instruments share one threshold, set so that "any instrument fired" holds the level."""
    nulls = {"rankme": _ar1(30, seed=20), "knn_acc": _ar1(30, seed=21, level=0.5)}
    rules = calibrate_panel(nulls, {"knn_acc": {"window": 3}}, alpha=0.05)
    assert rules["rankme"].threshold == rules["knn_acc"].threshold
    assert rules["knn_acc"].window == 3 and rules["rankme"].window == 1
    maxima = np.maximum(
        leave_one_out_max_z(nulls["rankme"], "rankme"), leave_one_out_max_z(nulls["knn_acc"], "knn_acc", window=3)
    )
    assert rules["rankme"].threshold >= maxima.max()
    fresh = {"rankme": _ar1(300, seed=22), "knn_acc": _ar1(300, seed=23, level=0.5)}
    fired = [any(call(fresh[s][i], rules[s])["fired"] for s in rules) for i in range(300)]
    assert np.mean(fired) < 0.12


def test_adjudication_lets_the_canary_arbitrate() -> None:
    """Adjudication lets the canary arbitrate."""
    arbiters = ["linear_acc", "knn_acc"]
    assert adjudicate({"rankme_proj": False, "linear_acc": False}, arbiters) == "quiet"
    assert adjudicate({"rankme_proj": True, "linear_acc": False}, arbiters) == "movement"
    assert adjudicate({"rankme_proj": False, "knn_acc": True}, arbiters) == "degradation"


def test_trailing_diversity_counts_distinct_recent_regimes() -> None:
    """Trailing diversity counts distinct recent regimes."""
    assert trailing_diversity([0, 0, 1, 2, 2, 2], 3).tolist() == [1, 1, 2, 3, 2, 1]


def test_spearman_handles_ties_and_monotone_maps() -> None:
    """Spearman handles ties and monotone maps."""
    assert spearman(np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0, 400.0])) == pytest.approx(1.0)
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)
    assert spearman(np.array([1.0, 1.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.894, abs=1e-3)
    assert np.isnan(spearman(np.ones(4), np.arange(4.0)))


def test_local_sd_follows_a_growing_spread_and_respects_its_floor() -> None:
    """Local sd follows a growing spread and respects its floor."""
    rng = np.random.default_rng(0)
    growing = rng.normal(size=(5, 100)) * np.linspace(0.0, 2.0, 100)
    sd = local_sd(growing, span=5, floor=0.25)
    assert sd[90] > 3 * sd[20]
    assert sd[0] >= 0.25 * np.sqrt(np.mean(growing**2)) - 1e-12


def test_a_growing_null_spread_does_not_false_alarm_late() -> None:
    """Drift-like nulls (spread grows from zero) stay quiet at the calibrated level."""

    def null(n: int, seed: int) -> np.ndarray:
        """Drift-like null trajectories: a random walk from zero."""
        rng = np.random.default_rng(seed)
        return np.cumsum(rng.normal(size=(n, 120)) * 0.05, axis=1)

    bands = calibrate(null(40, 13), "cka_drift", transform="none")
    assert false_alarm_rate(null(400, 14), bands) < 0.15


def test_leave_one_out_fired_refits_the_panel_and_flags_each_seed() -> None:
    """Leave-one-out refits bands and threshold without each seed, and flags a seed that is off the null."""
    null = _ar1(8, seed=15)
    null[3, 60:] -= 8.0  # one seed collapses: the others' bands and threshold must catch it
    fired = leave_one_out_fired({"rankme": null}, {}, alpha=0.05)
    assert len(fired) == 8 and all(isinstance(f, bool) for f in fired)
    assert fired[3] and sum(fired) <= 2


def test_short_batch_steps_flag_the_tiny_leg_flushes() -> None:
    """Short batch steps flag the tiny leg flushes."""
    legs = [
        {"start_step": 0, "end_step": 2, "n_images": 70},  # 32 + 32 + 6
        {"start_step": 3, "end_step": 3, "n_images": 20},
        {"start_step": 4, "end_step": 4, "n_images": 3},
    ]
    assert short_batch_steps(legs, 32, 8) == {2, 4}


def test_skipped_checkpoints_cannot_fire_and_do_not_break_calibration() -> None:
    """Skipped checkpoints cannot fire and do not break calibration."""
    null = _ar1(10, seed=17)
    null[:, 50] = np.nan
    bands = calibrate(null, "rankme")
    spike = _ar1(1, seed=18)[0]
    spike[50] = -100.0  # a wild reading at a skipped checkpoint
    spike[50] = np.nan
    assert np.isfinite(bands.sd).all() and not call(spike, bands)["fired"]
