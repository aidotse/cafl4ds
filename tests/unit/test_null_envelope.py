"""Unit tests for the P1.1.0 healthy-movement null (``cafl4ds.null_envelope``)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from cafl4ds.null_envelope import (
    canary_power,
    checkpoint_steps,
    effect_sizes,
    envelope,
    gap,
    movement,
    read_corpus,
    seed_matrix,
    signal_columns,
    t_multiplier,
    wilson_halfwidth,
)


def _rows(values: dict[int, list[float]], signal: str = "rankme", arm: str = "live") -> list[dict[str, object]]:
    """Corpus-like rows for one signal: ``{seed: per-checkpoint values}``."""
    return [
        {"backbone": "je", "seed": float(seed), "arm": arm, "checkpoint_index": float(i), signal: v}
        for seed, series in values.items()
        for i, v in enumerate(series)
    ]


def test_read_corpus_parses_numbers_and_blanks(tmp_path: Path) -> None:
    """Numbers become floats, blanks become ``None``, text stays text."""
    path = tmp_path / "health.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["backbone", "seed", "arm", "checkpoint_index", "era_name", "rankme", "loss"])
        writer.writerow(["je", "0", "live", "0", "daytime·clear", "12.5", ""])
    rows = read_corpus(path)
    assert rows == [
        {
            "backbone": "je",
            "seed": 0.0,
            "arm": "live",
            "checkpoint_index": 0.0,
            "era_name": "daytime·clear",
            "rankme": 12.5,
            "loss": None,
        }
    ]
    assert signal_columns(rows) == ["rankme"]


def test_seed_matrix_orders_seeds_and_checkpoints_and_filters_arm() -> None:
    """Rows are ordered by seed, columns by checkpoint; other arms are ignored."""
    rows = _rows({1: [3.0, 4.0], 0: [1.0, 2.0]}) + _rows({0: [9.0, 9.0]}, arm="b5")
    assert seed_matrix(rows, "rankme").tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_seed_matrix_rejects_misaligned_seeds() -> None:
    """Seeds with different checkpoint counts cannot be aligned."""
    with pytest.raises(ValueError, match="checkpoint counts"):
        seed_matrix(_rows({0: [1.0, 2.0], 1: [1.0]}), "rankme")


def test_envelope_is_mean_plus_minus_k_sample_sds() -> None:
    """The envelope is the per-checkpoint seed mean ± k sample SDs."""
    m = np.array([[1.0, 10.0], [3.0, 10.0]])
    env = envelope(m, k=2.0)
    assert env["mean"].tolist() == [2.0, 10.0]
    assert env["sd"][0] == pytest.approx(math.sqrt(2.0))
    assert env["lo"][0] == pytest.approx(2.0 - 2 * math.sqrt(2.0))
    assert env["hi"][1] == pytest.approx(10.0)


def test_movement_measures_net_change_and_largest_excursion() -> None:
    """Net change start→end, the largest excursion, and the range — each averaged over seeds."""
    m = np.array([[0.0, 3.0, 1.0], [2.0, 0.0, 2.0]])
    mv = movement(m)
    assert mv["final"] == pytest.approx((1.0 + 0.0) / 2)
    assert mv["max_abs"] == pytest.approx((3.0 + 2.0) / 2)
    assert mv["range"] == pytest.approx((3.0 + 2.0) / 2)


def test_effect_sizes_are_standardized_mean_differences() -> None:
    """D = (walk mean − IID mean) / pooled seed SD, per checkpoint."""
    walk = np.array([[2.0], [4.0]])
    iid = np.array([[0.0], [2.0]])
    assert effect_sizes(walk, iid)[0] == pytest.approx(2.0 / math.sqrt(2.0))


def test_effect_sizes_handle_zero_spread() -> None:
    """Zero spread in both arms reads 0 for equal means and nan (undefined) for different ones."""
    walk = np.array([[1.0, 2.0], [1.0, 2.0]])
    iid = np.array([[1.0, 1.0], [1.0, 1.0]])
    d = effect_sizes(walk, iid)
    assert d[0] == 0.0
    assert math.isnan(d[1])


def test_gap_reads_no_effect_for_identical_arms() -> None:
    """Identical arms: d≈0 everywhere, nothing significant, and p_max at its ceiling."""
    rng = np.random.default_rng(0)
    iid = rng.normal(0.0, 1.0, size=(5, 50))
    same = gap(iid.copy(), iid)
    assert same["max_abs_d"] == pytest.approx(0.0)
    assert same["frac_sig"] == 0.0
    assert same["p_max"] == 1.0
    assert same["n_perm"] == 252.0  # C(10, 5) relabellings


def test_gap_detects_a_large_shift_at_the_attainable_floor() -> None:
    """A walk shifted by many SDs sits at the smallest attainable p (2 / n_perm) and is significant everywhere."""
    rng = np.random.default_rng(1)
    iid = rng.normal(0.0, 1.0, size=(5, 50))
    walk = rng.normal(10.0, 1.0, size=(5, 50))
    shifted = gap(walk, iid)
    assert shifted["p_max"] == pytest.approx(2 / 252)
    assert shifted["frac_sig"] == 1.0
    assert shifted["final_d"] > 3.0
    assert shifted["final_gap"] == pytest.approx(float(walk[:, -1].mean() - iid[:, -1].mean()))


def test_gap_permutation_test_holds_its_level_under_the_null() -> None:
    """Two arms drawn from the same distribution reject at roughly the nominal family-wise rate."""
    rng = np.random.default_rng(2)
    rejections = 0
    trials = 200
    for _ in range(trials):
        base = rng.normal(size=(10, 30)).cumsum(axis=1)  # autocorrelated trajectories, like a drive
        rejections += gap(base[:5], base[5:])["p_max"] <= 0.05
    assert rejections / trials < 0.10


def test_gap_sign_follows_the_walk_direction() -> None:
    """A walk below the null reads a negative d."""
    iid = np.array([[1.0, 1.0], [1.2, 1.2], [0.8, 0.8]])
    assert gap(iid - 5.0, iid)["final_d"] < 0.0


def test_gap_rejects_mismatched_checkpoint_counts() -> None:
    """The matched control must share the walk's checkpoint grid."""
    with pytest.raises(ValueError, match="checkpoint counts"):
        gap(np.zeros((3, 4)), np.zeros((3, 5)))


def test_t_multiplier_uses_t_quantiles_and_falls_back_to_normal() -> None:
    """df=8 (5 vs 5 seeds) gives 2.306 + 0.889; large df gives the normal 2.80; untabulated df rounds down."""
    assert t_multiplier(8) == pytest.approx(3.195)
    assert t_multiplier(100) == pytest.approx(2.8016, abs=1e-3)
    assert t_multiplier(11) == t_multiplier(10)


def test_checkpoint_steps_reads_the_step_grid() -> None:
    """The step of each checkpoint, ordered by checkpoint index."""
    rows = [{"seed": 0.0, "arm": "live", "checkpoint_index": float(i), "step": float(2 * i)} for i in (2, 0, 1)]
    assert checkpoint_steps(rows) == [0, 2, 4]


def test_wilson_halfwidth_matches_the_known_value_and_shrinks_with_n() -> None:
    """The Wilson half-width at p=0.5, n=100 is ~0.096, and it shrinks as n grows."""
    assert wilson_halfwidth(0.5, 100) == pytest.approx(0.0962, abs=1e-3)
    assert wilson_halfwidth(0.5, 400) < wilson_halfwidth(0.5, 100)


def test_canary_power_reports_seed_spread_and_the_minimal_detectable_gap() -> None:
    """With a known across-seed SD, the MDD is t-multiplier(df=8) · sd · sqrt(2 / n_seeds)."""
    # five seeds offset by a fixed per-seed constant: across-seed SD is exact and the same at every step
    offsets = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    iid = 0.5 + offsets[:, None] + np.zeros((5, 20))
    power = canary_power(iid, n_query=135)
    sd = float(np.std(offsets, ddof=1))
    assert power["sd_seed"] == pytest.approx(sd)
    assert power["mdd"] == pytest.approx(t_multiplier(8) * sd * math.sqrt(2 / 5), rel=1e-6)
    assert power["sd_within"] == pytest.approx(0.0)  # flat trajectories: no checkpoint jitter
    assert power["ci95_halfwidth"] == pytest.approx(wilson_halfwidth(0.5, 135))
