"""P1.1.4 — the referee-power core (Part 4) and the false-alarm count (Part 1)."""

import math

import numpy as np
import pytest

from cafl4ds import referee_power as rp
from scripts.analyze_p114 import wilson_interval


def test_change_noise_point_is_the_seed_mean_bacc_fall() -> None:
    """The point loss is each seed's balanced-accuracy fall from t0, averaged."""
    labels = np.array([0, 0, 1, 1])
    start = np.array([[1, 1, 1, 0], [1, 0, 1, 1]], dtype=float)  # bacc 0.75, 0.75
    later = [np.array([[1, 0, 0, 0], [0, 0, 1, 1]], dtype=float)]  # bacc 0.25, 0.5
    out = rp.change_noise(start, later, labels, n_boot=50)
    assert out["loss"] == pytest.approx([0.375])
    assert out["seed_noise"].shape == (50, 1)
    assert out["image_noise"].shape == (50, 1)


def test_change_noise_components_are_centred_and_sized() -> None:
    """No change gives no noise; a random 10% flip gives the binomial image SD."""
    rng = np.random.default_rng(0)
    n, s = 4000, 5
    labels = np.repeat([0, 1], n // 2)
    start = rng.integers(0, 2, size=(s, n)).astype(float)
    later = [start.copy(), np.where(rng.random((s, n)) < 0.1, 1 - start, start)]
    out = rp.change_noise(start, later, labels, n_boot=2000)
    # no change at the first later checkpoint: no noise at all
    assert np.allclose(out["image_noise"][:, 0], 0) and np.allclose(out["seed_noise"][:, 0], 0)
    # 10% of images flip per seed: the per-seed change has mean ~0 and image SD ~ sqrt(0.1 / n / s)
    sd = out["image_noise"][:, 1].std()
    assert sd == pytest.approx(math.sqrt(0.1 / n / s), rel=0.15)
    assert abs(out["image_noise"][:, 1].mean()) < sd / 5


def test_mdd_matches_the_gaussian_formula_at_one_checkpoint() -> None:
    """One checkpoint, Gaussian noise: MDD = (z_0.95 + z_0.80) sd."""
    noise = np.random.default_rng(1).normal(0, 0.01, size=(200_000, 1))
    out = rp.mdd(noise, np.ones(1), alpha=0.05, power=0.8)
    assert out["mdd"] == pytest.approx((1.6449 + 0.8416) * 0.01, rel=0.02)


def test_mdd_grows_with_the_max_over_checkpoints_and_shrinks_for_a_sustained_loss() -> None:
    """The max over checkpoints costs power; a loss held at every one buys it back."""
    noise = np.random.default_rng(2).normal(0, 0.01, size=(100_000, 4))
    one = rp.mdd(noise[:, :1], np.ones(1), alpha=0.05)["mdd"]
    single = rp.mdd(noise, np.array([0, 0, 0, 1.0]), alpha=0.05)["mdd"]
    sustained = rp.mdd(noise, np.ones(4), alpha=0.05)["mdd"]
    assert single > one  # the max over four independent readings needs a larger loss at one of them
    assert sustained < single


def test_combined_noise_scales_each_component() -> None:
    """The image part scales as 1/sqrt(pool), the seed part as 1/sqrt(seeds)."""
    rng = np.random.default_rng(3)
    seed_noise, image_noise = rng.normal(0, 1, (50_000, 1)), rng.normal(0, 1, (50_000, 1))
    assert rp.combined_noise(seed_noise, image_noise).std() == pytest.approx(math.sqrt(2), rel=0.02)
    assert rp.combined_noise(seed_noise, image_noise, pool=4, seeds=4).std() == pytest.approx(math.sqrt(0.5), rel=0.02)


def test_pool_for_is_infinite_when_the_seed_part_alone_misses_the_target() -> None:
    """A seed part above the target forbids any pool; without one, the pool follows 1/sqrt(n)."""
    rng = np.random.default_rng(4)
    seed_noise, image_noise = rng.normal(0, 0.02, (20_000, 1)), rng.normal(0, 0.02, (20_000, 1))
    assert rp.pool_for(seed_noise, image_noise, np.ones(1), 0.02, alpha=0.05) == math.inf
    # with a negligible seed part, the pool needed follows 1/sqrt(n): target half the image MDD -> ~4x
    pool = rp.pool_for(seed_noise * 1e-6, image_noise, np.ones(1), 0.5 * 2.4865 * 0.02, alpha=0.05)
    assert pool == pytest.approx(4.0, rel=0.1)


def test_dense_grid_mdd_is_bonferroni() -> None:
    """The dense-grid bound is Bonferroni over the checkpoints."""
    assert rp.dense_grid_mdd(0.01, 1, alpha=0.05) == pytest.approx((1.6449 + 0.8416) * 0.01, rel=1e-3)
    assert rp.dense_grid_mdd(0.01, 10, alpha=0.05) > rp.dense_grid_mdd(0.01, 1, alpha=0.05)


def test_wilson_interval() -> None:
    """The Wilson interval matches its tabled value, and rejects k > n."""
    lo, hi = wilson_interval(1, 15)
    assert lo == pytest.approx(0.0119, abs=1e-3) and hi == pytest.approx(0.2983, abs=1e-3)
    with pytest.raises(ValueError):
        wilson_interval(3, 2)
