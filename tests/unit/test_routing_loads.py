"""Load analysis must distinguish probability ambiguity, hard concentration and training exposure."""

from __future__ import annotations

import numpy as np
import pytest

from cafl4ds.routing_loads import distribution_statistics, load_statistics, time_average


def test_load_statistics_have_meaningful_uniform_and_single_expert_limits() -> None:
    """Normalize counts, use population CV, and avoid treating effective count as active count."""
    uniform = distribution_statistics([10, 10, 10, 10])
    assert uniform["cv"] == 0 and uniform["entropy"] == pytest.approx(1)
    assert uniform["effective_experts"] == pytest.approx(4)
    concentrated = distribution_statistics([0, 0, 100, 0])
    assert concentrated["cv"] == pytest.approx(np.sqrt(3))
    assert concentrated["entropy"] == 0 and concentrated["effective_experts"] == 1
    assert concentrated["inactive_fraction"] == 0.75
    assert distribution_statistics([100, 0, 0, 0]) == concentrated
    for bad in ([0, 0], [float("nan"), 1], [-1, 2], [1]):
        with pytest.raises(ValueError):
            distribution_statistics(bad)


def test_time_average_accounts_for_irregular_checkpoints() -> None:
    """A cluster of closely spaced reads must not receive equal-time weight."""
    assert time_average([1, 2, 11], [0, 0, 1]) == pytest.approx(0.45)
    assert time_average([5], [0.7]) == 0.7
    with pytest.raises(ValueError):
        time_average([2, 1], [0, 1])


def test_post_init_load_contrasts_keep_seeds_and_layers_separate() -> None:
    """Reject initialization as learned concentration and separate hard loads from soft/training balance."""
    evaluations, training = {}, {}
    for seed in (0, 1):
        for arm in ("random", "high", "low"):
            hard = [0.7, 0.1, 0.1, 0.1] if arm == "high" else [0.25] * 4
            evaluations[(seed, arm)] = [
                {"step": 0, "l2_load": [1, 0, 0, 0], "l3_load": [1, 0, 0, 0]},
                *[
                    {
                        "step": step,
                        "l2_load": hard,
                        "l3_load": [0.25] * 4,
                        "l2_soft_mass": [0.25] * 4,
                        "l3_soft_mass": [0.25] * 4,
                    }
                    for step in (50, 100, 104)
                ],
            ]
            training[(seed, arm)] = [{"train_counts": {"l2": [128] * 4, "l3": [128] * 4}}]
    result = load_statistics(evaluations, training)
    high = result["summary"]["l2"]["high"]
    assert high["runs_with_majority"] == 2 and high["runs_with_inactive_read"] == 0
    assert high["training_pooled_cv"]["mean"] == 0
    assert high["soft_endpoint_entropy"]["mean"] == pytest.approx(1)
    assert high["endpoint_max_share"]["mean"] == pytest.approx(0.7)
    difference = result["contrasts"]["l2"]["high_minus_random"]["effects"]["time_mean_cv"]
    assert difference["mean"] > 0 and difference["ci95"][0] > 0
    assert result["contrasts"]["l3"]["high_minus_random"]["effects"]["time_mean_cv"]["mean"] == 0
    assert result["summary"]["l2"]["low"]["endpoint_entropy"]["sd"] == 0
