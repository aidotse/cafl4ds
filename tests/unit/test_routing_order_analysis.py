"""Matched order effects must not confuse a filter effect with a general order penalty."""

import copy
from typing import Any

import pytest

from cafl4ds.routing_order_analysis import order_effects


def _results() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    iid: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for seed in (0, 1):
        for arm in ("random", "high", "low"):
            old_mse = {"random": 1.0, "high": 1.2, "low": 0.9}[arm] * (seed + 1)
            new_mse = {"random": 1.1, "high": 1.21, "low": 0.99}[arm] * (seed + 1)
            for rows, mse in ((iid, old_mse), (blocked, new_mse)):
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "initial_hash": f"same-{seed}",
                        "steps": 100,
                        "entropy_change": 0.1,
                        "final": {"heldout_mse": mse, "balanced_knn": 0.5, "balanced_linear": 0.5},
                    }
                )
    return blocked, iid


def test_order_effect_and_filter_interaction_use_paired_seeds() -> None:
    """All arms can worsen while a filter's relative penalty shrinks under the changed order."""
    blocked, iid = _results()
    result = order_effects(blocked, iid)
    assert result["blocked_minus_iid"]["random"]["relative_mse"]["mean"] == pytest.approx(0.1)
    assert result["blocked_minus_iid"]["high"]["relative_mse"]["mean"] == pytest.approx(1.21 / 1.2 - 1)
    high = result["filter_by_order_interaction"]["high_minus_random"]["relative_mse_penalty_change_pp"]
    assert high["mean"] == pytest.approx(-10)
    assert high["ci95"] == pytest.approx([-10, -10])
    assert result["filter_by_order_interaction"]["low_minus_random"]["relative_mse_penalty_change_pp"][
        "mean"
    ] == pytest.approx(0)


def test_unmatched_conditions_are_rejected() -> None:
    """Prevent attributing a changed seed, initialization or training budget to ordering."""
    blocked, iid = _results()
    with pytest.raises(ValueError):
        order_effects(blocked[:-1], iid)
    for key, value in (("initial_hash", "different"), ("steps", 101)):
        other = copy.deepcopy(blocked)
        other[0][key] = value
        with pytest.raises(ValueError):
            order_effects(other, iid)
