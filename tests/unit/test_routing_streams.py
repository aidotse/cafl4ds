"""Regime blocking changes correlation without changing candidate budgets or leaking labels."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from cafl4ds.routing_streams import audit_regime_order, regime_blocked_order


def test_regime_order_preserves_iid_membership_tail_and_within_regime_shuffle() -> None:
    """Even a one-image regime survives; only cross-boundary batches are mixed."""
    shuffled = torch.tensor([6, 2, 7, 1, 0, 3, 4, 5, 9, 8])
    regimes = torch.tensor([0, 1, 0, 2, 1, 1, 2, 1, 2, 0])
    result = regime_blocked_order(shuffled, regimes, [0, 1, 2], 4)
    assert result.tolist() == [2, 0, 7, 1, 4, 5, 6, 3, 9, 8]
    assert torch.equal(result[:8].sort().values, shuffled[:8].sort().values)
    assert torch.equal(result[8:], shuffled[8:])
    audit = audit_regime_order(result, regimes, {0: "day·clear", 1: "dusk·rain", 2: "night·clear"}, 4)
    assert [r["regime"] for r in audit["segments"]] == [0, 1, 2]
    assert audit["candidate_count"] == 8 and audit["updates"] == 2
    assert audit["mixed_batch_count"] == 2
    assert [r["count"] for r in audit["segments"]] == [2, 4, 2]


def test_rare_regime_is_not_dropped_or_padded() -> None:
    """A regime smaller than a candidate batch remains in the stream with no repeated IDs."""
    regimes = torch.tensor([0] * 5 + [1] + [2] * 6)
    result = regime_blocked_order(torch.arange(12), regimes, [0, 1, 2], 4)
    assert len(result.unique()) == 12
    assert (regimes[result] == 1).sum() == 1
    audit = audit_regime_order(result, regimes, {0: "a", 1: "b", 2: "c"}, 4)
    assert audit["segments"][1]["count"] == 1
    assert audit["mixed_batch_indices"] == [1]


def test_invalid_regime_walk_is_rejected() -> None:
    """An omitted regime must not silently remove its examples."""
    with pytest.raises(ValueError):
        regime_blocked_order(torch.arange(8), torch.tensor([0, 1] * 4), [0], 4)
    with pytest.raises(ValueError):
        regime_blocked_order(torch.arange(8), torch.tensor([0, 1] * 4), [0, 1, 1], 4)
    with pytest.raises(ValueError):
        regime_blocked_order(torch.arange(3), torch.zeros(3), [0], 4)


def test_regime_config_changes_only_order_and_output() -> None:
    """Model, optimizer, batch sizes, seeds, panels, and horizon inherit the IID defaults exactly."""
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "cafl4ds/configs"), version_base=None
    ):
        iid = compose(config_name="routing_experiment")
        blocked = compose(config_name="routing_regime_experiment")
    assert iid.stream_order == "iid" and blocked.stream_order == "regime_blocked"
    for key in iid:
        if key not in {"stream_order", "output"}:
            assert iid[key] == blocked[key], key
