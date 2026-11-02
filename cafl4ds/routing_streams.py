"""Regime-blocked ordering with exactly matched IID candidate membership and batch budgets."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def regime_blocked_order(
    shuffled_order: torch.Tensor, regimes: torch.Tensor, regime_order: Sequence[int], batch_size: int
) -> torch.Tensor:
    """Group the retained IID prefix stably by regime; leave its discarded tail unchanged.

    Fixed-size batches are cut after concatenating blocks. Only boundary batches
    may mix regimes. Rare blocks are neither dropped nor padded to fill a batch.
    """
    if batch_size < 1 or shuffled_order.ndim != 1 or len(set(regime_order)) != len(regime_order):
        raise ValueError("Require a positive batch size, a flat order, and unique regime IDs.")
    retained = len(shuffled_order) // batch_size * batch_size
    if retained == 0:
        raise ValueError("The stream contains no complete candidate batch.")
    kept = shuffled_order[:retained]
    labels = regimes[kept]
    if not set(labels.tolist()).issubset(regime_order):
        raise ValueError("Every retained regime must appear in the requested walk.")
    grouped = torch.cat([kept[labels == regime] for regime in regime_order])
    return torch.cat([grouped, shuffled_order[retained:]])


def audit_regime_order(
    order: torch.Tensor, regimes: torch.Tensor, regime_names: Mapping[int, str], batch_size: int
) -> dict[str, Any]:
    """Record exact membership/order hashes, regime spans and boundary batches for one epoch."""
    retained = len(order) // batch_size * batch_size
    kept, dropped = order[:retained], order[retained:]
    labels = regimes[kept]
    starts = [0, *((labels[1:] != labels[:-1]).nonzero().flatten() + 1).tolist()]
    stops = [*starts[1:], retained]
    segments = []
    for start, stop in zip(starts, stops, strict=True):
        regime = int(labels[start])
        segments.append(
            {
                "regime": regime,
                "name": regime_names[regime],
                "start_candidate": start,
                "stop_candidate": stop,
                "count": stop - start,
                "start_update": start / batch_size,
                "stop_update": stop / batch_size,
            }
        )
    batches = labels.reshape(-1, batch_size)
    mixed = (batches != batches[:, :1]).any(1).nonzero().flatten().tolist()

    def digest(values: torch.Tensor) -> str:
        return hashlib.sha256(values.cpu().contiguous().numpy().tobytes()).hexdigest()

    return {
        "candidate_order_sha256": digest(kept),
        "candidate_membership_sha256": digest(kept.sort().values),
        "dropped_ids": dropped.tolist(),
        "candidate_count": retained,
        "updates": len(batches),
        "mixed_batch_indices": mixed,
        "mixed_batch_count": len(mixed),
        "segments": segments,
    }
