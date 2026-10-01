"""Aggregation — the weighted mean of the participating clients' ``state_dict``s.

The first of two server-side steps per round; :mod:`~cafl4ds.federated.server_optim` decides
what to do with the mean (under FedAvg it simply becomes the next global model). Each client is
weighted by the images it actually trained on this round, so a client whose filter admitted
little contributes little — the seam where selection-induced skew (**N-D**) enters. Only
floating-point tensors are averaged; integer buffers (e.g. ``num_batches_tracked``) are copied
from the first client.

A health-gated aggregator (**N-F**) would slot in here behind the same
``(states, weights) -> state`` shape.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

StateDict = dict[str, torch.Tensor]


def weights_from_samples(sample_counts: Sequence[int]) -> list[float]:
    """Normalize per-client training-sample counts into FedAvg mixing weights.

    Args:
        sample_counts: Images each client trained on this round (same order as the states).

    Returns:
        Weights summing to 1, or uniform weights if every count is zero.
    """
    total = float(sum(sample_counts))
    if total <= 0.0:
        n = len(sample_counts)
        return [1.0 / n] * n if n else []
    return [c / total for c in sample_counts]


def federated_average(states: Sequence[StateDict], weights: Sequence[float]) -> StateDict:
    """FedAvg: the weighted mean of client ``state_dict``s.

    Args:
        states: One ``state_dict`` per participating client (identical key sets).
        weights: Mixing weight per client (same order), typically from
            :func:`weights_from_samples`.

    Returns:
        The aggregated ``state_dict`` (fresh tensors; inputs are untouched).

    Raises:
        ValueError: If ``states`` is empty or its length differs from ``weights``.
    """
    if not states:
        raise ValueError("federated_average requires at least one client state.")
    if len(states) != len(weights):
        raise ValueError(f"states/weights length mismatch: {len(states)} vs {len(weights)}.")
    reference = states[0]
    aggregated: StateDict = {}
    for key, ref_tensor in reference.items():
        if ref_tensor.is_floating_point():
            acc = torch.zeros_like(ref_tensor)
            for state, weight in zip(states, weights, strict=True):
                acc += weight * state[key]
            aggregated[key] = acc
        else:
            aggregated[key] = ref_tensor.clone()
    return aggregated
