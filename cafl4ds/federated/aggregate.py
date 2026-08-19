"""Aggregation — combine client model updates into a new global model (FedAvg).

The server-side half of a federated round: given each participating client's updated weights
and how much local training backed them, produce the next global ``state_dict``. Phase-0 FL
ships **FedAvg** (McMahan et al. 2017) only — the sample-weighted mean of client weights.

Two deliberate details:

* **Sample weighting.** A client's contribution is weighted by the number of images it actually
  trained on this round (:attr:`~cafl4ds.loop.StepResult.num_trained` summed over the round), so
  a client whose filter admitted little contributes proportionally little. This is the seam
  where selection-induced skew (**N-D**) enters aggregation.
* **Non-float buffers.** ``state_dict`` mixes learnable float tensors with integer bookkeeping
  buffers (e.g. BatchNorm ``num_batches_tracked``). Averaging is applied to floating-point
  tensors only; non-float entries are carried over unchanged from the first client (they are
  step counters, not parameters).

Extension points (later phases): a health-gated aggregator that down-weights or drops a client
whose representation health has degraded (**N-F**), and FedProx's proximal term (a *client*-side
change, not here). Keep those behind the same ``(states, weights) -> state`` shape.
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
        Weights summing to 1. Falls back to a uniform average if every count is zero.
    """
    total = float(sum(sample_counts))
    if total <= 0.0:
        n = len(sample_counts)
        return [1.0 / n] * n if n else []
    return [c / total for c in sample_counts]


def federated_average(states: Sequence[StateDict], weights: Sequence[float]) -> StateDict:
    """FedAvg: the weighted mean of client ``state_dict``s.

    Floating-point tensors are averaged with ``weights``; non-float buffers (e.g.
    ``num_batches_tracked``) are copied from the first client unchanged.

    Args:
        states: One ``state_dict`` per participating client (identical key sets).
        weights: Mixing weight per client (same order); typically from
            :func:`weights_from_samples`. Need not be normalized, but usually is.

    Returns:
        The aggregated global ``state_dict`` (fresh tensors; inputs are untouched).

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
