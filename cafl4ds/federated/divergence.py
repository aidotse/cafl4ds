"""Per-client representation divergence — whether clients pull *apart* within a round.

The global readout measures the aggregate, which can hide the mechanism under study: two clients
drifting in opposite directions average back to near where they started (F2 hit exactly this).
These metrics read the divergence directly, on the backbone embedding of the **global** held-out
probe set, which is shared by all clients and disjoint from their training data:

* **anchor divergence** — each client's post-round representation against the broadcast weights.
* **cross-client divergence** — every client pair against each other.

Each is reported as a mean and a max; the max is usually more informative, since FedAvg is harmed
by its worst-diverging participant.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import torch

from cafl4ds import measurements
from cafl4ds.ssl.base import SSLMethod


def embed_probe(method: SSLMethod, images: torch.Tensor) -> torch.Tensor:
    """Embed a fixed probe set in eval mode, restoring the method's training flag afterwards.

    Mirrors :meth:`~cafl4ds.monitor.HealthMonitor.measure`.

    Args:
        method: The SSL method whose backbone embedding to read.
        images: A fixed probe batch ``[N, C, H, W]``.

    Returns:
        The pooled backbone embeddings ``[N, d]`` (no gradient).
    """
    was_training = method.training
    method.eval()
    try:
        return method.encode(images)
    finally:
        method.train(was_training)


def divergence_metrics(anchor_z: torch.Tensor, client_z: Sequence[torch.Tensor]) -> dict[str, float]:
    """Summarize how far a round's clients moved from the anchor, and from each other.

    Args:
        anchor_z: Probe embeddings ``[N, d]`` of the weights broadcast this round.
        client_z: Each participating client's probe embeddings ``[N, d]``, same probe set and
            same order as the round's participants.

    Returns:
        A flat ``metric -> value`` dict. Anchor divergence is always present; the
        ``cross_client_*`` keys need at least two participants and are **omitted** below that
        (a single client has no pair, and there is no honest value to report for one).

    Raises:
        ValueError: If ``client_z`` is empty.
    """
    if not client_z:
        raise ValueError("divergence_metrics needs at least one client embedding.")

    anchor_cka = [measurements.cka_drift(anchor_z, z) for z in client_z]
    anchor_cos = [measurements.cosine_drift(anchor_z, z) for z in client_z]
    metrics = {
        "client_anchor_cka_mean": _mean(anchor_cka),
        "client_anchor_cka_max": max(anchor_cka),
        "client_anchor_cosine_mean": _mean(anchor_cos),
        "client_anchor_cosine_max": max(anchor_cos),
    }
    if len(client_z) > 1:
        pairs = [measurements.cka_drift(a, b) for a, b in combinations(client_z, 2)]
        metrics["cross_client_cka_mean"] = _mean(pairs)
        metrics["cross_client_cka_max"] = max(pairs)
    return metrics


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean of a non-empty sequence."""
    return float(sum(values) / len(values))
