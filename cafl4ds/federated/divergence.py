"""Per-client representation divergence — whether clients pull *apart* within a round.

The global readout (the orchestrator's ``global_monitor``) measures the **aggregate**, which can
hide the very mechanism it is meant to detect: two clients drifting hard in opposite directions
average back to near where they started, so a flat global probe is perfectly consistent with
large underlying divergence. F2 hit exactly this — the global probe moved less than its own noise
across sharply skewed partitions, leaving "no heterogeneity effect" and "the readout cannot see
one" indistinguishable. These measurements read the divergence directly instead.

Both are computed on the **global** held-out probe set rather than per-client eval sets. A common
probe is what makes client numbers comparable, and the global pool is disjoint from every client's
training data by construction (:func:`~cafl4ds.federated.partition.holdout_split`) — whereas a
client's own eval set is skewed by the partition under study and is sized 0 by default.

Two quantities, each on the backbone embedding:

* **anchor divergence** — a client's post-round representation against the weights broadcast at
  the start of that round. How far local training pulled *this* client.
* **cross-client divergence** — every client pair against each other. Whether they pulled in
  *different* directions, which is what averaging then has to reconcile.

Each is reported as a mean and a max. The max is usually the more informative: FedAvg is harmed
by its worst-diverging participant, not by the typical one.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import torch

from cafl4ds import measurements
from cafl4ds.ssl.base import SSLMethod


def embed_probe(method: SSLMethod, images: torch.Tensor) -> torch.Tensor:
    """Embed a fixed probe set in eval mode, leaving the method's training flag as found.

    Mirrors :meth:`~cafl4ds.monitor.HealthMonitor.measure`: probes read the representation with
    BatchNorm/dropout in eval mode, and the caller's mode is restored so the training loop is
    unperturbed. ``encode`` moves the images to the backbone's device itself.

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
