"""Federated learning (the ``D`` factor): a synchronous FL simulation over per-client streams.

* :mod:`~cafl4ds.federated.partition` — split one dataset into per-client (non-IID) shards.
* :mod:`~cafl4ds.federated.client` — a resumable per-client :class:`~cafl4ds.loop.StreamingLoop`.
* :mod:`~cafl4ds.federated.aggregate` — the weighted mean of client ``state_dict``s.
* :mod:`~cafl4ds.federated.server_optim` — apply that mean (FedAvg, or the adaptive FedOpt family).
* :mod:`~cafl4ds.federated.proximal` — the client-side FedProx constraint.
* :mod:`~cafl4ds.federated.strategy` — one named algorithm: a server rule plus a client constraint.
* :mod:`~cafl4ds.federated.divergence` — per-client representation divergence.
* :mod:`~cafl4ds.federated.orchestrator` — the round loop tying them together.
"""

from cafl4ds.federated.aggregate import federated_average, weights_from_samples
from cafl4ds.federated.client import FederatedClient, RoundResult
from cafl4ds.federated.orchestrator import FederatedOrchestrator, RoundSummary
from cafl4ds.federated.partition import partition_source
from cafl4ds.federated.proximal import ProximalTerm
from cafl4ds.federated.server_optim import (
    AdaptiveServerOptimizer,
    FedAdagradServer,
    FedAdamServer,
    FedAvgServer,
    FedYogiServer,
    ServerOptimizer,
)
from cafl4ds.federated.strategy import FederatedStrategy

__all__ = [
    "AdaptiveServerOptimizer",
    "FedAdagradServer",
    "FedAdamServer",
    "FedAvgServer",
    "FedYogiServer",
    "FederatedClient",
    "FederatedOrchestrator",
    "FederatedStrategy",
    "ProximalTerm",
    "RoundResult",
    "RoundSummary",
    "ServerOptimizer",
    "federated_average",
    "partition_source",
    "weights_from_samples",
]
