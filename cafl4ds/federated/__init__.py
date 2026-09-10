"""Federated learning (the ``D`` factor): partition, per-client streaming, FedAvg aggregation.

Wraps the centralized streaming loop in a synchronous FedAvg simulation. The pieces:

* :mod:`~cafl4ds.federated.partition` — split one dataset into per-client (non-IID) shards.
* :mod:`~cafl4ds.federated.client` — a resumable per-client :class:`~cafl4ds.loop.StreamingLoop`.
* :mod:`~cafl4ds.federated.aggregate` — mix client ``state_dict``s into a weighted mean.
* :mod:`~cafl4ds.federated.server_optim` — apply that mean (FedAvg, or the adaptive FedOpt family).
* :mod:`~cafl4ds.federated.orchestrator` — the round loop tying them together.
"""

from cafl4ds.federated.aggregate import federated_average, weights_from_samples
from cafl4ds.federated.client import FederatedClient, RoundResult
from cafl4ds.federated.orchestrator import FederatedOrchestrator, RoundSummary
from cafl4ds.federated.partition import partition_source
from cafl4ds.federated.server_optim import (
    AdaptiveServerOptimizer,
    FedAdagradServer,
    FedAdamServer,
    FedAvgServer,
    FedYogiServer,
    ServerOptimizer,
)

__all__ = [
    "AdaptiveServerOptimizer",
    "FedAdagradServer",
    "FedAdamServer",
    "FedAvgServer",
    "FedYogiServer",
    "FederatedClient",
    "FederatedOrchestrator",
    "RoundResult",
    "RoundSummary",
    "ServerOptimizer",
    "federated_average",
    "partition_source",
    "weights_from_samples",
]
