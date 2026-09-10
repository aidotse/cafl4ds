"""Federated strategies — the single user-facing choice of *which FL algorithm* to run.

A federated algorithm is not one decision but two, taken at opposite ends of a round:

* **Server side** — how the aggregated client update becomes the next global model
  (:mod:`~cafl4ds.federated.server_optim`).
* **Client side** — what constraint, if any, local training is placed under
  (:mod:`~cafl4ds.federated.proximal`).

Published algorithms are *named points* in that two-dimensional space. FedAvg is "identity server
step, unconstrained client". FedAdam changes only the server half. FedProx changes only the client
half. That is why they compose: FedProx and FedAdam touch different halves of the round, so
running both is a coherent fourth option rather than a contradiction.

:class:`FederatedStrategy` bundles the two halves into one object, so a run selects an algorithm
by name (``strategy=fedadam``) instead of assembling it from parts. The bundling is a
convenience, not a restriction — a strategy is free to set both halves, and
``strategy/fedprox_fedadam.yaml`` is exactly that.

Adding a strategy is a config file, not code, whenever it is a new *combination* of existing
halves. Write a new class only when a genuinely new mechanism is needed — a new server rule goes
in :mod:`~cafl4ds.federated.server_optim`, a new client constraint alongside
:class:`~cafl4ds.federated.proximal.ProximalTerm`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cafl4ds.federated.proximal import ProximalTerm
from cafl4ds.federated.server_optim import FedAvgServer, ServerOptimizer


@dataclass(frozen=True)
class FederatedStrategy:
    """One named federated algorithm: a server-side rule plus a client-side constraint.

    Attributes:
        server_optimizer: How the server applies each round's aggregate. Defaults to
            :class:`~cafl4ds.federated.server_optim.FedAvgServer`, the identity step.
        proximal_mu: FedProx penalty strength for every client. ``0.0`` (the default) leaves
            local training unconstrained, which is FedAvg's client.
    """

    server_optimizer: ServerOptimizer = field(default_factory=FedAvgServer)
    proximal_mu: float = 0.0

    def __post_init__(self) -> None:
        """Reject a negative penalty at construction, so a bad config fails before any training.

        Raises:
            ValueError: If ``proximal_mu`` is negative.
        """
        if self.proximal_mu < 0.0:
            raise ValueError(f"proximal_mu must be >= 0; got {self.proximal_mu}.")

    @property
    def name(self) -> str:
        """Short algorithm name for run logs, e.g. ``fedavg`` or ``fedadam+fedprox(mu=0.1)``."""
        base = self.server_optimizer.name
        return base if self.proximal_mu <= 0.0 else f"{base}+fedprox(mu={self.proximal_mu})"

    def make_proximal(self) -> ProximalTerm:
        """Build a **fresh** proximal term for one client.

        Each client needs its own instance: the term holds that client's per-round anchor, so a
        shared one would have every client leashed to whichever client was broadcast to last.

        Returns:
            A new :class:`~cafl4ds.federated.proximal.ProximalTerm` at this strategy's ``mu``.
            At ``mu = 0`` it is an exact no-op rather than an approximation of one.
        """
        return ProximalTerm(mu=self.proximal_mu)
