"""Federated strategies — the single ``strategy=`` choice of which FL algorithm to run.

An FL algorithm is two decisions at opposite ends of a round: how the server applies the
aggregate (:mod:`~cafl4ds.federated.server_optim`), and what constraint local training is under
(:mod:`~cafl4ds.federated.proximal`). FedAvg is "identity server, unconstrained client"; FedAdam
changes only the server half, FedProx only the client half, so the two compose
(``strategy/fedprox_fedadam.yaml``). A new combination is a config file; only a new mechanism
needs code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cafl4ds.federated.proximal import ProximalTerm
from cafl4ds.federated.server_optim import FedAvgServer, ServerOptimizer


@dataclass(frozen=True)
class FederatedStrategy:
    """One named federated algorithm: a server-side rule plus a client-side constraint.

    Attributes:
        server_optimizer: How the server applies each round's aggregate (default: FedAvg's
            identity step).
        proximal_mu: FedProx penalty strength for every client; ``0.0`` leaves local training
            unconstrained.
    """

    server_optimizer: ServerOptimizer = field(default_factory=FedAvgServer)
    proximal_mu: float = 0.0

    def __post_init__(self) -> None:
        """Reject a negative penalty before any training.

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
        """Build a fresh proximal term for one client.

        Each client needs its own instance, since the term holds that client's per-round anchor.

        Returns:
            A new :class:`~cafl4ds.federated.proximal.ProximalTerm` at this strategy's ``mu``
            (an exact no-op at ``mu = 0``).
        """
        return ProximalTerm(mu=self.proximal_mu)
