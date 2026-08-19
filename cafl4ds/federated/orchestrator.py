"""The orchestrator — the synchronous FedAvg round loop over streaming clients.

The server side. Each round: broadcast the current global weights, let every still-active
client train one local round on the next slice of its stream, then FedAvg their updates into the
next global model. This is a single-process *simulation* — clients are objects iterated in turn,
no networking — which is the standard, reproducible setup for FL research at this scale.

Round cadence and stopping:

* A round is ``steps_per_round`` **stream steps** per client (see
  :class:`~cafl4ds.federated.client.FederatedClient`).
* Clients run **true single-pass** streams that continue across rounds, so they exhaust at
  different rounds under a skewed partition. An exhausted client simply stops participating; the
  run ends when no active clients remain (or an optional ``num_rounds`` cap is hit).
* Only clients that actually trained this round enter the average (a client that exhausted with
  nothing admitted contributes nothing).

Global health (the dependent variable) is logged once per round when a ``global_monitor`` is
supplied — the aggregated model is measured on a *global* held-out set, distinct from any
client's skewed local monitor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from loguru import logger

from cafl4ds.federated.aggregate import StateDict, federated_average, weights_from_samples
from cafl4ds.federated.client import FederatedClient, RoundResult
from cafl4ds.monitor import HealthMonitor
from cafl4ds.run_log import RunLogger


@dataclass(frozen=True)
class RoundSummary:
    """Per-round record: who participated and the resulting global health."""

    round_index: int
    """Zero-based round number."""
    participants: list[int]
    """Client ids that trained (and were aggregated) this round."""
    samples: int
    """Total images trained on across participants this round."""
    health: dict[str, float] | None
    """Global-model health after aggregation, or ``None`` if no ``global_monitor`` was set."""


class FederatedOrchestrator:
    """Runs synchronous FedAvg over a set of streaming clients until they exhaust."""

    def __init__(
        self,
        clients: Sequence[FederatedClient],
        steps_per_round: int,
        num_rounds: int | None = None,
        global_monitor: HealthMonitor | None = None,
        run_logger: RunLogger | None = None,
    ) -> None:
        """Configure the orchestrator.

        Args:
            clients: The federated clients (each with its own model, filter, and stream).
            steps_per_round: Stream steps each client advances per round.
            num_rounds: Optional hard cap on rounds; ``None`` runs until every client exhausts.
            global_monitor: Optional monitor (on a *global* held-out set) used to log the
                aggregated model's health each round.
            run_logger: Optional run log for the per-round global health series.

        Raises:
            ValueError: If ``clients`` is empty or ``steps_per_round < 1``.
        """
        if not clients:
            raise ValueError("FederatedOrchestrator requires at least one client.")
        if steps_per_round < 1:
            raise ValueError(f"steps_per_round must be >= 1; got {steps_per_round}.")
        self.clients = list(clients)
        self.steps_per_round = steps_per_round
        self.num_rounds = num_rounds
        self.global_monitor = global_monitor
        self.run_logger = run_logger

    def run(self) -> tuple[StateDict, list[RoundSummary]]:
        """Run the federated training loop to completion.

        Returns:
            The final global ``state_dict`` and the per-round summaries.
        """
        global_state = self.clients[0].get_weights()  # a single, shared starting point for all
        history: list[RoundSummary] = []
        round_index = 0
        while self._should_continue(round_index):
            active = [c for c in self.clients if not c.exhausted]
            if not active:
                break
            results: list[tuple[FederatedClient, RoundResult]] = []
            for client in active:
                client.set_weights(global_state)
                results.append((client, client.train_round(self.steps_per_round)))
            trained = [(c, r) for c, r in results if r.num_trained > 0]
            if not trained:
                break  # every active client exhausted with nothing admitted
            global_state = federated_average(
                [c.get_weights() for c, _ in trained],
                weights_from_samples([r.num_trained for _, r in trained]),
            )
            history.append(self._record_round(round_index, [r for _, r in trained], global_state))
            round_index += 1
        logger.info(f"federated run complete: {len(history)} rounds")
        if self.run_logger is not None:
            self.run_logger.close()
        return global_state, history

    def _should_continue(self, round_index: int) -> bool:
        """Whether another round is allowed under the optional ``num_rounds`` cap."""
        return self.num_rounds is None or round_index < self.num_rounds

    def _record_round(self, round_index: int, results: list[RoundResult], global_state: StateDict) -> RoundSummary:
        """Measure/log the aggregated model and build the round summary.

        Args:
            round_index: The round just completed.
            results: The results of the clients that trained this round.
            global_state: The freshly aggregated global weights.

        Returns:
            The :class:`RoundSummary` for this round.
        """
        samples = sum(r.num_trained for r in results)
        participants = [r.client_id for r in results]
        health: dict[str, float] | None = None
        if self.global_monitor is not None:
            # Measure the aggregated model on the global held-out set. The first client's model
            # is the load vessel — it is overwritten by set_weights at the next round anyway.
            vessel = self.clients[0].method
            vessel.load_state_dict(global_state)
            health = self.global_monitor.measure(vessel, round_index)
            if self.run_logger is not None:
                self.run_logger.log_health(round_index, era=-1, metrics=health)
        logger.info(f"round {round_index}: {len(participants)} clients, {samples} imgs trained")
        return RoundSummary(round_index=round_index, participants=participants, samples=samples, health=health)
