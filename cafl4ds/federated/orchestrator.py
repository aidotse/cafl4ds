"""The orchestrator — the synchronous federated round loop over streaming clients.

The server side. Each round: broadcast the current global weights, let every still-active
client train one local round on the next slice of its stream, aggregate their updates, then hand
the aggregate to the server optimizer to form the next global model. This is a single-process
*simulation* — clients are objects iterated in turn, no networking — which is the standard,
reproducible setup for FL research at this scale.

The two server-side decisions are separate objects, so the loop itself is algorithm-agnostic:
:func:`~cafl4ds.federated.aggregate.federated_average` decides *how the clients are mixed*, and
the injected :class:`~cafl4ds.federated.server_optim.ServerOptimizer` decides *what is done with
the mixture*. The default is :class:`~cafl4ds.federated.server_optim.FedAvgServer`, whose step is
the identity — so the default behaviour is plain FedAvg, unchanged.

Round cadence and stopping:

* A round is ``steps_per_round`` **stream steps** per client (see
  :class:`~cafl4ds.federated.client.FederatedClient`).
* Clients run **true single-pass** streams that continue across rounds, so they exhaust at
  different rounds under a skewed partition. An exhausted client simply stops participating; the
  run ends when no active clients remain (or an optional ``num_rounds`` cap is hit).
* Only clients that actually trained this round enter the average (a client that exhausted with
  nothing admitted contributes nothing).

The round's loss (participants' ``mean_loss``, weighted by ``num_trained`` — the same weighting FedAvg
uses to aggregate weights) is logged every round via ``run_logger``. Global health (the dependent
variable) is logged once per round when a ``global_monitor`` is supplied — the aggregated model is
measured on a *global* held-out set, distinct from any client's skewed local monitor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from loguru import logger

from cafl4ds.federated.aggregate import StateDict, federated_average, weights_from_samples
from cafl4ds.federated.client import FederatedClient, RoundResult
from cafl4ds.federated.server_optim import FedAvgServer, ServerOptimizer
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
    """Runs a synchronous federated round loop over streaming clients until they exhaust."""

    def __init__(
        self,
        clients: Sequence[FederatedClient],
        steps_per_round: int,
        num_rounds: int | None = None,
        global_monitor: HealthMonitor | None = None,
        run_logger: RunLogger | None = None,
        server_optimizer: ServerOptimizer | None = None,
    ) -> None:
        """Configure the orchestrator.

        Args:
            clients: The federated clients (each with its own model, filter, and stream).
            steps_per_round: Stream steps each client advances per round.
            num_rounds: Optional hard cap on rounds; ``None`` runs until every client exhausts.
            global_monitor: Optional monitor (on a *global* held-out set) used to log the
                aggregated model's health each round.
            run_logger: Optional run log for the per-round global health series.
            server_optimizer: How the aggregated client update becomes the next global model
                (see :mod:`~cafl4ds.federated.server_optim`). Defaults to
                :class:`~cafl4ds.federated.server_optim.FedAvgServer` — the identity step, i.e.
                plain FedAvg. Stateful implementations must not be shared between runs.

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
        self.server_optimizer: ServerOptimizer = server_optimizer if server_optimizer is not None else FedAvgServer()
        # Which state_dict entries the server optimizer may touch. Only *parameters* are
        # optimized: a state_dict also holds non-learned buffers (BatchNorm running statistics,
        # step counters), and those are estimates rather than things to descend on. This
        # orchestrator is the only component holding a model, so it is the only one that can
        # tell the two apart exactly — see `_apply_server_step`.
        self._param_keys = frozenset(dict(self.clients[0].method.named_parameters()))

    def run(self) -> tuple[StateDict, list[RoundSummary]]:
        """Run the federated training loop to completion.

        Returns:
            The final global ``state_dict`` and the per-round summaries.
        """
        global_state = self.clients[0].get_weights()  # a single, shared starting point for all
        history: list[RoundSummary] = []
        round_index = 0
        logger.info(f"server optimizer: {self.server_optimizer.name}")
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
            # Mix the clients, then apply the mixture. Every participant trained from
            # `global_state`, so the aggregate minus it is the round's pseudo-gradient — which is
            # what the server optimizer consumes (a no-op under FedAvg).
            aggregated = federated_average(
                [c.get_weights() for c, _ in trained],
                weights_from_samples([r.num_trained for _, r in trained]),
            )
            global_state = self._apply_server_step(global_state, aggregated)
            history.append(self._record_round(round_index, [r for _, r in trained], global_state))
            round_index += 1
        logger.info(f"federated run complete: {len(history)} rounds")
        if self.run_logger is not None:
            self.run_logger.close()
        return global_state, history

    def _should_continue(self, round_index: int) -> bool:
        """Whether another round is allowed under the optional ``num_rounds`` cap."""
        return self.num_rounds is None or round_index < self.num_rounds

    def _apply_server_step(self, old_state: StateDict, aggregated: StateDict) -> StateDict:
        """Run the server optimizer over the *parameters*, keeping buffers at their FedAvg mean.

        A ``state_dict`` is not uniformly optimizable. Alongside learnable parameters it carries
        buffers that are *measurements* of the data rather than variables to descend on — for
        SimSiam's BatchNorm heads, ``running_mean`` and ``running_var``. An adaptive step is
        actively wrong for those on two counts: momentum makes a statistic lag its own data, and
        the normalized update moves each coordinate by about ``lr`` no matter how little the
        clients changed it. A running *variance* also has to stay non-negative, and a step
        decoupled from the true change can walk it past zero, which turns the model's output into
        ``NaN``. Their plain weighted mean — what ``federated_average`` already produced — is the
        right answer, and is what FedAvg has always used.

        Args:
            old_state: The global weights broadcast at the start of the round.
            aggregated: The round's aggregated client weights.

        Returns:
            The next global ``state_dict``: server-stepped parameters over an aggregate-mean base,
            so every key is present and buffers pass through untouched.
        """
        params = {key: value for key, value in aggregated.items() if key in self._param_keys}
        stepped = self.server_optimizer.step({key: old_state[key] for key in params}, params)
        return {**aggregated, **stepped}

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
        if self.run_logger is not None:
            # Sample-weighted mean of the round's participants — the same weighting FedAvg uses
            # to aggregate their weights, so the logged "loss" matches what the average model was
            # actually trained towards this round.
            round_loss = sum(r.mean_loss * r.num_trained for r in results if r.mean_loss is not None) / samples
            self.run_logger.log_loss(round_index, era=-1, loss=round_loss)
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
