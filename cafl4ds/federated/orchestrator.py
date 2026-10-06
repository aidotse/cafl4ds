"""The orchestrator — the synchronous federated round loop over streaming clients.

Each round: broadcast the global weights, let every active client train one round on the next
slice of its stream, average the clients that trained
(:func:`~cafl4ds.federated.aggregate.federated_average`), and hand the average to the
:class:`~cafl4ds.federated.server_optim.ServerOptimizer` (FedAvg by default). This is a
single-process simulation: clients are objects iterated in turn, with no networking.

Clients run true single-pass streams, so under a skewed partition they exhaust at different
rounds and simply stop participating; the run ends when none remain (or at ``num_rounds``).

Per round, the run log records the participants' sample-weighted loss and, given a
``global_monitor``, the aggregated model's health on a global held-out set. ``track_divergence``
adds per-client divergence (:mod:`~cafl4ds.federated.divergence`), since an aggregate can hide
clients drifting in opposite directions. ``track_client_health`` also measures each client's
post-round model on that same global set, logged to the client's own run log.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from loguru import logger

from cafl4ds.federated.aggregate import StateDict, federated_average, weights_from_samples
from cafl4ds.federated.client import FederatedClient, RoundResult
from cafl4ds.federated.divergence import divergence_metrics, embed_probe
from cafl4ds.federated.server_optim import FedAvgServer, ServerOptimizer
from cafl4ds.monitor import HealthMonitor
from cafl4ds.run_log import RunLogger
from cafl4ds.ssl.base import SSLMethod


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
        track_divergence: bool = True,
        track_client_health: bool = False,
    ) -> None:
        """Configure the orchestrator.

        Args:
            clients: The federated clients (each with its own model, filter, and stream).
            steps_per_round: Stream steps each client advances per round.
            num_rounds: Optional hard cap on rounds; ``None`` runs until every client exhausts.
            global_monitor: Optional monitor on a *global* held-out set, used to log the
                aggregated model's health each round.
            run_logger: Optional run log for the per-round loss and health series.
            server_optimizer: How the aggregate becomes the next global model; defaults to
                :class:`~cafl4ds.federated.server_optim.FedAvgServer`. Must not be shared
                between runs.
            track_divergence: Also log per-client divergence each round. Needs
                ``global_monitor`` for its probe set and is inert without one.
            track_client_health: Also measure each participant's post-round model on the global
                held-out set, logged to that client's run log at ``step = round``. Needs
                ``global_monitor`` and is inert without one. Costs one health measurement per
                participant per round.

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
        # A shared probe set makes per-client numbers comparable; the global one is disjoint
        # from every client's training data.
        self.track_divergence = track_divergence and global_monitor is not None
        self.track_client_health = track_client_health and global_monitor is not None
        self.server_optimizer: ServerOptimizer = server_optimizer if server_optimizer is not None else FedAvgServer()
        # Only parameters are server-stepped; buffers stay at their FedAvg mean (see
        # `_apply_server_step`). The model is the one exact way to tell the two apart.
        self._param_keys = frozenset(dict(self.clients[0].method.named_parameters()))

    def run(self) -> tuple[StateDict, list[RoundSummary]]:
        """Run the federated training loop to completion.

        Returns:
            The final global ``state_dict`` (the shared part only, under ``keep_heads_local``)
            and the per-round summaries.
        """
        # A single starting point for all, private heads included, so local heads differ only by
        # what each client learns, not by their random init.
        initial = {k: v.detach().clone() for k, v in self.clients[0].method.state_dict().items()}
        for client in self.clients[1:]:
            client.load_weights(initial)
        global_state = self.clients[0].get_weights()
        history: list[RoundSummary] = []
        round_index = 0
        logger.info(f"server optimizer: {self.server_optimizer.name}")
        while self._should_continue(round_index):
            active = [c for c in self.clients if not c.exhausted]
            if not active:
                break
            for client in active:
                client.set_weights(global_state)
            # Embed the anchor before any local step, so the divergence baseline is exactly the
            # broadcast weights.
            anchor_z = self._embed_probe(active[0].method) if self.track_divergence else None
            results: list[tuple[FederatedClient, RoundResult]] = [
                (client, client.train_round(self.steps_per_round)) for client in active
            ]
            trained = [(c, r) for c, r in results if r.num_trained > 0]
            if not trained:
                break  # every active client exhausted with nothing admitted
            # Read divergence and client health before `_record_round` overwrites client 0's model.
            divergence = (
                divergence_metrics(anchor_z, [self._embed_probe(c.method) for c, _ in trained])
                if anchor_z is not None
                else {}
            )
            if self.track_client_health:
                self._record_client_health(round_index, [c for c, _ in trained])
            aggregated = federated_average(
                [c.get_weights() for c, _ in trained],
                weights_from_samples([r.num_trained for _, r in trained]),
            )
            global_state = self._apply_server_step(global_state, aggregated)
            history.append(self._record_round(round_index, [r for _, r in trained], global_state, divergence))
            round_index += 1
        logger.info(f"federated run complete: {len(history)} rounds")
        if self.run_logger is not None:
            self.run_logger.close()
        return global_state, history

    def _should_continue(self, round_index: int) -> bool:
        """Whether another round is allowed under the optional ``num_rounds`` cap."""
        return self.num_rounds is None or round_index < self.num_rounds

    def _embed_probe(self, method: SSLMethod) -> torch.Tensor:
        """Embed the global monitor's probe-query set with ``method``.

        Args:
            method: The model to read (a client after its round, or the broadcast anchor).

        Returns:
            Pooled backbone embeddings of the global probe-query set ``[N, d]``.

        Raises:
            RuntimeError: If no ``global_monitor`` is set.
        """
        monitor = self.global_monitor
        if monitor is None:
            raise RuntimeError("divergence tracking requires a global_monitor to supply the probe set.")
        return embed_probe(method, monitor.eval_sets.probe_query.images)

    def _record_client_health(self, round_index: int, clients: list[FederatedClient]) -> None:
        """Measure each client's post-round model on the global held-out set and log it.

        Args:
            round_index: The round just trained.
            clients: The clients that trained this round.
        """
        monitor = self.global_monitor
        if monitor is None:
            return
        for client in clients:
            client.loop.run_logger.log_health(round_index, era=-1, metrics=monitor.measure(client.method, round_index))

    def _apply_server_step(self, old_state: StateDict, aggregated: StateDict) -> StateDict:
        """Run the server optimizer over the parameters, keeping buffers at their FedAvg mean.

        Buffers such as BatchNorm ``running_mean`` / ``running_var`` are measurements, not
        variables to descend on; an adaptive step would move them by about ``lr`` regardless of
        the data and can push a variance below zero (``NaN`` outputs).

        Args:
            old_state: The global weights broadcast at the start of the round.
            aggregated: The round's aggregated client weights.

        Returns:
            The next global ``state_dict`` with every key present.
        """
        params = {key: value for key, value in aggregated.items() if key in self._param_keys}
        stepped = self.server_optimizer.step({key: old_state[key] for key in params}, params)
        return {**aggregated, **stepped}

    def _record_round(
        self,
        round_index: int,
        results: list[RoundResult],
        global_state: StateDict,
        divergence: dict[str, float],
    ) -> RoundSummary:
        """Measure/log the aggregated model and build the round summary.

        Args:
            round_index: The round just completed.
            results: The results of the clients that trained this round.
            global_state: The new global weights.
            divergence: This round's divergence metrics (empty when not tracked), merged into
                the logged health.

        Returns:
            The :class:`RoundSummary` for this round.
        """
        samples = sum(r.num_trained for r in results)
        participants = [r.client_id for r in results]
        if self.run_logger is not None:
            # Weighted like the FedAvg aggregate, so the loss matches what the global model was
            # trained towards.
            round_loss = sum(r.mean_loss * r.num_trained for r in results if r.mean_loss is not None) / samples
            self.run_logger.log_loss(round_index, era=-1, loss=round_loss)
        health: dict[str, float] | None = None
        if self.global_monitor is not None:
            # Client 0's model serves as the vessel; the next broadcast overwrites it anyway (its
            # private heads, if any, are kept).
            self.clients[0].load_weights(global_state)
            health = {**self.global_monitor.measure(self.clients[0].method, round_index), **divergence}
            if self.run_logger is not None:
                self.run_logger.log_health(round_index, era=-1, metrics=health)
        logger.info(f"round {round_index}: {len(participants)} clients, {samples} imgs trained")
        return RoundSummary(round_index=round_index, participants=participants, samples=samples, health=health)
