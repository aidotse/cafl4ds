"""The federated client — one local streaming learner, resumable across rounds.

A client owns exactly what a centralized run owns — a model, optimizer, selection filter,
monitor, and its *own* :class:`~cafl4ds.data.streams.EraStream` — bundled as a
:class:`~cafl4ds.loop.StreamingLoop`. Federation adds three things on top:

* **A persistent stream position.** The client holds one iterator over its stream and advances
  it a fixed number of steps per round, *continuing where it left off* — a single true
  single-pass stream sliced across rounds, never restarted. When the iterator is exhausted the
  client stops participating.
* **Weight exchange.** :meth:`set_weights` overwrites the *model only* with the broadcast global
  weights; the optimizer state and the filter's local state (e.g. a reservoir buffer) persist
  across rounds — they are the client's private streaming memory and are never synced.
* **A round of local work.** :meth:`train_round` pulls ``steps_per_round`` batches and runs each
  through :meth:`StreamingLoop.train_step` — the *same* selection + update path as centralized —
  reporting how much it trained on (for FedAvg weighting).

A "round" is measured in **stream steps** (batches pulled), not optimizer updates: every client
advances the same distance through its stream each round, so selection-induced differences in
*how much* each client trains surface as differences in the reported sample count rather than
being normalized away.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from loguru import logger

from cafl4ds.data.streams import StreamBatch
from cafl4ds.loop import StreamingLoop
from cafl4ds.ssl.base import SSLMethod

StateDict = dict[str, torch.Tensor]


@dataclass(frozen=True)
class RoundResult:
    """The outcome of one client round."""

    client_id: int
    """Which client produced this result."""
    steps_pulled: int
    """Batches pulled from the stream this round (<= ``steps_per_round``; short if exhausted)."""
    num_trained: int
    """Total images the client took an SSL update on this round (the FedAvg weight)."""
    mean_loss: float | None
    """Mean SSL loss over the round's updates, or ``None`` if no step trained."""
    exhausted: bool
    """Whether the client's stream ran out during this round."""


class FederatedClient:
    """One client: a resumable :class:`StreamingLoop` plus weight exchange."""

    def __init__(self, client_id: int, loop: StreamingLoop) -> None:
        """Wrap a per-client streaming loop as a federated client.

        Args:
            client_id: Stable identifier for logging/aggregation bookkeeping.
            loop: The client's local loop (its own model, optimizer, filter, monitor, stream).
        """
        self.client_id = client_id
        self.loop = loop
        self.loop.method.to(self.loop.device)
        self._iterator: Iterator[StreamBatch] = iter(loop.stream)
        self._step = 0
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        """Whether this client's stream has been fully consumed."""
        return self._exhausted

    @property
    def method(self) -> SSLMethod:
        """The client's local SSL model."""
        return self.loop.method

    def get_weights(self) -> StateDict:
        """Return a detached clone of the local model's ``state_dict`` (for aggregation)."""
        return {k: v.detach().clone() for k, v in self.loop.method.state_dict().items()}

    def set_weights(self, state: StateDict) -> None:
        """Load broadcast global weights into the local model (model only).

        The optimizer and the filter's local state are deliberately left untouched — under true
        streaming the client resumes its own optimization/replay memory against the new weights.

        Args:
            state: The global ``state_dict`` to load.
        """
        self.loop.method.load_state_dict(state)

    def train_round(self, steps_per_round: int) -> RoundResult:
        """Advance the local stream by ``steps_per_round`` and train on what is admitted.

        Args:
            steps_per_round: Number of batches to pull from the (persistent) stream this round.

        Returns:
            A :class:`RoundResult` summarizing the round (sample count drives FedAvg weighting).
        """
        self.loop.method.to(self.loop.device)
        steps_pulled, num_trained = 0, 0
        loss_sum, loss_count = 0.0, 0
        for _ in range(steps_per_round):
            batch = next(self._iterator, None)
            if batch is None:
                self._exhausted = True
                break
            steps_pulled += 1
            result = self.loop.train_step(batch, self._step)
            self._step += 1
            if result is None:
                continue
            num_trained += result.num_trained
            loss_sum += result.loss
            loss_count += 1
        mean_loss = loss_sum / loss_count if loss_count else None
        logger.debug(
            f"client {self.client_id}: pulled {steps_pulled}, trained on {num_trained} imgs, "
            f"mean_loss={mean_loss}, exhausted={self._exhausted}"
        )
        return RoundResult(
            client_id=self.client_id,
            steps_pulled=steps_pulled,
            num_trained=num_trained,
            mean_loss=mean_loss,
            exhausted=self._exhausted,
        )

    def measure_health(self, step: int) -> dict[str, float]:
        """Read the client's local representation health (its on-device monitor).

        Args:
            step: Global step index to tag the reading with.

        Returns:
            The monitor's flat metric dict for the current local model.
        """
        return self.loop.monitor.measure(self.loop.method, step)
