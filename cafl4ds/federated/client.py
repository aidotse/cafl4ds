"""The federated client — one local streaming learner, resumable across rounds.

A client owns what a centralized run owns (model, optimizer, filter, monitor, and its own
:class:`~cafl4ds.data.streams.EraStream`), bundled as a :class:`~cafl4ds.loop.StreamingLoop`.
It holds one persistent iterator over its stream, so rounds slice a single true single-pass
stream; once it is exhausted the client stops participating. With ``epochs > 1`` the client
restarts the stream (same order) until it has made that many passes — the federated counterpart
of :class:`~cafl4ds.loop.StreamingLoop`'s ``epochs``. A pass spans many rounds, so this adds
rounds, not local steps between averages. :meth:`FederatedClient.set_weights`
overwrites the model only — the optimizer and filter state (e.g. a reservoir buffer) are the
client's private memory and are never synced.

With ``keep_heads_local`` the method's :attr:`~cafl4ds.ssl.base.SSLMethod.local_heads` (SimSiam's
predictor, MAE's decoder) are private too: they are neither sent nor overwritten, so only the
rest of the model is averaged.

A round is measured in **stream steps** (batches pulled), not optimizer updates, so
selection-induced differences in how much each client trains show up in the reported sample
count rather than being normalized away.
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

    def __init__(self, client_id: int, loop: StreamingLoop, epochs: int = 1, keep_heads_local: bool = False) -> None:
        """Wrap a per-client streaming loop as a federated client.

        Args:
            client_id: Stable identifier for logging/aggregation bookkeeping.
            loop: The client's local loop (its own model, optimizer, filter, monitor, stream).
            epochs: Passes over the stream before the client is exhausted. ``1`` (default) is the
                single-pass stream.
            keep_heads_local: Keep the method's ``local_heads`` private: exchange and average
                only the rest of the model. ``False`` (default) exchanges the whole model.

        Raises:
            ValueError: If ``epochs < 1``.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1; got {epochs}.")
        self.client_id = client_id
        self.loop = loop
        self.epochs = epochs
        self.local_heads: tuple[str, ...] = loop.method.local_heads if keep_heads_local else ()
        if keep_heads_local and not self.local_heads:
            logger.warning(f"keep_heads_local: {loop.method.name} declares no local heads; sharing the whole model.")
        self.loop.method.to(self.loop.device)
        self._iterator: Iterator[StreamBatch] = iter(loop.stream)
        self._passes_done = 0
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

    def is_local(self, key: str) -> bool:
        """Whether ``state_dict`` entry ``key`` belongs to a private head (never exchanged)."""
        return any(key.startswith(f"{head}.") for head in self.local_heads)

    def get_weights(self) -> StateDict:
        """Return a detached clone of the shared part of the local ``state_dict`` (for aggregation)."""
        return {k: v.detach().clone() for k, v in self.loop.method.state_dict().items() if not self.is_local(k)}

    def load_weights(self, state: StateDict) -> None:
        """Load shared weights into the local model, leaving the private heads untouched.

        Args:
            state: A ``state_dict`` holding at least every shared entry. Private-head entries,
                if present (e.g. a full initial state), are loaded too.

        Raises:
            RuntimeError: If ``state`` has unknown keys or lacks a shared entry.
        """
        result = self.loop.method.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if not self.is_local(k)]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"state mismatch: missing {missing}, unexpected {list(result.unexpected_keys)}.")

    def set_weights(self, state: StateDict) -> None:
        """Load broadcast global weights into the local model (model only).

        Also re-ties the FedProx anchor, if any, so the leash is measured against this round's
        consensus. Private heads have no consensus, so they are not leashed.

        Args:
            state: The global ``state_dict`` to load.
        """
        self.load_weights(state)
        if self.loop.proximal is not None:
            self.loop.proximal.set_anchor(state)

    def train_round(self, steps_per_round: int) -> RoundResult:
        """Advance the local stream by ``steps_per_round`` and train on what is admitted.

        Args:
            steps_per_round: Number of batches to pull from the (persistent) stream this round.

        Returns:
            A :class:`RoundResult` summarizing the round.
        """
        self.loop.method.to(self.loop.device)
        steps_pulled, num_trained = 0, 0
        loss_sum, loss_count = 0.0, 0
        for _ in range(steps_per_round):
            batch = self._next_batch()
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

    def _next_batch(self) -> StreamBatch | None:
        """Pull the next batch, restarting the stream (same order) while passes remain.

        A pass that ends mid-round continues seamlessly into the next one, as in the centralized
        multi-epoch loop.

        Returns:
            The next batch, or ``None`` once the last pass is exhausted.
        """
        batch = next(self._iterator, None)
        if batch is None and self._passes_done + 1 < self.epochs:
            self._passes_done += 1
            self._iterator = iter(self.loop.stream)
            batch = next(self._iterator, None)
        return batch

    def measure_health(self, step: int) -> dict[str, float]:
        """Read the client's local representation health (its on-device monitor).

        Args:
            step: Global step index to tag the reading with.

        Returns:
            The monitor's flat metric dict for the current local model.
        """
        return self.loop.monitor.measure(self.loop.method, step)
