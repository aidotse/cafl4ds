"""Reservoir sampling + experience replay — the replay knob (Vitter 1985; baseline B1).

A fixed-capacity buffer maintained by **Vitter's Algorithm R**: after ``N`` frames have passed,
each is present with equal probability ``capacity / N``, so the buffer is a uniform random
sample of the whole (single-pass, unbounded) stream — no matter how correlated the arrival
order. Each step the model trains on the incoming frames **plus** a random replay draw from the
buffer, mixing past eras back into a correlated stream. That interleaving is the "dumb"
protective bar (B1): replay alone, no informativeness, partly counters forgetting.

The buffer is the **replay** role of the ``A`` factor (see :mod:`cafl4ds.filters.base`): it
runs after any admission filters, so what it stores and replays is the *admitted* stream —
upstream :class:`~cafl4ds.filters.dedup.SemDeDup` gives reservoir + dedup (baseline B1.5).
Sampling uses a private, seeded generator so a run is reproducible independently of whatever
else consumes the global RNG.
"""

from __future__ import annotations

import torch

from cafl4ds.filters.base import FilterContext, ReplayBuffer


class ReservoirReplay(ReplayBuffer):
    """Uniform reservoir buffer (Algorithm R) with per-step experience replay."""

    def __init__(self, capacity: int = 256, replay_batch: int = 32, seed: int = 0) -> None:
        """Configure the reservoir.

        Args:
            capacity: Maximum number of frames held in the buffer.
            replay_batch: Number of past frames replayed alongside each incoming batch (a
                uniform draw, without replacement, from the current buffer; clamped to the
                buffer's size, so early steps replay fewer).
            seed: Seed for the buffer's private RNG (reservoir replacement + replay sampling).

        Raises:
            ValueError: If ``capacity`` or ``replay_batch`` is not positive.
        """
        if capacity < 1 or replay_batch < 1:
            raise ValueError(f"capacity and replay_batch must be >= 1; got {capacity}, {replay_batch}.")
        self.capacity = capacity
        self.replay_batch = replay_batch
        self._buffer: list[torch.Tensor] = []
        self._seen = 0  # total frames observed so far (Algorithm R's running count N)
        self._gen = torch.Generator().manual_seed(seed)  # CPU generator; indices only

    def observe(self, images: torch.Tensor, ctx: FilterContext) -> torch.Tensor:
        """Replay from the buffer, then ingest the incoming frames (Algorithm R).

        Replay is sampled from the buffer *before* the incoming frames are ingested, so it is
        drawn from genuine history (never the just-arrived frames), then each incoming frame
        updates the reservoir.

        Args:
            images: The admitted images ``[K, C, H, W]`` for this step.
            ctx: Unused (reservoir replay is model-agnostic and random).

        Returns:
            The training batch ``[K + R, C, H, W]``: the incoming frames followed by ``R``
            replayed frames (``R = min(replay_batch, buffer size before this step)``).
        """
        del ctx  # reservoir selection is model-agnostic
        replay = self._sample_replay()
        for i in range(images.shape[0]):
            self._ingest(images[i])
        return images if replay is None else torch.cat([images, replay], dim=0)

    def _sample_replay(self) -> torch.Tensor | None:
        """Draw a uniform replay batch from the current buffer (without replacement).

        Returns:
            A ``[R, C, H, W]`` stack of replayed frames, or ``None`` if the buffer is empty.
        """
        if not self._buffer:
            return None
        r = min(self.replay_batch, len(self._buffer))
        idx = torch.randperm(len(self._buffer), generator=self._gen)[:r].tolist()
        return torch.stack([self._buffer[i] for i in idx], dim=0)

    def _ingest(self, image: torch.Tensor) -> None:
        """Update the reservoir with one frame via Vitter's Algorithm R.

        While the buffer is not full every frame is stored; once full, the ``N``-th frame
        (0-based index ``N``) replaces a uniformly-random slot with probability
        ``capacity / (N + 1)`` — the invariant that keeps the buffer a uniform sample.

        Args:
            image: A single frame ``[C, H, W]`` to (maybe) store; a detached clone is kept.
        """
        item = image.detach().clone()
        if len(self._buffer) < self.capacity:
            self._buffer.append(item)
        else:
            j = int(torch.randint(0, self._seen + 1, (1,), generator=self._gen).item())
            if j < self.capacity:
                self._buffer[j] = item
        self._seen += 1
