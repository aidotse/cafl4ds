"""The streaming SSL adaptation loop (Phase 0).

Ties the pieces together: pull batches from the :class:`~cafl4ds.data.streams.EraStream`, pass
each through the :class:`~cafl4ds.filters.base.Filter` (Phase 0: the B-floor accept-all knob),
run one self-supervised update on the accepted images, log the loss every step, and every
``eval_every`` steps read the :class:`~cafl4ds.monitor.HealthMonitor` and log the health
series. Loss and health land in the *same* run log as *separate* series so they can be read
side by side over steps (the Phase-0 exit criterion).

This is a single-pass streaming loop: no replay, no shuffling of the incoming order — the
model sees the correlated stream exactly as it arrives.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from loguru import logger
from torch import optim

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import Filter, FilterContext
from cafl4ds.monitor import HealthMonitor
from cafl4ds.run_log import RunLogger
from cafl4ds.ssl.base import SSLMethod

# BatchNorm heads (SimSiam) and per-patch stats need at least two samples in a batch.
_MIN_BATCH = 2


class StreamingLoop:
    """Runs the streaming SSL adaptation loop with a single selection filter."""

    def __init__(
        self,
        stream: Iterable[StreamBatch],
        method: SSLMethod,
        optimizer: optim.Optimizer,
        selection_filter: Filter,
        monitor: HealthMonitor,
        run_logger: RunLogger,
        eval_every: int = 5,
        grad_clip: float | None = 1.0,
        device: str = "cpu",
    ) -> None:
        """Build the loop.

        Args:
            stream: An iterable of :class:`~cafl4ds.data.streams.StreamBatch` (the ``F`` factor).
            method: The SSL method to adapt (the ``C`` factor).
            optimizer: Optimizer over ``method.parameters()``.
            selection_filter: The selection knob (the ``A`` factor); Phase 0 uses B-floor.
            monitor: The health monitor read every ``eval_every`` steps.
            run_logger: The run log receiving the loss and health series.
            eval_every: Run the monitor every this many steps (and always at the end).
            grad_clip: Global grad-norm clip value, or ``None`` to disable.
            device: Torch device to run on (``"cpu"`` for Phase-0 smoke; ``"hpu"`` later).
        """
        self.stream = stream
        self.method = method
        self.optimizer = optimizer
        self.selection_filter = selection_filter
        self.monitor = monitor
        self.run_logger = run_logger
        self.eval_every = eval_every
        self.grad_clip = grad_clip
        self.device = torch.device(device)

    def run(self) -> RunLogger:
        """Run the full single-pass stream, logging loss and health.

        Returns:
            The run logger (closed), for convenience.
        """
        self.method.to(self.device)
        last_step, last_era = -1, 0
        for batch in self.stream:
            step, last_era = batch.step, batch.era
            moved = StreamBatch(images=batch.images.to(self.device), era=batch.era, step=batch.step)
            accepted = self.selection_filter.select(moved, FilterContext(method=self.method, step=step))
            if accepted.shape[0] < _MIN_BATCH:
                logger.debug(f"step {step}: skipping batch of {accepted.shape[0]} (< {_MIN_BATCH}).")
                continue
            loss = self._update(accepted)
            self.run_logger.log_loss(step, batch.era, loss)
            if step % self.eval_every == 0:
                self.run_logger.log_health(step, batch.era, self.monitor.measure(self.method, step))
            last_step = step
        if last_step >= 0 and last_step % self.eval_every != 0:  # always end on a health reading
            self.run_logger.log_health(last_step, last_era, self.monitor.measure(self.method, last_step))
        logger.info("streaming loop complete\n" + self.run_logger.tabulate())
        self.run_logger.close()
        return self.run_logger

    def _update(self, images: torch.Tensor) -> float:
        """Run one SSL optimization step on the accepted images.

        Args:
            images: The accepted image batch ``[K, C, H, W]``.

        Returns:
            The scalar loss value for this step.
        """
        self.method.train()
        self.optimizer.zero_grad()
        loss = self.method.training_step(images)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.method.parameters(), self.grad_clip)
        self.optimizer.step()
        return float(loss.item())
