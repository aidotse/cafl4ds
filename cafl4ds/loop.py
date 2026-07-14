"""The streaming SSL adaptation loop (Phase 0).

Ties the pieces together: pull batches from the :class:`~cafl4ds.data.streams.EraStream`, pass
each through the :class:`~cafl4ds.filters.base.Filter` (Phase 0: the B-floor accept-all knob),
run one self-supervised update on the accepted images, log the loss every step, and every
``eval_every`` steps read the :class:`~cafl4ds.monitor.HealthMonitor` and log the health
series. Loss and health land in the *same* run log as *separate* series so they can be read
side by side over steps (the Phase-0 exit criterion).

By default this is a single-pass streaming loop: no replay, no shuffling of the incoming
order — the model sees the correlated stream exactly as it arrives. It also supports a
**multi-epoch** regime (``epochs > 1``, the stream re-iterated in the same order) with an
optional per-step **LR scheduler** — the fair training regime the P0.2.1 healthy-baseline
calibration needs (the correlated single pass is still the default, and the Phase-1 setting).
"""

from __future__ import annotations

from collections.abc import Iterable, Sized

import torch
from loguru import logger
from torch import optim
from torch.optim.lr_scheduler import LRScheduler

from cafl4ds.data.streams import StreamBatch
from cafl4ds.eval import PerEraProbe
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
        epochs: int = 1,
        scheduler: LRScheduler | None = None,
        grad_clip: float | None = 1.0,
        device: str = "cpu",
        era_evaluator: PerEraProbe | None = None,
    ) -> None:
        """Build the loop.

        Args:
            stream: An iterable of :class:`~cafl4ds.data.streams.StreamBatch` (the ``F`` factor).
            method: The SSL method to adapt (the ``C`` factor).
            optimizer: Optimizer over ``method.parameters()``.
            selection_filter: The selection knob (the ``A`` factor); Phase 0 uses B-floor.
            monitor: The health monitor read every ``eval_every`` (global) steps.
            run_logger: The run log receiving the loss and health series.
            eval_every: Run the monitor every this many steps (and always at the end).
            epochs: Number of passes over the stream. ``1`` (default) is the single-pass
                streaming setting; ``>1`` re-iterates the same stream in the same order (the
                multi-epoch calibration regime). Steps are numbered **globally** across epochs.
            scheduler: Optional LR scheduler stepped once per optimizer step (e.g.
                :func:`~cafl4ds.schedule.warmup_cosine_schedule`), or ``None`` for a flat LR.
            grad_clip: Global grad-norm clip value, or ``None`` to disable.
            device: Torch device to run on (``"cpu"`` for Phase-0 smoke; ``"hpu"`` later).
            era_evaluator: Optional probe-on-past evaluator (the *validating* axis). When set,
                the current encoder is probed over all seen eras at each era boundary and at the
                end, building the accuracy matrix behind Backward Transfer / Forgetting. ``None``
                (default) runs no downstream probing — the loop is unchanged.
        """
        self.stream = stream
        self.method = method
        self.optimizer = optimizer
        self.selection_filter = selection_filter
        self.monitor = monitor
        self.run_logger = run_logger
        self.eval_every = eval_every
        self.epochs = epochs
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.device = torch.device(device)
        self.era_evaluator = era_evaluator

    def run(self) -> RunLogger:
        """Run the stream (``epochs`` passes), logging loss and health over global steps.

        Returns:
            The run logger (closed), for convenience.
        """
        self.method.to(self.device)
        # Global step numbering across epochs. batch.step restarts at 0 each pass, so offset
        # by a fixed batches-per-epoch stride to keep steps monotonic (and the eval cadence /
        # drift reference consistent across a multi-epoch run). A single pass needs no offset.
        if self.epochs > 1 and not isinstance(self.stream, Sized):
            raise TypeError("multi-epoch loop (epochs > 1) requires a stream with a known length.")
        batches_per_epoch = len(self.stream) if isinstance(self.stream, Sized) else 0
        last_step, last_era = -1, 0
        prev_era: int | None = None
        for epoch in range(self.epochs):
            for batch in self.stream:
                step = epoch * batches_per_epoch + batch.step
                if prev_era is not None and batch.era != prev_era:
                    self._probe_past(prev_era)  # the era just ended — record its probe-on-past row
                prev_era = batch.era
                last_era = batch.era
                moved = StreamBatch(images=batch.images.to(self.device), era=batch.era, step=step)
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
        if last_step >= 0:
            self._probe_past(last_era)  # final row: current encoder over every era seen
        logger.info("streaming loop complete\n" + self.run_logger.tabulate())
        self.run_logger.close()
        return self.run_logger

    def _probe_past(self, era: int) -> None:
        """Record one probe-on-past row for the just-finished ``era`` (if an evaluator is set).

        Probes in eval mode (BatchNorm/dropout off, as the health monitor does), then restores
        the training mode so the loop is unperturbed.

        Args:
            era: Index of the era that just completed; the encoder is scored on eras ``0..era``.
        """
        if self.era_evaluator is None:
            return
        was_training = self.method.training
        self.method.eval()
        try:
            self.era_evaluator.record(self.method.encode, era)
        finally:
            self.method.train(was_training)

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
        if self.scheduler is not None:
            self.scheduler.step()
        return float(loss.item())
