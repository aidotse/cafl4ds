"""Warm a "generally competent" well for the P1.0.2 deployment drive.

Phase 0 read health *after* the representation had settled — a competent well, warmed until the loss
plateaus (P0.3.10's reconstruction-competence plateau) — because reading a still-settling from-scratch
model confounds "the model waking up" with "a response to the stream" (P0.3.9's fresh-head transient).
This module reproduces that discipline for the deployment prototype: it trains a backbone on a
**stationary** all-regime sample (the drive's nonstationarity is deliberately withheld) until the SSL
loss stops improving, then saves the **whole method** — encoder *and* head — as a reusable well.

Saving the whole method is the load-bearing detail. The Phase-0 checkpoint utilities are encoder-only;
resuming a warm encoder onto a *fresh* head would replay the very transient P0.3.9 spent a substudy
removing. A drive that loads this well resumes the model intact, so the trajectory starts from genuine
competence with no head warm-up to misread. The well is produced once per backbone and reused across
drives (identical start state → drive-to-drive comparisons carry no per-run warmup variance).

Compute-agnostic: a single ``device`` string, no hardware branching. The stationary diet comes from
:meth:`cafl4ds.data.regime.RegimeStream.stationary_batches` (the reserved training pool, reshuffled
regime-agnostic), so the well is competent on the same probe set the drive reads.

That last guarantee holds only if the well and the drive reserve the probe set (and the well slice)
identically. :func:`save_split` writes the stream's split fingerprint beside the well, and
:func:`check_split` lets a drive refuse a well reserved differently.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from loguru import logger

from cafl4ds import measurements
from cafl4ds.data.regime import RegimeStream
from cafl4ds.data.streams import EvalSets
from cafl4ds.ssl.base import SSLMethod

# Numerical floor so the plateau ratio is well-defined even at a near-zero loss.
_EPS = 1e-8
# The smallest batch an update runs on (the streaming loop's floor, cafl4ds.loop).
_MIN_BATCH = 2


def competence(method: SSLMethod, sets: EvalSets, knn_k: int = 20) -> dict[str, float]:
    """The method's kNN and linear probe accuracy on a labelled support/query pair (eval mode).

    Args:
        method: The method to read (its training mode is restored).
        sets: The labelled sets — for the well's stop, the stream's selection set.
        knn_k: Neighbours for the kNN probe.

    Returns:
        ``{"knn_acc": ..., "linear_acc": ...}``.
    """
    was_training = method.training
    method.eval()
    try:
        support = (sets.probe_support.images, sets.probe_support.labels)
        query = (sets.probe_query.images, sets.probe_query.labels)
        return {
            "knn_acc": measurements.knn_probe(method.encode, support, query, k=knn_k),
            "linear_acc": measurements.linear_probe(method.encode, support, query),
        }
    finally:
        method.train(was_training)


@dataclass(frozen=True)
class CompetenceCriterion:
    """A competence-plateau stopping rule — stop when the selection accuracy has stopped moving.

    The SSL loss is not a competence signal for every backbone (MAE's loss keeps falling while its
    features stall — "the loss lies", P0.5), so a deep well stops on labelled accuracy instead, read on
    the *selection* set, never the measurement probes. Compares the mean of ``key`` over the readings in
    the most recent ``window`` steps against the ``window`` before it; when the two differ by less than
    ``tol`` (absolute accuracy) **in either direction**, the well is on its plateau.

    Two-sided because a pretrained backbone's competence first *falls* under continued SSL — new heads,
    a new learning rate, a new loss (P1.1.0's ViT-B scout) — and can keep sliding slowly. A rule that only
    waits out gains would stop mid-slide, and the drive would inherit the rest of that slide as an
    apparent loss.

    Attributes:
        key: The accuracy to watch (``"knn_acc"`` / ``"linear_acc"``).
        window: Steps per comparison window.
        tol: The absolute change (up or down) between windows below which the well has settled.
        min_steps: Minimum steps before a plateau may be declared.
    """

    key: str = "knn_acc"
    window: int = 500
    tol: float = 0.01
    min_steps: int = 1000

    def settled(self, trace: list[dict[str, Any]], step: int) -> bool:
        """Whether the competence trace has plateaued by ``step``.

        Args:
            trace: The competence readings so far, each with a ``step`` and the watched ``key``.
            step: The current step count.

        Returns:
            ``True`` once ``min_steps`` and two full windows have passed and the recent window's mean
            moved less than ``tol`` from the previous one's, up or down.
        """
        if step < max(self.min_steps, 2 * self.window):
            return False
        recent = [r[self.key] for r in trace if step - self.window < r["step"] <= step]
        previous = [r[self.key] for r in trace if step - 2 * self.window < r["step"] <= step - self.window]
        if not recent or not previous:
            return False
        return abs(fmean(recent) - fmean(previous)) < self.tol


@dataclass(frozen=True)
class PlateauCriterion:
    """A loss-plateau stopping rule — stop when the loss stops improving.

    Compares the mean loss over the most recent ``window`` steps against the ``window`` before it; when
    the relative improvement falls below ``tol``, the loss has settled. ``min_steps`` guards against
    calling a plateau before the well has had a chance to learn.

    Attributes:
        window: Steps per comparison window.
        tol: Relative-improvement floor — below this, the loss is considered settled.
        min_steps: Minimum steps before a plateau may be declared.
    """

    window: int = 20
    tol: float = 1e-3
    min_steps: int = 40

    def settled(self, losses: list[float]) -> bool:
        """Whether the loss trace has plateaued.

        Args:
            losses: The per-step loss trace, in order.

        Returns:
            ``True`` once enough steps have run and the recent window improved on the previous one by
            less than ``tol`` (relative). A loss that stops falling — or drifts up in noise — settles.
        """
        if len(losses) < self.min_steps or len(losses) < 2 * self.window:
            return False
        previous = fmean(losses[-2 * self.window : -self.window])
        recent = fmean(losses[-self.window :])
        # Normalize on the combined magnitude so the ratio stays bounded in ~[-1, 1] even for a
        # signed loss (e.g. SimSiam's negative cosine) whose windows can straddle zero.
        relative_improvement = (previous - recent) / (abs(previous) + abs(recent) + _EPS)
        return relative_improvement < self.tol


def _check_stop(stop: str, select_every: int, selection: EvalSets | None) -> None:
    """Validate the warm-up's stopping mode against its selection reads (see :func:`warm_until_settled`)."""
    if stop not in ("loss", "competence", "budget"):
        raise ValueError(f"stop must be 'loss', 'competence' or 'budget'; got {stop!r}.")
    if stop == "competence" and select_every <= 0:
        raise ValueError("stop='competence' needs select_every > 0.")
    if select_every > 0 and selection is None:
        raise ValueError("a competence read needs a selection set (select_support/query_per_canary > 0).")


def _update(
    method: SSLMethod, optimizer: torch.optim.Optimizer, images: torch.Tensor, grad_clip: float | None
) -> float:
    """One optimizer step on a batch; returns its loss."""
    optimizer.zero_grad()
    loss = method.training_step(images)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(method.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach().cpu())


class _SelectionLog:
    """The warm-up's selection-competence trace: one reading per call, with the mean loss since the last."""

    def __init__(
        self,
        method: SSLMethod,
        sets: EvalSets | None,
        *,
        every: int,
        knn_k: int,
        extra: Callable[[], dict[str, float]] | None = None,
    ) -> None:
        self.method, self.sets, self.every, self.knn_k, self.extra = method, sets, every, knn_k, extra
        self.trace: list[dict[str, Any]] = []

    def read(self, losses: list[float]) -> None:
        """Append a reading at the current step (off when ``every`` is 0; never twice at one step)."""
        if self.every <= 0 or self.sets is None or (self.trace and self.trace[-1]["step"] == len(losses)):
            return
        row = {"step": float(len(losses)), "loss": fmean(losses[-self.every :]) if losses else None}
        self.trace.append(
            {**row, **competence(self.method, self.sets, self.knn_k), **(self.extra() if self.extra else {})}
        )
        logger.info(f"warm step {len(losses)}: selection {self.trace[-1]}")


def warm_until_settled(
    *,
    method: SSLMethod,
    stream: RegimeStream,
    optimizer: torch.optim.Optimizer,
    plateau: PlateauCriterion | None = None,
    competence_rule: CompetenceCriterion | None = None,
    max_steps: int = 2000,
    max_passes: int = 50,
    grad_clip: float | None = 1.0,
    device: str = "cpu",
    stop: str = "loss",
    select_every: int = 0,
    knn_k: int = 20,
    extra_read: Callable[[], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Train ``method`` on the stream's stationary diet until it settles (or a cap is hit).

    Args:
        method: The SSL method to warm (updated in place).
        stream: The regime stream whose reserved training pool supplies the stationary diet.
        optimizer: Optimizer over ``method.parameters()``.
        plateau: The loss-plateau rule (``stop="loss"``); defaults to :class:`PlateauCriterion`. Its
            ``window`` also smooths the summary's initial/final loss.
        competence_rule: The competence-plateau rule (``stop="competence"``); defaults to
            :class:`CompetenceCriterion`.
        max_steps: Hard cap on optimizer steps (the plateau usually stops first).
        max_passes: Cap on passes over the stationary pool (each pass is freshly reshuffled).
        grad_clip: Global grad-norm clip, or ``None`` to disable.
        device: Torch device (``cpu`` / ``cuda`` / ``hpu``).
        stop: ``"loss"`` (loss plateau), ``"competence"`` (selection-accuracy plateau) or ``"budget"``
            (run to ``max_steps`` / ``max_passes`` — the competence-ceiling scout).
        select_every: Read the selection-set competence every this many steps (and at step 0 and
            the end); ``0`` reads none. ``stop="competence"`` needs it.
        knn_k: Neighbours for the selection kNN probe.
        extra_read: A further read merged into every selection-trace row (e.g. the scout's drive-pool
            read, :mod:`cafl4ds.pool_read`); it is logged, never a stopping signal.

    Returns:
        A warm-up summary: the step count, whether it plateaued, the initial/final smoothed loss, the
        (possibly truncated) loss trace, and the selection ``competence_trace``.

    Raises:
        ValueError: If ``stop`` is unknown, or a competence read is asked of a stream with no
            selection set.
    """
    selection = stream.selection_sets
    _check_stop(stop, select_every, selection)
    loss_rule = plateau or PlateauCriterion()
    competence_stop = competence_rule or CompetenceCriterion()
    method.to(torch.device(device))
    method.train()
    losses: list[float] = []
    log = _SelectionLog(method, selection, every=select_every, knn_k=knn_k, extra=extra_read)
    log.read(losses)
    settled = False
    for _pass in range(max_passes):
        for batch in stream.stationary_batches():
            if batch.images.shape[0] < _MIN_BATCH:  # a 1-image tail cannot update (BatchNorm) — as in the loop
                continue
            losses.append(_update(method, optimizer, batch.images.to(torch.device(device)), grad_clip))
            if len(losses) % max(select_every, 1) == 0:
                log.read(losses)
            if len(losses) >= max_steps:
                break
            if (stop == "loss" and loss_rule.settled(losses)) or (
                stop == "competence" and competence_stop.settled(log.trace, len(losses))
            ):
                settled = True
                break
        if settled or len(losses) >= max_steps:
            break
    log.read(losses)  # the final state (a no-op if the last step was just read)
    trace = log.trace
    window = min(loss_rule.window, len(losses))
    summary = {
        "steps": len(losses),
        "plateaued": settled,
        "init_loss": fmean(losses[:window]) if losses else None,
        "final_loss": fmean(losses[-window:]) if losses else None,
        "loss_trace": losses,
        "competence_trace": trace,
    }
    logger.info(
        f"warm well: {summary['steps']} steps, plateaued={settled}, "
        f"loss {summary['init_loss']:.4f} → {summary['final_loss']:.4f}"
        if losses
        else "warm well: no steps run"
    )
    return summary


def save_well(method: SSLMethod, path: str | Path) -> Path:
    """Save the **whole method** (encoder + head) as a reusable warm well.

    Unlike :func:`cafl4ds.ssl.base.save_encoder_checkpoint` (encoder only), this preserves the head so
    a drive resumes the model intact — no fresh-head transient to misread (P0.3.9).

    Args:
        method: The warmed method to serialize.
        path: Destination path (parent directories are created).

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # CPU-move before serializing (device-agnostic on disk; avoids the Habana storage-copy bug).
    state = {k: v.detach().contiguous().cpu() for k, v in method.state_dict().items()}
    torch.save(state, destination)
    logger.info(f"saved warm well (full method) to {destination}")
    return destination


def load_well(method: SSLMethod, path: str | Path) -> None:
    """Load a full-method warm well into ``method`` in place.

    Args:
        method: The freshly built method to resume (its architecture must match the saved well).
        path: The well checkpoint written by :func:`save_well`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"warm well not found: {source}")
    method.load_state_dict(torch.load(source, map_location="cpu"))
    logger.info(f"resumed warm well (full method) from {source}")


def split_path(well_path: str | Path) -> Path:
    """The split-fingerprint file that sits beside a well (``je.pt`` → ``je.split.json``)."""
    return Path(well_path).with_suffix(".split.json")


def save_split(well_path: str | Path, fingerprint: dict[str, Any]) -> Path:
    """Save the well's split fingerprint (:meth:`RegimeStream.split_fingerprint`) beside it.

    Args:
        well_path: The well checkpoint the fingerprint describes.
        fingerprint: The split fingerprint of the stream the well was warmed on.

    Returns:
        The path written.
    """
    destination = split_path(well_path)
    destination.write_text(json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def check_split(well_path: str | Path, fingerprint: dict[str, Any]) -> None:
    """Refuse a well whose split fingerprint differs from the drive's.

    A well saved before fingerprints existed has none; that is logged as a warning, not refused.

    Args:
        well_path: The well checkpoint the drive resumes (or anchors to).
        fingerprint: The drive stream's split fingerprint.

    Raises:
        ValueError: If the saved fingerprint differs — the probes may not be disjoint from the
            well's training, or the familiarity level is not the one the drive claims.
    """
    source = split_path(well_path)
    if not source.is_file():
        logger.warning(f"no split fingerprint beside {well_path} — cannot verify probe/well disjointness")
        return
    saved = json.loads(source.read_text(encoding="utf-8"))
    differing = sorted(k for k in saved.keys() | fingerprint.keys() if saved.get(k) != fingerprint.get(k))
    if differing:
        detail = ", ".join(f"{k}: well={saved.get(k)!r} drive={fingerprint.get(k)!r}" for k in differing)
        raise ValueError(f"well {well_path} was reserved with a different split ({detail}).")
