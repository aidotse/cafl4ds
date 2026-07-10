"""Streaming datasets: order a :class:`~cafl4ds.data.sources.DataSource` into eras.

The stream is the ``F`` (data-order) factor of the experiment. Phase 0 uses a single
:class:`EraStream` with two orderings:

* ``order="class_blocked"`` — the **synthetic correlation** of Phase 1a: all of class 0, then
  all of class 1, …. Each class block is an **era**. This is the correlated stream whose
  degradation the instruments must catch.
* ``order="iid"`` — a shuffled single pass (one era). This is the *well-behaved reference*
  used to produce the ``I=pretrained`` warm-start checkpoint — correlation must **not** leak
  into the clean starting point.

Labels are used only to (a) build the class-blocked ordering and (b) construct held-out eval
sets; they never appear in a :class:`StreamBatch`, so they can never enter the SSL update.
Each stream reserves, per class, a disjoint held-out **probe support**, **probe query**, and
**per-era** eval set before ordering the remaining images into the training stream.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from cafl4ds.data.sources import DataSource


@dataclass(frozen=True)
class StreamBatch:
    """One batch delivered by a stream — images only, tagged with its era and step."""

    images: torch.Tensor
    """Image batch ``[B, C, H, W]``. Contains no labels, by construction."""
    era: int
    """Index of the era (class block) this batch belongs to."""
    step: int
    """Global step index of this batch within the single-pass stream."""


@dataclass(frozen=True)
class EvalSet:
    """A held-out labeled evaluation set. Labels are used HERE ONLY (probes/eval)."""

    images: torch.Tensor
    """Images ``[M, C, H, W]``."""
    labels: torch.Tensor
    """Integer labels ``[M]`` — used only by the probes, never by training."""


@dataclass(frozen=True)
class EvalSets:
    """The held-out eval sets a stream exposes to the health monitor."""

    probe_support: EvalSet
    """Balanced held-out set the probes fit on (kNN database / linear-head training)."""
    probe_query: EvalSet
    """Balanced held-out set the probes score on; also the *fixed reference* whose embeddings
    the drift instruments track across checkpoints."""
    per_era: dict[int, EvalSet] = field(default_factory=dict)
    """Per-era (per-class) held-out sets, reserved for later per-era forgetting measures."""


class EraStream:
    """A single-pass stream that orders a data source into eras.

    Reserves disjoint per-class held-out eval sets, then orders the remaining images either
    as class blocks (``"class_blocked"``) or shuffled (``"iid"``), and delivers them as
    label-free :class:`StreamBatch` batches.
    """

    def __init__(
        self,
        source: DataSource,
        batch_size: int = 32,
        order: str = "class_blocked",
        class_order: list[int] | None = None,
        support_per_class: int = 20,
        query_per_class: int = 20,
        era_eval_per_class: int = 10,
        max_train_per_class: int | None = None,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        """Build the stream (loads the source and constructs the splits eagerly).

        Args:
            source: The data source to order.
            batch_size: Number of images per delivered batch.
            order: ``"class_blocked"`` (eras = class blocks) or ``"iid"`` (one shuffled era).
            class_order: Explicit class ordering for ``"class_blocked"``; defaults to
                ``0, 1, …, num_classes - 1``.
            support_per_class: Images per class reserved for the probe support set.
            query_per_class: Images per class reserved for the probe query / drift set.
            era_eval_per_class: Images per class reserved for the per-era eval set.
            max_train_per_class: If set, cap the *training* images per class after the
                held-out reservations (keeps smoke runs short).
            drop_last: Whether to drop a final short batch.
            seed: RNG seed for the held-out sampling and IID shuffle.

        Raises:
            ValueError: If ``order`` is unknown, or a class has too few images to satisfy the
                requested held-out reservations.
        """
        if order not in ("class_blocked", "iid"):
            raise ValueError(f"unknown order {order!r}; expected 'class_blocked' or 'iid'.")
        self.batch_size = batch_size
        self.order = order
        self.drop_last = drop_last
        self._generator = torch.Generator().manual_seed(seed)

        images, labels = source.load()
        self._images = images
        classes = sorted(set(labels.tolist()))
        self._class_order = class_order if class_order is not None else classes

        train_indices_by_era: list[tuple[int, torch.Tensor]] = []
        support_idx, query_idx = [], []
        per_era_eval: dict[int, EvalSet] = {}
        for era, cls in enumerate(self._class_order):
            cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
            perm = cls_idx[torch.randperm(cls_idx.numel(), generator=self._generator)]
            need = support_per_class + query_per_class + era_eval_per_class
            if perm.numel() <= need:
                raise ValueError(
                    f"class {cls} has {perm.numel()} images but {need} are reserved for eval; "
                    "reduce the per-class reservations or use more data."
                )
            s, q, e = support_per_class, query_per_class, era_eval_per_class
            support_idx.append(perm[:s])
            query_idx.append(perm[s : s + q])
            era_eval = perm[s + q : s + q + e]
            train = perm[s + q + e :]
            if max_train_per_class is not None:
                train = train[:max_train_per_class]
            per_era_eval[era] = EvalSet(images[era_eval], labels[era_eval])
            train_indices_by_era.append((era, train))

        self._eval_sets = EvalSets(
            probe_support=self._make_eval(torch.cat(support_idx), labels),
            probe_query=self._make_eval(torch.cat(query_idx), labels),
            per_era=per_era_eval,
        )
        self._order_stream = self._build_order(train_indices_by_era)

    def _make_eval(self, idx: torch.Tensor, labels: torch.Tensor) -> EvalSet:
        """Materialize an :class:`EvalSet` from an index tensor.

        Args:
            idx: Indices into the loaded images/labels.
            labels: The full label vector.

        Returns:
            The corresponding held-out :class:`EvalSet`.
        """
        return EvalSet(self._images[idx], labels[idx])

    def _build_order(self, train_by_era: list[tuple[int, torch.Tensor]]) -> list[tuple[int, int]]:
        """Flatten per-era training indices into an ordered ``(era, image_index)`` stream.

        Args:
            train_by_era: ``(era, indices)`` pairs of training images per era.

        Returns:
            A list of ``(era, image_index)`` in the stream's delivery order.
        """
        ordered: list[tuple[int, int]] = []
        for era, idx in train_by_era:
            for i in idx.tolist():
                ordered.append((era, i))
        if self.order == "iid":
            # IID is a single era: shuffle globally and relabel every item to era 0 so batches
            # are not fragmented at (now meaningless) class boundaries.
            perm = torch.randperm(len(ordered), generator=self._generator).tolist()
            ordered = [(0, ordered[i][1]) for i in perm]
        return ordered

    @property
    def eval_sets(self) -> EvalSets:
        """The held-out eval sets (probe support/query and per-era)."""
        return self._eval_sets

    @property
    def num_eras(self) -> int:
        """Number of eras in the stream (class blocks, or 1 for IID)."""
        return 1 if self.order == "iid" else len(self._class_order)

    def __len__(self) -> int:
        """Number of batches the stream will deliver."""
        n = len(self._order_stream)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[StreamBatch]:
        """Yield label-free :class:`StreamBatch` batches in stream order.

        A batch is never split across eras: each batch carries a single era id (the era of
        its first sample), and a new batch is started at every era boundary.

        Yields:
            Successive :class:`StreamBatch` batches over the single-pass stream.
        """
        step = 0
        buffer_idx: list[int] = []
        buffer_era: int | None = None
        for era, image_index in self._order_stream:
            if buffer_era is not None and era != buffer_era and buffer_idx:
                yield self._make_batch(buffer_idx, buffer_era, step)
                step += 1
                buffer_idx = []
            buffer_era = era
            buffer_idx.append(image_index)
            if len(buffer_idx) == self.batch_size:
                yield self._make_batch(buffer_idx, era, step)
                step += 1
                buffer_idx = []
        if buffer_idx and not self.drop_last and buffer_era is not None:
            yield self._make_batch(buffer_idx, buffer_era, step)

    def _make_batch(self, idx: list[int], era: int, step: int) -> StreamBatch:
        """Assemble a :class:`StreamBatch` from buffered indices.

        Args:
            idx: Image indices for this batch.
            era: Era id for this batch.
            step: Global step index.

        Returns:
            The assembled label-free batch.
        """
        return StreamBatch(images=self._images[torch.tensor(idx, dtype=torch.long)], era=era, step=step)
