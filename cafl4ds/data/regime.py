"""The regime stream — the P1.0.2 deployment diet's attribute-driven era ordering.

Where :class:`~cafl4ds.data.streams.EraStream` orders a *single* class axis into eras (and labels its
eval sets on that same axis), the deployment diet needs the two-axis structure an
:class:`~cafl4ds.data.attributes.AttributeSource` provides: it orders **eras on the era-key axis**
(the driving regime) while reserving its held-out probe set on the **orthogonal canary axis** (so the
labelled canary reads something the diet did not sort by). It is otherwise a faithful sibling of
``EraStream`` — single-pass, a batch is never split across eras, and ``block_size`` is the same
correlation-strength knob — so it drops straight into the Phase-1 harness's ``run_stream_arm``.

Kept in its own module (importing only the stable :class:`~cafl4ds.data.streams.StreamBatch` /
:class:`~cafl4ds.data.streams.EvalSet` / :class:`~cafl4ds.data.streams.EvalSets` dataclasses) so the
Phase-0 stream is untouched.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from cafl4ds.data.attributes import AttributeSource
from cafl4ds.data.streams import EvalSet, EvalSets, StreamBatch


def iter_ordered_batches(
    order: list[tuple[int, int]], images: torch.Tensor, batch_size: int, drop_last: bool
) -> Iterator[StreamBatch]:
    """Yield label-free batches from an ordered ``(era, image_index)`` stream.

    A batch is never split across eras: each batch carries the era of its first sample, and a new
    batch starts at every era boundary (mirroring ``EraStream``'s contract exactly).

    Args:
        order: The ordered ``(era, image_index)`` delivery stream.
        images: The image tensor to index ``[N, C, H, W]``.
        batch_size: Images per batch.
        drop_last: Whether to drop the final short batch of the whole stream.

    Yields:
        Successive :class:`StreamBatch` batches.
    """
    step = 0
    buffer: list[int] = []
    buffer_era: int | None = None
    for era, image_index in order:
        if buffer_era is not None and era != buffer_era and buffer:
            yield StreamBatch(images=images[torch.tensor(buffer, dtype=torch.long)], era=buffer_era, step=step)
            step += 1
            buffer = []
        buffer_era = era
        buffer.append(image_index)
        if len(buffer) == batch_size:
            yield StreamBatch(images=images[torch.tensor(buffer, dtype=torch.long)], era=era, step=step)
            step += 1
            buffer = []
    if buffer and not drop_last and buffer_era is not None:
        yield StreamBatch(images=images[torch.tensor(buffer, dtype=torch.long)], era=buffer_era, step=step)


def count_ordered_batches(order: list[tuple[int, int]], batch_size: int, drop_last: bool) -> int:
    """Count the batches :func:`iter_ordered_batches` will yield (era-boundary flush included).

    Args:
        order: The ordered ``(era, image_index)`` stream.
        batch_size: Images per batch.
        drop_last: Whether the final short batch is dropped.

    Returns:
        The exact batch count.
    """
    count = 0
    buffer = 0
    buffer_era: int | None = None
    for era, _ in order:
        if buffer_era is not None and era != buffer_era and buffer:
            count += 1
            buffer = 0
        buffer_era = era
        buffer += 1
        if buffer == batch_size:
            count += 1
            buffer = 0
    if buffer and not drop_last and buffer_era is not None:
        count += 1
    return count


class RegimeStream:
    """A single-pass diet that orders images by driving regime and probes an orthogonal canary axis.

    Reserves a balanced held-out probe support/query on the **canary** axis, then orders the remaining
    images into eras that walk the regimes in ``regime_order`` — contiguous by default (maximal
    correlation), interleaved by ``block_size`` (the correlation-strength knob). Delivers label-free
    :class:`StreamBatch` batches whose ``era`` is the regime's position in the walk.
    """

    def __init__(
        self,
        source: AttributeSource,
        batch_size: int = 32,
        regime_order: list[int] | None = None,
        block_size: int | None = None,
        support_per_canary: int = 10,
        query_per_canary: int = 10,
        max_train_per_regime: int | None = None,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        """Build the stream (loads the source and constructs the splits eagerly).

        Args:
            source: The attributed data source (two axes: era-key regime + canary).
            batch_size: Images per delivered batch.
            regime_order: Explicit regime walk order; defaults to the source's shift-walk order.
            block_size: Correlation knob — ``None`` delivers each regime contiguously (maximal
                correlation, following the walk); an integer ``b`` round-robins the regimes in chunks
                of ``b`` (interleaving them). Pass a multiple of ``batch_size`` to avoid confounding
                correlation with a shrinking effective batch.
            support_per_canary: Images per canary class reserved for the probe support set.
            query_per_canary: Images per canary class reserved for the probe query / drift set.
            max_train_per_regime: If set, cap the training images per regime (bounds run length).
            drop_last: Whether to drop the final short batch.
            seed: RNG seed for the held-out sampling and within-regime shuffle.

        Raises:
            ValueError: If ``block_size`` is non-positive, or a canary class has too few images to
                satisfy the requested held-out reservations.
        """
        if block_size is not None and block_size < 1:
            raise ValueError(f"block_size must be a positive integer; got {block_size}.")
        self.batch_size = batch_size
        self.block_size = block_size
        self.drop_last = drop_last
        self._generator = torch.Generator().manual_seed(seed)

        attributed = source.load()
        self._images = attributed.images
        self.regime_names = attributed.regime_names
        self.canary_names = attributed.canary_names
        era_key, canary = attributed.era_key, attributed.canary

        # Reserve a balanced held-out probe set on the canary axis; the rest is the training pool.
        support_idx, query_idx, train_mask = [], [], torch.ones(self._images.shape[0], dtype=torch.bool)
        for cls in sorted(set(canary.tolist())):
            cls_idx = (canary == cls).nonzero(as_tuple=True)[0]
            perm = cls_idx[torch.randperm(cls_idx.numel(), generator=self._generator)]
            need = support_per_canary + query_per_canary
            if perm.numel() <= need:
                raise ValueError(
                    f"canary class {cls} has {perm.numel()} images but {need} are reserved for probes; "
                    "reduce support_per_canary / query_per_canary or use more data."
                )
            support_idx.append(perm[:support_per_canary])
            query_idx.append(perm[support_per_canary:need])
            train_mask[perm[:need]] = False

        self._eval_sets = EvalSets(
            probe_support=EvalSet(self._images[torch.cat(support_idx)], canary[torch.cat(support_idx)]),
            probe_query=EvalSet(self._images[torch.cat(query_idx)], canary[torch.cat(query_idx)]),
            per_era={},
        )

        requested = regime_order if regime_order is not None else attributed.regime_order
        self._order_stream = self._build_order(era_key, train_mask, requested, max_train_per_regime)

    def _build_order(
        self, era_key: torch.Tensor, train_mask: torch.Tensor, regime_order: list[int], max_train_per_regime: int | None
    ) -> list[tuple[int, int]]:
        """Order the training pool into eras walking ``regime_order`` (era = walk position).

        Args:
            era_key: Per-image regime id ``[N]``.
            train_mask: Boolean mask of training (non-reserved) images ``[N]``.
            regime_order: The regime ids to walk, in order (regimes with no training data are skipped).
            max_train_per_regime: Optional per-regime training cap.

        Returns:
            The ordered ``(era, image_index)`` stream (contiguous eras, or round-robin if
            ``block_size`` is set).
        """
        train_by_era: list[tuple[int, list[int]]] = []
        for era, regime in enumerate(regime_order):
            regime_idx = ((era_key == regime) & train_mask).nonzero(as_tuple=True)[0]
            if regime_idx.numel() == 0:
                continue
            shuffled = regime_idx[torch.randperm(regime_idx.numel(), generator=self._generator)].tolist()
            if max_train_per_regime is not None:
                shuffled = shuffled[:max_train_per_regime]
            train_by_era.append((era, shuffled))
        if self.block_size is not None:
            return self._round_robin(train_by_era, self.block_size)
        return [(era, i) for era, items in train_by_era for i in items]

    @staticmethod
    def _round_robin(train_by_era: list[tuple[int, list[int]]], block_size: int) -> list[tuple[int, int]]:
        """Round-robin the eras in chunks of ``block_size`` (the correlation-strength knob).

        Emits ``block_size`` images of the first era, then ``block_size`` of the next, … and repeats
        until every era is exhausted; the same-regime run length is ``block_size`` (a large value
        recovers contiguous regimes). The ``era`` tag stays the regime's walk position across
        recurrences.

        Args:
            train_by_era: ``(era, indices)`` pairs of training images per regime, in walk order.
            block_size: Number of consecutive same-era images per chunk.

        Returns:
            The round-robin ``(era, image_index)`` stream.
        """
        pointers = [0] * len(train_by_era)
        ordered: list[tuple[int, int]] = []
        while any(pointers[k] < len(train_by_era[k][1]) for k in range(len(train_by_era))):
            for k, (era, items) in enumerate(train_by_era):
                start = pointers[k]
                for i in items[start : start + block_size]:
                    ordered.append((era, i))
                pointers[k] = start + block_size
        return ordered

    @property
    def eval_sets(self) -> EvalSets:
        """The held-out eval sets (probe support/query on the canary axis)."""
        return self._eval_sets

    @property
    def num_eras(self) -> int:
        """Number of distinct eras (regimes with training data) in the walk."""
        return len({era for era, _ in self._order_stream})

    def __len__(self) -> int:
        """Number of batches the stream will deliver (era-boundary flushes included)."""
        return count_ordered_batches(self._order_stream, self.batch_size, self.drop_last)

    def __iter__(self) -> Iterator[StreamBatch]:
        """Yield label-free :class:`StreamBatch` batches in walk order."""
        return iter_ordered_batches(self._order_stream, self._images, self.batch_size, self.drop_last)
