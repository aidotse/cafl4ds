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

import hashlib
from collections.abc import Iterator
from typing import Any

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


def _batches(order: list[tuple[int, int]], batch_size: int, drop_last: bool) -> tuple[list[int], list[int]]:
    """Per-batch ``(sizes, eras)`` under :func:`iter_ordered_batches`'s chunking (era-boundary flush included)."""
    sizes: list[int] = []
    eras: list[int] = []
    buffer = 0
    buffer_era: int | None = None
    for era, _ in order:
        if buffer_era is not None and era != buffer_era and buffer:
            sizes.append(buffer)
            eras.append(buffer_era)
            buffer = 0
        buffer_era = era
        buffer += 1
        if buffer == batch_size:
            sizes.append(buffer)
            eras.append(era)
            buffer = 0
    if buffer and not drop_last and buffer_era is not None:
        sizes.append(buffer)
        eras.append(buffer_era)
    return sizes, eras


def _reserve_probes(
    canary: torch.Tensor,
    stratum: torch.Tensor | None,
    support_per_cell: int,
    query_per_cell: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reserve a balanced held-out probe support/query per canary class (or per ``(stratum, class)`` cell).

    Args:
        canary: Canary label per image ``[N]``.
        stratum: Optional stratum per image ``[N]``; when given, each ``(stratum, canary)`` cell is
            reserved separately.
        support_per_cell: Support images per cell.
        query_per_cell: Query images per cell.
        generator: The fixed reservation generator (seeded by ``canary_seed``).

    Returns:
        ``(support_idx, query_idx, train_mask)`` — the reserved indices and the remaining training mask.

    Raises:
        ValueError: If a cell has too few images for the reservation.
    """
    strata: list[int | None] = sorted(set(stratum.tolist())) if stratum is not None else [None]
    support_idx, query_idx = [], []
    train_mask = torch.ones(canary.shape[0], dtype=torch.bool)
    for st in strata:
        for cls in sorted(set(canary.tolist())):
            in_cell = canary == cls if stratum is None else (canary == cls) & (stratum == st)
            cls_idx = in_cell.nonzero(as_tuple=True)[0]
            perm = cls_idx[torch.randperm(cls_idx.numel(), generator=generator)]
            need = support_per_cell + query_per_cell
            if perm.numel() <= need:
                where = f"canary class {cls}" if st is None else f"(stratum {st}, canary class {cls}) cell"
                raise ValueError(
                    f"{where} has {perm.numel()} images but {need} are reserved for probes; "
                    "reduce support_per_canary / query_per_canary or use more data."
                )
            support_idx.append(perm[:support_per_cell])
            query_idx.append(perm[support_per_cell:need])
            train_mask[perm[:need]] = False
    return torch.cat(support_idx), torch.cat(query_idx), train_mask


def _carve_well(
    era_key: torch.Tensor,
    train_mask: torch.Tensor,
    well_fraction: float,
    well_overlap: float,
    generator: torch.Generator,
) -> list[int]:
    """Carve the warm well's pool out of the training pool, removing its own slice from the drive.

    Each regime reserves ``well_fraction`` of its pool for the well (cleared from ``train_mask`` *in
    place*, so the drive never streams it). With ``well_overlap``, that fraction of each regime's well
    pool is then swapped for drive frames — drawn after the slice, from the same fixed generator, so the
    drive pool is the same at every overlap level and higher levels swap in a superset of lower ones.

    Args:
        era_key: Regime id per image ``[N]``.
        train_mask: The training-pool mask (probes already removed); modified in place.
        well_fraction: Fraction of each regime's pool reserved for the well.
        well_overlap: Fraction of each regime's well pool drawn from the drive instead.
        generator: The fixed reservation generator (seeded by ``canary_seed``).

    Returns:
        The well's image indices (sorted).
    """
    regimes = sorted(set(era_key.tolist()))
    slices: dict[int, list[int]] = {}
    for regime in regimes:
        pool = ((era_key == regime) & train_mask).nonzero(as_tuple=True)[0]
        perm = pool[torch.randperm(pool.numel(), generator=generator)]
        slices[regime] = perm[: int(round(well_fraction * perm.numel()))].tolist()
    for idx in slices.values():
        if idx:
            train_mask[torch.tensor(idx, dtype=torch.long)] = False
    well: list[int] = []
    for regime in regimes:
        own = slices[regime]
        drive = ((era_key == regime) & train_mask).nonzero(as_tuple=True)[0]
        drive = drive[torch.randperm(drive.numel(), generator=generator)]
        swap = min(int(round(well_overlap * len(own))), drive.numel())
        well.extend(own[: len(own) - swap] + drive[:swap].tolist())
    return sorted(well)


class RegimeStream:
    """A single-pass diet that orders images by driving regime and probes an orthogonal canary axis.

    Reserves a balanced held-out probe support/query on the **canary** axis, then orders the remaining
    images into eras that walk the regimes in ``regime_order`` — contiguous by default (maximal
    correlation), interleaved by ``block_size`` (the correlation-strength knob). Delivers label-free
    :class:`StreamBatch` batches whose ``era`` is the regime's position in the walk.

    Two P1.1 options sit on top. ``order="iid"`` delivers the **matched IID control** — the exact
    images the walk delivers, in the walk's own batch-size sequence (so the step count and per-step
    batch size match the walk one-for-one), only shuffled across regimes, every batch tagged
    ``era=0``. Any walk-vs-IID gap at a given step is then attributable to the ordering alone.
    ``well_fraction`` carves a fixed per-regime slice of the training pool out for the warm well, so
    the drive streams frames the well never trained on. ``well_overlap`` then dials *familiarity* back
    in without moving anything else: the well's pool keeps its size, but that fraction of it is swapped
    for frames the drive also streams. The drive itself is identical at every overlap level.
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
        canary_seed: int = 0,
        order: str = "walk",
        stratify_canary: bool = False,
        well_fraction: float = 0.0,
        well_overlap: float = 0.0,
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
            support_per_canary: Images per canary class reserved for the probe support set (per
                ``(stratum, canary)`` cell when ``stratify_canary`` is set).
            query_per_canary: Images per canary class reserved for the probe query / drift set (per
                ``(stratum, canary)`` cell when ``stratify_canary`` is set).
            max_train_per_regime: If set, cap the training images per regime (bounds run length).
            drop_last: Whether to drop the final short batch.
            seed: RNG seed for the **within-regime shuffle** (frame order inside each era, and — with
                ``max_train_per_regime`` — which frames survive). Varies per drive across a seed
                ensemble; does **not** touch the held-out reservation (see ``canary_seed``).
            canary_seed: RNG seed for the **held-out canary reservation** — which support/query
                frames each scene sets aside. Deliberately *decoupled* from ``seed`` and fixed by
                default, so every drive in a seed ensemble (and the warm well, which reserves the
                same way) holds out the **identical** probe set, disjoint from all SSL training. A
                per-drive reservation would leave each drive probed on images an earlier drive's well
                had already seen unsupervised — a soft transductive leak (audit P1.0 A1).
            order: ``"walk"`` (the regime shift-walk) or ``"iid"`` (the matched IID control: the
                walk's images and batch sizes, shuffled across regimes; see the class docstring).
            stratify_canary: Reserve the probe set per ``(stratum, canary)`` cell rather than per
                canary class, and tag every probe image with its stratum (``EvalSet.groups``), so the
                probes can report accuracy per stratum (BDD: per ``timeofday``). Needs a source that
                defines strata.
            well_fraction: Fraction of each regime's training pool reserved for the warm well
                (drawn from ``canary_seed``, so it is identical across drives). :meth:`stationary_batches`
                then draws only from that slice and the drive only from the rest. ``0`` (default)
                shares the whole pool between well and drive.
            well_overlap: Fraction of the well's pool (per regime) drawn from the **drive's** frames
                instead of the well's reserved slice — the familiarity knob. The well pool keeps its
                size and the drive is unchanged, so only how much of the drive the well has already
                seen varies (``0`` = disjoint, ``1`` = the well trains only on drive frames). Drawn
                from ``canary_seed`` (fixed across drives). Needs ``well_fraction > 0``.

        Raises:
            ValueError: If ``block_size`` is non-positive, ``order``, ``well_fraction`` or
                ``well_overlap`` is invalid,
                ``stratify_canary`` is set on a source without strata, or a canary class (cell) has
                too few images to satisfy the requested held-out reservations.
        """
        if block_size is not None and block_size < 1:
            raise ValueError(f"block_size must be a positive integer; got {block_size}.")
        if order not in ("walk", "iid"):
            raise ValueError(f"order must be 'walk' or 'iid'; got {order!r}.")
        if not 0.0 <= well_fraction < 1.0:
            raise ValueError(f"well_fraction must be in [0, 1); got {well_fraction}.")
        if not 0.0 <= well_overlap <= 1.0:
            raise ValueError(f"well_overlap must be in [0, 1]; got {well_overlap}.")
        if well_overlap > 0.0 and well_fraction == 0.0:
            raise ValueError("well_overlap needs well_fraction > 0 (with no reserved slice there is nothing to swap).")
        self.order = order
        self.batch_size = batch_size
        self.block_size = block_size
        self.drop_last = drop_last
        self._generator = torch.Generator().manual_seed(seed)
        reserve_generator = torch.Generator().manual_seed(canary_seed)

        attributed = source.load()
        self._images = attributed.images
        self.regime_names = attributed.regime_names
        self.canary_names = attributed.canary_names
        self._object_categories = attributed.object_categories
        era_key, canary = attributed.era_key, attributed.canary
        self._canary = canary

        # Reserve a balanced held-out probe set on the canary axis; the rest is the training pool.
        # The reservation draws from `reserve_generator` (seeded by `canary_seed`, not the drive
        # `seed`), so the probe set is identical across every drive and the warm well — never a
        # per-drive split that would let a later drive train on an earlier drive's probe images.
        if stratify_canary and attributed.stratum is None:
            raise ValueError("stratify_canary=True needs a source that defines strata (AttributedImages.stratum).")
        stratum = attributed.stratum if stratify_canary else None
        support, query, train_mask = _reserve_probes(
            canary, stratum, support_per_canary, query_per_canary, reserve_generator
        )
        self._split_params: dict[str, Any] = {
            "n_images": int(canary.shape[0]),
            "canary_seed": int(canary_seed),
            "support_per_canary": int(support_per_canary),
            "query_per_canary": int(query_per_canary),
            "stratify_canary": bool(stratify_canary),
            "well_fraction": float(well_fraction),
            "well_overlap": float(well_overlap),
        }
        self._probe_indices = (support, query)
        self._eval_sets = EvalSets(
            probe_support=EvalSet(
                self._images[support], canary[support], stratum[support] if stratum is not None else None
            ),
            probe_query=EvalSet(self._images[query], canary[query], stratum[query] if stratum is not None else None),
            per_era={},
            group_names=dict(attributed.stratum_names) if stratum is not None else {},
        )

        # The well's slice of the training pool (fixed by `canary_seed`), removed from the drive.
        self._well_indices: list[int] | None = None
        if well_fraction > 0.0:
            self._well_indices = _carve_well(era_key, train_mask, well_fraction, well_overlap, reserve_generator)

        requested = regime_order if regime_order is not None else attributed.regime_order
        self._regime_order = requested
        self._order_stream = self._build_order(era_key, train_mask, requested, max_train_per_regime)
        walk_sizes, self._walk_batch_eras = _batches(self._order_stream, batch_size, drop_last)

        # The delivered `(era, image_index)` stream and its batch chunking, per `order`.
        if order == "walk":
            self._delivered = self._order_stream
            self._iid_chunks: list[list[int]] | None = None
        else:
            kept = [idx for _, idx in self._order_stream][: sum(walk_sizes)]
            perm = torch.randperm(len(kept), generator=self._generator).tolist()
            shuffled = [kept[p] for p in perm]
            starts = [sum(walk_sizes[:i]) for i in range(len(walk_sizes))]
            self._iid_chunks = [shuffled[a : a + n] for a, n in zip(starts, walk_sizes, strict=True)]
            self._delivered = [(0, idx) for idx in shuffled]

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

    def stationary_batches(self) -> Iterator[StreamBatch]:
        """Yield the training pool in fully shuffled, regime-agnostic order — the warmup diet.

        Where :meth:`__iter__` delivers the *nonstationary* regime walk, this reshuffles the same
        reserved training pool with no regime ordering (every batch tagged ``era=0``), so a warm-up
        pass sees a stationary all-regime mix. The reserved canary probe set is excluded exactly as in
        the drive, so a well warmed here is competent on the same probes the drive reads. With
        ``well_fraction > 0`` the pool is the well's own slice, disjoint from the drive.

        Yields:
            Stationary :class:`StreamBatch` batches over the training pool.
        """
        indices = self._well_indices if self._well_indices is not None else [idx for _, idx in self._order_stream]
        perm = torch.randperm(len(indices), generator=self._generator)
        order = [(0, indices[p]) for p in perm.tolist()]
        return iter_ordered_batches(order, self._images, self.batch_size, self.drop_last)

    @property
    def familiarity(self) -> dict[str, float] | None:
        """The *realized* well/drive overlap — how familiar the drive is to the well, as measured.

        ``well_overlap`` is a nominal per-regime fraction; rounding per regime makes the realized
        share differ slightly. ``well_side`` is the share of the well's pool the drive also streams;
        ``drive_side`` is the share of the drive's delivered frames the well trained on. ``None``
        when no well slice is carved (``well_fraction=0``: the well and drive share the whole pool).
        """
        if self._well_indices is None:
            return None
        well = set(self._well_indices)
        drive = {idx for _, idx in self._delivered}
        shared = len(well & drive)
        return {
            "well_side": shared / len(well) if well else 0.0,
            "drive_side": shared / len(drive) if drive else 0.0,
        }

    def split_fingerprint(self) -> dict[str, Any]:
        """A fingerprint of the held-out split: the probe and well reservations, plus their settings.

        The probes stay disjoint from a well's training only if the well and the drive reserve
        identically, so :mod:`cafl4ds.warmup` stores this beside a well and the drive refuses a well
        whose fingerprint differs. The ``digest`` hashes the reserved indices themselves (support,
        query, well); the settings are kept beside it to make a mismatch readable.

        Returns:
            The split settings and a ``digest`` over the reserved indices.
        """
        support, query = self._probe_indices
        well = self._well_indices if self._well_indices is not None else []
        digest = hashlib.sha256()
        for part in (support.tolist(), query.tolist(), well):
            digest.update(",".join(map(str, part)).encode())
            digest.update(b"|")
        return {**self._split_params, "digest": digest.hexdigest()}

    @property
    def eval_sets(self) -> EvalSets:
        """The held-out eval sets (probe support/query on the canary axis)."""
        return self._eval_sets

    @property
    def era_names(self) -> dict[int, str]:
        """Map each era (walk position) present in the stream to its human regime name.

        Resolves the era tag carried on every batch — its position in the regime walk — back to the
        source's ``(timeofday·weather)`` name, so the deploy corpus can label each leg. Only eras with
        training data (those actually delivered) appear.
        """
        if self.order == "iid":
            return {0: "iid"}
        present = {era for era, _ in self._order_stream}
        return {era: self.regime_names[self._regime_order[era]] for era in sorted(present)}

    def era_composition(self) -> dict[int, dict[str, Any]]:
        """Per-era label composition of the delivered training pool — the leg's self-description.

        Aggregates, for each era actually delivered, the canary (scene) histogram and — when the
        source carries detection labels (BDD) — the object-category histogram, over the images that
        era streams. This is the label-derived tag set the deploy corpus attaches to each leg, so a
        consumer can read what each ``sunny → rain → …`` block *contained* without the raw frames.

        Returns:
            ``era -> {"n_images", "scene_hist", "category_hist"}`` (histograms sorted for
            determinism; ``category_hist`` is empty when the source has no detection labels).
        """
        composition: dict[int, dict[str, Any]] = {}
        for era, idx in self._delivered:
            entry = composition.setdefault(era, {"n_images": 0, "scene_hist": {}, "category_hist": {}})
            entry["n_images"] += 1
            scene = self.canary_names[int(self._canary[idx])]
            entry["scene_hist"][scene] = entry["scene_hist"].get(scene, 0) + 1
            if self._object_categories is not None:
                for category, count in self._object_categories[idx].items():
                    entry["category_hist"][category] = entry["category_hist"].get(category, 0) + count
        for entry in composition.values():
            entry["scene_hist"] = dict(sorted(entry["scene_hist"].items()))
            entry["category_hist"] = dict(sorted(entry["category_hist"].items()))
        return composition

    @property
    def num_eras(self) -> int:
        """Number of distinct eras delivered (regimes with training data; ``1`` for the IID control)."""
        return len({era for era, _ in self._delivered})

    @property
    def leg_boundary_steps(self) -> list[int]:
        """The steps at which the **walk** opens a new leg (step 0 excluded).

        The state just *before* such a step is the end of the previous leg. Defined from the walk in
        both orders: under ``order="iid"`` the batch sizes match the walk one-for-one, so these are
        the matched steps at which to compare (or checkpoint) the IID control.
        """
        eras = self._walk_batch_eras
        return [step for step in range(1, len(eras)) if eras[step] != eras[step - 1]]

    def __len__(self) -> int:
        """Number of batches the stream will deliver (era-boundary flushes included)."""
        if self._iid_chunks is not None:
            return len(self._iid_chunks)
        return count_ordered_batches(self._order_stream, self.batch_size, self.drop_last)

    def __iter__(self) -> Iterator[StreamBatch]:
        """Yield label-free :class:`StreamBatch` batches (walk order, or the matched IID shuffle)."""
        if self._iid_chunks is not None:
            return (
                StreamBatch(images=self._images[torch.tensor(chunk, dtype=torch.long)], era=0, step=step)
                for step, chunk in enumerate(self._iid_chunks)
            )
        return iter_ordered_batches(self._order_stream, self._images, self.batch_size, self.drop_last)
