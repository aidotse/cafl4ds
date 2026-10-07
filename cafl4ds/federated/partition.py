"""Client partitioning — split one dataset into per-client shards (the FL ``D`` factor).

Each shard feeds its own client's :class:`~cafl4ds.data.streams.EraStream`. The partition scheme
is where non-IID structure enters: ``"dirichlet"`` is label skew (Hsu et al. 2019; small ``α`` →
clients specialize), ``"iid"`` is the uniform control, and ``"condition"`` is domain skew — clients
split by a per-image condition (BDD's weather / time of day) while sharing one label space.
Partitions are over indices; pixels are sliced into :class:`_ShardSource` views.

Call :func:`holdout_split` first, so the global monitor's pool is disjoint from every client's
data, then partition the remainder.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import torch
from loguru import logger

from cafl4ds.data.attributes import CONDITION_RANKS
from cafl4ds.data.sources import DataSource

# Per-image condition names by attribute, e.g. {"weather": [...], "timeofday": [...]}.
Conditions = dict[str, list[str]]


class _ShardSource(DataSource):
    """A pre-loaded per-client view over one partition's images/labels."""

    def __init__(
        self, images: torch.Tensor, labels: torch.Tensor, num_classes: int, conditions: Conditions | None = None
    ) -> None:
        """Wrap an already-sliced shard as a :class:`DataSource`.

        Args:
            images: The shard's images ``[n, C, H, W]``.
            labels: The shard's labels ``[n]`` (used only for the stream's ordering/eval sets).
            num_classes: The *global* class count (kept constant across shards so class ids are
                comparable even when a shard is missing some classes).
            conditions: The shard's per-image condition names, if the parent source had them.
        """
        self._images = images
        self._labels = labels
        self._num_classes = num_classes
        self.conditions = conditions

    @property
    def num_classes(self) -> int:
        """The global class count (not the number of classes present in this shard)."""
        return self._num_classes

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the shard's ``(images, labels)`` (already in memory)."""
        return self._images, self._labels


def dirichlet_partition(labels: torch.Tensor, num_clients: int, alpha: float, seed: int) -> list[torch.Tensor]:
    """Label-skewed non-IID partition: a ``Dirichlet(α)`` split of each class over clients.

    Args:
        labels: Integer labels ``[N]`` of the full dataset.
        num_clients: Number of client shards to produce.
        alpha: Dirichlet concentration. Small (e.g. 0.1) → sharp per-client class skew; large
            (e.g. 100) → nearly uniform.
        seed: RNG seed for reproducibility.

    Returns:
        One sorted index tensor per client (indices into ``labels``). A client may receive no
        images of a given class (or, at very small ``α``, an empty shard).
    """
    rng = np.random.default_rng(seed)
    labels_np = labels.numpy()
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for cls in np.unique(labels_np):
        idx_c = np.where(labels_np == cls)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        cuts = (np.cumsum(proportions)[:-1] * len(idx_c)).astype(int)
        for client, chunk in enumerate(np.split(idx_c, cuts)):
            client_indices[client].extend(chunk.tolist())
    return [torch.tensor(sorted(ix), dtype=torch.long) for ix in client_indices]


def iid_partition(labels: torch.Tensor, num_clients: int, seed: int) -> list[torch.Tensor]:
    """Uniform random partition (the control): each shard is a fair sample of the whole.

    Args:
        labels: Integer labels ``[N]`` (only its length is used).
        num_clients: Number of client shards to produce.
        seed: RNG seed for the shuffle.

    Returns:
        One sorted index tensor per client, of near-equal size.
    """
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(labels.shape[0], generator=generator)
    return [torch.sort(chunk).values for chunk in perm.tensor_split(num_clients)]


def _conditions_of(source: DataSource) -> Conditions | None:
    """The source's per-image condition names, or ``None`` if it carries none."""
    return getattr(source, "conditions", None)


def _take(conditions: Conditions | None, idx: torch.Tensor) -> Conditions | None:
    """Slice per-image condition names by an index tensor (``None`` passes through)."""
    if conditions is None:
        return None
    rows = idx.tolist()
    return {attr: [names[i] for i in rows] for attr, names in conditions.items()}


def holdout_split(
    source: DataSource, per_class: int, seed: int = 0, strata: str | None = None
) -> tuple[DataSource, DataSource]:
    """Carve a class-balanced hold-out pool out of ``source``, *before* any client sees it.

    The aggregated model must not be scored on images its clients trained on, so the global
    pool is split off first and the two pools are disjoint by construction. Both keep the
    source's original image order (so ``num_clients=1`` still matches a centralized run).

    With ``strata`` (a condition attribute, e.g. ``"weather"``), ``per_class`` images are held out
    per *(condition, class)* cell instead, so every condition is represented in the global readout —
    otherwise a natural-proportion draw is dominated by the common conditions and cannot see a
    model that degrades only on the rare ones. Cells too small to fund the draw are skipped with a
    warning (they stay entirely with the clients).

    Args:
        source: The full dataset.
        per_class: Images per class (per cell, with ``strata``) to reserve for the global monitor.
            Must exceed the global stream's own ``support + query + era_eval`` total, since that
            stream reserves its eval sets *from this pool*.
        seed: RNG seed for the per-class draw.
        strata: Condition attribute to stratify by, or ``None`` for class-only balance.

    Returns:
        ``(holdout, remainder)`` — the global monitor's pool, and the pool to partition.

    Raises:
        ValueError: If ``per_class < 1``, a class holds too few images to fund it (without
            ``strata``), ``strata`` names an attribute the source does not carry, or no cell can
            fund the draw.
    """
    if per_class < 1:
        raise ValueError(f"holdout per_class must be >= 1; got {per_class}.")
    images, labels = source.load()
    conditions = _conditions_of(source)
    if strata is not None and (conditions is None or strata not in conditions):
        raise ValueError(f"holdout strata {strata!r} is not a condition attribute of this source.")
    cell_of = conditions[strata] if strata is not None and conditions is not None else ["all"] * labels.numel()
    cell_names = sorted(set(cell_of))
    cell_ids = torch.tensor([cell_names.index(c) for c in cell_of], dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    held: list[torch.Tensor] = []
    skipped: list[str] = []
    for cell, cell_name in enumerate(cell_names):
        for cls in sorted(set(labels.tolist())):
            idx = ((labels == cls) & (cell_ids == cell)).nonzero(as_tuple=True)[0]
            perm = idx[torch.randperm(idx.numel(), generator=generator)]
            if perm.numel() <= per_class:
                if strata is None:
                    raise ValueError(
                        f"class {cls} has {perm.numel()} images but {per_class} are held out for the global "
                        "monitor; lower the global hold-out or use more data."
                    )
                skipped.append(f"{cell_name}/class{cls} ({perm.numel()})")
                continue
            held.append(perm[:per_class])
    if not held:
        raise ValueError(f"no ({strata}, class) cell holds more than {per_class} images; lower the hold-out.")
    if skipped:
        logger.warning(f"global hold-out: cells too small to hold out {per_class} from, left to clients: {skipped}")
    # Sort so both pools keep the source's original ordering (the num_clients=1 parity property).
    held_idx = torch.cat(held).sort().values
    mask = torch.ones(labels.numel(), dtype=torch.bool)
    mask[held_idx] = False
    rest_idx = mask.nonzero(as_tuple=True)[0]
    unit = "class" if strata is None else f"({strata}, class) cell"
    logger.info(
        f"global hold-out: {held_idx.numel()} images ({per_class}/{unit}) withheld from partitioning; "
        f"{rest_idx.numel()} remain for clients"
    )
    return (
        _ShardSource(images[held_idx], labels[held_idx], source.num_classes, _take(conditions, held_idx)),
        _ShardSource(images[rest_idx], labels[rest_idx], source.num_classes, _take(conditions, rest_idx)),
    )


def _condition_groups(conditions: Conditions, order: Sequence[str]) -> tuple[torch.Tensor, list[str]]:
    """Map each image to a condition group, with group ids ascending along the sort walk.

    A group is one combination of the ``order`` attributes' values (e.g. ``snowy·night``). Groups
    are ranked attribute by attribute — the first attribute is the primary key — using
    :data:`~cafl4ds.data.attributes.CONDITION_RANKS` (unknown values / attributes sort by name), so
    adjacent groups are similar conditions.

    Args:
        conditions: Per-image condition names by attribute.
        order: Attributes to sort by, primary first (e.g. ``["weather", "timeofday"]``).

    Returns:
        ``(group_id [N], group_names)`` — ids index ``group_names``, which are in walk order.

    Raises:
        ValueError: If ``order`` is empty or names an attribute the conditions lack.
    """
    if not order:
        raise ValueError("condition partition needs at least one attribute in `order`.")
    missing = [attr for attr in order if attr not in conditions]
    if missing:
        raise ValueError(f"condition attributes {missing} not carried by the source (has {sorted(conditions)}).")
    combos = list(zip(*(conditions[attr] for attr in order), strict=True))

    def rank(combo: tuple[str, ...]) -> tuple[tuple[int, str], ...]:
        return tuple((CONDITION_RANKS.get(attr, {}).get(v, 99), v) for attr, v in zip(order, combo, strict=True))

    walk = sorted(set(combos), key=rank)
    group_of = {combo: i for i, combo in enumerate(walk)}
    group_id = torch.tensor([group_of[c] for c in combos], dtype=torch.long)
    return group_id, ["·".join(combo) for combo in walk]


def _merge_to_natural(group_sizes: list[int], num_clients: int) -> list[int]:
    """Merge adjacent condition groups until ``num_clients`` remain; return each client's size.

    The smallest adjacent pair is merged first, so the clients stay as close to one pure condition
    each as the client count allows, at the conditions' natural (unequal) sizes.
    """
    sizes = list(group_sizes)
    while len(sizes) > num_clients:
        pair = min(range(len(sizes) - 1), key=lambda i: sizes[i] + sizes[i + 1])
        sizes[pair : pair + 2] = [sizes[pair] + sizes[pair + 1]]
    return sizes


def condition_partition(
    conditions: Conditions,
    num_clients: int,
    order: Sequence[str] = ("weather", "timeofday"),
    sizes: str = "equal",
    mix: float = 0.0,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Domain-skewed non-IID partition: sort images by condition, cut into contiguous client blocks.

    Images are lined up along a walk where similar conditions sit next to each other (see
    :func:`_condition_groups`), then cut into ``num_clients`` contiguous blocks. With
    ``sizes="equal"`` the cuts are at equal counts, so a client whose condition runs out is filled
    up with the *next most similar* condition, and a large condition spans several clients. With
    ``sizes="natural"`` the cuts fall on condition boundaries (adjacent conditions merged, smallest
    first, until ``num_clients`` remain), so clients hold their conditions' natural, unequal sizes.

    ``mix`` softens the split: that fraction of every client's images is pooled, shuffled, and dealt
    back in the same counts, so each client keeps its size but holds some images of other conditions.

    Args:
        conditions: Per-image condition names by attribute.
        num_clients: Number of client shards.
        order: Attributes to sort by, primary first.
        sizes: ``"equal"`` or ``"natural"``.
        mix: Fraction in ``[0, 1]`` of each client's images re-dealt across clients.
        seed: RNG seed (tie order within a condition, and the ``mix`` draw).

    Returns:
        One sorted index tensor per client (indices into the per-image conditions).

    Raises:
        ValueError: On an unknown ``sizes``, ``mix`` outside ``[0, 1]``, ``sizes="natural"`` with
            more clients than condition groups, or more clients than images.
    """
    if sizes not in ("equal", "natural"):
        raise ValueError(f"unknown condition-partition sizes {sizes!r}; expected 'equal' or 'natural'.")
    if not 0.0 <= mix <= 1.0:
        raise ValueError(f"mix must be in [0, 1]; got {mix}.")
    group_id, group_names = _condition_groups(conditions, order)
    if num_clients > group_id.numel():
        raise ValueError(f"{num_clients} clients but only {group_id.numel()} images.")
    generator = torch.Generator().manual_seed(seed)
    # Shuffle, then stable-sort by group: groups line up in walk order, ties in random order.
    perm = torch.randperm(group_id.numel(), generator=generator)
    walk = perm[torch.sort(group_id[perm], stable=True).indices]

    if sizes == "equal":
        blocks = list(walk.tensor_split(num_clients))
    else:
        group_sizes = torch.bincount(group_id, minlength=len(group_names)).tolist()
        if num_clients > len(group_sizes):
            raise ValueError(
                f"sizes='natural' gives at most one client per condition group; {num_clients} clients "
                f"but {len(group_sizes)} groups ({group_names})."
            )
        blocks = list(walk.split(_merge_to_natural(group_sizes, num_clients)))

    if mix > 0.0:
        counts = [round(mix * b.numel()) for b in blocks]
        picks = [b[torch.randperm(b.numel(), generator=generator)] for b in blocks]
        pool = torch.cat([p[:k] for p, k in zip(picks, counts, strict=True)])
        pool = pool[torch.randperm(pool.numel(), generator=generator)]
        dealt = pool.split(counts)
        blocks = [torch.cat([p[k:], d]) for p, k, d in zip(picks, counts, dealt, strict=True)]
    return [torch.sort(b).values for b in blocks]


def _log_composition(conditions: Conditions, order: Sequence[str], index_sets: list[torch.Tensor]) -> None:
    """Log each client's condition mix (top groups by share), so results can be read against it."""
    group_id, group_names = _condition_groups(conditions, order)
    for client, idx in enumerate(index_sets):
        counts = Counter(group_id[idx].tolist())
        top = ", ".join(f"{group_names[g]} {n / idx.numel():.0%}" for g, n in counts.most_common(4))
        rest = "" if len(counts) <= 4 else f", +{len(counts) - 4} more"
        logger.info(f"  client {client} ({idx.numel()} images): {top}{rest}")


def partition_source(
    source: DataSource,
    num_clients: int,
    scheme: str = "dirichlet",
    alpha: float = 0.5,
    seed: int = 0,
    order: Sequence[str] = ("weather", "timeofday"),
    sizes: str = "equal",
    mix: float = 0.0,
) -> list[DataSource]:
    """Split one data source into ``num_clients`` per-client sources.

    Args:
        source: The full dataset to partition (loaded once, here).
        num_clients: Number of client shards.
        scheme: ``"dirichlet"`` (label-skewed non-IID), ``"iid"`` (uniform control), or
            ``"condition"`` (domain-skewed; needs a source carrying per-image conditions).
        alpha: Dirichlet concentration for ``scheme="dirichlet"`` (ignored otherwise).
        seed: RNG seed for the partition.
        order: ``scheme="condition"``: attributes to sort by, primary first.
        sizes: ``scheme="condition"``: ``"equal"`` or ``"natural"`` client sizes.
        mix: ``scheme="condition"``: fraction of each client's images re-dealt across clients.

    Returns:
        A list of ``num_clients`` :class:`DataSource` shards, each ready to feed an
        :class:`~cafl4ds.data.streams.EraStream`.

    Raises:
        ValueError: If ``num_clients < 1``, ``scheme`` is unknown, or ``scheme="condition"`` on a
            source without per-image conditions.

    Note:
        Any per-class eval reservations a client stream makes come from its own shard; under
        sharp skew (small ``alpha``) a shard may not afford them.
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1; got {num_clients}.")
    images, labels = source.load()
    conditions = _conditions_of(source)
    if scheme == "dirichlet":
        index_sets = dirichlet_partition(labels, num_clients, alpha, seed)
        detail = f"alpha={alpha}"
    elif scheme == "iid":
        index_sets = iid_partition(labels, num_clients, seed)
        detail = "uniform"
    elif scheme == "condition":
        if conditions is None:
            raise ValueError(
                "partition scheme 'condition' needs a source with per-image conditions (e.g. data=bdd_fl)."
            )
        index_sets = condition_partition(conditions, num_clients, order, sizes, mix, seed)
        detail = f"order={list(order)}, sizes={sizes}, mix={mix}"
    else:
        raise ValueError(f"unknown partition scheme {scheme!r}; expected 'dirichlet', 'iid' or 'condition'.")
    shards: list[DataSource] = [
        _ShardSource(images[idx], labels[idx], source.num_classes, _take(conditions, idx)) for idx in index_sets
    ]
    shard_sizes = [int(idx.numel()) for idx in index_sets]
    logger.info(f"partition '{scheme}' ({detail}): {num_clients} clients, shard sizes {shard_sizes}")
    if scheme == "condition" and conditions is not None:
        _log_composition(conditions, order, index_sets)
    return shards
