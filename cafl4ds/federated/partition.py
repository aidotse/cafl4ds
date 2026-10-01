"""Client partitioning — split one dataset into per-client shards (the FL ``D`` factor).

Each shard feeds its own client's :class:`~cafl4ds.data.streams.EraStream`. The partition scheme
is where non-IID structure enters: ``"dirichlet"`` is label skew (Hsu et al. 2019; small ``α`` →
clients specialize), ``"iid"`` is the uniform control. Partitions are over indices; pixels are
sliced into :class:`_ShardSource` views.

Call :func:`holdout_split` first, so the global monitor's pool is disjoint from every client's
data, then partition the remainder.
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

from cafl4ds.data.sources import DataSource


class _ShardSource(DataSource):
    """A pre-loaded per-client view over one partition's images/labels."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, num_classes: int) -> None:
        """Wrap an already-sliced shard as a :class:`DataSource`.

        Args:
            images: The shard's images ``[n, C, H, W]``.
            labels: The shard's labels ``[n]`` (used only for the stream's ordering/eval sets).
            num_classes: The *global* class count (kept constant across shards so class ids are
                comparable even when a shard is missing some classes).
        """
        self._images = images
        self._labels = labels
        self._num_classes = num_classes

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


def holdout_split(source: DataSource, per_class: int, seed: int = 0) -> tuple[DataSource, DataSource]:
    """Carve a class-balanced hold-out pool out of ``source``, *before* any client sees it.

    The aggregated model must not be scored on images its clients trained on, so the global
    pool is split off first and the two pools are disjoint by construction. Both keep the
    source's original image order (so ``num_clients=1`` still matches a centralized run).

    Args:
        source: The full dataset.
        per_class: Images per class to reserve for the global monitor. Must exceed the global
            stream's own ``support + query + era_eval`` total, since that stream reserves its
            eval sets *from this pool*.
        seed: RNG seed for the per-class draw.

    Returns:
        ``(holdout, remainder)`` — the global monitor's pool, and the pool to partition.

    Raises:
        ValueError: If ``per_class < 1``, or a class holds too few images to fund it.
    """
    if per_class < 1:
        raise ValueError(f"holdout per_class must be >= 1; got {per_class}.")
    images, labels = source.load()
    generator = torch.Generator().manual_seed(seed)
    held: list[torch.Tensor] = []
    rest: list[torch.Tensor] = []
    for cls in sorted(set(labels.tolist())):
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(idx.numel(), generator=generator)]
        if perm.numel() <= per_class:
            raise ValueError(
                f"class {cls} has {perm.numel()} images but {per_class} are held out for the global "
                "monitor; lower the global hold-out or use more data."
            )
        held.append(perm[:per_class])
        rest.append(perm[per_class:])
    # Sort so both pools keep the source's original ordering (the num_clients=1 parity property).
    held_idx = torch.cat(held).sort().values
    rest_idx = torch.cat(rest).sort().values
    logger.info(
        f"global hold-out: {held_idx.numel()} images ({per_class}/class) withheld from partitioning; "
        f"{rest_idx.numel()} remain for clients"
    )
    return (
        _ShardSource(images[held_idx], labels[held_idx], source.num_classes),
        _ShardSource(images[rest_idx], labels[rest_idx], source.num_classes),
    )


def partition_source(
    source: DataSource,
    num_clients: int,
    scheme: str = "dirichlet",
    alpha: float = 0.5,
    seed: int = 0,
) -> list[DataSource]:
    """Split one data source into ``num_clients`` per-client sources.

    Args:
        source: The full dataset to partition (loaded once, here).
        num_clients: Number of client shards.
        scheme: ``"dirichlet"`` (label-skewed non-IID) or ``"iid"`` (uniform control).
        alpha: Dirichlet concentration for ``scheme="dirichlet"`` (ignored otherwise).
        seed: RNG seed for the partition.

    Returns:
        A list of ``num_clients`` :class:`DataSource` shards, each ready to feed an
        :class:`~cafl4ds.data.streams.EraStream`.

    Raises:
        ValueError: If ``num_clients < 1`` or ``scheme`` is unknown.

    Note:
        Any per-class eval reservations a client stream makes come from its own shard; under
        sharp skew (small ``alpha``) a shard may not afford them.
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1; got {num_clients}.")
    images, labels = source.load()
    if scheme == "dirichlet":
        index_sets = dirichlet_partition(labels, num_clients, alpha, seed)
    elif scheme == "iid":
        index_sets = iid_partition(labels, num_clients, seed)
    else:
        raise ValueError(f"unknown partition scheme {scheme!r}; expected 'dirichlet' or 'iid'.")
    shards: list[DataSource] = [_ShardSource(images[idx], labels[idx], source.num_classes) for idx in index_sets]
    sizes = [int(idx.numel()) for idx in index_sets]
    logger.info(f"partition '{scheme}' (alpha={alpha}): {num_clients} clients, shard sizes {sizes}")
    return shards
