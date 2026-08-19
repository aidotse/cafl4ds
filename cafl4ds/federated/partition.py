"""Client partitioning — split one dataset into per-client shards (the FL ``D`` factor).

Federated learning starts by handing each client its *own* slice of the data. This module
turns a single :class:`~cafl4ds.data.sources.DataSource` into ``N`` per-client sources; each
client then wraps its shard in its own :class:`~cafl4ds.data.streams.EraStream`, so a client
sees its *local* correlated stream. The **partition scheme** is where non-IID structure is
injected — the knob behind novelty claim **N-D** (selection × aggregation skew): independent
clients each over-selecting their local tail may skew the aggregate.

Two schemes:

* ``"dirichlet"`` — label-skewed non-IID (Hsu et al. 2019). For each class, a ``Dirichlet(α)``
  draw over clients splits that class's images. Small ``α`` → sharp skew (clients specialize);
  large ``α`` → near-uniform. The headline heterogeneity knob.
* ``"iid"`` — a uniform random split (the control): every client's shard is a fair sample of
  the whole, so any aggregation effect is *not* attributable to data skew.

A partition is over *indices only*; the pixels are sliced into per-client
:class:`_ShardSource` views. Downstream, each client's :class:`EraStream` reserves its own
held-out eval sets from its shard, so under sharp skew the per-class reservations must be sized
to what the smallest shard can afford (see the note in :func:`partition_source`).
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
        Each client's stream later reserves ``support/query/era_eval`` images *per class from
        its own shard*. Under sharp skew (small ``alpha``) a shard may hold too few images of a
        class to satisfy those reservations — size the per-class reservations to the smallest
        shard, or raise ``alpha``.
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
