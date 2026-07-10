"""Tests for the era streams and data sources.

The load-bearing guarantees: class-blocked ordering really is blocked by class; the held-out
eval sets are disjoint from training and from each other (so eval never leaks into the
update); and a :class:`~cafl4ds.data.streams.StreamBatch` carries no labels.
"""

import pytest
import torch

from cafl4ds.data.sources import DataSource, SyntheticSource
from cafl4ds.data.streams import EraStream, StreamBatch


class IdSource(DataSource):
    """A source whose images encode their own unique index (for disjointness checks)."""

    def __init__(self, num_classes: int, per_class: int) -> None:
        """Store the class/count layout."""
        self._num_classes = num_classes
        self.per_class = per_class

    @property
    def num_classes(self) -> int:
        """Return the class count."""
        return self._num_classes

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return images that are constant-valued at their global index, plus class labels."""
        n = self._num_classes * self.per_class
        images = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1, 1).expand(n, 3, 4, 4).clone()
        labels = torch.arange(n) // self.per_class
        return images, labels

    @staticmethod
    def ids(images: torch.Tensor) -> set[int]:
        """Recover the set of global indices encoded in a batch of images."""
        return {int(v) for v in images[:, 0, 0, 0].tolist()}


def test_synthetic_source_is_class_structured() -> None:
    """SyntheticSource yields the requested layout with labels in range."""
    imgs, labels = SyntheticSource(num_classes=3, per_class=10, img_size=8).load()
    assert imgs.shape == (30, 3, 8, 8)
    assert set(labels.tolist()) == {0, 1, 2}
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0


def test_class_blocked_ordering_is_blocked_by_era() -> None:
    """Class-blocked batches arrive in contiguous, non-decreasing era order."""
    stream = EraStream(
        IdSource(num_classes=4, per_class=40),
        batch_size=8,
        order="class_blocked",
        support_per_class=5,
        query_per_class=5,
        era_eval_per_class=5,
    )
    eras = [b.era for b in stream]
    assert stream.num_eras == 4
    assert eras == sorted(eras)  # non-decreasing
    assert set(eras) == {0, 1, 2, 3}


def test_batches_carry_no_labels() -> None:
    """A StreamBatch exposes only images (+ era/step) — labels cannot enter the update."""
    stream = EraStream(
        SyntheticSource(num_classes=2, per_class=40, img_size=8),
        batch_size=8,
        support_per_class=5,
        query_per_class=5,
        era_eval_per_class=5,
    )
    batch = next(iter(stream))
    assert isinstance(batch, StreamBatch)
    assert set(vars(batch)) == {"images", "era", "step"}


def test_eval_sets_disjoint_from_training_and_each_other() -> None:
    """Probe support/query, per-era eval, and training images are pairwise disjoint."""
    src = IdSource(num_classes=3, per_class=40)
    stream = EraStream(src, batch_size=8, support_per_class=5, query_per_class=5, era_eval_per_class=5)
    support = IdSource.ids(stream.eval_sets.probe_support.images)
    query = IdSource.ids(stream.eval_sets.probe_query.images)
    era_eval = set().union(*(IdSource.ids(e.images) for e in stream.eval_sets.per_era.values()))
    train = set().union(*(IdSource.ids(b.images) for b in stream))

    assert len(support) == 3 * 5
    assert len(query) == 3 * 5
    assert len(era_eval) == 3 * 5
    assert support.isdisjoint(query)
    assert support.isdisjoint(era_eval)
    assert query.isdisjoint(era_eval)
    assert train.isdisjoint(support | query | era_eval)


def test_iid_order_is_single_era_and_shuffled() -> None:
    """IID ordering collapses to one era and does not preserve class-blocked order."""
    src = IdSource(num_classes=4, per_class=40)
    stream = EraStream(src, batch_size=8, order="iid", support_per_class=5, query_per_class=5, era_eval_per_class=5)
    assert stream.num_eras == 1
    ordered_ids = [i for b in stream for i in IdSource.ids(b.images)]
    assert all(b.era == 0 for b in stream)
    # ids increase with class, so class-blocked order would be sorted; IID must not be.
    assert ordered_ids != sorted(ordered_ids)


def test_max_train_per_class_caps_training_images() -> None:
    """max_train_per_class limits training images per class after the eval reservations."""
    src = IdSource(num_classes=2, per_class=40)
    stream = EraStream(
        src,
        batch_size=4,
        support_per_class=5,
        query_per_class=5,
        era_eval_per_class=5,
        max_train_per_class=7,
    )
    train_ids = set().union(*(IdSource.ids(b.images) for b in stream))
    assert len(train_ids) == 2 * 7


def test_reservation_larger_than_class_raises() -> None:
    """Reserving more per class than available raises a clear error."""
    with pytest.raises(ValueError, match="reserved for eval"):
        EraStream(
            SyntheticSource(num_classes=2, per_class=10, img_size=8),
            support_per_class=5,
            query_per_class=5,
            era_eval_per_class=5,
        )


def test_unknown_order_raises() -> None:
    """An unknown ordering is rejected."""
    with pytest.raises(ValueError, match="unknown order"):
        EraStream(SyntheticSource(num_classes=2, per_class=40, img_size=8), order="bogus")
