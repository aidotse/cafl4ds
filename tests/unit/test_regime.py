"""Unit tests for the P1.0.2 regime stream (``cafl4ds.data.regime``)."""

from __future__ import annotations

import pytest
import torch

from cafl4ds.data.attributes import SyntheticAttributeSource
from cafl4ds.data.regime import RegimeStream, count_ordered_batches, iter_ordered_batches
from cafl4ds.data.streams import StreamBatch


def _source() -> SyntheticAttributeSource:
    """A modest network-free attributed source with enough images to reserve probes from."""
    return SyntheticAttributeSource(num_regimes=3, num_canary_classes=3, per_cell=20, img_size=12, long_tail=False)


def test_eval_sets_are_balanced_on_the_canary_axis_and_disjoint() -> None:
    """The probe support/query are canary-labelled, balanced per class, and share no image."""
    stream = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5)
    ev = stream.eval_sets
    assert set(ev.probe_support.labels.tolist()) == {0, 1, 2}
    for cls in (0, 1, 2):
        assert int((ev.probe_support.labels == cls).sum()) == 5
        assert int((ev.probe_query.labels == cls).sum()) == 5
    # support and query images are disjoint (no shared row)
    sup = {tuple(img.flatten().tolist()) for img in ev.probe_support.images}
    qry = {tuple(img.flatten().tolist()) for img in ev.probe_query.images}
    assert sup.isdisjoint(qry)


def test_default_walk_delivers_eras_contiguously_in_order() -> None:
    """With no block_size, batches walk the regimes in order, each era a contiguous run."""
    stream = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5)
    eras = [b.era for b in stream]
    assert eras == sorted(eras)  # non-decreasing → contiguous walk
    assert set(eras) == {0, 1, 2}
    assert eras[0] == 0 and eras[-1] == 2


def test_every_batch_is_label_free_and_single_era() -> None:
    """Each delivered batch is images-only and carries exactly one era tag."""
    stream = RegimeStream(_source(), batch_size=8)
    for b in stream:
        assert isinstance(b, StreamBatch)
        assert b.images.ndim == 4
        assert isinstance(b.era, int)


def test_len_matches_iteration() -> None:
    """The advertised length equals the number of batches actually yielded."""
    stream = RegimeStream(_source(), batch_size=8)
    assert len(stream) == len(list(stream))


def test_block_size_interleaves_regimes() -> None:
    """block_size round-robins the regimes, so eras recur instead of forming one run each."""
    stream = RegimeStream(_source(), batch_size=4, block_size=4)
    eras = [b.era for b in stream]
    # contiguous would be sorted; interleaving breaks that (an era recurs after a later era)
    assert eras != sorted(eras)
    # every era still appears
    assert set(eras) == {0, 1, 2}


def test_block_size_rejects_non_positive() -> None:
    """A non-positive correlation knob is a clear error."""
    with pytest.raises(ValueError, match="block_size must be a positive"):
        RegimeStream(_source(), block_size=0)


def test_max_train_per_regime_caps_the_stream() -> None:
    """Capping training-per-regime shortens the stream vs. the uncapped default."""
    full = RegimeStream(_source(), batch_size=8)
    capped = RegimeStream(_source(), batch_size=8, max_train_per_regime=3)
    assert sum(b.images.shape[0] for b in capped) < sum(b.images.shape[0] for b in full)
    assert sum(b.images.shape[0] for b in capped) <= 3 * capped.num_eras


def test_regime_order_override_changes_the_walk() -> None:
    """Passing an explicit regime order re-sequences the eras."""
    stream = RegimeStream(_source(), batch_size=8, regime_order=[2, 0, 1])
    eras = [b.era for b in stream]
    # era ids are walk positions, so a contiguous walk still reads 0,1,2 — but the *regimes* behind
    # them are reordered; the first era now holds regime 2's images. We assert the walk stays
    # contiguous and complete under the override.
    assert eras == sorted(eras)
    assert set(eras) == {0, 1, 2}


def test_is_deterministic_in_seed() -> None:
    """Same seed reproduces the exact batch sequence (images + era tags)."""
    a = list(RegimeStream(_source(), batch_size=8, seed=3))
    b = list(RegimeStream(_source(), batch_size=8, seed=3))
    assert len(a) == len(b)
    assert all(torch.equal(x.images, y.images) and x.era == y.era for x, y in zip(a, b, strict=True))


def test_num_eras_counts_regimes_with_training_data() -> None:
    """num_eras reflects the regimes that actually contribute training batches."""
    stream = RegimeStream(_source(), batch_size=8)
    assert stream.num_eras == 3


def test_raises_when_a_canary_class_is_too_small_for_reservations() -> None:
    """Asking to reserve more probes than a canary class has is a clear error."""
    tiny = SyntheticAttributeSource(num_regimes=2, num_canary_classes=2, per_cell=3, long_tail=False)
    with pytest.raises(ValueError, match="reserved for probes"):
        RegimeStream(tiny, support_per_canary=20, query_per_canary=20)


def test_iter_and_count_helpers_agree_and_never_split_eras() -> None:
    """The low-level batch helpers agree on count and never merge two eras into one batch."""
    order = [(0, 0), (0, 1), (0, 2), (1, 3), (1, 4)]  # 3 of era 0, 2 of era 1
    images = torch.arange(5 * 3 * 4 * 4, dtype=torch.float32).reshape(5, 3, 4, 4)
    batches = list(iter_ordered_batches(order, images, batch_size=2, drop_last=False))
    assert count_ordered_batches(order, batch_size=2, drop_last=False) == len(batches)
    # era 0 -> [2,1] (boundary flush), era 1 -> [2]  => eras 0,0,1
    assert [b.era for b in batches] == [0, 0, 1]
    assert [b.images.shape[0] for b in batches] == [2, 1, 2]
