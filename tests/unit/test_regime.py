"""Unit tests for the P1.0.2 regime stream (``cafl4ds.data.regime``)."""

from __future__ import annotations

import pytest
import torch

from cafl4ds.data.attributes import AttributedImages, AttributeSource, SyntheticAttributeSource
from cafl4ds.data.regime import RegimeStream, count_ordered_batches, iter_ordered_batches
from cafl4ds.data.streams import StreamBatch


def _source() -> SyntheticAttributeSource:
    """A modest network-free attributed source with enough images to reserve probes from."""
    return SyntheticAttributeSource(num_regimes=3, num_canary_classes=3, per_cell=20, img_size=12, long_tail=False)


class _CategorySource(AttributeSource):
    """A tiny attributed source carrying per-image object-category counts (a BDD label stand-in)."""

    def __init__(self, per_cell: int = 8) -> None:
        self.per_cell = per_cell

    @property
    def num_canary_classes(self) -> int:
        return 2

    def load(self) -> AttributedImages:
        images, era_key, canary, cats = [], [], [], []
        for regime in (0, 1):
            for scene in (0, 1):
                for _ in range(self.per_cell):
                    images.append(torch.rand(3, 8, 8))
                    era_key.append(regime)
                    canary.append(scene)
                    cats.append({"car": 1} if scene == 0 else {"person": 2})
        return AttributedImages(
            images=torch.stack(images),
            era_key=torch.tensor(era_key, dtype=torch.long),
            canary=torch.tensor(canary, dtype=torch.long),
            regime_order=[0, 1],
            regime_names={0: "daytime·clear", 1: "night·rainy"},
            canary_names={0: "highway", 1: "city street"},
            object_categories=cats,
        )


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


def _probe_images(stream: RegimeStream) -> tuple[set[tuple[float, ...]], set[tuple[float, ...]]]:
    """The stream's held-out (support, query) probe sets as hashable image contents."""
    ev = stream.eval_sets
    sup = {tuple(img.flatten().tolist()) for img in ev.probe_support.images}
    qry = {tuple(img.flatten().tolist()) for img in ev.probe_query.images}
    return sup, qry


def test_reservation_is_fixed_across_drive_seeds() -> None:
    """Different drive seeds reserve the *identical* probe set — the reservation is decoupled from seed.

    The A1 fix: the held-out canary probe set is drawn from ``canary_seed`` (fixed), not the drive
    ``seed``, so every drive in a seed ensemble (and the warm well) holds out the same images and no
    drive ever trains on another drive's probe queries.
    """
    a = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5, seed=0)
    b = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5, seed=7)
    assert _probe_images(a) == _probe_images(b)  # identical held-out probe set across drive seeds
    # ... while the within-regime training order genuinely varies with the drive seed
    walk_a = [tuple(img.flatten().tolist()) for batch in a for img in batch.images]
    walk_b = [tuple(img.flatten().tolist()) for batch in b for img in batch.images]
    assert set(walk_a) == set(walk_b)  # same training pool (identical reservation removed the same rows)
    assert walk_a != walk_b  # but a different frame order


def test_canary_seed_redraws_the_reservation() -> None:
    """Changing ``canary_seed`` (independently of the drive seed) re-draws the held-out probe set."""
    a = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5, canary_seed=0)
    b = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5, canary_seed=1)
    assert _probe_images(a) != _probe_images(b)


def test_num_eras_counts_regimes_with_training_data() -> None:
    """num_eras reflects the regimes that actually contribute training batches."""
    stream = RegimeStream(_source(), batch_size=8)
    assert stream.num_eras == 3


def test_raises_when_a_canary_class_is_too_small_for_reservations() -> None:
    """Asking to reserve more probes than a canary class has is a clear error."""
    tiny = SyntheticAttributeSource(num_regimes=2, num_canary_classes=2, per_cell=3, long_tail=False)
    with pytest.raises(ValueError, match="reserved for probes"):
        RegimeStream(tiny, support_per_canary=20, query_per_canary=20)


def test_era_names_map_walk_positions_to_regime_names() -> None:
    """era_names resolves each delivered era (walk position) to its source regime name."""
    stream = RegimeStream(_source(), batch_size=8)
    assert stream.era_names == {0: "regime0", 1: "regime1", 2: "regime2"}
    # the map only covers eras that actually deliver batches
    assert set(stream.era_names) == {b.era for b in stream}


def test_era_composition_scene_histogram_sums_to_image_count() -> None:
    """Each era's scene histogram totals its delivered image count; no category labels → empty hist."""
    stream = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5)
    comp = stream.era_composition()
    assert set(comp) == {b.era for b in stream}
    for entry in comp.values():
        assert sum(entry["scene_hist"].values()) == entry["n_images"]
        assert entry["category_hist"] == {}  # synthetic source carries no detection labels


def test_era_composition_aggregates_object_categories() -> None:
    """With detection labels, each era's category histogram sums the per-image box counts."""
    stream = RegimeStream(_CategorySource(per_cell=8), batch_size=4, support_per_canary=2, query_per_canary=2)
    comp = stream.era_composition()
    # each era has both scenes; per training image: highway→{car:1}, city street→{person:2}
    for era, entry in comp.items():
        highway = entry["scene_hist"].get("highway", 0)
        city = entry["scene_hist"].get("city street", 0)
        assert entry["category_hist"] == {"car": highway, "person": 2 * city}, era


def test_stationary_batches_cover_the_training_pool_without_probes() -> None:
    """The stationary diet reshuffles the exact training pool — same images, no era ordering."""
    stream = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5)
    walk = list(stream)
    stationary = list(stream.stationary_batches())
    # every stationary batch is tagged era 0 (no regime ordering) ...
    assert {b.era for b in stationary} == {0}
    # ... and the image content matches the walk's training pool exactly (a permutation of it)
    walk_imgs = {tuple(img.flatten().tolist()) for b in walk for img in b.images}
    stat_imgs = {tuple(img.flatten().tolist()) for b in stationary for img in b.images}
    assert walk_imgs == stat_imgs


def test_iter_and_count_helpers_agree_and_never_split_eras() -> None:
    """The low-level batch helpers agree on count and never merge two eras into one batch."""
    order = [(0, 0), (0, 1), (0, 2), (1, 3), (1, 4)]  # 3 of era 0, 2 of era 1
    images = torch.arange(5 * 3 * 4 * 4, dtype=torch.float32).reshape(5, 3, 4, 4)
    batches = list(iter_ordered_batches(order, images, batch_size=2, drop_last=False))
    assert count_ordered_batches(order, batch_size=2, drop_last=False) == len(batches)
    # era 0 -> [2,1] (boundary flush), era 1 -> [2]  => eras 0,0,1
    assert [b.era for b in batches] == [0, 0, 1]
    assert [b.images.shape[0] for b in batches] == [2, 1, 2]


def _contents(batches: list[StreamBatch]) -> list[tuple[float, ...]]:
    """Every delivered image as hashable content, in delivery order."""
    return [tuple(img.flatten().tolist()) for b in batches for img in b.images]


def test_iid_order_is_the_walk_shuffled_with_matched_batch_sizes() -> None:
    """The IID control delivers the walk's exact images, at the walk's per-step batch sizes, reshuffled.

    Matching the step count and every batch size one-for-one is what makes a walk-vs-IID comparison
    at a given step attributable to the ordering alone (not to a shorter run or a smaller batch).
    """
    walk = list(RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5, seed=2))
    iid_stream = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5, seed=2, order="iid")
    iid = list(iid_stream)
    assert len(iid) == len(walk) == len(iid_stream)
    assert [b.images.shape[0] for b in iid] == [b.images.shape[0] for b in walk]
    assert [b.step for b in iid] == list(range(len(iid)))
    assert {b.era for b in iid} == {0}
    assert sorted(_contents(iid)) == sorted(_contents(walk))  # the same images ...
    assert _contents(iid) != _contents(walk)  # ... in a different order
    assert iid_stream.num_eras == 1
    assert iid_stream.era_names == {0: "iid"}


def test_iid_order_mixes_regimes_within_batches() -> None:
    """The IID shuffle crosses regime boundaries — some batch holds images from more than one regime."""
    src = _source()
    loaded = src.load()
    regime_of = {tuple(img.flatten().tolist()): int(r) for img, r in zip(loaded.images, loaded.era_key, strict=True)}
    iid = list(RegimeStream(src, batch_size=8, order="iid"))
    assert any(len({regime_of[tuple(img.flatten().tolist())] for img in b.images}) > 1 for b in iid)


def test_iid_order_is_deterministic_in_seed_and_varies_across_seeds() -> None:
    """The IID shuffle reproduces at fixed seed and is redrawn by a different seed."""
    a = _contents(list(RegimeStream(_source(), batch_size=8, order="iid", seed=1)))
    b = _contents(list(RegimeStream(_source(), batch_size=8, order="iid", seed=1)))
    c = _contents(list(RegimeStream(_source(), batch_size=8, order="iid", seed=2)))
    assert a == b
    assert a != c


def test_leg_boundary_steps_mark_walk_era_changes_in_both_orders() -> None:
    """Leg boundaries are the walk's era-change steps, and the IID control reports the same steps."""
    walk_stream = RegimeStream(_source(), batch_size=8)
    eras = [b.era for b in walk_stream]
    expected = [s for s in range(1, len(eras)) if eras[s] != eras[s - 1]]
    assert walk_stream.leg_boundary_steps == expected
    assert len(expected) == walk_stream.num_eras - 1
    assert RegimeStream(_source(), batch_size=8, order="iid").leg_boundary_steps == expected


def test_rejects_an_unknown_order() -> None:
    """``order`` must be walk or iid."""
    with pytest.raises(ValueError, match="order"):
        RegimeStream(_source(), order="sorted")


def test_stratified_canary_balances_every_stratum_scene_cell() -> None:
    """With ``stratify_canary`` each (stratum, canary) cell reserves the per-cell counts, tagged by stratum."""
    stream = RegimeStream(_source(), batch_size=8, support_per_canary=3, query_per_canary=4, stratify_canary=True)
    ev = stream.eval_sets
    assert ev.group_names == {0: "early", 1: "late"}
    for split, per_cell in ((ev.probe_support, 3), (ev.probe_query, 4)):
        assert split.groups is not None
        cells: dict[tuple[int, int], int] = {}
        for g, y in zip(split.groups.tolist(), split.labels.tolist(), strict=True):
            cells[(g, y)] = cells.get((g, y), 0) + 1
        assert cells == {(g, y): per_cell for g in (0, 1) for y in range(3)}
    # the probes stay disjoint from the training pool
    sup, qry = _probe_images(stream)
    train = set(_contents(list(stream)))
    assert not (sup | qry) & train


def test_stratified_canary_tags_probes_with_their_true_stratum() -> None:
    """Every reserved probe image carries the stratum of the regime it was drawn from."""
    src = _source()
    loaded = src.load()
    assert loaded.stratum is not None
    stratum_of = {tuple(img.flatten().tolist()): int(s) for img, s in zip(loaded.images, loaded.stratum, strict=True)}
    ev = RegimeStream(src, support_per_canary=3, query_per_canary=4, stratify_canary=True).eval_sets
    assert ev.probe_query.groups is not None
    for img, g in zip(ev.probe_query.images, ev.probe_query.groups.tolist(), strict=True):
        assert stratum_of[tuple(img.flatten().tolist())] == g


def test_unstratified_probes_carry_no_groups() -> None:
    """The default reservation is ungrouped (the P1.0.2 behaviour)."""
    ev = RegimeStream(_source(), support_per_canary=3, query_per_canary=3).eval_sets
    assert ev.probe_query.groups is None
    assert ev.group_names == {}


def test_stratify_canary_needs_a_source_with_strata() -> None:
    """Stratifying on a source that defines no strata is a clear error."""
    with pytest.raises(ValueError, match="strata"):
        RegimeStream(_CategorySource(), support_per_canary=2, query_per_canary=2, stratify_canary=True)


def test_well_fraction_carves_a_disjoint_fixed_well_pool() -> None:
    """``well_fraction`` gives the well its own slice: disjoint from drive and probes, fixed across seeds."""
    a = RegimeStream(_source(), batch_size=8, support_per_canary=3, query_per_canary=3, well_fraction=0.5, seed=0)
    b = RegimeStream(_source(), batch_size=8, support_per_canary=3, query_per_canary=3, well_fraction=0.5, seed=5)
    well_a = set(_contents(list(a.stationary_batches())))
    well_b = set(_contents(list(b.stationary_batches())))
    drive = set(_contents(list(a)))
    sup, qry = _probe_images(a)
    assert well_a == well_b  # the well's slice does not depend on the drive seed
    assert not well_a & drive
    assert not well_a & (sup | qry)
    full = set(_contents(list(RegimeStream(_source(), batch_size=8, support_per_canary=3, query_per_canary=3))))
    assert well_a | drive == full  # well + drive partition the training pool
    assert abs(len(well_a) - len(drive)) <= a.num_eras  # ~half each (per-regime rounding)


def test_well_fraction_keeps_every_regime_in_the_drive() -> None:
    """The well slice is taken per regime, so the drive still walks every regime."""
    stream = RegimeStream(_source(), batch_size=8, well_fraction=0.5)
    assert stream.num_eras == 3


@pytest.mark.parametrize("fraction", [-0.1, 1.0])
def test_well_fraction_rejects_out_of_range(fraction: float) -> None:
    """``well_fraction`` must lie in [0, 1)."""
    with pytest.raises(ValueError, match="well_fraction"):
        RegimeStream(_source(), well_fraction=fraction)


def _overlap_stream(overlap: float, seed: int = 0) -> RegimeStream:
    """A well/drive split at ``well_fraction=0.5`` with the given familiarity overlap."""
    return RegimeStream(
        _source(),
        batch_size=8,
        support_per_canary=3,
        query_per_canary=3,
        well_fraction=0.5,
        well_overlap=overlap,
        seed=seed,
    )


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_well_overlap_sets_the_familiar_share_at_a_fixed_well_size(overlap: float) -> None:
    """The well pool keeps its size while ``overlap`` of it is drawn from the drive's frames."""
    base = _overlap_stream(0.0)
    stream = _overlap_stream(overlap)
    well = _contents(list(stream.stationary_batches()))
    drive = set(_contents(list(stream)))
    assert len(well) == len(_contents(list(base.stationary_batches())))  # fixed well size
    familiar = sum(img in drive for img in well) / len(well)
    assert abs(familiar - overlap) <= 0.1  # per-regime rounding on a small pool
    sup, qry = _probe_images(stream)
    assert not set(well) & (sup | qry)  # the probes stay out of the well at every level


def test_well_overlap_leaves_the_drive_identical() -> None:
    """The drive — images, order and batching — is the same at every overlap level."""
    reference = list(_overlap_stream(0.0, seed=3))
    for overlap in (0.25, 0.5, 1.0):
        drive = list(_overlap_stream(overlap, seed=3))
        assert len(drive) == len(reference)
        assert all(torch.equal(a.images, b.images) and a.era == b.era for a, b in zip(reference, drive, strict=True))


def test_well_overlap_well_is_fixed_across_drive_seeds() -> None:
    """The overlapped well pool is drawn from ``canary_seed``, so every drive seed shares it."""
    a = set(_contents(list(_overlap_stream(0.5, seed=0).stationary_batches())))
    b = set(_contents(list(_overlap_stream(0.5, seed=4).stationary_batches())))
    assert a == b


def test_well_overlap_levels_are_nested() -> None:
    """A higher overlap swaps in a superset of the lower level's drive frames (one shared draw order)."""
    drive = set(_contents(list(_overlap_stream(0.0))))
    low = set(_contents(list(_overlap_stream(0.25).stationary_batches()))) & drive
    high = set(_contents(list(_overlap_stream(0.75).stationary_batches()))) & drive
    assert low <= high


def test_well_overlap_needs_a_well_fraction() -> None:
    """Overlap without a reserved slice is rejected."""
    with pytest.raises(ValueError, match="well_fraction"):
        RegimeStream(_source(), well_overlap=0.5)


@pytest.mark.parametrize("overlap", [-0.1, 1.1])
def test_well_overlap_rejects_out_of_range(overlap: float) -> None:
    """``well_overlap`` must lie in [0, 1]."""
    with pytest.raises(ValueError, match="well_overlap"):
        RegimeStream(_source(), well_fraction=0.5, well_overlap=overlap)


@pytest.mark.parametrize("overlap", [0.0, 0.5, 1.0])
def test_familiarity_reports_the_realized_shared_shares(overlap: float) -> None:
    """``familiarity`` measures the shared frames from both sides; it tracks the nominal overlap."""
    stream = _overlap_stream(overlap)
    well = _contents(list(stream.stationary_batches()))
    drive = _contents(list(stream))
    shared = len(set(well) & set(drive))
    familiarity = stream.familiarity
    assert familiarity is not None
    assert familiarity["well_side"] == pytest.approx(shared / len(well))
    assert familiarity["drive_side"] == pytest.approx(shared / len(drive))
    assert abs(familiarity["well_side"] - overlap) <= 0.1


def test_familiarity_is_undefined_without_a_well_slice() -> None:
    """With no carved well (``well_fraction=0``) there is no well/drive split to measure."""
    assert RegimeStream(_source(), batch_size=8).familiarity is None


def test_split_fingerprint_is_shared_across_drive_seeds_and_orders() -> None:
    """The fingerprint describes the reservation only, so the seed and the order leave it unchanged."""
    reference = _overlap_stream(0.5, seed=0).split_fingerprint()
    assert _overlap_stream(0.5, seed=4).split_fingerprint() == reference
    iid = RegimeStream(
        _source(),
        batch_size=8,
        support_per_canary=3,
        query_per_canary=3,
        well_fraction=0.5,
        well_overlap=0.5,
        order="iid",
    )
    assert iid.split_fingerprint() == reference


def test_split_fingerprint_changes_with_the_reservation() -> None:
    """A different familiarity level or probe draw gives a different digest."""
    reference = _overlap_stream(0.5).split_fingerprint()["digest"]
    assert _overlap_stream(0.25).split_fingerprint()["digest"] != reference
    redrawn = RegimeStream(
        _source(),
        batch_size=8,
        support_per_canary=3,
        query_per_canary=3,
        well_fraction=0.5,
        well_overlap=0.5,
        canary_seed=1,
    )
    assert redrawn.split_fingerprint()["digest"] != reference


def _select_stream(seed: int = 0, overlap: float = 0.5) -> RegimeStream:
    """A well/drive split with stratified probes and a selection set."""
    return RegimeStream(
        _source(),
        batch_size=8,
        support_per_canary=3,
        query_per_canary=3,
        well_fraction=0.5,
        well_overlap=overlap,
        select_support_per_canary=2,
        select_query_per_canary=2,
        seed=seed,
    )


def test_selection_set_is_balanced_and_disjoint_from_probes_well_and_drive() -> None:
    """The selection set holds its per-scene counts and shares no image with the probes, well or drive."""
    stream = _select_stream(overlap=1.0)  # a fully familiar well spans the most drive frames
    sets = stream.selection_sets
    assert sets is not None
    for ev, n in ((sets.probe_support, 2), (sets.probe_query, 2)):
        assert torch.bincount(ev.labels).tolist() == [n] * 3
    selection = {tuple(img.flatten().tolist()) for ev in (sets.probe_support, sets.probe_query) for img in ev.images}
    sup, qry = _probe_images(stream)
    well = set(_contents(list(stream.stationary_batches())))
    drive = set(_contents(list(stream)))
    assert not selection & (sup | qry | well | drive)


def test_selection_set_is_fixed_across_drive_seeds() -> None:
    """Drawn from ``canary_seed``, the selection set is the same for every drive seed."""
    a, b = _select_stream(seed=0).selection_sets, _select_stream(seed=5).selection_sets
    assert a is not None and b is not None
    assert torch.equal(a.probe_query.images, b.probe_query.images)


def test_no_selection_set_leaves_every_reservation_unchanged() -> None:
    """Off by default, and turning it off reproduces the stream bit for bit (no extra generator draw)."""
    off = _overlap_stream(0.5)
    assert off.selection_sets is None
    reference = RegimeStream(
        _source(), batch_size=8, support_per_canary=3, query_per_canary=3, well_fraction=0.5, well_overlap=0.5
    )
    assert off.split_fingerprint() == reference.split_fingerprint()


def test_selection_set_enters_the_split_fingerprint() -> None:
    """A well warmed with a selection set and a drive without one reserve differently."""
    assert _select_stream().split_fingerprint()["digest"] != _overlap_stream(0.5).split_fingerprint()["digest"]


def test_selection_set_needs_both_support_and_query() -> None:
    """A half-specified selection set is rejected."""
    with pytest.raises(ValueError, match="selection set"):
        RegimeStream(_source(), select_support_per_canary=2)


def test_labelled_pool_returns_the_well_and_drive_frames_with_their_labels() -> None:
    """The labelled pools hold exactly the well's and the drive's frames, labelled on the canary axis."""
    stream = _select_stream(overlap=0.0)
    well, drive = stream.labelled_pool("well"), stream.labelled_pool("drive")
    assert {tuple(img.flatten().tolist()) for img in well.images} == set(_contents(list(stream.stationary_batches())))
    assert {tuple(img.flatten().tolist()) for img in drive.images} == set(_contents(list(stream)))
    assert well.labels.shape[0] == well.images.shape[0] and set(drive.labels.tolist()) <= {0, 1, 2}
    with pytest.raises(ValueError, match="which"):
        stream.labelled_pool("probe")


@pytest.mark.parametrize("overlap", [0.0, 0.5, 1.0])
def test_a_drive_disjoint_well_pool_never_shares_a_drive_frame(overlap: float) -> None:
    """Above overlap 0 the raw well pool shares drive frames; the disjoint pool drops exactly those."""
    stream = _select_stream(overlap=overlap)
    raw = {tuple(img.flatten().tolist()) for img in stream.labelled_pool("well").images}
    disjoint = {tuple(img.flatten().tolist()) for img in stream.labelled_pool("well", drive_disjoint=True).images}
    drive = {tuple(img.flatten().tolist()) for img in stream.labelled_pool("drive").images}
    assert disjoint == raw - drive
    if overlap == 0.0:
        assert disjoint == raw
    if overlap == 1.0:
        assert not disjoint
