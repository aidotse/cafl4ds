"""Unit tests for the P1.0.2 attribute sources (``cafl4ds.data.attributes``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from cafl4ds.data.attributes import (
    BDD100KSource,
    SyntheticAttributeSource,
    _index_scenes,
    _rank_regimes,
    as_float,
    resize_images,
)


def test_synthetic_source_has_two_axes_with_expected_shapes() -> None:
    """Images plus aligned integer era-key and canary axes, all with N rows."""
    src = SyntheticAttributeSource(num_regimes=3, num_canary_classes=4, per_cell=8, img_size=12, long_tail=False)
    a = src.load()
    n = a.images.shape[0]
    assert a.images.shape == (n, 3, 12, 12)
    assert a.era_key.shape == (n,) and a.canary.shape == (n,)
    assert a.images.dtype == torch.float32
    assert 0.0 <= float(a.images.min()) and float(a.images.max()) <= 1.0
    assert set(a.era_key.tolist()) == {0, 1, 2}
    assert set(a.canary.tolist()) == {0, 1, 2, 3}
    assert src.num_canary_classes == 4


def test_synthetic_regime_order_is_identity_and_named() -> None:
    """The default walk is 0..R-1 and every regime / canary id has a name."""
    src = SyntheticAttributeSource(num_regimes=4, num_canary_classes=3, per_cell=6)
    a = src.load()
    assert a.regime_order == [0, 1, 2, 3]
    assert set(a.regime_names) == {0, 1, 2, 3}
    assert set(a.canary_names) == {0, 1, 2}


def test_synthetic_long_tail_thins_later_regimes() -> None:
    """Long-tail makes later regimes rarer than earlier ones (mirrors rare driving conditions)."""
    src = SyntheticAttributeSource(num_regimes=4, num_canary_classes=2, per_cell=16, long_tail=True)
    a = src.load()
    counts = [int((a.era_key == r).sum()) for r in range(4)]
    assert counts[0] > counts[-1]
    assert all(earlier >= later for earlier, later in zip(counts, counts[1:], strict=False))


def test_synthetic_canary_axis_is_independent_of_regime() -> None:
    """Each regime carries every canary class — eras are not separable by the probe axis."""
    src = SyntheticAttributeSource(num_regimes=3, num_canary_classes=3, per_cell=4, long_tail=False)
    a = src.load()
    for r in range(3):
        in_regime = a.canary[a.era_key == r]
        assert set(in_regime.tolist()) == {0, 1, 2}


def test_synthetic_is_deterministic_in_seed() -> None:
    """Same seed reproduces the images and both axes exactly."""
    a = SyntheticAttributeSource(seed=7).load()
    b = SyntheticAttributeSource(seed=7).load()
    assert torch.equal(a.images, b.images)
    assert torch.equal(a.era_key, b.era_key) and torch.equal(a.canary, b.canary)


def test_rank_regimes_follows_the_shift_walk() -> None:
    """Regime ids are assigned daytime→night, clear→adverse, so id 0 is the earliest regime."""
    regimes = {("night", "clear"), ("daytime", "clear"), ("daytime", "rainy"), ("dawn/dusk", "clear")}
    order, regime_id, names = _rank_regimes(regimes)
    assert order == [0, 1, 2, 3]
    assert regime_id[("daytime", "clear")] == 0
    assert regime_id[("daytime", "rainy")] == 1
    assert regime_id[("dawn/dusk", "clear")] == 2
    assert regime_id[("night", "clear")] == 3
    assert names[0] == "daytime·clear"


def test_index_scenes_is_sorted_and_contiguous() -> None:
    """Scenes map to 0..S-1 in sorted order (deterministic canary ids)."""
    scene_id, names = _index_scenes({"highway", "city street", "residential"})
    assert scene_id == {"city street": 0, "highway": 1, "residential": 2}
    assert names == {0: "city street", 1: "highway", 2: "residential"}


def test_resize_images_is_a_noop_when_already_sized() -> None:
    """Resize leaves an already-correct-size tensor untouched (identity), else reshapes."""
    imgs = torch.rand(2, 3, 16, 16)
    assert resize_images(imgs, 16) is imgs
    assert resize_images(imgs, 8).shape == (2, 3, 8, 8)


def _write_bdd_fixture(root: Path, records: list[dict[str, object]], split: str = "train") -> None:
    """Write a tiny canonical-layout BDD100K fixture (images + attributes JSON) under ``root``."""
    images_dir = root / "images" / "100k" / split
    images_dir.mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for rec in records:
        name = rec["name"]
        assert isinstance(name, str)
        # only write a file for records that should be found (skip a deliberately-missing one)
        if rec.get("_write", True):
            Image.new("RGB", (32, 24), color=(100, 50, 50)).save(images_dir / name)
    labels = [
        {"name": r["name"], "attributes": r["attributes"], **({"labels": r["labels"]} if "labels" in r else {})}
        for r in records
    ]
    (root / "labels" / f"bdd100k_labels_images_{split}.json").write_text(json.dumps(labels), encoding="utf-8")


def test_synthetic_source_caches_the_decode() -> None:
    """Repeated ``load()`` on one source returns the identical cached object (decode once — A2)."""
    src = SyntheticAttributeSource(num_regimes=2, num_canary_classes=2, per_cell=4, img_size=8, long_tail=False)
    first = src.load()
    assert src.load() is first


def test_bdd_source_caches_the_decode(tmp_path: Path) -> None:
    """The BDD source decodes once and reuses the cache — so threading it through arms is cheap (A2)."""
    records: list[dict[str, object]] = [
        {"name": "a.jpg", "attributes": {"timeofday": "daytime", "weather": "clear", "scene": "highway"}},
        {"name": "b.jpg", "attributes": {"timeofday": "night", "weather": "rainy", "scene": "city street"}},
    ]
    _write_bdd_fixture(tmp_path, records)
    src = BDD100KSource(str(tmp_path), img_size=16)
    first = src.load()
    assert src.load() is first


def test_bdd_source_parses_layout_and_maps_both_axes(tmp_path: Path) -> None:
    """The real canonical layout parses into resized images with regime + scene axes."""
    records: list[dict[str, object]] = [
        {"name": "a.jpg", "attributes": {"timeofday": "daytime", "weather": "clear", "scene": "highway"}},
        {"name": "b.jpg", "attributes": {"timeofday": "night", "weather": "rainy", "scene": "city street"}},
        {"name": "c.jpg", "attributes": {"timeofday": "daytime", "weather": "clear", "scene": "city street"}},
    ]
    _write_bdd_fixture(tmp_path, records)
    a = BDD100KSource(str(tmp_path), img_size=16).load()
    assert a.images.shape == (3, 3, 16, 16)
    # daytime·clear ranks before night·rainy → regime id 0 is the daytime·clear pair
    assert a.regime_order == [0, 1]
    assert a.regime_names[0] == "daytime·clear"
    assert set(a.canary.tolist()) == {0, 1}  # two scenes
    assert a.canary_names[0] == "city street"  # sorted


def test_bdd_source_counts_detection_categories_per_image(tmp_path: Path) -> None:
    """The detection boxes are aggregated into per-image category counts aligned to the images."""
    records: list[dict[str, object]] = [
        {
            "name": "a.jpg",
            "attributes": {"timeofday": "daytime", "weather": "clear", "scene": "highway"},
            "labels": [{"category": "car"}, {"category": "car"}, {"category": "person"}, {"note": "no-category"}],
        },
        {
            "name": "b.jpg",
            "attributes": {"timeofday": "night", "weather": "rainy", "scene": "city street"},
            "labels": [],
        },
    ]
    _write_bdd_fixture(tmp_path, records)
    a = BDD100KSource(str(tmp_path), img_size=16).load()
    assert a.object_categories == [{"car": 2, "person": 1}, {}]


def test_bdd_source_skips_undefined_attributes_and_missing_files(tmp_path: Path) -> None:
    """Records with an undefined attribute, or a missing image file, are dropped."""
    records: list[dict[str, object]] = [
        {"name": "ok.jpg", "attributes": {"timeofday": "daytime", "weather": "clear", "scene": "highway"}},
        {"name": "undef.jpg", "attributes": {"timeofday": "undefined", "weather": "clear", "scene": "highway"}},
        {
            "name": "gone.jpg",
            "attributes": {"timeofday": "night", "weather": "snowy", "scene": "tunnel"},
            "_write": False,
        },
    ]
    _write_bdd_fixture(tmp_path, records)
    a = BDD100KSource(str(tmp_path)).load()
    assert a.images.shape[0] == 1  # only the valid, present record survives


def test_bdd_source_respects_max_images(tmp_path: Path) -> None:
    """The max_images cap bounds how many attribute-valid images are loaded."""
    records: list[dict[str, object]] = [
        {"name": f"{i}.jpg", "attributes": {"timeofday": "daytime", "weather": "clear", "scene": "highway"}}
        for i in range(5)
    ]
    _write_bdd_fixture(tmp_path, records)
    a = BDD100KSource(str(tmp_path), max_images=2).load()
    assert a.images.shape[0] == 2


def test_bdd_source_drops_rare_canary_scenes_below_min_count(tmp_path: Path) -> None:
    """min_canary_count drops scenes with too few images to support a probe; 0 keeps every scene."""
    records: list[dict[str, object]] = []
    records.extend(
        {"name": f"c{i}.jpg", "attributes": {"timeofday": "daytime", "weather": "clear", "scene": "city street"}}
        for i in range(5)
    )
    records.extend(
        {"name": f"t{i}.jpg", "attributes": {"timeofday": "night", "weather": "clear", "scene": "tunnel"}}
        for i in range(2)
    )
    _write_bdd_fixture(tmp_path, records)
    # keep-all: both scenes present
    keep_all = BDD100KSource(str(tmp_path))
    keep_all.load()
    assert keep_all.num_canary_classes == 2
    # threshold 3: the 2-image "tunnel" scene is dropped, its images excluded
    src = BDD100KSource(str(tmp_path), min_canary_count=3)
    filtered = src.load()
    assert src.num_canary_classes == 1
    assert set(filtered.canary_names.values()) == {"city street"}
    assert filtered.images.shape[0] == 5


def test_bdd_source_raises_on_missing_root(tmp_path: Path) -> None:
    """A missing image directory is a clear FileNotFoundError, not an opaque crash."""
    with pytest.raises(FileNotFoundError, match="BDD100K images not found"):
        BDD100KSource(str(tmp_path / "nope")).load()


def test_bdd_source_uint8_frames_match_the_float_decode_bit_for_bit(tmp_path: Path) -> None:
    """``pixel_dtype=uint8`` holds a quarter of the memory; as_float recovers the float32 decode exactly."""
    records: list[dict[str, object]] = [
        {"name": "a.jpg", "attributes": {"timeofday": "daytime", "weather": "clear", "scene": "highway"}},
        {"name": "b.jpg", "attributes": {"timeofday": "night", "weather": "rainy", "scene": "city street"}},
    ]
    _write_bdd_fixture(tmp_path, records)
    reference = BDD100KSource(str(tmp_path), img_size=16).load().images
    compact = BDD100KSource(str(tmp_path), img_size=16, pixel_dtype="uint8").load().images
    assert reference.dtype == torch.float32 and compact.dtype == torch.uint8
    assert torch.equal(as_float(compact), reference)
    legacy = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
    with Image.open(tmp_path / "images" / "100k" / "train" / "a.jpg") as im:
        assert torch.equal(legacy(im.convert("RGB")), reference[0])  # the pre-uint8 decode path
    with pytest.raises(ValueError, match="pixel_dtype"):
        BDD100KSource(str(tmp_path), pixel_dtype="float16")


def test_bdd_source_cache_reuses_an_identical_decode_and_keys_on_the_config(tmp_path: Path) -> None:
    """A cached load equals the fresh decode; a different size is a different cache entry."""
    records: list[dict[str, object]] = [
        {"name": "a.jpg", "attributes": {"timeofday": "daytime", "weather": "clear", "scene": "highway"}},
        {"name": "b.jpg", "attributes": {"timeofday": "night", "weather": "rainy", "scene": "city street"}},
    ]
    _write_bdd_fixture(tmp_path / "bdd", records)
    cache = tmp_path / "cache"
    fresh = BDD100KSource(str(tmp_path / "bdd"), img_size=16, pixel_dtype="uint8", cache_dir=str(cache)).load()
    assert len(list(cache.glob("*.pt"))) == 1
    again = BDD100KSource(str(tmp_path / "bdd"), img_size=16, pixel_dtype="uint8", cache_dir=str(cache))
    cached = again.load()
    assert torch.equal(cached.images, fresh.images) and torch.equal(cached.canary, fresh.canary)
    assert cached.regime_names == fresh.regime_names and again.num_canary_classes == 2
    BDD100KSource(str(tmp_path / "bdd"), img_size=8, pixel_dtype="uint8", cache_dir=str(cache)).load()
    assert len(list(cache.glob("*.pt"))) == 2


def test_bdd_source_scene_whitelist_and_center_crop(tmp_path: Path) -> None:
    """``scenes`` keeps only the listed scenes; ``center_crop`` keeps the aspect ratio (sides dropped)."""
    records: list[dict[str, object]] = [
        {"name": "a.jpg", "attributes": {"timeofday": "daytime", "weather": "clear", "scene": "highway"}},
        {"name": "b.jpg", "attributes": {"timeofday": "night", "weather": "rainy", "scene": "parking lot"}},
    ]
    _write_bdd_fixture(tmp_path, records)
    kept = BDD100KSource(str(tmp_path), img_size=16, scenes=["highway"]).load()
    assert kept.canary.numel() == 1 and kept.canary_names == {0: "highway"}
    # a frame whose left and right thirds differ from its centre: the crop sees only the centre
    frame = Image.new("RGB", (48, 16), color=(0, 0, 0))
    frame.paste((255, 255, 255), (16, 0, 32, 16))
    frame.save(tmp_path / "images" / "100k" / "train" / "a.jpg", quality=100)
    crop = BDD100KSource(str(tmp_path), img_size=16, resize_mode="center_crop", scenes=["highway"]).load().images
    squash = BDD100KSource(str(tmp_path), img_size=16, scenes=["highway"]).load().images
    assert crop.mean() > 0.9 and squash.mean() < 0.6
    with pytest.raises(ValueError, match="resize_mode"):
        BDD100KSource(str(tmp_path), resize_mode="pad")
