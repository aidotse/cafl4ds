"""Unit tests for the off-domain control sources (``cafl4ds.data.offdomain``) on synthetic archives."""

from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from cafl4ds.data.offdomain import COCOSource, Places365Source, crop_to_aspect
from cafl4ds.data.regime import RegimeStream


def _jpeg(width: int, height: int, value: int) -> bytes:
    """A solid-colour JPEG with a white centre column (so the crop is visible)."""
    pixels = np.full((height, width, 3), value, dtype=np.uint8)
    pixels[:, width // 2 - 2 : width // 2 + 2] = 255
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _coco(tmp_path: Path) -> tuple[Path, Path]:
    """Ten images: 0-5 people only, 6-7 with a car, 8 unannotated, 9 a person and a bus."""
    images: list[dict[str, Any]] = [{"id": i, "file_name": f"{i:012d}.jpg"} for i in range(10)]
    cats = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}, {"id": 6, "name": "bus"}]
    anns = [{"image_id": i, "category_id": 1} for i in [*range(8), 9]]
    anns += [{"image_id": 6, "category_id": 3}, {"image_id": 7, "category_id": 3}, {"image_id": 9, "category_id": 6}]
    annotations = tmp_path / "ann.zip"
    with zipfile.ZipFile(annotations, "w") as z:
        z.writestr(
            "annotations/instances_train2017.json",
            json.dumps({"images": images, "categories": cats, "annotations": anns}),
        )
    archive = tmp_path / "train2017.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for im in images:
            z.writestr(f"train2017/{im['file_name']}", _jpeg(64, 48, 10 * im["id"]))
    return archive, annotations


def test_crop_to_aspect_centres_the_largest_window() -> None:
    """A wide image loses its sides, a tall one its top and bottom."""
    assert crop_to_aspect(200, 90, 16 / 9) == (20, 0, 180, 90)
    assert crop_to_aspect(160, 160, 16 / 9) == (0, 35, 160, 125)
    assert crop_to_aspect(256, 256, 4 / 3) == (0, 32, 256, 224)


def test_coco_drops_street_and_unannotated_images_and_caches_without_the_archive(tmp_path: Path) -> None:
    """Only the people-only images are eligible; the decode is seed-fixed, square and cached."""
    archive, annotations = _coco(tmp_path)
    kwargs: dict[str, Any] = {
        "archive": str(archive),
        "annotations": str(annotations),
        "n_images": 4,
        "exclude": ["car", "bus"],
        "crop_aspect": 16 / 9,
        "img_size": 16,
        "cache_dir": str(tmp_path / "cache"),
        "workers": 1,
    }
    source = COCOSource(**kwargs)
    assert source.eligible() == [f"train2017/{i:012d}.jpg" for i in range(6)]
    data = source.load()
    assert data.images.shape == (4, 3, 16, 16) and data.images.dtype == torch.uint8
    assert data.canary.tolist() == [0, 1, 0, 1] and data.regime_order == [0]
    centre, edge = data.images[..., 7:9].float().mean(), data.images[..., :3].float().mean()
    assert centre > edge + 50  # the centre column survives the crop + squash
    with pytest.raises(ValueError):
        COCOSource(**{**kwargs, "exclude": ["spaceship"], "cache_dir": None}).eligible()
    archive.unlink()
    annotations.unlink()
    assert torch.equal(COCOSource(**kwargs).load().images, data.images)  # from the cache alone


def test_places_filters_categories_reads_the_tar_and_feeds_a_regime_stream(tmp_path: Path) -> None:
    """Excluded categories (including two-level ones) are dropped; the drawn images stream as one regime."""
    categories = tmp_path / "categories.txt"
    categories.write_text("/a/airfield 0\n/b/bus_station/indoor 1\n/s/street 2\n", encoding="utf-8")
    lines, archive = [], tmp_path / "train.tar"
    with tarfile.open(archive, "w") as tar:
        for cat in ("a/airfield", "b/bus_station/indoor", "s/street"):
            for i in range(20):
                data = _jpeg(32, 32, i)
                info = tarfile.TarInfo(f"data_256/{cat}/{i:08d}.jpg")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                lines.append(f"/{cat}/{i:08d}.jpg 0")
    file_list = tmp_path / "list.txt"
    file_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    source = Places365Source(
        file_list=str(file_list),
        categories=str(categories),
        archive=str(archive),
        n_images=16,
        exclude=["bus_station/indoor", "street"],
        crop_aspect=4 / 3,
        img_size=8,
        workers=1,
    )
    assert all("/a/airfield/" in m for m in source.eligible())
    stream = RegimeStream(source, batch_size=4, support_per_canary=2, query_per_canary=2, order="iid")
    assert len(stream) == 2  # 16 images - 2 x (2 + 2) probes = 8 = two batches
    with pytest.raises(ValueError):
        Places365Source(
            file_list=str(file_list),
            categories=str(categories),
            archive=str(archive),
            n_images=41,
            exclude=["street"],
            crop_aspect=1.0,
        ).selected()
