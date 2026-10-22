"""Off-domain control images — P1.2.4's attribution controls.

P1.2.1 found that JE adapted on BDD beats the frozen encoder. P1.2.4 asks whether the same training on
*non-driving* images would do as well. These sources supply those images in the shape the regime stream
expects, so an off-domain drive runs through the deploy harness exactly like the BDD IID control:

* :class:`COCOSource` — COCO train2017, read straight from ``train2017.zip``, with every image carrying a
  street-content annotation (``exclude``) left out, and images with no annotation at all left out too
  (their content is unverified).
* :class:`Places365Source` — Places365-Standard at 256 px, read from its tar, with the road, traffic and
  BDD-scene categories (``exclude``) left out.

Each draws ``n_images`` at random (``seed``) from the eligible set and applies P1.2.4's geometry rule:
centre-crop to ``crop_aspect`` (width / height), then squash to ``img_size`` square. With COCO's
full-resolution images and ``crop_aspect=16/9`` that is exactly a BDD frame's 16:9 → 1:1 squash; Places'
256 × 256 images are already squashed from (mostly) 4:3 originals, so ``crop_aspect=4/3`` (256 × 192)
gives the same net distortion.

The images carry no labels the drive needs. The regime stream still reserves canary probes, so each
image gets a placeholder canary — two arbitrary halves by position — and a single regime. The in-run
canary readings of an off-domain drive therefore mean nothing; the drive is read afterwards on BDD val.

The decode is cached (uint8, ``cache_dir``) under a key that does not touch the archive, so the cache
still loads after the archive is deleted (P1.2.4 deletes them to free disk).
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from abc import abstractmethod
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from PIL import Image

from cafl4ds.data.attributes import AttributedImages, AttributeSource

_CHUNK = 512  # images per decode task


def crop_to_aspect(width: int, height: int, aspect: float) -> tuple[int, int, int, int]:
    """The centred ``(left, top, right, bottom)`` box of the largest ``aspect`` (width / height) window."""
    if width / height > aspect:
        w = round(height * aspect)
        left = (width - w) // 2
        return left, 0, left + w, height
    h = round(width / aspect)
    top = (height - h) // 2
    return 0, top, width, top + h


def _decode_one(data: bytes, aspect: float, img_size: int) -> np.ndarray:
    """JPEG bytes -> uint8 ``[3, img_size, img_size]``: centre-crop to ``aspect``, then squash."""
    with Image.open(io.BytesIO(data)) as im:
        rgb = im.convert("RGB")
        box = crop_to_aspect(rgb.width, rgb.height, aspect)
        out = rgb.crop(box).resize((img_size, img_size), Image.Resampling.BILINEAR)
    return np.asarray(out, dtype=np.uint8).transpose(2, 0, 1).copy()


def _decode_chunk(blobs: list[bytes], aspect: float, img_size: int) -> np.ndarray:
    return np.stack([_decode_one(b, aspect, img_size) for b in blobs])


class OffDomainSource(AttributeSource):
    """Shared selection, decode and caching for the off-domain sources (one regime, placeholder canary)."""

    name = "offdomain"

    def __init__(
        self,
        archive: str,
        n_images: int,
        exclude: list[str],
        crop_aspect: float,
        img_size: int = 224,
        seed: int = 0,
        pixel_dtype: str = "uint8",
        cache_dir: str | None = None,
        workers: int = 8,
    ) -> None:
        """Configure the source.

        Args:
            archive: The image archive (COCO's ``train2017.zip``, Places' ``train_256_places365standard.tar``).
            n_images: Images drawn at random from the eligible set.
            exclude: The street / road content left out (COCO category names; Places category paths).
            crop_aspect: The centre-crop aspect (width / height) applied before the square squash.
            img_size: Side length of the square output.
            seed: The draw's seed.
            pixel_dtype: ``"uint8"`` (as the BDD caches) or ``"float32"`` in ``[0, 1]``.
            cache_dir: Save / reuse the decode here (``None`` = off).
            workers: Decode processes.

        Raises:
            ValueError: If ``pixel_dtype`` is unknown or ``n_images`` is not positive.
        """
        if pixel_dtype not in ("float32", "uint8"):
            raise ValueError(f"pixel_dtype must be 'float32' or 'uint8'; got {pixel_dtype!r}.")
        if n_images <= 0:
            raise ValueError(f"n_images must be positive; got {n_images}.")
        self.archive = Path(archive)
        self.n_images = int(n_images)
        self.exclude = sorted(str(e) for e in exclude)
        self.crop_aspect = float(crop_aspect)
        self.img_size = int(img_size)
        self.seed = int(seed)
        self.pixel_dtype = pixel_dtype
        self.cache_dir = cache_dir
        self.workers = int(workers)
        self._cache: AttributedImages | None = None

    @property
    def num_canary_classes(self) -> int:
        """The placeholder canary's two halves."""
        return 2

    @abstractmethod
    def eligible(self) -> list[str]:
        """Archive member names of every image that passes the content filter, sorted."""

    @abstractmethod
    def read(self, members: list[str]) -> Iterator[tuple[str, bytes]]:
        """Yield ``(member, bytes)`` for exactly ``members``, in any order."""

    def selected(self) -> list[str]:
        """The ``n_images`` members drawn (``seed``) from :meth:`eligible`, in sorted order."""
        pool = self.eligible()
        if len(pool) < self.n_images:
            raise ValueError(f"{self.name}: only {len(pool)} eligible images for n_images={self.n_images}.")
        draw = np.random.default_rng(self.seed).choice(len(pool), size=self.n_images, replace=False)
        return [pool[i] for i in sorted(draw)]

    def _cache_path(self) -> Path | None:
        if not self.cache_dir:
            return None
        key = json.dumps(
            [self.name, self.archive.name, self.n_images, self.exclude, self.crop_aspect, self.img_size, self.seed]
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return Path(self.cache_dir) / f"{self.name}_{self.img_size}px_{self.pixel_dtype}_{digest}.pt"

    def load(self) -> AttributedImages:
        """The drawn images (cached on the instance and, with ``cache_dir``, on disk)."""
        if self._cache is None:
            path = self._cache_path()
            if path is not None and path.is_file():
                self._cache = torch.load(path, weights_only=False)
                logger.info(f"{self.name}: loaded {self.n_images} cached images from {path}")
            else:
                self._cache = self._decode()
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partial = path.with_suffix(".partial")
                    torch.save(self._cache, partial)
                    partial.replace(path)
                    logger.info(f"{self.name}: cached the decode at {path}")
        return self._cache

    def _decode(self) -> AttributedImages:
        members = self.selected()
        position = {m: i for i, m in enumerate(members)}
        frames = torch.empty((len(members), 3, self.img_size, self.img_size), dtype=torch.uint8)
        names: list[str] = []
        blobs: list[bytes] = []
        with ProcessPoolExecutor(self.workers) as pool:
            pending = []
            for name, data in self.read(members):
                names.append(name)
                blobs.append(data)
                if len(blobs) == _CHUNK:
                    pending.append((names, pool.submit(_decode_chunk, blobs, self.crop_aspect, self.img_size)))
                    names, blobs = [], []
            if blobs:
                pending.append((names, pool.submit(_decode_chunk, blobs, self.crop_aspect, self.img_size)))
            seen = 0
            for chunk_names, future in pending:
                decoded = torch.from_numpy(future.result())
                for name, frame in zip(chunk_names, decoded, strict=True):
                    frames[position[name]] = frame
                seen += len(chunk_names)
        if seen != len(members):
            raise ValueError(f"{self.name}: read {seen} of {len(members)} selected images from {self.archive}.")
        logger.info(f"{self.name}: decoded {len(members)} images at {self.img_size}px from {self.archive}")
        n = len(members)
        return AttributedImages(
            images=frames if self.pixel_dtype == "uint8" else frames.float().div_(255),
            era_key=torch.zeros(n, dtype=torch.long),
            canary=torch.arange(n, dtype=torch.long) % 2,
            regime_order=[0],
            regime_names={0: self.name},
            canary_names={0: "placeholder_a", 1: "placeholder_b"},
        )


class COCOSource(OffDomainSource):
    """COCO train2017 without street content (see the module docstring)."""

    name = "coco"

    def __init__(self, annotations: str, **kwargs: object) -> None:
        """``annotations``: ``annotations_trainval2017.zip`` (or the extracted ``instances_train2017.json``)."""
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.annotations = Path(annotations)

    def _instances(self) -> Any:  # noqa: ANN401 — COCO's parsed JSON
        if self.annotations.suffix == ".zip":
            with zipfile.ZipFile(self.annotations) as z:
                return json.loads(z.read("annotations/instances_train2017.json"))
        return json.loads(self.annotations.read_text(encoding="utf-8"))

    def eligible(self) -> list[str]:
        """Images with at least one annotation and none in an excluded category."""
        data = self._instances()
        names = {c["id"]: c["name"] for c in data["categories"]}
        unknown = set(self.exclude) - set(names.values())
        if unknown:
            raise ValueError(f"coco: unknown excluded categories {sorted(unknown)}.")
        annotated, street = set(), set()
        for ann in data["annotations"]:
            annotated.add(ann["image_id"])
            if names[ann["category_id"]] in self.exclude:
                street.add(ann["image_id"])
        keep = annotated - street
        return sorted(f"train2017/{im['file_name']}" for im in data["images"] if im["id"] in keep)

    def read(self, members: list[str]) -> Iterator[tuple[str, bytes]]:
        """Random access into the zip."""
        with zipfile.ZipFile(self.archive) as z:
            for name in members:
                yield name, z.read(name)


class Places365Source(OffDomainSource):
    """Places365-Standard (256 px) without road, traffic and BDD-scene categories (see the module docstring)."""

    name = "places365"

    def __init__(self, file_list: str, categories: str, **kwargs: object) -> None:
        """``file_list``: ``places365_train_standard.txt``; ``categories``: ``categories_places365.txt``."""
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.file_list = Path(file_list)
        self.categories = Path(categories)

    def eligible(self) -> list[str]:
        """Images whose category (e.g. ``bus_station/indoor``) is not excluded."""
        known = {line.split()[0][3:] for line in self.categories.read_text(encoding="utf-8").splitlines() if line}
        unknown = set(self.exclude) - known
        if unknown:
            raise ValueError(f"places365: unknown excluded categories {sorted(unknown)}.")
        keep = []
        for line in self.file_list.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            path = line.split()[0]  # /a/airfield/00000001.jpg
            if path[3:].rsplit("/", 1)[0] not in self.exclude:
                keep.append(f"data_256{path}")
        return sorted(keep)

    def read(self, members: list[str]) -> Iterator[tuple[str, bytes]]:
        """One sequential pass over the tar, yielding the selected members."""
        wanted = set(members)
        with tarfile.open(self.archive, mode="r|") as tar:
            for info in tar:
                if info.name in wanted:
                    handle = tar.extractfile(info)
                    if handle is not None:
                        yield info.name, handle.read()
