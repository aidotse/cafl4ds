"""Data sources: produce ``(images, labels)`` tensors for the streams to order.

A :class:`DataSource` is the *raw material* a stream orders into eras — it decouples the
stream/ordering logic from where the pixels come from. Two sources exist in Phase 0:

* :class:`STL10Source` — the real STL-10 labeled split (Coates et al. 2011), resized tiny for
  CPU. Labels are carried only so the stream can build class-blocked ordering and held-out
  eval sets; they never reach the SSL update.
* :class:`SyntheticSource` — class-structured Gaussian blobs, network-free, for fast unit
  tests and the fastest smoke runs.

Both return images as ``float32`` ``[N, C, H, W]`` in ``[0, 1]`` and integer labels ``[N]``.
Adding BDD100K/ZOD later means adding a new source, not touching the stream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias
from loguru import logger
from torchvision.datasets import STL10


class DataSource(ABC):
    """Produces ``(images, labels)`` tensors for a stream to order into eras."""

    @abstractmethod
    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the full dataset into memory.

        Returns:
            A tuple ``(images, labels)`` with ``images`` of shape ``[N, C, H, W]``
            (``float32`` in ``[0, 1]``) and integer ``labels`` of shape ``[N]``.
        """

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of distinct classes in the source."""


class STL10Source(DataSource):
    """The real STL-10 labeled split, resized to a tiny CPU-friendly size."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 32,
        max_per_class: int | None = None,
    ) -> None:
        """Configure the STL-10 source.

        Args:
            root: Directory holding the downloaded ``stl10_binary`` (torchvision layout).
            split: Which labeled split to load (``"train"`` or ``"test"``).
            img_size: Side length to bilinearly resize the 96px images to.
            max_per_class: If set, keep at most this many images per class (tiny runs).
        """
        self.root = root
        self.split = split
        self.img_size = img_size
        self.max_per_class = max_per_class

    @property
    def num_classes(self) -> int:
        """STL-10 has 10 classes."""
        return 10

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load, resize, and (optionally) per-class-subsample the STL-10 split.

        Returns:
            ``(images, labels)`` with images ``[N, 3, img_size, img_size]`` in ``[0, 1]``.

        Raises:
            FileNotFoundError: If the STL-10 binaries are not present under ``root``.
        """
        if not (Path(self.root) / "stl10_binary").is_dir():
            raise FileNotFoundError(
                f"STL-10 binaries not found under {self.root}. Download once with "
                "torchvision.datasets.STL10(root=..., split=..., download=True)."
            )
        ds = STL10(root=self.root, split=self.split, download=False)
        images = torch.from_numpy(ds.data).float() / 255.0  # [N, 3, 96, 96]
        labels = torch.from_numpy(ds.labels).long()
        if self.max_per_class is not None:
            images, labels = _subsample_per_class(images, labels, self.max_per_class)
        images = F.interpolate(images, size=self.img_size, mode="bilinear", align_corners=False, antialias=True)
        logger.info(f"STL10Source: loaded {images.shape[0]} images ({self.split}) at {self.img_size}px")
        return images, labels


class SyntheticSource(DataSource):
    """Class-structured Gaussian images — network-free, for tests and fast smoke runs."""

    def __init__(
        self,
        num_classes: int = 4,
        per_class: int = 64,
        img_size: int = 16,
        channels: int = 3,
        noise: float = 0.3,
        seed: int = 0,
    ) -> None:
        """Configure the synthetic source.

        Args:
            num_classes: Number of classes to generate.
            per_class: Images per class.
            img_size: Image side length.
            channels: Number of channels.
            noise: Standard deviation of the per-pixel Gaussian noise around each class mean.
            seed: RNG seed for reproducibility.
        """
        self._num_classes = num_classes
        self.per_class = per_class
        self.img_size = img_size
        self.channels = channels
        self.noise = noise
        self.seed = seed

    @property
    def num_classes(self) -> int:
        """Return the configured class count."""
        return self._num_classes

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate class-structured images (a distinct random mean pattern per class).

        Returns:
            ``(images, labels)`` with images ``[N, C, img_size, img_size]`` in ``[0, 1]``,
            where a class's images cluster around its own random pattern (so probes are
            learnable and the effective rank is meaningful).
        """
        g = torch.Generator().manual_seed(self.seed)
        shape = (self.channels, self.img_size, self.img_size)
        images, labels = [], []
        for c in range(self._num_classes):
            mean = torch.rand(shape, generator=g)
            block = mean.unsqueeze(0) + self.noise * torch.randn(self.per_class, *shape, generator=g)
            images.append(block.clamp_(0.0, 1.0))
            labels.append(torch.full((self.per_class,), c, dtype=torch.long))
        return torch.cat(images), torch.cat(labels)


def _subsample_per_class(
    images: torch.Tensor, labels: torch.Tensor, max_per_class: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep at most ``max_per_class`` images of each class (first-occurring, order-preserving).

    Args:
        images: All images ``[N, C, H, W]``.
        labels: All labels ``[N]``.
        max_per_class: Cap on images retained per class.

    Returns:
        The subsampled ``(images, labels)``.
    """
    keep: list[int] = []
    counts: dict[int, int] = {}
    for i, y in enumerate(labels.tolist()):
        if counts.get(y, 0) < max_per_class:
            keep.append(i)
            counts[y] = counts.get(y, 0) + 1
    idx = torch.tensor(keep, dtype=torch.long)
    return images[idx], labels[idx]
