"""The SSL-method interface (the ``C`` flag) and shared init/checkpoint utilities.

An :class:`SSLMethod` bundles the shared encoder with a self-supervised objective. The
streaming loop only ever calls two things on it:

* :meth:`SSLMethod.training_step` — compute a self-supervised loss on a batch of **raw
  images** (no labels ever enter here); the loop backprops and steps the optimizer.
* :meth:`SSLMethod.encode` — map images to the pooled backbone embedding the health
  instruments and probes read (no gradient).

The ``C`` factor of the experiment matrix selects the concrete method (:mod:`cafl4ds.ssl.mae`
or :mod:`cafl4ds.ssl.simsiam`); the ``I`` factor selects the encoder initialization, applied
here via :func:`apply_encoder_init` (``from_scratch`` leaves the random init in place;
``pretrained`` loads a checkpoint produced by an *IID* pre-pass, so stream correlation never
contaminates the starting point).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from loguru import logger
from torch import nn

from cafl4ds.models.vit import TinyViTEncoder


class SSLMethod(nn.Module, ABC):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """A self-supervised method: a shared encoder plus a training objective."""

    def __init__(self, encoder: TinyViTEncoder) -> None:
        """Store the shared backbone encoder.

        Args:
            encoder: The :class:`~cafl4ds.models.vit.TinyViTEncoder` backbone under study.
        """
        super().__init__()
        self.encoder = encoder

    @property
    @abstractmethod
    def name(self) -> str:
        """Short method identifier (e.g. ``"mae"``, ``"simsiam"``) for logging."""

    @abstractmethod
    def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
        """Compute the self-supervised loss on a batch of raw images.

        Args:
            imgs: A batch of images ``[B, C, H, W]``. Labels never enter this path.

        Returns:
            A scalar loss tensor with gradient, ready for ``.backward()``.
        """

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return the pooled backbone embedding used by the instruments/probes.

        Args:
            imgs: A batch of images ``[B, C, H, W]``.

        Returns:
            The pooled embedding ``[B, embed_dim]`` (no gradient tracking).
        """
        with torch.no_grad():
            return self.encoder.embed(imgs)


def load_encoder_checkpoint(encoder: TinyViTEncoder, checkpoint: str | Path) -> None:
    """Load encoder weights from a checkpoint saved by :func:`save_encoder_checkpoint`.

    Args:
        encoder: The encoder to load weights into (in place).
        checkpoint: Path to a ``state_dict`` file for the encoder.

    Raises:
        FileNotFoundError: If ``checkpoint`` does not exist.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"pretrained checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    encoder.load_state_dict(state)
    logger.info(f"loaded pretrained encoder weights from {path}")


def save_encoder_checkpoint(encoder: TinyViTEncoder, checkpoint: str | Path) -> None:
    """Save an encoder's ``state_dict`` (produced by the IID pre-pass).

    Args:
        encoder: The encoder whose weights to save.
        checkpoint: Destination path (parent directories are created).
    """
    path = Path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), path)
    logger.info(f"saved pretrained encoder weights to {path}")


def apply_encoder_init(
    encoder: TinyViTEncoder, mode: str = "from_scratch", checkpoint: str | Path | None = None
) -> None:
    """Apply the ``I`` (initialization) factor to the encoder.

    Args:
        encoder: The encoder to initialize.
        mode: ``"from_scratch"`` (keep the random init) or ``"pretrained"`` (warm-start from
            ``checkpoint``).
        checkpoint: Path to the warm-start checkpoint; required when ``mode`` is
            ``"pretrained"``.

    Raises:
        ValueError: If ``mode`` is unknown, or ``"pretrained"`` without a ``checkpoint``.
    """
    if mode == "from_scratch":
        logger.info("encoder init: from_scratch (random weights)")
        return
    if mode == "pretrained":
        if checkpoint is None:
            raise ValueError("init mode 'pretrained' requires a checkpoint path.")
        load_encoder_checkpoint(encoder, checkpoint)
        return
    raise ValueError(f"unknown init mode {mode!r}; expected 'from_scratch' or 'pretrained'.")
