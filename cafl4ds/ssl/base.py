"""The SSL-method interface (the ``C`` flag) and shared init/checkpoint utilities.

An :class:`SSLMethod` bundles the shared encoder with a self-supervised objective. The
streaming loop only ever calls two things on it:

* :meth:`SSLMethod.training_step` — compute a self-supervised loss on a batch of **raw
  images** (no labels ever enter here); the loop backprops and steps the optimizer.
* :meth:`SSLMethod.encode` — map images to the pooled backbone embedding the health
  instruments and probes read (no gradient).

The ``C`` factor of the experiment matrix selects the concrete method (:mod:`cafl4ds.ssl.mae`
or :mod:`cafl4ds.ssl.simsiam`); the ``I`` factor selects the initialization, applied here via
:func:`apply_method_init` (``from_scratch`` leaves the random init in place; ``pretrained``
loads a checkpoint produced by an *IID* pre-pass, so stream correlation never contaminates the
starting point).
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

    def per_sample_loss(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return the *per-sample* self-supervised loss ``[B]`` (no gradient).

        Where :meth:`training_step` reduces to a single scalar, this keeps the loss for each
        image separately — the free per-frame informativeness signal the plan calls out (MAE's
        per-patch reconstruction error; SimSiam's per-view negative cosine). The loss-gate knob
        (:class:`~cafl4ds.filters.loss_gate.LossGate`) reads it to keep the high-loss frames;
        later phases reuse it for novelty scoring. It is a *selection heuristic*, computed under
        ``no_grad`` and never backpropagated, so it is free to draw its own augmentation/mask.

        Args:
            imgs: A batch of raw images ``[B, C, H, W]``. Labels never enter this path.

        Returns:
            A detached ``[B]`` tensor of per-sample losses (same orientation as the scalar
            :meth:`training_step`: lower is better).

        Raises:
            NotImplementedError: If the concrete method does not define a per-sample loss.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define a per-sample loss.")

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return the pooled backbone embedding used by the instruments/probes.

        Args:
            imgs: A batch of images ``[B, C, H, W]``.

        Returns:
            The pooled embedding ``[B, embed_dim]`` (no gradient tracking).
        """
        with torch.no_grad():
            return self.encoder.embed(imgs)

    def embedding_surfaces(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Named frozen embedding surfaces for the health instruments (no gradient).

        The health monitor reads the collapse geometry at each surface. The **default** is a
        single ``"backbone"`` surface — the pooled encoder embedding (:meth:`encode`) that the
        probes and RankMe calibration (P0.2.1) read. Joint-embedding methods override this to
        add a ``"proj"`` surface (the projector/expander output), where the VICReg
        variance/covariance and Wang & Isola alignment/uniformity terms are canonically defined
        and where SimSiam's collapse fingerprint (the terminal projector BatchNorm) lives. This
        lets P0.2.2 calibrate each instrument at *both* surfaces (see
        ``docs/experiments/phase0/P0.2.2.md``).

        Args:
            imgs: A batch of images ``[B, C, H, W]``.

        Returns:
            A ``surface_name -> embedding [B, d]`` dict; the ``"backbone"`` key is always present.
        """
        return {"backbone": self.encode(imgs)}

    def make_views(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return a positive pair (two augmented views) for pair-based instruments, or ``None``.

        Alignment (Wang & Isola 2020) needs a positive pair. Joint-embedding methods override
        this to return two independently augmented views; single-view methods (MAE) return
        ``None``, signalling that pair-based instruments are not applicable and should be
        skipped.

        Args:
            imgs: A batch of raw images ``[B, C, H, W]``.

        Returns:
            A ``(view_a, view_b)`` pair, or ``None`` if the method has no positive-pair notion.
        """
        return None


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
    # Move tensors to CPU before serializing: the checkpoint is device-agnostic on disk
    # (loading uses ``map_location="cpu"``), and saving an ``hpu`` state_dict directly trips a
    # Habana storage-copy bug. ``.contiguous()`` normalizes strides so the CPU copy is clean.
    state = {k: v.detach().contiguous().cpu() for k, v in encoder.state_dict().items()}
    torch.save(state, path)
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


def load_method_checkpoint(method: SSLMethod, checkpoint: str | Path) -> None:
    """Load a warm start into ``method``, accepting either checkpoint shape.

    The ``encoder.``-prefixed keys of a :func:`save_method_checkpoint` file identify it as a full
    method, so the shape is read off the file rather than configured twice: a full checkpoint
    restores the objective's heads too, while a bare encoder ``state_dict``
    (:func:`save_encoder_checkpoint`, or an externally-sourced backbone) loads into the encoder
    and leaves the heads random. The log line says which happened, because it matters:

    Restoring the heads (MAE's decoder, SimSiam's projector/predictor) is what makes a *short*
    warm-started run interpretable. Leave them random and the first steps backpropagate through an
    untrained head, transiently degrading the restored encoder before it recovers — the warm-up
    confound audit P0.3 caught. On a single-pass streaming or federated run that transient can
    consume most of the horizon and mask the effect under study.

    Args:
        method: The method to load weights into (in place). Must be built from the same ``ssl``
            and ``encoder`` config as the checkpoint.
        checkpoint: Path to a ``state_dict`` file — full method or encoder-only.

    Raises:
        FileNotFoundError: If ``checkpoint`` does not exist.
        RuntimeError: If the keys or shapes do not match ``method`` — a config mismatch between
            the pre-pass and this run (e.g. different ``decoder_dim``), not a corrupt file.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"pretrained method checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    if any(key.startswith("encoder.") for key in state):
        method.load_state_dict(state)
        logger.info(f"loaded pretrained method weights (encoder + heads) from {path}")
    else:
        method.encoder.load_state_dict(state)
        logger.info(f"loaded pretrained ENCODER-ONLY weights from {path}; heads keep their random init")


def save_method_checkpoint(method: SSLMethod, checkpoint: str | Path) -> None:
    """Save a method's ``state_dict`` — encoder + objective heads — as the pre-pass warm start.

    Args:
        method: The method whose weights to save.
        checkpoint: Destination path (parent directories are created).
    """
    path = Path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same CPU/contiguous normalization as save_encoder_checkpoint: device-agnostic on disk, and
    # it sidesteps the Habana storage-copy bug when serializing an `hpu` state_dict directly.
    state = {k: v.detach().contiguous().cpu() for k, v in method.state_dict().items()}
    torch.save(state, path)
    logger.info(f"saved pretrained method weights (encoder + heads) to {path}")


def apply_method_init(method: SSLMethod, mode: str = "from_scratch", checkpoint: str | Path | None = None) -> None:
    """Apply the ``I`` (initialization) factor to a whole method.

    The method-level entry point the run scripts use: ``from_scratch`` keeps the random init,
    ``pretrained`` restores a pre-pass checkpoint — encoder *and* heads (see
    :func:`load_method_checkpoint`). Delegates ``from_scratch`` to :func:`apply_encoder_init` so
    the documented no-op contract has a single implementation.

    Args:
        method: The method to initialize.
        mode: ``"from_scratch"`` or ``"pretrained"``.
        checkpoint: Path to the warm-start checkpoint; required when ``mode`` is ``"pretrained"``.

    Raises:
        ValueError: If ``mode`` is unknown, or ``"pretrained"`` without a ``checkpoint``.
    """
    if mode == "pretrained":
        if checkpoint is None:
            raise ValueError("init mode 'pretrained' requires a checkpoint path.")
        load_method_checkpoint(method, checkpoint)
        return
    apply_encoder_init(method.encoder, mode, checkpoint)
