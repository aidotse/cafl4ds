"""Masked Autoencoder (MAE) SSL method — the default ``C`` backbone (He et al. 2022).

MAE masks a large fraction of image patches, encodes the visible ones, and reconstructs the
missing pixels; the per-patch reconstruction error is a free label-free informativeness
signal (used by novelty filters in later phases). MAE is **collapse-resistant** — a constant
code cannot reconstruct diverse pixels — so its Phase-1 degradation mode is
forgetting/overspecialization, not representational collapse; the collapse demonstration uses
the joint-embedding method (:mod:`cafl4ds.ssl.simsiam`) instead.
"""

from __future__ import annotations

from typing import cast

import torch
from torchvision.transforms import v2

from cafl4ds.models.heads import MAEDecoder
from cafl4ds.models.vit import TinyViTEncoder, patchify
from cafl4ds.ssl.augment import make_light_augment
from cafl4ds.ssl.base import SSLMethod


class MAE(SSLMethod):
    """Masked Autoencoder over the shared :class:`~cafl4ds.models.vit.TinyViTEncoder`."""

    def __init__(
        self,
        encoder: TinyViTEncoder,
        decoder: MAEDecoder,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        augment: v2.Transform | None = None,
    ) -> None:
        """Build the MAE method.

        Args:
            encoder: The shared backbone encoder.
            decoder: The pixel-reconstruction decoder.
            mask_ratio: Fraction of patches masked each step.
            norm_pix_loss: Whether to normalize each target patch (per-patch mean/var) before
                the reconstruction loss, as in the MAE paper (usually more stable).
            augment: Light spatial augmentation applied before masking; defaults to
                :func:`~cafl4ds.ssl.augment.make_light_augment` sized to the encoder.
        """
        super().__init__(encoder)
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        img_size = int(round(encoder.num_patches**0.5)) * encoder.patch_size
        self.augment = augment if augment is not None else make_light_augment(img_size)

    @property
    def name(self) -> str:
        """Return the method identifier."""
        return "mae"

    def _reconstruction_target(self, imgs: torch.Tensor) -> torch.Tensor:
        """Patchify images into reconstruction targets, optionally per-patch normalized.

        Args:
            imgs: Images ``[B, C, H, W]``.

        Returns:
            Target patches ``[B, N, p * p * C]``.
        """
        target = patchify(imgs, self.encoder.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()
        return target

    def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
        """Compute the masked reconstruction loss on the masked patches only.

        Args:
            imgs: A batch of raw images ``[B, C, H, W]``.

        Returns:
            The mean squared reconstruction error over masked patches (scalar).
        """
        views = self.augment(imgs)
        latent, mask, ids_restore = self.encoder.forward_encoder(views, mask_ratio=self.mask_ratio)
        assert mask is not None and ids_restore is not None  # noqa: S101 - guaranteed by mask_ratio > 0
        pred = self.decoder(latent, ids_restore)
        target = self._reconstruction_target(views)
        loss_per_patch = (pred - target).pow(2).mean(dim=-1)  # [B, N]
        return cast(torch.Tensor, (loss_per_patch * mask).sum() / mask.sum().clamp_min(1.0))
