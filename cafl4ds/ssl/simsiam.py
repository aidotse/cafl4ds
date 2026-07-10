"""SimSiam SSL method — the joint-embedding ``C`` backbone (Chen & He 2021).

SimSiam is the simplest of the joint-embedding family: a projector + predictor with a
**stop-gradient**, and *no* negatives and *no* momentum/EMA target (that is BYOL, a later
drop-in). Its learning signal is pulling together two augmented views, which makes it
susceptible to representational **collapse** — the degradation mode the Phase-1 positive
control will elicit and the health instruments (rank, alignment/uniformity) will catch. The
health monitor reads the *encoder* embedding, never the projector/predictor outputs.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

from cafl4ds.models.heads import MLPHead
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.augment import TwoView, make_ssl_augment
from cafl4ds.ssl.base import SSLMethod


def _neg_cosine(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity with the target branch stop-gradient'd.

    Args:
        p: Predictor output of one branch ``[B, d]`` (gradient flows).
        z: Projector output of the other branch ``[B, d]`` (detached — stop-gradient).

    Returns:
        The mean negative cosine similarity (scalar); minimized when ``p`` aligns with ``z``.
    """
    p = torch.nn.functional.normalize(p, dim=1)
    z = torch.nn.functional.normalize(z.detach(), dim=1)
    return -(p * z).sum(dim=1).mean()


class SimSiam(SSLMethod):
    """SimSiam over the shared :class:`~cafl4ds.models.vit.TinyViTEncoder`."""

    def __init__(
        self,
        encoder: TinyViTEncoder,
        projector: MLPHead,
        predictor: MLPHead,
        augment: v2.Transform | None = None,
    ) -> None:
        """Build the SimSiam method.

        Args:
            encoder: The shared backbone encoder.
            projector: The projection MLP (3-layer, last BatchNorm) mapping the pooled
                embedding to the latent space.
            predictor: The prediction MLP (2-layer, no last BatchNorm) applied to one branch.
            augment: Per-view augmentation; defaults to
                :func:`~cafl4ds.ssl.augment.make_ssl_augment` sized to the encoder.
        """
        super().__init__(encoder)
        self.projector = projector
        self.predictor = predictor
        img_size = int(round(encoder.num_patches**0.5)) * encoder.patch_size
        base_augment = augment if augment is not None else make_ssl_augment(img_size)
        self.two_view = TwoView(base_augment)

    @property
    def name(self) -> str:
        """Return the method identifier."""
        return "simsiam"

    def _branch(self, view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode → project → predict one augmented view.

        Args:
            view: An augmented image batch ``[B, C, H, W]``.

        Returns:
            A ``(prediction, projection)`` pair, each ``[B, latent_dim]``.
        """
        z = self.projector(self.encoder.embed(view))
        p = self.predictor(z)
        return p, z

    def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
        """Compute the symmetric stop-gradient negative-cosine loss on two views.

        Args:
            imgs: A batch of raw images ``[B, C, H, W]``. Labels never enter this path.

        Returns:
            The symmetrized SimSiam loss (scalar), in ``[-1, 1]`` (lower is better).
        """
        view_1, view_2 = self.two_view(imgs)
        p1, z1 = self._branch(view_1)
        p2, z2 = self._branch(view_2)
        return 0.5 * (_neg_cosine(p1, z2) + _neg_cosine(p2, z1))
