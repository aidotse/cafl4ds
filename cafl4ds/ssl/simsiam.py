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


def _neg_cosine(p: torch.Tensor, z: torch.Tensor, stop_grad: bool = True) -> torch.Tensor:
    """Negative cosine similarity, with the target branch optionally stop-gradient'd.

    Args:
        p: Predictor output of one branch ``[B, d]`` (gradient flows).
        z: Projector output of the other branch ``[B, d]``.
        stop_grad: When ``True`` (SimSiam as published) the target ``z`` is detached — the
            stop-gradient that prevents collapse. When ``False`` the gradient flows through
            both branches, removing that anti-collapse mechanism (the positive-control
            ablation, see :class:`SimSiam`).

    Returns:
        The mean negative cosine similarity (scalar); minimized when ``p`` aligns with ``z``.
    """
    p = torch.nn.functional.normalize(p, dim=1)
    z = torch.nn.functional.normalize(z.detach() if stop_grad else z, dim=1)
    return -(p * z).sum(dim=1).mean()


class SimSiam(SSLMethod):
    """SimSiam over the shared :class:`~cafl4ds.models.vit.TinyViTEncoder`.

    SimSiam avoids collapse with two coupled mechanisms: a **predictor** on one branch and a
    **stop-gradient** on the target branch (Chen & He 2021). The ``anti_collapse`` flag toggles
    *both* off together, giving the documented collapse ablation used as the Phase-0 positive
    control (P0.2): with the predictor bypassed (``p = z``) and no stop-gradient, the objective
    reduces to ``-cosine(z1, z2)`` with gradients flowing through both branches, whose trivial
    global optimum maps every input to one constant vector (cosine → 1, loss → −1). Collapse is
    then mathematically forced and scale-independent — which is exactly why the toy CPU/HPU
    regime suffices to calibrate the collapse instruments (RankMe, alignment/uniformity).
    """

    def __init__(
        self,
        encoder: TinyViTEncoder,
        projector: MLPHead,
        predictor: MLPHead,
        augment: v2.Transform | None = None,
        anti_collapse: bool = True,
    ) -> None:
        """Build the SimSiam method.

        Args:
            encoder: The shared backbone encoder.
            projector: The projection MLP (3-layer, last BatchNorm) mapping the pooled
                embedding to the latent space.
            predictor: The prediction MLP (2-layer, no last BatchNorm) applied to one branch.
            augment: Per-view augmentation; defaults to
                :func:`~cafl4ds.ssl.augment.make_ssl_augment` sized to the encoder.
            anti_collapse: When ``True`` (default) run SimSiam as published — predictor + stop-
                gradient. When ``False`` disable **both** anti-collapse mechanisms (predictor
                bypassed, stop-gradient off): the forced-collapse positive control (P0.2).
        """
        super().__init__(encoder)
        self.projector = projector
        self.predictor = predictor
        self.anti_collapse = anti_collapse
        img_size = int(round(encoder.num_patches**0.5)) * encoder.patch_size
        base_augment = augment if augment is not None else make_ssl_augment(img_size)
        self.two_view = TwoView(base_augment)

    @property
    def name(self) -> str:
        """Return the method identifier (``"simsiam_collapse"`` for the PC ablation)."""
        return "simsiam" if self.anti_collapse else "simsiam_collapse"

    def _branch(self, view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode → project (→ predict, unless collapsing) one augmented view.

        Args:
            view: An augmented image batch ``[B, C, H, W]``.

        Returns:
            A ``(prediction, projection)`` pair, each ``[B, latent_dim]``. With
            ``anti_collapse=False`` the predictor is removed from the path, so ``p = z``.
        """
        z = self.projector(self.encoder.embed(view))
        p = self.predictor(z) if self.anti_collapse else z
        return p, z

    def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
        """Compute the symmetric negative-cosine loss on two views.

        With ``anti_collapse=True`` this is the published stop-gradient objective; with it
        ``False`` the stop-gradient is dropped (and the predictor already bypassed in
        :meth:`_branch`), yielding the forced-collapse positive control.

        Args:
            imgs: A batch of raw images ``[B, C, H, W]``. Labels never enter this path.

        Returns:
            The symmetrized SimSiam loss (scalar), in ``[-1, 1]`` (lower is better).
        """
        view_1, view_2 = self.two_view(imgs)
        p1, z1 = self._branch(view_1)
        p2, z2 = self._branch(view_2)
        sg = self.anti_collapse
        return 0.5 * (_neg_cosine(p1, z2, stop_grad=sg) + _neg_cosine(p2, z1, stop_grad=sg))
