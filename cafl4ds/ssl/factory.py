"""Factories that build an :class:`~cafl4ds.ssl.base.SSLMethod` from an encoder + scalars.

The SSL methods take *pre-built* heads (decoder / projector / predictor) so they stay
decoupled from any particular sizing. These factories bridge the gap for Hydra: a config
supplies plain scalar hyper-parameters, the run script supplies the shared ``encoder``, and
the factory sizes the heads from the encoder's own dimensions. This keeps the ``ssl`` config
group a flat list of numbers while the head shapes stay consistent with the backbone.
"""

from __future__ import annotations

from cafl4ds.models.heads import MAEDecoder, MLPHead
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.mae import MAE
from cafl4ds.ssl.simsiam import SimSiam


def build_mae(
    encoder: TinyViTEncoder,
    mask_ratio: float = 0.75,
    decoder_dim: int = 64,
    decoder_depth: int = 2,
    decoder_heads: int = 4,
    decoder_mlp_ratio: float = 2.0,
    norm_pix_loss: bool = True,
) -> MAE:
    """Build an :class:`~cafl4ds.ssl.mae.MAE` with a decoder sized to the encoder.

    Args:
        encoder: The shared backbone encoder.
        mask_ratio: Fraction of patches masked each step.
        decoder_dim: Decoder token width.
        decoder_depth: Number of decoder transformer blocks.
        decoder_heads: Attention heads per decoder block (must divide ``decoder_dim``).
        decoder_mlp_ratio: Decoder MLP hidden width as a multiple of ``decoder_dim``.
        norm_pix_loss: Whether to per-patch normalize the reconstruction targets.

    Returns:
        The assembled MAE method.
    """
    decoder = MAEDecoder(
        num_patches=encoder.num_patches,
        encoder_dim=encoder.embed_dim,
        patch_size=encoder.patch_size,
        in_chans=encoder.in_chans,
        decoder_dim=decoder_dim,
        depth=decoder_depth,
        num_heads=decoder_heads,
        mlp_ratio=decoder_mlp_ratio,
    )
    return MAE(encoder, decoder, mask_ratio=mask_ratio, norm_pix_loss=norm_pix_loss)


def build_simsiam(
    encoder: TinyViTEncoder,
    proj_hidden: int = 256,
    proj_dim: int = 128,
    pred_hidden: int = 64,
    anti_collapse: bool = True,
) -> SimSiam:
    """Build a :class:`~cafl4ds.ssl.simsiam.SimSiam` with heads sized to the encoder.

    Args:
        encoder: The shared backbone encoder.
        proj_hidden: Hidden width of the 3-layer projector.
        proj_dim: Output width of the projector (and the predictor's input/output).
        pred_hidden: Hidden width of the 2-layer predictor bottleneck.
        anti_collapse: Keep SimSiam's predictor + stop-gradient (``True``, the healthy
            control) or disable both for the forced-collapse positive control (``False``).

    Returns:
        The assembled SimSiam method.
    """
    projector = MLPHead(encoder.embed_dim, proj_hidden, proj_dim, num_layers=3, last_bn=True)
    predictor = MLPHead(proj_dim, pred_hidden, proj_dim, num_layers=2, last_bn=False)
    return SimSiam(encoder, projector, predictor, anti_collapse=anti_collapse)
