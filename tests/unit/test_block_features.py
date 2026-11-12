"""Unit tests for the per-block feature reader (``cafl4ds.block_features``, P1.1.2)."""

from __future__ import annotations

import pytest
import torch

from cafl4ds.block_features import block_features, layer_names
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.factory import build_mae, build_simsiam


def _encoder() -> TinyViTEncoder:
    """A small encoder with perturbed weights, so the blocks differ from one another."""
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=16, patch_size=4, embed_dim=16, depth=3, num_heads=2)
    with torch.no_grad():
        for p in encoder.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return encoder


def _images(n: int = 10) -> torch.Tensor:
    """A small random image batch."""
    return torch.rand(n, 3, 16, 16, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("build", [build_mae, build_simsiam])
def test_layers_and_surfaces_match_the_monitor(build: object) -> None:
    """One ``[N, d]`` read per block plus the method's surfaces, the latter identical to the monitor's."""
    method = build(_encoder())  # type: ignore[operator]
    images = _images()
    feats = block_features(method, images, batch_size=4)
    method.eval()  # the monitor reads in eval mode (SimSiam's projector BatchNorm uses running stats)
    surfaces = method.embedding_surfaces(images)
    assert list(feats) == layer_names(3, list(surfaces))
    assert all(z.shape[0] == images.shape[0] and z.dtype == torch.float32 for z in feats.values())
    for name, z in surfaces.items():
        assert torch.allclose(feats[name], z, atol=1e-6)


def test_block_reads_are_the_pooled_block_outputs() -> None:
    """Each ``block<i>`` is the patch-token mean after block ``i`` of an unmasked forward."""
    method = build_mae(_encoder())
    encoder = method.encoder
    images = _images()
    feats = block_features(method, images)
    with torch.no_grad():
        x = encoder._embed_patches(images)  # pylint: disable=protected-access
        cls = (encoder.cls_token + encoder.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for i, blk in enumerate(encoder.blocks):
            x = blk(x)
            assert torch.allclose(feats[f"block{i + 1:02d}"], x[:, 1:, :].mean(dim=1), atol=1e-6)


def test_batching_hooks_and_mode_are_clean() -> None:
    """Batch size does not change the reads; hooks are removed and the training mode restored."""
    method = build_simsiam(_encoder())
    method.train()
    images = _images()
    one, many = block_features(method, images, batch_size=images.shape[0]), block_features(method, images, 3)
    assert all(torch.allclose(one[k], many[k], atol=1e-6) for k in one)
    assert method.training
    assert all(not blk._forward_hooks for blk in method.encoder.blocks)  # pylint: disable=protected-access
    with pytest.raises(ValueError):
        block_features(method, images, batch_size=0)
