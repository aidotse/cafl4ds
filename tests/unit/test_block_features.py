"""Unit tests for the per-block feature reader (``cafl4ds.block_features``, P1.1.2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from cafl4ds.block_features import _one_pass, block_features, layer_names
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.dino import build_dino
from cafl4ds.ssl.factory import build_mae, build_simsiam

sys.path.insert(0, str(Path(__file__).parent))
from test_p125 import _DINO_SMALL, _tiny_vit  # noqa: E402


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


def test_a_method_reading_its_encoder_twice_keeps_one_pass() -> None:
    """DINO's surfaces run the teacher twice: the reads are one pass's, the same as a single-pass method's."""
    method = build_mae(_encoder())
    once = block_features(method, _images())
    surfaces = method.embedding_surfaces

    def twice(imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        method.encoder.embed(imgs)
        return surfaces(imgs)

    method.embedding_surfaces = twice  # type: ignore[method-assign]
    again = block_features(method, _images())
    assert list(again) == list(once)
    assert all(torch.allclose(once[k], again[k], atol=1e-6) for k in once)


def test_one_pass_rejects_partial_or_differing_repeats() -> None:
    """A capture that is not whole passes, or a repeat that differs, cannot be read."""
    a, b = torch.zeros(2, 3), torch.ones(2, 3)
    assert _one_pass([a, b, a, b], 2) == [a, b]
    with pytest.raises(RuntimeError, match="captured"):
        _one_pass([a, b, a], 2)
    with pytest.raises(RuntimeError, match="differ"):
        _one_pass([a, b, b, b], 2)


def test_dino_reads_its_teacher_once() -> None:
    """DINO's block reads are its teacher's (the encoder read), one pass, with both surfaces."""
    method = build_dino(_tiny_vit(), **_DINO_SMALL)
    feats = block_features(method, _images())
    assert list(feats) == ["block01", "block02", "backbone", "proj"]
    with torch.no_grad():
        method.eval()
        x = method.encoder._embed_patches(_images())  # pylint: disable=protected-access
        cls = (method.encoder.cls_token + method.encoder.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = method.encoder.blocks[0](torch.cat([cls, x], dim=1))
    assert torch.allclose(feats["block01"], x[:, 1:, :].mean(dim=1), atol=1e-6)
