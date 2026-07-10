"""Tests for the SSL methods and the init/checkpoint utilities.

Checks the two things the loop relies on: ``training_step`` returns a differentiable scalar
whose value can be driven down by optimization, and ``encode`` yields the pooled backbone
embedding. Also covers the ``I`` (init) factor: from-scratch, pretrained warm-start, and the
checkpoint round-trip.
"""

from pathlib import Path

import pytest
import torch

from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.base import apply_encoder_init, load_encoder_checkpoint, save_encoder_checkpoint
from cafl4ds.ssl.factory import build_mae, build_simsiam


def _encoder() -> TinyViTEncoder:
    """Return a tiny encoder for fast tests."""
    return TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)


def _batch(n: int = 8, size: int = 16) -> torch.Tensor:
    """Return a small random image batch."""
    g = torch.Generator()
    g.manual_seed(0)
    return torch.rand(n, 3, size, size, generator=g)


@pytest.mark.parametrize("build", [build_mae, build_simsiam])
def test_training_step_returns_differentiable_scalar(build: object) -> None:
    """Each method's training step is a scalar tensor with a gradient path to the encoder."""
    method = build(_encoder())  # type: ignore[operator]
    loss = method.training_step(_batch())
    assert loss.ndim == 0 and loss.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in method.encoder.parameters())


@pytest.mark.parametrize("build", [build_mae, build_simsiam])
def test_loss_decreases_when_overfitting_a_fixed_batch(build: object) -> None:
    """Optimizing a fixed batch drives the SSL loss down (the objective is learnable)."""
    torch.manual_seed(0)
    method = build(_encoder())  # type: ignore[operator]
    method.train()
    opt = torch.optim.AdamW(method.parameters(), lr=1e-3)
    x = _batch()
    first = method.training_step(x).item()
    for _ in range(30):
        opt.zero_grad()
        loss = method.training_step(x)
        loss.backward()
        opt.step()
    assert method.training_step(x).item() < first


@pytest.mark.parametrize("build", [build_mae, build_simsiam])
def test_encode_returns_pooled_embedding_without_grad(build: object) -> None:
    """Encode yields ``[B, embed_dim]`` detached embeddings for the instruments."""
    method = build(_encoder())  # type: ignore[operator]
    z = method.encode(_batch(n=5))
    assert z.shape == (5, method.encoder.embed_dim)
    assert not z.requires_grad


def test_checkpoint_roundtrip_and_pretrained_init(tmp_path: Path) -> None:
    """Saving then applying a pretrained init reproduces the encoder weights exactly."""
    src = _encoder()
    ckpt = tmp_path / "enc.pt"
    save_encoder_checkpoint(src, ckpt)

    dst = _encoder()
    # Sanity: a fresh encoder differs before loading.
    ref_param = next(iter(src.state_dict().values()))
    assert not torch.allclose(ref_param, next(iter(dst.state_dict().values())))

    apply_encoder_init(dst, mode="pretrained", checkpoint=ckpt)
    for a, b in zip(src.state_dict().values(), dst.state_dict().values(), strict=True):
        assert torch.allclose(a, b)


def test_from_scratch_init_is_a_noop() -> None:
    """from_scratch init leaves the encoder's random weights untouched."""
    enc = _encoder()
    before = {k: v.clone() for k, v in enc.state_dict().items()}
    apply_encoder_init(enc, mode="from_scratch")
    for k, v in enc.state_dict().items():
        assert torch.allclose(v, before[k])


def test_pretrained_without_checkpoint_raises() -> None:
    """Pretrained init requires a checkpoint path."""
    with pytest.raises(ValueError, match="requires a checkpoint"):
        apply_encoder_init(_encoder(), mode="pretrained", checkpoint=None)


def test_unknown_init_mode_raises() -> None:
    """An unknown init mode is rejected."""
    with pytest.raises(ValueError, match="unknown init mode"):
        apply_encoder_init(_encoder(), mode="warm")


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    """Loading a non-existent checkpoint raises a clear error."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_encoder_checkpoint(_encoder(), tmp_path / "nope.pt")
