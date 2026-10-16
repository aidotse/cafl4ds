"""Unit tests for drive branching — resume, surgical damage, recovery pass (``cafl4ds.branch``)."""

from __future__ import annotations

import pytest
import torch

from cafl4ds.branch import ResumedStream, damage_top_blocks, recovery_batches, top_blocks
from cafl4ds.data.attributes import SyntheticAttributeSource
from cafl4ds.data.regime import RegimeStream
from cafl4ds.models.vit import TinyViTEncoder


def _stream(**kwargs: object) -> RegimeStream:
    """A small three-regime walk."""
    source = SyntheticAttributeSource(num_regimes=3, num_canary_classes=2, per_cell=12, img_size=8, long_tail=False)
    return RegimeStream(source, batch_size=8, support_per_canary=3, query_per_canary=3, **kwargs)  # type: ignore[arg-type]


def _encoder() -> TinyViTEncoder:
    """A trained-looking encoder: non-default LayerNorms, so a reinit visibly resets them."""
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=8, patch_size=4, embed_dim=16, depth=4, num_heads=2)
    with torch.no_grad():
        for p in encoder.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return encoder


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """A detached copy of a module's state dict."""
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def test_resumed_stream_delivers_the_tail_of_the_rebuilt_drive() -> None:
    """From step ``s`` on, the resumed stream is the original drive batch for batch, steps kept."""
    full = list(_stream(seed=3))
    start = len(full) // 2
    resumed = ResumedStream(_stream(seed=3), start)
    tail = list(resumed)
    assert len(resumed) == len(tail) == len(full) - start
    assert [b.step for b in tail] == [b.step for b in full[start:]]
    assert all(torch.equal(a.images, b.images) and a.era == b.era for a, b in zip(full[start:], tail, strict=True))


def test_resumed_stream_rejects_a_negative_start() -> None:
    """Resumed stream rejects a negative start."""
    with pytest.raises(ValueError, match="start"):
        ResumedStream(_stream(), -1)


def test_recovery_batches_draw_the_well_pool_and_wrap_passes() -> None:
    """The recovery pass streams only well-pool frames, numbered 0..n-1, re-shuffling past one pass."""
    stream = _stream(well_fraction=0.5)
    one_pass = list(stream.stationary_batches())
    n = len(one_pass) + 2
    batches = recovery_batches(_stream(well_fraction=0.5), n)
    assert [b.step for b in batches] == list(range(n))
    assert all(b.era == 0 for b in batches)
    drive = torch.cat([b.images for b in _stream(well_fraction=0.5)])
    for b in batches:  # no recovery image is a drive image (well_overlap = 0)
        assert not any(torch.equal(img, d) for img in b.images for d in drive)


def test_recovery_batches_need_a_positive_count() -> None:
    """Recovery batches need a positive count."""
    with pytest.raises(ValueError, match="n_steps"):
        recovery_batches(_stream(), 0)


def test_top_blocks_are_the_last_k() -> None:
    """Top blocks are the last k."""
    encoder = _encoder()
    assert top_blocks(encoder, 2) == list(encoder.blocks)[2:]
    assert top_blocks(encoder, 0) == []
    with pytest.raises(ValueError, match="k must be"):
        top_blocks(encoder, 5)


@pytest.mark.parametrize(("mode", "k", "scale"), [("none", 2, 1.0), ("reinit", 0, 0.0), ("noise", 2, 0.0)])
def test_dose_zero_leaves_the_encoder_untouched(mode: str, k: int, scale: float) -> None:
    """Dose zero leaves the encoder untouched."""
    encoder = _encoder()
    before = _state(encoder)
    out = damage_top_blocks(encoder, mode, k, scale=scale)
    assert out["rel_change"] == 0.0
    assert all(torch.equal(before[name], v) for name, v in encoder.state_dict().items())


@pytest.mark.parametrize(("mode", "scale"), [("reinit", 0.0), ("noise", 0.5)])
def test_damage_touches_only_the_top_blocks(mode: str, scale: float) -> None:
    """Damage touches only the top blocks."""
    encoder = _encoder()
    before = _state(encoder)
    out = damage_top_blocks(encoder, mode, 2, scale=scale, seed=1)
    assert out["rel_change"] > 0.0
    after = encoder.state_dict()
    changed = {name for name in before if not torch.equal(before[name], after[name])}
    assert changed and all(name.startswith(("blocks.2.", "blocks.3.")) for name in changed)


def test_reinit_resets_layernorms_and_noise_spares_them() -> None:
    """Reinit resets layernorms and noise spares them."""
    reinit, noised = _encoder(), _encoder()
    damage_top_blocks(reinit, "reinit", 1)
    damage_top_blocks(noised, "noise", 1, scale=0.5)
    assert torch.equal(reinit.blocks[3].norm1.weight, torch.ones_like(reinit.blocks[3].norm1.weight))
    assert torch.equal(noised.blocks[3].norm1.weight, _encoder().blocks[3].norm1.weight)


def test_noise_dose_scales_the_dent() -> None:
    """A larger noise scale moves the weights further (relative change ≈ scale)."""
    small = damage_top_blocks(_encoder(), "noise", 2, scale=0.1)["rel_change"]
    large = damage_top_blocks(_encoder(), "noise", 2, scale=0.4)["rel_change"]
    assert small < large
    assert large == pytest.approx(0.4, rel=0.2)


def test_damage_is_seeded_and_leaves_the_global_rng_alone() -> None:
    """Same seed, same damage; and the global RNG continues exactly as if no damage were drawn."""
    a, b = _encoder(), _encoder()
    torch.manual_seed(7)
    damage_top_blocks(a, "noise", 2, scale=0.3, seed=5)
    after_damage = torch.rand(3)
    torch.manual_seed(7)
    damage_top_blocks(b, "reinit", 2, seed=5)
    assert torch.equal(torch.rand(3), after_damage)
    c = _encoder()
    damage_top_blocks(c, "noise", 2, scale=0.3, seed=5)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), c.state_dict().values(), strict=True))


def test_damage_rejects_bad_arguments() -> None:
    """Damage rejects bad arguments."""
    with pytest.raises(ValueError, match="mode"):
        damage_top_blocks(_encoder(), "melt", 1)
    with pytest.raises(ValueError, match="scale"):
        damage_top_blocks(_encoder(), "noise", 1, scale=-1.0)
