"""Sparse dispatch, balancing gradients, and MoE ViT/SSL integration tests."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch import nn

from cafl4ds.models.vit import Block, FeedForwardOutput, Mlp, MoEMlp, MoEViTEncoder, TinyViTEncoder
from cafl4ds.ssl.base import SSLMethod
from cafl4ds.ssl.factory import build_barlow, build_mae, build_simsiam


def _encoder(routing_level: str = "token", top_k: int = 1) -> MoEViTEncoder:
    """Make a CPU-sized encoder with two MoE layers and biased routers."""
    model = MoEViTEncoder(
        img_size=16,
        patch_size=8,
        embed_dim=16,
        depth=3,
        num_heads=2,
        num_experts=3,
        top_k=top_k,
        routing_level=routing_level,
    )
    with torch.no_grad():
        for block in model.blocks:
            if isinstance(block.mlp, MoEMlp):
                block.mlp.router.bias.copy_(torch.tensor([0.4, 0.0, -0.2]))
    return model


@pytest.mark.parametrize("routing_level", ["token", "image"])
@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_sparse_dispatch_matches_independent_token_reference(routing_level: str, top_k: int) -> None:
    """Sparse accumulation equals a literal token-by-token weighted expert sum."""
    torch.manual_seed(4)
    layer = MoEMlp(4, 7, num_experts=3, top_k=top_k, routing_level=routing_level).eval()
    x = torch.randn(2, 5, 4)
    scores = layer.router(x.mean(dim=1)) if routing_level == "image" else layer.router(x)
    weights, ids = scores.softmax(dim=-1).topk(top_k, dim=-1)
    expected = torch.zeros_like(x)
    for b in range(2):
        for n in range(5):
            index = (b,) if routing_level == "image" else (b, n)
            for slot in range(top_k):
                expert_id = int(ids[index][slot])
                expected[b, n] += weights[index][slot] * layer.experts[expert_id](x[b, n])
    result = layer(x)
    torch.testing.assert_close(result.features, expected)
    assert result.routing_probs.shape == (2, 5, 3)
    torch.testing.assert_close(result.routing_probs.sum(dim=-1), torch.ones(2, 5))
    if routing_level == "image":
        torch.testing.assert_close(result.routing_probs, result.routing_probs[:, :1].expand(-1, 5, -1))


@pytest.mark.parametrize("routing_level", ["token", "image"])
def test_top_one_retains_probability_gradient_and_skips_unused_experts(routing_level: str) -> None:
    """Top-1 keeps the full-softmax weight and trains the router without balancing."""
    layer = MoEMlp(4, 7, num_experts=3, routing_level=routing_level).eval()
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.bias.copy_(torch.tensor([-1.0, 2.0, 0.0]))
    x = torch.randn(2, 5, 4, requires_grad=True)
    called: list[int] = []
    handles = [
        expert.register_forward_hook(lambda _m, _args, _out, i=i: called.append(i))
        for i, expert in enumerate(layer.experts)
    ]
    result = layer(x)
    for handle in handles:
        handle.remove()
    assert called == [1]
    probability = torch.tensor([-1.0, 2.0, 0.0]).softmax(dim=0)[1]
    torch.testing.assert_close(result.features, probability * layer.experts[1](x.reshape(-1, 4)).reshape_as(x))
    result.features.square().sum().backward()
    assert layer.router.weight.grad is not None
    assert layer.router.weight.grad.abs().sum() > 0
    assert layer.router.bias.grad is not None and (layer.router.bias.grad != 0).all()
    assert all(p.grad is None for i in (0, 2) for p in layer.experts[i].parameters())


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
@pytest.mark.parametrize("routing_level", ["token", "image"])
def test_auxiliary_loss_trains_top_one_router(noise_std: float, routing_level: str) -> None:
    """Balancing independently supplies router gradients, with or without noise."""
    torch.manual_seed(3)
    layer = MoEMlp(4, 7, num_experts=3, routing_level=routing_level, router_noise_std=noise_std)
    with torch.no_grad():
        layer.router.bias.copy_(torch.tensor([0.5, 0.0, -0.2]))
    result = layer(torch.randn(8, 5, 4))
    assert result.aux_loss.ndim == 0 and torch.isfinite(result.aux_loss)
    result.aux_loss.backward()
    assert layer.router.weight.grad is not None and layer.router.weight.grad.abs().sum() > 0


def test_load_loss_is_differentiable_without_importance_term() -> None:
    """Expected-load balancing has its own gradient, not just a hard-count penalty."""
    torch.manual_seed(2)
    layer = MoEMlp(4, 7, num_experts=3, importance_weight=0)
    result = layer(torch.randn(8, 5, 4))
    result.aux_loss.backward()
    assert layer.router.weight.grad is not None and layer.router.weight.grad.abs().sum() > 0


def test_balancing_minimum_is_uniform_and_counts_images_once() -> None:
    """Both losses vanish for uniform routing probabilities; replicated tokens add no image load."""
    layer = MoEMlp(4, 7, num_experts=3)
    uniform = torch.zeros(12, 3)
    peaked = torch.tensor([2.0, 0.0, -1.0]).expand(12, -1)
    assert layer._balancing_loss(uniform, uniform, 0.3) == 0
    assert layer._balancing_loss(peaked, peaked, 0.3) > 0
    image_layer = MoEMlp(4, 7, num_experts=3, routing_level="image").eval()
    x = torch.randn(4, 1, 4)
    torch.testing.assert_close(image_layer(x).aux_loss, image_layer(x.expand(-1, 5, -1)).aux_loss)


@pytest.mark.parametrize("routing_level", ["token", "image"])
def test_masked_encoder_composes_with_mae_and_averages_layer_losses(routing_level: str) -> None:
    """MAE sees only visible tokens, while the decoder remains dense and reconstructs all patches."""
    model = _encoder(routing_level)
    layer_losses: list[torch.Tensor] = []
    token_counts: list[int] = []

    def capture(_module: nn.Module, args: tuple[torch.Tensor], output: FeedForwardOutput) -> None:
        layer_losses.append(output.aux_loss)
        token_counts.append(args[0].shape[1])

    handles = [block.mlp.register_forward_hook(capture) for block in model.blocks if isinstance(block.mlp, MoEMlp)]
    images = torch.rand(4, 3, 16, 16)
    result = model.forward_with_aux(images, mask_ratio=0.75)
    assert result.tokens.shape == (4, 2, 16)
    assert result.mask is not None and result.ids_restore is not None
    assert result.mask.shape == result.ids_restore.shape == (4, 4)
    assert token_counts == [2, 2]  # CLS plus one visible patch
    torch.testing.assert_close(result.aux_loss, torch.stack(layer_losses).mean())
    for handle in handles:
        handle.remove()
    method = build_mae(model, decoder_dim=16, decoder_heads=2, decoder_depth=1)
    assert all(isinstance(block.mlp, Mlp) for block in method.decoder.blocks)
    torch.autograd.backward(method.training_step(images))
    assert all(block.mlp.router.weight.grad is not None for block in model.blocks if isinstance(block.mlp, MoEMlp))


@pytest.mark.parametrize("routing_level", ["token", "image"])
def test_routing_is_sequential_deterministic_and_read_only(routing_level: str) -> None:
    """Diagnostic probabilities match actual eval execution without changing RNG, modes, or gradients."""
    model = _encoder(routing_level)
    images = torch.rand(4, 3, 16, 16)
    model.eval()
    expected: dict[int, torch.Tensor] = {}
    handles = []
    for index, block in enumerate(model.blocks):
        if isinstance(block.mlp, MoEMlp):

            def capture(_m: nn.Module, _args: tuple[torch.Tensor], out: FeedForwardOutput, i: int = index) -> None:
                assert out.routing_probs is not None
                expected[i] = out.routing_probs.detach().clone()

            handles.append(block.mlp.register_forward_hook(capture))
    model(images)
    for handle in handles:
        handle.remove()
    model.train()
    model.blocks[0].eval()  # Preserve mixed modes too.
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    modes = [module.training for module in model.modules()]
    params = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    actual = model.routing(images)
    again = model.routing(images)
    assert set(actual) == {1, 2}
    for index, probs in actual.items():
        torch.testing.assert_close(probs, expected[index])
        assert torch.equal(probs, again[index]) and not probs.requires_grad
        assert probs.shape == (4, 5, 3)
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [module.training for module in model.modules()]
    assert all(torch.equal(params[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is not None and torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    assert len(model.attention_maps(images)) == 3


def test_dense_interfaces_and_checkpoint_layout_are_preserved() -> None:
    """A shared dense core retains vanilla computation and state-dict names."""
    model = TinyViTEncoder(img_size=16, embed_dim=16, depth=2, num_heads=2).eval()
    images = torch.rand(2, 3, 16, 16)
    patches = model._embed_patches(images)
    cls = (model.cls_token + model.pos_embed[:, :1]).expand(2, -1, -1)
    expected = torch.cat([cls, patches], dim=1)
    for block in model.blocks:
        block = cast(Block, block)
        expected = expected + block.attn(block.norm1(expected))
        expected = expected + block.mlp(block.norm2(expected))
    expected = model.norm(expected)
    result = model.forward_with_aux(images)
    torch.testing.assert_close(result.tokens, expected)
    assert result.aux_loss == 0
    torch.testing.assert_close(model(images), expected)
    torch.testing.assert_close(model.embed(images), expected[:, 1:].mean(dim=1))
    assert len(model.forward_encoder(images)) == 3
    assert "blocks.0.mlp.fc1.weight" in model.state_dict()
    # Selecting no MoE blocks produces the identical dense checkpoint surface.
    dense_variant = MoEViTEncoder(img_size=16, embed_dim=16, depth=2, num_heads=2, moe_blocks=[])
    dense_variant.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(dense_variant(images), expected)


@pytest.mark.parametrize("routing_level", ["token", "image"])
@pytest.mark.parametrize(
    "factory,options,views",
    [
        (build_mae, {}, 1),
        (build_simsiam, {}, 2),
        (build_simsiam, {"anti_collapse": False}, 2),
        (build_simsiam, {"collapse_alpha": 0.5}, 2),
        (build_barlow, {}, 2),
        (build_barlow, {"anti_collapse": False}, 2),
    ],
)
def test_ssl_adds_aux_from_each_forward_once(
    routing_level: str, factory: Callable[..., SSLMethod], options: dict[str, object], views: int
) -> None:
    """All objective branches add the mean of actual view losses and propagate router gradients."""
    torch.manual_seed(7)
    model = _encoder(routing_level)
    method = factory(model, aux_weight=0, **options)
    images = torch.rand(4, 3, 16, 16)
    torch.manual_seed(8)
    task_loss = method.training_step(images)
    aux_losses: list[torch.Tensor] = []

    def capture(_module: nn.Module, _args: tuple[torch.Tensor], out: FeedForwardOutput) -> None:
        aux_losses.append(out.aux_loss)

    handles = [block.mlp.register_forward_hook(capture) for block in model.blocks if isinstance(block.mlp, MoEMlp)]
    method.aux_weight = 0.7
    torch.manual_seed(8)
    combined = method.training_step(images)
    for handle in handles:
        handle.remove()
    assert len(aux_losses) == views * 2
    torch.testing.assert_close(combined, task_loss + 0.7 * torch.stack(aux_losses).mean())
    torch.autograd.backward(combined)
    for block in model.blocks:
        if isinstance(block.mlp, MoEMlp):
            grad = block.mlp.router.weight.grad
            assert grad is not None and grad.abs().sum() > 0


@pytest.mark.parametrize("factory", [build_mae, build_simsiam])
def test_selection_scores_exclude_auxiliary_loss(factory: Callable[..., SSLMethod]) -> None:
    """Changing aux_weight does not change loss-gate per-image scores."""
    method = factory(_encoder(), aux_weight=0)
    images = torch.rand(4, 3, 16, 16)
    torch.manual_seed(3)
    scores = method.per_sample_loss(images)
    method.aux_weight = 100
    torch.manual_seed(3)
    torch.testing.assert_close(method.per_sample_loss(images), scores)


def test_legacy_embedding_encoder_supplies_zero_auxiliary_loss() -> None:
    """Joint-embedding methods continue accepting wrappers without explicit aux methods."""

    class LegacyEncoder(nn.Module):  # type: ignore[misc]
        embed_dim = 4
        num_patches = 4
        patch_size = 8

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 4)

        def embed(self, imgs: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.linear(imgs.mean(dim=(2, 3))))

    method = build_simsiam(cast(TinyViTEncoder, LegacyEncoder()))
    features, aux = method._embed_with_aux(torch.rand(4, 3, 16, 16))
    assert features.shape == (4, 4) and aux == 0
    torch.autograd.backward(method.training_step(torch.rand(4, 3, 16, 16)))


@pytest.mark.parametrize("ssl", ["mae", "simsiam", "barlow"])
def test_hydra_moe_encoder_and_aux_weight_configuration(ssl: str) -> None:
    """The existing config composition selects MoE and both SSL auxiliary weights."""
    config_dir = str(Path(__file__).resolve().parents[2] / "cafl4ds" / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="loop", overrides=["encoder=moe_tiny", f"ssl={ssl}", "img_size=16"])
    model = instantiate(config.encoder)
    assert isinstance(model, MoEViTEncoder) and model.moe_blocks == (2, 3)
    assert cast(MoEMlp, cast(Block, model.blocks[2]).mlp).top_k == 1
    assert config.ssl.aux_weight == 0.01
    method = instantiate(config.ssl, encoder=model)
    assert method.aux_weight == 0.01


@pytest.mark.parametrize(
    "options",
    [
        {"top_k": 0},
        {"top_k": 5},
        {"num_experts": 0},
        {"routing_level": "invalid"},
        {"router_noise_std": -1},
        {"load_weight": -1},
        {"importance_weight": float("nan")},
    ],
)
def test_invalid_moe_options_fail_early(options: dict[str, Any]) -> None:
    """Invalid router/expert/loss options are rejected at construction."""
    with pytest.raises(ValueError):
        MoEMlp(4, 8, **options)


@pytest.mark.parametrize(
    "options",
    [
        {"moe_blocks": [-1]},
        {"moe_blocks": [4]},
        {"moe_blocks": [2, 2]},
        {"balancing_mode": "ema"},
        {"balancing_mode": "bias"},
        {"depth": 0},
    ],
)
def test_invalid_encoder_options_fail_early(options: dict[str, Any]) -> None:
    """Block selection and deferred balancing modes have explicit validation."""
    with pytest.raises(ValueError):
        MoEViTEncoder(**options)
