"""Scientific invariants for entropy reduction, exact admission, and the task-gradient gate."""

from __future__ import annotations

import copy

import pytest
import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.routing_entropy import RoutingEntropyHalf
from cafl4ds.measurements import routing_entropy
from cafl4ds.models.vit import MoEViTEncoder, TinyViTEncoder
from cafl4ds.ssl.factory import build_mae


def test_entropy_excludes_cls_and_averages_entropies_before_layers() -> None:
    """Confident different experts remain zero-entropy even when the average distribution is uniform."""
    p = torch.tensor([[[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    q = torch.full_like(p, 0.5)
    result = routing_entropy({2: p, 3: q})
    torch.testing.assert_close(result.token_normalized, torch.full((1, 2), 0.5))
    assert result.image_normalized.item() == pytest.approx(0.5)
    assert result.image_nats.item() == pytest.approx(0.5 * torch.log(torch.tensor(2.0)).item())
    assert result.layer_image_normalized[2].item() == 0
    assert not result.token_nats.requires_grad
    permuted = routing_entropy({2: p.flip(-1), 3: q.flip(-1)})
    torch.testing.assert_close(result.image_normalized, permuted.image_normalized)


def test_normalization_respects_each_layers_expert_count() -> None:
    """Uniform routers have normalized entropy one even with different expert counts."""
    result = routing_entropy({0: torch.full((2, 4, 2), 0.5), 1: torch.full((2, 4, 4), 0.25)})
    torch.testing.assert_close(result.image_normalized, torch.ones(2))


@pytest.mark.parametrize(
    "p",
    [
        torch.tensor([[[float("nan"), 0.0], [1.0, 0.0]]]),
        torch.ones(1, 3, 1),
        torch.ones(1, 3, 2),
        -torch.ones(1, 3, 2),
        torch.ones(1, 1, 2) / 2,
    ],
)
def test_invalid_probability_tensors_fail(p: torch.Tensor) -> None:
    """Never turn malformed or non-finite routing into silently altered admission."""
    with pytest.raises(ValueError):
        routing_entropy({0: p})
    with pytest.raises(ValueError):
        routing_entropy({})
    with pytest.raises(ValueError):
        routing_entropy({0: torch.full((1, 3, 2), 0.5), 1: torch.full((1, 4, 2), 0.5)})


def _method() -> torch.nn.Module:
    torch.manual_seed(5)
    return build_mae(
        MoEViTEncoder(img_size=16, embed_dim=16, depth=3, num_heads=2), decoder_dim=16, decoder_heads=2, decoder_depth=1
    )


@pytest.mark.parametrize("strategy", ["random", "high", "low"])
def test_admission_is_exact_read_only_and_resumable(strategy: str) -> None:
    """The same candidate identities and private RNG reproduce admission without modifying training state."""
    method = _method()
    method.train()
    method.encoder.blocks[0].eval()
    images = torch.rand(8, 3, 16, 16)
    batch = StreamBatch(images, 0, 0, torch.arange(80, 88))
    modes = [m.training for m in method.modules()]
    before = {k: v.clone() for k, v in method.state_dict().items()}
    for parameter in method.parameters():
        parameter.grad = torch.ones_like(parameter)
    rng = torch.get_rng_state().clone()
    selector = RoutingEntropyHalf(strategy, 3)
    saved = copy.deepcopy(selector.state_dict())
    chosen = selector.select(batch, FilterContext(method, 0))
    record = selector.last_record
    assert record is not None
    assert chosen.shape[0] == 4
    scores = record.entropy.image_normalized
    accepted = record.accepted_indices
    rejected = torch.tensor([i for i in range(8) if i not in accepted.tolist()])
    if strategy == "high":
        assert scores[accepted].min() >= scores[rejected].max()
    if strategy == "low":
        assert scores[accepted].max() <= scores[rejected].min()
    torch.testing.assert_close(chosen, images[accepted])
    assert record.sample_ids.tolist() == list(range(80, 88))
    assert torch.equal(torch.get_rng_state(), rng)
    assert modes == [m.training for m in method.modules()]
    assert all(torch.equal(before[k], v) for k, v in method.state_dict().items())
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in method.parameters())
    clone = RoutingEntropyHalf(strategy)
    clone.load_state_dict(saved)
    torch.testing.assert_close(chosen, clone.select(batch, FilterContext(method, 0)))


def test_uniform_ties_are_randomized_with_independent_rng() -> None:
    """Exactly tied scores reproduce random-half admission rather than favoring image positions."""
    method = _method()
    for block in method.encoder.blocks:
        if hasattr(block.mlp, "router"):
            with torch.no_grad():
                block.mlp.router.weight.zero_()
                block.mlp.router.bias.zero_()
    batch = StreamBatch(torch.rand(8, 3, 16, 16), 0, 0, torch.arange(8))
    random = RoutingEntropyHalf("random", 42)
    high = RoutingEntropyHalf("high", 42)
    low = RoutingEntropyHalf("low", 42)
    expected = random.select(batch, FilterContext(method, 0))
    torch.testing.assert_close(expected, high.select(batch, FilterContext(method, 0)))
    torch.testing.assert_close(expected, low.select(batch, FilterContext(method, 0)))


def test_invalid_selection_setup_fails() -> None:
    """Require token routing, complete even batches, and audit identities."""
    method = _method()
    selector = RoutingEntropyHalf()
    for batch in (
        StreamBatch(torch.rand(5, 3, 16, 16), 0, 0, torch.arange(5)),
        StreamBatch(torch.rand(8, 3, 16, 16), 0, 0),
    ):
        with pytest.raises(ValueError):
            selector.select(batch, FilterContext(method, 0))
    batch = StreamBatch(torch.rand(8, 3, 16, 16), 0, 0, torch.arange(8))
    method.encoder = TinyViTEncoder(img_size=16, embed_dim=16, depth=3, num_heads=2)
    with pytest.raises(ValueError, match="token-routed"):
        selector.select(batch, FilterContext(method, 0))
    with pytest.raises(ValueError):
        RoutingEntropyHalf("bogus")
    with pytest.raises(ValueError):
        selector.load_state_dict(RoutingEntropyHalf("high").state_dict())


def test_mae_reconstruction_trains_top_one_router_without_auxiliary_loss() -> None:
    """Prove the requested gate carries gradients from actual masked reconstruction alone."""
    method = _method()
    method.aux_weight = 0
    noise = torch.rand(4, method.encoder.num_patches)
    losses = method.loss_components(torch.rand(4, 3, 16, 16), augmented=True, mask_noise=noise)
    torch.testing.assert_close(losses.total, losses.reconstruction)
    losses.total.backward()
    for block in method.encoder.blocks:
        if hasattr(block.mlp, "router"):
            assert block.mlp.router.weight.grad.abs().sum() > 0
