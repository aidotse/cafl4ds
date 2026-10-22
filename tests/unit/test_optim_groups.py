"""Unit tests for the fine-tuning parameter groups (``cafl4ds.optim_groups``)."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.optim_groups import layer_id, param_groups


class _Method(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """A tiny ViT encoder plus a head — the shape of an SSL method."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = TinyViTEncoder(img_size=8, patch_size=4, embed_dim=8, depth=3, num_heads=2)
        self.head = nn.Linear(8, 4)


def _lr_of(groups: list[dict[str, Any]], param: torch.Tensor) -> tuple[float, float]:
    """The ``(lr, weight_decay)`` of the group holding ``param``."""
    (group,) = [g for g in groups if any(p is param for p in g["params"])]
    return group["lr"], group["weight_decay"]


def test_layer_ids_number_the_encoder_from_its_input() -> None:
    """Embeddings are layer 0, block i is i + 1, the final norm and the heads are depth + 1."""
    assert layer_id("encoder.pos_embed", 3) == 0
    assert layer_id("encoder.patch_embed.weight", 3) == 0
    assert layer_id("encoder.blocks.0.attn.qkv.weight", 3) == 1
    assert layer_id("encoder.blocks.2.mlp.fc1.bias", 3) == 3
    assert layer_id("encoder.norm.weight", 3) == 4
    assert layer_id("decoder.blocks.0.attn.qkv.weight", 3) == 4


def test_no_groups_requested_is_one_flat_decayed_group() -> None:
    """Neither option: one group, every trainable parameter, the base lr and decay."""
    method = _Method()
    (group,) = param_groups(method, lr=1e-3, weight_decay=0.05)
    assert len(group["params"]) == len(list(method.parameters()))
    assert group["lr"] == 1e-3 and group["weight_decay"] == 0.05


def test_layer_decay_and_no_decay_scale_and_exempt_the_right_parameters() -> None:
    """Early blocks get geometrically smaller lrs; biases, norms and tokens get no decay."""
    method = _Method()
    groups = param_groups(method, lr=1.0, weight_decay=0.05, layer_decay=0.5, no_decay=True)
    enc = method.encoder
    assert _lr_of(groups, method.head.weight) == (1.0, 0.05)
    assert _lr_of(groups, enc.norm.weight) == (1.0, 0.0)
    assert _lr_of(groups, enc.blocks[2].attn.qkv.weight) == (0.5, 0.05)
    assert _lr_of(groups, enc.blocks[0].attn.qkv.bias) == (0.125, 0.0)
    assert _lr_of(groups, enc.cls_token) == (0.0625, 0.0)
    assert _lr_of(groups, enc.patch_embed.weight) == (0.0625, 0.05)
    assert sum(len(g["params"]) for g in groups) == len(list(method.parameters()))
    torch.optim.AdamW(groups)  # a valid optimizer spec


def test_frozen_parameters_are_left_out_and_bad_decay_raises() -> None:
    """A frozen parameter is not handed to the optimizer; a layer decay outside (0, 1] is refused."""
    method = _Method()
    method.encoder.pos_embed.requires_grad_(False)
    groups = param_groups(method, lr=1.0, weight_decay=0.0, no_decay=True)
    assert all(p is not method.encoder.pos_embed for g in groups for p in g["params"])
    with pytest.raises(ValueError):
        param_groups(method, lr=1.0, weight_decay=0.0, layer_decay=0.0)
