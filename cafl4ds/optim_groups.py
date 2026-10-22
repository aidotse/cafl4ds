"""Optimizer parameter groups for adapting a pretrained encoder (P1.2.2 / P1.2.4).

A drive's optimizer otherwise sees ``method.parameters()`` as one group: every weight moves at the same
rate and AdamW's decoupled decay pulls every pretrained weight toward zero. Fine-tuning practice
(BEiT, MAE) protects a pretrained encoder in two ways, both selectable here:

* **Layer-wise lr decay** — the encoder's blocks are numbered from the input (embeddings 0, block *i*
  as *i* + 1, the final norm and every non-encoder parameter — the SSL heads — as depth + 1), and a
  parameter at layer *l* trains at ``lr * layer_decay ** (depth + 1 - l)``. Early blocks move least.
* **A no-decay group** — biases, norm weights (every parameter with ≤ 1 dimension), position
  embeddings and the ``cls`` / ``mask`` tokens get no weight decay.

With neither option the result is a single group equal to ``method.parameters()``, so a drive that
does not ask for groups is unchanged.
"""

from __future__ import annotations

import re
from typing import Any

from torch import nn

_NO_DECAY_NAMES = ("pos_embed", "cls_token", "mask_token")
_BLOCK = re.compile(r"^encoder\.blocks\.(\d+)\.")


def layer_id(name: str, depth: int) -> int:
    """The layer a method parameter belongs to, counted from the input.

    0 for the encoder's embeddings, ``i + 1`` for its block ``i``, and ``depth + 1`` for the encoder's
    final norm and everything outside the encoder.
    """
    match = _BLOCK.match(name)
    if match:
        return int(match.group(1)) + 1
    if name.startswith("encoder.") and not name.startswith("encoder.norm."):
        return 0
    return depth + 1


def _no_decay(name: str, param: nn.Parameter) -> bool:
    return param.ndim <= 1 or name.rsplit(".", 1)[-1] in _NO_DECAY_NAMES


def param_groups(
    method: nn.Module,
    lr: float,
    weight_decay: float,
    layer_decay: float | None = None,
    no_decay: bool = False,
) -> list[dict[str, Any]]:
    """Group ``method``'s trainable parameters by lr scale and decay.

    Args:
        method: The SSL method (its encoder at ``method.encoder``, blocks at ``encoder.blocks``).
        lr: The base learning rate (the top layer's).
        weight_decay: The decay of the decayed group.
        layer_decay: Per-block lr decay factor in ``(0, 1]``, or ``None`` for a flat lr.
        no_decay: Exempt biases, norms, position embeddings and tokens from weight decay.

    Returns:
        Optimizer parameter groups, each with its own ``lr`` and ``weight_decay`` (empty groups dropped).

    Raises:
        ValueError: If ``layer_decay`` is outside ``(0, 1]``.
    """
    if layer_decay is not None and not 0.0 < layer_decay <= 1.0:
        raise ValueError(f"layer_decay must be in (0, 1], got {layer_decay}.")
    depth = len(method.encoder.blocks) if layer_decay is not None else 0
    groups: dict[tuple[int, bool], list[nn.Parameter]] = {}
    for name, param in method.named_parameters():
        if not param.requires_grad:
            continue
        layer = layer_id(name, depth) if layer_decay is not None else 0
        groups.setdefault((layer, no_decay and _no_decay(name, param)), []).append(param)
    return [
        {
            "params": params,
            "lr": lr * (layer_decay ** (depth + 1 - layer) if layer_decay is not None else 1.0),
            "weight_decay": 0.0 if exempt else weight_decay,
        }
        for (layer, exempt), params in sorted(groups.items())
    ]
