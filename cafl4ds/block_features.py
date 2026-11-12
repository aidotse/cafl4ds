"""Per-block encoder features — the input to P1.1.2's layer-wise lenses.

The health monitor reads one pooled embedding per checkpoint: the encoder's final, normed output
(:meth:`~cafl4ds.models.vit.TinyViTEncoder.embed`). P1.1.2's lenses (per-block CKA / Procrustes
against the well, intrinsic dimension per block) need the representation *after every transformer
block*, to localize where a change lives. :func:`block_features` captures each block's output with
forward hooks during the same pass that produces the monitor's surfaces, so the final layers are
exactly what the monitor reads — no re-implementation of the forward that could drift from it.

Each block output is pooled like :meth:`~cafl4ds.models.vit.TinyViTEncoder.embed`: the mean over the
patch tokens, ``cls`` dropped. Layer names: ``block01`` … ``block<depth>`` (pre final norm), then the
method's surfaces (``backbone``, plus ``proj`` for a joint-embedding method).
"""

from __future__ import annotations

import torch
from torch import nn

from cafl4ds.ssl.base import SSLMethod


def layer_names(depth: int, surfaces: list[str]) -> list[str]:
    """The ordered layer names :func:`block_features` returns for a ``depth``-block encoder."""
    return [f"block{i + 1:02d}" for i in range(depth)] + list(surfaces)


def block_features(method: SSLMethod, images: torch.Tensor, batch_size: int = 128) -> dict[str, torch.Tensor]:
    """Pooled per-block features and the monitor's surfaces for ``images``, on the CPU.

    Args:
        method: The SSL method to read (eval mode for the pass; its training mode is restored).
        images: The probe images ``[N, C, H, W]`` (any device; batches are moved to the encoder's).
        batch_size: Images per forward pass (bounds accelerator memory).

    Returns:
        ``layer -> [N, d]`` float32 CPU tensors, ordered as :func:`layer_names`.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0; got {batch_size}.")
    blocks: list[nn.Module] = list(method.encoder.blocks)
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, _inputs: object, output: torch.Tensor) -> None:
        captured.append(output[:, 1:, :].mean(dim=1).detach().float().cpu())

    chunks: dict[str, list[torch.Tensor]] = {}
    handles = [block.register_forward_hook(hook) for block in blocks]
    was_training = method.training
    method.eval()
    try:
        with torch.no_grad():
            for start in range(0, images.shape[0], batch_size):
                captured.clear()
                surfaces = method.embedding_surfaces(images[start : start + batch_size])
                if len(captured) != len(blocks):
                    raise RuntimeError(f"captured {len(captured)} block outputs for {len(blocks)} blocks.")
                names = layer_names(len(blocks), list(surfaces))
                for name, z in zip(names, [*captured, *surfaces.values()], strict=True):
                    chunks.setdefault(name, []).append(z.detach().float().cpu())
    finally:
        for handle in handles:
            handle.remove()
        method.train(was_training)
    return {name: torch.cat(parts) for name, parts in chunks.items()}
