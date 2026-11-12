"""Branch a drive from a saved checkpoint — resume it, damage it, or run a recovery pass (P1.1.1).

P1.1.0 saved the full method at every leg boundary of its drives (:mod:`cafl4ds.leg_checkpoints`).
P1.1.1 reads them three ways, all built here so the Phase-0 :class:`~cafl4ds.loop.StreamingLoop`
runs untouched:

* **Resume.** :class:`ResumedStream` re-delivers a drive from step ``s`` on — the same batches the
  original drive streamed from there, since the stream is rebuilt bit-identically from its seed
  (P1.0.1's data certificate). Loading ``step<s>.pt`` and driving this stream continues the drive
  from the saved state. The optimizer restarts fresh (a checkpoint stores the model, not AdamW's
  moments), so a resumed arm is compared against a *dose-0* arm resumed the same way, never against
  the original drive.
* **Damage.** :func:`damage_top_blocks` is the surgical, dose-controlled positive control: it
  re-initializes, or adds weight noise to, the top ``k`` encoder blocks. It draws from its own seeded
  generator, so a damaged arm and its dose-0 twin consume the global RNG identically — the same
  augmentation and masking draws — and differ only in the damage.
* **Recover.** :func:`recovery_batches` streams the well's stationary pool for a fixed number of
  steps: the short IID pass after a drive, read as savings (porting P0.3.5's recoverability check).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from torch import nn

from cafl4ds.data.regime import RegimeStream
from cafl4ds.data.streams import StreamBatch
from cafl4ds.models.vit import TinyViTEncoder

DAMAGE_MODES = ("none", "reinit", "noise")


class ResumedStream:
    """A stream wrapper that skips every batch before ``start`` — the drive resumed from a checkpoint."""

    def __init__(self, stream: Iterable[StreamBatch], start: int) -> None:
        """Wrap ``stream``.

        Args:
            stream: The rebuilt drive stream (sized, as the loop's multi-epoch guard expects).
            start: The first step to deliver — the ``s`` of the ``step<s>.pt`` checkpoint resumed.
                Batches keep their original ``step`` index, so the resumed arm's checkpoints sit on
                the original drive's grid.

        Raises:
            ValueError: If ``start`` is negative.
        """
        if start < 0:
            raise ValueError(f"start must be >= 0; got {start}.")
        self.stream = stream
        self.start = start

    def __len__(self) -> int:
        """The batches still to deliver."""
        return max(len(self.stream) - self.start, 0)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[StreamBatch]:
        """Yield the wrapped stream's batches from ``start`` on."""
        return (batch for batch in self.stream if batch.step >= self.start)


def recovery_batches(stream: RegimeStream, n_steps: int) -> list[StreamBatch]:
    """The recovery pass: ``n_steps`` batches of the well's stationary pool, numbered ``0..n_steps-1``.

    Draws :meth:`~cafl4ds.data.regime.RegimeStream.stationary_batches` — the well's own shuffled,
    all-regime pool, disjoint from the drive at ``well_overlap=0`` — re-shuffling for another pass if
    ``n_steps`` outruns one. Materialized as a list so it is sized for the loop.

    Args:
        stream: The rebuilt drive stream (its seed fixes the shuffle).
        n_steps: Batches to deliver (``> 0``).

    Returns:
        The recovery batches, every one tagged ``era=0``.

    Raises:
        ValueError: If ``n_steps`` is not positive.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0; got {n_steps}.")
    batches: list[StreamBatch] = []
    while len(batches) < n_steps:
        before = len(batches)
        for batch in stream.stationary_batches():
            batches.append(StreamBatch(images=batch.images, era=0, step=len(batches)))
            if len(batches) == n_steps:
                break
        if len(batches) == before:
            raise ValueError("the well pool is empty — no recovery batches to draw.")
    return batches


def top_blocks(encoder: TinyViTEncoder, k: int) -> list[nn.Module]:
    """The last ``k`` transformer blocks of ``encoder`` (the ones nearest its output).

    Raises:
        ValueError: If ``k`` is outside ``[0, depth]``.
    """
    depth = len(encoder.blocks)
    if not 0 <= k <= depth:
        raise ValueError(f"k must be in [0, {depth}]; got {k}.")
    return list(encoder.blocks)[depth - k :]


def _reinit_block(block: nn.Module, generator: torch.Generator) -> None:
    """Re-draw a block's Linears (truncated normal, std 0.02, zero bias) and reset its LayerNorms, in place."""
    for m in block.modules():
        if isinstance(m, nn.Linear):
            fresh = torch.empty(m.weight.shape)
            nn.init.trunc_normal_(fresh, std=0.02, generator=generator)
            m.weight.copy_(fresh)
            if m.bias is not None:
                m.bias.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.weight.fill_(1.0)
            m.bias.zero_()


def damage_top_blocks(
    encoder: TinyViTEncoder, mode: str, k: int, scale: float = 0.0, seed: int = 0
) -> dict[str, float]:
    """Damage the top ``k`` encoder blocks in place — the surgical positive control.

    ``reinit`` re-draws every Linear (truncated normal, std 0.02, zero bias) and resets every
    LayerNorm, as :meth:`~cafl4ds.models.vit.TinyViTEncoder._init_weights` does at construction: the
    blocks' learned function is erased, while the residual stream still carries the blocks below.
    ``noise`` adds ``scale`` × the tensor's own standard deviation of Gaussian noise to every weight
    matrix (biases and LayerNorm parameters untouched), a graded dent that keeps the blocks' function
    partly intact. ``k = 0``, ``mode="none"`` or a zero noise ``scale`` leave the encoder unchanged.

    The draws come from a CPU generator seeded by ``seed`` inside a forked RNG, so the global RNG —
    and with it every later augmentation and masking draw — is left exactly where it was.

    Args:
        encoder: The encoder to damage (any device).
        mode: ``"none"``, ``"reinit"`` or ``"noise"``.
        k: How many top blocks to damage.
        scale: Noise standard deviation as a multiple of each weight tensor's std (``noise`` only).
        seed: Seed for the damage draws.

    Returns:
        ``{"rel_change"}`` — the relative L2 change of the damaged blocks' weight matrices (0 if none).

    Raises:
        ValueError: If ``mode`` is unknown, ``k`` is out of range, or ``scale`` is negative.
    """
    if mode not in DAMAGE_MODES:
        raise ValueError(f"mode must be one of {DAMAGE_MODES}; got {mode!r}.")
    if scale < 0.0:
        raise ValueError(f"scale must be >= 0; got {scale}.")
    blocks = top_blocks(encoder, k)
    linears = [m for block in blocks for m in block.modules() if isinstance(m, nn.Linear)]
    before = [m.weight.detach().clone() for m in linears]
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        if mode == "reinit":
            for block in blocks:
                _reinit_block(block, generator)
        elif mode == "noise" and scale > 0.0:
            for m in linears:
                noise = torch.randn(m.weight.shape, generator=generator).to(m.weight.device)
                m.weight.add_(noise * (scale * m.weight.std()))
    num = sum(float((m.weight.detach() - b).pow(2).sum()) for m, b in zip(linears, before, strict=True))
    den = sum(float(b.pow(2).sum()) for b in before)
    return {"rel_change": (num / den) ** 0.5 if den > 0 else 0.0}
