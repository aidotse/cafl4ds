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

import itertools
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


class TruncatedStream:
    """A stream wrapper that stops after ``n_steps`` batches — the drive's opening, on the full drive's schedules."""

    def __init__(self, stream: Iterable[StreamBatch], n_steps: int) -> None:
        """Wrap ``stream``.

        Args:
            stream: The drive stream (sized).
            n_steps: How many of its batches to deliver.

        Raises:
            ValueError: If ``n_steps`` is not positive.
        """
        if n_steps <= 0:
            raise ValueError(f"n_steps must be > 0; got {n_steps}.")
        self.stream = stream
        self.n_steps = n_steps

    def __len__(self) -> int:
        """The batches delivered."""
        return min(len(self.stream), self.n_steps)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[StreamBatch]:
        """Yield the wrapped stream's first ``n_steps`` batches."""
        return itertools.islice(self.stream, self.n_steps)


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


def _linears(blocks: list[nn.Module]) -> list[nn.Linear]:
    return [m for block in blocks for m in block.modules() if isinstance(m, nn.Linear)]


def _rel_change(linears: list[nn.Linear], before: list[torch.Tensor]) -> float:
    num = sum(float((m.weight.detach() - b).pow(2).sum()) for m, b in zip(linears, before, strict=True))
    den = sum(float(b.pow(2).sum()) for b in before)
    return (num / den) ** 0.5 if den > 0 else 0.0


def _add_noise(linears: list[nn.Linear], twins: list[nn.Linear], scale: float, generator: torch.Generator) -> None:
    """Add ``scale`` × each weight's own SD of Gaussian noise; each twin weight gets the identical delta."""
    for i, m in enumerate(linears):
        noise = torch.randn(m.weight.shape, generator=generator).to(m.weight.device)
        delta = noise * (scale * m.weight.std())
        m.weight.add_(delta)
        if twins:
            twins[i].weight.add_(delta.to(twins[i].weight.device))


def damage_top_blocks(
    encoder: TinyViTEncoder,
    mode: str,
    k: int,
    scale: float = 0.0,
    seed: int = 0,
    twin: TinyViTEncoder | None = None,
) -> dict[str, float]:
    """Damage the top ``k`` encoder blocks in place — the surgical positive control.

    ``reinit`` re-draws every Linear (truncated normal, std 0.02, zero bias) and resets every
    LayerNorm, as :meth:`~cafl4ds.models.vit.TinyViTEncoder._init_weights` does at construction: the
    blocks' learned function is erased, while the residual stream still carries the blocks below.
    ``noise`` adds ``scale`` × the tensor's own standard deviation of Gaussian noise to every weight
    matrix (biases and LayerNorm parameters untouched), a graded dent that keeps the blocks' function
    partly intact. ``k = 0``, ``mode="none"`` or a zero noise ``scale`` leave the encoder unchanged.

    ``twin`` is a second backbone of the same architecture that must take the *same* damage — DINO's
    student, whose moving average is the encoder read (P1.1.4). It gets the identical perturbation:
    the same re-drawn weights under ``reinit``, and under ``noise`` the same additive delta (sized by
    ``encoder``'s tensors), so a twin equal to ``encoder`` stays equal, and their difference is kept.

    The draws come from a CPU generator seeded by ``seed`` inside a forked RNG, so the global RNG —
    and with it every later augmentation and masking draw — is left exactly where it was. Without a
    twin the draws are exactly P1.1.1's.

    Args:
        encoder: The encoder to damage (any device).
        mode: ``"none"``, ``"reinit"`` or ``"noise"``.
        k: How many top blocks to damage.
        scale: Noise standard deviation as a multiple of each weight tensor's std (``noise`` only).
        seed: Seed for the damage draws.
        twin: A backbone to damage identically (``None``: the encoder alone).

    Returns:
        ``{"rel_change"}`` — the relative L2 change of the damaged blocks' weight matrices (0 if none);
        with a twin, also ``twin_rel_change``.

    Raises:
        ValueError: If ``mode`` is unknown, ``k`` is out of range, ``scale`` is negative, or the twin's
            blocks do not match the encoder's.
    """
    if mode not in DAMAGE_MODES:
        raise ValueError(f"mode must be one of {DAMAGE_MODES}; got {mode!r}.")
    if scale < 0.0:
        raise ValueError(f"scale must be >= 0; got {scale}.")
    blocks = top_blocks(encoder, k)
    linears = _linears(blocks)
    twin_blocks = top_blocks(twin, k) if twin is not None else []
    twin_linears = _linears(twin_blocks)
    if twin is not None and [m.weight.shape for m in twin_linears] != [m.weight.shape for m in linears]:
        raise ValueError("the twin's top blocks do not match the encoder's.")
    before = [m.weight.detach().clone() for m in linears]
    twin_before = [m.weight.detach().clone() for m in twin_linears]
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        if mode == "reinit":
            for block in blocks:
                _reinit_block(block, generator)
            for block, other in zip(blocks, twin_blocks, strict=False):
                other.load_state_dict(block.state_dict())
        elif mode == "noise" and scale > 0.0:
            _add_noise(linears, twin_linears, scale, generator)
    out = {"rel_change": _rel_change(linears, before)}
    if twin is not None:
        out["twin_rel_change"] = _rel_change(twin_linears, twin_before)
    return out


def damage_method(method: nn.Module, mode: str, k: int, scale: float = 0.0, seed: int = 0) -> dict[str, float]:
    """Damage a method's read encoder, and its ``student`` backbone alike when it trains one (DINO).

    A method whose read encoder is a moving average of a trained student must lose the same weights in
    both: damage to the average alone would be pulled back toward the intact student, and damage to
    the student alone would reach the read encoder only diluted, over hundreds of steps. Every other
    state (heads, DINO's centre) is left as it stands.
    """
    student = getattr(method, "student", None)
    twin = student if isinstance(student, TinyViTEncoder) and student is not method.encoder else None
    return damage_top_blocks(method.encoder, mode, k, scale=scale, seed=seed, twin=twin)
