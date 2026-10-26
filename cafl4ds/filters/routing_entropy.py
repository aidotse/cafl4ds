"""Exact half-batch admission by live routing entropy, with resumable random decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import Filter, FilterContext
from cafl4ds.measurements import RoutingEntropy, routing_entropy
from cafl4ds.models.vit import MoEMlp, MoEViTEncoder


@dataclass
class SelectionRecord:
    """One detached pre-update decision; labels never enter this record."""

    step: int
    sample_ids: torch.Tensor
    accepted_indices: torch.Tensor
    entropy: RoutingEntropy


class RoutingEntropyHalf(Filter):
    """Keep exactly half the images using random, highest, or lowest entropy.

    All modes score for common instrumentation. A private CPU generator breaks ties
    without affecting training RNG. Full probability tensors are not retained.
    """

    def __init__(self, strategy: str = "random", seed: int = 0) -> None:
        """Choose an admission rule and independent random/tie-breaking seed."""
        if strategy not in {"random", "high", "low"}:
            raise ValueError("strategy must be random, high, or low.")
        self.strategy = strategy
        self.generator = torch.Generator().manual_seed(seed)
        self.last_record: SelectionRecord | None = None

    def state_dict(self) -> dict[str, Any]:
        """Persist the random admission/tie stream for exact resume."""
        return {"strategy": self.strategy, "rng": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore only a compatible selection policy."""
        if state["strategy"] != self.strategy:
            raise ValueError("Cannot restore a different admission strategy.")
        self.generator.set_state(state["rng"].cpu())

    @torch.no_grad()  # type: ignore[misc]
    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Score before updating, choose half, and retain the accepted stream order."""
        b = len(batch.images)
        if b < 4 or b % 2:
            raise ValueError("Half-batch admission requires an even batch with at least four images.")
        encoder = ctx.method.encoder
        if not isinstance(encoder, MoEViTEncoder) or not encoder.moe_blocks:
            raise ValueError("Routing entropy selection requires a token-routed MoE encoder.")
        if any(isinstance(m, MoEMlp) and m.routing_level != "token" for m in encoder.modules()):
            raise ValueError("Image routing is not a token-routing experiment.")
        if batch.sample_ids is None or batch.sample_ids.shape != (b,):
            raise ValueError("Selection audit requires stable sample_ids aligned with images.")
        entropy = routing_entropy(encoder.routing(batch.images))
        scores = entropy.image_normalized.cpu()
        permutation = torch.randperm(b, generator=self.generator)
        if self.strategy == "random":
            accepted = permutation[: b // 2]
        else:
            ranked = torch.argsort(scores[permutation], descending=self.strategy == "high", stable=True)
            accepted = permutation[ranked[: b // 2]]
        accepted = accepted.sort().values
        self.last_record = SelectionRecord(ctx.step, batch.sample_ids.detach().cpu().clone(), accepted, entropy)
        return batch.images[accepted.to(batch.images.device)]
