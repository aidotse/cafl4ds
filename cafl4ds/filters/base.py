"""The selection-filter interface (the ``A`` factor / independent variable).

The filter is the knob the whole study turns. In Phase 0 only the **B-floor** exists
(:class:`~cafl4ds.filters.accept_all.AcceptAll`): accept every incoming batch, no replay, raw
stream order — the honest floor of the streaming loop. Later phases add novelty (F-a),
coverage (F-b), and the health-steerable filter (F-c) behind this same interface.

The :class:`FilterContext` deliberately carries the live model and step so that
model-in-the-loop filters (the *fast edge*) and, later, health state (the *slow edge*) have
what they need without changing the loop's call site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.ssl.base import SSLMethod


@dataclass
class FilterContext:
    """Everything a filter may consult when selecting from a batch."""

    method: SSLMethod
    """The live SSL method — its encoder scores informativeness against the *current* model
    (the fast edge). B-floor ignores it."""
    step: int
    """The current global stream step."""


class Filter(ABC):
    """Selects which images from an incoming stream batch enter the SSL update."""

    @abstractmethod
    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Return the images accepted from ``batch`` for this update.

        Args:
            batch: The incoming label-free stream batch.
            ctx: The live filter context (model, step).

        Returns:
            The accepted images ``[K, C, H, W]`` with ``0 <= K <= B`` (possibly the whole
            batch, possibly empty).
        """
