"""The FedProx proximal term (Li et al. 2020) — a leash tying local training to the global model.

Under a skewed partition each client's local optimum differs, so local training pulls the
clients apart (**client drift**). An adaptive server reconciles drift after the fact; FedProx
prevents it by having each client minimize::

    L_prox(w) = L_ssl(w) + (mu / 2) * ||w - w_global||^2

The penalty's gradient, ``mu * (w - w_global)``, is added straight to ``param.grad`` after
``backward()``. That is equivalent to rewriting the loss, and keeps the logged loss the pure SSL
loss. ``mu = 0`` is an exact no-op (FedAvg's client); Li et al. sweep ``{0.001, 0.01, 0.1, 1}``.

The anchor ``w_global`` is re-tied every round to the broadcast weights, so it limits
within-round drift without pinning the model to its initialization. Only parameters are
penalized; buffers such as BatchNorm running statistics are not.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from loguru import logger

StateDict = dict[str, torch.Tensor]


class ProximalTerm:
    """FedProx's proximal penalty: a per-round anchor plus the gradient force towards it.

    One instance per client. Each round, call :meth:`set_anchor` when the client receives the
    broadcast weights, then :meth:`apply_gradient` after every ``backward()``.
    """

    def __init__(self, mu: float = 0.0) -> None:
        """Configure the proximal term.

        Args:
            mu: Penalty strength. ``0.0`` (the default) disables the term entirely.

        Raises:
            ValueError: If ``mu`` is negative.
        """
        if mu < 0.0:
            raise ValueError(f"proximal mu must be >= 0; got {mu}.")
        self.mu = mu
        self._anchor: StateDict = {}
        self.last_penalty = 0.0
        """The penalty value ``(mu / 2) * ||w - w_global||^2`` at the most recent step."""

    @property
    def enabled(self) -> bool:
        """Whether the term does anything: a positive ``mu`` and an anchor to pull towards."""
        return self.mu > 0.0 and bool(self._anchor)

    def set_anchor(self, state: StateDict) -> None:
        """Tie the leash to the weights the client just received (stored as detached clones).

        Args:
            state: The broadcast global ``state_dict``. Buffer entries are stored but never read.
        """
        self._anchor = {key: value.detach().clone() for key, value in state.items()}

    def apply_gradient(self, named_parameters: Iterable[tuple[str, torch.Tensor]]) -> float:
        """Add ``mu * (w - w_global)`` to each parameter's gradient, in place.

        Call after ``backward()`` and before clipping, so the force is part of the gradient that
        is clipped and applied. Parameters without a gradient are skipped; they are never
        stepped, so their drift is zero anyway.

        Args:
            named_parameters: The model's ``named_parameters()``.

        Returns:
            The penalty ``(mu / 2) * ||w - w_global||^2`` at this step, or ``0.0`` when disabled.
        """
        if not self.enabled:
            return 0.0
        squared_drift = 0.0
        for name, param in named_parameters:
            anchor = self._anchor.get(name)
            if anchor is None or param.grad is None:
                continue
            drift = param.detach() - anchor.to(param.device)
            param.grad.add_(drift, alpha=self.mu)
            squared_drift += float(drift.pow(2).sum())
        self.last_penalty = 0.5 * self.mu * squared_drift
        logger.debug(f"proximal term: mu={self.mu}, penalty={self.last_penalty:.6g}")
        return self.last_penalty
