"""The FedProx proximal term — a leash tethering a client's local update to the global model.

FedProx (Li et al. 2020, *Federated Optimization in Heterogeneous Networks*) attacks the same
problem as the adaptive server optimizers in :mod:`~cafl4ds.federated.server_optim`, but from the
opposite end of the round. Under a skewed partition each client's local optimum sits somewhere
different, so local training pulls the clients apart — **client drift**. Averaging drifted models
gives a global model that is nobody's optimum.

The two cures are complementary, and that is worth being precise about:

* A **server** optimizer lets clients drift, then reconciles the disagreement when applying the
  average. It is a *post-hoc* correction.
* **FedProx** stops the drift happening in the first place, by changing what each client
  *optimizes*. It is a *preventative* constraint.

They compose: FedProx is a client-side change and FedOpt is a server-side one, so a run can use
both at once. Nothing here touches aggregation.

How it works
------------

Each client adds a penalty for straying from the weights it was handed at the start of the round.
Writing ``w`` for the local weights and ``w_global`` for that broadcast anchor, the client
minimizes::

    L_prox(w) = L_ssl(w) + (mu / 2) * ||w - w_global||^2

The gradient of that penalty is just ``mu * (w - w_global)`` — a force pulling every parameter
back towards the anchor, proportional to how far it has strayed. So rather than rewriting the
loss, this module adds that force directly to ``param.grad`` after ``backward()``. The two are
mathematically identical, and adding to the gradient keeps the SSL loss computation untouched:
the number in the run log stays the *SSL* loss, comparable across every run in the project.

``mu`` is the leash length, and it interpolates between two familiar regimes:

* ``mu = 0`` — no leash. Exactly FedAvg's client, and an exact no-op in this code (the fast path
  returns before touching a single gradient), so the default cannot perturb an existing run.
* ``mu -> infinity`` — an infinitely short leash. The client cannot move at all, and federation
  stops learning.

Useful values are small: Li et al. sweep ``{0.001, 0.01, 0.1, 1.0}`` and the useful setting is
data-dependent, so treat it as a knob to sweep rather than a constant to trust.

The anchor is per-round, not per-run
------------------------------------

The penalty is measured against the weights broadcast *this* round, refreshed every round by
:meth:`~cafl4ds.federated.client.FederatedClient.set_weights`. This is the whole point: the leash
is re-tied to the current consensus each round, so it limits *within-round* divergence without
ever forbidding the federation from moving. An anchor frozen at the run's start would instead
pin the model to its initialization.

Parameters only
---------------

The penalty applies to learnable parameters. A ``state_dict`` also carries buffers that measure
the data rather than being optimized — BatchNorm ``running_mean`` / ``running_var``, step
counters — and a gradient force on a measurement is meaningless. Iterating
``model.named_parameters()`` excludes them by construction, which is the same distinction
:meth:`~cafl4ds.federated.orchestrator.FederatedOrchestrator._apply_server_step` makes on the
server side.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from loguru import logger

StateDict = dict[str, torch.Tensor]


class ProximalTerm:
    """FedProx's proximal penalty: a per-round anchor plus the gradient force towards it.

    One instance belongs to one client — it holds that client's anchor and its last measured
    penalty. Clients must not share an instance.

    Usage is two calls, in this order each round:

    #. :meth:`set_anchor` when the client receives the broadcast global weights.
    #. :meth:`apply_gradient` after every ``backward()``, before the optimizer steps.
    """

    def __init__(self, mu: float = 0.0) -> None:
        """Configure the proximal term.

        Args:
            mu: Penalty strength. ``0.0`` (the default) disables the term entirely, recovering
                FedAvg's client exactly; larger values tie the client more tightly to the
                broadcast weights. Sweep rather than assume — see the module docstring.

        Raises:
            ValueError: If ``mu`` is negative (a negative penalty would *reward* drift).
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
        """Tie the leash to the weights the client just received (start of a round).

        Stores detached clones, so later local training cannot move the anchor underneath the
        penalty. Call this every round; the anchor is deliberately *not* the run's initial
        weights.

        Args:
            state: The broadcast global ``state_dict``. Buffers may be present and are simply
                never read — only parameter entries are looked up by :meth:`apply_gradient`.
        """
        self._anchor = {key: value.detach().clone() for key, value in state.items()}

    def apply_gradient(self, named_parameters: Iterable[tuple[str, torch.Tensor]]) -> float:
        """Add ``mu * (w - w_global)`` to each parameter's gradient, in place.

        Call after ``backward()`` and before the optimizer steps, so the proximal force is part
        of the gradient that is clipped and applied.

        A parameter with no gradient is skipped. That is safe rather than an approximation: the
        anchor equals the weights at the start of the round, and a parameter the SSL loss never
        touches is never stepped, so its drift — and hence its proximal gradient — stays zero.

        Args:
            named_parameters: The model's ``named_parameters()``. Buffers are excluded by
                construction, since a gradient force on a running statistic is meaningless.

        Returns:
            The penalty ``(mu / 2) * ||w - w_global||^2`` at this step, or ``0.0`` when the term
            is disabled. Zero at the first step of a round, where the local weights *are* the
            anchor.
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
