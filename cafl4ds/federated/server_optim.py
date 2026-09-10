"""Server optimizers — how the server *applies* an aggregated client update (FedOpt).

The second half of a federated round, downstream of :mod:`~cafl4ds.federated.aggregate`. Mixing
and applying are two separate decisions, and this module owns the second one:

* **Aggregation** (``aggregate.py``) answers *how do I combine the clients?* — it produces one
  averaged ``state_dict`` from many, weighted by how much each client trained.
* **The server optimizer** (here) answers *what do I do with that average?* — it turns the
  averaged weights into the next global model, possibly using memory of previous rounds.

FedAvg's answer to the second question is "just use it", which is why the distinction is easy to
miss: the average *is* the new global model. FedOpt (Reddi et al. 2021, *Adaptive Federated
Optimization*) makes the distinction load-bearing by reading the round as one step of an
optimizer that lives on the server.

The pseudo-gradient
-------------------

The trick is to treat the round's aggregate as a *gradient*. Every client starts the round from
the same broadcast global weights ``x``, so the difference
``delta = avg_i(client weights) - x`` is the direction the clients collectively moved — a
stand-in for a gradient step that the server never computed itself. Call it the
**pseudo-gradient**. Once you have it, any first-order optimizer can drive the server: FedAvg is
plain SGD on ``delta`` with a learning rate of 1, and the adaptive variants below are
Adam/Yogi/Adagrad on ``delta`` instead.

Note the identity this relies on: averaging the client weights and then differencing equals
averaging the client *deltas*, because every client shares the same starting point
``x``. That equality is what lets this module take the already-averaged ``state_dict``
from ``federated_average`` rather than needing each client's delta. It holds only as long as the
orchestrator broadcasts the global weights to every participant before local training.

Why adaptivity helps under skew
-------------------------------

Under a non-IID partition (the **D** factor) clients pull in conflicting directions, so
``delta`` is large on the coordinates they agree about and small-but-noisy on the ones
they don't. FedAvg treats both the same. An adaptive server divides each coordinate by its own
running gradient scale, which damps the coordinates that only ever disagree.

Two hyperparameters deserve care, because their FedOpt meaning differs from their local-Adam
meaning:

* **``lr``** (the server learning rate) has no FedAvg counterpart — FedAvg's is
  implicitly 1.0, since it takes the client step whole. It is *not* the client learning rate in
  ``configs/optim/`` and should not be set from it. Because the adaptive update is normalized
  (see ``tau``), ``lr`` is roughly the per-coordinate step *in weight space, per round*.
* **``tau``** (the "degree of adaptivity") sits where local Adam puts ``eps``, but it
  is a real knob, not numerical safety. The ratio ``m / (sqrt(v) + tau)`` has magnitude
  ~1 per coordinate, so with a local-Adam-sized ``eps`` of ``1e-8`` every coordinate would move
  by almost exactly ``±lr`` — sign-SGD, discarding the magnitude information about
  *which* coordinates the clients actually agreed on. A larger ``tau`` (~``1e-3``) lets
  genuinely small deltas stay small.

Adding an algorithm
-------------------

Anything with the shape ``(old_state, aggregated_state) -> state`` is a server optimizer; the
:class:`ServerOptimizer` protocol is structural, so no base class is required. In practice the
adaptive family differs *only* in how the second moment accumulates, so subclass
:class:`AdaptiveServerOptimizer` and implement :meth:`~AdaptiveServerOptimizer._second_moment`
(plus :meth:`~AdaptiveServerOptimizer._debias_second_moment` if the accumulator is an EMA). The
three FedOpt variants below are each one line of real arithmetic on top of that.

Not everything belongs here. Server *momentum* without a second moment (FedAvgM) is a natural
fit — a subclass with a constant denominator. FedProx's proximal term is a *client*-side change
and belongs in :mod:`~cafl4ds.federated.client`. Health-gated down-weighting of a degraded
client (**N-F**) changes the *mixing weights*, not their application, so it belongs in
:mod:`~cafl4ds.federated.aggregate`.

What a server optimizer is given
--------------------------------

**Parameters only.** A ``state_dict`` is not uniformly optimizable: alongside learnable
parameters it carries buffers that *measure* the data rather than being descended on — BatchNorm
``running_mean`` / ``running_var``, and integer counters like ``num_batches_tracked``. The caller
(:meth:`~cafl4ds.federated.orchestrator.FederatedOrchestrator._apply_server_step`) therefore
passes in the parameter subset and keeps buffers at the plain weighted mean that
``federated_average`` already produced.

That split is the caller's job because only it holds a model, and ``named_parameters()`` is the
one exact way to tell a parameter from a buffer — dtype cannot, since running statistics are
floats too. Getting it wrong is not cosmetic: an adaptive step moves every coordinate by about
``lr`` regardless of how little the clients changed it, so a running variance drifts at a rate
set by the *hyperparameters* rather than by the data, and can cross zero into ``NaN``.

Implementations here still skip non-float tensors as a second line of defence, and always return
the full key set they were handed, so a result can go straight to ``load_state_dict`` (which is
strict by default).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import torch
from loguru import logger

from cafl4ds.federated.aggregate import StateDict


class ServerOptimizer(Protocol):
    """Applies one round's aggregated client update to produce the next global model.

    Structural (``Protocol``): any object with a matching :meth:`step` is a server optimizer.
    Implementations are stateful across rounds in general (the adaptive ones carry moment
    estimates), so one instance belongs to one run.
    """

    def step(self, old_state: StateDict, aggregated_state: StateDict) -> StateDict:
        """Produce the next global ``state_dict`` for one round.

        Args:
            old_state: The global weights broadcast at the start of the round.
            aggregated_state: The aggregated client weights for the round, typically from
                :func:`~cafl4ds.federated.aggregate.federated_average`.

        Returns:
            The next global ``state_dict``, with the same key set as the inputs.
        """
        ...

    @property
    def name(self) -> str:
        """Short algorithm name, for run logs and round-loop banners."""
        ...


class FedAvgServer:
    """FedAvg: take the aggregate as-is (McMahan et al. 2017).

    The identity server step, and the Phase-0 baseline. Equivalently, SGD on the pseudo-gradient
    with a learning rate of exactly 1.0 — the server accepts the clients' step whole, keeping no
    memory between rounds.
    """

    @property
    def name(self) -> str:
        """Short algorithm name, for run logs and round-loop banners."""
        return "fedavg"

    def step(self, old_state: StateDict, aggregated_state: StateDict) -> StateDict:
        """Return ``aggregated_state`` unchanged.

        Args:
            old_state: Ignored; FedAvg keeps no memory of the previous global model.
            aggregated_state: The aggregated client weights, which *are* the next global model.

        Returns:
            ``aggregated_state``, unmodified.
        """
        del old_state  # FedAvg is memoryless by construction
        return aggregated_state


class AdaptiveServerOptimizer(ABC):
    """Shared skeleton for the FedOpt adaptive servers (Reddi et al. 2021, Algorithm 2).

    Runs one adaptive optimizer step per round on the pseudo-gradient
    ``delta = avg_i(client weights) - x`` (see the module docstring):

    .. code-block:: text

        m_t = beta_1 * m_{t-1} + (1 - beta_1) * delta      # first moment (shared)
        v_t = <subclass rule>(v_{t-1}, delta)              # second moment (per-algorithm)
        x_{t+1} = x_t + lr * m_hat / (sqrt(v_hat) + tau)

    Moment state is one tensor per floating-point parameter, allocated lazily on the first round
    (the key set is not known until a ``state_dict`` arrives).

    Bias correction is **optional and on by default**, which departs from the published
    algorithm — Reddi et al. omit it. The default is deliberate: uncorrected moments are badly
    underestimated for their first ``~1/(1 - beta)`` updates, which is a real fraction of a
    federated run measured in *rounds* rather than steps. Set ``bias_correction=False`` to
    reproduce the paper exactly. Note that it interacts with ``tau``: correction inflates the early
    ``sqrt(v)``, changing how much ``tau`` dominates in the opening rounds.
    """

    def __init__(
        self,
        lr: float = 1.0e-2,
        beta_1: float = 0.9,
        tau: float = 1.0e-3,
        bias_correction: bool = True,
        v_init: float = 0.0,
    ) -> None:
        """Configure the shared adaptive-server hyperparameters.

        Args:
            lr: Server learning rate. Roughly the per-coordinate weight-space step
                per round, since the update is normalized — unrelated to the client learning
                rate in ``configs/optim/``, and typically much larger in nominal terms.
            beta_1: First-moment (server momentum) decay in ``[0, 1)``.
            tau: Denominator constant, the degree of adaptivity. Larger values are
                *less* adaptive, approaching plain server momentum; see the module docstring on
                why a local-Adam ``1e-8`` is the wrong scale here.
            bias_correction: Whether to debias the moment estimates. ``False`` matches the
                published FedOpt algorithms.
            v_init: Constant the second moment starts at. Zero suits the EMA accumulator in
                :class:`FedAdamServer`; the accumulate-only rules (:class:`FedYogiServer`,
                :class:`FedAdagradServer`) can benefit from a small positive value, since they
                have no decay to shrink an over-large early estimate.

        Raises:
            ValueError: If ``lr`` or ``tau`` is non-positive, ``beta_1`` is outside ``[0, 1)``,
                or ``v_init`` is negative.
        """
        if lr <= 0.0:
            raise ValueError(f"server lr must be > 0; got {lr}.")
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError(f"beta_1 must be in [0, 1); got {beta_1}.")
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0; got {tau}.")
        if v_init < 0.0:
            raise ValueError(f"v_init must be >= 0; got {v_init}.")
        self.lr = lr
        self.beta_1 = beta_1
        self.tau = tau
        self.bias_correction = bias_correction
        self.v_init = v_init
        self.round = 0  # completed steps; the bias-correction exponent t
        self.m: StateDict = {}
        self.v: StateDict = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Short algorithm name, for run logs and round-loop banners."""

    @abstractmethod
    def _second_moment(self, v: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Accumulate one parameter's second moment — the only per-algorithm arithmetic.

        Args:
            v: The previous second-moment estimate for this parameter.
            delta: This round's pseudo-gradient for this parameter.

        Returns:
            The updated second-moment estimate.
        """

    def _debias_second_moment(self, v: torch.Tensor) -> torch.Tensor:
        """Debias the second moment. Default: identity.

        Only an exponential moving average is biased towards its initialization, so the
        accumulate-only rules (Yogi, Adagrad) leave this alone and
        :class:`FedAdamServer` overrides it.

        Args:
            v: The raw second-moment estimate.

        Returns:
            The debiased estimate.
        """
        return v

    def step(self, old_state: StateDict, aggregated_state: StateDict) -> StateDict:
        """Apply one adaptive server step to the round's aggregate.

        Args:
            old_state: The global weights broadcast at the start of the round.
            aggregated_state: The aggregated client weights for the round.

        Returns:
            The next global ``state_dict``: adapted floating-point parameters, plus every
            non-float buffer carried through from ``aggregated_state`` unchanged.

        Raises:
            ValueError: If the two states have different key sets.
        """
        if old_state.keys() != aggregated_state.keys():
            raise ValueError("old/aggregated state key sets differ; cannot apply a server step.")
        self.round += 1
        updated: StateDict = {}
        for key, aggregated in aggregated_state.items():
            if not aggregated.is_floating_point():
                updated[key] = aggregated  # integer bookkeeping buffer, not a parameter
                continue
            previous = old_state[key]
            delta = aggregated - previous
            if key not in self.m:  # lazy allocation: the key set arrives with the first round
                self.m[key] = torch.zeros_like(previous)
                self.v[key] = torch.full_like(previous, self.v_init)
            self.m[key] = self.beta_1 * self.m[key] + (1.0 - self.beta_1) * delta
            self.v[key] = self._second_moment(self.v[key], delta)
            m_hat = self.m[key]
            if self.bias_correction:
                m_hat = m_hat / (1.0 - self.beta_1**self.round)
            v_hat = self._debias_second_moment(self.v[key])
            updated[key] = previous + self.lr * m_hat / (v_hat.sqrt() + self.tau)
        logger.debug(f"{self.name} server step {self.round}: lr={self.lr}, tau={self.tau}")
        return updated


class FedAdamServer(AdaptiveServerOptimizer):
    """FedAdam: an exponential moving average of the squared pseudo-gradient.

    The second moment forgets at rate ``beta_2``, so the per-coordinate scale tracks *recent*
    rounds. This is the variant to reach for first.

    On ``beta_2``: the local-Adam default of ``0.999`` implies a memory of ~1000 updates, which
    never warms up over a federated run of ~100 *rounds*. The default here is ``0.99``.
    """

    def __init__(
        self,
        lr: float = 1.0e-2,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1.0e-3,
        bias_correction: bool = True,
        v_init: float = 0.0,
    ) -> None:
        """Configure FedAdam.

        Args:
            lr: Server learning rate. See :class:`AdaptiveServerOptimizer`.
            beta_1: First-moment decay in ``[0, 1)``.
            beta_2: Second-moment decay in ``[0, 1)``; the horizon is ~``1/(1 - beta_2)`` rounds.
            tau: Degree of adaptivity. See :class:`AdaptiveServerOptimizer`.
            bias_correction: Whether to debias both moments.
            v_init: Second-moment initialization; ``0.0`` is standard for an EMA.

        Raises:
            ValueError: If ``beta_2`` is outside ``[0, 1)``, or per the base class.
        """
        super().__init__(lr=lr, beta_1=beta_1, tau=tau, bias_correction=bias_correction, v_init=v_init)
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError(f"beta_2 must be in [0, 1); got {beta_2}.")
        self.beta_2 = beta_2

    @property
    def name(self) -> str:
        """Short algorithm name, for run logs and round-loop banners."""
        return "fedadam"

    def _second_moment(self, v: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """EMA of the squared pseudo-gradient: ``beta_2 * v + (1 - beta_2) * delta^2``."""
        return self.beta_2 * v + (1.0 - self.beta_2) * delta * delta

    def _debias_second_moment(self, v: torch.Tensor) -> torch.Tensor:
        """Divide out the EMA's bias towards its zero initialization."""
        if not self.bias_correction:
            return v
        return v / (1.0 - self.beta_2**self.round)


class FedYogiServer(AdaptiveServerOptimizer):
    """FedYogi: additive second-moment updates, so the scale cannot collapse.

    Adam's multiplicative EMA can shrink ``v`` fast when a few small deltas arrive, and a
    small denominator means a large step — the classic Adam instability. Yogi (Zaheer et al. 2018)
    changes ``v`` by an amount proportional to ``delta^2`` rather than to ``v`` itself, so it
    *decreases* only gently:

    .. code-block:: text

        v_t = v_{t-1} - (1 - beta_2) * delta^2 * sign(v_{t-1} - delta^2)

    Worth trying when FedAdam shows round-to-round spikes. Because the rule is additive rather
    than a true EMA, no second-moment debiasing is applied; a small positive ``v_init`` is the
    usual way to keep the first few rounds from over-stepping.
    """

    def __init__(
        self,
        lr: float = 1.0e-2,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1.0e-3,
        bias_correction: bool = True,
        v_init: float = 1.0e-6,
    ) -> None:
        """Configure FedYogi.

        Args:
            lr: Server learning rate. See :class:`AdaptiveServerOptimizer`.
            beta_1: First-moment decay in ``[0, 1)``.
            beta_2: Second-moment step size in ``[0, 1)``.
            tau: Degree of adaptivity. See :class:`AdaptiveServerOptimizer`.
            bias_correction: Whether to debias the *first* moment (the second is left as-is).
            v_init: Second-moment initialization; a small positive value by default.

        Raises:
            ValueError: If ``beta_2`` is outside ``[0, 1)``, or per the base class.
        """
        super().__init__(lr=lr, beta_1=beta_1, tau=tau, bias_correction=bias_correction, v_init=v_init)
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError(f"beta_2 must be in [0, 1); got {beta_2}.")
        self.beta_2 = beta_2

    @property
    def name(self) -> str:
        """Short algorithm name, for run logs and round-loop banners."""
        return "fedyogi"

    def _second_moment(self, v: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Yogi's sign-gated additive update — grows like Adam, shrinks gently."""
        squared = delta * delta
        return v - (1.0 - self.beta_2) * squared * torch.sign(v - squared)


class FedAdagradServer(AdaptiveServerOptimizer):
    """FedAdagrad: a running *sum* of squared pseudo-gradients, so the step only ever decays.

    With no decay term, ``v`` grows monotonically and the effective per-coordinate step
    shrinks over the run — a built-in annealing schedule. That suits a fixed-horizon run and
    sparse, rarely-updated coordinates; it is the wrong choice for a long run whose gradient
    scale shifts, since the early rounds permanently set the denominator.

    ``beta_1`` still applies to the first moment. No second-moment debiasing: the accumulator is
    a sum, not an EMA, so it has no initialization bias to remove.
    """

    @property
    def name(self) -> str:
        """Short algorithm name, for run logs and round-loop banners."""
        return "fedadagrad"

    def _second_moment(self, v: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Accumulate squared pseudo-gradients: ``v + delta^2``."""
        return v + delta * delta
