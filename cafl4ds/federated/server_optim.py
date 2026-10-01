"""Server optimizers — how the server *applies* an aggregated client update (FedOpt).

The second server-side step, after :mod:`~cafl4ds.federated.aggregate`. FedOpt (Reddi et al.
2021) treats the round as one step of an optimizer that lives on the server, driven by the
**pseudo-gradient** ``delta = avg_i(client weights) - x``, where ``x`` is the broadcast global
model. Because every client starts from the same ``x``, differencing the averaged weights equals
averaging the client deltas. FedAvg is SGD on ``delta`` with learning rate 1; the adaptive
variants run Adam / Yogi / Adagrad on it instead, damping coordinates the clients disagree on.

Two hyperparameters mean something different from local Adam:

* ``lr`` is the *server* rate, unrelated to the client rate in ``configs/optim/``. Because the
  update is normalized, it is roughly the per-coordinate step in weight space per round.
* ``tau`` sits where Adam puts ``eps`` but is a real knob. With ``eps``-sized ``1e-8`` every
  coordinate would move by about ``±lr`` (sign-SGD); ``~1e-3`` lets small deltas stay small.

Server optimizers receive **parameters only**: the orchestrator keeps buffers (e.g. BatchNorm
running statistics) at their plain weighted mean, since an adaptive step would move them by
about ``lr`` regardless of the data and can push a running variance below zero. As a second line
of defence, non-float tensors are skipped here too.

To add an adaptive variant, subclass :class:`AdaptiveServerOptimizer` and implement
:meth:`~AdaptiveServerOptimizer._second_moment`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import torch
from loguru import logger

from cafl4ds.federated.aggregate import StateDict


class ServerOptimizer(Protocol):
    """Applies one round's aggregated client update to produce the next global model.

    Structural: any object with a matching :meth:`step` and :attr:`name` qualifies. Stateful in
    general, so one instance belongs to one run.
    """

    def step(self, old_state: StateDict, aggregated_state: StateDict) -> StateDict:
        """Produce the next global ``state_dict`` for one round.

        Args:
            old_state: The global weights broadcast at the start of the round.
            aggregated_state: The round's aggregated client weights.

        Returns:
            The next global ``state_dict``, with the same key set as the inputs.
        """
        ...

    @property
    def name(self) -> str:
        """Short algorithm name, for run logs."""
        ...


class FedAvgServer:
    """FedAvg (McMahan et al. 2017): take the aggregate as-is — the identity server step."""

    @property
    def name(self) -> str:
        """Short algorithm name, for run logs."""
        return "fedavg"

    def step(self, old_state: StateDict, aggregated_state: StateDict) -> StateDict:
        """Return ``aggregated_state`` unchanged.

        Args:
            old_state: Ignored.
            aggregated_state: The aggregated client weights, which *are* the next global model.

        Returns:
            ``aggregated_state``, unmodified.
        """
        del old_state  # FedAvg is memoryless by construction
        return aggregated_state


class AdaptiveServerOptimizer(ABC):
    """Shared skeleton for the FedOpt adaptive servers (Reddi et al. 2021, Algorithm 2).

    .. code-block:: text

        m_t = beta_1 * m_{t-1} + (1 - beta_1) * delta      # first moment (shared)
        v_t = <subclass rule>(v_{t-1}, delta)              # second moment (per-algorithm)
        x_{t+1} = x_t + lr * m_hat / (sqrt(v_hat) + tau)

    Moment tensors are allocated lazily on the first round. Bias correction is on by default,
    which departs from the paper: uncorrected moments are underestimated for ~``1/(1 - beta)``
    updates, a real fraction of a run measured in rounds. Set ``bias_correction=False`` to
    reproduce the paper exactly.
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
            lr: Server learning rate (see the module docstring).
            beta_1: First-moment (server momentum) decay in ``[0, 1)``.
            tau: Degree of adaptivity; larger is less adaptive (see the module docstring).
            bias_correction: Whether to debias the moment estimates.
            v_init: Initial value of the second moment.

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
        """Short algorithm name, for run logs."""

    @abstractmethod
    def _second_moment(self, v: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Accumulate one parameter's second moment — the only per-algorithm arithmetic.

        Args:
            v: The previous second-moment estimate.
            delta: This round's pseudo-gradient.

        Returns:
            The updated second-moment estimate.
        """

    def _debias_second_moment(self, v: torch.Tensor) -> torch.Tensor:
        """Debias the second moment. Identity by default; only an EMA needs it (FedAdam)."""
        return v

    def step(self, old_state: StateDict, aggregated_state: StateDict) -> StateDict:
        """Apply one adaptive server step to the round's aggregate.

        Args:
            old_state: The global weights broadcast at the start of the round.
            aggregated_state: The round's aggregated client weights.

        Returns:
            The next global ``state_dict``: stepped float tensors, plus non-float tensors carried
            through unchanged.

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
    """FedAdam: an EMA of the squared pseudo-gradient. The variant to reach for first.

    ``beta_2`` defaults to ``0.99`` rather than local Adam's ``0.999``, whose ~1000-update memory
    never warms up over a run of ~100 rounds.
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
            lr: Server learning rate.
            beta_1: First-moment decay in ``[0, 1)``.
            beta_2: Second-moment decay in ``[0, 1)``; the horizon is ~``1/(1 - beta_2)`` rounds.
            tau: Degree of adaptivity.
            bias_correction: Whether to debias both moments.
            v_init: Initial value of the second moment.

        Raises:
            ValueError: If ``beta_2`` is outside ``[0, 1)``, or per the base class.
        """
        super().__init__(lr=lr, beta_1=beta_1, tau=tau, bias_correction=bias_correction, v_init=v_init)
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError(f"beta_2 must be in [0, 1); got {beta_2}.")
        self.beta_2 = beta_2

    @property
    def name(self) -> str:
        """Short algorithm name, for run logs."""
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
    """FedYogi (Zaheer et al. 2018): an additive second moment that grows like Adam's but shrinks gently.

    .. code-block:: text

        v_t = v_{t-1} - (1 - beta_2) * delta^2 * sign(v_{t-1} - delta^2)

    Worth trying when FedAdam shows round-to-round spikes. Not an EMA, so the second moment is
    not debiased; a small positive ``v_init`` keeps the first rounds from over-stepping.
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
            lr: Server learning rate.
            beta_1: First-moment decay in ``[0, 1)``.
            beta_2: Second-moment step size in ``[0, 1)``.
            tau: Degree of adaptivity.
            bias_correction: Whether to debias the first moment.
            v_init: Initial value of the second moment.

        Raises:
            ValueError: If ``beta_2`` is outside ``[0, 1)``, or per the base class.
        """
        super().__init__(lr=lr, beta_1=beta_1, tau=tau, bias_correction=bias_correction, v_init=v_init)
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError(f"beta_2 must be in [0, 1); got {beta_2}.")
        self.beta_2 = beta_2

    @property
    def name(self) -> str:
        """Short algorithm name, for run logs."""
        return "fedyogi"

    def _second_moment(self, v: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Yogi's sign-gated additive update."""
        squared = delta * delta
        return v - (1.0 - self.beta_2) * squared * torch.sign(v - squared)


class FedAdagradServer(AdaptiveServerOptimizer):
    """FedAdagrad: a running *sum* of squared pseudo-gradients, so the step only ever shrinks.

    A built-in annealing schedule: suits a fixed-horizon run, but the early rounds permanently set
    the denominator. Not an EMA, so the second moment is not debiased.
    """

    @property
    def name(self) -> str:
        """Short algorithm name, for run logs."""
        return "fedadagrad"

    def _second_moment(self, v: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Accumulate squared pseudo-gradients: ``v + delta^2``."""
        return v + delta * delta
