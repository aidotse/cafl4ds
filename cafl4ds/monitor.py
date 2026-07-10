"""The health monitor — the *dependent variable* (the thermometer).

The monitor periodically reads the SSL backbone with the Phase-0 instruments
(:mod:`cafl4ds.measurements`) on a **fixed** held-out probe set, and returns a flat
metric dictionary the loop logs as the ``health`` series. In Phase 0 it is a pure readout;
in Phase 3 the same signal drives the monitor→filter controller (the slow edge), so the
measurement surface is deliberately kept separate from any action.

Metrics reported:

* ``rankme`` — effective rank of the probe embeddings (the core collapse readout).
* ``cka_drift`` / ``cosine_drift`` — representation drift of the fixed probe set vs. its
  first-checkpoint embeddings (content drift and coordinate-frame churn).
* ``uniformity`` / ``offdiag_cov`` / ``mean_feature_var`` — extra label-free geometry.
* ``knn_acc`` / ``linear_acc`` — downstream probe accuracy on frozen features (labels used
  HERE ONLY).

Drift is undefined at the first checkpoint (no reference yet), so it is reported as ``0.0``
there and against the stored ``t0`` embeddings thereafter.
"""

from __future__ import annotations

import torch

from cafl4ds import measurements
from cafl4ds.data.streams import EvalSets
from cafl4ds.ssl.base import SSLMethod


class HealthMonitor:
    """Runs the representation-health instruments on a fixed held-out probe set."""

    def __init__(
        self,
        eval_sets: EvalSets,
        knn_k: int = 20,
        run_knn: bool = True,
        run_linear: bool = True,
    ) -> None:
        """Configure the monitor.

        Args:
            eval_sets: The stream's held-out eval sets (probe support/query, per-era).
            knn_k: Number of neighbours for the kNN probe.
            run_knn: Whether to compute the kNN probe (labels used HERE ONLY).
            run_linear: Whether to compute the linear probe (labels used HERE ONLY).
        """
        self.eval_sets = eval_sets
        self.knn_k = knn_k
        self.run_knn = run_knn
        self.run_linear = run_linear
        self._z_ref0: torch.Tensor | None = None

    def measure(self, method: SSLMethod, step: int) -> dict[str, float]:
        """Compute the health metrics for the current model state.

        Args:
            method: The live SSL method (its encoder supplies the frozen embedding).
            step: The current global step (recorded in the returned dict).

        Returns:
            A flat ``metric -> value`` dictionary (all Python floats).
        """
        was_training = method.training
        method.eval()
        try:
            query = self.eval_sets.probe_query
            z_query = method.encode(query.images)  # [M, d], no grad
            metrics: dict[str, float] = {
                "step": float(step),
                "rankme": measurements.rankme(z_query),
                "uniformity": measurements.uniformity(z_query),
                "offdiag_cov": measurements.offdiag_covariance(z_query),
                "mean_feature_var": float(measurements.feature_variance(z_query).mean().item()),
            }
            metrics.update(self._drift(z_query))
            if self.run_knn:
                metrics["knn_acc"] = measurements.knn_probe(
                    method.encode,
                    (self.eval_sets.probe_support.images, self.eval_sets.probe_support.labels),
                    (query.images, query.labels),
                    k=self.knn_k,
                )
            if self.run_linear:
                metrics["linear_acc"] = measurements.linear_probe(
                    method.encode,
                    (self.eval_sets.probe_support.images, self.eval_sets.probe_support.labels),
                    (query.images, query.labels),
                )
            return metrics
        finally:
            method.train(was_training)

    def _drift(self, z_query: torch.Tensor) -> dict[str, float]:
        """Compute drift of the fixed probe set vs. its first-checkpoint embeddings.

        The first call stores the reference embeddings and reports zero drift; later calls
        compare against that stored reference.

        Args:
            z_query: Current embeddings of the fixed probe-query set.

        Returns:
            ``{"cka_drift": ..., "cosine_drift": ...}``.
        """
        if self._z_ref0 is None:
            self._z_ref0 = z_query.clone()
            return {"cka_drift": 0.0, "cosine_drift": 0.0}
        return {
            "cka_drift": measurements.cka_drift(self._z_ref0, z_query),
            "cosine_drift": measurements.cosine_drift(self._z_ref0, z_query),
        }
