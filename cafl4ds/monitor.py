"""The health monitor — the *dependent variable* (the thermometer).

The monitor periodically reads the SSL backbone with the Phase-0 instruments
(:mod:`cafl4ds.measurements`) on a **fixed** held-out probe set, and returns a flat
metric dictionary the loop logs as the ``health`` series. In Phase 0 it is a pure readout;
in Phase 3 the same signal drives the monitor→filter controller (the slow edge), so the
measurement surface is deliberately kept separate from any action.

Metrics reported (P0.2.2 reads the geometry suite at **every** surface the method exposes via
:meth:`~cafl4ds.ssl.base.SSLMethod.embedding_surfaces` — ``"backbone"`` always, plus ``"proj"``
for joint-embedding methods; the projector-surface metrics carry a ``_proj`` suffix):

* ``rankme`` — effective rank of the probe embeddings (the core collapse readout).
* ``mean_feature_var`` — mean per-dimension variance (VICReg variance term); read on
  **L2-normalized** embeddings so it measures directional spread, not raw scale (P0.2.2).
* ``offdiag_cov`` — mean squared off-diagonal covariance (VICReg redundancy term); likewise on
  L2-normalized embeddings.
* ``uniformity`` — spread on the hypersphere (Wang & Isola); L2-normalizes internally.
* ``alignment`` — positive-pair closeness (Wang & Isola); needs a positive pair, so it is
  reported only for methods that expose one (skipped for MAE). The pair is a **fixed** pair of
  augmented views of the probe set, drawn once and reused across checkpoints.
* ``cka_drift`` / ``cosine_drift`` — representation drift of the fixed probe set vs. its
  first-checkpoint (backbone) embeddings (content drift and coordinate-frame churn).
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


def _surface_suffix(name: str) -> str:
    """Metric-key suffix for a surface: none for ``backbone``, ``_<name>`` otherwise."""
    return "" if name == "backbone" else f"_{name}"


class HealthMonitor:
    """Runs the representation-health instruments on a fixed held-out probe set."""

    def __init__(
        self,
        eval_sets: EvalSets,
        knn_k: int = 20,
        run_knn: bool = True,
        run_linear: bool = True,
        run_alignment: bool = True,
        align_seed: int = 0,
    ) -> None:
        """Configure the monitor.

        Args:
            eval_sets: The stream's held-out eval sets (probe support/query, per-era).
            knn_k: Number of neighbours for the kNN probe.
            run_knn: Whether to compute the kNN probe (labels used HERE ONLY).
            run_linear: Whether to compute the linear probe (labels used HERE ONLY).
            run_alignment: Whether to compute alignment (needs a positive pair; auto-skipped
                for methods that expose none).
            align_seed: Seed used to draw the fixed alignment view-pair (deterministic and
                isolated from the training RNG), so alignment is comparable across checkpoints.
        """
        self.eval_sets = eval_sets
        self.knn_k = knn_k
        self.run_knn = run_knn
        self.run_linear = run_linear
        self.run_alignment = run_alignment
        self.align_seed = align_seed
        self._z_ref0: torch.Tensor | None = None
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None
        self._views_cached = False

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
            surfaces = method.embedding_surfaces(query.images)  # {"backbone": [M, d], ...}
            metrics: dict[str, float] = {"step": float(step)}
            for name, z in surfaces.items():
                metrics.update(self._geometry(name, z))
            # Drift is tracked on the backbone surface only — the representation under study,
            # matching the P0.2.1 RankMe calibration reference.
            metrics.update(self._drift(surfaces["backbone"]))
            if self.run_alignment:
                metrics.update(self._alignment(method))
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

    def _geometry(self, name: str, z: torch.Tensor) -> dict[str, float]:
        """Collapse-geometry instruments for one embedding surface.

        RankMe is read on the raw embedding (scale-invariant; exactly the P0.2.1 quantity). The
        VICReg variance/covariance terms are read on the **L2-normalized** embedding, so they
        measure directional collapse rather than a raw change of scale (a representation can
        shrink in norm without collapsing) — the normalization the P0.2.2 calibration uses.
        Uniformity L2-normalizes internally.

        Args:
            name: Surface name (``"backbone"``, ``"proj"``, …).
            z: The surface embedding ``[M, d]``.

        Returns:
            ``{metric+suffix: value}`` for this surface.
        """
        s = _surface_suffix(name)
        z_unit = torch.nn.functional.normalize(z, dim=1)
        return {
            f"rankme{s}": measurements.rankme(z),
            f"mean_feature_var{s}": float(measurements.feature_variance(z_unit).mean().item()),
            f"offdiag_cov{s}": measurements.offdiag_covariance(z_unit),
            f"uniformity{s}": measurements.uniformity(z),
        }

    def _alignment(self, method: SSLMethod) -> dict[str, float]:
        """Positive-pair alignment per surface, on a fixed view-pair of the probe set.

        The pair is drawn once (under an isolated, fixed RNG so it neither varies across
        checkpoints nor perturbs the training RNG) and reused, so the alignment series reflects
        the model moving, not the augmentation resampling. Returns ``{}`` for methods that
        expose no positive pair.

        Args:
            method: The live SSL method.

        Returns:
            ``{"alignment"+suffix: value}`` per surface, or ``{}`` if unsupported.
        """
        views = self._view_pair(method)
        if views is None:
            return {}
        surf_a = method.embedding_surfaces(views[0])
        surf_b = method.embedding_surfaces(views[1])
        return {
            f"alignment{_surface_suffix(name)}": measurements.alignment(surf_a[name], surf_b[name]) for name in surf_a
        }

    def _view_pair(self, method: SSLMethod) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Draw (once) and cache a deterministic positive pair of the probe-query set.

        Uses a fixed seed and restores the global RNG state afterwards, so the pair is stable
        across checkpoints and the draw does not disturb training-order/augmentation RNG.

        Args:
            method: The live SSL method (supplies the augmentation via ``make_views``).

        Returns:
            The cached ``(view_a, view_b)`` pair, or ``None`` if the method exposes none.
        """
        if not self._views_cached:
            rng_state = torch.random.get_rng_state()
            try:
                torch.manual_seed(self.align_seed)
                self._views = method.make_views(self.eval_sets.probe_query.images)
            finally:
                torch.random.set_rng_state(rng_state)
            self._views_cached = True
        return self._views

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
