"""Tests for the health monitor.

Confirms the monitor reports the expected metric surface, that drift is zero at the first
checkpoint and becomes positive once the representation moves, and that measuring never
leaves the method in eval mode (it must restore training).
"""

import torch

from cafl4ds.data.sources import SyntheticSource
from cafl4ds.data.streams import EraStream
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.monitor import HealthMonitor
from cafl4ds.ssl.factory import build_mae

_EXPECTED_KEYS = {
    "step",
    "rankme",
    "uniformity",
    "offdiag_cov",
    "mean_feature_var",
    "cka_drift",
    "cosine_drift",
    "knn_acc",
    "linear_acc",
}


def _method_and_monitor() -> tuple[object, HealthMonitor]:
    """Build a tiny method and a monitor over a synthetic stream's eval sets."""
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    method = build_mae(encoder)
    stream = EraStream(
        SyntheticSource(num_classes=3, per_class=40, img_size=16),
        support_per_class=8,
        query_per_class=8,
        era_eval_per_class=5,
    )
    return method, HealthMonitor(stream.eval_sets, knn_k=5)


def test_measure_reports_expected_metrics() -> None:
    """A measurement returns exactly the expected metric keys, all finite floats."""
    method, monitor = _method_and_monitor()
    metrics = monitor.measure(method, step=0)  # type: ignore[arg-type]
    assert set(metrics) == _EXPECTED_KEYS
    assert all(isinstance(v, float) and v == v for v in metrics.values())  # not NaN


def test_first_checkpoint_has_zero_drift_then_positive() -> None:
    """Drift is zero at the first checkpoint and positive after the encoder changes."""
    method, monitor = _method_and_monitor()
    first = monitor.measure(method, step=0)  # type: ignore[arg-type]
    assert first["cka_drift"] == 0.0 and first["cosine_drift"] == 0.0

    opt = torch.optim.AdamW(method.parameters(), lr=1e-2)  # type: ignore[attr-defined]
    x = SyntheticSource(num_classes=3, per_class=8, img_size=16).load()[0]
    for _ in range(10):
        opt.zero_grad()
        loss = method.training_step(x)  # type: ignore[attr-defined]
        loss.backward()
        opt.step()

    later = monitor.measure(method, step=10)  # type: ignore[arg-type]
    assert later["cosine_drift"] > 0.0


def test_measure_restores_training_mode() -> None:
    """The monitor must not leave the method stuck in eval mode."""
    method, monitor = _method_and_monitor()
    method.train()  # type: ignore[attr-defined]
    monitor.measure(method, step=0)  # type: ignore[arg-type]
    assert method.training  # type: ignore[attr-defined]
