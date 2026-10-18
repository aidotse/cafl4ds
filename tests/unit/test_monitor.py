"""Tests for the health monitor.

Confirms the monitor reports the expected metric surface, that drift is zero at the first
checkpoint and becomes positive once the representation moves, and that measuring never
leaves the method in eval mode (it must restore training).
"""

import copy
from dataclasses import replace

import torch
from loguru import logger

from cafl4ds.data.sources import SyntheticSource
from cafl4ds.data.streams import EraStream
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.monitor import HealthMonitor
from cafl4ds.ssl.factory import build_mae, build_simsiam

# MAE exposes only the backbone surface (no projector → no `_proj` keys), but it now returns a
# positive pair from `make_views` (two augmentation draws, for the P0.5 alignment-on-trial read),
# so `alignment` is reported at the backbone.
_EXPECTED_KEYS = {
    "step",
    "rankme",
    "uniformity",
    "alignment",
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


def _simsiam_and_monitor() -> tuple[object, HealthMonitor]:
    """Build a tiny SimSiam (two surfaces + positive pair) and a monitor over its eval sets."""
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    method = build_simsiam(encoder, proj_hidden=64, proj_dim=32, pred_hidden=32)
    stream = EraStream(
        SyntheticSource(num_classes=3, per_class=40, img_size=16),
        support_per_class=8,
        query_per_class=8,
        era_eval_per_class=5,
    )
    return method, HealthMonitor(stream.eval_sets, knn_k=5)


def test_joint_embedding_reports_both_surfaces_and_alignment() -> None:
    """SimSiam backone and projector surfaces.

    SimSiam exposes backbone + projector surfaces and a positive pair, so the monitor emits
    the geometry suite at both surfaces (``_proj`` keys) plus alignment at both.
    """
    method, monitor = _simsiam_and_monitor()
    metrics = monitor.measure(method, step=0)  # type: ignore[arg-type]
    surfaced = {"rankme", "mean_feature_var", "offdiag_cov", "uniformity", "alignment"}
    expected = {"step", "cka_drift", "cosine_drift", "knn_acc", "linear_acc"}
    expected |= surfaced | {f"{m}_proj" for m in surfaced}
    assert set(metrics) == expected
    assert all(isinstance(v, float) and v == v for v in metrics.values())  # all finite


def test_drift_surfaces_adds_projector_drift() -> None:
    """P0.6.1: ``drift_surfaces`` tracks drift at the projector too, keeping the backbone keys.

    By default drift is backbone-only (unsuffixed ``cka_drift`` / ``cosine_drift``). With
    ``drift_surfaces=True`` a joint-embedding method additionally reports ``cka_drift_proj`` /
    ``cosine_drift_proj`` — the past-data-free reader's second surface — and the backbone keys are
    unchanged. Both are zero at the first checkpoint (no reference yet).
    """
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    method = build_simsiam(encoder, proj_hidden=64, proj_dim=32, pred_hidden=32)
    stream = EraStream(
        SyntheticSource(num_classes=3, per_class=40, img_size=16),
        support_per_class=8,
        query_per_class=8,
        era_eval_per_class=5,
    )
    monitor = HealthMonitor(stream.eval_sets, knn_k=5, run_knn=False, run_linear=False, drift_surfaces=True)
    first = monitor.measure(method, step=0)
    assert {"cka_drift", "cosine_drift", "cka_drift_proj", "cosine_drift_proj"} <= set(first)
    assert first["cka_drift"] == 0.0 and first["cka_drift_proj"] == 0.0  # both surfaces zero at t0

    opt = torch.optim.AdamW(method.parameters(), lr=1e-2)
    x = SyntheticSource(num_classes=3, per_class=8, img_size=16).load()[0]
    for _ in range(10):
        opt.zero_grad()
        loss = method.training_step(x)
        loss.backward()
        opt.step()

    later = monitor.measure(method, step=10)
    assert later["cosine_drift"] > 0.0 and later["cosine_drift_proj"] > 0.0  # both surfaces move


def test_alignment_view_pair_is_fixed_across_checkpoints() -> None:
    """Alignment view across checkpoints.

    The alignment view-pair is drawn once and reused, so it is identical across calls and
    does not consume the global RNG (two measurements draw the same pair).
    """
    method, monitor = _simsiam_and_monitor()
    monitor.measure(method, step=0)  # type: ignore[arg-type]
    first = monitor._views  # cached pair
    assert first is not None
    monitor.measure(method, step=1)  # type: ignore[arg-type]
    assert monitor._views is first  # same cached object, not re-drawn


def test_p052_readers_reported_when_enabled_for_mae() -> None:
    """Enabling the P0.5.2 candidate readers adds their keys for an MAE backbone.

    Clusterability, mean attention distance, and alignment-under-stronger-aug are off by default
    (so existing runs are unchanged); switching them on must emit exactly those three extra finite
    keys and nothing else.
    """
    method, _ = _method_and_monitor()
    stream = EraStream(
        SyntheticSource(num_classes=3, per_class=40, img_size=16),
        support_per_class=8,
        query_per_class=8,
        era_eval_per_class=5,
    )
    monitor = HealthMonitor(
        stream.eval_sets,
        knn_k=5,
        run_clusterability=True,
        run_attn_distance=True,
        run_alignment_strong=True,
    )
    metrics = monitor.measure(method, step=0)  # type: ignore[arg-type]
    extra = {"clusterability", "attn_distance", "alignment_strong"}
    assert extra <= set(metrics)
    assert set(metrics) == _EXPECTED_KEYS | extra
    assert all(isinstance(metrics[k], float) and metrics[k] == metrics[k] for k in extra)


def _perturb(method: object) -> None:
    """Move the encoder's weights (a stand-in for training) so its representation changes."""
    with torch.no_grad():
        for p in method.encoder.parameters():  # type: ignore[attr-defined]
            p.add_(0.05 * torch.randn_like(p))


def test_anchor_drift_to_the_same_model_reads_zero_then_tracks_movement() -> None:
    """Anchored to a frozen copy of itself, the first read is 0 — and drift grows once the model moves."""
    method, monitor = _method_and_monitor()
    monitor.anchor_drift(copy.deepcopy(method))  # type: ignore[arg-type]
    first = monitor.measure(method, step=0)  # type: ignore[arg-type]
    assert abs(first["cka_drift"]) < 1e-5 and abs(first["cosine_drift"]) < 1e-5
    _perturb(method)
    later = monitor.measure(method, step=1)  # type: ignore[arg-type]
    assert later["cka_drift"] > 1e-4 and later["cosine_drift"] > 1e-4


def test_anchor_drift_reads_nonzero_at_the_first_checkpoint_against_another_model() -> None:
    """Anchored to a *different* model, drift is non-zero from the very first checkpoint (no self-reset)."""
    method, monitor = _method_and_monitor()
    reference = copy.deepcopy(method)
    _perturb(reference)
    monitor.anchor_drift(reference)  # type: ignore[arg-type]
    first = monitor.measure(method, step=0)  # type: ignore[arg-type]
    assert first["cka_drift"] > 1e-4 and first["cosine_drift"] > 1e-4


def test_anchor_drift_restores_the_reference_training_mode() -> None:
    """Anchoring reads the reference in eval mode and puts it back the way it was."""
    method, monitor = _method_and_monitor()
    reference = copy.deepcopy(method)
    reference.train()  # type: ignore[attr-defined]
    monitor.anchor_drift(reference)  # type: ignore[arg-type]
    assert reference.training  # type: ignore[attr-defined]


def test_anchor_without_a_drift_surface_warns_and_falls_back_to_the_first_checkpoint() -> None:
    """An anchor lacking a surface the run tracks (a backbone-only MAE under a JE run) is flagged."""
    _, mae_monitor = _method_and_monitor()
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    live = build_simsiam(encoder=encoder, proj_hidden=32, proj_dim=16, pred_hidden=16)
    monitor = HealthMonitor(mae_monitor.eval_sets, knn_k=5, drift_surfaces=True)
    monitor.anchor_drift(build_mae(TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)))
    warnings: list[str] = []
    sink = logger.add(warnings.append, level="WARNING")
    try:
        first = monitor.measure(live, step=0)
    finally:
        logger.remove(sink)
    assert any("no 'proj' surface" in w for w in warnings)
    assert first["cka_drift_proj"] == 0.0 and first["cka_drift"] > 1e-4


def test_grouped_probe_query_reports_per_group_accuracies() -> None:
    """A grouped probe-query adds ``<probe>_<group>`` accuracies whose size-weighted mean is the pooled one."""
    method, monitor = _method_and_monitor()
    ev = monitor.eval_sets
    groups = torch.arange(ev.probe_query.labels.numel()) % 2  # two arbitrary query groups
    grouped = replace(ev, probe_query=replace(ev.probe_query, groups=groups), group_names={0: "dawn/dusk", 1: "night"})
    metrics = HealthMonitor(grouped, knn_k=5).measure(method, step=0)  # type: ignore[arg-type]
    n0, n1 = int((groups == 0).sum()), int((groups == 1).sum())
    for probe in ("knn_acc", "linear_acc"):
        assert {f"{probe}_dawn_dusk", f"{probe}_night"} <= set(metrics)
        weighted = (n0 * metrics[f"{probe}_dawn_dusk"] + n1 * metrics[f"{probe}_night"]) / (n0 + n1)
        assert abs(weighted - metrics[probe]) < 1e-9
