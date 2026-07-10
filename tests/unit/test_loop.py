"""End-to-end streaming-loop test — the Phase-0 exit criterion in miniature.

A short run must complete and leave a single run log whose health series carries the SSL loss
and the health metrics (rankme, drift, probe) side by side over multiple steps, for both ``C``
backbones. Uses the network-free synthetic source so the test is self-contained.
"""

import torch

from cafl4ds.data.sources import SyntheticSource
from cafl4ds.data.streams import EraStream
from cafl4ds.filters.accept_all import AcceptAll
from cafl4ds.loop import StreamingLoop
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.monitor import HealthMonitor
from cafl4ds.run_log import RunLogger, read_run, tabulate
from cafl4ds.ssl.factory import build_mae, build_simsiam

_HEALTH_KEYS = {"loss", "rankme", "cka_drift", "cosine_drift", "knn_acc", "linear_acc"}


def _run_loop(build: object, log_path: str) -> list[dict[str, object]]:
    """Run a short synthetic streaming loop and return the logged records."""
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    method = build(encoder)  # type: ignore[operator]
    stream = EraStream(
        SyntheticSource(num_classes=3, per_class=48, img_size=16),
        batch_size=12,
        support_per_class=8,
        query_per_class=8,
        era_eval_per_class=5,
    )
    monitor = HealthMonitor(stream.eval_sets, knn_k=5)
    run_logger = RunLogger(log_path, run_name="test")
    loop = StreamingLoop(
        stream=stream,
        method=method,
        optimizer=torch.optim.AdamW(method.parameters(), lr=1e-3),
        selection_filter=AcceptAll(),
        monitor=monitor,
        run_logger=run_logger,
        eval_every=2,
    )
    loop.run()
    return read_run(log_path)


def test_mae_loop_logs_loss_and_health_side_by_side(tmp_path: object) -> None:
    """The MAE run log has a loss series and a health series with all metrics over steps."""
    records = _run_loop(build_mae, str(tmp_path / "mae.jsonl"))  # type: ignore[operator]
    loss = [r for r in records if r["series"] == "loss"]
    health = [r for r in records if r["series"] == "health"]
    assert len(loss) >= 3
    assert len(health) >= 2  # metrics tracked over multiple checkpoints
    for r in health:
        assert _HEALTH_KEYS <= set(r)
    # Drift is defined relative to the first checkpoint.
    assert health[0]["cka_drift"] == 0.0
    assert any(r["cosine_drift"] > 0.0 for r in health[1:])  # type: ignore[operator]


def test_simsiam_loop_runs_and_logs_health(tmp_path: object) -> None:
    """The SimSiam (joint-embedding) backbone runs through the same loop and logs health."""
    records = _run_loop(build_simsiam, str(tmp_path / "simsiam.jsonl"))  # type: ignore[operator]
    health = [r for r in records if r["series"] == "health"]
    assert len(health) >= 2
    assert all(_HEALTH_KEYS <= set(r) for r in health)


def test_tabulate_renders_health_rows(tmp_path: object) -> None:
    """The side-by-side table lists the metric columns and one row per checkpoint."""
    records = _run_loop(build_mae, str(tmp_path / "t.jsonl"))  # type: ignore[operator]
    health = [r for r in records if r["series"] == "health"]
    table = tabulate(health)
    for col in ("loss", "rankme", "cka_drift", "knn_acc"):
        assert col in table
    assert table.count("\n") >= len(health) + 1  # header + separator + rows


def test_tabulate_handles_empty() -> None:
    """Tabulating no records yields a placeholder rather than erroring."""
    assert tabulate([]) == "(no health records)"
