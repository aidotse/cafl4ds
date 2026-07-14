"""End-to-end streaming-loop test — the Phase-0 exit criterion in miniature.

A short run must complete and leave a single run log whose health series carries the SSL loss
and the health metrics (rankme, drift, probe) side by side over multiple steps, for both ``C``
backbones. Uses the network-free synthetic source so the test is self-contained.
"""

from pathlib import Path

import torch

from cafl4ds.data.sources import SyntheticSource
from cafl4ds.data.streams import EraStream
from cafl4ds.filters.accept_all import AcceptAll
from cafl4ds.loop import StreamingLoop
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.monitor import HealthMonitor
from cafl4ds.run_log import RunLogger, read_run, tabulate
from cafl4ds.schedule import warmup_cosine_schedule
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


def test_multi_epoch_with_scheduler_numbers_steps_globally(tmp_path: Path) -> None:
    """``epochs>1`` re-iterates the stream with globally-increasing steps, stepping the LR sched.

    The P0.2.1 regime: several passes over an IID stream with a warmup+cosine schedule. The run
    log's steps must span more than a single epoch (so the extra passes actually happened) and
    the scheduler must have moved the LR off its warmup floor by the end.
    """
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    method = build_simsiam(encoder)
    stream = EraStream(
        SyntheticSource(num_classes=3, per_class=48, img_size=16),
        batch_size=12,
        order="iid",
        support_per_class=8,
        query_per_class=8,
        era_eval_per_class=5,
    )
    batches_per_epoch = len(stream)
    epochs = 3
    optimizer = torch.optim.AdamW(method.parameters(), lr=1e-3)
    scheduler = warmup_cosine_schedule(optimizer, total_steps=epochs * batches_per_epoch, warmup_frac=0.1)
    run_logger = RunLogger(str(tmp_path / "multi.jsonl"), run_name="multi")
    StreamingLoop(
        stream=stream,
        method=method,
        optimizer=optimizer,
        selection_filter=AcceptAll(),
        monitor=HealthMonitor(stream.eval_sets, knn_k=5),
        run_logger=run_logger,
        eval_every=batches_per_epoch,
        epochs=epochs,
        scheduler=scheduler,
    ).run()

    records = read_run(str(tmp_path / "multi.jsonl"))
    loss_steps = [r["step"] for r in records if r["series"] == "loss"]
    assert max(loss_steps) >= 2 * batches_per_epoch  # steps span past the first epoch (global numbering)
    assert len(loss_steps) == len(set(loss_steps))  # no duplicate step ids across epochs
    assert optimizer.param_groups[0]["lr"] < 1e-3  # cosine decayed the LR below the base by the end
