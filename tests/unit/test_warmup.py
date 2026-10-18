"""Unit tests for the P1.0.2 warm-the-well step (``cafl4ds.warmup``)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from cafl4ds.data.attributes import SyntheticAttributeSource
from cafl4ds.data.regime import RegimeStream
from cafl4ds.data.streams import StreamBatch
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.base import SSLMethod
from cafl4ds.ssl.factory import build_simsiam
from cafl4ds.warmup import (
    CompetenceCriterion,
    PlateauCriterion,
    check_split,
    load_well,
    save_split,
    save_well,
    warm_until_settled,
)


def _stream(seed: int = 0) -> RegimeStream:
    """A small synthetic regime stream (3 regimes × 3 canary classes)."""
    source = SyntheticAttributeSource(num_regimes=3, num_canary_classes=3, per_cell=16, img_size=16, long_tail=False)
    return RegimeStream(source, batch_size=8, support_per_canary=6, query_per_canary=6, seed=seed)


def _method() -> SSLMethod:
    """A tiny SimSiam method matching the 16px synthetic images."""
    encoder = TinyViTEncoder(img_size=16, patch_size=8, in_chans=3, embed_dim=32, depth=2, num_heads=2, mlp_ratio=2.0)
    return build_simsiam(encoder=encoder, proj_hidden=32, proj_dim=16, pred_hidden=16)


def test_plateau_needs_min_steps() -> None:
    """A short trace never plateaus, however flat, until min_steps is reached."""
    criterion = PlateauCriterion(window=5, tol=1e-2, min_steps=20)
    assert criterion.settled([1.0] * 10) is False


def test_plateau_fires_on_a_flat_trace() -> None:
    """A flat trace past min_steps has ~zero relative improvement and settles."""
    criterion = PlateauCriterion(window=5, tol=1e-3, min_steps=10)
    assert criterion.settled([0.5] * 40) is True


def test_plateau_holds_open_while_improving() -> None:
    """A steadily decreasing loss keeps improving faster than tol, so it does not settle."""
    criterion = PlateauCriterion(window=5, tol=1e-3, min_steps=10)
    decreasing = [1.0 - 0.02 * i for i in range(40)]
    assert criterion.settled(decreasing) is False


def test_warm_until_settled_reduces_loss_and_reports() -> None:
    """Warming a from-scratch SimSiam lowers the smoothed loss and returns a summary."""
    method = _method()
    stream = _stream()
    optimizer = torch.optim.AdamW(method.parameters(), lr=1e-3)
    summary = warm_until_settled(
        method=method,
        stream=stream,
        optimizer=optimizer,
        plateau=PlateauCriterion(window=5, tol=1e-3, min_steps=10),
        max_steps=200,
        device="cpu",
    )
    assert summary["steps"] > 0
    assert math.isfinite(summary["init_loss"]) and math.isfinite(summary["final_loss"])
    assert len(summary["loss_trace"]) == summary["steps"]


def test_warm_until_settled_respects_the_step_cap() -> None:
    """When a plateau cannot be declared, warming stops exactly at max_steps (across passes)."""
    method = _method()
    stream = _stream()
    optimizer = torch.optim.AdamW(method.parameters(), lr=1e-3)
    summary = warm_until_settled(
        method=method,
        stream=stream,
        optimizer=optimizer,
        plateau=PlateauCriterion(window=5, tol=1e-3, min_steps=10_000),  # min_steps unreachable → never settles
        max_steps=24,
        device="cpu",
    )
    assert summary["steps"] == 24
    assert summary["plateaued"] is False


def test_save_and_load_well_roundtrips_the_full_method(tmp_path: Path) -> None:
    """The well saves the whole method (encoder + head) and resumes it bit-for-bit."""
    method = _method()
    stream = _stream()
    optimizer = torch.optim.AdamW(method.parameters(), lr=1e-3)
    warm_until_settled(method=method, stream=stream, optimizer=optimizer, max_steps=12, device="cpu")

    path = save_well(method, tmp_path / "wells" / "je_well.pt")
    assert path.is_file()

    # a fresh method (different random init) is made identical once the well is loaded
    resumed = _method()
    before = torch.cat([p.flatten() for p in resumed.parameters()])
    load_well(resumed, path)
    after = torch.cat([p.flatten() for p in resumed.parameters()])
    original = torch.cat([p.flatten() for p in method.parameters()])
    assert not torch.equal(before, after)  # loading changed the fresh weights
    assert torch.equal(after, original)  # ...to match the saved well exactly (head included)


def test_load_well_missing_file_raises(tmp_path: Path) -> None:
    """Pointing a drive at a non-existent well is a hard error, not a silent from-scratch run."""
    with pytest.raises(FileNotFoundError, match="warm well not found"):
        load_well(_method(), tmp_path / "absent.pt")


def test_check_split_accepts_a_well_reserved_like_the_drive(tmp_path: Path) -> None:
    """A well saved with the drive's own fingerprint passes the check."""
    well = tmp_path / "well.pt"
    save_split(well, _stream().split_fingerprint())
    check_split(well, _stream(seed=3).split_fingerprint())  # the drive seed does not enter the split


def test_check_split_refuses_a_well_reserved_differently(tmp_path: Path) -> None:
    """A mismatched reservation is refused, naming the setting that differs."""
    well = tmp_path / "well.pt"
    save_split(well, _stream().split_fingerprint())
    other = RegimeStream(
        SyntheticAttributeSource(num_regimes=2, num_canary_classes=2, per_cell=10, img_size=16, long_tail=False),
        batch_size=4,
        support_per_canary=2,
        query_per_canary=2,
        canary_seed=7,
    )
    with pytest.raises(ValueError, match="canary_seed"):
        check_split(well, other.split_fingerprint())


def test_check_split_only_warns_for_a_well_without_a_fingerprint(tmp_path: Path) -> None:
    """A well saved before fingerprints existed is not refused (there is nothing to compare)."""
    check_split(tmp_path / "legacy.pt", _stream().split_fingerprint())


def _select_stream() -> RegimeStream:
    """The small stream with a selection set reserved."""
    source = SyntheticAttributeSource(num_regimes=3, num_canary_classes=3, per_cell=16, img_size=16, long_tail=False)
    return RegimeStream(
        source,
        batch_size=8,
        support_per_canary=6,
        query_per_canary=6,
        select_support_per_canary=3,
        select_query_per_canary=3,
    )


def _trace(values: list[float], every: int = 100) -> list[dict[str, float]]:
    """A competence trace read every ``every`` steps from step 0."""
    return [{"step": float(i * every), "knn_acc": v} for i, v in enumerate(values)]


def test_competence_criterion_waits_for_min_steps_and_two_windows() -> None:
    """No plateau is declared before ``min_steps`` or two full windows of readings."""
    rule = CompetenceCriterion(window=200, tol=0.01, min_steps=300)
    flat = _trace([0.5] * 10)
    assert not rule.settled(flat, 200)
    assert rule.settled(flat, 400)


def test_competence_criterion_keeps_going_while_accuracy_climbs() -> None:
    """A window that still gains ``tol`` is not a plateau; a flat one is."""
    rule = CompetenceCriterion(window=200, tol=0.01, min_steps=0)
    climbing = _trace([0.40, 0.42, 0.44, 0.46, 0.48])
    assert not rule.settled(climbing, 400)
    assert rule.settled(_trace([0.40, 0.42, 0.44, 0.44, 0.44, 0.44, 0.44]), 600)


def test_competence_criterion_waits_out_a_slide_too() -> None:
    """Two-sided: a pretrained well still sliding after its warm-up dip is not on its plateau yet."""
    rule = CompetenceCriterion(window=200, tol=0.01, min_steps=0)
    assert not rule.settled(_trace([0.60, 0.58, 0.56, 0.54, 0.52]), 400)
    assert rule.settled(_trace([0.60, 0.58, 0.56, 0.54, 0.54, 0.54, 0.54]), 600)


def test_budget_stop_runs_to_max_steps_and_logs_the_selection_trace() -> None:
    """``stop="budget"`` ignores the plateaus; the trace reads at step 0, every N steps, and the end."""
    torch.manual_seed(0)
    method = _method()
    summary = warm_until_settled(
        method=method,
        stream=_select_stream(),
        optimizer=torch.optim.AdamW(method.parameters(), lr=1e-3),
        max_steps=7,
        stop="budget",
        select_every=3,
        knn_k=3,
    )
    assert summary["steps"] == 7 and not summary["plateaued"]
    assert [r["step"] for r in summary["competence_trace"]] == [0.0, 3.0, 6.0, 7.0]
    assert all(0.0 <= r["knn_acc"] <= 1.0 and 0.0 <= r["linear_acc"] <= 1.0 for r in summary["competence_trace"])


def test_an_extra_read_lands_in_every_selection_row_without_stopping_the_warm_up() -> None:
    """``extra_read`` (the scout's drive-pool read) joins each trace row; it is logged, never a stop."""
    torch.manual_seed(0)
    method = _method()
    calls: list[int] = []

    def extra() -> dict[str, float]:
        calls.append(1)
        return {"drive_knn_bacc": 0.5}

    summary = warm_until_settled(
        method=method,
        stream=_select_stream(),
        optimizer=torch.optim.AdamW(method.parameters(), lr=1e-3),
        max_steps=4,
        stop="budget",
        select_every=2,
        knn_k=3,
        extra_read=extra,
    )
    assert summary["steps"] == 4 and len(calls) == len(summary["competence_trace"]) == 3
    assert all(r["drive_knn_bacc"] == 0.5 and "knn_acc" in r for r in summary["competence_trace"])


def test_competence_stop_fires_on_a_selection_plateau() -> None:
    """``stop="competence"`` ends the warm-up once the selection accuracy stops gaining."""
    torch.manual_seed(0)
    method = _method()
    summary = warm_until_settled(
        method=method,
        stream=_select_stream(),
        optimizer=torch.optim.AdamW(method.parameters(), lr=1e-3),
        competence_rule=CompetenceCriterion(window=2, tol=1.0, min_steps=0),  # any change < 1 settles
        max_steps=50,
        stop="competence",
        select_every=1,
        knn_k=3,
    )
    assert summary["plateaued"] and summary["steps"] == 4


def test_competence_reads_need_a_selection_set() -> None:
    """A selection read on a stream without a selection set, or a competence stop without reads, is refused."""
    method = _method()
    optimizer = torch.optim.AdamW(method.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="selection set"):
        warm_until_settled(method=method, stream=_stream(), optimizer=optimizer, select_every=2)
    with pytest.raises(ValueError, match="select_every"):
        warm_until_settled(method=method, stream=_select_stream(), optimizer=optimizer, stop="competence")


class _TailStream:
    """A stationary diet whose every pass ends on a 1-image batch (an awkward pool size)."""

    selection_sets = None

    def stationary_batches(self) -> list[StreamBatch]:
        return [
            StreamBatch(images=torch.rand(4, 3, 16, 16), era=0, step=0),
            StreamBatch(torch.rand(1, 3, 16, 16), 0, 1),
        ]


def test_a_one_image_tail_batch_is_skipped_not_trained_on() -> None:
    """A 1-image batch cannot pass the projector BatchNorm; the warm-up skips it as the loop does."""
    method = _method()
    summary = warm_until_settled(
        method=method,
        stream=_TailStream(),  # type: ignore[arg-type]
        optimizer=torch.optim.AdamW(method.parameters(), lr=1e-3),
        max_steps=3,
        max_passes=5,
        stop="budget",
    )
    assert summary["steps"] == 3
