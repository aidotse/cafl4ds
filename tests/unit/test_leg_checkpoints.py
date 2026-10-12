"""Unit tests for the leg-boundary checkpointer (``cafl4ds.leg_checkpoints``)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cafl4ds.data.attributes import SyntheticAttributeSource
from cafl4ds.data.regime import RegimeStream
from cafl4ds.leg_checkpoints import CheckpointingStream


def _stream() -> RegimeStream:
    """A small three-regime walk."""
    source = SyntheticAttributeSource(num_regimes=3, num_canary_classes=2, per_cell=12, img_size=8, long_tail=False)
    return RegimeStream(source, batch_size=8, support_per_canary=3, query_per_canary=3)


def _drive(wrapped: CheckpointingStream, module: torch.nn.Linear) -> None:
    """Consume the stream like the loop does: train on each batch only after it is yielded.

    The "training" stamps the module's bias with the number of completed steps, so a checkpoint's
    bias says exactly which state it captured.
    """
    for batch in wrapped:
        with torch.no_grad():
            module.bias.fill_(float(batch.step + 1))


def test_checkpoints_land_at_leg_boundaries_and_the_end(tmp_path: Path) -> None:
    """One checkpoint per leg boundary plus ``final.pt``, indexed in ``index.json``."""
    stream = _stream()
    module = torch.nn.Linear(1, 1)
    wrapped = CheckpointingStream(stream, module, stream.leg_boundary_steps, tmp_path)
    assert len(wrapped) == len(stream)
    _drive(wrapped, module)
    expected = [f"step{s:05d}.pt" for s in stream.leg_boundary_steps] + ["final.pt"]
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == sorted(expected)
    index = json.loads((tmp_path / "index.json").read_text())
    assert [e["file"] for e in index] == expected


def test_a_boundary_checkpoint_captures_the_end_of_the_previous_leg(tmp_path: Path) -> None:
    """The checkpoint before step ``s`` holds the state after exactly ``s`` completed steps."""
    stream = _stream()
    module = torch.nn.Linear(1, 1)
    _drive(CheckpointingStream(stream, module, stream.leg_boundary_steps, tmp_path), module)
    for step in stream.leg_boundary_steps:
        state = torch.load(tmp_path / f"step{step:05d}.pt")
        assert float(state["bias"]) == float(step)
    final = torch.load(tmp_path / "final.pt")
    assert float(final["bias"]) == float(len(stream))
    index = json.loads((tmp_path / "index.json").read_text())
    assert index[-1]["completed_steps"] == len(stream)


def test_wrapping_does_not_change_the_delivered_batches(tmp_path: Path) -> None:
    """The wrapper is transparent: same batches, same order."""
    stream = _stream()
    plain = list(stream)
    wrapped = list(CheckpointingStream(stream, torch.nn.Linear(1, 1), [], tmp_path))
    assert len(plain) == len(wrapped)
    assert all(torch.equal(a.images, b.images) and a.era == b.era for a, b in zip(plain, wrapped, strict=True))
