"""Unit tests for P1.1.3 item 5's bookkeeping (``scripts/read_forgetting_lenses.py``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "read_forgetting_lenses.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_read_forgetting_lenses", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gate(pc: float, healthy: float, passed: bool = True) -> dict[str, object]:
    return {"gate": {"pc_backward_transfer": pc, "healthy_backward_transfer": healthy, "passed": passed}}


def test_reproduction_needs_both_arms_within_tolerance_and_the_same_verdict() -> None:
    """A BWT off by more than the tolerance on either arm, or a flipped verdict, fails the reproduction."""
    reproduction = _module().reproduction
    assert reproduction(_gate(-0.073, 0.012), _gate(-0.075, 0.013), 0.005)["ok"]
    assert not reproduction(_gate(-0.075, 0.020), _gate(-0.075, 0.013), 0.005)["ok"]
    assert not reproduction(_gate(-0.075, 0.013, passed=False), _gate(-0.075, 0.013), 0.005)["ok"]


def test_paired_diffs_are_harm_signed_per_lens() -> None:
    """More drift on PC reads positive; a RankMe fall on PC (relative to healthy) reads positive too."""
    module = _module()
    pc = [
        module.changes(
            {"cka_drift@block01": 0.05, "rankme@block01": 20.0}, {"cka_drift@block01": 0.0, "rankme@block01": 25.0}
        )
    ]
    healthy = [{"cka_drift@block01": 0.02, "rankme@block01": -1.0}]
    diffs = module.paired_diffs(pc, healthy, ["cka_drift@block01", "rankme@block01"])
    assert diffs[0] == pytest.approx(np.array([0.03, 4.0]))
