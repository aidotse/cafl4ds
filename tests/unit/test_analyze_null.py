"""Unit tests for the P1.1.0 null report's familiarity read (``scripts/analyze_null.py``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_null.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_analyze_null", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_familiarity_trend_counts_every_signal_and_field_and_states_the_floor() -> None:
    """One test per signal and movement field, with the smallest p reported beside how many were run."""
    module = _module()
    per_level = {
        f"o={o}": {"signals": {s: {"iid_movement": {"final": v, "max_abs": abs(v) + k}} for s, v in vals.items()}}
        for k, (o, vals) in enumerate(
            [(0.0, {"a": 1.0, "b": 3.0}), (0.5, {"a": 2.0, "b": 1.0}), (1.0, {"a": 3.0, "b": 2.0})]
        )
    }
    trend = module.familiarity_trend(per_level)
    assert trend["n_tests"] == 4 and trend["p_floor"] == pytest.approx(2 / 6)
    assert trend["tests"]["a/final"]["rho"] == pytest.approx(1.0)
    assert trend["min_p"] == pytest.approx(2 / 6)
