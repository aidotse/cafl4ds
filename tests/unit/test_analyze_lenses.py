"""Unit tests for the P1.1.2 screen's bookkeeping (``scripts/analyze_lenses.py``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

from cafl4ds.reading_protocol import fit_bands

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_lenses.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_analyze_lenses", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(effects: dict[str, float], canary: float, weight: float) -> dict[str, object]:
    return {
        "effects": {s: {"immediate": e, "end": np.nan} for s, e in effects.items()},
        "canary_immediate": canary,
        "canary_end": np.nan,
        "rel_change_encoder": weight,
    }


def test_screen_counts_called_quiet_and_reinit4_seeds_per_reader() -> None:
    """A reader firing on every called seed and on re-init 4 scores full sensitivity and zero specificity there."""
    null = np.random.default_rng(0).normal(size=(5, 6)) + 10.0
    rules = {s: fit_bands(null, s, span=1, threshold=3.0) for s in ("twonn@block01", "twonn@block02", "rankme")}
    arms = {
        "iid/noise1.0": {"0": _seed({"twonn@block01": 9.0, "twonn@block02": 0.1, "rankme": -9.0}, -0.15, 1.0)},
        "iid/reinit4": {"0": _seed({"twonn@block01": 0.2, "twonn@block02": -8.0, "rankme": -9.0}, 0.0, 0.66)},
        "iid/noise0.3": {"0": _seed({"twonn@block01": 0.1, "twonn@block02": 0.1, "rankme": 9.0}, 0.0, 0.3)},
    }
    p111 = {
        "branches": {
            "iid/noise1.0": {"fired_share": {"knn_acc": 1.0, "linear_acc": 1.0}},
            "iid/reinit4": {"fired_share": {"knn_acc": 0.0, "linear_acc": 0.0}},
            "iid/noise0.3": {"fired_share": {"knn_acc": 0.0, "linear_acc": 0.0}},
        }
    }
    readers = {"twonn": ["twonn@block01", "twonn@block02"], "rankme": ["rankme"]}
    out = _module().screen(arms, rules, readers, p111)
    twonn = out["twonn"]["immediate"]
    assert twonn["fires_on_called"] == "1/1" and twonn["quiet_on_reinit4"] == "0/1"  # two-sided: -8 fires
    assert twonn["quiet_on_quiet"] == "1/2" and twonn["layers_fired_on_reinit4"] == ["block02"]
    rankme = out["rankme"]["immediate"]  # effects are harm-signed: -9 (a rise) stays quiet, +9 (a fall) fires
    assert rankme["fires_on_called"] == "0/1" and rankme["quiet_on_quiet"] == "1/2"
    assert rankme["quiet_on_reinit4"] == "1/1"
    assert "end" not in out["rankme"]  # no finite end effects -> no end read
    # twonn places re-init 4 (8.0) above one noise dose (0.1), below the other (9.0)
    assert twonn["ladder"] == {"noise1.0": 9.0, "reinit4": 8.0, "noise0.3": 0.1}
    assert twonn["reinit4_above_noise_doses"] == 1
