"""Unit tests for P1.1.3 item 5's bookkeeping (``scripts/read_forgetting_lenses.py``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from omegaconf import OmegaConf

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


def test_unpaired_reads_each_forgetting_arm_alone_against_the_healthy_arms() -> None:
    """PC arms far above the healthy arms' drift fire; the healthy arms' own leave-one-out calls stay quiet."""
    rng = np.random.default_rng(3)
    healthy = [{"cka_drift@block05": float(v)} for v in 0.01 * np.exp(rng.normal(size=6) * 0.1)]
    pc = [{"cka_drift@block05": 0.05}, {"cka_drift@block05": 0.0101}]
    out = _module().unpaired(pc, healthy, ["cka_drift@block05"], alpha=0.05, floor=0.25)
    assert [r["fired"] for r in out["pc"]] == [True, False]
    assert out["summary"]["loo_false_alarms"] == "0/6"


def test_select_blocks_keeps_only_layers_where_pc_wins_on_every_pair() -> None:
    """A layer where any pair has PC at or below its twin is dropped."""
    pc = [{"cka_drift@a": 2.0, "cka_drift@b": 2.0}, {"cka_drift@a": 2.0, "cka_drift@b": 1.0}]
    healthy = [{"cka_drift@a": 1.0, "cka_drift@b": 1.0}, {"cka_drift@a": 1.0, "cka_drift@b": 1.0}]
    assert _module().select_blocks(pc, healthy, ["cka_drift@a", "cka_drift@b"]) == ["cka_drift@a"]


def test_leave_one_pair_out_chooses_the_panel_without_the_held_out_pair() -> None:
    """A layer that separates only on the held-out pair is not chosen for it; it is chosen for the others."""
    rng = np.random.default_rng(0)
    n = 6
    base = 0.01 * np.exp(rng.normal(size=n) * 0.1)
    healthy = [{"cka_drift@good": float(b), "cka_drift@lucky": float(b)} for b in base]
    pc = [{"cka_drift@good": 0.05, "cka_drift@lucky": float(b) * 0.5} for b in base]
    pc[0]["cka_drift@lucky"] = 0.05  # separates on pair 0 alone
    out = _module().leave_one_pair_out(pc, healthy, alpha=0.05, floor=0.25)
    assert out["folds"][0]["panel"] == ["cka_drift@good"]
    assert all(f["panel"] == ["cka_drift@good"] for f in out["folds"][1:])
    assert out["summary"]["all_healthy"] == {"fires_on_pc": f"{n}/{n}", "fires_on_twin": f"0/{n}"}


def test_remap_mounts_swaps_whole_path_components_only() -> None:
    """Recorded container mounts become host paths; a longer name that merely shares the prefix is left alone."""
    cfg = OmegaConf.create(
        {
            "encoder": {"state_dict_path": "/mnt/cifar100/mae.pth"},
            "data": [{"root": "/mnt/stl10"}, {"root": "/mnt/stl100/x"}],
            "steps": 3,
        }
    )
    mounts = [{"from": "/mnt/cifar100", "to": "/host/data/cifar100"}, {"from": "/mnt/stl10/", "to": "/host/stl10"}]
    out = _module().remap_mounts(cfg, mounts)
    assert out.encoder.state_dict_path == "/host/data/cifar100/mae.pth"
    assert [d.root for d in out.data] == ["/host/stl10", "/mnt/stl100/x"]
    assert out.steps == 3
