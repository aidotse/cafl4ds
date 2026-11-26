"""P1.1.4 Parts 2 and 3 — the branch read: kinds, the referee's paired reads, the control and the reader bar."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from cafl4ds.reading_protocol import calibrate_panel
from scripts.analyze_p114_branches import (
    _lens_inputs,
    _on_grid,
    branch_kinds,
    control_verdict,
    kind_order,
    lens_matrix,
    paired_referee,
    panel_calls,
    reader_bar,
)


class _Stub:
    """A ladder read keyed on (prefix, seed, checkpoint, key); unread = nan."""

    def __init__(self, reads: dict[tuple[str, int, str, str], float]) -> None:
        self.reads = reads

    def read(self, prefix: str, seed: int, checkpoint: str, key: str) -> float:
        return self.reads.get((prefix, seed, checkpoint, key), math.nan)


def test_kind_order_puts_the_twin_then_reinits_then_doses_by_size() -> None:
    """Reading order: dose0, re-inits by depth, then noise by its SD multiple (0p875 after 0p65)."""
    kinds = ["noise0p875", "reinit4", "noise0p5", "dose0", "noise0p65", "reinit2"]
    assert kind_order(kinds) == ["dose0", "reinit2", "reinit4", "noise0p5", "noise0p65", "noise0p875"]


def test_kind_order_rejects_unknown_kinds_and_a_missing_twin() -> None:
    """A stray dir name or a set without its dose-0 twin cannot be read."""
    with pytest.raises(ValueError, match="unknown"):
        kind_order(["dose0", "noise1"])
    with pytest.raises(ValueError, match="twin"):
        kind_order(["reinit2", "noise0p5"])


def test_branch_kinds_keeps_the_arm_exact(tmp_path: Path) -> None:
    """``mae`` must not collect ``mae_je``'s branches; non-branch dirs are ignored."""
    for name in ("mae_dose0", "mae_reinit2", "mae_noise0p4", "mae_je_dose0", "mae_je_noise0p3", "mae_notes"):
        (tmp_path / "branch" / name).mkdir(parents=True)
    assert branch_kinds(tmp_path, "branch", "mae") == ["dose0", "reinit2", "noise0p4"]
    assert branch_kinds(tmp_path, "branch", "mae_je") == ["dose0", "noise0p3"]


def test_paired_referee_subtracts_the_twin_and_the_walk_seed_by_seed() -> None:
    """Damage = branch − twin, restart = twin − walk, both in points; the twins' SD across seeds."""
    key, ck = "scene|linear", "end"
    branches = _Stub(
        {
            ("dose0", 1, ck, key): 0.60,
            ("dose0", 2, ck, key): 0.62,
            ("noise0p5", 1, ck, key): 0.50,
            ("noise0p5", 2, ck, key): 0.54,
        }
    )
    walks = _Stub({("walk", 1, ck, key): 0.61, ("walk", 2, ck, key): 0.62})
    out = paired_referee(branches, walks, ["dose0", "noise0p5"], [1, 2], [key], [ck])
    assert out["damage"]["noise0p5"][key][ck]["per_seed"] == pytest.approx([-10.0, -8.0])
    assert out["damage"]["noise0p5"][key][ck]["mean"] == pytest.approx(-9.0)
    assert out["restart"][key][ck]["per_seed"] == pytest.approx([-1.0, 0.0])
    assert out["twin_sd"][key][ck] == pytest.approx(100 * 0.02 / math.sqrt(2))
    assert "dose0" not in out["damage"]


def test_control_verdict_reads_linear_and_finetune_rungs_only() -> None:
    """A kNN loss never disqualifies; a linear or fine-tune mean under −k twin SDs does."""
    referee = {
        "twin_sd": {"scene|knn": {"end": 0.5}, "scene|linear": {"end": 0.5}, "scene|ft100": {"end": 0.5}},
        "damage": {
            "reinit2": {
                "scene|knn": {"end": {"mean": -5.0}},
                "scene|linear": {"end": {"mean": -0.4}},
                "scene|ft100": {"end": {"mean": -0.6}},
            }
        },
    }
    out = control_verdict(referee, "reinit2", 1.0)
    assert not out["confirmed"]
    assert [b["key"] for b in out["below"]] == ["scene|ft100"]
    assert control_verdict(referee, "reinit2", 2.0)["confirmed"]


def test_on_grid_drops_readings_off_the_null_grid() -> None:
    """A 600-step drive's last reading at 599 is not on the 25-step grid."""
    rows = [{"step": s} for s in (0, 25, 575, 599)]
    kept, dropped = _on_grid(rows, [0, 25, 50, 575, 600])
    assert [r["step"] for r in kept] == [0, 25, 575]
    assert dropped == [599]


def _entry(first: dict[str, list[int | None]], twin_fired: bool, walk_fired: list[bool]) -> dict[str, Any]:
    def calls(steps: list[int | None]) -> dict[int, Any]:
        return {
            s + 1: {
                "fired": st is not None,
                "first_step": st,
                "fired_after_first": st is not None and st > 850,
                "final_z": 5.0 if st is not None else 0.0,
            }
            for s, st in enumerate(steps)
        }

    kinds = {k: calls(v) for k, v in first.items()}
    kinds["dose0"] = calls([900 if twin_fired else None, None, None])
    return {
        "threshold": 4.0,
        "loo": {"fired": 1, "of": 10},
        "first_reading": 850,
        "control": "reinit2",
        "walks": {i: {"fired": f} for i, f in enumerate(walk_fired)},
        "kinds": kinds,
    }


_CFG = OmegaConf.create({"branches": {"loo_max": 2, "walks_quiet": 4, "late_by": 1269}})


def test_reader_bar_passes_a_reader_firing_at_once_and_quiet_elsewhere() -> None:
    """Every called seed at the first reading, quiet twins, walks and control: a pass."""
    e = _entry({"noise0p5": [850, 850, 850], "reinit2": [None, None, None]}, False, [False] * 4 + [True])
    bar = reader_bar(e, {"noise0p5": [1, 2, 3]}, _CFG)
    assert bar["pass"] and bar["specific"] and bar["sensitive"]


def test_reader_bar_fails_on_a_late_fire_a_twin_or_the_walks() -> None:
    """A fire after the first reading is only `sensitive_by_late`; a twin or two walks fail specificity."""
    late = _entry({"noise0p5": [850, 900, 850], "reinit2": [None] * 3}, False, [False] * 5)
    bar = reader_bar(late, {"noise0p5": [1, 2, 3]}, _CFG)
    assert not bar["sensitive"] and bar["sensitive_by_late"] and not bar["pass"]
    twin = _entry({"noise0p5": [850] * 3, "reinit2": [None] * 3}, True, [False] * 5)
    assert not reader_bar(twin, {"noise0p5": [1, 2, 3]}, _CFG)["specific"]
    walks = _entry({"noise0p5": [850] * 3, "reinit2": [None] * 3}, False, [True, True, False, False, False])
    assert not reader_bar(walks, {"noise0p5": [1, 2, 3]}, _CFG)["walks"]


def test_reader_bar_reads_an_uncalled_seed_as_no_demand() -> None:
    """Only called seeds count for sensitivity; a control fire fails specificity."""
    e = _entry({"noise0p5": [850, None, 850], "reinit2": [None, 900, None]}, False, [False] * 5)
    bar = reader_bar(e, {"noise0p5": [1, 3]}, _CFG)
    assert bar["sensitive"] and bar["control"] is False and not bar["pass"]


def test_lens_matrix_reads_the_table_by_run_seed_and_stem() -> None:
    """Each (seed, stem) cell from ``features/<run>/s<seed>/<stem>.pt``; an unread checkpoint is nan."""
    table = {
        "features/walk/s1/step00100.pt": {"cka_drift@block01": 0.1},
        "features/walk/s1/final.pt": {"cka_drift@block01": 0.3},
        "features/walk/s2/step00100.pt": {"cka_drift@block01": 0.2},
    }
    m = lens_matrix(table, "walk", [1, 2], ["step00100", "final"], ["cka_drift@block01"])["cka_drift@block01"]
    assert m[0].tolist() == [0.1, 0.3]
    assert m[1, 0] == 0.2 and math.isnan(m[1, 1])


def test_panel_calls_fire_on_the_layer_that_leaves_the_bands() -> None:
    """A drift jump in one layer at the first branch checkpoint fires the panel there, naming the layer."""
    rng = np.random.default_rng(0)
    grid = [100, 212, 423, 846, 1269, 1692]
    nulls = {f"cka_drift@block0{i}": 0.1 + 0.01 * rng.standard_normal((10, 6)) for i in (1, 2)}
    rules = calibrate_panel(nulls, dict.fromkeys(nulls, {"span": 3, "floor": 0.25}), 0.05)
    drive = {s: np.full((1, 3), 0.1) for s in nulls}
    drive["cka_drift@block02"][0, 0] = 0.9
    out = panel_calls(drive, rules, [3, 4, 5], grid[3:], [1])
    assert out[1]["fired"] and out[1]["first_step"] == 846
    assert out[1]["fired_layers"] == ["block02"]


def test_reader_bar_skips_the_twins_start_state() -> None:
    """A twin firing only at its first reading (the walk's own checkpoint, before the restart) stays quiet."""
    e = _entry({"noise0p5": [850] * 3, "reinit2": [None] * 3}, False, [False] * 5)
    e["kinds"]["dose0"][1] = {"fired": True, "first_step": 850, "fired_after_first": False, "final_z": 1.0}
    assert reader_bar(e, {"noise0p5": [1, 2, 3]}, _CFG)["twins"]
    e["kinds"]["dose0"][1]["fired_after_first"] = True
    assert not reader_bar(e, {"noise0p5": [1, 2, 3]}, _CFG)["twins"]


def test_lens_inputs_read_a_promoted_cache_and_write_nothing_beside_the_features(tmp_path: Path) -> None:
    """The per-block screen's inputs come from the features dir's own cache; nothing is computed or written there."""
    fdir = tmp_path / "p114" / "features" / "arm"
    fdir.mkdir(parents=True)
    keys = [
        f"features/arm_{order}/s{seed}/{stem}.pt"
        for order in ("iid", "walk")
        for seed in (1, 2)
        for stem in ("step00100", "final")
    ]
    (fdir / "index.json").write_text(json.dumps([{"features": k} for k in keys]), encoding="utf-8")
    cache = {k: {"cka_drift@block04": float(i)} for i, k in enumerate(keys)}
    (fdir / "lens_values.json").write_text(json.dumps(cache), encoding="utf-8")
    (tmp_path / "p114" / "branch" / "arm_dose0").mkdir(parents=True)  # its features unread: nan
    config = OmegaConf.create(
        {
            "out": str(tmp_path / "report" / "branches.json"),
            "branches": {
                "dir": "p114/branch",
                "seeds": [1, 2],
                "lens": {"workers": 1, "stems": ["step00100", "final"], "branch_stems": ["final"], "end_step": 1692},
            },
        }
    )
    arm = OmegaConf.create(
        {
            "branches": {
                "features": {
                    "dir": "p114/features/arm",
                    "iid": "arm_iid",
                    "iid_seeds": [1, 2],
                    "walk": "arm_walk",
                    "walk_seeds": [1, 2],
                }
            }
        }
    )
    before = sorted(p.name for p in fdir.iterdir())
    inputs = _lens_inputs("arm", arm, config, tmp_path)
    assert inputs["iid"]["cka_drift@block04"].tolist() == [[0.0, 1.0], [2.0, 3.0]]
    assert inputs["grid"] == [100, 1692] and list(inputs["branch"]) == ["dose0"]
    assert sorted(p.name for p in fdir.iterdir()) == before
