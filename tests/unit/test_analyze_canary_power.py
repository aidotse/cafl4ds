"""Unit tests for the B2 injection read (``scripts/analyze_canary_power.py``)."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from omegaconf import OmegaConf

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_canary_power.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_analyze_canary_power", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corpus(run: Path, seeds: list[int], rng: np.random.Generator) -> None:
    """A healthy ``corpus/health.csv``: a canary near 0.6 with reading noise, 8 checkpoints per seed."""
    (run / "corpus").mkdir(parents=True)
    with (run / "corpus" / "health.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "arm", "checkpoint_index", "step", "knn_acc"])
        for seed in seeds:
            for i in range(8):
                writer.writerow([seed, "live", i, 10 * i, 0.6 + 0.01 * rng.normal()])


def _config(tmp_path: Path) -> object:
    return OmegaConf.create(
        {
            "backbones": ["je"],
            "seeds_dirs": [str(tmp_path / "a_{bb}"), str(tmp_path / "b_{bb}")],
            "t_mdd_dir": str(tmp_path / "a_{bb}"),
            "arbiters": ["knn_acc"],
            "shapes": ["sustained", "single"],
            "n_inject": 5,
            "n_groups": 30,
            "alpha": 0.05,
            "power": 0.8,
            "seed": 0,
            "n_query_pooled": 405,
            "n_query_stratum": 135,
            "out": str(tmp_path / "out" / "report.json"),
        }
    )


def test_a_sustained_drop_is_caught_smaller_than_a_single_reading_drop(tmp_path: Path) -> None:
    """The max-over-checkpoints test accumulates a sustained drop; one reading's drop must be larger to be caught."""
    rng = np.random.default_rng(0)
    _corpus(tmp_path / "a_je", [0, 1, 2, 3, 4], rng)
    _corpus(tmp_path / "b_je", [5, 6, 7, 8, 9], rng)
    report = _module().analyze(_config(tmp_path))
    entry = report["backbones"]["je"]
    assert entry["seeds"] == list(range(10))
    injected = entry["channels"]["knn_acc"]["injected"]
    assert 0 < injected["sustained"]["mdd"] < injected["single"]["mdd"]
    assert json.loads((tmp_path / "out" / "report.json").read_text())["n_inject"] == 5


def test_seeds_shared_between_dirs_are_rejected(tmp_path: Path) -> None:
    """A seed in both dirs would enter the injection twice."""
    rng = np.random.default_rng(1)
    _corpus(tmp_path / "a_je", [0, 1, 2], rng)
    _corpus(tmp_path / "b_je", [2, 3], rng)
    with pytest.raises(ValueError, match=r"\[2\]"):
        _module().analyze(_config(tmp_path))
