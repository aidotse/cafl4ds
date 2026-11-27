"""Unit tests for the P1.1.1 reading report's corpus pooling (``scripts/analyze_reading.py``)."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_reading.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_analyze_reading", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corpus(run: Path, seeds: list[int]) -> None:
    """A two-checkpoint ``corpus/health.csv`` for ``seeds``."""
    (run / "corpus").mkdir(parents=True)
    with (run / "corpus" / "health.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "arm", "checkpoint_index", "step", "rankme"])
        for seed in seeds:
            for i in range(2):
                writer.writerow([seed, "live", i, 10 * i, 1.0 + seed])


def test_union_pools_disjoint_seeds_and_skips_missing_dirs(tmp_path: Path) -> None:
    """Disjoint seed sets pool; a dir without a corpus is skipped."""
    _corpus(tmp_path / "a_je", [1, 2])
    _corpus(tmp_path / "b_je", [3])
    rows = _module()._union([str(tmp_path / "a_{bb}"), str(tmp_path / "b_{bb}"), str(tmp_path / "c_{bb}")], "je")
    assert sorted({int(r["seed"]) for r in rows}) == [1, 2, 3]


def test_union_rejects_a_seed_in_two_dirs(tmp_path: Path) -> None:
    """A seed in two dirs would have one run's readings overwrite the other's in the seed matrix."""
    _corpus(tmp_path / "a_je", [1, 2])
    _corpus(tmp_path / "b_je", [2, 3])
    with pytest.raises(ValueError, match="seed 2"):
        _module()._union([str(tmp_path / "a_{bb}"), str(tmp_path / "b_{bb}")], "je")
