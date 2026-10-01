"""Unit tests for ``scripts/plot_federated_health.py`` — log parsing and figure output."""

import json
from pathlib import Path

import pytest

from scripts.plot_federated_health import load_health, load_run, plot_health


def _write_log(path: Path, records: list[dict[str, object]]) -> None:
    """Write ``records`` as a JSONL run log."""
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _health(step: int, **metrics: float) -> dict[str, object]:
    """Build one ``health`` record."""
    return {"run": "r", "series": "health", "step": step, "era": -1, **metrics}


def test_load_health_keeps_numeric_health_metrics_only(tmp_path: Path) -> None:
    """Loss lines, bookkeeping fields, booleans, and non-finite values are all left out."""
    path = tmp_path / "global_run_log.jsonl"
    _write_log(
        path,
        [
            {"run": "r", "series": "loss", "step": 0, "era": -1, "loss": 1.0},
            _health(0, rankme=3.0, finite=True),
            {**_health(1, rankme=4.0), "knn_acc": float("nan")},
        ],
    )
    assert load_health(path) == {"rankme": [(0, 3.0), (1, 4.0)]}


def test_load_run_collects_global_and_clients_with_health(tmp_path: Path) -> None:
    """Clients are keyed by id; a client log without health records is skipped."""
    _write_log(tmp_path / "global_run_log.jsonl", [_health(0, rankme=1.0)])
    _write_log(tmp_path / "client1_run_log.jsonl", [_health(0, rankme=2.0)])
    _write_log(tmp_path / "client0_run_log.jsonl", [{"run": "c0", "series": "loss", "step": 0, "era": 0, "loss": 1.0}])
    global_series, clients = load_run(tmp_path)
    assert global_series == {"rankme": [(0, 1.0)]}
    assert list(clients) == [1]


def test_load_run_requires_a_global_log(tmp_path: Path) -> None:
    """A directory without a global log is not a federated run."""
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path)


def test_plot_health_writes_an_image(tmp_path: Path) -> None:
    """A global-only metric and a shared metric both get a panel."""
    global_series = {"rankme": [(0, 1.0), (1, 2.0)], "client_anchor_cka_max": [(0, 0.1), (1, 0.2)]}
    clients = {0: {"rankme": [(0, 1.5), (1, 2.5)]}}
    out = plot_health(global_series, clients, None, tmp_path / "health.png", "test")
    assert out.is_file() and out.stat().st_size > 0


def test_plot_health_rejects_unknown_metrics(tmp_path: Path) -> None:
    """Asking only for absent metrics fails loudly, listing what is available."""
    with pytest.raises(ValueError, match="available"):
        plot_health({"rankme": [(0, 1.0)]}, {}, ["nope"], tmp_path / "x.png", "test")
