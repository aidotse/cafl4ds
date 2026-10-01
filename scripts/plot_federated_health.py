#!/usr/bin/env python3
"""Plot a federated run's health series over rounds: one panel per metric, one line per client.

Reads the run logs ``run_federated.py`` writes into its Hydra output directory: ``global_*.jsonl``
(the aggregated model, plus the divergence summaries) and ``client<i>_*.jsonl`` (each client's
post-round model, present when the run set ``track_client_health=true``). The global model is
drawn as a thick grey line *underneath* the thin coloured client lines, so a client
that coincides with it (e.g. the last one still training) stays visible; metrics that only the global log
carries (e.g. ``client_anchor_cka_max``) get the global line alone.

Usage::

    uv run python scripts/plot_federated_health.py outputs/2026-10-01/12-00-00
    uv run python scripts/plot_federated_health.py RUN_DIR --metrics rankme knn_acc --out health.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl
from matplotlib.artist import Artist
from matplotlib.axes import Axes

mpl.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

# Record fields that are bookkeeping, not health metrics.
_NON_METRICS = frozenset({"run", "series", "step", "era"})

Series = dict[str, list[tuple[int, float]]]  # metric -> [(round, value), ...]


def load_health(path: Path) -> Series:
    """Read the ``health`` records of one run log into per-metric ``(round, value)`` series.

    Args:
        path: A JSONL run log.

    Returns:
        Each numeric metric's points, in file order. Non-finite values are dropped.
    """
    series: Series = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("series") != "health":
            continue
        for key, value in record.items():
            if key in _NON_METRICS or isinstance(value, bool) or not isinstance(value, int | float):
                continue
            if math.isfinite(value):
                series.setdefault(key, []).append((int(record["step"]), float(value)))
    return series


def load_run(run_dir: Path) -> tuple[Series, dict[int, Series]]:
    """Load the global and per-client health series of one federated run directory.

    Args:
        run_dir: The Hydra output directory of a ``run_federated.py`` run.

    Returns:
        ``(global_series, {client_id: client_series})``; clients without health records are left out.

    Raises:
        FileNotFoundError: If ``run_dir`` holds no ``global_*.jsonl`` log.
    """
    global_logs = sorted(run_dir.glob("global_*.jsonl"))
    if not global_logs:
        raise FileNotFoundError(f"no global_*.jsonl run log under {run_dir}")
    clients: dict[int, Series] = {}
    for path in run_dir.glob("client*_*.jsonl"):
        match = re.match(r"client(\d+)_", path.name)
        if match and (series := load_health(path)):
            clients[int(match.group(1))] = series
    return load_health(global_logs[0]), dict(sorted(clients.items()))


def _unique_legend(axes: Iterable[Axes]) -> tuple[list[Artist], list[str]]:
    """Collect every line label once across panels (a panel may lack some clients' lines)."""
    handles: list[Artist] = []
    labels: list[str] = []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    return handles, labels


def plot_health(
    global_series: Series, clients: dict[int, Series], metrics: list[str] | None, out: Path, title: str
) -> Path:
    """Draw one panel per metric (global line + one line per client) and save the figure.

    Args:
        global_series: The aggregated model's series.
        clients: Each client's series, by client id.
        metrics: Metrics to plot, in order; ``None`` plots every metric found.
        out: Destination image path.
        title: Figure title.

    Returns:
        ``out``.

    Raises:
        ValueError: If none of the requested metrics is present.
    """
    found = set(global_series).union(*(set(s) for s in clients.values()))
    names = [m for m in metrics if m in found] if metrics else sorted(found)
    if not names:
        raise ValueError(f"none of the requested metrics are in the logs; available: {sorted(found)}")
    cols = min(3, len(names))
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False, sharex=True)
    colours = plt.get_cmap("tab10")
    for ax, name in zip(axes.flat, names, strict=False):
        for cid, series in clients.items():
            if name in series:
                x, y = zip(*series[name], strict=True)
                ax.plot(x, y, lw=1.2, color=colours(cid % 10), zorder=2, label=f"client {cid}")
        if name in global_series:
            x, y = zip(*global_series[name], strict=True)
            ax.plot(x, y, lw=3.0, color="black", alpha=0.45, zorder=1, label="global")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("round")
    fig.legend(*_unique_legend(axes.flat[: len(names)]), loc="upper right", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.9, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    """Parse the CLI and write the health figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Hydra output directory of a run_federated.py run")
    parser.add_argument("--metrics", nargs="+", default=None, help="metrics to plot (default: all)")
    parser.add_argument("--out", type=Path, default=None, help="image path (default: RUN_DIR/health.png)")
    args = parser.parse_args()
    global_series, clients = load_run(args.run_dir)
    out = plot_health(global_series, clients, args.metrics, args.out or args.run_dir / "health.png", args.run_dir.name)
    print(f"wrote {out} ({len(clients)} clients)")


if __name__ == "__main__":
    main()
