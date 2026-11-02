"""Matched comparisons of regime-blocked and IID routing experiments, without retraining controls."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import t as student_t

from cafl4ds.routing_experiment import ARMS, digest_file, tensor_hash, write_json


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def _interval(values: list[float]) -> dict[str, Any]:
    """Descriptive paired t interval; the input has one difference per training seed."""
    array = np.array(values)
    mean = float(array.mean())
    half = (
        float(student_t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / math.sqrt(len(array)))
        if len(array) > 1
        else None
    )
    return {"mean": mean, "ci95": [mean - half, mean + half] if half is not None else None}


def _verify_candidate_traces(blocked: Path, iid: Path, manifest: dict[str, Any]) -> None:
    """Compare the manifest with actual observed candidate IDs in both stream conditions."""
    for seed, epochs in manifest["seeds"].items():
        control_rows = _read_rows(iid / f"seed_{seed}" / "random" / "selection.jsonl")
        blocked_rows = _read_rows(blocked / f"seed_{seed}" / "random" / "selection.jsonl")
        for epoch in epochs:
            control = [
                sample_id for row in control_rows if row["epoch"] == epoch["epoch"] for sample_id in row["sample_ids"]
            ]
            observed = [
                sample_id for row in blocked_rows if row["epoch"] == epoch["epoch"] for sample_id in row["sample_ids"]
            ]
            if (
                tensor_hash(torch.tensor(control, dtype=torch.long).sort().values)
                != epoch["candidate_membership_sha256"]
            ):
                raise ValueError("Actual IID candidate membership differs from the regime stream.")
            if tensor_hash(torch.tensor(observed, dtype=torch.long)) != epoch["candidate_order_sha256"]:
                raise ValueError("Actual blocked candidate order differs from the stream manifest.")


def _verify_matching(blocked: Path, iid: Path) -> dict[str, Any]:
    """Verify settings, panels and actual retained candidate IDs before attributing effects to order."""
    new = json.loads((blocked / "protocol.json").read_text())
    old = json.loads((iid / "protocol.json").read_text())
    if new["config"].get("stream_order") != "regime_blocked" or old["config"].get("stream_order", "iid") != "iid":
        raise ValueError("Expected a regime-blocked experiment and an IID reference.")
    new_settings = {k: v for k, v in new["config"].items() if k not in {"stream_order", "output"}}
    old_settings = {k: v for k, v in old["config"].items() if k not in {"stream_order", "output"}}
    if new_settings != old_settings or new["panels"] != old["panels"] or new["runtime"] != old["runtime"]:
        raise ValueError("Ordering comparison requires identical settings, panels and software versions.")
    if new["corpus"] != old["corpus"] or new["gate_semantics"] != old["gate_semantics"]:
        raise ValueError("Ordering comparison changed the corpus or gate semantics.")
    manifest = json.loads((blocked / "stream_manifest.json").read_text())
    _verify_candidate_traces(blocked, iid, manifest)
    return {
        "matched_settings": True,
        "matched_panels": True,
        "matched_runtime": True,
        "matched_corpus_and_gate": True,
        "matched_actual_candidate_membership_each_epoch": True,
        "matched_actual_blocked_order_to_manifest": True,
        "iid_protocol_hash": old["protocol_hash"],
        "regime_protocol_hash": new["protocol_hash"],
    }


def order_effects(blocked_runs: list[dict[str, Any]], iid_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate within-policy order effects from changes in each filter's penalty versus random."""
    blocked = {(r["seed"], r["arm"]): r for r in blocked_runs}
    iid = {(r["seed"], r["arm"]): r for r in iid_runs}
    if not blocked or blocked.keys() != iid.keys():
        raise ValueError("Ordering comparison requires matching seed/policy pairs.")
    seeds = sorted({seed for seed, _ in blocked})
    if set(blocked) != {(seed, arm) for seed in seeds for arm in ARMS}:
        raise ValueError("Every seed needs all three admission policies.")
    rows = []
    for (seed, arm), current in sorted(blocked.items()):
        control = iid[(seed, arm)]
        if current["initial_hash"] != control["initial_hash"] or current["steps"] != control["steps"]:
            raise ValueError("Initial weights or training budget differ between stream conditions.")
        a, b = current["final"], control["final"]
        rows.append(
            {
                "seed": seed,
                "arm": arm,
                "relative_mse": a["heldout_mse"] / b["heldout_mse"] - 1,
                "entropy_change": current["entropy_change"] - control["entropy_change"],
                "balanced_knn_pp": 100 * (a["balanced_knn"] - b["balanced_knn"]),
                "balanced_linear_pp": 100 * (a["balanced_linear"] - b["balanced_linear"]),
            }
        )
    summaries = {
        arm: {
            metric: _interval([r[metric] for r in rows if r["arm"] == arm])
            for metric in ("relative_mse", "entropy_change", "balanced_knn_pp", "balanced_linear_pp")
        }
        for arm in ARMS
    }
    interactions = {}
    for arm in ("high", "low"):
        differences = []
        for seed in seeds:
            regime_penalty = (
                blocked[(seed, arm)]["final"]["heldout_mse"] / blocked[(seed, "random")]["final"]["heldout_mse"] - 1
            )
            iid_penalty = iid[(seed, arm)]["final"]["heldout_mse"] / iid[(seed, "random")]["final"]["heldout_mse"] - 1
            differences.append(100 * (regime_penalty - iid_penalty))
        interactions[arm + "_minus_random"] = {
            "relative_mse_penalty_change_pp": _interval(differences),
            "seed_differences_pp": dict(zip(seeds, differences, strict=True)),
        }
    return {
        "per_seed": rows,
        "blocked_minus_iid": summaries,
        "filter_by_order_interaction": interactions,
        "interval_note": "descriptive paired 95% t intervals across seeds; no multiplicity-adjusted tests",
    }


def compare_ordering(blocked: Path, iid: Path) -> dict[str, Any]:
    """Create matched order comparisons, regime timing, and combined diagnostic figures."""
    blocked, iid = blocked.resolve(), iid.resolve()
    verification = _verify_matching(blocked, iid)
    new = json.loads((blocked / "comparison.json").read_text())
    old = json.loads((iid / "comparison.json").read_text())
    result = {
        **order_effects(new["runs"], old["runs"]),
        "verification": verification,
        "source_sha256": digest_file(Path(__file__)),
        "iid_directory": str(iid),
    }
    directory = blocked / "analysis"
    directory.mkdir(exist_ok=True)
    write_json(directory / "order_comparison.json", result)
    with (directory / "order_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["per_seed"][0]))
        writer.writeheader()
        writer.writerows(result["per_seed"])
    (directory / "routing_order_analysis.py").write_text(Path(__file__).read_text())
    _write_regime_acceptance(blocked, directory, new["runs"])
    _plot_comparison(blocked, iid, directory, new["runs"])
    _plot_timeline(blocked, directory)
    _write_comparison(directory, result)
    return result


def _write_regime_acceptance(blocked: Path, directory: Path, runs: list[dict[str, Any]]) -> None:
    """Expose joint-regime admission, including rare blocks affected by mixed boundary batches."""
    membership = json.loads((blocked / "membership.json").read_text())
    rows = []
    for run in runs:
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        path = blocked / f"seed_{run['seed']}" / run["arm"] / "selection.jsonl"
        for selection in _read_rows(path):
            for regime, values in selection["groups"]["regime"].items():
                counts[regime][0] += values["candidate"]
                counts[regime][1] += values["accepted"]
        for regime, (candidate, accepted) in sorted(counts.items(), key=lambda item: int(item[0])):
            rows.append(
                {
                    "seed": run["seed"],
                    "arm": run["arm"],
                    "regime": regime,
                    "name": membership["regime_names"][regime],
                    "candidate": candidate,
                    "accepted": accepted,
                    "rate": accepted / candidate,
                }
            )
    with (directory / "regime_acceptance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_comparison(blocked: Path, iid: Path, directory: Path, runs: list[dict[str, Any]]) -> None:
    """Use identical axes for solid regime-blocked and dashed IID trajectories."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    colors = {"random": "#566573", "high": "#c0392b", "low": "#2471a3"}
    metrics = (
        ("entropy_mean", "Fixed-panel router entropy"),
        ("heldout_mse", "Natural-query reconstruction MSE"),
        ("balanced_linear", "Balanced scene linear accuracy"),
        ("l2_load_cv", "Layer 2 clean load CV"),
        ("l3_load_cv", "Layer 3 clean load CV"),
        ("rankme", "Fixed-panel RankMe"),
    )
    for arm in ARMS:
        for root, style in ((blocked, "-"), (iid, "--")):
            traces = [
                _read_rows(root / f"seed_{r['seed']}" / arm / "evaluation.jsonl") for r in runs if r["arm"] == arm
            ]
            for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
                steps = [r["step"] for r in traces[0] if metric in r]
                values = np.array([[r[metric] for r in trace if metric in r] for trace in traces])
                mean = values.mean(0)
                ax.plot(steps, mean, color=colors[arm], linestyle=style, linewidth=1.6)
                if root == blocked and len(values) > 1:
                    sem = values.std(0, ddof=1) / np.sqrt(len(values))
                    ax.fill_between(steps, mean - sem, mean + sem, color=colors[arm], alpha=0.12)
                ax.set(title=title, xlabel="Optimizer updates")
                ax.grid(alpha=0.2)
    handles = [Line2D([0], [0], color=colors[arm], label=arm) for arm in ARMS]
    handles.extend(
        [
            Line2D([0], [0], color="black", linestyle=style, label=name)
            for style, name in (("-", "regime blocked"), ("--", "IID"))
        ]
    )
    fig.legend(handles=handles, ncols=5, loc="outside lower center")
    seed_count = len({r["seed"] for r in runs})
    fig.suptitle(f"Regime ordering versus IID: {seed_count} paired seeds; shading is regime ±1 SEM")
    fig.savefig(directory / "order_comparison.png", dpi=160)
    plt.close(fig)


def _plot_timeline(blocked: Path, directory: Path) -> None:
    """Show exact candidate spans for one representative seed; retain all seeds in the manifest."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    manifest = json.loads((blocked / "stream_manifest.json").read_text())
    seed = min(manifest["seeds"], key=int)
    regimes = manifest["regime_order"]
    fig, ax = plt.subplots(figsize=(12, max(4, len(regimes) * 0.32)), constrained_layout=True)
    for epoch in manifest["seeds"][seed]:
        offset = epoch["epoch"] * epoch["updates"]
        for segment in epoch["segments"]:
            index = regimes.index(segment["regime"])
            ax.broken_barh(
                [(offset + segment["start_update"], segment["stop_update"] - segment["start_update"])],
                (index - 0.4, 0.8),
                facecolors=plt.get_cmap("tab20")(index % 20),
            )
            if segment["stop_update"] - segment["start_update"] < 2:
                ax.plot(
                    offset + (segment["start_update"] + segment["stop_update"]) / 2,
                    index,
                    marker="|",
                    markersize=12,
                    color=plt.get_cmap("tab20")(index % 20),
                )
        if epoch["epoch"]:
            ax.axvline(offset, linestyle="--", color="black", linewidth=0.8)
    ax.set(
        yticks=range(len(regimes)),
        yticklabels=[manifest["regime_names"][str(r)] for r in regimes],
        xlabel="Optimizer updates (ticks mark regime blocks shorter than two updates)",
        title=f"Regime walk, seed {seed}: the same walk repeats each epoch",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(directory / "stream_timeline.png", dpi=160)
    plt.close(fig)


def _write_comparison(directory: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Regime-blocked versus shuffled BDD",
        "",
        "Settings, panels, software, initial weights, budgets and retained candidate membership are matched.",
        "",
        "| Filter | Relative MSE change (%) | Router-entropy change difference | "
        "kNN change (pp) | Linear change (pp) |",
        "| -- | --: | --: | --: | --: |",
    ]
    for arm in ARMS:
        cells = []
        for metric, factor, digits in (
            ("relative_mse", 100, 2),
            ("entropy_change", 1, 5),
            ("balanced_knn_pp", 1, 2),
            ("balanced_linear_pp", 1, 2),
        ):
            effect = result["blocked_minus_iid"][arm][metric]
            interval = effect["ci95"]
            cells.append(
                f"{factor * effect['mean']:+.{digits}f}"
                + (f" [{factor * interval[0]:+.{digits}f}, {factor * interval[1]:+.{digits}f}]" if interval else "")
            )
        lines.append("| " + arm + " | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Filter-by-order interaction",
            "",
            "Change in each filter's relative MSE penalty versus random, in percentage points.",
            "Negative values mean the filter's relative penalty shrank under regime ordering.",
            "",
            "| Filter versus random | Change in relative MSE penalty (pp) |",
            "| -- | --: |",
        ]
    )
    for name, data in result["filter_by_order_interaction"].items():
        effect = data["relative_mse_penalty_change_pp"]
        interval = effect["ci95"]
        value = f"{effect['mean']:+.2f}" + (f" [{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval else "")
        lines.append(f"| {name} | {value} |")
    lines.extend(
        [
            "",
            "Brackets are descriptive paired 95% t intervals across training seeds.",
            "Regime ordering is a label-defined non-IID proxy over still images, with two passes as in the control.",
            "Global panel/probe changes do not, by themselves, establish past-regime forgetting.",
            "",
            "![Regime timing](stream_timeline.png)",
            "",
            "![Order comparison](order_comparison.png)",
        ]
    )
    (directory / "order_comparison.md").write_text("\n".join(lines) + "\n")
