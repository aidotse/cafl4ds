"""Post-run analysis of routing, selection composition, expert use and paired learning effects."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from cafl4ds.routing_experiment import ARMS, digest_file, write_json


def records(path: Path) -> Iterator[dict[str, Any]]:
    """Read potentially large selection logs a row at a time."""
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a rectangular, easily reusable table."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def acceptance_rates(root: Path, membership: dict[str, Any]) -> list[dict[str, Any]]:
    """Audit scene, weather and time-of-day only after admission decisions have been made."""
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in records(root / "selection.jsonl"):
        for scene, values in row["groups"]["scene"].items():
            target = counts[("scene", membership["scene_names"][scene])]
            target[0] += values["candidate"]
            target[1] += values["accepted"]
        for regime, values in row["groups"]["regime"].items():
            name = membership["regime_names"][regime]
            if "·" in name:
                time_of_day, weather = name.split("·", 1)
                for axis, group in (("time_of_day", time_of_day), ("weather", weather)):
                    target = counts[(axis, group)]
                    target[0] += values["candidate"]
                    target[1] += values["accepted"]
    return [
        {"axis": axis, "group": group, "candidate": total, "accepted": admitted, "rate": admitted / total}
        for (axis, group), (total, admitted) in sorted(counts.items())
    ]


def _endpoint(row: dict[str, Any]) -> dict[str, Any]:
    """Keep one record per independent training seed and arm, with an init-matched B5 read."""
    final, initial = row["final"], row["initial"]
    layers = sorted(k.removesuffix("_minimum_load") for k in final if k.endswith("_minimum_load"))
    values = {
        "seed": row["seed"],
        "arm": row["arm"],
        "steps": row["steps"],
        "entropy": final["entropy_mean"],
        "entropy_change": row["entropy_change"],
        "heldout_mse": final["heldout_mse"],
        "balanced_mse": final["balanced_mse"],
        "balanced_knn": final["balanced_knn"],
        "balanced_linear": final["balanced_linear"],
        "natural_knn": final["natural_knn"],
        "natural_linear": final["natural_linear"],
        "initial_mse": initial["heldout_mse"],
        "initial_knn": initial["balanced_knn"],
        "initial_linear": initial["balanced_linear"],
        "knn_gain_over_b5_pp": 100 * (final["balanced_knn"] - initial["balanced_knn"]),
        "linear_gain_over_b5_pp": 100 * (final["balanced_linear"] - initial["balanced_linear"]),
        "rankme": final["rankme"],
        "cosine_drift": final["cosine_drift"],
        "entropy_reconstruction_spearman": final["entropy_reconstruction_spearman"],
        "unique_accepted": row["unique_accepted"],
        "unique_training_coverage": row["unique_accepted"] / row["train_pool"],
        **{name + "_seconds": value for name, value in row["elapsed_seconds"].items()},
    }
    for layer in layers:
        for suffix in (
            "minimum_load",
            "dead_fraction",
            "load_entropy",
            "route_change_initial",
            "rolling_starved_fraction",
        ):
            values[layer + "_" + suffix] = final.get(layer + "_" + suffix)
    sensitivity = final["mask_sensitivity"]
    values["mask_rank_correlation"] = sensitivity["spearman_each_then_mean"][-1] if sensitivity["defined"] else None
    values["mask_half_overlap"] = (
        sensitivity["half_overlap_each_then_mean"]["high"][-1] if sensitivity["defined"] else None
    )
    return values


def _group_means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize seed-level measurements; checkpoint/token counts are never replicate counts."""
    summary = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        stats = {}
        for name in selected[0]:
            if name in {"arm", "seed"}:
                continue
            values = [r[name] for r in selected if r[name] is not None]
            stats[name] = {
                "mean": float(np.mean(values)) if values else None,
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "n_seeds": len(values),
            }
        summary[arm] = stats
    return summary


def analyze(output: Path) -> dict[str, Any]:
    """Require the complete frozen ensemble, then write reusable tables, figures and a report."""
    output = output.resolve()
    comparison = json.loads((output / "comparison.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    membership = json.loads((output / "membership.json").read_text())
    runs = comparison["runs"]
    expected = {(int(seed), arm) for seed in protocol["config"]["seeds"] for arm in ARMS}
    if {(r["seed"], r["arm"]) for r in runs} != expected:
        raise ValueError("Analysis requires every planned seed and all three arms.")
    directory = output / "analysis"
    directory.mkdir(exist_ok=True)
    endpoints = [_endpoint(row) for row in runs]
    _write_csv(directory / "seed_summary.csv", endpoints)
    rates: list[dict[str, Any]] = []
    trajectories = {}
    training = {}
    for run in runs:
        seed, arm = run["seed"], run["arm"]
        root = output / f"seed_{seed}" / arm
        rates.extend({"seed": seed, "arm": arm, **row} for row in acceptance_rates(root, membership))
        trajectories[(seed, arm)] = list(records(root / "evaluation.jsonl"))
        training[(seed, arm)] = list(records(root / "training.jsonl"))
    _write_csv(directory / "acceptance_rates.csv", rates)
    grouped = _group_means(endpoints)
    report = {
        "arms": grouped,
        "contrasts": comparison["contrasts"],
        "protocol_hash": protocol["protocol_hash"],
        "replicate_unit": "training seed",
        "analysis_source_sha256": digest_file(Path(__file__)),
    }
    write_json(directory / "summary.json", report)
    shutil.copyfile(Path(__file__), directory / "routing_analysis.py")
    _plot_diagnostics(directory, trajectories, training)
    _plot_experts(directory, trajectories)
    _plot_acceptance(directory, rates)
    _plot_selection_entropy(output, directory, sorted({r["seed"] for r in runs}))
    _write_report(directory, grouped, comparison["contrasts"], protocol)
    return report


def _plot_diagnostics(
    directory: Path,
    traces: dict[tuple[int, str], list[dict[str, Any]]],
    training: dict[tuple[int, str], list[dict[str, Any]]],
) -> None:
    """Plot model-side changes separately from mechanically selected-set entropy."""
    import matplotlib  # noqa: PLC0415 - optional headless plotting

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    colors = {"random": "#566573", "high": "#c0392b", "low": "#2471a3"}
    for arm in ARMS:
        matching = [rows for (_, policy), rows in traces.items() if policy == arm]
        specs = (
            ("entropy_mean", "Fixed-panel normalized entropy"),
            ("heldout_mse", "Natural held-out reconstruction MSE"),
            ("entropy_reconstruction_spearman", "Entropy vs reconstruction error (Spearman)"),
        )
        for ax, (key, title) in zip(axes[0], specs, strict=True):
            steps = [r["step"] for r in matching[0]]
            values = np.array([[r[key] for r in rows] for rows in matching], dtype=float)
            ax.plot(steps, values.mean(0), color=colors[arm], label=arm)
            ax.set(title=title, xlabel="Optimizer updates")
            ax.grid(alpha=0.2)
        tr = [rows for (_, policy), rows in training.items() if policy == arm]
        ends = list(range(50, len(tr[0]) + 1, 50))
        layers = sorted(
            k.removesuffix("_selected_probability") for k in tr[0][0] if k.endswith("_selected_probability")
        )
        for ax, keys, title in (
            (axes[1, 0], ["reconstruction"], "Training reconstruction (50-update mean)"),
            (axes[1, 1], [layer + "_selected_probability" for layer in layers], "Noisy training selected gate weight"),
            (axes[1, 2], [layer + "_router_grad_norm" for layer in layers], "Router gradient norm before clipping"),
        ):
            values = [
                [np.mean([row[key] for row in rows[end - 50 : end] for key in keys]) for end in ends] for rows in tr
            ]
            ax.plot(ends, np.mean(values, axis=0), color=colors[arm], label=arm)
            ax.set(title=title, xlabel="Optimizer updates")
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.savefig(directory / "routing_and_learning.png", dpi=160)
    plt.close(fig)


def _plot_experts(directory: Path, traces: dict[tuple[int, str], list[dict[str, Any]]]) -> None:
    """Keep expert IDs within seed; averaging arbitrary expert identities across seeds hides starvation."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    seeds = sorted({seed for seed, _ in traces})
    layers = sorted(
        k.removesuffix("_load")
        for k, value in next(iter(traces.values()))[0].items()
        if k.endswith("_load") and isinstance(value, list)
    )
    for layer in layers:
        fig, axes = plt.subplots(len(seeds), 3, squeeze=False, figsize=(12, 2.1 * len(seeds)), constrained_layout=True)
        for row, seed in enumerate(seeds):
            for col, arm in enumerate(ARMS):
                values = traces[(seed, arm)]
                loads = np.array([r[layer + "_load"] for r in values])
                axes[row, col].stackplot(
                    [r["step"] for r in values], loads.T, labels=[f"Expert {i}" for i in range(loads.shape[1])]
                )
                axes[row, col].set(ylim=(0, 1), title=f"Seed {seed} · {arm}", ylabel="Patch assignment share")
                if row == len(seeds) - 1:
                    axes[row, col].set_xlabel("Optimizer updates")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncols=4, fontsize=8, loc="outside lower center")
        fig.suptitle(f"Layer {layer.removeprefix('l')}: deterministic full-image routing; expert IDs are seed-specific")
        fig.savefig(directory / f"{layer}_expert_loads.png", dpi=160)
        plt.close(fig)


def _plot_acceptance(directory: Path, rows: list[dict[str, Any]]) -> None:
    """Show attribute bias as mean within-seed acceptance, relative to the 50% global budget."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    axes_names = sorted({r["axis"] for r in rows})
    fig, axes = plt.subplots(
        1, len(axes_names), squeeze=False, figsize=(5 * len(axes_names), 5), constrained_layout=True
    )
    for ax, axis in zip(axes.flat, axes_names, strict=True):
        groups = sorted({r["group"] for r in rows if r["axis"] == axis})
        values = np.array(
            [
                [
                    np.mean([r["rate"] for r in rows if r["axis"] == axis and r["group"] == group and r["arm"] == arm])
                    for arm in ARMS
                ]
                for group in groups
            ]
        )
        rendered = ax.imshow(values, vmin=0, vmax=1, cmap="RdBu_r", aspect="auto")
        for row, _group in enumerate(groups):
            for col in range(3):
                ax.text(
                    col,
                    row,
                    f"{100 * values[row, col]:.0f}%",
                    ha="center",
                    va="center",
                    color="white" if abs(values[row, col] - 0.5) > 0.28 else "black",
                )
        ax.set(
            xticks=range(3),
            xticklabels=ARMS,
            yticks=range(len(groups)),
            yticklabels=groups,
            title=axis.replace("_", " "),
        )
    fig.colorbar(rendered, ax=axes.ravel().tolist(), label="Within-group acceptance rate")
    fig.savefig(directory / "selection_bias.png", dpi=160)
    plt.close(fig)


def _format_stat(stats: dict[str, Any], key: str, factor: float = 1, digits: int = 3) -> str:
    """Format seed means and sample SDs, without implying an extra replicate count."""
    row = stats[key]
    if row["mean"] is None:
        return "undefined"
    value = f"{row['mean'] * factor:.{digits}f}"
    return value + (f" ± {row['sd'] * factor:.{digits}f}" if row["sd"] is not None else "")


def _plot_selection_entropy(output: Path, directory: Path, seeds: list[int]) -> None:
    """Show the admission rule's mechanical effect separately from fixed-reference model changes."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, arm in zip(axes, ARMS, strict=True):
        traces = [list(records(output / f"seed_{seed}" / arm / "selection.jsonl")) for seed in seeds]
        width = min(50, len(traces[0]))
        ends = list(range(width, len(traces[0]) + 1, width))
        if ends[-1] != len(traces[0]):
            ends.append(len(traces[0]))
        for key, color in (("candidate_mean", "#566573"), ("accepted_mean", "#c0392b"), ("rejected_mean", "#2471a3")):
            values = np.array(
                [[np.mean([r[key] for r in trace[max(0, end - width) : end]]) for end in ends] for trace in traces]
            )
            mean = values.mean(0)
            ax.plot(ends, mean, label=key.removesuffix("_mean"), color=color)
            if len(values) > 1:
                sem = values.std(0, ddof=1) / np.sqrt(len(values))
                ax.fill_between(ends, mean - sem, mean + sem, color=color, alpha=0.15)
        ax.set(title=arm, xlabel="Optimizer updates", ylabel="Normalized routing entropy")
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle(
        "Incoming and admitted entropy: rolling means, ±1 SEM across seeds\nEach panel uses its own y-axis scale"
    )
    fig.savefig(directory / "selection_entropy.png", dpi=160)
    plt.close(fig)


def _write_report(
    directory: Path, grouped: dict[str, Any], contrasts: dict[str, Any], protocol: dict[str, Any]
) -> None:
    """Describe observations and their limits without selecting a favorable metric after seeing results."""
    lines = [
        "# MoE routing entropy filtering: measured comparison",
        "",
        "Values are seed mean ± sample standard deviation.",
        "",
        "| Arm | Entropy change | Natural MSE | Balanced kNN (%) | Balanced linear (%) | "
        "Unique training coverage (%) |",
        "| -- | --: | --: | --: | --: | --: |",
    ]
    for arm in ARMS:
        stats = grouped[arm]
        values = [
            _format_stat(stats, "entropy_change", digits=4),
            _format_stat(stats, "heldout_mse", digits=4),
            _format_stat(stats, "balanced_knn", 100, 1),
            _format_stat(stats, "balanced_linear", 100, 1),
            _format_stat(stats, "unique_training_coverage", 100, 1),
        ]
        lines.append("| " + arm + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Paired contrasts",
        "",
        "Descriptive 95% t intervals across paired training seeds. These are not multiplicity-adjusted tests.",
        "",
        "| Contrast | Entropy-change difference | Relative MSE difference (%) | kNN difference (pp) | "
        "Linear difference (pp) |",
        "| -- | --: | --: | --: | --: |",
    ]
    for contrast, data in contrasts.items():
        cells = []
        for key, scale in (
            ("entropy_change_difference", 1),
            ("relative_mse_difference", 100),
            ("balanced_knn_pp", 1),
            ("balanced_linear_pp", 1),
        ):
            effect = data["effects"][key]
            interval = effect["ci95"]
            digits = 5 if key == "entropy_change_difference" else 3
            cells.append(
                f"{scale * effect['mean']:+.{digits}f}"
                + (f" [{scale * interval[0]:+.{digits}f}, {scale * interval[1]:+.{digits}f}]" if interval else "")
            )
        lines.append("| " + contrast + " | " + " | ".join(cells) + " |")
    config = protocol["config"]
    per_epoch = len(protocol["panels"]["train"]) // config["batch_size"]
    steps = per_epoch * config["epochs"]
    if config["max_steps"] is not None:
        steps = min(steps, config["max_steps"])
    lines += [
        "",
        "## Checks and interpretation",
        "",
        f"Frozen protocol: `{protocol['protocol_hash']}`.",
        "Paired initial weights and full candidate orders were verified. "
        f"Each run makes {steps} updates with exactly half-batch admission.",
        "The fixed reference panel, rather than selected-set entropy, measures model routing changes.",
        "Expert-load plots keep seed-specific expert identities; entropy does not establish a healthy direction.",
        "The initial checkpoint is each seed's frozen-random B5. "
        "Its reconstruction and canary values are retained in seed_summary.csv.",
        "Selection bias is audited after decisions using scene/weather/time attributes; "
        "those attributes never score training images.",
        "These small-model results do not establish forgetting, semantic expert specialization, "
        "or a benefit from health-based control.",
        "Resource figures include instrumentation. "
        "The random selector would not require an encoder scoring pass in deployment.",
        "",
        "![Routing and learning](routing_and_learning.png)",
        "",
        "![Candidate, accepted and rejected entropy](selection_entropy.png)",
        "",
        "![Selection bias](selection_bias.png)",
    ]
    for path in sorted(directory.glob("l*_expert_loads.png")):
        lines += ["", f"![{path.stem}]({path.name})"]
    (directory / "results.md").write_text("\n".join(lines) + "\n")
