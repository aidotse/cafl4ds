"""Analyze seed-level WP-M0 MAE/BDD results and export scientific figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["dense_active", "dense_total", "moe_native"]
LABELS = ["Dense active", "Dense total", "Native MoE"]
COLORS = ["#2463a5", "#d88a19", "#167d60"]
METRICS = ["heldout_mse", "knn_acc", "linear_acc", "rankme", "generalization_gap"]


def read_csv(path):
    """Read optional numeric cells as NaN for plotting only."""
    with Path(path).open() as f:
        return [{k: float(v) if v else np.nan for k, v in r.items()} for r in csv.DictReader(f)]


def statistics(values):
    """Student-t uncertainty uses independent training seeds, never checkpoints."""
    values = np.asarray(values, dtype=float)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
    return {"values": values.tolist(), "mean": mean, "sd": sd, "ci95": [mean - half, mean + half]}


def save_figure(fig, directory, name):
    """Export both raster preview and editable vector figure."""
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    """Require complete primary runs and report parity, utilization, cost, and integration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    directory = root / "analysis"
    directory.mkdir(exist_ok=True)
    summaries, health, losses = {}, {}, {}
    for arm in ARMS:
        summaries[arm], health[arm], losses[arm] = [], [], []
        for seed in range(5):
            run = root / "runs" / arm / f"s{seed}"
            assert (run / "complete.json").exists(), f"Incomplete run: {run}"
            s = json.loads((run / "warm_summary.json").read_text())
            assert s["steps"] == 2000 and s["endpoint_reload"]["embeddings_exact"]
            summaries[arm].append(s)
            health[arm].append(read_csv(run / "health.csv"))
            losses[arm].append(read_csv(run / "losses.csv"))
    pairing = json.loads((root / "runs_pairing.json").read_text())
    assert len(pairing) == 5 and all(p["train_order_matches"] and p["sampled_views_and_masks_match"] for p in pairing)
    aggregate = {
        arm: {metric: statistics([s["endpoint"][metric] for s in summaries[arm]]) for metric in METRICS} for arm in ARMS
    }
    for arm in ARMS:
        for metric in [
            "training_seconds",
            "evaluation_seconds",
            "total_wall_seconds",
            "images_per_second",
            "seconds_per_update",
            "peak_mps_allocated_mib",
            "peak_mps_driver_mib",
        ]:
            aggregate[arm][metric] = statistics([s[metric] for s in summaries[arm]])
    comparisons = {}
    for baseline in ARMS[:2]:
        pairs = {}
        for metric in METRICS:
            moe = [s["endpoint"][metric] for s in summaries["moe_native"]]
            dense = [s["endpoint"][metric] for s in summaries[baseline]]
            delta = np.asarray(moe) - np.asarray(dense)
            if metric in {"knn_acc", "linear_acc"}:
                delta *= 100
            elif metric == "heldout_mse":
                delta = delta / np.asarray(dense) * 100
            pairs[metric] = statistics(delta)
        pairs["mse_parity"] = "supported" if pairs["heldout_mse"]["ci95"][1] <= 5 else "inconclusive_or_failed"
        for metric in ["knn_acc", "linear_acc"]:
            pairs[f"{metric}_parity"] = "supported" if pairs[metric]["ci95"][0] >= -3 else "inconclusive_or_failed"
        comparisons[baseline] = pairs
    smoke = {}
    for arm in ARMS:
        smoke[arm] = json.loads((root / "smoke" / arm / "complete.json").read_text())
    seed_rows = []
    for arm in ARMS:
        for s in summaries[arm]:
            row = {
                "arm": arm,
                "seed": s["seed"],
                "steps": s["steps"],
                "images_seen": s["images_seen"],
                **{k: s["endpoint"][k] for k in METRICS},
                "training_seconds": s["training_seconds"],
                "images_per_second": s["images_per_second"],
                "selected_well_step": s["well_selection"]["selected_step"],
                "well_status": s["well_selection"]["status"],
            }
            if arm == "moe_native":
                row.update(
                    {
                        k: v
                        for k, v in s["endpoint"].items()
                        if k.startswith(("route_", "dead_", "min_expert", "masked_dead", "expert_"))
                    }
                )
            seed_rows.append(row)
    columns = list(dict.fromkeys(k for row in seed_rows for k in row))
    with (directory / "seed_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(seed_rows)
    result = {
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "smoke": smoke,
        "pairing": pairing,
        "parity_tolerances": {"reconstruction_percent": 5, "canary_pp": -3},
        "complete_runs": 15,
        "training_seed_unit": True,
    }
    (directory / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ai, arm in enumerate(ARMS):
        for si in range(5):
            hs = health[arm][si]
            for ax, key in [(axs[0, 0], "heldout_mse"), (axs[0, 1], "knn_acc"), (axs[1, 0], "rankme")]:
                valid = [r for r in hs if np.isfinite(r.get(key, np.nan))]
                ax.plot(
                    [r["step"] for r in valid],
                    [r[key] * (100 if key == "knn_acc" else 1) for r in valid],
                    color=COLORS[ai],
                    alpha=0.2,
                    lw=0.9,
                )
            ls = losses[arm][si]
            axs[1, 1].plot(
                [r["step"] for r in ls][49::50],
                [np.mean([r["reconstruction_loss"] for r in ls][j - 50 : j]) for j in range(50, 2001, 50)],
                color=COLORS[ai],
                alpha=0.2,
            )
        for ax, key in [(axs[0, 0], "heldout_mse"), (axs[0, 1], "knn_acc"), (axs[1, 0], "rankme")]:
            keys = sorted(set(r["step"] for r in health[arm][0] if np.isfinite(r.get(key, np.nan))))
            means = [
                np.mean([next(r[key] for r in health[arm][si] if r["step"] == step) for si in range(5)])
                for step in keys
            ]
            ax.plot(
                keys, np.asarray(means) * (100 if key == "knn_acc" else 1), color=COLORS[ai], lw=2.1, label=LABELS[ai]
            )
        axs[1, 1].plot(
            range(50, 2001, 50),
            [
                np.mean([np.mean([r["reconstruction_loss"] for r in losses[arm][si]][j - 50 : j]) for si in range(5)])
                for j in range(50, 2001, 50)
            ],
            color=COLORS[ai],
            lw=2.1,
        )
    for ax, title, ylabel in zip(
        axs.flatten(),
        [
            "Fixed held-out reconstruction",
            "Frozen kNN canary",
            "Representation effective rank",
            "Training reconstruction (50-step mean)",
        ],
        ["Normalized-pixel MSE", "Accuracy (%)", "RankMe", "Normalized-pixel MSE"],
        strict=False,
    ):
        ax.set(title=title, xlabel="Optimizer updates", ylabel=ylabel)
    axs[0, 1].axhline(25, color="gray", ls="--", lw=1, label="Chance")
    axs[0, 0].legend(frameon=False)
    fig.suptitle("MAE / BDD train · MPS · five seeds per architecture\nBold: seed mean; faint: individual seeds")
    save_figure(fig, directory, "learning_curves")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for j, baseline in enumerate(ARMS[:2]):
        for seed in range(5):
            axs[0].scatter(
                j + (seed - 2) * 0.035, comparisons[baseline]["heldout_mse"]["values"][seed], color=COLORS[j], s=35
            )
        for k, metric in enumerate(["knn_acc", "linear_acc"]):
            x = j * 3 + k
            stat = comparisons[baseline][metric]
            axs[1].scatter(np.full(5, x) + np.linspace(-0.1, 0.1, 5), stat["values"], color=COLORS[j])
            axs[1].errorbar(
                x,
                stat["mean"],
                yerr=[[stat["mean"] - stat["ci95"][0]], [stat["ci95"][1] - stat["mean"]]],
                color="black",
                fmt="_",
                capsize=4,
            )
        stat = comparisons[baseline]["heldout_mse"]
        axs[0].errorbar(
            j,
            stat["mean"],
            yerr=[[stat["mean"] - stat["ci95"][0]], [stat["ci95"][1] - stat["mean"]]],
            color="black",
            fmt="_",
            capsize=4,
        )
    axs[0].axhline(5, color="#ba3535", ls="--", label="5% parity boundary")
    axs[1].axhline(-3, color="#ba3535", ls="--", label="−3 pp parity boundary")
    for ax in axs:
        ax.axhline(0, color="gray", lw=0.8)
        ax.legend(frameon=False)
    axs[0].set(
        xticks=[0, 1], xticklabels=LABELS[:2], ylabel="MoE minus dense MSE (%)", title="Reconstruction: lower is better"
    )
    axs[1].set(
        xticks=[0, 1, 3, 4],
        xticklabels=["Active\nkNN", "Active\nLinear", "Total\nkNN", "Total\nLinear"],
        ylabel="MoE minus dense accuracy (pp)",
        title="Canary: higher is better",
    )
    fig.suptitle("Paired seed differences · bars show descriptive 95% Student-t intervals")
    save_figure(fig, directory, "paired_differences")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for j, layer in enumerate([2, 3]):
        loads = np.array(
            [[s["endpoint"][f"route_load_l{layer}_e{e}"] for e in range(4)] for s in summaries["moe_native"]]
        )
        image = axs[j].imshow(loads, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for seed in range(5):
            for e in range(4):
                axs[j].text(
                    e,
                    seed,
                    f"{loads[seed, e] * 100:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if loads[seed, e] > 0.5 else "black",
                )
        axs[j].set(
            title=f"MoE block {layer} (zero-based)",
            xticks=range(4),
            yticks=range(5),
            xlabel="Expert",
            ylabel="Training seed",
        )
        axs[j].grid(False)
    fig.colorbar(image, ax=axs, label="Fraction of 6,400 held-out patch decisions")
    fig.suptitle("Deterministic expert use at update 2,000 · CLS excluded")
    save_figure(fig, directory, "expert_loads")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for j, (key, ylabel) in enumerate(
        [("seconds_per_update", "Synchronized seconds / update"), ("images_per_second", "Training images / second")]
    ):
        means = [aggregate[a][key]["mean"] for a in ARMS]
        sd = [aggregate[a][key]["sd"] for a in ARMS]
        axs[j].bar(LABELS, means, yerr=sd, color=COLORS, alpha=0.85, capsize=4)
        axs[j].set(ylabel=ylabel)
    fig.suptitle("Observed training cost · includes controlled RNG overhead; excludes evaluation")
    save_figure(fig, directory, "training_cost")

    lines = [
        "# WP-M0 MAE/BDD native warm-well results",
        "",
        "All 15 primary runs completed 2,000 optimizer updates on MPS.",
        "",
        "| Arm | Held-out MSE | kNN (%) | Linear (%) | RankMe | Train seconds | Images/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, label in zip(ARMS, LABELS, strict=False):
        a = aggregate[arm]
        cells = []
        for key, mult, digits in [
            ("heldout_mse", 1, 4),
            ("knn_acc", 100, 1),
            ("linear_acc", 100, 1),
            ("rankme", 1, 2),
            ("training_seconds", 1, 1),
            ("images_per_second", 1, 0),
        ]:
            cells.append(f"{a[key]['mean'] * mult:.{digits}f} ± {a[key]['sd'] * mult:.{digits}f}")
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Values are seed mean ± sample standard deviation. Pure held-out reconstruction is averaged over eight fixed masks/augmentation draws per image.",
        "",
        "## Paired comparison",
        "",
        "| Dense reference | MoE MSE difference (%) | MoE kNN difference (pp) | MoE linear difference (pp) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for baseline, label in zip(ARMS[:2], LABELS[:2], strict=False):
        cells = []
        for metric in ["heldout_mse", "knn_acc", "linear_acc"]:
            s = comparisons[baseline][metric]
            cells.append(f"{s['mean']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]")
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Intervals use five paired seed outcomes. Checkpoint rows and mask draws are not independent replicates.",
        "",
        "The predefined practical boundaries are +5% reconstruction MSE and −3 percentage points canary accuracy. "
        "A mean within the boundary is insufficient if its interval crosses the boundary.",
        "",
    ]
    for baseline, label in zip(ARMS[:2], LABELS[:2], strict=False):
        p = comparisons[baseline]
        lines.append(
            f"- Against {label}: MSE parity {p['mse_parity']}; kNN parity {p['knn_acc_parity']}; linear parity {p['linear_acc_parity']}."
        )
    lines += ["", "## Well selection and expert use", ""]
    for arm, label in zip(ARMS, LABELS, strict=False):
        selected = [s["well_selection"]["selected_step"] for s in summaries[arm]]
        lines.append(
            f"- {label}: certified plateau steps by seed: {selected}. A missing step is cap-limited, not proof of convergence."
        )
    for layer in [2, 3]:
        s = summaries["moe_native"]
        dead = [v["endpoint"][f"dead_expert_frac_l{layer}"] * 4 for v in s]
        masked = [v["endpoint"][f"masked_dead_expert_frac_l{layer}"] * 4 for v in s]
        minimum = [v["endpoint"][f"min_expert_load_l{layer}"] * 100 for v in s]
        cka = [v["endpoint"][f"expert_output_cka_l{layer}"] for v in s]
        lines.append(
            f"- Block {layer}: full-image dead experts {dead}; masked-query dead experts {masked}; minimum full-image patch loads (%) {[round(x, 2) for x in minimum]}; mean expert-output CKA {np.mean(cka):.4f}."
        )
    lines.append(
        f"- Every expert received patch assignments in the final 200 training updates: {all(s['all_experts_trained_final_200'] for s in summaries['moe_native'])}."
    )
    lines += ["", "## Integration acceptance", ""]
    for arm, label in zip(ARMS, LABELS, strict=False):
        manifest = json.loads((root / "smoke" / arm / "corpus/manifest.json").read_text())
        lines.append(
            f"- {label}: Tier A {smoke[arm]['tier_a']}, Tier B {smoke[arm]['tier_b']}; certified plateau well used: {manifest['certified_well']}."
        )
    lines += [
        "",
        "The full training pool was used for each seed-0 accept-all integration walk, with a health read every 50 batches. "
        "These loose sanity checks do not establish parity or forgetting protection.",
        "",
        "## Protocol and limits",
        "",
        "- BDD train: 61,555 usable images, four scenes, 18 regimes; fixed support/query sets have 100 images each. Training pool: 61,355.",
        "- Actual image order hashes match across all three arms for each seed. Augmentation and masking hashes match at the first five and every 100th update.",
        "- Native MoE uses four experts, top-1 token routing, blocks 2/3, batch balancing, and auxiliary coefficient 0.01. Dense total capacity differs by 0.25%.",
        "- Training cost includes per-step RNG isolation and MoE router-noise isolation. It excludes evaluation and checkpoint writes. It is an observed experiment-run cost, not an optimized sparsity benchmark.",
        "- MPS memory figures are sampled allocator/driver maxima, not continuous peaks; RSS is the process lifetime peak in a sequential ensemble, not an independent per-arm peak.",
        "- EMA/bias balancing, upcycling, and JE cross-view consistency remain deferred. Routing measurements are candidate signals.",
        "- Fixed-budget endpoint quality is distinct from certified warm-well convergence. Five seeds and 100 fixed query images provide limited precision.",
        "- The training reference panel is sampled from the natural-frequency pool, while the query panel is scene-balanced. Their reconstruction gap is descriptive and can reflect composition differences; it is not a clean overfitting estimate.",
        "- The deployment smoke uses an endpoint checkpoint when seed 0 has no certified plateau. Its Tier-A memory-flat check is not enabled; finite signals, channel completeness, and trust completeness are checked.",
        "",
        "![Learning curves](learning_curves.png)",
        "",
        "![Paired differences](paired_differences.png)",
        "",
        "![Expert loads](expert_loads.png)",
        "",
        "![Training cost](training_cost.png)",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines))
    print(json.dumps({"aggregate": aggregate, "paired_comparisons": comparisons, "smoke": smoke}, indent=2))


if __name__ == "__main__":
    main()
