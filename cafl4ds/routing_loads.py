"""Exploratory expert-load statistics, keeping layers and training seeds separate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from cafl4ds.routing_experiment import ARMS, digest_file


def distribution_statistics(values: Sequence[float]) -> dict[str, float]:
    """Summarize a hard load, soft mass, or assignment-count vector without expert-label dependence.

    CV uses population SD. Entropy is normalized by log(E); effective expert count
    is exp(unnormalized entropy), a concentration measure rather than a count of active experts.
    """
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Expert loads must be finite nonnegative vectors with positive mass and E >= 2.")
    p = p / p.sum()
    positive = p[p > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    return {
        "max_share": float(p.max()),
        "min_share": float(p.min()),
        "cv": float(p.std() / p.mean()),
        "entropy": entropy / math.log(len(p)),
        "effective_experts": math.exp(entropy),
        "inactive_fraction": float((p == 0).mean()),
    }


def time_average(steps: Sequence[int], values: Sequence[float]) -> float:
    """Trapezoidal average over recorded steps; unequal checkpoint spacing must not imply extra weight."""
    x, y = np.asarray(steps), np.asarray(values, dtype=float)
    if len(x) == 0 or x.shape != y.shape or not np.isfinite(y).all() or (np.diff(x) <= 0).any():
        raise ValueError("Expected aligned finite values at strictly increasing steps.")
    if len(x) == 1:
        return float(y[0])
    return float(((y[:-1] + y[1:]) * np.diff(x) / 2).sum() / (x[-1] - x[0]))


def _run_statistics(
    seed: int, arm: str, layer: str, evaluations: list[dict[str, Any]], training: list[dict[str, Any]]
) -> dict[str, Any]:
    """Separate post-initialization fixed-panel trajectories, endpoints, and real training assignments."""
    post = [row for row in evaluations if row["step"] > 0]
    if not post or not training:
        raise ValueError("Load analysis requires post-initialization measurements and training counts.")
    steps = [row["step"] for row in post]
    hard = [distribution_statistics(row[layer + "_load"]) for row in post]
    batches = np.array([row["train_counts"][layer] for row in training], dtype=float)
    batch_stats = [distribution_statistics(row) for row in batches]
    pooled = distribution_statistics(batches.sum(0))
    soft = distribution_statistics(post[-1][layer + "_soft_mass"])
    peak_index = int(np.argmax([row["max_share"] for row in hard]))
    return {
        "seed": seed,
        "arm": arm,
        "layer": layer,
        "expert_count": len(post[-1][layer + "_load"]),
        "window_start": steps[0],
        "window_end": steps[-1],
        "post_init_reads": len(post),
        **{"endpoint_" + key: value for key, value in hard[-1].items()},
        "time_mean_cv": time_average(steps, [r["cv"] for r in hard]),
        "time_mean_entropy": time_average(steps, [r["entropy"] for r in hard]),
        "time_mean_max_share": time_average(steps, [r["max_share"] for r in hard]),
        "peak_max_share": hard[peak_index]["max_share"],
        "peak_step": steps[peak_index],
        "majority_reads": sum(r["max_share"] > 0.5 for r in hard),
        "inactive_reads": sum(r["inactive_fraction"] > 0 for r in hard),
        "inactive_steps": [step for step, r in zip(steps, hard, strict=True) if r["inactive_fraction"] > 0],
        "majority_endpoint": hard[-1]["max_share"] > 0.5,
        "soft_endpoint_max_share": soft["max_share"],
        "soft_endpoint_entropy": soft["entropy"],
        "training_pooled_cv": pooled["cv"],
        "training_pooled_max_share": pooled["max_share"],
        "training_batch_mean_cv": float(np.mean([r["cv"] for r in batch_stats])),
        "training_batch_max_cv": max(r["cv"] for r in batch_stats),
        "training_inactive_batches": sum(r["inactive_fraction"] > 0 for r in batch_stats),
    }


def _aggregate(rows: list[dict[str, Any]], layers: list[str]) -> dict[str, Any]:
    """Average label-invariant scalars across seeds, never expert identity columns."""
    metrics = (
        "endpoint_max_share",
        "endpoint_min_share",
        "endpoint_cv",
        "endpoint_entropy",
        "endpoint_effective_experts",
        "time_mean_cv",
        "time_mean_entropy",
        "time_mean_max_share",
        "peak_max_share",
        "soft_endpoint_max_share",
        "soft_endpoint_entropy",
        "training_pooled_cv",
        "training_pooled_max_share",
        "training_batch_mean_cv",
        "training_batch_max_cv",
    )
    summary: dict[str, Any] = {}
    for layer in layers:
        summary[layer] = {}
        for arm in ARMS:
            group = [r for r in rows if r["layer"] == layer and r["arm"] == arm]
            statistics: dict[str, Any] = {
                key: {
                    "mean": float(np.mean([r[key] for r in group])),
                    "sd": float(np.std([r[key] for r in group], ddof=1)) if len(group) > 1 else None,
                }
                for key in metrics
            }
            statistics.update(
                {
                    "n_seeds": len(group),
                    "post_init_reads_per_seed": [r["post_init_reads"] for r in group],
                    "runs_with_majority": sum(r["majority_reads"] > 0 for r in group),
                    "majority_endpoint_runs": sum(r["majority_endpoint"] for r in group),
                    "runs_with_inactive_read": sum(r["inactive_reads"] > 0 for r in group),
                    "inactive_read_count": sum(r["inactive_reads"] for r in group),
                    "training_inactive_batch_count": sum(r["training_inactive_batches"] for r in group),
                }
            )
            summary[layer][arm] = statistics
    return summary


def _paired_effects(rows: list[dict[str, Any]], layers: list[str]) -> dict[str, Any]:
    """Descriptive paired t intervals for post-hoc load contrasts; seeds are the replicates."""
    contrasts: dict[str, Any] = {}
    for layer in layers:
        contrasts[layer] = {}
        for arm in ("high", "low"):
            pairs = []
            for row in rows:
                if row["layer"] != layer or row["arm"] != arm:
                    continue
                base = next(
                    r for r in rows if r["seed"] == row["seed"] and r["arm"] == "random" and r["layer"] == layer
                )
                pairs.append(
                    {
                        "seed": row["seed"],
                        **{
                            key: row[key] - base[key]
                            for key in ("time_mean_cv", "endpoint_cv", "endpoint_max_share", "endpoint_entropy")
                        },
                    }
                )
            effects = {}
            for key in ("time_mean_cv", "endpoint_cv", "endpoint_max_share", "endpoint_entropy"):
                values = np.array([r[key] for r in pairs])
                mean = float(values.mean())
                half = (
                    float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
                    if len(values) > 1
                    else None
                )
                effects[key] = {"mean": mean, "ci95": [mean - half, mean + half] if half is not None else None}
            contrasts[layer][arm + "_minus_random"] = {"paired_seeds": pairs, "effects": effects}
    return contrasts


def load_statistics(
    evaluations: Mapping[tuple[int, str], list[dict[str, Any]]],
    training: Mapping[tuple[int, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Summarize existing loads without retraining or changing the original endpoint protocol."""
    if not evaluations or set(evaluations) != set(training):
        raise ValueError("Evaluation/training runs must match and be nonempty.")
    seeds = sorted({seed for seed, _ in evaluations})
    if set(evaluations) != {(seed, arm) for seed in seeds for arm in ARMS}:
        raise ValueError("Load comparisons require all three arms within every seed.")
    first = next(iter(evaluations.values()))[0]
    layers = sorted(
        key.removesuffix("_load") for key, value in first.items() if key.endswith("_load") and isinstance(value, list)
    )
    if not layers:
        raise ValueError("No per-layer expert load vectors found.")
    rows = [
        _run_statistics(seed, arm, layer, evaluations[(seed, arm)], training[(seed, arm)])
        for seed, arm in sorted(evaluations)
        for layer in layers
    ]
    return {
        "analysis_type": "post-hoc descriptive; not a new training run or preregistered primary outcome",
        "replicate_unit": "training seed",
        "source_sha256": digest_file(Path(__file__)),
        "definitions": {
            "hard_load": "fraction of fixed-panel patch argmax assignments, excluding CLS",
            "balanced_cv": 0,
            "balanced_entropy": 1,
            "window": "first positive evaluation step through endpoint; excludes random-initialization step zero",
            "time_average": "trapezoidal interpolation across recorded steps; not a continuous observation",
            "majority": "one expert gets strictly more than half of fixed-panel patch assignments",
            "inactive": "zero patch assignments on a panel or training batch, not proof of no gradients",
            "training": "actual noisy masked augmented accepted-image assignments; pooled and per-batch",
            "intervals": "descriptive paired 95% t intervals; exploratory and not multiplicity-adjusted",
        },
        "per_seed": rows,
        "summary": _aggregate(rows, layers),
        "contrasts": _paired_effects(rows, layers),
    }
