"""Post-hoc reader certification for P0.6 (audit C1 + C3, [analysis], no new runs).

Reads the promoted P0.6.0 A-only gates (eb40 `aonly_nseed/` and eb60 `deepen_eb60/`)
and scores the label-free projector readers against a *healthy-vs-healthy seed-noise
null* — the P0.5.2 discipline P0.6 skipped. For each reader it reports:

  * fires (matched PC-vs-healthy margin > 0) across the five seeds;
  * the effect ratio mean|matched margin| / mean(healthy-vs-healthy |pairwise diff|),
    i.e. how many x the PC-vs-healthy separation sits above the no-forgetting null
    (the P0.5.2 "Nx above the healthy/healthy null" figure);
  * how many matched margins clear the null mean / null max;
  * the Spearman rank correlation of the matched margin with crater depth (-pc_bwt)
    across the five seeds — does a deeper crater move the reader more (quality/
    magnitude) or is the fire flat vs depth (a far-shift regime fingerprint)?

n=5, so the Spearman rho is coarse (not significant); read it as directional only.

Run: `python analyze.py` from the repo root (paths are relative to it).
"""

from __future__ import annotations

import glob
import json
import os
from itertools import combinations

EB40 = "docs/experiments/phase0/artifacts/P0.6.0/aonly_nseed/comparison_aonly_seed*.json"
EB60 = "docs/experiments/phase0/artifacts/P0.6.0/deepen_eb60/*seed*.json"

# reader key -> (pc field, healthy field); alignment/uniformity rise & drift are all
# "PC > healthy = fires" (rises with forgetting), scored on the same zero-margin rule.
READERS = {
    "cka_drift": ("pc_cka_drift", "healthy_cka_drift"),
    "cosine_drift": ("pc_cosine_drift", "healthy_cosine_drift"),
    "alignment_proj": ("proj_align_pc_rise", "proj_align_healthy_rise"),
    "uniformity_proj": ("proj_uniformity_pc_rise", "proj_uniformity_healthy_rise"),
}


def _mean(xs):
    return sum(xs) / len(xs)


def _spearman(a, b):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        rk = [0] * len(xs)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk

    ra, rb = rank(a), rank(b)
    n = len(a)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def _load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        g = json.load(open(f))["gate"]
        r = g["reported"]
        row = {"pc_bwt": g["pc_backward_transfer"], "h_bwt": g["healthy_backward_transfer"]}
        for key, (pc_f, h_f) in READERS.items():
            row[key] = (r[pc_f], r[h_f])
        rows.append(row)
    return rows


def _score(rows):
    craters = [-r["pc_bwt"] for r in rows]  # crater depth, positive
    out = {}
    for key in READERS:
        pc = [r[key][0] for r in rows]
        h = [r[key][1] for r in rows]
        matched = [p - hh for p, hh in zip(pc, h)]  # paired PC-vs-healthy margin
        null = [abs(h[i] - h[j]) for i, j in combinations(range(len(h)), 2)]  # 10 pairs
        out[key] = {
            "fires_matched_positive": sum(1 for m in matched if m > 0),
            "matched_margins": [round(m, 4) for m in matched],
            "min_matched_margin": round(min(matched), 4),
            "null_mean": round(_mean(null), 4),
            "null_max": round(max(null), 4),
            "effect_ratio_over_null": round(_mean([abs(m) for m in matched]) / _mean(null), 2),
            "clears_null_mean": sum(1 for m in matched if m > _mean(null)),
            "clears_null_max": sum(1 for m in matched if m > max(null)),
            "crater_depth_spearman": round(_spearman(matched, craters), 2),
        }
    return out


def main():
    result = {
        "note": "n=5 Spearman rho is coarse/not significant; directional only. "
        "effect_ratio is mean|PC-healthy margin| / mean(healthy-vs-healthy pairwise |diff|).",
        "eb40": _score(_load(EB40)),
        "eb60": _score(_load(EB60)),
    }
    # D2: the eb60 seed where a real two-sided crater is caught by alignment but missed by drift.
    r60 = _load(EB60)
    result["eb60_per_seed_complementarity"] = [
        {
            "seed": i,
            "two_sided_bwt_diff": round(r["pc_bwt"] - r["h_bwt"], 4),
            "cka_margin": round(r["cka_drift"][0] - r["cka_drift"][1], 4),
            "cka_separates": r["cka_drift"][0] > r["cka_drift"][1],
            "align_margin": round(r["alignment_proj"][0] - r["alignment_proj"][1], 4),
            "align_separates": r["alignment_proj"][0] > r["alignment_proj"][1],
        }
        for i, r in enumerate(r60)
    ]
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "null_analysis.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
