# P0.6 reader certification against a healthy-vs-healthy null (audit C1 + C3)

Post-hoc `[analysis]`, no new runs. Re-scores the label-free projector readers on the promoted A-only gates
(`aonly_nseed/` eb40, `deepen_eb60/` eb60) against a **healthy-vs-healthy seed-noise null** — the pairwise separations
among the five A-only healthy arms, the P0.5.2 floor P0.6 skipped. `analyze.py` regenerates `null_analysis.json`. All
Spearman ρ are n=5 (coarse, not significant) — directional only.

## The alignment-vs-geometry question (C1 / A1 / D1)

| reader | fires (matched>0) | effect ratio vs null (eb40 / eb60) | crater-depth ρ (eb40 / eb60) | verdict |
| -- | -- | -- | -- | -- |
| `uniformity_proj` | 3/5 eb40, 4/5 eb60 (sign-flips) | 2.10 / 2.18 | +0.5 / −0.1 | **refuted** — mis-signals (flips) |
| `alignment_proj` | **5/5 both** | 1.29 / 3.76 | −0.1 / −0.5 | **candidate regime-level 2nd reader** |
| `cka_drift` | 5/5 eb40, 4/5 eb60 | 2.15 / 1.84 | −0.1 / **+0.9** | calibrated; magnitude-linked at the deep well |
| `cosine_drift` | 5/5 both | 5.34 / 1.75 | −0.3 / −0.1 | calibrated (past-data-free at proj) |

- `alignment_proj` **separates two-sided 5/5 same-direction at both eb40 and eb60** and **clears the healthy null** —
    thinly at eb40 (1.29×, only 2/5 seeds clear the null *max*; min margin +0.0135) and clearly at the deeper eb60
    (3.76×, 5/5 clear the null *mean*). So "refuted" is wrong for it.
- But it **does not track crater depth** (ρ −0.1 eb40, −0.5 eb60) — it reads the far-shift *regime*, not forgetting
    *magnitude*, the same limitation the P0.5 audit found for `alignment_strong`. → a **candidate second,
    independent-mechanism reader** (invariance geometry, not representation displacement), certified against seed noise,
    but not a magnitude-calibrated quality instrument.
- `uniformity_proj` genuinely mis-signals — 2 sign-flips (seeds 2, 3 at eb40) — the "geometry does not port" negative is
    earned by it alone.

## Complementarity (D2)

At **eb60 seed 3** the crater is real two-sided (BWT diff −0.075) but `cka` drift **misses** (margin −0.003,
`cka_separates=False`) while `alignment_proj` **fires** (+0.146). So the readers are complementary — neither dominates
on the sub-bar-but-two-sided seeds — which argues for keeping the panel plural, not crowning one reader.

## Drift magnitude vs crater depth (C3)

At the deeper eb60 well `cka_drift` (needs the retained-canary) **tracks crater depth** (ρ +0.9) — real
forgetting-specificity beyond the co-firing boolean. But the deployable **past-data-free** `cosine_drift` does **not**
(ρ −0.1), and at the shallow eb40 well neither tracks (noise-dominated). So the magnitude-vs-depth link exists only for
the canary-dependent content-drift reader at the deep well; the past-data-free reader stays boolean-only. The no-control
deployment disambiguation remains Phase-1.
