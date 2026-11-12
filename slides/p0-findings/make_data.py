"""Extract the Phase 0 deck's numbers into assets/data.js.

Run from the repo root: `uv run python slides/p0-findings/make_data.py`.

The deck itself never runs Python: its ECharts charts read `window.DeckData` from that file. Re-run this only when the
numbers change.

Each table cites its source. Most read the tracked run artifacts (`docs/experiments/phase0/artifacts/`). Where only a
substudy doc records the numbers, they are transcribed from it. The per-step and per-checkpoint traces were never
tracked, so three tables read local Hydra run dirs under `outputs/` (gitignored); each says which run it reads and is
checked against the tracked artifact where one exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "assets" / "data.js"
ART = ROOT / "docs/experiments/phase0/artifacts"
RUNS = ROOT / "outputs"

TABLES: dict[str, list[dict[str, Any]]] = {}


def _clean(v: object) -> object:
    """Plain JSON values, floats rounded to 5 significant digits."""
    if isinstance(v, float):
        return float(f"{v:.5g}")
    return v


def write(name: str, rows: list[dict[str, Any]]) -> None:
    """Register one table for data.js, under `name`."""
    TABLES[name] = [{k: _clean(v) for k, v in r.items()} for r in rows]
    print(f"{name}: {len(rows)} rows")


def load(path: str | Path) -> dict[str, Any]:
    """Read one JSON object, relative to the artifacts dir unless absolute."""
    p = Path(path)
    data: dict[str, Any] = json.loads((p if p.is_absolute() else ART / p).read_text())
    return data


def check(ok: bool, what: str) -> None:
    """Stop if a local run no longer matches the tracked artifact it stands in for."""
    if not ok:
        raise ValueError(what)


# --- The scoreboard -------------------------------------------------------------------------------------------------


def scoreboard() -> None:
    """Failure mode x reader, from the Failure Modes table and Handover of docs/experiments/phase0/index.md.

    status: works / catch (works, with a load-bearing catch) / labels (works, but needs labels) / fails (blind,
    confounded or mis-signals) / guard (used only as a precondition). Cells not listed were not a reader for that mode.
    """
    cells = [
        ("Collapse (JE)", "Rank", "catch", "Fires in every corner; thins to ~2× on the 2× bar where Phase 1 reads it"),
        ("Collapse (JE)", "Projector geometry", "works", "Variance: point vs redundancy; covariance: redundancy"),
        ("Collapse (JE)", "Backbone geometry", "fails", "Genuine blind spots and false alarms"),
        ("Collapse (JE)", "Gradient norm", "guard", "Stays quiet on collapse: it is specific to divergence"),
        ("Forgetting (MAE)", "Labelled probe", "labels", "BWT / FM, seed-stable 5/5, but needs labels"),
        (
            "Forgetting (MAE)",
            "Loss / recon gap",
            "fails",
            "Mostly the replay arm improving; 10× false fire on grayscale",
        ),
        ("Forgetting (MAE)", "Drift, old batch", "fails", "Quiet at the output (Phase 1 finds it per block, P1.1.3)"),
        ("Forgetting (JE)", "Labelled probe", "labels", "BWT ground truth"),
        ("Forgetting (JE)", "Drift, old batch", "works", "Tracks the crater two-sided, 5/5"),
        ("Forgetting (JE)", "Drift, no old data", "catch", "Only the projector's frame churn, 5/5; backbone 3/5"),
        ("Forgetting (JE)", "Projector geometry", "catch", "Uniformity mis-signals; alignment reads the regime only"),
        ("Forgetting (JE)", "Rank", "guard", "Full rank throughout: forgetting, not collapse"),
        ("Instability (MAE)", "Gradient norm", "catch", "~5 steps ahead of the NaN, only with SGD + a raw loss"),
        ("Instability (MAE)", "Loss / recon gap", "fails", "The loss shows a runaway only as it goes NaN"),
        ("Quality loss (MAE)", "Rank", "fails", "Blind by construction: the bad rep is full-rank"),
        (
            "Quality loss (MAE)",
            "Backbone geometry",
            "fails",
            "Uniformity / clusterability say the bad rep is healthier",
        ),
        ("Quality loss (MAE)", "Loss / recon gap", "fails", "The loss lies: lower on the degraded arm, every seed"),
        ("Quality loss (MAE)", "Labelled probe", "labels", "The only reader: a labelled canary"),
    ]
    write("scoreboard", [{"mode": m, "reader": r, "status": s, "note": n} for m, r, s, n in cells])


# --- P0.2 collapse --------------------------------------------------------------------------------------------------

ENVELOPE = [
    ("rankme", "proj", "Rank"),
    ("mean_feature_var", "proj", "Variance"),
    ("offdiag_cov", "proj", "Covariance"),
    ("uniformity", "proj", "Uniformity"),
    ("rankme", "backbone", "Rank"),
    ("mean_feature_var", "backbone", "Variance"),
    ("offdiag_cov", "backbone", "Covariance"),
    ("uniformity", "backbone", "Uniformity"),
]


def collapse_heatmap() -> None:
    """P0.2.2 E1 corners (point collapse) and P0.2.3 (redundancy collapse): min-across-4-seeds separation.

    Ratios fire at 2.0, uniformity (a gap) at 0.5; `bar` carries each one's threshold. Source: envelope[].separation.
    """
    corners = [
        ("P0.2.2/E1/iid_40ep_seed{}.json", "IID, 40 epochs"),
        ("P0.2.2/E1/corr_40ep_seed{}.json", "Correlated, 40 epochs"),
        ("P0.2.2/E1/iid_1pass_seed{}.json", "IID, one pass"),
        ("P0.2.2/E1/corr_1pass_seed{}.json", "Correlated, one pass"),
        ("P0.2.3/seed_{}.json", "Redundancy collapse"),
    ]
    rows = []
    for pattern, corner in corners:
        envs = [load(pattern.format(s))["envelope"] for s in range(4)]
        for metric, surface, label in ENVELOPE:
            seps = [next(e for e in env if e["metric"] == metric and e["surface"] == surface) for env in envs]
            rows.append(
                {
                    "corner": corner,
                    "reader": label,
                    "surface": "projector" if surface == "proj" else "backbone",
                    "sep": min(s["separation"] for s in seps),
                    "bar": seps[0]["threshold"],
                }
            )
    write("collapse_heatmap", rows)


def collapse_trajectory() -> None:
    """P0.2.2 D1 rider: IID x 40 epochs, 16 seeds, healthy vs forced-collapse RankMe per checkpoint.

    Local run dirs outputs/D1_rider/iid_seed{0-15} (their summary reproduces the tracked E3_16seed/summary.json).
    """
    runs = [load(RUNS / f"D1_rider/iid_seed{s}/comparison.json") for s in range(16)]
    rows = []
    for arm, label in (("healthy", "Healthy"), ("pc", "Forced collapse")):
        for i, point in enumerate(runs[0][arm]):
            for key, surface in (("rankme_proj", "projector"), ("rankme", "backbone")):
                vals = sorted(r[arm][i][key] for r in runs)
                rows.append(
                    {
                        "arm": label,
                        "surface": surface,
                        "epoch": point["step"] / 32,
                        "median": median(vals),
                        "lo": vals[0],
                        "hi": vals[-1],
                    }
                )
    write("collapse_trajectory", rows)


def collapse_dose() -> None:
    """P0.2.5: projector RankMe and linear probe along the partial-collapse dial, both knobs.

    Coarse ladder P0.2.5/{blend,softsg}/seed_{0-3}.json; refined low-severity ladder P0.2.5/lowres/*_seed{0,1}.json.
    """
    rows = []
    for ladder, pattern, seeds in (
        ("coarse", "P0.2.5/{k}/seed_{s}.json", 4),
        ("fine", "P0.2.5/lowres/{k}_seed{s}.json", 2),
    ):
        for knob, name in (("blend", "Loss blend"), ("softsg", "Soft stop-gradient")):
            for s in range(seeds):
                for rung in load(pattern.format(k=knob, s=s))["ladder"]:
                    f = rung["final"]
                    rows.append(
                        {
                            "ladder": ladder,
                            "knob": name,
                            "seed": s,
                            "severity": rung["severity"],
                            "rank_proj": f["rankme_proj"],
                            "rank_backbone": f["rankme"],
                            "linear": f["linear_acc"],
                        }
                    )
    write("collapse_dose", rows)


def collapse_margin() -> None:
    """P0.2.2: per-seed projector-RankMe separation in the four E1 corners and the E4 single-pass LR sweep."""
    rows = []
    for corner, label in (
        ("iid_40ep", "IID, 40 ep"),
        ("corr_40ep", "Corr., 40 ep"),
        ("iid_1pass", "IID, 1 pass"),
        ("corr_1pass", "Corr., 1 pass"),
    ):
        for s in range(4):
            env = load(f"P0.2.2/E1/{corner}_seed{s}.json")["envelope"]
            sep = next(e for e in env if e["metric"] == "rankme" and e["surface"] == "proj")["separation"]
            rows.append({"group": "Training regime", "label": label, "seed": s, "sep": sep})
    for mult in ("0.25x", "0.5x", "1x", "2x", "4x"):
        for s in range(4):
            env = load(f"P0.2.2/E4/lr{mult}_seed{s}.json")["envelope"]
            sep = next(e for e in env if e["metric"] == "rankme" and e["surface"] == "proj")["separation"]
            rows.append({"group": "LR, IID 1 pass", "label": f"{mult.replace('x', '×')} LR", "seed": s, "sep": sep})
    write("collapse_margin", rows)


# --- P0.3 forgetting on MAE -----------------------------------------------------------------------------------------


def _bwt(path: str) -> tuple[float, float, dict[str, Any]]:
    g = load(path)
    return g["gate"]["pc_backward_transfer"], g["gate"]["healthy_backward_transfer"], g


def forgetting_hunt() -> None:
    """Every MAE forgetting vehicle, per seed: PC and healthy-twin BWT (fires at PC BWT <= -0.05).

    `fires` follows each substudy's own rule: PC <= -0.05 with the twin holding (|BWT| <= 0.03) up to P0.3.9;
    two-sided (PC - twin <= -0.05) in P0.3.10. STL-10 is transcribed from P0.3.2.md (200-image probes) and
    CIFAR-100 from P0.3.4.md (min / median / max only); the rest read the artifacts.
    """
    rows: list[dict[str, Any]] = []

    def add(config: str, study: str, pc: float, h: float, fires: bool, kind: str = "seed") -> None:
        rows.append({"config": config, "study": study, "pc": pc, "healthy": h, "fires": fires, "kind": kind})

    stl_pc = [0.024, -0.001, 0.026, 0.026, 0.026]
    stl_h = [0.078, 0.010, 0.063, 0.015, 0.044]
    for p, h in zip(stl_pc, stl_h, strict=True):
        add("STL-10, from scratch", "P0.3.2", p, h, False)
    for p, h in zip([-0.003, 0.011, 0.080], [0.029, 0.055, 0.084], strict=True):
        add("CIFAR-100, from scratch", "P0.3.4", p, h, False, kind="min/median/max")

    def absolute(pc: float, h: float) -> bool:
        return pc <= -0.05 and abs(h) <= 0.03

    for s in range(5):
        p, h, _ = _bwt(f"P0.3.4/n5_lr1e-4_seed{s}.json")
        add("Supervised ViT-B (upper limit)", "P0.3.4", p, h, absolute(p, h))
    for config, pattern in (
        ("Pretrained MAE, far split", "P0.3.9/c1_seedsweep/comparison_lr3e-4_seed{}.json"),
        ("+ deeper well", "P0.3.9/d1redo_decfirst/comparison_decfirst_seed{}.json"),
        ("+ deeper well, 2× task-B time", "P0.3.9/d1redo_epochsb40/comparison_epochsb40_seed{}.json"),
    ):
        for s in range(5):
            p, h, _ = _bwt(pattern.format(s))
            add(config, "P0.3.9", p, h, absolute(p, h))
    for config, domain in (("Far domain: satellite", "eurosat"), ("Far domain: textures", "dtd")):
        for s in range(5):
            p, h, _ = _bwt(f"P0.3.10/{domain}_nseed/comparison_{domain}_seed{s}.json")
            add(config, "P0.3.10", p, h, p - h <= -0.05)
    write("forgetting_hunt", rows)


def forgetting_depth() -> None:
    """Task-A accuracy before (R00) and after (R10) task B, per seed; the from-scratch MAE floor is ~0.38-0.40.

    Source: pc.matrix in the P0.3.9 / P0.3.10 artifacts; the floor is savings.summary.scratch in the eb40 savings run.
    """
    rows = []
    for config, pattern in (
        ("Pretrained MAE, far split", "P0.3.9/c1_seedsweep/comparison_lr3e-4_seed{}.json"),
        ("+ deeper well, 2× task-B time", "P0.3.9/d1redo_epochsb40/comparison_epochsb40_seed{}.json"),
        ("Far domain: satellite", "P0.3.10/eurosat_nseed/comparison_eurosat_seed{}.json"),
        ("Far domain: textures", "P0.3.10/dtd_nseed/comparison_dtd_seed{}.json"),
    ):
        for s in range(5):
            m = load(pattern.format(s))["pc"]["matrix"]
            rows.append({"config": config, "seed": s, "before": m["0"]["0"], "after": m["1"]["0"]})
    sav = load("P0.3.9/d1redo_epochsb40/comparison_epochsb40_savings_seed4.json")["savings"]["summary"]["scratch"]
    rows.append({"config": "floor", "seed": None, "before": sav["final_acc"], "after": sav["max_acc"]})
    write("forgetting_depth", rows)


def recon_gap() -> None:
    """The label-free MAE reader: recon-loss gap (PC - twin) vs two-sided BWT (PC - twin), per run.

    Source: gate.reported.recon_forget_gap and gate.{pc,healthy}_backward_transfer.
    """
    rows = []
    for config, pattern, seeds in (
        ("Pretrained MAE, far split", "P0.3.9/c1_seedsweep/comparison_lr3e-4_seed{}.json", 5),
        ("+ deeper well", "P0.3.9/d1redo_decfirst/comparison_decfirst_seed{}.json", 5),
        ("+ deeper well, 2× task-B time", "P0.3.9/d1redo_epochsb40/comparison_epochsb40_seed{}.json", 5),
        ("Far domain: satellite", "P0.3.10/eurosat_nseed/comparison_eurosat_seed{}.json", 5),
        ("Far domain: textures", "P0.3.10/dtd_nseed/comparison_dtd_seed{}.json", 5),
        ("Scrambled task B", "P0.3.8/comparison_scrambled_lr3e-4_seed0.json", 1),
        ("Grayscale task B", "P0.3.8/comparison_grayscale_lr3e-4_seed0.json", 1),
    ):
        for s in range(seeds):
            p, h, g = _bwt(pattern.format(s))
            rows.append({"config": config, "seed": s, "bwt": p - h, "gap": g["gate"]["reported"]["recon_forget_gap"]})
    write("recon_gap", rows)


# --- P0.4 instability -----------------------------------------------------------------------------------------------


def divergence_traces() -> None:
    """P0.4.0 per-step grad norm, SGD vs AdamW, seed 0 (step 0-40; a NaN ends a trace).

    The tracked artifacts dropped the traces, so this reads a re-run of the documented commands
    (outputs/p0-deck/p040_{sgd,adamw}); its summaries match P0.4.0/{dose_seed0,adamw_seed0} exactly, checked below.
    """
    rows = []
    for optim, tracked, lr_healthy, lrs in (
        ("SGD", "dose_seed0", 1.0, (3.0, 10.0)),
        ("AdamW", "adamw_seed0", 1e-3, (10.0, 100.0)),
    ):
        run = load(RUNS / f"p0-deck/p040_{optim.lower()}/comparison.json")
        ref = load(f"P0.4.0/{tracked}/comparison.json")
        for mine, theirs in zip(run["sweep"], ref["sweep"], strict=True):
            same_nan = mine["summary"]["first_nonfinite_step"] == theirs["first_nonfinite_step"]
            same_peak = abs(mine["summary"]["peak_grad_norm"] / theirs["peak_grad_norm"] - 1) < 1e-6
            check(same_nan and same_peak, f"{optim} lr {theirs['lr']} does not match P0.4.0/{tracked}")
        arms = [("healthy", run["healthy"])] + [(f"lr {s['summary']['lr']:g}", s) for s in run["sweep"]]
        for name, arm in arms:
            if name != "healthy" and float(name[3:]) not in lrs:
                continue
            label = f"healthy (lr {lr_healthy:g})" if name == "healthy" else name
            for t in arm["trace"][:41]:
                rows.append(
                    {
                        "optim": optim,
                        "arm": label,
                        "step": t["step"],
                        "grad_norm": t["grad_norm"] if t["finite"] else None,
                        "nan": not t["finite"],
                    }
                )
        rows.append(
            {"optim": optim, "arm": "alarm", "step": None, "grad_norm": run["gate"]["blowup_level"], "nan": False}
        )
    write("divergence_traces", rows)


def shock_traces() -> None:
    """P0.4.1 per-step grad norm around a spliced burst (steps 300-307), seed 0, lr 1.

    The tracked artifacts dropped op_traces, so this reads the local run dirs of the documented runs; the noise and
    scrambled runs are checked against P0.4.1/{gate_seed0,dataset_scrambled} gate.fire_ratio.
    """
    runs = [
        ("16-19-40", "control", "No burst (control)", None),
        ("16-19-40", "shock", "Saturating noise burst", "gate_seed0"),
        ("16-14-10", "shock", "CIFAR-100 burst", None),
        ("16-14-33", "shock", "Grayscale CIFAR burst", None),
        ("16-14-56", "shock", "Phase-scrambled burst", "dataset_scrambled"),
    ]
    rows = []
    for d, arm, label, tracked in runs:
        r = load(RUNS / f"2026-08-06/{d}/comparison.json")
        if tracked:
            ref = load(f"P0.4.1/{tracked}/comparison.json")["gate"]["fire_ratio"]
            check(abs(r["gate"]["fire_ratio"] / ref - 1) < 1e-3, f"{d} does not match P0.4.1/{tracked}")
        for t in r["op_traces"][arm]:
            if t["step"] >= 240:
                rows.append({"arm": label, "step": t["step"], "grad_norm": t["grad_norm"], "loss": t["loss"]})
    write("shock_traces", rows)


def shock_lead() -> None:
    """P0.4.1 burst-size dose: per-seed lead (steps from burst onset to the loss NaN) and runaway count.

    Source: P0.4.1/scale_dose_seeded/dose.json rows (3 seeds, HPU).
    """
    rows = []
    for r in load("P0.4.1/scale_dose_seeded/dose.json")["rows"]:
        for s, lead in r["lead_by_seed"].items():
            rows.append(
                {
                    "scale": r["burst_scale"],
                    "seed": int(s),
                    "lead": lead,
                    "fire": r["fire_by_seed"][s],
                    "runaways": r["n_runaway"],
                }
            )
    write("shock_lead", rows)


# --- P0.5 quality on MAE --------------------------------------------------------------------------------------------


def loss_lies() -> None:
    """P0.5.2 N=5 (over-powered decoder vs healthy, both mask 0.75): final loss vs kNN and linear probe, per seed."""
    rows = []
    for s in range(5):
        g = load(f"P0.5.2/nseed_seed{s}.json")["gate"]
        for arm, label in (("healthy", "Healthy MAE"), ("shortcut", "Over-powered decoder")):
            rows.append(
                {
                    "arm": label,
                    "seed": s,
                    "loss": g[f"{arm}_loss_floor"],
                    "knn": g[f"{arm}_probe_final"],
                    "linear": g[f"{arm}_linear_final"],
                }
            )
    write("loss_lies", rows)


def quality_readers() -> None:
    """P0.5.2 N=5: each candidate reader's degrade gap (positive = reads the bad arm as worse) over its fire bar.

    Source: gate.reader_verdict.candidates[].{reader, degrade_gap, threshold}; RankMe from envelope separation.
    """
    names = {
        "alignment_strong": "Alignment, strong augment",
        "alignment": "Alignment",
        "clusterability": "Clusterability",
        "uniformity": "Uniformity",
        "attn_distance": "Attention distance",
    }
    rows = []
    for s in range(5):
        r = load(f"P0.5.2/nseed_seed{s}.json")
        for c in r["gate"]["reader_verdict"]["candidates"]:
            if c["reader"] in names:
                rows.append(
                    {
                        "reader": names[c["reader"]],
                        "seed": s,
                        "gap": c["degrade_gap"],
                        "bar": c["threshold"],
                        "score": c["degrade_gap"] / c["threshold"],
                    }
                )
        rank = next(e for e in r["envelope"] if e["metric"] == "rankme")
        rows.append(
            {"reader": "Rank", "seed": s, "gap": rank["separation"], "bar": 2.0, "score": (rank["separation"] - 1)}
        )
    write("quality_readers", rows)


def decoder_dose() -> None:
    """P0.5.3 decoder dose-down (seed 0): probe crater (healthy - shortcut) and the alignment_strong gap per decoder.

    Sources: P0.5.3/dec*_seed0.json, P0.5.2/nseed_seed0.json (512x8) and P0.5.2/lever_mask_e96.json (the mask lever);
    64x2 is the healthy decoder itself, so its crater is zero by construction.
    """

    def read(path: str) -> dict[str, Any]:
        g = load(path)["gate"]
        a = next(c for c in g["reader_verdict"]["candidates"] if c["reader"] == "alignment_strong")
        return {
            "knn_crater": g["healthy_probe_final"] - g["shortcut_probe_final"],
            "linear_crater": g["healthy_linear_final"] - g["shortcut_linear_final"],
            "align_gap": a["degrade_gap"],
            "align_bar": a["threshold"],
        }

    rows = [
        {"lever": "decoder", "decoder": "64×2 (healthy)", "knn_crater": 0.0, "linear_crater": 0.0, "align_gap": None}
    ]
    for dec, path in (
        ("128×4", "P0.5.3/dec128x4_seed0.json"),
        ("256×6", "P0.5.3/dec256x6_seed0.json"),
        ("384×8", "P0.5.3/dec384x8_seed0.json"),
        ("512×8", "P0.5.2/nseed_seed0.json"),
    ):
        rows.append({"lever": "decoder", "decoder": dec, **read(path)})
    rows.append({"lever": "mask", "decoder": "low mask ratio", **read("P0.5.2/lever_mask_e96.json")})
    write("decoder_dose", rows)


# --- P0.6 forgetting on JE ------------------------------------------------------------------------------------------


def je_vs_mae() -> None:
    """P0.6.0: the same far split on JE (A-only twin) and on MAE, per seed: two-sided BWT and backbone CKA drift.

    Sources: P0.6.0/aonly_nseed and P0.6.0/mae_contrast, gate.{pc,healthy}_backward_transfer and gate.reported.
    """
    rows = []
    for vehicle, pattern in (
        ("JE (SimSiam)", "P0.6.0/aonly_nseed/comparison_aonly_seed{}.json"),
        ("MAE", "P0.6.0/mae_contrast/comparison_mae_seed{}.json"),
    ):
        for s in range(5):
            p, h, g = _bwt(pattern.format(s))
            rep = g["gate"]["reported"]
            rows.append(
                {
                    "vehicle": vehicle,
                    "seed": s,
                    "bwt": p - h,
                    "drift_pc": rep["pc_cka_drift"],
                    "drift_healthy": rep["healthy_cka_drift"],
                }
            )
    write("je_vs_mae", rows)


def je_readers() -> None:
    """P0.6 per-seed margin (forgetting arm - A-only twin) for each label-free reader.

    A1 (task-A canary): P0.6.0/reader_null_analysis/null_analysis.json eb40.<reader>.matched_margins.
    C3 (current stream, no old data): P0.6.1/comparison_aonly_seed*.json gate.reported.cs_<reader>_{pc,healthy}.
    """
    rows = []
    null = load("P0.6.0/reader_null_analysis/null_analysis.json")["eb40"]
    for key, label in (
        ("cka_drift", "Backbone drift (CKA)"),
        ("cosine_drift", "Backbone drift (cosine)"),
        ("alignment_proj", "Projector alignment"),
        ("uniformity_proj", "Projector uniformity"),
    ):
        for s, m in enumerate(null[key]["matched_margins"]):
            rows.append({"reference": "Saved task-A batch (A1)", "reader": label, "seed": s, "margin": m})
    for key, label in (
        ("cosine_drift_proj", "Projector drift (cosine)"),
        ("cka_drift_proj", "Projector drift (CKA)"),
        ("cosine_drift", "Backbone drift (cosine)"),
        ("cka_drift", "Backbone drift (CKA)"),
    ):
        for s in range(5):
            rep = load(f"P0.6.1/comparison_aonly_seed{s}.json")["gate"]["reported"]
            rows.append(
                {
                    "reference": "Current stream only (C3)",
                    "reader": label,
                    "seed": s,
                    "margin": rep[f"cs_{key}_pc"] - rep[f"cs_{key}_healthy"],
                }
            )
    write("je_readers", rows)


if __name__ == "__main__":
    for fn in (
        scoreboard,
        collapse_heatmap,
        collapse_trajectory,
        collapse_dose,
        collapse_margin,
        forgetting_hunt,
        forgetting_depth,
        recon_gap,
        divergence_traces,
        shock_traces,
        shock_lead,
        loss_lies,
        quality_readers,
        decoder_dose,
        je_vs_mae,
        je_readers,
    ):
        fn()
    OUT.write_text(
        "/* Generated by make_data.py from the run artifacts and substudy docs; do not edit. */\n"
        f"window.DeckData = {json.dumps(TABLES, separators=(',', ':'))};\n"
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
