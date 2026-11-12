"""Extract the deck's numbers into assets/data.js.

Run from the repo root: `uv run python slides/p1-findings/make_data.py`.

The deck itself never runs Python: its ECharts charts read `window.DeckData` from that file. Re-run this only when the
numbers change.

Each table cites its source. Where a substudy doc's table is the record (P1.1.1, P1.2.1, P1.2.2, P1.2.4), the numbers
are transcribed from it; everything else is read from the run artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "assets" / "data.js"
P125 = ROOT / "outputs/p125"


TABLES: dict[str, list[dict[str, Any]]] = {}


def _clean(v: object) -> object:
    """Plain JSON values, floats rounded to 4 places (the deck shows at most 0.1 pt)."""
    if isinstance(v, np.floating | float):
        return round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v


def write(name: str, rows: list[dict[str, Any]]) -> None:
    """Register one table for data.js, under `name`."""
    key = name.removesuffix(".csv")
    TABLES[key] = [{k: _clean(v) for k, v in r.items()} for r in rows]
    print(f"{key}: {len(rows)} rows")


def canary_power() -> None:
    """P1.1.1 surgical damage: linear dent (mean over the branch) against its dose-0 twin; P1.1.1.md tables."""
    table = {
        ("JE", "IID"): [(0.3, -0.028, 0), (0.5, -0.075, 3), (0.75, -0.141, 3), (1.0, -0.158, 3)],
        ("JE", "walk"): [(0.3, -0.030, 1), (0.5, -0.084, 3), (0.75, -0.134, 3), (1.0, -0.175, 3)],
        ("MAE", "IID"): [(0.3, 0.004, 0), (0.5, -0.027, 0), (0.75, -0.068, 3), (1.0, -0.109, 3)],
        ("MAE", "walk"): [(0.3, -0.006, 0), (0.5, -0.037, 1), (0.75, -0.086, 3), (1.0, -0.104, 3)],
    }
    rows = [
        {"backbone": b, "order": o, "sigma": s, "dent_pts": 100 * d, "fires": f}
        for (b, o), pts in table.items()
        for s, d, f in pts
    ]
    write("canary_power", rows)


def compression_vs_collapse() -> None:
    """P1.1.1: the 54-step walk branch vs forced collapse (alpha 1, lr 1e-4), seeds 0-2 mean; P1.1.1.md table."""
    rows: list[dict[str, Any]] = [
        {"arm": "Correlated walk (lr 3e-5)", "surface": "Projector rank", "start": 10.9, "end": 8.4},
        {"arm": "Correlated walk (lr 3e-5)", "surface": "Backbone rank", "start": 19.6, "end": 19.1},
        {"arm": "Forced collapse (lr 1e-4)", "surface": "Projector rank", "start": 10.6, "end": 6.9},
        {"arm": "Forced collapse (lr 1e-4)", "surface": "Backbone rank", "start": 18.8, "end": 11.6},
    ]
    for r in rows:
        r["change_pct"] = 100 * (r["end"] - r["start"]) / r["start"]
    write("compression_vs_collapse", rows)


def cka_blocks() -> None:
    """P1.1.3 item 5: per-block CKA drift, forgetting (PC) vs healthy arm, 5 seeds; forgetting_lenses.json."""
    r = json.loads((ROOT / "docs/experiments/phase1/artifacts/P1.1.3/forgetting_lenses.json").read_text())
    rows = []
    for layer in [f"block{i:02d}" for i in range(1, 13)] + ["backbone"]:
        e = r["per_reader"][f"cka_drift@{layer}"]
        rows.append(
            {
                "layer": "output" if layer == "backbone" else str(int(layer[5:])),
                "forgetting": e["pc_mean"],
                "healthy": e["healthy_mean"],
                "seeds_separating": e["n_positive"],
            }
        )
    write("cka_blocks", rows)


def gates() -> None:
    """Linear-rung gain over the init's own frozen start, scene and traffic light, 95% seed x image intervals.

    P1.2.1 / P1.2.2 / P1.2.4 from their docs' tables; P1.2.5 from outputs/p125/adaptation_report.json.
    """
    rows = [
        ("P1.2.1", "JE (MAE-init), first recipe", "scene", 0.025, 0.008, 0.041),
        ("P1.2.1", "JE (MAE-init), first recipe", "traffic light", 0.040, 0.022, 0.061),
        ("P1.2.1", "MAE, first recipe", "scene", -0.043, -0.060, -0.026),
        ("P1.2.1", "MAE, first recipe", "traffic light", -0.011, -0.030, 0.007),
        ("P1.2.2", "MAE, repaired recipe", "scene", -0.014, -0.029, 0.002),
        ("P1.2.2", "MAE, repaired recipe", "traffic light", -0.005, -0.023, 0.012),
        ("P1.2.4", "JE (MAE-init), protections", "scene", 0.038, 0.023, 0.053),
        ("P1.2.4", "JE (MAE-init), protections", "traffic light", 0.031, 0.011, 0.051),
    ]
    out = [{"study": s, "arm": a, "task": t, "delta": d, "lo": lo, "hi": hi} for s, a, t, d, lo, hi in rows]
    rep = json.loads((P125 / "adaptation_report.json").read_text())
    names = {"mae_je": "JE (MAE-init), re-run", "dino_simsiam": "SimSiam (DINO-init)", "dino_dino": "DINO (DINO-init)"}
    for arm, label in names.items():
        for task, key in (("scene", "scene"), ("traffic light", "presence_cell:traffic light")):
            c = rep["arms"][arm]["gate_contrasts"][key]["linear"]
            out.append(
                {"study": "P1.2.5", "arm": label, "task": task, "delta": c["delta"], "lo": c["lo"], "hi": c["hi"]}
            )
    part_g = json.loads((P125 / "partG.json").read_text())
    names = {
        "dino_simsiam_p9e-5": "SimSiam (DINO-init), lr 9e-5",
        "dino_simsiam_np": "SimSiam (DINO-init), no protections",
        "mae_je_np": "JE (MAE-init), no protections",
    }
    for arm, label in names.items():
        for task, key in (("scene", "scene"), ("traffic light", "presence_cell:traffic light")):
            c = part_g["arms"][arm]["gate_contrasts"][key]["linear"]
            out.append(
                {"study": "P1.2.5 G", "arm": label, "task": task, "delta": c["delta"], "lo": c["lo"], "hi": c["hi"]}
            )
    write("gates", out)


def protections() -> None:
    """P1.2.5 Part G: protected minus unprotected (both lr 3e-5), scene, per checkpoint and rung.

    Source: partG.json `recipe`.
    """
    rep = json.loads((P125 / "partG.json").read_text())
    frames = {"t00100": 3200, "t00212": 6800, "t00423": 13500, "t00846": 27100, "t01269": 40600, "end": 53900}
    rows = []
    for arm, label in (("dino_simsiam_np", "SimSiam (DINO-init)"), ("mae_je_np", "JE (MAE-init)")):
        for point, tasks in rep["recipe"][arm]["points"].items():
            for rung, d in tasks["scene"].items():
                if rung in ("linear", "knn", "ft100"):
                    rows.append(
                        {
                            "arm": label,
                            "rung": rung,
                            "frames": frames[point],
                            "delta": d["delta"],
                            "lo": d["lo"],
                            "hi": d["hi"],
                        }
                    )
    write("protections", rows)


def curve() -> None:
    """P1.2.5 Part B: scene gain over the frozen start at each checkpoint, linear and kNN (5 seeds).

    Source: adaptation_report.json.
    """
    rep = json.loads((P125 / "adaptation_report.json").read_text())
    frames = {"t00100": 3200, "t00212": 6800, "t00423": 13500, "t00846": 27100, "t01269": 40600, "end": 53900}
    rows = []
    for arm, label in (
        ("mae_je", "JE (MAE-init)"),
        ("dino_simsiam", "SimSiam (DINO-init)"),
        ("dino_dino", "DINO (DINO-init)"),
    ):
        for point, c in rep["arms"][arm]["curve"]["scene"]["points"].items():
            for rung in ("linear", "knn"):
                d = c[rung]
                rows.append(
                    {
                        "arm": label,
                        "rung": rung,
                        "frames": frames[point],
                        "delta": d["delta"],
                        "lo": d["lo"],
                        "hi": d["hi"],
                    }
                )
    write("curve", rows)


SIZE_BINS = {"0:0.005102": "under 1 patch", "0.005102:0.02": "1–4 patches", "0.02:inf": "4+ patches"}
PARTF = ROOT / "docs/experiments/phase1/artifacts/P1.2.5"


def sizes() -> None:
    """P1.2.5 Part F1: end-of-drive gain by object size, averaged over the read categories; partF_*_sizes.json."""
    drives = (
        ("partF_p125_sizes.json", "dino_simsiam", "SimSiam (DINO-init)"),
        ("partF_p125_sizes.json", "mae_je", "JE (MAE-init)"),
        ("partF_p121_sizes.json", "je", "JE (MAE-init), P1.2.1 recipe"),
        ("partF_p121_sizes.json", "mae", "MAE"),
    )
    rows = []
    for fname, arm, label in drives:
        r = json.loads((PARTF / fname).read_text())
        for task, t in r["arms"][arm].items():
            if not task.startswith("size_cell:"):
                continue
            category, lo, hi = task.split(":")[1:]
            for rung in ("linear", "knn"):
                end = t[rung]["curve"]["all"]["end"]
                rows.append(
                    {
                        "drive": label,
                        "category": category,
                        "size": SIZE_BINS[f"{lo}:{hi}"],
                        "rung": rung,
                        "delta": end["delta"],
                        "lo": end["lo"],
                        "hi": end["hi"],
                    }
                )
    write("sizes", rows)


def regime_curve() -> None:
    """P1.2.5 Part F2: scene gain by time-of-day slice at each checkpoint; partF_p125_gate.json."""
    r = json.loads((PARTF / "partF_p125_gate.json").read_text())
    steps = {"t00100": 100, "t00212": 212, "t00423": 423, "t00846": 846, "t01269": 1269, "end": 1692}
    rows = []
    for arm, label in (("dino_simsiam", "SimSiam (DINO-init)"), ("mae_je", "JE (MAE-init)")):
        for rung in ("linear", "knn"):
            for sl, name in (("tod:daytime", "daytime"), ("tod:night", "night")):
                for point, d in r["arms"][arm]["scene"][rung]["curve"][sl].items():
                    rows.append(
                        {
                            "arm": label,
                            "rung": rung,
                            "slice": name,
                            "step": steps[point],
                            "delta": d["delta"],
                            "lo": d["lo"],
                            "hi": d["hi"],
                        }
                    )
    write("regime_curve", rows)


def breadth() -> None:
    """P1.2.5 Part C: linear and kNN gain on the panel (5 seeds); adaptation_report.json."""
    rep = json.loads((P125 / "adaptation_report.json").read_text())
    labels = {
        "weather": "weather",
        "large_cell:car,bus,truck:0.02": "large vehicle",
        "presence_cell:bus": "bus",
        "presence_cell:pedestrian": "pedestrian",
    }
    rows = []
    for arm, label in (("mae_je", "JE (MAE-init)"), ("dino_simsiam", "SimSiam (DINO-init)")):
        tasks = rep["arms"][arm]["breadth"]["tasks"]
        for key, name in labels.items():
            for rung in ("linear", "knn"):
                c = tasks[key][rung]
                rows.append(
                    {"arm": label, "task": name, "rung": rung, "delta": c["delta"], "lo": c["lo"], "hi": c["hi"]}
                )
    write("breadth", rows)


def attribution() -> None:
    """P1.2.5 Part D: BDD IID gain minus each generic-image control's (linear); adaptation_report.json."""
    rep = json.loads((P125 / "adaptation_report.json").read_text())
    rows = []
    for arm, label in (("mae_je", "JE (MAE-init)"), ("dino_simsiam", "SimSiam (DINO-init)")):
        for ctrl, cname in (("coco", "COCO"), ("places365", "Places365")):
            for task, key in (("scene", "scene"), ("traffic light", "presence_cell:traffic light")):
                c = rep["arms"][arm]["attribution"]["controls"][ctrl]["tasks"][key]
                rows.append(
                    {"arm": label, "control": cname, "task": task, "delta": c["delta"], "lo": c["lo"], "hi": c["hi"]}
                )
    write("attribution", rows)


def dino_lr() -> None:
    """P1.2.5 Part E: DINO's own objective, unswitched, seed 0, at three lrs; each drive's corpus/health.csv."""
    runs = {
        "9e-5": P125 / "drive/dino_dino_lr9e-5",
        "3e-4": P125 / "partE/hi/dino_lr3e-4",
        "1e-3": P125 / "partE/hi/dino_lr1e-3",
    }
    rows = []
    for lr, d in runs.items():
        with (d / "corpus/health.csv").open() as f:
            for r in csv.DictReader(f):
                if not r.get("rankme"):
                    continue
                step = int(float(r["step"]))
                if step > 600:
                    continue
                rows.append(
                    {
                        "lr": lr,
                        "step": step,
                        "projector_rank": float(r["rankme_proj"]),
                        "backbone_rank": float(r["rankme"]),
                        "canary_knn": float(r["knn_acc"]),
                        "canary_linear": float(r["linear_acc"]),
                    }
                )
    write("dino_lr", rows)


def dino_switches() -> None:
    """P1.2.5 Part E: end-of-drive reads per arm (unswitched / centring off / sharpening off); health.csv."""
    runs = {
        ("9e-5", "unswitched"): P125 / "drive/dino_dino_lr9e-5",
        ("9e-5", "centring off"): P125 / "partE/nocenter_full",
        ("9e-5", "sharpening off"): P125 / "partE/nosharpen_full",
    }
    for lr in ("3e-4", "1e-3"):
        for arm, name in (("dino", "unswitched"), ("nocenter", "centring off"), ("nosharpen", "sharpening off")):
            runs[(lr, name)] = P125 / f"partE/hi/{arm}_lr{lr}"
    rows = []
    for (lr, arm), d in runs.items():
        with (d / "corpus/health.csv").open() as f:
            last = [r for r in csv.DictReader(f) if r.get("rankme")][-1]
        rows.append(
            {
                "lr": lr,
                "arm": arm,
                "steps": int(float(last["step"])) + 1,
                "projector_rank": float(last["rankme_proj"]),
                "backbone_rank": float(last["rankme"]),
                "canary_knn": float(last["knn_acc"]),
                "canary_linear": float(last["linear_acc"]),
            }
        )
    write("dino_switches", rows)


def floors() -> None:
    """P1.2.5: absolute val reads (scene, linear and kNN) of the frozen starts and the adapted walk ends."""

    def enc(path: str, name: str) -> dict[str, Any]:
        bacc: dict[str, Any] = json.loads((P125 / path / "summary.json").read_text())["encoders"][name]["bacc"]
        return bacc

    rows = []

    def add(label: str, kind: str, b: dict[str, Any]) -> None:
        rows.append(
            {
                "encoder": label,
                "kind": kind,
                "scene_linear": b["scene|linear"]["pooled"],
                "scene_knn": b["scene|knn"]["pooled"],
            }
        )

    add("DINO", "frozen", enc("floors/dino", "dino"))
    add("Supervised ViT-B", "frozen", enc("floors/sup", "sup"))
    add("MAE", "frozen", enc("ladder/mae_je", "floor"))
    for arm, label in (("mae_je", "JE (MAE-init)"), ("dino_simsiam", "SimSiam (DINO-init)")):
        s = json.loads((P125 / f"ladder/{arm}/summary.json").read_text())["encoders"]
        ends = [s[f"walk_s{i}_end"]["bacc"] for i in range(5)]
        rows.append(
            {
                "encoder": label,
                "kind": "adapted",
                "scene_linear": float(np.mean([e["scene|linear"]["pooled"] for e in ends])),
                "scene_knn": float(np.mean([e["scene|knn"]["pooled"] for e in ends])),
            }
        )
    write("floors", rows)


if __name__ == "__main__":
    for fn in (
        canary_power,
        compression_vs_collapse,
        cka_blocks,
        gates,
        curve,
        sizes,
        regime_curve,
        breadth,
        attribution,
        dino_lr,
        dino_switches,
        floors,
        protections,
    ):
        fn()
    OUT.write_text(
        "/* Generated by make_data.py from the run artifacts and substudy docs; do not edit. */\n"
        f"window.DeckData = {json.dumps(TABLES, separators=(',', ':'))};\n"
    )
    print(f"wrote {OUT}")
