"""Extract the deck's numbers into assets/data.js.

Run from the repo root: `uv run python slides/p1-findings/make_p1_data.py`.

The deck itself never runs Python: its ECharts charts read `window.DeckData` from that file. Re-run this when the
numbers change, and whenever the phase index or an audit changes: the status slide parses both.

Each table cites its source. Everything is read from tracked files: the phase index and audits, the promoted run
artifacts under docs/experiments/phase1/artifacts/, and, where a substudy doc's table is the record (P1.1.1, P1.2.1,
P1.2.2, P1.2.4), numbers transcribed from that doc.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "assets" / "data.js"
EXP = ROOT / "docs/experiments"
ART = EXP / "phase1/artifacts"
P125 = ART / "P1.2.5"

# The checkpoints of a P1.2.5 drive, as steps and as frames streamed (32 per step).
STEPS = {"t00100": 100, "t00212": 212, "t00423": 423, "t00846": 846, "t01269": 1269, "end": 1692}
FRAMES = {"t00100": 3200, "t00212": 6800, "t00423": 13500, "t00846": 27100, "t01269": 40600, "end": 53900}

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
    TABLES[name] = [{k: _clean(v) for k, v in r.items()} for r in rows]
    print(f"{name}: {len(rows)} rows")


def load(rel: str) -> dict[str, Any]:
    """A tracked JSON artifact, relative to docs/experiments/phase1/artifacts/."""
    data: dict[str, Any] = json.loads((ART / rel).read_text())
    return data


# --- Phase status, parsed from the phase index and the audits -------------------------------------------------------

STUDY_LABELS = {
    "P1.0": "Build the loop",
    "P1.1": "Read degradation",
    "P1.2": "Adapt vs freeze",
    "P1.3": "Stability–plasticity",
    "P1.4": "Collapse",
    "P1.5": "Init × pressure",
    "P1.6": "FL + monitor",
}


def _cells(line: str) -> list[str]:
    """Split a markdown table row on its unescaped pipes."""
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def _status(cell: str) -> tuple[str, str, str]:
    """(category, label, note) from a Status / Outcome cell: `<marker> **<label>** (<finding>)`."""
    m = re.match(r"(✅\\?\*?|🟡|🔲)\s*\*\*(.+?)\*\*\s*(.*)", cell)
    if not m:
        raise SystemExit(f"unparsed status cell: {cell[:80]!r}")
    marker, label, note = m.groups()
    note = note.strip()
    if note.startswith("(") and note.endswith(")"):
        note = note[1:-1]
    if marker.startswith("✅"):
        cat = "complete"
    elif marker == "🟡":
        cat = "runs complete"
    elif "Designed" in label:
        cat = "designed"
    elif note.startswith("folded"):
        cat = "folded"
    else:
        cat = "not started"
    return cat, label.strip(" —-"), note


def _title(cell: str) -> str:
    """A study cell's short title: the link text, or the text before its tag or dash."""
    m = re.match(r"\[(.+?)\]\(", cell)
    t = m.group(1) if m else re.split(r" `\[| — ", cell)[0]
    return t.strip()


def _audit(study: str) -> dict[str, Any]:
    """An audit's state: none, open (findings pending, no disposition yet) or closed."""
    p = EXP / f"audits/{study}.md"
    if not p.exists():
        return {"audit": "none", "pending": 0, "findings": 0}
    s = p.read_text()
    findings = len(re.findall(r"^- \*\*[A-Z]\d+ — ", s, flags=re.M)) or len(
        re.findall(r"^\*\*[A-Z]\d+ — ", s, flags=re.M)
    )
    pending = len(re.findall(r"\*\*Conclusion\.?\*\*\s*(?:—\s*)?\*Pending", s))
    disp = re.search(r"^## [A-Z]\. Disposition\s*\n+(.*)", s, flags=re.M)
    placeholder = bool(disp and disp.group(1).startswith("*Placeholder"))
    return {"audit": "open" if placeholder or pending else "closed", "pending": pending, "findings": findings}


def status() -> None:
    """One row per sub-study (or not-yet-split study) in the Phase 1 index, with its study's audit state.

    Source: docs/experiments/phase1/index.md (Studies table) and docs/experiments/audits/P1.x.md.
    """
    rows = []
    for line in (EXP / "phase1/index.md").read_text().splitlines():
        if not line.startswith("| P1."):
            continue
        sid, study_cell, _, status_cell = _cells(line)[:4]
        study = ".".join(sid.split(".")[:2])
        cat, label, note = _status(status_cell)
        rows.append(
            {
                "id": sid,
                "study": study,
                "study_label": STUDY_LABELS.get(study, study),
                "title": _title(study_cell),
                "status": cat,
                "label": label,
                "note": note,
                **_audit(study),
            }
        )
    write("status", rows)


def audit_p11() -> None:
    """The P1.1 audit's findings: group, title, the remediation work types it asks for, and whether it is worked.

    Source: docs/experiments/audits/P1.1.md (lettered sections; `[tag]` remediation types; Conclusion lines).
    """
    s = (EXP / "audits/P1.1.md").read_text()
    groups = dict(re.findall(r"^## ([A-Z])\. (.+)$", s, flags=re.M))
    blocks = re.split(r"^- \*\*(?=[A-Z]\d+ — )", s, flags=re.M)[1:]
    rows = []
    for b in blocks:
        head = b.split("**", 1)[0]
        fid, title = head.split(" — ", 1)
        rem = re.search(r"\*\*Remediation\.\*\*(.*?)\*\*Conclusion\.\*\*", b, flags=re.S)
        tags = re.findall(r"`\[([^\]]+)\]`", rem.group(1) if rem else "")
        kinds = sorted({re.sub(r"defer.*", "defer", t) for t in tags}) or ["none"]
        rows.append(
            {
                "id": fid,
                "group": fid[0],
                "group_name": groups[fid[0]],
                "title": title.rstrip("."),
                "kinds": kinds,
                "pending": bool(re.search(r"\*\*Conclusion\.\*\*\s*\*Pending", b)),
            }
        )
    write("audit_p11", rows)


def coverage() -> None:
    """Which reading tools exist for each arm P1.3 runs, and the P1.1.4 part that rebuilds them.

    Transcribed from the P1.1 audit (C1, D1-D5) and P1.1.4.md's design (arms table, parts table). `state`: `old` (P1.1
    calibrated it, on a setup P1.3 no longer runs), `none_found` (P1.1 searched; nothing passed), `missing` (never
    built for this arm).
    """
    old_setup = "P1.1's setup: 119-step drive on BDD val, fresh heads, no protections"
    table = [
        ("DINO-init SimSiam", "Healthy bands", "missing", "No drive from a DINO start in P1.1 (D2)", "1a, 1b"),
        ("DINO-init SimSiam", "Damage of known size", "missing", "No surgical damage on a DINO start (D2)", "2"),
        ("DINO-init SimSiam", "Label-free reader", "missing", "Never screened on a DINO start (D4)", "2"),
        ("DINO-init SimSiam", "Referee power, past regimes", "missing", "No plan sets it (D3)", "4"),
        ("MAE-init JE", "Healthy bands", "old", old_setup + " (D1)", "1a, 1b"),
        ("MAE-init JE", "Damage of known size", "old", "Canary calls 6–8 pts, misses ~3; " + old_setup, "2"),
        (
            "MAE-init JE",
            "Label-free reader",
            "none_found",
            "Backbone CKA drift nearly blind to JE damage; nearly every JE scalar fires on the healthy walk (D4)",
            "2",
        ),
        (
            "MAE-init JE",
            "Referee power, past regimes",
            "old",
            "3-class canary, 3 time-of-day strata of 18 regimes (D3)",
            "4",
        ),
        ("MAE", "Healthy bands", "old", "Small 64-d × 2 decoder, since repaired (C1, D1)", "1b"),
        ("MAE", "Damage of known size", "old", "Small decoder (C1)", "3"),
        (
            "MAE",
            "Label-free reader",
            "old",
            "CKA drift, blocks 4–11: 5/5 against its twin; alone, 4/5 on the chosen blocks, 3/5 on all (B1); "
            "small decoder",
            "3",
        ),
        ("MAE", "Referee power, past regimes", "old", "3-class canary, 3 time-of-day strata of 18 regimes (D3)", "4"),
        ("DINO-init DINO (provisional)", "Healthy bands", "missing", "Also its collapse watch (D5)", "1a, 1b"),
        (
            "DINO-init DINO (provisional)",
            "Damage of known size",
            "missing",
            "The encoder read is the teacher, an average of the student: damage must hit both (D5)",
            "2",
        ),
        ("DINO-init DINO (provisional)", "Label-free reader", "missing", "Never screened (D5)", "2"),
        ("DINO-init DINO (provisional)", "Referee power, past regimes", "missing", "No plan sets it (D3)", "4"),
    ]
    write(
        "coverage",
        [{"arm": a, "piece": p, "state": s, "note": n, "part": part} for a, p, s, n, part in table],
    )


def walk() -> None:
    """The correlated walk: one full pass of BDD train in regime order; P1.2.5/walk_segments.json."""
    rows, start = [], 0
    for s in load("P1.2.5/walk_segments.json")["segments"]:
        rows.append(
            {
                "regime": s["regime"],
                "time_of_day": s["time_of_day"],
                "weather": s["weather"],
                "start": start,
                "end": start + s["n_images"],
                "n_images": s["n_images"],
            }
        )
        start += s["n_images"]
    write("walk", rows)


# --- P1.1 ---------------------------------------------------------------------------------------------------------


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
    r = load("P1.1.3/forgetting_lenses.json")
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


# --- P1.2 ---------------------------------------------------------------------------------------------------------

GATE_TASKS = (("scene", "scene"), ("traffic light", "presence_cell:traffic light"))


def gates() -> None:
    """Linear-rung gain over the init's own frozen start, scene and traffic light, 95% seed x image intervals.

    P1.2.1 / P1.2.2 / P1.2.4 from their docs' tables; P1.2.5 from P1.2.5/adaptation_report.json and partG.json.
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
    for rel, study, names in (
        (
            "P1.2.5/adaptation_report.json",
            "P1.2.5",
            {"mae_je": "JE (MAE-init), re-run", "dino_simsiam": "SimSiam (DINO-init)", "dino_dino": "DINO (DINO-init)"},
        ),
        (
            "P1.2.5/partG.json",
            "P1.2.5 G",
            {
                "dino_simsiam_p9e-5": "SimSiam (DINO-init), lr 9e-5",
                "dino_simsiam_np": "SimSiam (DINO-init), no protections",
                "mae_je_np": "JE (MAE-init), no protections",
            },
        ),
    ):
        rep = load(rel)
        for arm, label in names.items():
            for task, key in GATE_TASKS:
                c = rep["arms"][arm]["gate_contrasts"][key]["linear"]
                out.append(
                    {"study": study, "arm": label, "task": task, "delta": c["delta"], "lo": c["lo"], "hi": c["hi"]}
                )
    write("gates", out)


def protections() -> None:
    """P1.2.5 Part G: protected minus unprotected (both lr 3e-5), scene, per checkpoint and rung; partG.json recipe."""
    rep = load("P1.2.5/partG.json")
    rows = []
    for arm, label in (("dino_simsiam_np", "SimSiam (DINO-init)"), ("mae_je_np", "JE (MAE-init)")):
        for point, tasks in rep["recipe"][arm]["points"].items():
            for rung, d in tasks["scene"].items():
                if rung in ("linear", "knn", "ft100"):
                    rows.append(
                        {
                            "arm": label,
                            "rung": rung,
                            "frames": FRAMES[point],
                            "delta": d["delta"],
                            "lo": d["lo"],
                            "hi": d["hi"],
                        }
                    )
    write("protections", rows)


def curve() -> None:
    """P1.2.5 Part B: scene gain over the frozen start at each checkpoint, linear and kNN (5 seeds).

    Source: P1.2.5/adaptation_report.json.
    """
    rep = load("P1.2.5/adaptation_report.json")
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
                        "frames": FRAMES[point],
                        "delta": d["delta"],
                        "lo": d["lo"],
                        "hi": d["hi"],
                    }
                )
    write("curve", rows)


SIZE_BINS = {"0:0.005102": "under 1 patch", "0.005102:0.02": "1–4 patches", "0.02:inf": "4+ patches"}


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
        r = load(f"P1.2.5/{fname}")
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
    r = load("P1.2.5/partF_p125_gate.json")
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
                            "step": STEPS[point],
                            "delta": d["delta"],
                            "lo": d["lo"],
                            "hi": d["hi"],
                        }
                    )
    write("regime_curve", rows)


def breadth() -> None:
    """P1.2.5 Part C: linear and kNN gain on the panel (5 seeds); P1.2.5/adaptation_report.json."""
    rep = load("P1.2.5/adaptation_report.json")
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
    """P1.2.5 Part D: BDD IID gain minus each generic-image control's (linear); P1.2.5/adaptation_report.json."""
    rep = load("P1.2.5/adaptation_report.json")
    rows = []
    for arm, label in (("mae_je", "JE (MAE-init)"), ("dino_simsiam", "SimSiam (DINO-init)")):
        for ctrl, cname in (("coco", "COCO"), ("places365", "Places365")):
            for task, key in GATE_TASKS:
                c = rep["arms"][arm]["attribution"]["controls"][ctrl]["tasks"][key]
                rows.append(
                    {"arm": label, "control": cname, "task": task, "delta": c["delta"], "lo": c["lo"], "hi": c["hi"]}
                )
    write("attribution", rows)


def dino_lr() -> None:
    """P1.2.5 Part E: DINO's own objective, unswitched, seed 0, first 600 steps at three lrs.

    Source: P1.2.5/partE_health_early.json.
    """
    rows = []
    for lr, trace in load("P1.2.5/partE_health_early.json")["runs"].items():
        for r in trace:
            rows.append(
                {
                    "lr": lr,
                    "step": r["step"],
                    "projector_rank": r["rankme_proj"],
                    "backbone_rank": r["rankme"],
                    "canary_knn": r["knn_acc"],
                    "canary_linear": r["linear_acc"],
                }
            )
    write("dino_lr", rows)


def floors() -> None:
    """P1.2.5: pooled val reads (scene, linear and kNN) of the frozen starts and the adapted walk ends (5-seed mean).

    Source: P1.2.5/floors/{dino,sup}/summary.json and P1.2.5/ladder/<arm>/summary.json.
    """

    def bacc(path: str, name: str) -> dict[str, float]:
        b: dict[str, float] = load(f"P1.2.5/{path}/summary.json")["encoders"][name]["bacc"]
        return b

    rows = []
    for label, path, name in (
        ("DINO", "floors/dino", "dino"),
        ("Supervised ViT-B", "floors/sup", "sup"),
        ("MAE", "ladder/mae_je", "floor"),
    ):
        b = bacc(path, name)
        rows.append(
            {"encoder": label, "kind": "frozen", "scene_linear": b["scene|linear"], "scene_knn": b["scene|knn"]}
        )
    for arm, label in (("mae_je", "JE (MAE-init)"), ("dino_simsiam", "SimSiam (DINO-init)")):
        ends = [bacc(f"ladder/{arm}", f"walk_s{i}_end") for i in range(5)]
        rows.append(
            {
                "encoder": label,
                "kind": "adapted",
                "scene_linear": float(np.mean([e["scene|linear"] for e in ends])),
                "scene_knn": float(np.mean([e["scene|knn"] for e in ends])),
            }
        )
    write("floors", rows)


if __name__ == "__main__":
    for fn in (
        status,
        audit_p11,
        coverage,
        walk,
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
        floors,
        protections,
    ):
        fn()
    OUT.write_text(
        "/* Generated by make_p1_data.py from the phase index, the audits, the run artifacts and substudy docs; "
        "do not edit. */\n"
        f"window.DeckData = {json.dumps(TABLES, separators=(',', ':'))};\n"
    )
    print(f"wrote {OUT}")
