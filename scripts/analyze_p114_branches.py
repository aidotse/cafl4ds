"""P1.1.4 Parts 2 and 3 — the damage branches, read on the referee and by every reader (``mode=branches``).

Each arm's branches (``branch/<arm>_<kind>``, seeds 1-3) resume a healthy walk at step 846 with a
damage: ``dose0`` (the twin: no damage, the same restart), ``reinit<k>`` (the specificity control,
where it counts as one) and ``noise0p<d>`` (the doses). Read here, per arm:

* ``referee`` — the val ladder at the branch's start state (after the damage), step 1,269 and the end.
  ``damage``: each branch minus its twin, seed by seed (the restart cancels). ``restart``: each twin
  minus its own walk at the same seed and checkpoint. ``twin_sd``: the twins' spread across seeds;
* ``control`` — the specificity control confirmed on the referee: its linear and fine-tune reads must
  not fall below the twin's (a seed-mean paired difference below ``-sd_multiple`` twin SDs at any read).
  An unconfirmed re-init is neither control nor dose (``descriptive``): its fires are reported, never
  scored; a re-init on an arm with no designated control (none passed the pilot) is one more dose;
* ``adjudication`` — Part 1b's three-panel rule (bands and thresholds on IID seeds 1-10, read on the
  branch's stretch of the grid) on every branch, twin and walk-from-the-branch-point: the label, and so
  which doses are *called* (labelled ``degradation``);
* ``screen`` — every label-free reader as its own single-drive panel (P1.1.2's screen): its threshold,
  leave-one-out false alarms, the walks, the twins, the control, the doses (first fire), the restart
  effect (twin minus walk, in band SDs), the paired branch-minus-twin contrast, the natural failures,
  and the pass bar;
* ``lens_screen`` — once the block features exist (``features/<arm>/index.json``): P1.1.2's per-block
  lenses (one panel per lens over every layer) and P1.1.3's middle-block panel, calibrated on the IID
  drives' saved checkpoints and read on the same drives at theirs, with the same bar.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from cafl4ds.null_envelope import checkpoint_steps
from cafl4ds.reading_protocol import Bands, adjudicate_split, calibrate_panel, call, zscores
from scripts.analyze_p114 import (
    FinetunePanel,
    LadderSet,
    _load,
    _loo_calls,
    _seeds,
    three_panels,
)
from scripts.analyze_reading import _matrix, _settings, read_arm, skipped_steps

_KIND = re.compile(r"dose0|reinit(\d+)|noise0p(\d+)")


class _Reads(Protocol):
    def read(self, prefix: str, seed: int, checkpoint: str, key: str) -> float: ...


def kind_order(kinds: list[str]) -> list[str]:
    """Branch kinds in reading order: the twin, the re-inits, then the noise doses by size.

    Raises:
        ValueError: If a kind is not ``dose0``, ``reinit<k>`` or ``noise0p<d>``, or the twin is missing.
    """
    bad = [k for k in kinds if not _KIND.fullmatch(k)]
    if bad:
        raise ValueError(f"unknown branch kinds {bad}.")
    if "dose0" not in kinds:
        raise ValueError("no dose-0 twin among the branches.")

    def key(k: str) -> tuple[int, float]:
        m = _KIND.fullmatch(k)
        if k == "dose0" or m is None:
            return (0, 0.0)
        if m.group(1) is not None:
            return (1, float(m.group(1)))
        return (2, float("0." + m.group(2)))

    return sorted(kinds, key=key)


def branch_kinds(root: Path, branch_dir: str, arm: str) -> list[str]:
    """The arm's own branch kinds on disk (``<arm>_<kind>``; ``mae_*`` must not catch ``mae_je_*``)."""
    names = [p.name[len(arm) + 1 :] for p in (root / branch_dir).glob(f"{arm}_*") if p.is_dir()]
    return kind_order([n for n in names if _KIND.fullmatch(n)])


def paired_referee(
    branches: _Reads, walks: _Reads, kinds: list[str], seeds: list[int], keys: list[str], checkpoints: list[str]
) -> dict[str, Any]:
    """The referee's paired reads: each branch minus its twin, each twin minus its walk, the twins' spread.

    Every difference is in points (balanced accuracy × 100); ``nan`` where a read is missing.
    """

    def diff(a: float, b: float) -> float:
        return 100.0 * (a - b)

    def summary(per_seed: list[float]) -> dict[str, Any]:
        finite = [x for x in per_seed if np.isfinite(x)]
        return {"per_seed": per_seed, "mean": float(np.mean(finite)) if finite else math.nan}

    twin = {(k, c, s): branches.read("dose0", s, c, k) for k in keys for c in checkpoints for s in seeds}
    out: dict[str, Any] = {"seeds": seeds, "checkpoints": checkpoints, "restart": {}, "twin_sd": {}, "damage": {}}
    for k in keys:
        out["restart"][k] = {
            c: summary([diff(twin[k, c, s], walks.read("walk", s, c, k)) for s in seeds]) for c in checkpoints
        }
        out["twin_sd"][k] = {c: 100.0 * float(np.std([twin[k, c, s] for s in seeds], ddof=1)) for c in checkpoints}
    for kind in kinds:
        if kind == "dose0":
            continue
        out["damage"][kind] = {
            k: {c: summary([diff(branches.read(kind, s, c, k), twin[k, c, s]) for s in seeds]) for c in checkpoints}
            for k in keys
        }
    return out


def control_verdict(referee: dict[str, Any], control: str, sd_multiple: float) -> dict[str, Any]:
    """Does the control keep the information on the referee? Linear and fine-tune reads, every checkpoint.

    A read *falls below* the twin when the seed-mean paired difference is under ``-sd_multiple`` times
    the twins' spread at that read. The control is confirmed when no read falls below.
    """
    below = []
    for key, by_ck in referee["damage"][control].items():
        if key.rsplit("|", 1)[-1] not in ("linear", "ft10", "ft100"):
            continue
        for ck, entry in by_ck.items():
            bound = -sd_multiple * referee["twin_sd"][key][ck]
            if np.isfinite(entry["mean"]) and entry["mean"] < bound:
                below.append({"key": key, "checkpoint": ck, "mean": entry["mean"], "bound": bound})
    return {"control": control, "sd_multiple": sd_multiple, "below": below, "confirmed": not below}


def _on_grid(rows: list[dict[str, Any]], grid: list[int]) -> tuple[list[dict[str, Any]], list[int]]:
    """The rows whose step is on the null grid, and the steps dropped (a short drive's last reading)."""
    on = set(grid)
    dropped = sorted({int(r["step"]) for r in rows} - on)
    return [r for r in rows if int(r["step"]) in on], dropped


def _restricted(rule: Bands, grid: list[int], steps: list[int]) -> Bands:
    position = {s: i for i, s in enumerate(grid)}
    return rule.restrict([position[s] for s in steps])


def _calls(rows: list[dict[str, Any]], signal: str, rule: Bands, grid: list[int], skip: set[int]) -> dict[int, Any]:
    """Per seed: did ``signal`` fire on the drive's stretch of the grid, first when, its ``z`` trajectory.

    ``z`` is signed so that positive is the reader's harmful direction; ``min_z`` is its furthest move the
    other way, and ``z_first`` its value at the drive's first reading (both descriptive: the bar is one-sided).
    """
    steps = checkpoint_steps(rows)
    sub = _restricted(rule, grid, steps)
    m = _matrix(rows, signal, skip)
    out = {}
    for k, seed in enumerate(_seeds(rows)):
        c = call(m[k], sub)
        out[seed] = {
            "fired": c["fired"],
            "first_step": steps[c["first"]] if c["first"] is not None else None,
            "max_z": c["max_z"],
            "min_z": float(np.nanmin(z)) if np.isfinite(z := zscores(m[k], sub)).any() else math.nan,
            "z_first": float(z[0]),
            "final_z": c["final_z"],
            "fired_after_first": bool((np.nan_to_num(z[1:], nan=-np.inf) > sub.threshold).any()),
            "z": z,
            "steps": steps,
        }
    return out


def reader_bar(entry: dict[str, Any], called: dict[str, list[int]], config: DictConfig) -> dict[str, Any]:
    """Part 2's pass bar for one reader, from its calls (P1.1.2's bar, P1.1.3's single-drive conditions).

    * sensitive — fires on every called seed of every called dose at the branch's first reading (and,
      reported beside it, by step ``late_by``);
    * specific — at most ``loo_max`` leave-one-out false alarms; quiet on at least ``walks_quiet`` of the
      walks over the whole drive; quiet on every twin at every reading after its first; quiet on the
      control, if any.

    The twin's first reading is skipped: at dose 0 it is the walk's own step-846 checkpoint, saved before
    the restart, so a fire there is a walk fire (already counted by the walk bar), not a restart effect.
    """
    bc = config.branches
    first = entry["first_reading"]
    doses = {k: v for k, v in entry["kinds"].items() if k in called}
    pairs = [(k, s) for k, seeds in called.items() for s in seeds]
    at_first = [doses[k][s]["first_step"] == first for k, s in pairs]
    late = [doses[k][s]["first_step"] is not None and doses[k][s]["first_step"] <= int(bc.late_by) for k, s in pairs]
    twins = entry["kinds"]["dose0"]
    control = entry["kinds"].get(entry.get("control") or "", {})
    walks_quiet = sum(not w["fired"] for w in entry["walks"].values())
    bar = {
        "loo": entry["loo"]["fired"] <= int(bc.loo_max),
        "walks": walks_quiet >= int(bc.walks_quiet),
        "twins": not any(t["fired_after_first"] for t in twins.values()),
        "twins_at_end": all(t["final_z"] <= entry["threshold"] for t in twins.values()),
        "control": not any(c["fired"] for c in control.values()) if control else None,
        "sensitive": bool(pairs) and all(at_first),
        "sensitive_by_late": bool(pairs) and all(late),
    }
    specific = bar["loo"] and bar["walks"] and bar["twins"] and bar["control"] is not False
    return bar | {"specific": specific, "pass": specific and bar["sensitive"]}


def screen_reader(
    signal: str,
    drives: dict[str, Any],
    called: dict[str, list[int]],
    control: str | None,
    config: DictConfig,
) -> dict[str, Any]:
    """One label-free reader as its own single-drive panel, read on every drive of the arm."""
    iid, grid, skip = drives["iid"], drives["grid"], drives["skip"]
    settings = _settings([signal], list(config.arbiters), config)
    null = {signal: _matrix(iid, signal, skip)}
    rule = calibrate_panel(null, settings, float(config.alpha))[signal]
    loo = [c["fired"] for c in _loo_calls(null, settings, float(config.alpha), grid)]
    walks = _calls(drives["walk"], signal, rule, grid, skip)
    stretch = _calls(drives["walk_stretch"], signal, rule, grid, skip)
    kinds = {k: _calls(rows, signal, rule, grid, skip) for k, rows in drives["branches"].items()}
    first_reading = checkpoint_steps(drives["branches"]["dose0"])[0]
    restart = {}
    for seed, twin in kinds["dose0"].items():
        walk = stretch.get(seed)
        if walk is None:
            continue
        both = [i for i, s in enumerate(twin["steps"]) if s in set(walk["steps"])]
        wz = {s: walk["z"][i] for i, s in enumerate(walk["steps"])}
        d = np.array([twin["z"][i] - wz[twin["steps"][i]] for i in both])
        restart[seed] = {
            "mean_dz": float(np.nanmean(d)) if np.isfinite(d).any() else math.nan,
            "twin_fired": twin["fired"],
            "walk_fired": walk["fired"],
            "walk_max_z": walk["max_z"],
        }
    paired: dict[str, dict[int, Any]] = {}
    for kind, by_seed in kinds.items():
        if kind == "dose0":
            continue
        paired[kind] = {}
        for seed, b in by_seed.items():
            t = kinds["dose0"].get(seed)
            if t is None:
                continue
            d = (np.asarray(b["z"]) - np.asarray(t["z"])) / math.sqrt(2.0)
            over = np.nan_to_num(d, nan=-np.inf) > rule.threshold
            paired[kind][seed] = {
                "max_dz": float(np.nanmax(d)),
                "first_step": b["steps"][int(np.argmax(over))] if over.any() else None,
            }
    natural = {name: _calls(rows, signal, rule, grid, skip) for name, rows in drives["natural"].items()}

    def slim(calls: dict[int, Any]) -> dict[int, Any]:
        return {s: {k: v for k, v in c.items() if k not in ("z", "steps")} for s, c in calls.items()}

    entry: dict[str, Any] = {
        "threshold": float(rule.threshold),
        "loo": {"fired": int(sum(loo)), "of": len(loo)},
        "first_reading": first_reading,
        "control": control,
        "walks": slim(walks),
        "kinds": {k: slim(v) for k, v in kinds.items()},
        "restart": restart,
        "paired": paired,
        "natural": {n: slim(v) for n, v in natural.items()},
    }
    entry["bar"] = reader_bar(entry, called, config)
    return entry


def adjudicate(
    rows: list[dict[str, Any]],
    rules: dict[str, Bands],
    groups: dict[str, list[str]],
    drive: tuple[list[int], set[int], list[str]],
    ft: FinetunePanel | None,
    config: DictConfig,
    source: tuple[str, LadderSet | None],
) -> dict[int, Any]:
    """Part 1b's split label on each seed of one drive, read on its own stretch of the grid.

    ``drive`` is the null grid, the skipped steps and the arbiters; ``source`` the ladder prefix and dirs
    for the fine-tune corroboration of a kNN fire (``None`` dirs = the arm's own).
    """
    grid, skip, arbiters = drive
    prefix, ladders = source
    read = read_arm(rows, rules, grid, arbiters, skip)
    out = {}
    for seed, entry in read["per_seed"].items():
        calls = entry["calls"]
        fired = {p: any(calls[s]["fired"] for s in sig) for p, sig in groups.items()}
        first = {
            p: min((calls[s]["first_step"] for s in sig if calls[s]["fired"]), default=None)
            for p, sig in groups.items()
        }
        drop = None
        if first["canary_knn"] is not None and ft is not None:
            check = ft.check(prefix, int(seed), float(first["canary_knn"]), ladders)
            drop = check[str(config.finetune_rule)]
        out[int(seed)] = {
            "fired": fired,
            "first_step": first,
            "finetune_drop": drop,
            "label": adjudicate_split(fired, drop),
        }
    return out


def _drives(name: str, arm: DictConfig, config: DictConfig, root: Path) -> dict[str, Any]:
    """Every corpus the arm's branch read needs, each cut to the null grid."""
    bc = arm.branches
    iid = _load(root, list(arm.iid), list(arm.iid_seeds))
    grid = checkpoint_steps(iid)
    skip = skipped_steps(root / str(arm.walk[0]), config)
    kinds = branch_kinds(root, str(config.branches.dir), name)
    branches = {k: _load(root, [f"{config.branches.dir}/{name}_{k}"], None) for k in kinds}
    start = checkpoint_steps(branches["dose0"])[0]
    walk = _load(root, list(arm.walk), None)
    seeds = _seeds(branches["dose0"])
    natural, dropped = {}, {}
    for nat, dirs in (bc.get("natural") or {}).items():
        natural[nat], dropped[nat] = _on_grid(_load(root, list(dirs), None), grid)
    return {
        "iid": iid,
        "grid": grid,
        "skip": skip,
        "kinds": kinds,
        "branches": branches,
        "walk": walk,
        "walk_stretch": [r for r in walk if int(r["step"]) >= start and int(r["seed"]) in set(seeds)],
        "natural": natural,
        "natural_dropped_steps": dropped,
        "seeds": seeds,
        "start": start,
    }


def read_branches(name: str, arm: DictConfig, config: DictConfig, root: Path) -> dict[str, Any]:
    """Parts 2 and 3 for one arm: the referee's paired reads, the control, the labels and the reader screen."""
    bc, ab = config.branches, arm.branches
    drives = _drives(name, arm, config, root)
    kinds, seeds = drives["kinds"], drives["seeds"]
    control = str(ab.control) if ab.get("control") else None
    walk_ladders = LadderSet(root, list(arm.ladders))
    branch_ladders = LadderSet(root, [{"dir": str(ab.ladder), "names": "seed"}])
    keys = [f"{t}|{r}" for t in config.referee_null.tasks for r in config.referee_null.rungs]
    referee = paired_referee(branch_ladders, walk_ladders, kinds, seeds, keys, list(bc.checkpoints))
    report: dict[str, Any] = {
        "kinds": kinds,
        "seeds": seeds,
        "start": drives["start"],
        "control_kind": control,
        "referee": referee,
        "natural_dropped_steps": drives["natural_dropped_steps"],
    }
    if control is not None:
        report["control"] = control_verdict(referee, control, float(bc.control_sd_multiple))
        # A re-init counts as the control only once the referee confirms it. One that passed the pilot but
        # not the referee is neither a control nor a dose: its cost sits between "none" and the smallest
        # dose, below what Part 1b's label resolves on one drive, so a fire there can be neither held
        # against a reader nor demanded of it. It is read descriptively (`descriptive`).
        if not report["control"]["confirmed"]:
            control = None
    report["control_used"] = control
    descriptive = [str(ab.control)] if ab.get("control") and control is None else []
    report["descriptive"] = descriptive

    bb = str(arm.backbone)
    arbiters = list(config.arbiters)
    iid, grid, skip = drives["iid"], drives["grid"], drives["skip"]
    groups = three_panels(iid, list(config.signals[bb]), arbiters)
    rules: dict[str, Bands] = {}
    for panel in groups.values():
        nulls = {s: _matrix(iid, s, skip) for s in panel}
        rules.update(calibrate_panel(nulls, _settings(panel, arbiters, config), float(config.alpha)))
    all_iid = _seeds(_load(root, list(arm.iid), None))
    ft = FinetunePanel(walk_ladders, config, all_iid)
    drive = (grid, skip, arbiters)
    labels = {
        kind: adjudicate(rows, rules, groups, drive, ft, config, (kind, branch_ladders))
        for kind, rows in drives["branches"].items()
    }
    labels["walk_from_start"] = adjudicate(drives["walk_stretch"], rules, groups, drive, ft, config, ("walk", None))
    report["adjudication"] = labels
    # The doses: the noise ladder, and a re-init only where none was designated a control (it failed the pilot).
    called = {
        k: [s for s, e in labels[k].items() if e["label"] == "degradation"]
        for k in kinds
        if k.startswith("noise") or (k.startswith("reinit") and k != control and k not in descriptive)
    }
    report["called"] = called

    unreadable = set(ab.get("unreadable") or [])
    readers = [s for s in bc.readers[bb] if s not in unreadable and any(isinstance(r.get(s), float) for r in iid)]
    report["unreadable"] = sorted(unreadable)
    report["screen"] = {s: screen_reader(s, drives, called, control, config) for s in readers}
    _log(name, report)
    fdir = ab.get("features")
    if fdir is not None and (root / str(fdir.dir) / "index.json").is_file():
        report["lens_screen"] = lens_screen(name, arm, config, root, called, control)
    elif fdir is not None:
        logger.info(f"[{name}] no block features yet under {fdir.dir}: the per-block readers are not read.")
    return report


def _log(name: str, report: dict[str, Any]) -> None:
    """Print the referee's damage and restart, the control, the labels and each reader's bar."""
    ref = report["referee"]
    cks = ref["checkpoints"]
    lines = [f"[{name}] branches {report['kinds']}, seeds {report['seeds']}, from step {report['start']}"]
    lines.append(
        "  referee restart (twin - walk, pts): "
        + "; ".join(f"{k} " + " ".join(f"{ref['restart'][k][c]['mean']:+.1f}" for c in cks) for k in ref["restart"])
    )
    for kind, by_key in ref["damage"].items():
        lines.append(
            f"  referee {kind} - twin (pts): "
            + "; ".join(f"{k} " + " ".join(f"{by_key[k][c]['mean']:+.1f}" for c in cks) for k in by_key)
        )
    if "control" in report:
        c = report["control"]
        below = ", ".join(f"{b['key']}@{b['checkpoint']} {b['mean']:+.1f} < {b['bound']:+.1f}" for b in c["below"])
        lines.append(f"  control {c['control']}: {'confirmed' if c['confirmed'] else 'NOT confirmed'} {below}")
    for kind, by_seed in report["adjudication"].items():
        lines.append(f"  label {kind}: " + ", ".join(f"s{s} {e['label']}" for s, e in by_seed.items()))
    lines.append(f"  called: {report['called']}")
    for s, e in report["screen"].items():
        b = e["bar"]
        flags = " ".join(f"{k}={'-' if v is None else int(v)}" for k, v in b.items())
        dz = np.nanmean([r["mean_dz"] for r in e["restart"].values()]) if e["restart"] else math.nan
        lines.append(f"  reader {s:<22} thr {e['threshold']:.2f} restart dz {dz:+.2f}  {flags}")
    logger.info("\n".join(lines))


# --- the per-block readers (P1.1.2's lenses, P1.1.3's middle-block panel), on the saved checkpoints ---------


def _stem_step(stem: str, end: int) -> int:
    """A checkpoint file stem's drive step (``step00846`` → 846, ``final`` → ``end``)."""
    return end if stem == "final" else int(stem.removeprefix("step"))


def lens_matrix(
    table: dict[str, dict[str, float]], run: str, seeds: list[int], stems: list[str], signals: list[str]
) -> dict[str, np.ndarray]:
    """``{signal: [seeds, stems]}`` from a lens table (``nan`` where a checkpoint was not read)."""
    out = {s: np.full((len(seeds), len(stems)), np.nan) for s in signals}
    for i, seed in enumerate(seeds):
        for j, stem in enumerate(stems):
            row = table.get(f"features/{run}/s{seed}/{stem}.pt")
            if row is None:
                continue
            for s in signals:
                out[s][i, j] = row.get(s, np.nan)
    return out


def panel_calls(
    mats: dict[str, np.ndarray], rules: dict[str, Bands], index: list[int], steps: list[int], seeds: list[int]
) -> dict[int, Any]:
    """One panel's call per seed on a stretch of the grid: any layer firing, first when, and its ``z``."""
    sub = {s: rules[s].restrict(index) for s in mats}
    out: dict[int, Any] = {}
    for i, seed in enumerate(seeds):
        zs = {s: zscores(mats[s][i], sub[s]) for s in mats}
        if not any(np.isfinite(z).any() for z in zs.values()):
            continue
        calls = {s: call(mats[s][i], sub[s]) for s in mats}
        firsts = [c["first"] for c in calls.values() if c["first"] is not None]
        stack = np.array(list(zs.values()))
        out[seed] = {
            "fired": bool(firsts),
            "first_step": steps[min(firsts)] if firsts else None,
            "fired_layers": sorted(s.split("@", 1)[1] for s, c in calls.items() if c["fired"]),
            "max_z": float(np.nanmax(stack)),
            "min_z": float(np.nanmin(stack)),
            "z_first": float(np.nanmax(stack[:, 0])),
            "final_z": float(np.nanmax(stack[:, -1])),
            "fired_after_first": bool(
                any((np.nan_to_num(zs[s][1:], nan=-np.inf) > sub[s].threshold).any() for s in zs)
            ),
            "z": zs,
        }
    return out


def _slim(calls: dict[int, Any]) -> dict[int, Any]:
    return {s: {k: v for k, v in c.items() if k != "z"} for s, c in calls.items()}


def _lens_inputs(name: str, arm: DictConfig, config: DictConfig, root: Path) -> dict[str, Any]:
    """Every lens matrix the per-block screen reads: the IID null, the walks, the branches, the natural failures."""
    from scripts.analyze_lenses import lens_table  # noqa: PLC0415  (spawns workers; only this mode needs it)

    lc, fc = config.branches.lens, arm.branches.features
    out_dir = Path(to_absolute_path(str(config.out))).parent

    def _table(fdir: str) -> dict[str, dict[str, float]]:
        """A features dir's lens values: its own cache read, new values cached beside the report."""
        return lens_table(root / fdir, int(lc.workers), out_dir / f"lens_values_{Path(fdir).name}.json")

    table = _table(str(fc.dir))
    stems, b_stems = list(lc.stems), list(lc.branch_stems)
    grid = [_stem_step(s, int(lc.end_step)) for s in stems]
    signals = sorted({s for row in table.values() for s in row})
    b_index = [stems.index(s) for s in b_stems]
    seeds = list(config.branches.get("seeds", [1, 2, 3]))
    natural = {}
    for nat, spec in (fc.get("natural") or {}).items():
        ntable = _table(str(spec.dir)) if spec.get("dir") else table
        natural[nat] = (lens_matrix(ntable, str(spec.run), list(spec.seeds), stems, signals), list(spec.seeds))
    return {
        "stems": stems,
        "grid": grid,
        "signals": signals,
        "iid": lens_matrix(table, str(fc.iid), list(fc.iid_seeds), stems, signals),
        "iid_seeds": list(fc.iid_seeds),
        "walk": lens_matrix(table, str(fc.walk), list(fc.walk_seeds), stems, signals),
        "walk_seeds": list(fc.walk_seeds),
        "b_index": b_index,
        "b_steps": [grid[i] for i in b_index],
        "seeds": seeds,
        "branch": {
            k: lens_matrix(table, f"{name}_{k}", seeds, b_stems, signals)
            for k in branch_kinds(root, str(config.branches.dir), name)
        },
        "natural": natural,
    }


def _paired_panel(
    by_kind: dict[str, dict[int, Any]], panel: list[str], threshold: float, steps: list[int]
) -> dict[str, Any]:
    """Each branch against its twin per layer, (z_branch − z_twin)/√2: the panel's max and its first fire."""
    paired: dict[str, dict[int, Any]] = {}
    for kind, calls in by_kind.items():
        if kind == "dose0":
            continue
        paired[kind] = {}
        for seed, c in calls.items():
            t = by_kind["dose0"].get(seed)
            if t is None:
                continue
            d = np.array([(c["z"][s] - t["z"][s]) / math.sqrt(2.0) for s in panel])
            cols = (np.nan_to_num(d, nan=-np.inf) > threshold).any(axis=0)
            paired[kind][seed] = {
                "max_dz": float(np.nanmax(d)),
                "first_step": steps[int(np.argmax(cols))] if cols.any() else None,
            }
    return paired


def _lens_panel(
    panel: list[str], x: dict[str, Any], config: DictConfig, called: dict[str, list[int]], control: str | None
) -> dict[str, Any]:
    """One per-block panel: calibrated on the IID null, read on every drive, with Part 2's bar."""
    lc = config.branches.lens
    settings = dict.fromkeys(panel, {"span": int(lc.span), "floor": float(lc.floor)})
    nulls = {s: x["iid"][s] for s in panel}
    rules = calibrate_panel(nulls, settings, float(lc.alpha))
    loo = [c["fired"] for c in _loo_calls(nulls, settings, float(lc.alpha), x["grid"])]
    full = list(range(len(x["stems"])))
    b_index, b_steps = x["b_index"], x["b_steps"]
    walks = panel_calls({s: x["walk"][s] for s in panel}, rules, full, x["grid"], x["walk_seeds"])
    stretch = panel_calls({s: x["walk"][s][:, b_index] for s in panel}, rules, b_index, b_steps, x["walk_seeds"])
    by_kind = {
        k: panel_calls({s: m[s] for s in panel}, rules, b_index, b_steps, x["seeds"]) for k, m in x["branch"].items()
    }
    restart = {
        seed: {
            "mean_dz": float(np.nanmean([twin["z"][s] - stretch[seed]["z"][s] for s in panel])),
            "twin_fired": twin["fired"],
            "walk_fired": stretch[seed]["fired"],
        }
        for seed, twin in by_kind["dose0"].items()
        if seed in stretch
    }
    entry: dict[str, Any] = {
        "layers": [s.split("@", 1)[1] for s in panel],
        "threshold": float(rules[panel[0]].threshold),
        "loo": {"fired": int(sum(loo)), "of": len(loo)},
        "first_reading": b_steps[0],
        "control": control,
        "walks": _slim(walks),
        "kinds": {k: _slim(v) for k, v in by_kind.items()},
        "restart": restart,
        "paired": _paired_panel(by_kind, panel, float(rules[panel[0]].threshold), b_steps),
        "natural": {
            n: _slim(panel_calls({s: m[s] for s in panel}, rules, full, x["grid"], seeds))
            for n, (m, seeds) in x["natural"].items()
        },
    }
    entry["bar"] = reader_bar(entry, called, config)
    return entry


def lens_screen(
    name: str, arm: DictConfig, config: DictConfig, root: Path, called: dict[str, list[int]], control: str | None
) -> dict[str, Any]:
    """The per-block readers: one panel per lens over every layer, and P1.1.3's middle-block panel.

    Calibrated on the IID drives' saved checkpoints (span and floor as P1.1.2's per-block panels); the
    walks read at every checkpoint, each branch at its start state (after the damage), step 1,269 and its end.
    """
    lc = config.branches.lens
    x = _lens_inputs(name, arm, config, root)
    by_lens = {lens: [s for s in x["signals"] if s.split("@", 1)[0] == lens] for lens in lc.lenses}
    fixed = {p: [f"cka_drift@{layer}" for layer in layers] for p, layers in lc.panels.items()}
    out: dict[str, Any] = {
        "grid": dict(zip(x["stems"], x["grid"], strict=True)),
        "iid_seeds": x["iid_seeds"],
        "panels": {},
    }
    for pname, signals in (by_lens | fixed).items():
        panel = [s for s in signals if s in x["iid"] and np.isfinite(x["iid"][s]).all()]
        if panel:
            out["panels"][pname] = _lens_panel(panel, x, config, called, control)
    lines = [f"[{name}] per-block readers: IID seeds {x['iid_seeds']}, grid {x['stems']}"]
    for pname, e in out["panels"].items():
        flags = " ".join(f"{k}={'-' if v is None else int(v)}" for k, v in e["bar"].items())
        lines.append(f"  panel {pname:<12} thr {e['threshold']:.2f} loo {e['loo']['fired']}/{e['loo']['of']}  {flags}")
    logger.info("\n".join(lines))
    return out
