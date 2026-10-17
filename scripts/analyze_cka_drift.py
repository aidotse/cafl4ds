"""P1.1.3 items 1-4 — can CKA drift tell information loss from benign adaptation?

P1.1.2 left CKA drift against the well as MAE's one label-free damage reader, but it fires on healthy
walks and on optimizer restarts. Four reads of the existing corpora, no new runs; the report (``out``)
holds one section per item:

* ``jolt`` (item 3) — each dose-0 twin (the drive resumed at step 65 with a fresh AdamW, nothing else)
  against the original drive, step by step, in units of the IID null's seed spread
  (:func:`cafl4ds.cka_reads.jolt`): how big the restart's kick to drift is, and how long it lasts;
* ``past`` (item 1) — drift on the strata the walk has left behind (:func:`cafl4ds.cka_reads.stratum_values`),
  a panel calibrated on the IID drives' night-block checkpoints. Damage is read immediately (the
  rebuilt post-damage state against step 65, no training between); healthy walks are read at the
  night-block checkpoints; the pooled backbone read faces the same grid and bar;
* ``walk_null`` (item 2, preliminary) — pooled ``cka_drift`` calibrated on the 5 healthy *walk*
  drives: their leave-one-out false alarms, then each walk branch's restart-free damage record read
  against bands that leave its parent seed out;
* ``lr_walk`` (item 4) — JE's lr-1e-4 walk minus its IID control, against the same gap at lr 3e-5
  (:func:`cafl4ds.cka_reads.gap_of_gaps`) — all four branch at step 65 with a fresh optimizer, so the
  restart cancels.

Example::

    uv run python scripts/analyze_cka_drift.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from cafl4ds.cka_reads import gap_of_gaps, jolt, stratum_signal, stratum_values
from cafl4ds.jsonio import dumps_valid
from cafl4ds.lenses import fires, twin_effect
from cafl4ds.null_envelope import checkpoint_steps, read_corpus, seed_matrix
from cafl4ds.reading_protocol import (
    Bands,
    calibrate_panel,
    call,
    leave_one_out_fired,
    short_batch_steps,
    zscores,
)

logger.remove()
logger.add(sys.stdout, level="INFO")


def _path(template: str, **fmt: str) -> Path:
    return Path(to_absolute_path(template.format(**fmt)))


def _skip(config: DictConfig, bb: str) -> set[int]:
    """The short-batch steps the scalar reads blank (P1.1.1)."""
    segments = json.loads(
        (_path(str(config.null_root)) / "warm" / f"{bb}_o0.0_walk" / "corpus" / "segments.json").read_text()
    )
    return short_batch_steps(segments["segments"], int(config.batch_size), int(config.min_batch))


def _by_step(rows: list[dict[str, Any]], seed: int, signal: str) -> dict[int, float]:
    return {
        int(r["step"]): float(r[signal])
        for r in rows
        if int(r["seed"]) == seed and r.get("arm") == "live" and isinstance(r.get(signal), float)
    }


def _seeds(rows: list[dict[str, Any]]) -> list[int]:
    return sorted({int(r["seed"]) for r in rows})


def _matrix(rows: list[dict[str, Any]], signal: str, grid: list[int], skip: set[int]) -> np.ndarray:
    """``[seeds, steps]`` of ``signal`` on ``grid``, short-batch steps blanked."""
    m = seed_matrix(rows, signal)
    m[:, [i for i, s in enumerate(grid) if s in skip]] = np.nan
    return m


def scalar_bands(config: DictConfig, bb: str, signal: str, skip: set[int]) -> tuple[Bands, list[int]]:
    """``signal`` calibrated as a panel of its own on P1.1.1's 10 IID seeds at every step, and its step grid."""
    rows = [r for d in config.scalars.calibration for r in read_corpus(_path(str(d), bb=bb) / "corpus" / "health.csv")]
    grid = checkpoint_steps(rows)
    settings = {"span": int(config.scalars.span), "floor": float(config.floor)}
    bands = calibrate_panel({signal: _matrix(rows, signal, grid, skip)}, {signal: settings}, float(config.alpha))
    return bands[signal], grid


# --- item 3: the optimizer-restart jolt ---------------------------------------------------------


def restart_jolt(config: DictConfig, bb: str) -> dict[str, Any]:
    """Each dose-0 twin against the original drive of the same seed and order, per step, harm-signed."""
    skip = _skip(config, bb)
    bands, grid = scalar_bands(config, bb, "cka_drift", skip)
    out: dict[str, Any] = {"threshold": bands.threshold, "orders": {}}
    for order in ("iid", "walk"):
        orig = read_corpus(_path(str(config.null_root)) / "warm" / f"{bb}_o0.0_{order}" / "corpus" / "health.csv")
        twin = read_corpus(
            _path(str(config.root)) / "branch" / f"{bb}_{order}_{config.dose0}" / "corpus" / "health.csv"
        )
        per_seed: dict[str, Any] = {}
        for seed in _seeds(twin):
            t, o = _by_step(twin, seed, "cka_drift"), _by_step(orig, seed, "cka_drift")
            steps = [s for s in sorted(set(t) & set(o)) if s not in skip]
            gap = np.array([twin_effect(t[s], o[s], bands, grid.index(s)) for s in steps])
            idx = [grid.index(s) for s in steps]
            fired = {
                name: [
                    s
                    for s, z in zip(steps, zscores(np.array([row[s] for s in steps]), bands.restrict(idx)), strict=True)
                    if z > bands.threshold
                ]
                for name, row in (("twin", t), ("original", o))
            }
            tc, oc = _by_step(twin, seed, "linear_acc"), _by_step(orig, seed, "linear_acc")
            per_seed[str(seed)] = {
                **jolt(gap, steps, bands.threshold, int(config.jolt.settle)),
                "gap_first_steps": {str(s): round(float(g), 2) for s, g in list(zip(steps, gap, strict=True))[:8]},
                "fired_steps": fired,
                "canary_gap_mean": float(np.mean([tc[s] - oc[s] for s in steps])),
            }
        out["orders"][order] = per_seed
    return out


# --- item 1: past-regime drift ------------------------------------------------------------------


def _stratum_table(fdir: Path, names: dict[int, str], groups: torch.Tensor, cache: Path) -> dict[str, dict[str, float]]:
    """Per-stratum drift at the backbone for every indexed checkpoint, cached."""
    table: dict[str, dict[str, float]] = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
    index = json.loads((fdir / "index.json").read_text(encoding="utf-8"))
    todo = [e["features"] for e in index if e["features"] not in table]
    if todo:
        logger.info(f"{fdir.name}: per-stratum drift on {len(todo)} checkpoints")
        well = torch.load(fdir / "features" / "well.pt")
        for key in todo:
            table[key] = stratum_values(torch.load(fdir / key), well, groups, names)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(table), encoding="utf-8")
    return table


def _category(p111: dict[str, Any], key: str) -> str:
    """``called`` if P1.1.1's pooled canary fired on every seed of the arm, ``quiet`` if on none, else ``partial``."""
    shares = p111["branches"][key]["fired_share"]
    share = max(shares["knn_acc"], shares["linear_acc"])
    return "called" if share == 1.0 else "quiet" if share == 0.0 else "partial"


def _harm(effects: list[float]) -> float:
    finite = [e for e in effects if np.isfinite(e)]
    return max(finite) if finite else float("nan")


def past_regime(config: DictConfig) -> dict[str, Any]:
    """Past-strata drift vs pooled drift: calibrate on IID night-block checkpoints, read damage and walks."""
    bb = str(config.past.backbone)
    fdir = _path(str(config.features), bb=bb)
    probe = torch.load(fdir / "probe.pt")
    groups = torch.cat([probe["support_groups"], probe["query_groups"]])
    names = {int(k): str(v) for k, v in probe["group_names"].items()}
    table = _stratum_table(fdir, names, groups, _path(str(config.out)).parent / f"stratum_values_{bb}.json")
    pooled = json.loads((fdir / "lens_values.json").read_text(encoding="utf-8"))
    for key, values in table.items():
        values.update({s: pooled[key][s] for s in ("cka_drift@backbone", "dcka_drift@backbone")})
    stems = list(config.past.grid)
    strata = [str(s) for s in config.past.strata]
    panels = {
        "past_dcka": [stratum_signal("dcka_drift", "backbone", s) for s in strata],
        "past_cka": [stratum_signal("cka_drift", "backbone", s) for s in strata],
        "current_dcka": [stratum_signal("dcka_drift", "backbone", "night")],
        "pooled_cka": ["cka_drift@backbone"],
        "pooled_dcka": ["dcka_drift@backbone"],
    }

    def null(run: str, sig: str) -> np.ndarray:
        return np.array([[table[f"features/{run}/s{s}/{stem}.pt"][sig] for stem in stems] for s in range(5)])

    settings = {"span": int(config.past.span), "floor": float(config.floor)}
    p111 = json.loads(_path(str(config.reading_report)).read_text(encoding="utf-8"))[bb]
    root = _path(str(config.root))
    canary = str(config.past.canary)
    out: dict[str, Any] = {"grid": stems, "panels": {}}
    for name, panel in panels.items():
        iid = {sig: null(f"{bb}_o0.0_iid", sig) for sig in panel}
        walk = {sig: null(f"{bb}_o0.0_walk", sig) for sig in panel}
        rules = calibrate_panel(iid, dict.fromkeys(panel, settings), float(config.alpha))
        loo = leave_one_out_fired(iid, dict.fromkeys(panel, settings), float(config.alpha))
        walk_calls = [
            max((call(walk[sig][i], rules[sig]) for sig in panel), key=lambda c: c["max_z"]) for i in range(5)
        ]
        arms: dict[str, Any] = {}
        for arm_dir in sorted((root / "branch").glob(f"{bb}_*")):
            key = arm_dir.name[len(f"{bb}_") :].replace("_", "/", 1)
            if key.endswith(f"/{config.dose0}"):
                continue
            order = key.split("/")[0]
            twin = f"{bb}_{order}_{config.dose0}"
            seeds: dict[str, Any] = {}
            for report in sorted((arm_dir / "reports").glob("*.json")):
                data = json.loads(report.read_text(encoding="utf-8"))
                seed, rec = int(data["config"]["seed"]), data["config"]["branch"]
                post = table[f"features/{arm_dir.name}/s{seed}/post_damage.pt"]
                pre = table[f"features/{bb}_o0.0_{order}/s{seed}/step00065.pt"]
                final, twin_final = (table[f"features/{r}/s{seed}/final.pt"] for r in (arm_dir.name, twin))
                imm = [twin_effect(post[s], pre[s], rules[s], 0) for s in panel]
                end = [twin_effect(final[s], twin_final[s], rules[s], len(stems) - 1) for s in panel]
                seeds[str(seed)] = {
                    "immediate": _harm(imm),
                    "end": _harm(end),
                    "fired_immediate": any(fires(e, rules[s]) for e, s in zip(imm, panel, strict=True)),
                    "canary_immediate": {
                        k: rec["post"][k] - rec["pre"][k]
                        for k in (canary, *(f"{canary}_{s.replace('/', '_')}" for s in [*strata, "night"]))
                    },
                }
            arms[key] = {"category": _category(p111, key), "seeds": seeds}
        out["panels"][name] = {
            "threshold": rules[panel[0]].threshold,
            "fpr_loo": float(np.mean(loo)),
            "walk": [{"fired": c["fired"], "max_z": c["max_z"]} for c in walk_calls],
            "arms": arms,
            "summary": _summarize(arms, walk_calls),
        }
    return out


def _summarize(arms: dict[str, Any], walk_calls: list[Any]) -> dict[str, Any]:
    """The pass bar's counts: fires on called seeds, quiet on re-init 4, quiet walks, and the margin."""
    called = [s for a in arms.values() if a["category"] == "called" for s in a["seeds"].values()]
    reinit4 = [s for k, a in arms.items() if k.endswith("reinit4") for s in a["seeds"].values()]
    worst_walk = max(c["max_z"] for c in walk_calls)
    harms = np.array([s["immediate"] for s in called])
    return {
        "fires_on_called": f"{sum(s['fired_immediate'] for s in called)}/{len(called)}",
        "quiet_on_reinit4": f"{sum(not s['fired_immediate'] for s in reinit4)}/{len(reinit4)}",
        "reinit4_max": max(s["immediate"] for s in reinit4),
        "quiet_walks": f"{sum(not c['fired'] for c in walk_calls)}/{len(walk_calls)}",
        "called_min": float(harms.min()),
        "called_median": float(np.median(harms)),
        "worst_walk_z": worst_walk,
        "margin_median": float(np.median(harms) / worst_walk),
        "margin_min": float(harms.min() / worst_walk),
    }


# --- item 2: the walk-aware null (preliminary) --------------------------------------------------


def walk_null(config: DictConfig) -> dict[str, Any]:
    """Pooled ``cka_drift`` on bands from the healthy walks; each walk branch read out of its parent seed."""
    bb = str(config.walk_null.backbone)
    skip = _skip(config, bb)
    rows = read_corpus(_path(str(config.null_root)) / "warm" / f"{bb}_o0.0_walk" / "corpus" / "health.csv")
    grid = checkpoint_steps(rows)
    null = _matrix(rows, "cka_drift", grid, skip)
    settings = {"cka_drift": {"span": int(config.scalars.span), "floor": float(config.floor)}}
    alpha = float(config.alpha)
    full = calibrate_panel({"cka_drift": null}, settings, alpha)["cka_drift"]
    loo = leave_one_out_fired({"cka_drift": null}, settings, alpha)
    iid_bands, iid_grid = scalar_bands(config, bb, "cka_drift", skip)
    p111 = json.loads(_path(str(config.reading_report)).read_text(encoding="utf-8"))[bb]
    root = _path(str(config.root))
    arms: dict[str, Any] = {}
    for arm_dir in sorted((root / "branch").glob(f"{bb}_walk_*")):
        key = f"walk/{arm_dir.name[len(f'{bb}_walk_') :]}"
        trajectory = read_corpus(arm_dir / "corpus" / "health.csv")
        seeds: dict[str, Any] = {}
        for report in sorted((arm_dir / "reports").glob("*.json")):
            data = json.loads(report.read_text(encoding="utf-8"))
            seed, rec = int(data["config"]["seed"]), data["config"]["branch"]
            bands = calibrate_panel({"cka_drift": np.delete(null, seed, axis=0)}, settings, alpha)["cka_drift"]
            # The restart-free reading: the damage applied to the step-65 state, before any update.
            pre_step = max(s for s in grid if s < int(rec["start_step"]) and s not in skip)
            effect = twin_effect(rec["post"]["cka_drift"], rec["pre"]["cka_drift"], bands, grid.index(pre_step))
            iid_effect = twin_effect(
                rec["post"]["cka_drift"], rec["pre"]["cka_drift"], iid_bands, iid_grid.index(pre_step)
            )
            # The rest of the branch, described: its own trajectory against the out-of-seed walk bands.
            values = _by_step(trajectory, seed, "cka_drift")
            steps = [s for s in sorted(values) if s not in skip]
            rest = call(np.array([values[s] for s in steps]), bands.restrict([grid.index(s) for s in steps]))
            seeds[str(seed)] = {
                "immediate": effect,
                "fired": fires(effect, bands),
                "threshold": bands.threshold,
                "immediate_vs_iid_null": iid_effect,
                "branch_fired": rest["fired"],
                "branch_first_step": steps[rest["first"]] if rest["first"] is not None else None,
                "branch_max_z": rest["max_z"],
            }
        category = "dose0" if key.endswith(str(config.dose0)) else _category(p111, key)
        arms[key] = {"category": category, "seeds": seeds}
    called = [s for a in arms.values() if a["category"] == "called" for s in a["seeds"].values()]
    reinit4 = [s for k, a in arms.items() if k.endswith("reinit4") for s in a["seeds"].values()]
    return {
        "threshold_all5": full.threshold,
        "fired_loo": [bool(f) for f in loo],
        "arms": arms,
        "summary": {
            "loo_false_alarms": f"{sum(loo)}/{len(loo)}",
            "fires_on_called": f"{sum(s['fired'] for s in called)}/{len(called)}",
            "quiet_on_reinit4": f"{sum(not s['fired'] for s in reinit4)}/{len(reinit4)}",
            "called_min_margin": min(s["immediate"] / s["threshold"] for s in called),
            "reinit4_max_ratio": max(s["immediate"] / s["threshold"] for s in reinit4),
        },
    }


# --- item 4: JE's lr-1e-4 walk ------------------------------------------------------------------


def lr_walk(config: DictConfig) -> dict[str, Any]:
    """The gap of gaps per seed at the drive's end (and its peak), with the canary's own gap of gaps."""
    bb, cfg = str(config.lr_walk.backbone), config.lr_walk
    skip = _skip(config, bb)
    bands, grid = scalar_bands(config, bb, "cka_drift", skip)
    runs = {
        f"{lr}_{order}": read_corpus(_path(str(cfg[lr][order])) / "corpus" / "health.csv")
        for lr in ("hi", "lo")
        for order in ("walk", "iid")
    }
    canary, window = str(cfg.canary), int(cfg.canary_window)
    per_seed: dict[str, Any] = {}
    for seed in _seeds(runs["hi_walk"]):
        drift = {k: _by_step(rows, seed, "cka_drift") for k, rows in runs.items()}
        acc = {k: _by_step(rows, seed, canary) for k, rows in runs.items()}
        steps = [s for s in sorted(set.intersection(*(set(v) for v in drift.values()))) if s not in skip]
        gaps = [
            gap_of_gaps(
                drift["hi_walk"][s], drift["hi_iid"][s], drift["lo_walk"][s], drift["lo_iid"][s], bands, grid.index(s)
            )
            for s in steps
        ]
        tail = steps[-window:]
        per_seed[str(seed)] = {
            "end_step": steps[-1],
            "gap_end": gaps[-1],
            "fired_end": fires(gaps[-1], bands),
            "gap_peak": float(np.nanmax(gaps)),
            "gap_peak_step": steps[int(np.nanargmax(gaps))],
            "raw_end": {k: drift[k][steps[-1]] for k in drift},
            # The canary's gap of gaps, in points (negative = the lr-1e-4 walk loses more than lr 3e-5's).
            "canary_gap_end": float(
                np.mean([(acc["hi_walk"][s] - acc["hi_iid"][s]) - (acc["lo_walk"][s] - acc["lo_iid"][s]) for s in tail])
            ),
        }
    return {
        "threshold": bands.threshold,
        "seeds": per_seed,
        "summary": f"{sum(v['fired_end'] for v in per_seed.values())}/{len(per_seed)}",
    }


def analyze(config: DictConfig) -> dict[str, Any]:
    """Build the P1.1.3 items 1-4 report (Hydra-free core)."""
    report = {
        "jolt": {bb: restart_jolt(config, bb) for bb in config.jolt.backbones},
        "past": past_regime(config),
        "walk_null": walk_null(config),
        "lr_walk": lr_walk(config),
    }
    out = _path(str(config.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps_valid(report) + "\n", encoding="utf-8")
    _log(report)
    logger.info(f"wrote {out}")
    return report


def _log(report: dict[str, Any]) -> None:
    for bb, entry in report["jolt"].items():
        for order, seeds in entry["orders"].items():
            for seed, j in seeds.items():
                logger.info(
                    f"jolt {bb} {order} s{seed}: peak {j['peak']:.1f} @ {j['peak_step']}, "
                    f"over c {j['over_threshold']}, over 2sd {j['over_2sd']}, "
                    f"settled |gap| {j['settled_abs_median']:.1f}"
                )
    for name, panel in report["past"]["panels"].items():
        logger.info(f"past {name}: c={panel['threshold']:.2f} loo={panel['fpr_loo']:.1f} {panel['summary']}")
    logger.info(f"walk_null: {report['walk_null']['summary']}")
    logger.info(f"lr_walk: {report['lr_walk']['summary']} (c={report['lr_walk']['threshold']:.2f})")


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="cka_drift")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run the four reads and write the report."""
    analyze(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
