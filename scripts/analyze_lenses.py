"""P1.1.2 Stage 1 — screen the label-free readers against P1.1.1's damage branches.

A candidate reader earns a place only if it **fires on damage** the canary calls, **stays quiet on
damage that costs nothing** (re-initializing the top 4 blocks moves the weights a lot and the canary
not at all — a reader firing there reads the weights, not the information: the P0.5.3 trap), and
**stays quiet on the healthy null**. Two families are screened, per backbone:

* **lenses** (:mod:`cafl4ds.lenses`) on the features ``scripts/extract_block_features.py`` saved — one
  panel per lens, its layers sharing a threshold calibrated on the 5 healthy IID drives at their 18
  leg checkpoints (:mod:`cafl4ds.reading_protocol`);
* **scalars** — every label-free instrument the corpus logs, each a panel of its own, calibrated on
  P1.1.1's 10 healthy IID seeds at every step.

Each damaged arm × seed is read twice against its twin, in null seed-spread units
(:func:`cafl4ds.lenses.twin_effect`): **immediate** (the state right after the damage against the
state right before — no training in between) and **end** (the branch's last state against the dose-0
branch's). The report (``out``) holds, per backbone:

* ``lens_panels`` / ``scalar_panels`` — thresholds, leave-one-out false alarms, and (lenses) which
  layers fire on each walk seed read against the IID null;
* ``scalar_benign`` — each scalar's calls on movement that costs no information: the healthy walk,
  fresh IID seeds, and the dose-0 twins (the optimizer restart alone);
* ``arms`` — per arm and seed: the canary dent, the encoder-wide weight change, every reader's effects;
* ``screen`` — per reader: the share of canary-called seeds it fires on, the share of canary-quiet
  ones it stays quiet on (re-init 4 separately), the Spearman rank of its effect against the
  canary dent and against the weight change, and where it places re-init 4 on its noise-dose ladder;
* ``walk_localization`` — per lens and layer, the walk-vs-IID effect size at the block ends.

Example::

    uv run python scripts/analyze_lenses.py
"""

import json
import multiprocessing
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig
from torch import nn

from cafl4ds import branch
from cafl4ds.jsonio import dumps_valid
from cafl4ds.lenses import LENSES, fires, lens_values, twin_effect
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.null_envelope import checkpoint_steps, effect_sizes, read_corpus, seed_matrix
from cafl4ds.reading_protocol import (
    Bands,
    calibrate_panel,
    call,
    direction,
    leave_one_out_fired,
    short_batch_steps,
    spearman,
)

logger.remove()
logger.add(sys.stdout, level="INFO")

_WELL: dict[str, dict[str, torch.Tensor]] = {}


def _lens_job(job: tuple[str, str]) -> tuple[str, dict[str, float]]:
    """Worker: every lens of one saved checkpoint against the well (the well loaded once per process)."""
    path, well = job
    torch.set_num_threads(2)
    if well not in _WELL:
        _WELL[well] = torch.load(well)
    return path, lens_values(torch.load(path), _WELL[well])


def lens_table(fdir: Path, workers: int) -> dict[str, dict[str, float]]:
    """``features/<run>/s<seed>/<name>.pt -> {signal: value}`` for every indexed checkpoint, cached in the dir."""
    cache = fdir / "lens_values.json"
    table: dict[str, dict[str, float]] = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
    index = json.loads((fdir / "index.json").read_text(encoding="utf-8"))
    todo = [
        (str(fdir / e["features"]), str(fdir / "features" / "well.pt")) for e in index if e["features"] not in table
    ]
    if todo:
        logger.info(f"{fdir.name}: computing lenses on {len(todo)} checkpoints")
        # spawn, not fork: the parent has run torch ops by now, and forked OpenMP threads deadlock.
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for path, values in pool.imap_unordered(_lens_job, todo):
                table[str(Path(path).relative_to(fdir))] = values
        cache.write_text(json.dumps(table), encoding="utf-8")
    return table


def _harm(effect: float, signal: str) -> float:
    """An effect's size in its harmful direction (its magnitude for a two-sided reader)."""
    return abs(effect) if direction(signal) == 0 else effect


def encoder_change(checkpoint: Path, mode: str, k: int, scale: float, seed: int) -> float:
    """Relative L2 change of *all* the encoder blocks' weight matrices under a damage (not just the damaged ones)."""
    state = torch.load(checkpoint, map_location="cpu")
    encoder = TinyViTEncoder(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0)
    blocks = {key[len("encoder.") :]: v for key, v in state.items() if key.startswith("encoder.blocks.")}
    encoder.load_state_dict(blocks, strict=False)  # the blocks only: all the damage touches
    linears = [m for block in encoder.blocks for m in block.modules() if isinstance(m, nn.Linear)]
    before = [m.weight.detach().clone() for m in linears]
    branch.damage_top_blocks(encoder, mode, k, scale, seed=seed)
    num = sum(float((m.weight.detach() - b).pow(2).sum()) for m, b in zip(linears, before, strict=True))
    return float((num / sum(float(b.pow(2).sum()) for b in before)) ** 0.5)


def _grid(null_root: Path, bb: str) -> list[tuple[str, int]]:
    """The leg-checkpoint grid of the drives: ``(file stem, step)`` in order."""
    index = json.loads((null_root / "warm" / f"{bb}_o0.0_iid" / "checkpoints" / "s0" / "index.json").read_text())
    return [(Path(e["file"]).stem, int(e["batches_seen"])) for e in index]


def _lens_nulls(
    table: dict[str, dict[str, float]], run: str, stems: list[str], seeds: list[int]
) -> dict[str, np.ndarray]:
    """Every lens signal's ``[seeds, checkpoints]`` matrix over a drive's leg checkpoints."""
    rows = [[table[f"features/{run}/s{s}/{stem}.pt"] for stem in stems] for s in seeds]
    return {sig: np.array([[c[sig] for c in row] for row in rows]) for sig in rows[0][0]}


def _panels_by_lens(signals: list[str]) -> dict[str, list[str]]:
    return {lens: [s for s in signals if s.split("@")[0] == lens] for lens in LENSES}


def calibrate_lenses(
    iid: dict[str, np.ndarray], walk: dict[str, np.ndarray], grid: list[tuple[str, int]], config: DictConfig
) -> tuple[dict[str, Bands], dict[str, Any]]:
    """One panel per lens on the IID null; its leave-one-out false alarms; the walk seeds' calls per layer."""
    settings = {"span": int(config.span), "floor": float(config.floor)}
    rules: dict[str, Bands] = {}
    report: dict[str, Any] = {}
    for lens, panel in _panels_by_lens(list(iid)).items():
        nulls = {s: iid[s] for s in panel}
        rules.update(calibrate_panel(nulls, dict.fromkeys(panel, settings), float(config.alpha)))
        loo = leave_one_out_fired(nulls, dict.fromkeys(panel, settings), float(config.alpha))
        walk_calls = []
        for i in range(next(iter(walk.values())).shape[0]):
            calls = {s.split("@")[1]: call(walk[s][i], rules[s]) for s in panel}
            fired = {layer: c for layer, c in calls.items() if c["fired"]}
            walk_calls.append(
                {
                    "fired_layers": sorted(fired),
                    "first_step": min(
                        (grid[c["first"]][1] for c in fired.values() if c["first"] is not None), default=None
                    ),
                    "max_z": max(c["max_z"] for c in calls.values()),
                }
            )
        report[lens] = {"threshold": rules[panel[0]].threshold, "fpr_loo": float(np.mean(loo)), "walk": walk_calls}
    return rules, report


def _skip(null_root: Path, bb: str, config: DictConfig) -> set[int]:
    segments = json.loads((null_root / "warm" / f"{bb}_o0.0_walk" / "corpus" / "segments.json").read_text())
    return short_batch_steps(segments["segments"], int(config.batch_size), int(config.min_batch))


def calibrate_scalars(
    bb: str, config: DictConfig, skip: set[int]
) -> tuple[dict[str, Bands], list[int], dict[str, Any]]:
    """Each label-free scalar as a panel of its own on P1.1.1's calibration seeds (short-batch readings blanked).

    A panel of its own, not P1.1.1's shared label-free panel: each scalar is screened as a candidate
    reader by itself, and a shared threshold would be set by the spikiest member (grad norm).
    """
    dirs = [Path(to_absolute_path(d.format(bb=bb))) for d in config.scalars.calibration]
    rows = [r for d in dirs for r in read_corpus(d / "corpus" / "health.csv")]
    grid = checkpoint_steps(rows)
    blank = [i for i, step in enumerate(grid) if step in skip]
    settings = {"span": int(config.scalars.span), "floor": float(config.floor)}
    rules: dict[str, Bands] = {}
    panels: dict[str, Any] = {}
    for s in config.scalars.signals[bb]:
        matrix = seed_matrix(rows, s)
        matrix[:, blank] = np.nan
        rules.update(calibrate_panel({s: matrix}, {s: settings}, float(config.alpha)))
        loo = leave_one_out_fired({s: matrix}, {s: settings}, float(config.alpha))
        panels[s] = {"threshold": rules[s].threshold, "fpr_loo": float(np.mean(loo))}
    return rules, grid, panels


def benign_calls(
    bb: str, rules: dict[str, Bands], grid: list[int], skip: set[int], config: DictConfig
) -> dict[str, Any]:
    """Each scalar's own calls on movement that costs no information.

    Three kinds: the healthy walk (P1.1.0, seeds 0-4), fresh IID seeds (P1.1.1's check, 10-14), and the
    dose-0 twins (an optimizer restart only, seeds 0-2).
    """
    null_root, root = Path(to_absolute_path(str(config.null_root))), Path(to_absolute_path(str(config.root)))
    runs = {
        "walk": null_root / "warm" / f"{bb}_o0.0_walk",
        "fresh_iid": root / "heldout2" / f"{bb}_iid",
        "restart_iid": root / "branch" / f"{bb}_iid_{config.dose0}",
        "restart_walk": root / "branch" / f"{bb}_walk_{config.dose0}",
    }
    out: dict[str, Any] = {}
    for name, run in runs.items():
        rows = read_corpus(run / "corpus" / "health.csv")
        steps = checkpoint_steps(rows)
        idx = [grid.index(step) for step in steps]
        blank = [i for i, step in enumerate(steps) if step in skip]
        for signal, bands in rules.items():
            matrix = seed_matrix(rows, signal)
            matrix[:, blank] = np.nan
            calls = [call(row, bands.restrict(idx)) for row in matrix]
            out.setdefault(signal, {})[name] = {
                "fired": f"{sum(c['fired'] for c in calls)}/{len(calls)}",
                "max_z": [round(c["max_z"], 2) for c in calls],
            }
    return out


def _by_step(rows: list[dict[str, Any]], seed: int, signal: str) -> dict[int, float]:
    return {
        int(r["step"]): float(r[signal]) for r in rows if int(r["seed"]) == seed and isinstance(r.get(signal), float)
    }


def _canary_end(arm: dict[int, float], twin: dict[int, float], skip: set[int], window: int) -> float:
    steps = [s for s in sorted(set(arm) & set(twin)) if s not in skip][-window:]
    return float(np.mean([arm[s] - twin[s] for s in steps]))


def read_arms(
    bb: str,
    table: dict[str, dict[str, float]],
    rules: dict[str, Bands],
    grid: list[tuple[str, int]],
    scalar_rules: dict[str, Bands],
    scalar_grid: list[int],
    skip: set[int],
    config: DictConfig,
) -> dict[str, Any]:
    """Every branch arm × seed: canary dents, encoder-wide weight change, and every reader's twin effects."""
    root = Path(to_absolute_path(str(config.root)))
    stems = [stem for stem, _ in grid]
    i_pre, i_end = stems.index("step00065"), len(stems) - 1
    end_step = max(s for s in scalar_grid if s not in skip)
    # The pre-damage reading sits just before step 65; its spread is the nearest earlier reading's (step 64
    # may be absent: the loop skips a batch the filter leaves too small to update on).
    i_scalar_pre = max(i for i, s in enumerate(scalar_grid) if s < 65)
    i_scalar_end = scalar_grid.index(end_step)
    out: dict[str, Any] = {}
    for order in ("iid", "walk"):
        twin_name = f"{bb}_{order}_{config.dose0}"
        twin_rows = read_corpus(root / "branch" / twin_name / "corpus" / "health.csv")
        for arm_dir in sorted((root / "branch").glob(f"{bb}_{order}_*")):
            if arm_dir.name == twin_name:
                continue
            rows = read_corpus(arm_dir / "corpus" / "health.csv")
            arm: dict[str, Any] = {}
            for report in sorted((arm_dir / "reports").glob("*.json")):
                data = json.loads(report.read_text(encoding="utf-8"))
                seed, rec = int(data["config"]["seed"]), data["config"]["branch"]
                dmg = rec["damage"]
                parent = Path(to_absolute_path(str(rec["checkpoint"]).replace("/workspace/", "")))
                post = table[f"features/{arm_dir.name}/s{seed}/post_damage.pt"]
                pre = table[f"features/{bb}_o0.0_{order}/s{seed}/step00065.pt"]
                final = table[f"features/{arm_dir.name}/s{seed}/final.pt"]
                twin_final = table[f"features/{twin_name}/s{seed}/final.pt"]
                effects: dict[str, dict[str, float]] = {}
                for sig in pre:
                    effects[sig] = {
                        "immediate": twin_effect(post[sig], pre[sig], rules[sig], i_pre),
                        "end": twin_effect(final[sig], twin_final[sig], rules[sig], i_end),
                    }
                for sig, bands in scalar_rules.items():
                    a, t = _by_step(rows, seed, sig), _by_step(twin_rows, seed, sig)
                    imm = rec["post"].get(sig), rec["pre"].get(sig)
                    effects[sig] = {
                        "immediate": twin_effect(imm[0], imm[1], bands, i_scalar_pre) if None not in imm else np.nan,
                        "end": twin_effect(a[end_step], t[end_step], bands, i_scalar_end),
                    }
                canary = str(config.canary)
                arm[str(seed)] = {
                    "canary_immediate": rec["post"][canary] - rec["pre"][canary],
                    "canary_end": _canary_end(
                        _by_step(rows, seed, canary), _by_step(twin_rows, seed, canary), skip, int(config.canary_window)
                    ),
                    "rel_change_damaged": dmg["rel_change"],
                    "rel_change_encoder": encoder_change(parent, dmg["mode"], int(dmg["k"]), float(dmg["scale"]), seed),
                    "effects": effects,
                }
            out[f"{order}/{arm_dir.name[len(f'{bb}_{order}_') :]}"] = arm
    return out


def _category(p111: dict[str, Any], key: str) -> str:
    """``called`` if P1.1.1's pooled canary fired on every seed of the arm, ``quiet`` if on none, else ``partial``."""
    shares = p111["branches"][key]["fired_share"]
    share = max(shares["knn_acc"], shares["linear_acc"])
    return "called" if share == 1.0 else "quiet" if share == 0.0 else "partial"


def screen(
    arms: dict[str, Any], rules: dict[str, Bands], readers: dict[str, list[str]], p111: dict[str, Any]
) -> dict[str, Any]:
    """Per reader (a lens panel or one scalar): sensitivity, specificity and rank agreement, per read.

    ``ladder`` is the median harm per dose (over seeds and both orders). Re-init 4 moves the encoder's
    weights by ~0.66 — between noise σ0.5 and σ0.75 — at no canary cost, so
    ``reinit4_above_noise_doses`` places it on the reader's noise ladder: a reader of the lever puts
    it mid-ladder, a reader of the information loss below the smallest dose.
    """
    out: dict[str, Any] = {}
    for reader, signals in readers.items():
        entry: dict[str, Any] = {}
        for read in ("immediate", "end"):
            cells = []
            for key, seeds in arms.items():
                for seed in seeds.values():
                    effects = [seed["effects"][s][read] for s in signals]
                    if not np.isfinite(effects).any():
                        continue
                    harm = max(_harm(e, s) for e, s in zip(effects, signals, strict=True) if np.isfinite(e))
                    fired = [s.split("@")[-1] for e, s in zip(effects, signals, strict=True) if fires(e, rules[s])]
                    cells.append(
                        (key, _category(p111, key), harm, fired, seed[f"canary_{read}"], seed["rel_change_encoder"])
                    )
            if not cells:
                continue
            called = [c for c in cells if c[1] == "called"]
            quiet = [c for c in cells if c[1] == "quiet"]
            reinit4 = [c for c in quiet if c[0].endswith("reinit4")]
            harm = np.array([c[2] for c in cells])
            ladder = {
                dose: float(np.median([c[2] for c in cells if c[0].split("/")[1] == dose]))
                for dose in dict.fromkeys(c[0].split("/")[1] for c in cells)
            }
            noise = [v for dose, v in ladder.items() if dose.startswith("noise")]
            entry[read] = {
                "fires_on_called": f"{sum(bool(c[3]) for c in called)}/{len(called)}",
                "quiet_on_quiet": f"{sum(not c[3] for c in quiet)}/{len(quiet)}",
                "quiet_on_reinit4": f"{sum(not c[3] for c in reinit4)}/{len(reinit4)}",
                "spearman_canary_loss": spearman(harm, -np.array([c[4] for c in cells])),
                "spearman_weight_change": spearman(harm, np.array([c[5] for c in cells])),
                "layers_fired_on_reinit4": sorted({layer for c in reinit4 for layer in c[3]}),
                "ladder": ladder,
                "reinit4_above_noise_doses": sum(v < ladder["reinit4"] for v in noise) if "reinit4" in ladder else None,
            }
        out[reader] = entry
    return out


def walk_localization(
    iid: dict[str, np.ndarray], walk: dict[str, np.ndarray], grid: list[tuple[str, int]]
) -> dict[str, Any]:
    """Per lens and layer, the walk-vs-IID effect size (Cohen's d per checkpoint) at the time-of-day block ends."""
    marks = {"daytime_end": "step00055", "dawn_dusk_end": "step00065", "night_end": "final"}
    stems = [stem for stem, _ in grid]
    return {
        sig: {name: float(effect_sizes(walk[sig], iid[sig])[stems.index(stem)]) for name, stem in marks.items()}
        for sig in iid
    }


def analyze(config: DictConfig) -> dict[str, Any]:
    """Build the Stage-1 screen report (Hydra-free core)."""
    null_root = Path(to_absolute_path(str(config.null_root)))
    p111_all = json.loads(Path(to_absolute_path(str(config.reading_report))).read_text(encoding="utf-8"))
    report: dict[str, Any] = {}
    for bb in config.backbones:
        fdir = Path(to_absolute_path(str(config.features).format(bb=bb)))
        table = lens_table(fdir, int(config.workers))
        grid = _grid(null_root, bb)
        stems, seeds = [stem for stem, _ in grid], list(range(5))
        iid = _lens_nulls(table, f"{bb}_o0.0_iid", stems, seeds)
        walk = _lens_nulls(table, f"{bb}_o0.0_walk", stems, seeds)
        rules, lens_panels = calibrate_lenses(iid, walk, grid, config)
        skip = _skip(null_root, bb, config)
        scalar_rules, scalar_grid, scalar_panels = calibrate_scalars(bb, config, skip)
        arms = read_arms(bb, table, rules, grid, scalar_rules, scalar_grid, skip, config)
        # Lens panels as `<lens>@*` (every layer), so they never collide with the same-named scalars.
        readers = {f"{lens}@*": panel for lens, panel in _panels_by_lens(list(iid)).items()}
        readers.update({s: [s] for s in scalar_rules})
        report[bb] = {
            "grid": grid,
            "lens_panels": lens_panels,
            "scalar_panels": scalar_panels,
            "scalar_benign": benign_calls(bb, scalar_rules, scalar_grid, skip, config),
            "arms": arms,
            "screen": screen(arms, {**rules, **scalar_rules}, readers, p111_all[bb]),
            "walk_localization": walk_localization(iid, walk, grid),
        }
        _log(bb, report[bb])
    out = Path(to_absolute_path(str(config.out)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps_valid(report) + "\n", encoding="utf-8")  # an inapplicable read (nan) becomes null
    logger.info(f"wrote {out}")
    return report


def _log(bb: str, entry: dict[str, Any]) -> None:
    logger.info(f"== {bb}: lens panels (threshold, LOO false alarms, walk seeds firing)")
    for lens, panel in entry["lens_panels"].items():
        n_walk = sum(bool(w["fired_layers"]) for w in panel["walk"])
        logger.info(f"  {lens:18s} c={panel['threshold']:.2f} loo={panel['fpr_loo']:.1f} walk={n_walk}/5")
    for s, sp in entry["scalar_panels"].items():
        logger.info(f"  {s:18s} c={sp['threshold']:.2f} loo={sp['fpr_loo']:.1f}")
    logger.info(
        f"== {bb}: screen (immediate | end): fires on called, quiet on quiet, quiet on reinit4, rho canary / weight"
    )
    for reader, reads in entry["screen"].items():
        parts = []
        for read in ("immediate", "end"):
            r = reads.get(read)
            parts.append(
                "—"
                if r is None
                else f"{r['fires_on_called']:>5} {r['quiet_on_quiet']:>5} {r['quiet_on_reinit4']:>4} "
                f"{r['spearman_canary_loss']:+.2f}/{r['spearman_weight_change']:+.2f}"
            )
        logger.info(f"  {reader:22s} {parts[0]:32s} | {parts[1]}")


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="lenses")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Screen the readers and write the report."""
    analyze(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
