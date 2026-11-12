"""P1.1.1 reading protocol — calibrate single-trajectory rules on the null, then read every arm with them.

For each backbone, calibrates one rule per instrument on the healthy IID drives — the P1.1.0 deep
drives plus the first held-out seeds (``calibration``) — with one threshold per panel
(:mod:`cafl4ds.reading_protocol`), then reports:

* ``rules`` / ``panels`` — each rule's spread, each panel's threshold, and the panel's false-alarm
  rate leave-one-out (bands and threshold refitted without each seed) and on fresh IID seeds the
  calibration never saw (``check``);
* ``walk`` — the P1.1.0 walk seeds read by the rules: which instruments fire, when, and the
  adjudication (``quiet`` / ``movement`` / ``degradation``, the canary arbitrating);
* ``reversion`` — whether the walk's excursion tracks how mixed its recent diet is (Spearman of the
  walk-minus-IID gap against the trailing regime count, per seed);
* ``branches`` — every resumed drive (``<root>/branch/<bb>_<order>_<dose>``): the rules' calls on the
  branch's own grid, and the dent against the dose-0 twin of the same order and seed;
* ``recovery`` — every recovery pass (``<root>/recover/<name>``): the arbiters' trajectories;
* ``dial`` (JE) — the warm collapse dial (``<root>/dial/je_<order>_alpha<a>_<tag>``): the rules' calls, and the
  dent against the ``alpha0.0`` arm of the same order and tag.

Example::

    uv run python scripts/analyze_reading.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from cafl4ds.null_envelope import checkpoint_steps, read_corpus, seed_matrix
from cafl4ds.reading_protocol import (
    Bands,
    adjudicate,
    calibrate_panel,
    call,
    leave_one_out_fired,
    short_batch_steps,
    spearman,
    trailing_diversity,
)

logger.remove()
logger.add(sys.stdout, level="INFO")


def _union(dirs: list[str], bb: str) -> list[dict[str, Any]] | None:
    """The corpus rows of every run dir in ``dirs`` (``{bb}`` filled in) that has one, pooled (seeds disjoint)."""
    pooled = [r for d in dirs if (rows := _rows(Path(to_absolute_path(d.format(bb=bb))))) is not None for r in rows]
    return pooled or None


def _rows(path: Path) -> list[dict[str, Any]] | None:
    """A run dir's corpus rows, or ``None`` if it has none."""
    csv = path / "corpus" / "health.csv"
    if not csv.is_file():
        return None
    return read_corpus(csv)


def _reports(path: Path) -> dict[int, dict[str, Any]]:
    """A run dir's per-seed reports, keyed by seed."""
    out = {}
    for report in sorted((path / "reports").glob("*.json")):
        data = json.loads(report.read_text(encoding="utf-8"))
        out[int(data["config"]["seed"])] = data
    return out


def _seeds(rows: list[dict[str, Any]]) -> list[int]:
    return sorted({int(r["seed"]) for r in rows if r.get("arm") == "live"})


def _present(rows: list[dict[str, Any]], signals: list[str]) -> list[str]:
    return [s for s in signals if any(isinstance(r.get(s), float) for r in rows)]


def _matrix(rows: list[dict[str, Any]], signal: str, skip: set[int]) -> np.ndarray:
    """``seed_matrix`` with the checkpoints at ``skip`` steps blanked (``nan``) — readings the rule skips."""
    matrix = seed_matrix(rows, signal)
    blank = [i for i, step in enumerate(checkpoint_steps(rows)) if step in skip]
    matrix[:, blank] = np.nan
    return matrix


def skipped_steps(walk_dir: Path, config: DictConfig) -> set[int]:
    """The steps after a short batch, from the walk's leg map (the IID control shares its batch sizes)."""
    path = walk_dir / "corpus" / "segments.json"
    if not path.is_file():
        return set()
    segments = json.loads(path.read_text(encoding="utf-8"))["segments"]
    return short_batch_steps(segments, int(config.batch_size), int(config.min_batch))


def _window(signal: str, arbiters: list[str], config: DictConfig) -> int:
    return int(config.window_canary if signal in arbiters else config.window_label_free)


def _settings(panel: list[str], arbiters: list[str], config: DictConfig) -> dict[str, dict[str, Any]]:
    """Each rule's band settings: its trailing-mean window, the local-SD span and floor."""
    return {
        s: {"window": _window(s, arbiters, config), "span": int(config.span), "floor": float(config.floor)}
        for s in panel
    }


def panels(iid: list[dict[str, Any]], signals: list[str], arbiters: list[str]) -> dict[str, list[str]]:
    """The two panels present in the null corpus: ``label_free`` and ``canary``."""
    return {"label_free": _present(iid, signals), "canary": _present(iid, arbiters)}


def calibrate_rules(
    iid: list[dict[str, Any]], signals: list[str], arbiters: list[str], config: DictConfig, skip: set[int]
) -> dict[str, Bands]:
    """One calibrated rule per instrument present in the null corpus, each panel sharing one threshold."""
    rules = {}
    for panel in panels(iid, signals, arbiters).values():
        nulls = {s: _matrix(iid, s, skip) for s in panel}
        rules.update(calibrate_panel(nulls, _settings(panel, arbiters, config), float(config.alpha)))
    return rules


def read_arm(
    rows: list[dict[str, Any]], rules: dict[str, Bands], grid: list[int], arbiters: list[str], skip: set[int]
) -> dict[str, Any]:
    """Call every seed of an arm with every rule, on the arm's own stretch of the null grid.

    Args:
        rows: The arm's corpus rows.
        rules: The calibrated rules (on the full null grid).
        grid: The null's checkpoint steps.
        arbiters: The canary channels (they arbitrate the adjudication).
        skip: Steps whose readings the rules skip (after a short batch).

    Returns:
        ``{"steps", "per_seed": {seed: {"calls", "verdict"}}, "fired_share"}``.

    Raises:
        ValueError: If the arm checkpoints at a step the null does not.
    """
    steps = checkpoint_steps(rows)
    position = {step: i for i, step in enumerate(grid)}
    missing = [s for s in steps if s not in position]
    if missing:
        raise ValueError(f"arm checkpoints at steps the null does not: {missing[:5]}")
    index = [position[s] for s in steps]
    per_seed: dict[int, Any] = {}
    for signal, bands in rules.items():
        sub = bands.restrict(index)
        matrix = _matrix(rows, signal, skip)
        for k, seed in enumerate(_seeds(rows)):
            entry = per_seed.setdefault(seed, {"calls": {}})
            out: dict[str, Any] = dict(call(matrix[k], sub))
            out["first_step"] = steps[out["first"]] if out["first"] is not None else None
            entry["calls"][signal] = out
    for entry in per_seed.values():
        entry["verdict"] = adjudicate({s: bool(c["fired"]) for s, c in entry["calls"].items()}, arbiters)
        # Transient or persisting: is any arbiter still over its threshold at the arm's last reading?
        entry["persisting"] = any(
            entry["calls"][a]["final_z"] > rules[a].threshold for a in arbiters if a in entry["calls"]
        )
    n = max(len(per_seed), 1)
    fired_share = {s: sum(e["calls"][s]["fired"] for e in per_seed.values()) / n for s in rules}
    return {"steps": [steps[0], steps[-1]], "per_seed": per_seed, "fired_share": fired_share}


def reversion(walk: list[dict[str, Any]], iid: list[dict[str, Any]], signals: list[str], window: int) -> dict[str, Any]:
    """Does the walk's excursion track how mixed its recent diet is? Two reads of the walk-minus-IID gap.

    * ``per_seed`` / ``mean`` / ``same_sign`` — Spearman, per seed, of the gap against the trailing
      regime count over ``window`` batches (the first ``window`` checkpoints skipped while it fills).
    * ``blocks`` — the event-locked read: per seed, the mean gap over the last ``window`` readings of
      each time-of-day block of the walk (``daytime`` → ``dawn/dusk`` → ``night``, from the leg names).
      A diet-tracking excursion shrinks at the end of the short, mixed dawn/dusk block.
    """
    first = min(int(r["seed"]) for r in walk)
    ordered = sorted((r for r in walk if int(r["seed"]) == first), key=lambda r: r["checkpoint_index"])
    eras = [int(r["era_index"]) for r in ordered]
    blocks = [str(r["era_name"]).split("·", 1)[0] for r in ordered]
    ends = {b: max(i for i, x in enumerate(blocks) if x == b) for b in dict.fromkeys(blocks)}
    diversity = trailing_diversity(eras, window)
    out: dict[str, Any] = {}
    for signal in _present(walk, signals):
        gap = seed_matrix(walk, signal) - np.nanmean(seed_matrix(iid, signal), axis=0)
        rhos = [spearman(row[window:], diversity[window:]) for row in gap]
        out[signal] = {
            "per_seed": rhos,
            "mean": float(np.nanmean(rhos)),
            "same_sign": int(abs(np.sign(rhos).sum())),
            "blocks": {
                b: [float(np.nanmean(row[max(0, end - window + 1) : end + 1])) for row in gap]
                for b, end in ends.items()
            },
        }
    return out


def _dose_label(name: str, bb: str, order: str) -> str:
    return name[len(f"{bb}_{order}_") :]


def read_branches(
    root: Path, bb: str, rules: dict[str, Bands], grid: list[int], config: DictConfig, skip: set[int]
) -> dict[str, Any]:
    """Read every resumed drive of backbone ``bb``: rule calls, and the paired dent against its dose-0 twin."""
    arbiters = list(config.arbiters)
    out: dict[str, Any] = {}
    for order in ("iid", "walk"):
        dirs = sorted((root / "branch").glob(f"{bb}_{order}_*"))
        twin_dir = root / "branch" / f"{bb}_{order}_{config.dose0}"
        twin = _rows(twin_dir)
        for path in dirs:
            rows = _rows(path)
            if rows is None:
                continue
            dose = _dose_label(path.name, bb, order)
            entry = read_arm(rows, rules, grid, arbiters, skip)
            reports = _reports(path)
            entry["damage"] = {s: r["config"]["branch"]["damage"] for s, r in reports.items()}
            entry["immediate_dent"] = {
                s: {a: r["config"]["branch"]["post"].get(a, np.nan) - r["config"]["branch"]["pre"].get(a, np.nan)
                    for a in ("knn_acc", "linear_acc", "rankme", "cka_drift")}
                for s, r in reports.items()
            }  # fmt: skip
            if twin is not None and path != twin_dir:
                entry["dent_vs_dose0"] = _paired_dent(rows, twin, [*arbiters, *config.signals[bb]])
            out[f"{order}/{dose}"] = entry
    return out


def _paired_dent(rows: list[dict[str, Any]], twin: list[dict[str, Any]], signals: list[str]) -> dict[str, Any]:
    """Branch minus dose-0 twin, seed by seed: the mean over the branch and at its last checkpoint."""
    shared = sorted(set(_seeds(rows)) & set(_seeds(twin)))
    out = {}
    for signal in _present(rows, signals):
        a, b = seed_matrix(rows, signal), seed_matrix(twin, signal)
        ia = [_seeds(rows).index(s) for s in shared]
        ib = [_seeds(twin).index(s) for s in shared]
        diff = a[ia] - b[ib]
        out[signal] = {"mean": float(np.nanmean(diff)), "final": float(np.nanmean(diff[:, -1])), "n_seeds": len(shared)}
    return out


def read_dial(
    root: Path, rules: dict[str, Bands], grid: list[int], config: DictConfig, skip: set[int]
) -> dict[str, Any]:
    """Read the JE collapse dial (``<root>/dial/je_<order>_alpha<a>_<tag>``): calls, and the dent against ``alpha0.0``.

    Each arm is compared seed by seed with the ``alpha0.0`` arm of the same order and tag (same lr, same resume), since
    the null bands were calibrated at the drive's own lr and a dial arm at a higher lr moves more even at ``alpha=0``.
    """
    arbiters = list(config.arbiters)
    out: dict[str, Any] = {}
    for path in sorted((root / "dial").glob("je_*_alpha*_*")):
        rows = _rows(path)
        if rows is None:
            continue
        order, rest = path.name[len("je_") :].split("_alpha", 1)
        alpha, tag = rest.split("_", 1)
        entry = read_arm(rows, rules, grid, arbiters, skip)
        entry["alpha"] = float(alpha)
        twin_dir = path.parent / f"je_{order}_alpha0.0_{tag}"
        twin = _rows(twin_dir)
        if twin is not None and path != twin_dir:
            entry["dent_vs_alpha0"] = _paired_dent(rows, twin, [*arbiters, *config.signals["je"]])
        out[path.name] = entry
    return out


def read_recovery(root: Path, arbiters: list[str], signals: list[str]) -> dict[str, Any]:
    """Every recovery pass: each signal's seed-mean trajectory (start, every 5th step, end)."""
    out: dict[str, Any] = {}
    for path in sorted((root / "recover").glob("*")):
        rows = _rows(path)
        if rows is None:
            continue
        entry = {}
        for signal in _present(rows, [*arbiters, *signals]):
            mean = np.nanmean(seed_matrix(rows, signal), axis=0)
            entry[signal] = {"start": float(mean[0]), "end": float(mean[-1]), "trace": mean[::5].round(4).tolist()}
        out[path.name] = entry
    return out


def analyze(config: DictConfig) -> dict[str, Any]:
    """Build the full reading report (Hydra-free core, unit-testable)."""
    null_root, root = Path(to_absolute_path(str(config.null_root))), Path(to_absolute_path(str(config.root)))
    arbiters = list(config.arbiters)
    report: dict[str, Any] = {}
    for bb in config.backbones:
        iid, walk = _union(list(config.calibration), bb), _rows(null_root / "warm" / f"{bb}_o0.0_walk")
        if iid is None:
            logger.warning(f"{bb}: no null corpus — skipped")
            continue
        signals = list(config.signals[bb])
        skip = skipped_steps(null_root / "warm" / f"{bb}_o0.0_walk", config)
        rules = calibrate_rules(iid, signals, arbiters, config, skip)
        grid = checkpoint_steps(iid)
        check = _union(list(config.check), bb)
        entry: dict[str, Any] = {"calibration_seeds": _seeds(iid), "rules": {}, "panels": {}}
        for name, panel in panels(iid, signals, arbiters).items():
            settings = _settings(panel, arbiters, config)
            loo = leave_one_out_fired({s: _matrix(iid, s, skip) for s in panel}, settings, float(config.alpha))
            held = (
                {s: [bool(call(row, rules[s])["fired"]) for row in _matrix(check, s, skip)] for s in panel}
                if check
                else {}
            )
            for s in panel:
                entry["rules"][s] = {
                    "panel": name,
                    "sd_median": float(np.median(rules[s].sd)),
                    "rho": rules[s].rho,
                    "window": settings[s]["window"],
                    "direction": rules[s].direction,
                    "fired_check": float(np.mean(held[s])) if held else None,
                }
            entry["panels"][name] = {
                "signals": panel,
                "threshold": rules[panel[0]].threshold,
                "fpr_loo": float(np.mean(loo)),
                "fpr_check": float(np.mean(np.any(list(held.values()), axis=0))) if held else None,
                "check_seeds": _seeds(check) if check else [],
            }
        if walk is not None:
            entry["walk"] = read_arm(walk, rules, grid, arbiters, skip)
            entry["reversion"] = reversion(walk, iid, signals, int(config.diversity_window))
        entry["branches"] = read_branches(root, bb, rules, grid, config, skip)
        entry["skipped_steps"] = sorted(skip)
        entry["recovery"] = read_recovery(root, arbiters, signals)
        if bb == "je":
            entry["dial"] = read_dial(root, rules, grid, config, skip)
        report[bb] = entry
        _log(bb, entry)
    out = Path(to_absolute_path(str(config.out)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_finite(report), indent=2, default=float, allow_nan=False), encoding="utf-8")
    logger.info(f"wrote {out}")
    return report


def _finite(value: object) -> object:
    """``value`` with every non-finite float (``nan``, ``±inf``) replaced by ``None`` — strict JSON."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    if isinstance(value, float | np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _log(bb: str, entry: dict[str, Any]) -> None:
    """Print the rules' false-alarm rates and the walk's per-instrument firing share."""
    lines = [f"[{bb}] calibrated on seeds {entry['calibration_seeds']}; rule: fired on check seeds / walk fired share"]
    walk = entry.get("walk", {}).get("fired_share", {})
    for s, r in entry["rules"].items():
        held = "-" if r["fired_check"] is None else f"{r['fired_check']:.2f}"
        lines.append(f"  {s:<22} {held:>5}  {walk.get(s, float('nan')):.2f}")
    for name, p in entry["panels"].items():
        held = "-" if p["fpr_check"] is None else f"{p['fpr_check']:.2f}"
        lines.append(
            f"  panel {name}: threshold {p['threshold']:.2f}, FPR loo {p['fpr_loo']:.2f}, "
            f"check {held} (seeds {p['check_seeds']})"
        )
    if "walk" in entry:
        verdicts = [
            e["verdict"] + ("+persisting" if e["persisting"] else "") for e in entry["walk"]["per_seed"].values()
        ]
        lines.append(f"  walk verdicts: {verdicts}")
    for name, rev in entry.get("reversion", {}).items():
        ends = "  ".join(f"{b} {np.mean(v):+.3f}" for b, v in rev["blocks"].items())
        lines.append(
            f"  reversion {name}: rho {rev['mean']:+.2f} ({rev['same_sign']}/5 same sign); block-end gap {ends}"
        )
    for name, b in entry["branches"].items():
        fired = {s: v for s, v in b["fired_share"].items() if v > 0}
        lines.append(f"  branch {name}: fired {fired}")
    logger.info("\n".join(lines))


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="reading_protocol")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Read the null and the P1.1.1 branches into the reading report."""
    analyze(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
