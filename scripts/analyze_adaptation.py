"""P1.2 analysis — adapted-vs-frozen (B5) contrasts from the readout ladder's per-image correctness.

Reads the directories ``scripts/read_adaptation.py`` wrote (``pool.npz`` + one ``<encoder>.npz`` each)
and bootstraps every contrast over seeds and query images (:func:`cafl4ds.adaptation.bootstrap_delta`).
Few-shot rows are averaged over their paired label draws first.

* ``mode=p120`` — P1.2.0's descriptive contrasts on P1.1's artifacts, per pool (``heldout`` = the
  BDD-train pool, ``canary`` = P1.1.0's val probes) and backbone: the pretraining framing (well − B5,
  drive − B5), the deployment framing (drive end − its well), and random − B5. Not a gate.
* ``mode=p121`` — P1.2.1's pre-registered gate (:func:`cafl4ds.adaptation.gate_verdict`) per backbone
  on the end of the walk, plus Δ at every saved checkpoint.
* ``mode=p124`` — P1.2.4's attribution verdict per off-domain control
  (:func:`cafl4ds.adaptation.attribution_verdict`), and the confirmed recipe against the best frozen floor.
* ``mode=p125`` — P1.2.5's reads per init against its own frozen floor: the gate (with the seed-level test
  beside the bootstrap, and whether the ``cls`` readout reverses it), breadth over the task panel, the
  time-to-payoff curve, the walk-vs-IID secondary and the attribution controls; then which DINO arm carries
  forward to P1.3, and the claim level.
* ``mode=p125f`` — P1.2.5's Part F (exploratory): per arm, Δ against the floor within time-of-day and weather
  slices at every walk checkpoint, the IID end per slice, and window contrasts (a slice's change between two
  checkpoints, or its difference from another slice's); optionally each task's frequency in BDD-train.
* ``mode=p125g`` — P1.2.5's Part G: each unprotected arm read as ``p125`` reads its arms, the recipe contrast
  (protected − unprotected) at every checkpoint, and the every-rung curve of the re-read confirmation drives.

Writes ``<root>/adaptation_report.json`` and prints the tables. CPU only; cap ``threads`` beside HPU jobs.

Example::

    uv run python scripts/analyze_adaptation.py mode=p120 root=outputs/p120
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig
from threadpoolctl import threadpool_limits

from cafl4ds import adaptation

logger.remove()
logger.add(sys.stdout, level="INFO")


class Ladder:
    """One reader output directory: the shared query metadata and lazily loaded per-encoder arrays."""

    def __init__(self, path: Path) -> None:
        """Open ``path`` (it must hold ``pool.npz``)."""
        self.path = path
        pool = np.load(path / "pool.npz")
        self.query = pool["query_index"]
        self.stratum = pool["stratum"]
        self.regime = pool["regime"]
        # Each task's frames within the shared query (older reads hold no ``pos|`` keys: every frame).
        tasks = [k.split("|", 1)[1] for k in pool.files if k.startswith("labels|")]
        self.pos = {t: pool[f"pos|{t}"] if f"pos|{t}" in pool.files else np.arange(self.query.size) for t in tasks}
        self.labels = {t: pool[f"labels|{t}"][self.pos[t]] for t in tasks}
        # the per-patch task's patches per class per query frame (P1.2.5)
        self.counts = {k.split("|", 1)[1]: pool[k] for k in pool.files if k.startswith("counts|")}
        self.summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        self._cache: dict[str, Any] = {}

    def has(self, name: str) -> bool:
        """Whether encoder ``name`` was read here."""
        return (self.path / f"{name.replace('/', '__')}.npz").is_file()

    def rows(self, names: list[str], key: str) -> np.ndarray | None:
        """``[S, N]`` correctness of ``names`` for ``task|rung`` (draws averaged), or ``None`` if any lacks it."""
        if not names:
            return None
        out = []
        for name in names:
            if name not in self._cache:
                self._cache[name] = np.load(self.path / f"{name.replace('/', '__')}.npz")
            arrays = self._cache[name]
            if key not in arrays.files:
                return None
            out.append(arrays[key].mean(axis=0))
        return np.stack(out)


def _contrast(
    adapted: np.ndarray | None, reference: np.ndarray | None, labels: np.ndarray, config: DictConfig
) -> dict[str, float] | None:
    if adapted is None or reference is None:
        return None
    return adaptation.bootstrap_delta(adapted, reference, labels, n_boot=int(config.n_boot), seed=int(config.seed))


def _strata(
    adapted: np.ndarray, reference: np.ndarray, ladder: Ladder, task: str, config: DictConfig
) -> dict[str, dict[str, float]]:
    """The contrast within each time-of-day stratum (strata missing a class are skipped)."""
    names = ladder.summary.get("stratum_names", {})
    labels = ladder.labels[task]
    stratum = ladder.stratum[ladder.pos[task]]
    out = {}
    for sid in np.unique(stratum):
        keep = stratum == sid
        if np.unique(labels[keep]).size == np.unique(labels).size:
            result = _contrast(adapted[:, keep], reference[:, keep], labels[keep], config)
            if result is not None:
                out[names.get(str(int(sid)), str(int(sid)))] = result
    return out


def _check_aligned(a: Ladder, b: Ladder) -> None:
    same_tasks = a.labels.keys() == b.labels.keys() and all(np.array_equal(a.pos[t], b.pos[t]) for t in a.pos)
    if not np.array_equal(a.query, b.query) or not same_tasks:
        raise ValueError(f"{a.path} and {b.path} read different query sets; their contrasts would not pair.")


def p120(config: DictConfig, root: Path) -> dict[str, Any]:
    """P1.2.0's descriptive contrasts (see the module docstring)."""
    report: dict[str, Any] = {}
    seeds = range(int(config.n_seeds))
    for pool in config.pools:
        if not (root / pool / "je").is_dir():
            logger.warning(f"no {pool} ladder under {root}; skipped")
            continue
        refs = Ladder(root / pool / "je")  # B5 and the random init are read once, in the JE list
        report[pool] = {}
        for bb in config.backbones:
            ladder = refs if bb == "je" else Ladder(root / pool / bb)
            _check_aligned(refs, ladder)
            groups = {"well": [f"{bb}_well"]}
            for order in ("walk", "iid"):
                for point in ("mid", "end"):
                    groups[f"{order}_{point}"] = [f"{bb}_{order}_s{s}_{point}" for s in seeds]
            rows: dict[str, Any] = {}
            for key in sorted(np.load(refs.path / "b5.npz").files):
                labels = refs.labels[key.split("|", 1)[0]]
                b5 = refs.rows(["b5"], key)
                if b5 is None:
                    continue
                got = {g: ladder.rows(n, key) for g, n in groups.items()}
                entry: dict[str, Any] = {
                    "bacc": {
                        "b5": adaptation.balanced_accuracy(b5[0], labels),
                        **{
                            g: float(np.mean([adaptation.balanced_accuracy(r, labels) for r in v]))
                            for g, v in got.items()
                            if v is not None
                        },
                    },
                    "vs_b5": {g: _contrast(v, b5, labels, config) for g, v in got.items() if v is not None},
                    "vs_well": {
                        g: _contrast(got[g], got["well"], labels, config)
                        for g in ("walk_end", "iid_end")
                        if got.get(g) is not None and got["well"] is not None
                    },
                }
                if got["walk_end"] is not None:
                    entry["walk_end_vs_b5_by_stratum"] = _strata(
                        got["walk_end"], b5, refs, key.split("|", 1)[0], config
                    )
                if bb == "je" and refs.has("random"):
                    rand = refs.rows(["random"], key)
                    entry["bacc"]["random"] = (
                        adaptation.balanced_accuracy(rand[0], labels) if rand is not None else None
                    )
                    entry["random_vs_b5"] = _contrast(rand, b5, labels, config)
                rows[key] = entry
            report[pool][bb] = rows
    return report


def p121(config: DictConfig, root: Path) -> dict[str, Any]:
    """P1.2.1's gate per backbone, and Δ at every saved checkpoint."""
    report: dict[str, Any] = {}
    for bb in config.backbones:
        ladder = Ladder(root / "ladder" / bb)
        tasks = list(config.tasks)
        ends = [f"walk_s{s}_end" for s in range(int(config.n_seeds))]
        b5_name = str(config.b5_name)
        contrasts: dict[str, dict[str, dict[str, float]]] = {}
        for task in tasks:
            labels = ladder.labels[task]
            contrasts[task] = {}
            for rung in config.rungs:
                result = _contrast(
                    ladder.rows(ends, f"{task}|{rung}"), ladder.rows([b5_name], f"{task}|{rung}"), labels, config
                )
                if result is not None:
                    contrasts[task][rung] = result
        gate = adaptation.gate_verdict(
            {t: contrasts[t] for t in config.gate_tasks},
            str(config.primary[bb]),
            delta_floor=float(config.delta_floor),
            alpha=float(config.alpha),
        )
        # Δ(t): every walk checkpoint suffix (e.g. t00100, end), all rungs present
        suffixes = sorted(
            {m.group(1) for p in ladder.path.glob("walk_s*_*.npz") if (m := re.match(r"walk_s\d+_(\w+)\.npz", p.name))}
        )
        trajectory: dict[str, Any] = {}
        for suffix in suffixes:
            names = [f"walk_s{s}_{suffix}" for s in range(int(config.n_seeds)) if ladder.has(f"walk_s{s}_{suffix}")]
            trajectory[suffix] = {
                f"{task}|{rung}": _contrast(
                    ladder.rows(names, f"{task}|{rung}"),
                    ladder.rows([b5_name], f"{task}|{rung}"),
                    ladder.labels[task],
                    config,
                )
                for task in tasks
                for rung in config.rungs
            }
        report[bb] = {
            "gate": gate,
            "contrasts": contrasts,
            "trajectory": trajectory,
            "secondary": _p121_secondary(config, root, bb, ladder),
        }
    return report


def p124(config: DictConfig, root: Path) -> dict[str, Any]:
    """P1.2.4's pre-registered reads: attribution per off-domain control, and the confirmation against the best floor.

    Attribution (A1): D_c = the BDD IID drive's arm − the control's arm (both adapted; B5 cancels), on the
    primary rung, per task, with both arms' seeds resampled (:func:`cafl4ds.adaptation.bootstrap_contrast`).
    The floor test: the confirmed recipe against, per task, the frozen backbone reading highest on the
    primary rung. Every ladder must have read the same query frames.
    """
    p = config.p124
    rung, tasks = str(p.rung), list(config.gate_tasks)
    bdd = Ladder(Path(to_absolute_path(str(p.bdd_ladder))))
    controls = Ladder(Path(to_absolute_path(str(p.control_ladder))))
    _check_aligned(bdd, controls)
    iid = [f"iid_s{s}_end" for s in range(int(config.n_iid_seeds))]
    attribution: dict[str, Any] = {}
    for control in p.controls:
        arms = [f"{control}_s{s}_end" for s in range(int(p.n_control_seeds))]
        per_task: dict[str, Any] = {}
        for task in tasks:
            key, labels = f"{task}|{rung}", bdd.labels[task]
            first, second, b5 = bdd.rows(iid, key), controls.rows(arms, key), bdd.rows([str(config.b5_name)], key)
            if first is None or second is None or b5 is None:
                raise ValueError(f"missing {key} reads for the {control} contrast.")
            gain = adaptation.bootstrap_delta(first, b5, labels, n_boot=int(config.n_boot), seed=int(config.seed))
            contrast = adaptation.bootstrap_contrast(
                first, second, labels, n_boot=int(config.n_boot), seed=int(config.seed)
            )
            per_task[task] = {
                **contrast,
                "gain_bdd": gain["delta"],
                "gain_control": adaptation.bootstrap_delta(second, b5, labels, n_boot=int(config.n_boot))["delta"],
                "share_stream": contrast["delta"] / gain["delta"] if gain["delta"] else None,
            }
        attribution[control] = {
            **adaptation.attribution_verdict(
                per_task, generic_ceiling=float(p.generic_ceiling), alpha=float(config.alpha)
            ),
            "tasks": per_task,
        }
    report: dict[str, Any] = {
        "attribution": attribution,
        "attribution_claim": adaptation.weakest_attribution([a["verdict"] for a in attribution.values()]),
    }
    if p.get("confirm_ladder"):
        confirm = Ladder(Path(to_absolute_path(str(p.confirm_ladder))))
        ends = [f"walk_s{s}_end" for s in range(int(config.n_seeds))]
        floors = {
            str(config.b5_name): bdd,
            **{str(n): Ladder(Path(to_absolute_path(str(d)))) for n, d in p.floors.items()},
        }
        for ladder in floors.values():
            _check_aligned(confirm, ladder)
        vs_floor: dict[str, Any] = {}
        for task in tasks:
            key, labels = f"{task}|{rung}", confirm.labels[task]
            reads = {}
            for name, ladder in floors.items():
                row = ladder.rows([name], key)
                if row is not None:
                    reads[name] = (adaptation.balanced_accuracy(row[0], labels), row)
            best = max(reads, key=lambda n: reads[n][0])
            vs_floor[task] = {
                "floor": best,
                "floor_reads": {n: r[0] for n, r in reads.items()},
                **adaptation.bootstrap_delta(
                    confirm.rows(ends, key), reads[best][1], labels, n_boot=int(config.n_boot), seed=int(config.seed)
                ),
            }
        beats = adaptation.holm({t: c["p_one_sided"] for t, c in vs_floor.items()}, float(config.alpha))
        report["vs_best_floor"] = {"beats": any(beats.values()), "holm": beats, "tasks": vs_floor}
    return report


def _bacc_rows(ladder: Ladder, rows: np.ndarray, task: str) -> np.ndarray:
    """Each row's balanced accuracy (patch-level for the per-patch task)."""
    if adaptation.is_patch_task(task):
        return np.array([adaptation.patch_balanced_accuracy(r, ladder.counts[task]) for r in rows])
    return np.array([adaptation.balanced_accuracy(r, ladder.labels[task]) for r in rows])


def _versus_floor(
    ladder: Ladder, adapted: list[str], floor: str, task: str, key: str, config: DictConfig
) -> dict[str, Any] | None:
    """Δ (adapted − floor): the bootstrap (it decides) and the seed-level one-sample *t*-test beside it."""
    rows, ref = ladder.rows(adapted, key), ladder.rows([floor], key)
    if rows is None or ref is None:
        return None
    if adaptation.is_patch_task(task):
        boot = adaptation.bootstrap_delta_patches(
            rows, ref[0], ladder.counts[task], n_boot=int(config.n_boot), seed=int(config.seed)
        )
    else:
        boot = adaptation.bootstrap_delta(
            rows, ref, ladder.labels[task], n_boot=int(config.n_boot), seed=int(config.seed)
        )
    out: dict[str, Any] = {**boot}
    if rows.shape[0] > 1:
        out["seed_level"] = adaptation.seed_ttest(
            _bacc_rows(ladder, rows, task), float(_bacc_rows(ladder, ref, task)[0])
        )
    return out


def _between_arms(
    first: tuple[Ladder, list[str]], second: tuple[Ladder, list[str]], task: str, key: str, config: DictConfig
) -> dict[str, Any] | None:
    """First − second, both adapted: the two-sample bootstrap and the one-sided Welch test beside it (image tasks)."""
    a, b = first[0].rows(first[1], key), second[0].rows(second[1], key)
    if a is None or b is None or adaptation.is_patch_task(task):
        return None
    out: dict[str, Any] = adaptation.bootstrap_contrast(
        a, b, first[0].labels[task], n_boot=int(config.n_boot), seed=int(config.seed)
    )
    if a.shape[0] > 1 and b.shape[0] > 1:
        out["seed_level"] = adaptation.seed_welch(_bacc_rows(first[0], a, task), _bacc_rows(second[0], b, task))
    return out


def _holm_both(contrasts: dict[str, dict[str, Any]], alpha: float) -> dict[str, Any]:
    """Holm over the family by the bootstrap (the decision) and by the seed-level test, and whether they disagree."""
    boot = adaptation.holm({t: c["p_one_sided"] for t, c in contrasts.items()}, alpha)
    seed = (
        adaptation.holm({t: c["seed_level"]["p_one_sided"] for t, c in contrasts.items()}, alpha)
        if all("seed_level" in c for c in contrasts.values())
        else None
    )
    return {"bootstrap": boot, "seed_level": seed, "disagree": seed is not None and seed != boot}


def _checkpoint_order(ladder: Ladder) -> list[str]:
    """The walk's checkpoint suffixes in drive order (``t00100`` …, then ``end``)."""
    suffixes = {
        m.group(1) for p in ladder.path.glob("walk_s*_*.npz") if (m := re.match(r"walk_s\d+_(\w+)\.npz", p.name))
    }
    return sorted(suffixes, key=lambda s: (s == "end", int(s[1:]) if s != "end" else 0))


def _walk_vs_iid(config: DictConfig, ladder: Ladder, ends: list[str], iid: list[str]) -> dict[str, Any]:
    """P1.2.1's secondary under the confirmed recipe: walk end against IID end, pooled and per stratum."""
    secondary: dict[str, Any] = {}
    names = ladder.summary.get("stratum_names", {})
    for task in config.gate_tasks:
        labels, stratum = ladder.labels[task], ladder.stratum[ladder.pos[task]]
        for rung in ("knn", "linear"):
            key = f"{task}|{rung}"
            walk_rows, iid_rows = ladder.rows(ends, key), ladder.rows(iid, key)
            entry: dict[str, Any] = {"pooled": _between_arms((ladder, ends), (ladder, iid), task, key, config)}
            if walk_rows is None or iid_rows is None:
                secondary[key] = entry
                continue
            for sid in np.unique(stratum):
                keep = stratum == sid
                if np.unique(labels[keep]).size == np.unique(labels).size:
                    entry[names.get(str(int(sid)), str(int(sid)))] = adaptation.bootstrap_contrast(
                        walk_rows[:, keep],
                        iid_rows[:, keep],
                        labels[keep],
                        n_boot=int(config.n_boot),
                        seed=int(config.seed),
                    )
            secondary[key] = entry
    return secondary


def _curve(config: DictConfig, ladder: Ladder, tasks: list[str]) -> dict[str, Any]:
    """Δ against the floor at each walk checkpoint, per task: kNN and linear, and every other rung read there.

    Time to payoff is the first checkpoint whose linear Δ has its lower bound above zero.
    """
    floor, seeds = str(config.p125.floor_name), range(int(config.n_seeds))
    curve: dict[str, Any] = {}
    for task in tasks:
        points: dict[str, Any] = {}
        for suffix in _checkpoint_order(ladder):
            names = [f"walk_s{s}_{suffix}" for s in seeds if ladder.has(f"walk_s{s}_{suffix}")]
            reads = {r: _versus_floor(ladder, names, floor, task, f"{task}|{r}", config) for r in config.curve_rungs}
            points[suffix] = {r: c for r, c in reads.items() if c is not None or r in ("knn", "linear")}
        payoff = next((s for s, pt in points.items() if pt["linear"] is not None and pt["linear"]["lo"] > 0), None)
        curve[task] = {"points": points, "time_to_payoff": payoff}
    return curve


def _p125_arm(config: DictConfig, ladder: Ladder, arm: str) -> dict[str, Any]:
    """One init's reads against its own frozen floor: the gate, breadth, the curve and the walk-vs-IID secondary."""
    p = config.p125
    floor, seeds = str(p.floor_name), range(int(config.n_seeds))
    ends = [f"walk_s{s}_end" for s in seeds]
    iid = [f"iid_s{s}_end" for s in range(int(config.n_iid_seeds)) if ladder.has(f"iid_s{s}_end")]
    alpha = float(config.alpha)
    out: dict[str, Any] = {}

    gate_contrasts: dict[str, dict[str, Any]] = {}
    for task in config.gate_tasks:
        gate_contrasts[task] = {}
        for rung in config.rungs:
            for at in ("", "@cls"):
                result = _versus_floor(ladder, ends, floor, task, f"{task}|{rung}{at}", config)
                if result is not None:
                    gate_contrasts[task][f"{rung}{at}"] = result
    primary = str(p.primary_rung)
    gate = adaptation.gate_verdict(
        {t: {r: c for r, c in rungs.items() if "@" not in r} for t, rungs in gate_contrasts.items()},
        primary,
        delta_floor=float(config.delta_floor),
        alpha=alpha,
    )
    gate["tests"] = _holm_both({t: gate_contrasts[t][primary] for t in config.gate_tasks}, alpha)
    gate["cls_reverses"] = {
        t: bool(np.sign(rungs[f"{primary}@cls"]["delta"]) != np.sign(rungs[primary]["delta"]))
        for t, rungs in gate_contrasts.items()
        if f"{primary}@cls" in rungs
    }
    out["gate"], out["gate_contrasts"] = gate, gate_contrasts

    panel = [t for t in p.panel if t in ladder.labels]
    breadth: dict[str, dict[str, dict[str, Any]]] = {}
    for t in panel:
        reads = {r: _versus_floor(ladder, ends, floor, t, f"{t}|{r}", config) for r in ("linear", "knn")}
        breadth[t] = {r: c for r, c in reads.items() if c is not None}
    family = {t: rungs["linear"] for t, rungs in breadth.items() if "linear" in rungs}
    tests = _holm_both(family, alpha) if family else {"bootstrap": {}, "seed_level": None, "disagree": False}
    gains = sum(tests["bootstrap"].values())
    out["breadth"] = {
        "tasks": breadth,
        "tests": tests,
        "gains": gains,
        "of": len(family),
        "most": gains > len(family) / 2,
    }

    out["curve"] = _curve(config, ladder, [*config.gate_tasks, *panel])

    if iid:
        out["walk_vs_iid"] = _walk_vs_iid(config, ladder, ends, iid)
    return out


def _p125_attribution(config: DictConfig, ladder: Ladder, controls: dict[str, int]) -> dict[str, Any]:
    """P1.2.4's attribution under the confirmed recipe: D_c = the BDD IID arm − each control's arm, per gate task."""
    iid = [f"iid_s{s}_end" for s in range(int(config.p125.n_attribution_iid))]
    rung = str(config.p125.primary_rung)
    per_control: dict[str, Any] = {}
    for control, n in controls.items():
        arms = [f"{control}_s{s}_end" for s in range(int(n))]
        reads = {t: _between_arms((ladder, iid), (ladder, arms), t, f"{t}|{rung}", config) for t in config.gate_tasks}
        tasks = {t: c for t, c in reads.items() if c is not None}
        if len(tasks) != len(reads):
            raise ValueError(f"missing {rung} reads for the {control} contrast in {ladder.path}.")
        verdict = adaptation.attribution_verdict(
            tasks, generic_ceiling=float(config.p125.generic_ceiling), alpha=float(config.alpha)
        )
        per_control[control] = {**verdict, "tests": _holm_both(tasks, float(config.alpha)), "tasks": tasks}
    return {
        "controls": per_control,
        "claim": adaptation.weakest_attribution([c["verdict"] for c in per_control.values()]) if per_control else None,
    }


def _carry_forward(config: DictConfig, ladders: dict[str, Ladder], arms: dict[str, Any]) -> dict[str, Any]:
    """Which DINO arm becomes P1.3's JE vehicle (the rule fixed in P1.2.5's design).

    One passing arm carries forward. If both pass, DINO-init DINO does, unless the two-sample contrast
    (SimSiam − DINO, primary rung) excludes zero in SimSiam's favour on at least one gate task and in DINO's
    favour on none. If neither passes, the MAE-init hybrid stays.
    """
    dino, simsiam = str(config.p125.dino_arm), str(config.p125.simsiam_arm)
    passing = [a for a in (dino, simsiam) if a in arms and arms[a]["gate"]["verdict"] == adaptation.PASS]
    if not passing:
        return {"vehicle": str(config.p125.fallback_vehicle), "passing": [], "contrast": None}
    if len(passing) == 1:
        return {"vehicle": passing[0], "passing": passing, "contrast": None}
    ends = [f"walk_s{s}_end" for s in range(int(config.n_seeds))]
    rung = str(config.p125.primary_rung)
    contrast = {
        t: _between_arms((ladders[simsiam], ends), (ladders[dino], ends), t, f"{t}|{rung}", config)
        for t in config.gate_tasks
    }
    for_simsiam = any(c is not None and c["lo"] > 0 for c in contrast.values())
    for_dino = any(c is not None and c["hi"] < 0 for c in contrast.values())
    return {"vehicle": simsiam if for_simsiam and not for_dino else dino, "passing": passing, "contrast": contrast}


def p125(config: DictConfig, root: Path) -> dict[str, Any]:
    """P1.2.5's pre-registered reads (docs/experiments/phase1/P1.2.5.md), per init, then the claim.

    Every arm's ladder (``<root>/ladder/<arm>``) holds its frozen floor (``p125.floor_name``), the walk's
    checkpoints (``walk_s<k>_<t>``), its IID ends and, where run, the attribution controls' ends. Every ladder
    must have read the same query frames.
    """
    p = config.p125
    ladders = {str(a): Ladder(root / "ladder" / str(a)) for a in p.arms if (root / "ladder" / str(a)).is_dir()}
    first = next(iter(ladders.values()))
    for ladder in ladders.values():
        _check_aligned(first, ladder)
    arms = {a: _p125_arm(config, ladder, a) for a, ladder in ladders.items()}
    for arm, controls in (p.get("attribution") or {}).items():
        if arm in ladders:
            arms[arm]["attribution"] = _p125_attribution(config, ladders[arm], dict(controls))
    carry = _carry_forward(config, ladders, arms)
    dino_arms = [a for a in (str(p.dino_arm), str(p.simsiam_arm)) if a in arms]
    passing = [a for a in dino_arms if arms[a]["gate"]["verdict"] == adaptation.PASS]
    if passing:
        broadest = max(passing, key=lambda a: arms[a]["breadth"]["gains"])
        level = "Strong" if arms[broadest]["breadth"]["most"] else "Qualified"
        claim: dict[str, str | None] = {"level": level, "backbone": broadest}
    else:
        claim = {"level": "Weak", "backbone": None}
    floors = {}
    for name, path in (p.get("reference_floors") or {}).items():
        ref = Ladder(Path(to_absolute_path(str(path))))
        _check_aligned(first, ref)
        floors[str(name)] = {k: v["pooled"] for k, v in ref.summary["encoders"][str(name)]["bacc"].items()}
    return {"arms": arms, "carry_forward": carry, "claim": claim, "reference_floors": floors}


def _recipe_contrast(config: DictConfig, first: Ladder, second: Ladder) -> dict[str, Any]:
    """First − second (two adapted arms, same start) at every checkpoint both read, per task and rung.

    At the end, the primary rung over the gate tasks is tested with Holm; everything else is descriptive.
    """
    _check_aligned(first, second)
    seeds = range(int(config.n_seeds))
    tasks = [t for t in [*config.gate_tasks, *config.p125.panel] if t in first.labels]
    rungs = [f"{r}{at}" for r in config.curve_rungs for at in ("", "@cls")]
    points: dict[str, Any] = {}
    shared = set(_checkpoint_order(second))
    for suffix in [s for s in _checkpoint_order(first) if s in shared]:
        names = [f"walk_s{s}_{suffix}" for s in seeds]
        points[suffix] = {}
        for task in tasks:
            reads = {r: _between_arms((first, names), (second, names), task, f"{task}|{r}", config) for r in rungs}
            points[suffix][task] = {r: c for r, c in reads.items() if c is not None}
    primary = str(config.p125.primary_rung)
    end = {t: points["end"][t][primary] for t in config.gate_tasks if primary in points.get("end", {}).get(t, {})}
    return {"points": points, "tests": _holm_both(end, float(config.alpha)) if end else None}


def p125g(config: DictConfig, root: Path) -> dict[str, Any]:
    """P1.2.5's Part G: the protections ablation, and the fine-tune rungs through the drive.

    Each unprotected arm (``<root>/ladder/<arm>``) is read as Part A–D read theirs; beside it, the recipe contrast
    (protected − unprotected) at every checkpoint; and the curve, every rung, of each re-read confirmation drive
    (``<root>/<p125g.ft_dir>/<arm>``).
    """
    g = config.p125g
    out: dict[str, Any] = {"arms": {}, "recipe": {}, "curve": {}}
    for arm in g.arms:
        ladder = Ladder(root / "ladder" / str(arm))
        out["arms"][str(arm)] = _p125_arm(config, ladder, str(arm))
        controls = (g.get("attribution") or {}).get(str(arm))
        if controls:
            out["arms"][str(arm)]["attribution"] = _p125_attribution(config, ladder, dict(controls))
    for unprotected, protected in (g.get("pairs") or {}).items():
        out["recipe"][str(unprotected)] = {
            "protected": str(protected),
            **_recipe_contrast(
                config, Ladder(root / "ladder" / str(protected)), Ladder(root / "ladder" / str(unprotected))
            ),
        }
    for arm in g.ft_arms:
        ladder = Ladder(root / str(g.ft_dir) / str(arm))
        out["curve"][str(arm)] = _curve(config, ladder, [t for t in config.gate_tasks if t in ladder.labels])
    return out


def _slice_mask(ladder: Ladder, task: str, spec: str) -> np.ndarray:
    """Frames of ``task`` in slice ``spec``: ``tod:<time of day>``, ``regime:<time of day>·<weather>``, ``all``."""
    if spec == "all":
        return np.ones(ladder.pos[task].size, dtype=bool)
    kind, name = spec.split(":", 1)
    if kind == "tod":
        names = {v: int(k) for k, v in ladder.summary["stratum_names"].items()}
        return ladder.stratum[ladder.pos[task]] == names[name]
    if kind == "regime":
        names = {v: int(k) for k, v in ladder.summary["regime_names"].items()}
        return ladder.regime[ladder.pos[task]] == names[name]
    raise ValueError(f"unknown slice {spec!r}; expected 'all', 'tod:<name>' or 'regime:<name>'.")


def _slice_ok(labels: np.ndarray, keep: np.ndarray, all_classes: np.ndarray, min_per_class: int) -> bool:
    return all((labels[keep] == c).sum() >= min_per_class for c in all_classes)


def _partf_arm(config: DictConfig, ladder: Ladder, tasks: list[str]) -> dict[str, Any]:
    """Part F2 for one arm: Δ vs the floor per slice at every walk checkpoint, the IID end, and the window contrasts.

    A window contrast is a signed sum of within-slice changes between two walk checkpoints of the same seeds
    (:func:`cafl4ds.adaptation.bootstrap_combination`): e.g. tracking = the night slice at the end minus at step 846.
    """
    pf = config.p125f
    seeds, floor = range(int(config.n_seeds)), [str(pf.floor_name)]
    iid = [f"iid_s{s}_end" for s in range(int(config.n_iid_seeds)) if ladder.has(f"iid_s{s}_end")]
    suffixes = _checkpoint_order(ladder)
    nb, sd, mpc = int(config.n_boot), int(config.seed), int(pf.min_per_class)
    out: dict[str, Any] = {}
    for task in tasks:
        labels = ladder.labels[task]
        classes = np.unique(labels)
        masks = {s: _slice_mask(ladder, task, s) for s in ["all", *pf.slices]}
        usable = {s: m for s, m in masks.items() if _slice_ok(labels, m, classes, mpc)}
        entry: dict[str, Any] = {
            "n_frames": int(labels.size),
            "n_positive": int((labels == 1).sum()) if classes.size == 2 else None,
            "slice_sizes": {s: int(m.sum()) for s, m in masks.items()},
        }
        for rung in pf.rungs:
            key = f"{task}|{rung}"
            ref = ladder.rows(floor, key)
            if ref is None:
                continue
            walk = {sfx: ladder.rows([f"walk_s{s}_{sfx}" for s in seeds], key) for sfx in suffixes}
            curve = {
                s: {
                    sfx: adaptation.bootstrap_delta(w[:, m], ref[:, m], labels[m], n_boot=nb, seed=sd)
                    for sfx, w in walk.items()
                    if w is not None
                }
                for s, m in usable.items()
            }
            iid_rows = ladder.rows(iid, key) if iid else None
            end = walk.get("end")
            iid_read: dict[str, Any] = {}
            if iid_rows is not None and end is not None:
                for s, m in usable.items():
                    iid_read[s] = {
                        "iid_vs_floor": adaptation.bootstrap_delta(
                            iid_rows[:, m], ref[:, m], labels[m], n_boot=nb, seed=sd
                        ),
                        "walk_vs_iid": adaptation.bootstrap_contrast(
                            end[:, m], iid_rows[:, m], labels[m], n_boot=nb, seed=sd
                        ),
                    }
            windows: dict[str, Any] = {}
            for name, win in pf.windows.items():
                a, b = walk.get(str(win["from"])), walk.get(str(win["to"]))
                terms = [(1.0, s) for s in win["slices"]] + [(-1.0, s) for s in win.get("against", [])]
                if a is None or b is None or any(s not in usable for _, s in terms):
                    windows[name] = None
                    continue
                parts = [
                    (w, (b - a)[:, masks[s]], labels[masks[s]]) for w, s in terms
                ]  # each term: the same seeds' change, within its slice
                windows[name] = adaptation.bootstrap_combination(parts, n_boot=nb, seed=sd)
                windows[name]["terms"] = {
                    s: adaptation.bootstrap_combination(
                        [(1.0, (b - a)[:, masks[s]], labels[masks[s]])], n_boot=nb, seed=sd
                    )["delta"]
                    for _, s in terms
                }
            entry[rung] = {"curve": curve, "iid": iid_read, "windows": windows}
        out[task] = entry
    return out


def _train_frequency(config: DictConfig, tasks: list[str]) -> dict[str, float]:
    """Each task's positive share over BDD-train frames of the study's scenes (the pool the drive samples from)."""
    pf = config.p125f
    records = json.loads(Path(to_absolute_path(str(pf.train_labels))).read_text(encoding="utf-8"))
    scenes, frame = set(pf.scenes), float(pf.frame_area)
    boxes, categories = [], []
    for rec in records:
        attrs = rec.get("attributes") or {}
        if attrs.get("scene") not in scenes or "undefined" in (attrs.get("timeofday"), attrs.get("weather")):
            continue
        frame_boxes = [
            (str(b["category"]), (b["box2d"]["x2"] - b["box2d"]["x1"]) * (b["box2d"]["y2"] - b["box2d"]["y1"]) / frame)
            for b in rec.get("labels") or []
            if isinstance(b.get("box2d"), dict)
        ]
        boxes.append(frame_boxes)
        counts: dict[str, int] = {}
        for c, _ in frame_boxes:
            counts[c] = counts.get(c, 0) + 1
        categories.append(counts)
    n = len(boxes)
    zeros = torch.zeros(n, dtype=torch.long)  # a label-only pool: no images are read
    pool = adaptation.HeldOutPool(zeros, zeros, zeros, zeros, categories=categories, boxes=boxes)
    return {t: float((adaptation.task_labels(pool, t).numpy() == 1).mean()) for t in tasks} | {"n_frames": float(n)}


def p125f(config: DictConfig, root: Path) -> dict[str, Any]:
    """P1.2.5's Part F (exploratory, no pass bar): F2's regime windows on every arm, and F1's end-of-drive table.

    Reads ``<root>/<p125f.ladder_dir>/<arm>``. With ``p125f.train_labels`` set, also each task's frequency in
    BDD-train, for F1's gain-against-frequency read.
    """
    pf = config.p125f
    ladders = {str(a): Ladder(root / str(pf.ladder_dir) / str(a)) for a in pf.arms}
    first = next(iter(ladders.values()))
    for ladder in ladders.values():
        _check_aligned(first, ladder)
    tasks = [str(t) for t in pf.tasks] if pf.tasks else list(first.labels)
    readable = [t for t in tasks if (first.labels[t] == 1).sum() >= int(pf.min_positive)]
    report: dict[str, Any] = {
        "tasks": tasks,
        "read": readable,
        "skipped": {t: int((first.labels[t] == 1).sum()) for t in tasks if t not in readable},
        "arms": {a: _partf_arm(config, ladder, readable) for a, ladder in ladders.items()},
    }
    if pf.get("train_labels"):
        report["train_frequency"] = _train_frequency(config, readable)
    return report


def _walk_position(segments: list[dict[str, Any]], step: int) -> int:
    """The walk leg (era index) that delivered batch ``step - 1`` — the last one a checkpoint before ``step`` saw."""
    seen = [seg["era_index"] for seg in segments if seg["start_step"] <= max(step - 1, 0)]
    return int(max(seen)) if seen else 0


def _p121_secondary(config: DictConfig, root: Path, bb: str, ladder: Ladder) -> dict[str, Any]:
    """P1.2.1's pre-registered secondary reads (not gating).

    * ``iid_vs_b5`` / ``walk_vs_iid`` — the IID control's end against B5, and the walk's end against it
      (5 vs ``n_iid_seeds`` seeds, descriptive).
    * ``by_stratum`` — the walk's end against B5 within each time-of-day stratum.
    * ``by_regime`` — at every saved walk checkpoint, Δ against B5 on the query frames whose regime the
      walk has already **left**, is **in**, or has **not reached**: does it adapt to what it sees, forget
      what it left, transfer ahead?
    """
    seeds = range(int(config.n_seeds))
    walk_end = [f"walk_s{s}_end" for s in seeds]
    iid_end = [f"iid_s{s}_end" for s in range(int(config.get("n_iid_seeds", 0)))]
    b5 = [str(config.b5_name)]
    segments_file = root / "drive" / f"{bb}_walk" / "corpus" / "segments.json"
    segments = json.loads(segments_file.read_text())["segments"] if segments_file.is_file() else []
    era_of_name = {seg["era_name"]: seg["era_index"] for seg in segments}
    regime_names = {int(k): v for k, v in ladder.summary.get("regime_names", {}).items()}
    total = max((seg["end_step"] for seg in segments), default=0) + 1
    out: dict[str, Any] = {"iid_vs_b5": {}, "walk_vs_iid": {}, "by_stratum": {}, "by_regime": {}}
    for task in config.tasks:
        labels = ladder.labels[task]
        era = np.array([era_of_name.get(regime_names.get(int(r), ""), -1) for r in ladder.regime[ladder.pos[task]]])
        for rung in config.rungs:
            key = f"{task}|{rung}"
            walk, iid, ref = ladder.rows(walk_end, key), ladder.rows(iid_end, key), ladder.rows(b5, key)
            if walk is None or ref is None:
                continue
            if iid_end and iid is not None:
                out["iid_vs_b5"][key] = _contrast(iid, ref, labels, config)
                out["walk_vs_iid"][key] = _contrast(walk, iid, labels, config)
            out["by_stratum"][key] = _strata(walk, ref, ladder, task, config)
        for rung in ("knn", "linear") if segments else ():  # every checkpoint carries the frozen rungs
            key = f"{task}|{rung}"
            ref = ladder.rows(b5, key)
            rows: dict[str, Any] = {}
            for path in sorted(ladder.path.glob("walk_s0_*.npz")):
                suffix = path.stem.split("_", 2)[2]
                step = total if suffix == "end" else int(suffix[1:])
                here = _walk_position(segments, step)
                adapted = ladder.rows([f"walk_s{s}_{suffix}" for s in seeds], key)
                if adapted is None or ref is None:
                    continue
                groups = {"left": (era >= 0) & (era < here), "in": era == here, "ahead": era > here}
                rows[suffix] = {"step": step, "era": here}
                for name, keep in groups.items():
                    enough = keep.sum() >= 30 and np.unique(labels[keep]).size == np.unique(labels).size
                    rows[suffix][name] = (
                        _contrast(adapted[:, keep], ref[:, keep], labels[keep], config) if enough else None
                    )
            out["by_regime"][key] = rows
    return out


def _print_p120(report: dict[str, Any]) -> None:
    for pool, by_bb in report.items():
        for bb, rows in by_bb.items():
            print(f"\n== {pool} · {bb} ==")
            for key, entry in rows.items():
                b = entry["bacc"]
                print(
                    f"{key:28s} B5 {b['b5']:.3f} | "
                    + " ".join(f"{g} {v:.3f}" for g, v in b.items() if g != "b5" and v is not None)
                )
                for frame in ("vs_b5", "vs_well"):
                    for g, c in entry.get(frame, {}).items():
                        if c is not None:
                            ci = f"[{c['lo']:+.3f}, {c['hi']:+.3f}]"
                            print(f"    {g:9s} {frame:7s} Δ {c['delta']:+.3f} {ci} p {c['p_one_sided']:.3f}")


def _print_report(mode: str, report: dict[str, Any]) -> None:
    """Print the mode's summary lines."""
    if mode == "p120":
        _print_p120(report)
    elif mode == "p124":
        print(json.dumps(report, indent=2))
    elif mode == "p125f":
        print(f"read {len(report['read'])} tasks; skipped (positives < min): {report['skipped']}")
    elif mode == "p125g":
        for arm, entry in report["arms"].items():
            print(f"{arm}: {entry['gate']['verdict']} (breadth {entry['breadth']['gains']}/{entry['breadth']['of']})")
        for arm, entry in report["recipe"].items():
            print(f"{entry['protected']} − {arm}, end: {entry['tests']}")
    elif mode == "p125":
        for arm, entry in report["arms"].items():
            print(f"{arm}: {entry['gate']['verdict']} (breadth {entry['breadth']['gains']}/{entry['breadth']['of']})")
        print(f"carry forward: {report['carry_forward']['vehicle']}; claim: {report['claim']}")
    else:
        for bb, entry in report.items():
            print(f"{bb}: {entry['gate']}")


def run(config: DictConfig) -> dict[str, Any]:
    """Run the configured analysis and write the report (Hydra-free core)."""
    root = Path(to_absolute_path(str(config.root)))
    modes = {"p120": p120, "p121": p121, "p124": p124, "p125": p125, "p125f": p125f, "p125g": p125g}
    if str(config.mode) not in modes:
        raise ValueError(f"mode must be one of {sorted(modes)}; got {config.mode!r}.")
    with threadpool_limits(int(config.threads)):
        report = modes[str(config.mode)](config, root)
    out = root / str(config.get("report") or "adaptation_report.json")
    out.write_text(
        json.dumps({"mode": str(config.mode), "n_boot": int(config.n_boot), **report}, indent=2), encoding="utf-8"
    )
    _print_report(str(config.mode), report)
    logger.info(f"wrote {out}")
    return report


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="analyze_adaptation")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run the P1.2 adapted-vs-frozen analysis."""
    run(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
