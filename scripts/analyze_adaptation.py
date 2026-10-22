"""P1.2 analysis — adapted-vs-frozen (B5) contrasts from the readout ladder's per-image correctness.

Reads the directories ``scripts/read_adaptation.py`` wrote (``pool.npz`` + one ``<encoder>.npz`` each)
and bootstraps every contrast over seeds and query images (:func:`cafl4ds.adaptation.bootstrap_delta`).
Few-shot rows are averaged over their paired label draws first.

* ``mode=p120`` — P1.2.0's descriptive contrasts on P1.1's artifacts, per pool (``heldout`` = the
  BDD-train pool, ``canary`` = P1.1.0's val probes) and backbone: the pretraining framing (well − B5,
  drive − B5), the deployment framing (drive end − its well), and random − B5. Not a gate.
* ``mode=p121`` — P1.2.1's pre-registered gate (:func:`cafl4ds.adaptation.gate_verdict`) per backbone
  on the end of the walk, plus Δ at every saved checkpoint.

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


def run(config: DictConfig) -> dict[str, Any]:
    """Run the configured analysis and write the report (Hydra-free core)."""
    root = Path(to_absolute_path(str(config.root)))
    modes = {"p120": p120, "p121": p121}
    if str(config.mode) not in modes:
        raise ValueError(f"mode must be one of {sorted(modes)}; got {config.mode!r}.")
    with threadpool_limits(int(config.threads)):
        report = modes[str(config.mode)](config, root)
    out = root / "adaptation_report.json"
    out.write_text(
        json.dumps({"mode": str(config.mode), "n_boot": int(config.n_boot), **report}, indent=2), encoding="utf-8"
    )
    if config.mode == "p120":
        _print_p120(report)
    else:
        for bb, entry in report.items():
            print(f"{bb}: {entry['gate']}")
    logger.info(f"wrote {out}")
    return report


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="analyze_adaptation")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run the P1.2 adapted-vs-frozen analysis."""
    run(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
