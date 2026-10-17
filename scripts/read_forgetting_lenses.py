"""P1.1.3 item 5 — does P0.3.9's natural MAE forgetting show anywhere CKA-like lenses can see it?

Phase 0 read CKA drift on the pooled output only, with the plain estimator, and found it faint on
P0.3.9's seed-stable forgetting. Here the re-run saved each arm's full method after phase A and after
phase B (``save_states``), so every lens (:mod:`cafl4ds.lenses`) is read at every block, on the
task-A support + query images, from post-A to post-B — PC (forgets) against healthy (replay), paired
by seed. Three steps, in order:

1. **Reproduce.** Each seed's BWT, both arms, must sit within ``tolerance`` of P0.3.9's promoted value,
   with the same gate verdict. Otherwise the states are not read.
2. **Read.** Rebuild the task-A probe from the run's config and check it against the fingerprint the
   run saved; read every state's per-block features (:func:`cafl4ds.block_features.block_features`);
   each arm's change is its post-B lens value against its own post-A anchor, minus the post-A value.
   Both arms' post-A states should be identical (a shared phase A); the report records how close.
3. **Test.** Per seed, PC minus healthy, harm-signed, for every tested lens at every layer. One exact
   sign-flip test of the maximum paired t over all of them (:func:`cafl4ds.cka_reads.sign_flip_max_t`)
   — the search over lenses and layers paid for inside the test.

Example::

    uv run python scripts/read_forgetting_lenses.py
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cafl4ds import measurements, warmup
from cafl4ds.block_features import block_features
from cafl4ds.cka_reads import sign_flip_max_t
from cafl4ds.jsonio import dumps_valid
from cafl4ds.lenses import lens_values
from cafl4ds.reading_protocol import direction, spearman

logger.remove()
logger.add(sys.stdout, level="INFO")


def _harness() -> ModuleType:
    """``scripts/positive_control_forgetting.py`` as a module (it is a script, not a package)."""
    path = Path(__file__).resolve().parent / "positive_control_forgetting.py"
    spec = importlib.util.spec_from_file_location("_forgetting_harness", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reproduction(new: dict[str, Any], promoted: dict[str, Any], tolerance: float) -> dict[str, Any]:
    """Each arm's BWT against the promoted run's, and whether the gate verdict matches."""
    out: dict[str, Any] = {}
    for arm in ("pc", "healthy"):
        key = f"{arm}_backward_transfer"
        out[key] = {"new": new["gate"][key], "promoted": promoted["gate"][key]}
        out[key]["ok"] = abs(out[key]["new"] - out[key]["promoted"]) <= tolerance
    out["gate_same"] = new["gate"]["passed"] == promoted["gate"]["passed"]
    out["ok"] = all(v["ok"] for k, v in out.items() if k.endswith("transfer")) and out["gate_same"]
    return out


def changes(post_b: dict[str, float], post_a: dict[str, float]) -> dict[str, float]:
    """Each lens's change over phase B: the post-B reading minus the post-A one (drift's post-A is ~0)."""
    return {sig: post_b[sig] - post_a[sig] for sig in post_b}


def paired_diffs(pc: list[dict[str, float]], healthy: list[dict[str, float]], readers: list[str]) -> np.ndarray:
    """``[seeds, readers]`` — PC's change minus healthy's, harm-signed (positive = PC moved the harmful way)."""
    return np.array([[(p[r] - h[r]) * (direction(r) or 1) for r in readers] for p, h in zip(pc, healthy, strict=True)])


def read_seed(config: DictConfig, harness: ModuleType, seed: int) -> dict[str, Any]:
    """Every lens's phase-B change per arm for one seed, plus the anchor check and the recorded-drift check."""
    run = Path(to_absolute_path(str(config.runs).format(seed=seed)))
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    split = harness._task_split(cfg)
    saved = json.loads((run / "states" / "task_a_probe.json").read_text(encoding="utf-8"))
    if harness.task_a_fingerprint(split) != saved:
        raise ValueError(f"{run}: the rebuilt task-A probe does not match the run's fingerprint.")
    a = split["A"]
    images = torch.cat([a["support"].images, a["query"].images])
    query = torch.arange(images.shape[0]) >= a["support"].images.shape[0]
    method = instantiate(cfg.ssl, encoder=instantiate(cfg.encoder)).to(config.device)
    comparison = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
    states = {
        arm: {ph: run / "states" / f"{name}_post_{ph}.pt" for ph in ("a", "b")} for arm, name in config.arms.items()
    }
    anchors = [torch.load(states[arm]["a"]) for arm in config.arms]
    out: dict[str, Any] = {
        "anchor_max_abs_diff": max(float((anchors[0][k] - anchors[1][k]).abs().max()) for k in anchors[0]),
        "pc_bwt": comparison["gate"]["pc_backward_transfer"],
    }
    for arm in config.arms:
        feats = {}
        for ph in ("a", "b"):
            warmup.load_well(method, states[arm][ph])
            feats[ph] = block_features(method, images, batch_size=int(config.batch_size))
        out[arm] = {
            "change": changes(lens_values(feats["b"], feats["a"]), lens_values(feats["a"], feats["a"])),
            # Phase 0's read: plain CKA drift on the pooled output, task-A query only — must equal the record.
            "phase0_cka_drift": measurements.cka_drift(feats["a"]["backbone"][query], feats["b"]["backbone"][query]),
            "recorded_cka_drift": comparison[arm]["cka_drift"],
        }
    return out


def analyze(config: DictConfig) -> dict[str, Any]:
    """Reproduce, read and test; write the report (Hydra-free core)."""
    torch.set_num_threads(int(config.threads))
    seeds = [int(s) for s in config.seeds]
    report: dict[str, Any] = {"reproduction": {}}
    for seed in seeds:
        new = json.loads((Path(to_absolute_path(str(config.runs).format(seed=seed))) / "comparison.json").read_text())
        promoted = json.loads(Path(to_absolute_path(str(config.promoted).format(seed=seed))).read_text())
        report["reproduction"][str(seed)] = reproduction(new, promoted, float(config.tolerance))
    out = Path(to_absolute_path(str(config.out)))
    out.parent.mkdir(parents=True, exist_ok=True)
    if not all(r["ok"] for r in report["reproduction"].values()):
        out.write_text(dumps_valid(report) + "\n", encoding="utf-8")
        raise SystemExit(f"the re-run does not reproduce P0.3.9's promoted BWT; states left unread ({out}).")
    harness = _harness()
    per_seed = {seed: read_seed(config, harness, seed) for seed in seeds}
    pc = [per_seed[s]["pc"]["change"] for s in seeds]
    healthy = [per_seed[s]["healthy"]["change"] for s in seeds]
    readers = [r for r in pc[0] if r.split("@")[0] in set(config.tested)]
    diffs = paired_diffs(pc, healthy, readers)
    test = sign_flip_max_t(diffs, readers)
    per_lens = {
        lens: sign_flip_max_t(
            diffs[:, [i for i, r in enumerate(readers) if r.startswith(f"{lens}@")]],
            [r for r in readers if r.startswith(f"{lens}@")],
        )
        for lens in config.tested
    }
    every = sorted(pc[0])
    all_diffs = paired_diffs(pc, healthy, every)
    bwt = np.array([per_seed[s]["pc_bwt"] for s in seeds])
    report.update(
        {
            "seeds": {str(s): dict(per_seed[s]) for s in seeds},
            "test": test,
            "per_lens": per_lens,
            "per_reader": {
                r: {
                    "diffs": all_diffs[:, i].tolist(),
                    "n_positive": int((all_diffs[:, i] > 0).sum()),
                    "pc_mean": float(np.mean([p[r] for p in pc])),
                    "healthy_mean": float(np.mean([h[r] for h in healthy])),
                    # Deeper craters (more negative BWT) with larger PC-minus-healthy change read negative.
                    "spearman_vs_pc_bwt": spearman(all_diffs[:, i], bwt),
                }
                for i, r in enumerate(every)
            },
        }
    )
    out.write_text(dumps_valid(report) + "\n", encoding="utf-8")
    logger.info(
        f"test: t_max {test['t_max']:.2f} at {test['reader']}, p = {test['p']:.4f} ({test['n_positive']}/5 seeds)"
    )
    for lens, t in per_lens.items():
        logger.info(f"  {lens:18s} t_max {t['t_max']:.2f} at {t['reader']}, p = {t['p']:.4f}")
    logger.info(f"wrote {out}")
    return report


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="forgetting_lenses")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Reproduce, read the saved states per block, and test."""
    analyze(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
