r"""P1.2 readout ladder — score encoders on held-out frames for the adapted-vs-frozen (B5) contrast.

Reads any list of encoders — the off-the-shelf weights (``pretrained``, B5), a random init
(``random``), a saved well (``well``) or a drive checkpoint (``checkpoint``) — on one held-out pool,
on the plan's ascending-label ladder: frozen kNN and linear probes, and a few-shot fine-tune at each
``fewshot.ks`` labels per class (``fewshot.draws`` label draws, the same images for every encoder).
Every reading is saved as **per-image correctness** (``<out>/<encoder>.npz``) beside the shared query
metadata (``<out>/pool.npz``) and a balanced-accuracy summary (``<out>/summary.json``), so
``cafl4ds.adaptation.bootstrap_delta`` can resample seeds and images.

The pool (``pool.kind``):

* ``heldout`` — frames from a BDD split none of the encoders trained on (``pool.split``; the first
  ``pool.max_images`` attribute-valid frames of the canary scenes), split once by ``pool.seed`` into a
  fit pool and a query set. P1.2.0 reads BDD *train* for the val-trained P1.1 models; P1.2.1 reads
  BDD *val* for its train-driven models.
* ``canary`` — the regime stream's reserved canary probes (support = fit, query = query), for
  continuity with P1.1's in-run canary. Pass the drive's diet overrides so the reservation matches.

Modes: ``read`` (the ladder), ``select_lr`` (the few-shot learning-rate pick, on the first encoder —
B5 — against a validation reservation from the fit pool) and ``screen`` (the presence-task screen:
headroom and overlap with the scene, and the pre-registered choice).

Example::

    uv run python scripts/read_adaptation.py data=bdd bdd_root=/abs/path encoder=vit_b16_mae \\
        img_size=224 device=hpu backbone=mae encoders_file=encoders.yaml
"""

import json
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds import adaptation, warmup
from cafl4ds.adaptation import FineTuneRecipe, HeldOutPool
from cafl4ds.data.attributes import AttributedImages
from cafl4ds.models.vit import TinyViTEncoder

logger.remove()
logger.add(sys.stdout, level="INFO")


def build_pool(config: DictConfig) -> tuple[HeldOutPool, np.ndarray, np.ndarray]:
    """The held-out pool and its disjoint ``(fit, query)`` indices (see the module docstring)."""
    kind = str(config.pool.kind)
    if kind == "heldout":
        overrides = {k: config.pool[k] for k in ("split", "max_images") if config.pool.get(k) is not None}
        data: AttributedImages = instantiate(config.data, **overrides).load()
        pool = HeldOutPool(
            images=data.images,
            scene=data.canary,
            stratum=data.stratum if data.stratum is not None else torch.zeros_like(data.canary),
            regime=data.era_key,
            categories=data.object_categories,
            stratum_names=dict(data.stratum_names),
            regime_names=dict(data.regime_names),
        )
        fit, query = adaptation.split_pool(len(pool), float(config.pool.fit_fraction), int(config.pool.seed))
        return pool, fit, query
    if kind == "canary":
        sets = instantiate(config.stream).eval_sets
        support, probe = sets.probe_support, sets.probe_query
        groups = [s.groups if s.groups is not None else torch.zeros_like(s.labels) for s in (support, probe)]
        pool = HeldOutPool(
            images=torch.cat([support.images, probe.images]),
            scene=torch.cat([support.labels, probe.labels]),
            stratum=torch.cat(groups),
            regime=torch.full((support.labels.numel() + probe.labels.numel(),), -1, dtype=torch.long),
            stratum_names=dict(sets.group_names),
        )
        n_fit = support.labels.numel()
        return pool, np.arange(n_fit), np.arange(n_fit, len(pool))
    raise ValueError(f"pool.kind must be 'heldout' or 'canary'; got {kind!r}.")


def _encoder_specs(config: DictConfig) -> list[dict[str, Any]]:
    """The encoders to read: ``encoders`` from the config, then any listed in ``encoders_file``."""
    listed = OmegaConf.create(list(config.get("encoders") or []))
    if config.get("encoders_file"):
        listed = OmegaConf.create([*listed, *OmegaConf.load(to_absolute_path(str(config.encoders_file)))])
    specs: list[dict[str, Any]] = [dict(e) for e in OmegaConf.to_container(listed, resolve=True)]
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"encoder names must be unique; got {names}.")
    return specs


def load_encoder(config: DictConfig, spec: dict[str, Any]) -> torch.nn.Module:
    """Build one encoder from its spec.

    ``pretrained`` is the encoder as configured; ``random`` the same architecture at the from-scratch
    init (the input resize / normalization kept); ``well`` / ``checkpoint`` a saved full method of the
    configured SSL family, whose encoder is returned.
    """
    kind = str(spec["kind"])
    encoder = instantiate(config.encoder)
    if kind == "pretrained":
        return encoder
    if kind == "random":
        fresh = TinyViTEncoder(
            img_size=int(config.img_size),
            patch_size=int(encoder.patch_size),
            embed_dim=int(encoder.embed_dim),
            depth=len(encoder.blocks),
            num_heads=int(config.encoder.num_heads),
            mlp_ratio=float(config.encoder.mlp_ratio),
        )
        missing, unexpected = encoder.load_state_dict(fresh.state_dict(), strict=False)
        if unexpected or any(not k.startswith("_") for k in missing):
            raise ValueError(f"random re-init mismatch: missing={missing} unexpected={unexpected}")
        return encoder
    if kind in ("well", "checkpoint"):
        method = instantiate(config.ssl, encoder=encoder)
        warmup.load_well(method, to_absolute_path(str(spec["path"])))
        return method.encoder
    raise ValueError(f"encoder kind must be pretrained | random | well | checkpoint; got {kind!r}.")


def _recipe(config: DictConfig, lr: float | None = None) -> FineTuneRecipe:
    fs = config.fewshot
    return FineTuneRecipe(
        lr=float(lr if lr is not None else fs.lr),
        head_lr_mult=float(fs.head_lr_mult),
        weight_decay=float(fs.weight_decay),
        steps=int(fs.steps),
        batch_size=int(fs.batch_size),
    )


def _fit_subsample(labels: np.ndarray, fit: np.ndarray, config: DictConfig) -> np.ndarray:
    """The task's probe-fit subsample (class-balanced, fixed by the pool seed — the same for every encoder)."""
    return adaptation.balanced_indices(labels, fit, int(config.probe_per_class), int(config.pool.seed))


def _summarize(correct: np.ndarray, labels: np.ndarray, strata: np.ndarray, names: dict[int, str]) -> dict[str, float]:
    """Pooled and per-stratum balanced accuracy of per-image correctness (draws averaged)."""
    mean = correct.mean(axis=0)
    out = {"pooled": adaptation.balanced_accuracy(mean, labels)}
    for sid in np.unique(strata):
        keep = strata == sid
        if np.unique(labels[keep]).size == np.unique(labels).size:
            out[names.get(int(sid), str(sid))] = adaptation.balanced_accuracy(mean[keep], labels[keep])
    return out


def read(config: DictConfig, out_dir: Path) -> dict[str, Any]:
    """Score every encoder on the ladder; write per-image correctness and a summary.

    Each task reads the frames it uses (:func:`cafl4ds.adaptation.task_indices`): all of them for the scene
    and full-frame presence, a labelled or cell-balanced subset otherwise. ``pool.npz`` records each task's
    positions within the shared query (``pos|<task>``), so the analysis aligns correctness to its frames.
    """
    pool, fit, query = build_pool(config)
    seed = int(config.pool.seed)
    tasks = list(config.tasks)
    labels = {t: adaptation.task_labels(pool, t).numpy() for t in tasks}
    t_fit = {t: adaptation.task_indices(pool, t, fit, seed) for t in tasks}
    t_query = {t: adaptation.task_indices(pool, t, query, seed) for t in tasks}
    pos = {t: np.searchsorted(query, t_query[t]) for t in tasks}
    strata = pool.stratum.numpy()[query]
    np.savez_compressed(
        out_dir / "pool.npz",
        query_index=query,
        stratum=strata,
        regime=pool.regime.numpy()[query],
        **{f"labels|{t}": labels[t][query] for t in tasks},
        **{f"pos|{t}": pos[t] for t in tasks},
    )
    summary: dict[str, Any] = {
        "pool": {"kind": str(config.pool.kind), "n_fit": int(fit.size), "n_query": int(query.size)},
        "n_query_by_task": {t: int(t_query[t].size) for t in tasks},
        "stratum_names": pool.stratum_names,
        "regime_names": pool.regime_names,
        "encoders": {},
    }
    recipe = _recipe(config)
    ft_tasks = config.fewshot.get("tasks")
    for spec in _encoder_specs(config):
        name = spec["name"]
        encoder = load_encoder(config, spec).to(config.device)
        features = adaptation.encode(encoder, pool.images, str(config.device))
        arrays: dict[str, np.ndarray] = {}
        for task in tasks:
            sub = _fit_subsample(labels[task], t_fit[task], config)
            for rung, correct in adaptation.frozen_correctness(
                features, labels[task], sub, t_query[task], int(config.knn_k)
            ).items():
                arrays[f"{task}|{rung}"] = correct[None, :]
            if not spec.get("fewshot", bool(config.fewshot.enabled)) or (ft_tasks is not None and task not in ft_tasks):
                continue
            for k in config.fewshot.ks:
                rows = []
                for draw in range(int(config.fewshot.draws)):
                    train = adaptation.balanced_indices(labels[task], sub, int(k), seed=draw)
                    pred = adaptation.fine_tune_predict(
                        encoder,
                        pool.images[train],
                        torch.from_numpy(labels[task][train]),
                        pool.images[t_query[task]],
                        recipe,
                        seed=draw,
                        device=str(config.device),
                    )
                    rows.append((pred == labels[task][t_query[task]]).astype(np.float32))
                arrays[f"{task}|ft{k}"] = np.stack(rows)
        np.savez_compressed(out_dir / f"{name.replace('/', '__')}.npz", **arrays)
        summary["encoders"][name] = {
            "spec": spec,
            "bacc": {
                key: _summarize(
                    arr,
                    labels[key.split("|")[0]][t_query[key.split("|")[0]]],
                    strata[pos[key.split("|")[0]]],
                    pool.stratum_names,
                )
                for key, arr in arrays.items()
            },
        }
        logger.info(f"{name}: { {k: round(v['pooled'], 4) for k, v in summary['encoders'][name]['bacc'].items()} }")
        del encoder
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def select_lr(config: DictConfig, out_dir: Path) -> dict[str, Any]:
    """Pick the few-shot learning rate on the first encoder (B5) against a validation reservation."""
    pool, fit, _ = build_pool(config)
    task = str(config.fewshot.select_task)
    labels = adaptation.task_labels(pool, task).numpy()
    sub = _fit_subsample(labels, fit, config)
    val = adaptation.balanced_indices(
        labels, np.setdiff1d(fit, sub), int(config.fewshot.val_per_class), int(config.pool.seed) + 1
    )
    spec = _encoder_specs(config)[0]
    encoder = load_encoder(config, spec).to(config.device)
    k = int(config.fewshot.select_k)
    scores: dict[str, list[float]] = {}
    for lr in config.fewshot.lr_grid:
        scores[str(lr)] = []
        for draw in range(int(config.fewshot.draws)):
            train = adaptation.balanced_indices(labels, sub, k, seed=draw)
            pred = adaptation.fine_tune_predict(
                encoder,
                pool.images[train],
                torch.from_numpy(labels[train]),
                pool.images[val],
                _recipe(config, float(lr)),
                seed=draw,
                device=str(config.device),
            )
            scores[str(lr)].append(adaptation.balanced_accuracy((pred == labels[val]).astype(float), labels[val]))
        logger.info(f"lr {lr}: val bacc {scores[str(lr)]}")
    best = max(scores, key=lambda lr: float(np.mean(scores[lr])))
    report = {"encoder": spec, "task": task, "k": k, "n_val": int(val.size), "scores": scores, "best_lr": float(best)}
    (out_dir / "lr_select.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"few-shot lr: {best}")
    return report


def _tuned_bacc(
    config: DictConfig,
    encoder: torch.nn.Module,
    pool: HeldOutPool,
    labels: np.ndarray,
    train: np.ndarray,
    query: np.ndarray,
    recipe: FineTuneRecipe,
    draw: int,
) -> float:
    """Balanced accuracy on ``query`` of ``encoder`` fine-tuned on ``train`` (a copy; the encoder is untouched)."""
    pred = adaptation.fine_tune_predict(
        encoder,
        pool.images[train],
        torch.from_numpy(labels[train]),
        pool.images[query],
        recipe,
        seed=draw,
        device=str(config.device),
    )
    truth = labels[query]
    return adaptation.balanced_accuracy((pred == truth).astype(float), truth)


def screen(config: DictConfig, out_dir: Path) -> dict[str, Any]:
    """The readout-task screen (P1.2.0 part 4) on the first encoder (B5).

    Per task — the ``screen.reference`` (the scene canary) and each ``screen.candidates`` — on the frames
    the task uses (:func:`cafl4ds.adaptation.task_indices`): B5's frozen kNN / linear and few-shot
    (``fewshot.select_k`` per class, ``fewshot.draws`` draws) reads, a full-label fine-tune ceiling, the
    overlap with scene + time of day, and the headroom (ceiling − B5 linear). The rule picks, among
    candidates with overlap ≤ ``max_overlap`` and headroom ≥ ``min_headroom``, the largest headroom.
    """
    pool, fit, query = build_pool(config)
    spec = _encoder_specs(config)[0]
    encoder = load_encoder(config, spec).to(config.device)
    features = adaptation.encode(encoder, pool.images, str(config.device))
    context = torch.stack([pool.scene, pool.stratum], dim=1).numpy()
    sc = config.screen
    seed = int(config.pool.seed)
    ceiling_recipe = FineTuneRecipe(**{**_recipe(config).__dict__, "steps": int(sc.ceiling_steps)})
    rows: dict[str, dict[str, Any]] = {}
    for task in [str(sc.reference), *[str(c) for c in sc.candidates]]:
        labels = adaptation.task_labels(pool, task).numpy()
        t_fit, t_query = (adaptation.task_indices(pool, task, idx, seed) for idx in (fit, query))
        sub = _fit_subsample(labels, t_fit, config)
        truth = labels[t_query]
        frozen = adaptation.frozen_correctness(features, labels, sub, t_query, int(config.knn_k))

        k = int(config.fewshot.select_k)
        fewshot = [
            _tuned_bacc(
                config,
                encoder,
                pool,
                labels,
                adaptation.balanced_indices(labels, sub, k, seed=d),
                t_query,
                _recipe(config),
                d,
            )
            for d in range(int(config.fewshot.draws))
        ]
        ceiling_train = adaptation.balanced_indices(labels, t_fit, int(sc.ceiling_per_class), seed)
        ceiling = _tuned_bacc(config, encoder, pool, labels, ceiling_train, t_query, ceiling_recipe, 0)
        row: dict[str, Any] = {
            "n_fit": int(t_fit.size),
            "n_query": int(t_query.size),
            "class_share": {str(c): float((truth == c).mean()) for c in np.unique(truth)},
            "overlap": adaptation.scene_overlap(context, labels, sub, t_query),
            "b5_knn": adaptation.balanced_accuracy(frozen["knn"], truth),
            "b5_linear": adaptation.balanced_accuracy(frozen["linear"], truth),
            f"b5_ft{k}": float(np.mean(fewshot)),
            "ceiling": ceiling,
        }
        row["headroom"] = row["ceiling"] - row["b5_linear"]
        rows[task] = row
        logger.info(f"{task}: {row}")
    eligible = [
        c
        for c in map(str, sc.candidates)
        if rows[c]["overlap"] <= float(sc.max_overlap) and rows[c]["headroom"] >= float(sc.min_headroom)
    ]
    choice = max(eligible, key=lambda c: rows[c]["headroom"]) if eligible else None
    report = {"encoder": spec, "rule": OmegaConf.to_container(sc), "tasks": rows, "choice": choice}
    (out_dir / "screen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"screen choice: {choice}")
    return report


def run(config: DictConfig, out_dir: Path) -> dict[str, Any]:
    """Dispatch on ``mode`` (Hydra-free core, unit-testable on the synthetic source)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config.seed))
    torch.set_num_threads(int(config.threads))
    modes = {"read": read, "select_lr": select_lr, "screen": screen}
    if str(config.mode) not in modes:
        raise ValueError(f"mode must be one of {sorted(modes)}; got {config.mode!r}.")
    with threadpool_limits(int(config.threads)):
        return modes[str(config.mode)](config, out_dir)


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="read_adaptation")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run the readout ladder in the configured mode."""
    run(config, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
