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
* ``selection`` — the stream's train-side selection set (``stream.select_*_per_canary``), never trained
  on: where P1.2.2 / P1.2.4 / P1.2.5 choose their recipes without reading val. Rebuilt from the source, so it
  carries every label the source does (P1.2.5 reads its whole task panel there).

``readout`` chooses what each encoder is read on (its own ``embed``, the patch-token ``mean`` or the ``cls``
token), and ``cls_rungs`` adds frozen rungs on the ``cls`` token beside it (P1.2.5). The per-patch task
(``drivable``) is read on patch tokens (``patch.*``).

Modes: ``read`` (the ladder), ``select_lr`` (the few-shot learning-rate pick, on the first encoder —
the frozen start — against a validation reservation from the fit pool) and ``screen`` (the task screen:
headroom and overlap with the scene, and the pre-registered choice or panel).

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


def _attributed_pool(data: AttributedImages, index: torch.Tensor | None = None) -> HeldOutPool:
    """A :class:`HeldOutPool` over ``data`` (or its ``index`` frames, in that order) with every label it carries."""
    pick = index if index is not None else torch.arange(data.canary.shape[0])
    rows = pick.tolist()
    return HeldOutPool(
        images=data.images[pick],
        scene=data.canary[pick],
        stratum=data.stratum[pick] if data.stratum is not None else torch.zeros_like(data.canary[pick]),
        regime=data.era_key[pick],
        categories=[data.object_categories[i] for i in rows] if data.object_categories is not None else None,
        stratum_names=dict(data.stratum_names),
        regime_names=dict(data.regime_names),
        boxes=[data.object_boxes[i] for i in rows] if data.object_boxes is not None else None,
        drivable=data.drivable[pick] if data.drivable is not None else None,
    )


def build_pool(config: DictConfig) -> tuple[HeldOutPool, np.ndarray, np.ndarray]:
    """The held-out pool and its disjoint ``(fit, query)`` indices (see the module docstring)."""
    kind = str(config.pool.kind)
    if kind == "heldout":
        overrides = {k: config.pool[k] for k in ("split", "max_images") if config.pool.get(k) is not None}
        pool = _attributed_pool(instantiate(config.data, **overrides).load())
        fit, query = adaptation.split_pool(len(pool), float(config.pool.fit_fraction), int(config.pool.seed))
        return pool, fit, query
    if kind == "selection":  # rebuilt from the source, so every task it labels can be read (P1.2.5)
        source = instantiate(config.data)
        indices = instantiate(config.stream, source=source).selection_indices
        if indices is None:
            raise ValueError("pool.kind=selection needs the drive's stream.select_*_per_canary reservation.")
        support, probe = indices
        pool = _attributed_pool(source.load(), torch.cat([support, probe]))
        return pool, np.arange(support.numel()), np.arange(support.numel(), len(pool))
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
    raise ValueError(f"pool.kind must be 'heldout', 'canary' or 'selection'; got {kind!r}.")


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
    configured SSL family, whose encoder is returned; ``interp`` the weight interpolation
    ``(1 - alpha) * a + alpha * b`` of two encoder specs (P1.2.4's WiSE-FT arm; non-float buffers from ``b``).
    """
    kind = str(spec["kind"])
    if kind == "interp":
        alpha = float(spec["alpha"])
        start, end = (load_encoder(config, dict(spec[k])).state_dict() for k in ("a", "b"))
        mixed = {k: torch.lerp(start[k], v, alpha) if v.is_floating_point() else v for k, v in end.items()}
        encoder = instantiate(config.encoder)
        encoder.load_state_dict(mixed)
        return encoder
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
    raise ValueError(f"encoder kind must be pretrained | random | well | checkpoint | interp; got {kind!r}.")


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


def _summarize(
    correct: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray,
    names: dict[int, str],
    counts: np.ndarray | None = None,
) -> dict[str, float]:
    """Pooled and per-stratum balanced accuracy of per-image correctness (draws averaged).

    With ``counts`` (the per-patch task) ``correct`` holds correct patches per class per frame, and the
    accuracy is patch-level.
    """
    mean = correct.mean(axis=0)
    if counts is not None:
        out = {"pooled": adaptation.patch_balanced_accuracy(mean, counts)}
        for sid in np.unique(strata):
            keep = strata == sid
            out[names.get(int(sid), str(sid))] = adaptation.patch_balanced_accuracy(mean[keep], counts[keep])
        return out
    out = {"pooled": adaptation.balanced_accuracy(mean, labels)}
    for sid in np.unique(strata):
        keep = strata == sid
        if np.unique(labels[keep]).size == np.unique(labels).size:
            out[names.get(int(sid), str(sid))] = adaptation.balanced_accuracy(mean[keep], labels[keep])
    return out


def _readout_encoder(config: DictConfig, encoder: torch.nn.Module) -> torch.nn.Module:
    """``encoder`` as read on ``config.readout``: its own ``embed``, or wrapped to read the ``mean`` / ``cls``."""
    readout = str(config.readout)
    return encoder if readout == "embed" else adaptation.Readout(encoder, readout)


def _frozen_features(config: DictConfig, encoder: torch.nn.Module, images: torch.Tensor) -> dict[str, np.ndarray]:
    """The frozen rungs' features: the configured readout (key ``""``), plus the ``cls`` token (``@cls``) if asked."""
    device, readout = str(config.device), str(config.readout)
    if readout == "embed" and not config.cls_rungs:
        return {"": adaptation.encode(encoder, images, device)}
    pooled = adaptation.encode_readouts(encoder, images, device)
    features = {"": adaptation.encode(encoder, images, device) if readout == "embed" else pooled[readout]}
    if config.cls_rungs:
        features["@cls"] = pooled["cls"]
    return features


def _patch_frames(pool: HeldOutPool, fit: np.ndarray, query: np.ndarray, config: DictConfig) -> tuple[np.ndarray, ...]:
    """The per-patch task's frames: seed-fixed subsets of the fit and query frames that have a drivable map."""
    labelled = adaptation.task_labels(pool, adaptation.DRIVABLE).numpy() >= 0
    rng = np.random.default_rng(int(config.pool.seed))
    picks = []
    for frames, size in ((fit, int(config.patch.fit_frames)), (query, int(config.patch.query_frames))):
        usable = frames[labelled[frames]]
        picks.append(np.sort(rng.permutation(usable)[:size]))
    return picks[0], picks[1]


def _patch_setup(pool: HeldOutPool, fit: np.ndarray, query: np.ndarray, config: DictConfig) -> dict[str, Any]:
    """The per-patch task's frames (pool indices), their patch labels, fit / query positions and query patch counts."""
    p_fit, p_query = _patch_frames(pool, fit, query, config)
    frames = np.concatenate([p_fit, p_query])
    labels = adaptation.patch_labels(pool).numpy()[frames]
    at_query = np.arange(p_fit.size, frames.size)
    return {
        "frames": frames,
        "query_frames": p_query,
        "labels": labels,
        "fit": np.arange(p_fit.size),
        "query": at_query,
        "counts": adaptation.patch_counts(labels[at_query]),
    }


def _read_patches(
    config: DictConfig, encoder: torch.nn.Module, pool: HeldOutPool, patch: dict[str, Any]
) -> dict[str, np.ndarray]:
    """The frozen per-patch rungs on one encoder (correct patches per class per query frame)."""
    tokens = adaptation.encode_patches(encoder, pool.images[patch["frames"]], str(config.device))
    return adaptation.patch_correctness(
        tokens,
        patch["labels"],
        patch["fit"],
        patch["query"],
        int(config.knn_k),
        int(config.patch.per_class),
        int(config.pool.seed),
        rungs=("knn", "linear") if config.patch.knn else ("linear",),
    )


def read(config: DictConfig, out_dir: Path) -> dict[str, Any]:
    """Score every encoder on the ladder; write per-image correctness and a summary.

    Each task reads the frames it uses (:func:`cafl4ds.adaptation.task_indices`): all of them for the scene
    and full-frame presence, a labelled or cell-balanced subset otherwise. ``pool.npz`` records each task's
    positions within the shared query (``pos|<task>``), so the analysis aligns correctness to its frames.

    The per-patch task (``drivable``) reads patch tokens on seed-fixed subsets of the fit and query frames
    (``patch.*``). Its arrays hold correct patches per class per query frame, and ``pool.npz`` holds the
    patch counts beside them (``counts|drivable``), from which the analysis scores patch-level balanced accuracy.
    """
    pool, fit, query = build_pool(config)
    seed = int(config.pool.seed)
    tasks = [t for t in config.tasks if not adaptation.is_patch_task(t)]
    patch_tasks = [t for t in config.tasks if adaptation.is_patch_task(t)]
    labels = {t: adaptation.task_labels(pool, t).numpy() for t in [*tasks, *patch_tasks]}
    t_fit = {t: adaptation.task_indices(pool, t, fit, seed) for t in tasks}
    t_query = {t: adaptation.task_indices(pool, t, query, seed) for t in tasks}
    patch = _patch_setup(pool, fit, query, config) if patch_tasks else {}
    t_query |= dict.fromkeys(patch_tasks, patch.get("query_frames"))
    pos = {t: np.searchsorted(query, t_query[t]) for t in t_query}
    strata = pool.stratum.numpy()[query]
    np.savez_compressed(
        out_dir / "pool.npz",
        query_index=query,
        stratum=strata,
        regime=pool.regime.numpy()[query],
        **{f"labels|{t}": labels[t][query] for t in labels},
        **{f"pos|{t}": pos[t] for t in pos},
        **{f"counts|{t}": patch["counts"] for t in patch_tasks},
    )
    summary: dict[str, Any] = {
        "pool": {"kind": str(config.pool.kind), "n_fit": int(fit.size), "n_query": int(query.size)},
        "readout": str(config.readout),
        "n_query_by_task": {t: int(t_query[t].size) for t in t_query},
        "stratum_names": pool.stratum_names,
        "regime_names": pool.regime_names,
        "encoders": {},
    }
    recipe = _recipe(config)
    ft_tasks = config.fewshot.get("tasks")
    for spec in _encoder_specs(config):
        name = spec["name"]
        encoder = load_encoder(config, spec).to(config.device)
        read_as = _readout_encoder(config, encoder)
        features = _frozen_features(config, encoder, pool.images)
        if config.get("blocks"):  # P1.2.2's per-block probes: frozen rungs on each listed block too
            blocks = adaptation.encode_blocks(read_as, pool.images, str(config.device), list(config.blocks))
            features |= {f"@block{b:02d}": f for b, f in blocks.items()}
        arrays: dict[str, np.ndarray] = {}
        for task in tasks:
            sub = _fit_subsample(labels[task], t_fit[task], config)
            for at, feats in features.items():
                for rung, correct in adaptation.frozen_correctness(
                    feats, labels[task], sub, t_query[task], int(config.knn_k)
                ).items():
                    arrays[f"{task}|{rung}{at}"] = correct[None, :]
            if not spec.get("fewshot", bool(config.fewshot.enabled)) or (ft_tasks is not None and task not in ft_tasks):
                continue
            for k in config.fewshot.ks:
                rows = []
                for draw in range(int(config.fewshot.draws)):
                    train = adaptation.balanced_indices(labels[task], sub, int(k), seed=draw)
                    pred = adaptation.fine_tune_predict(
                        read_as,
                        pool.images[train],
                        torch.from_numpy(labels[task][train]),
                        pool.images[t_query[task]],
                        recipe,
                        seed=draw,
                        device=str(config.device),
                    )
                    rows.append((pred == labels[task][t_query[task]]).astype(np.float32))
                arrays[f"{task}|ft{k}"] = np.stack(rows)
        if patch_tasks:
            rungs = _read_patches(config, encoder, pool, patch)
            arrays |= {f"{t}|{rung}": counts[None] for t in patch_tasks for rung, counts in rungs.items()}
        np.savez_compressed(out_dir / f"{name.replace('/', '__')}.npz", **arrays)
        summary["encoders"][name] = {
            "spec": spec,
            "bacc": {
                key: _summarize(
                    arr,
                    labels[key.split("|")[0]][t_query[key.split("|")[0]]],
                    strata[pos[key.split("|")[0]]],
                    pool.stratum_names,
                    patch.get("counts") if adaptation.is_patch_task(key.split("|")[0]) else None,
                )
                for key, arr in arrays.items()
            },
        }
        logger.info(f"{name}: { {k: round(v['pooled'], 4) for k, v in summary['encoders'][name]['bacc'].items()} }")
        del encoder, read_as
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
    encoder = _readout_encoder(config, load_encoder(config, spec).to(config.device))
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
    report = {
        "encoder": spec,
        "readout": str(config.readout),
        "task": task,
        "k": k,
        "n_val": int(val.size),
        "scores": scores,
        "best_lr": float(best),
    }
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


def _screen_patch_task(
    config: DictConfig, encoder: torch.nn.Module, pool: HeldOutPool, fit: np.ndarray, query: np.ndarray
) -> dict[str, Any]:
    """The screen's row for the per-patch task: frozen patch probes, the context overlap and a fine-tuned ceiling.

    Read at patch level throughout. The overlap fits scene + time of day (one-hot, repeated on every patch of a
    frame) to the patch labels; the ceiling fine-tunes the encoder with a per-patch head on
    ``2 * screen.ceiling_per_class`` fit frames. There is no few-shot read (its *k* counts frames per class).
    """
    sc, seed = config.screen, int(config.pool.seed)
    p_fit, p_query = _patch_frames(pool, fit, query, config)
    frames = np.concatenate([p_fit, p_query])
    labels = adaptation.patch_labels(pool).numpy()
    at_fit, at_query = np.arange(p_fit.size), np.arange(p_fit.size, frames.size)
    counts = adaptation.patch_counts(labels[p_query])
    tokens = adaptation.encode_patches(encoder, pool.images[frames], str(config.device))
    frozen = adaptation.patch_correctness(
        tokens, labels[frames], at_fit, at_query, int(config.knn_k), int(config.patch.per_class), seed
    )
    context = torch.stack([pool.scene, pool.stratum], dim=1).numpy()[frames]
    onehot = np.concatenate(
        [(context[:, [k]] == np.unique(context[:, k])).astype(np.float32) for k in range(2)], axis=1
    )
    overlap = adaptation.patch_correctness(
        np.repeat(onehot[:, None, :], labels.shape[1], axis=1),
        labels[frames],
        at_fit,
        at_query,
        1,
        int(config.patch.per_class),
        seed,
        rungs=("linear",),
    )["linear"]
    usable = fit[adaptation.task_labels(pool, adaptation.DRIVABLE).numpy()[fit] >= 0]
    train = np.sort(np.random.default_rng(seed).permutation(usable)[: 2 * int(sc.ceiling_per_class)])
    recipe = FineTuneRecipe(**{**_recipe(config).__dict__, "steps": int(sc.ceiling_steps)})
    pred = adaptation.fine_tune_patch_predict(
        encoder,
        pool.images[train],
        torch.from_numpy(labels[train]),
        pool.images[p_query],
        recipe,
        seed=0,
        device=str(config.device),
    )
    truth = labels[p_query]
    correct = np.stack([((pred == truth) & (truth == c)).sum(axis=1) for c in (0, 1)], axis=1).astype(np.float32)
    row: dict[str, Any] = {
        "n_fit": int(p_fit.size),
        "n_query": int(p_query.size),
        "class_share": {str(c): float(counts[:, c].sum() / counts.sum()) for c in (0, 1)},
        "overlap": adaptation.patch_balanced_accuracy(overlap, counts),
        "b5_knn": adaptation.patch_balanced_accuracy(frozen["knn"], counts),
        "b5_linear": adaptation.patch_balanced_accuracy(frozen["linear"], counts),
        f"b5_ft{int(config.fewshot.select_k)}": None,
        "ceiling": adaptation.patch_balanced_accuracy(correct, counts),
    }
    row["headroom"] = row["ceiling"] - row["b5_linear"]
    return row


def screen(config: DictConfig, out_dir: Path) -> dict[str, Any]:
    """The readout-task screen (P1.2.0 part 4; P1.2.5 Part C) on the first encoder (the frozen start).

    Per task — the ``screen.reference`` (the scene canary) and each ``screen.candidates`` — on the frames
    the task uses (:func:`cafl4ds.adaptation.task_indices`): the encoder's frozen kNN / linear and few-shot
    (``fewshot.select_k`` per class, ``fewshot.draws`` draws) reads, a full-label fine-tune ceiling, the
    overlap with scene + time of day, and the headroom (ceiling − frozen linear). The per-patch task is read
    at patch level (:func:`_screen_patch_task`). The rule keeps the candidates with overlap ≤ ``max_overlap``
    and headroom ≥ ``min_headroom``, ranks them by headroom, and picks up to ``max_tasks`` (P1.2.0 picked one;
    P1.2.5's panel takes up to four). The ``b5_*`` keys name the first encoder's reads, whatever it is.
    """
    pool, fit, query = build_pool(config)
    spec = _encoder_specs(config)[0]
    encoder = load_encoder(config, spec).to(config.device)
    read_as = _readout_encoder(config, encoder)
    features = _frozen_features(config, encoder, pool.images)[""]
    context = torch.stack([pool.scene, pool.stratum], dim=1).numpy()
    sc = config.screen
    seed = int(config.pool.seed)
    ceiling_recipe = FineTuneRecipe(**{**_recipe(config).__dict__, "steps": int(sc.ceiling_steps)})
    rows: dict[str, dict[str, Any]] = {}
    for task in [str(sc.reference), *[str(c) for c in sc.candidates]]:
        if adaptation.is_patch_task(task):
            rows[task] = _screen_patch_task(config, encoder, pool, fit, query)
            logger.info(f"{task}: {rows[task]}")
            continue
        labels = adaptation.task_labels(pool, task).numpy()
        t_fit, t_query = (adaptation.task_indices(pool, task, idx, seed) for idx in (fit, query))
        sub = _fit_subsample(labels, t_fit, config)
        truth = labels[t_query]
        frozen = adaptation.frozen_correctness(features, labels, sub, t_query, int(config.knn_k))

        k = int(config.fewshot.select_k)
        fewshot = [
            _tuned_bacc(
                config,
                read_as,
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
        ceiling = _tuned_bacc(config, read_as, pool, labels, ceiling_train, t_query, ceiling_recipe, 0)
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
    ranked = sorted(eligible, key=lambda c: rows[c]["headroom"], reverse=True)
    panel = ranked[: int(sc.max_tasks)] if sc.get("max_tasks") else ranked
    report = {
        "encoder": spec,
        "readout": str(config.readout),
        "rule": OmegaConf.to_container(sc),
        "tasks": rows,
        "choice": panel[0] if panel else None,
        "panel": panel,
    }
    (out_dir / "screen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"screen choice: {report['choice']}; panel: {panel}")
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
