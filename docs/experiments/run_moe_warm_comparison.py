"""Reproducible WP-M0 native MAE/BDD warm-well experiment and integration check.

Run with the repository Python environment and explicit MPS access. This research runner
keeps observations outside the production training interfaces and persists every result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from cafl4ds import deploy, deploy_corpus, harness, measurements, warmup
from cafl4ds.data.attributes import AttributedImages, BDD100KSource, _count_categories, _rank_regimes
from cafl4ds.data.regime import RegimeStream
from cafl4ds.health_trust import CANARY_SIGNALS, DEFAULT_LABEL_FREE_SIGNALS, Backbone
from cafl4ds.models.vit import MoEMlp
from cafl4ds.monitor import HealthMonitor

ROOT = Path(__file__).resolve().parents[2]
ARMS = {
    "dense_active": ["encoder=tiny", "encoder.mlp_ratio=2.0"],
    "dense_total": ["encoder=tiny", "encoder.mlp_ratio=5.0"],
    "moe_native": ["encoder=moe_tiny"],
}
PROBE_STEPS = {0, 200, 500, 1000, 2000}


def write_json(path, obj):
    """Write strict JSON atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    """Write a rectangular table with optional values left empty."""
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def sha_bytes(value):
    """Hash a tensor/array/string reproducibly."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        value = value.tobytes()
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    """Hash files without reading them all into RAM."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def isolated_rng(seed=None):
    """Isolate CPU and MPS RNG; evaluation never consumes the training stream."""
    cpu = torch.get_rng_state()
    mps = torch.mps.get_rng_state()
    try:
        if seed is not None:
            torch.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(cpu)
        torch.mps.set_rng_state(mps)


class CachedSource:
    """Read the exact BDD float32 resize cache with the ordinary regime stream."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "records.json").read_text())
        self.array = np.load(self.directory / "images.npy", mmap_mode="r+")
        self._cache = AttributedImages(
            images=torch.from_numpy(self.array),
            era_key=torch.tensor(self.meta["era_key"]),
            canary=torch.tensor(self.meta["canary"]),
            regime_order=self.meta["regime_order"],
            regime_names={int(k): v for k, v in self.meta["regime_names"].items()},
            canary_names={int(k): v for k, v in self.meta["canary_names"].items()},
            object_categories=self.meta["object_categories"],
        )

    def load(self):
        """Expose cached attributed images."""
        return self._cache

    @property
    def num_canary_classes(self):
        """Number of retained scenes."""
        return len(self._cache.canary_names)


def prepare_data(directory, bdd_root):
    """Validate every label/image pair and cache the unchanged 64-pixel BDD diet."""
    directory = Path(directory)
    if (directory / "complete.json").exists():
        return CachedSource(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = BDD100KSource(str(bdd_root), split="train", img_size=64, legacy_labels=False, min_canary_count=100)
    image_dir, label_dir = source._paths()
    records, failures = [], Counter()
    label_hash = hashlib.sha256()
    for rec in source._records(label_dir):
        label_hash.update(json.dumps(rec, sort_keys=True).encode())
        attrs = rec["attributes"]
        values = [attrs.get(k) for k in ("timeofday", "weather", "scene")]
        if any(v in {None, "", "undefined"} for v in values):
            failures["unknown_attributes"] += 1
            continue
        path = image_dir / rec["name"]
        if not path.is_file():
            failures["missing_image"] += 1
            continue
        records.append((rec["name"], tuple(values[:2]), values[2], _count_categories(rec)))
    counts = Counter(r[2] for r in records)
    records = [r for r in records if counts[r[2]] >= 100]
    assert records and len({r[0] for r in records}) == len(records)
    regime_order, regime_ids, regime_names = _rank_regimes({r[1] for r in records})
    canary_names = dict(enumerate(sorted({r[2] for r in records})))
    canary_ids = {v: k for k, v in canary_names.items()}
    disk_bytes = len(records) * 3 * 64 * 64 * 4
    assert shutil.disk_usage(directory).free > disk_bytes + 2_000_000_000, "Insufficient experiment disk space"
    array = np.lib.format.open_memmap(
        directory / "images.npy", mode="w+", dtype=np.float32, shape=(len(records), 3, 64, 64)
    )
    resize = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    def decode(rec):
        with Image.open(image_dir / rec[0]) as img:
            return resize(img.convert("RGB")).numpy()

    started = time.perf_counter()
    # Bounded batches keep ThreadPoolExecutor from queuing the entire decoded dataset.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(records), 256):
            for offset, image in enumerate(pool.map(decode, records[start : start + 256])):
                array[start + offset] = image
            if start % 4096 == 0:
                print(f"DATA {start}/{len(records)} decoded", flush=True)
    array.flush()
    meta = {
        "names": [r[0] for r in records],
        "era_key": [regime_ids[r[1]] for r in records],
        "canary": [canary_ids[r[2]] for r in records],
        "object_categories": [r[3] for r in records],
        "regime_order": regime_order,
        "regime_names": regime_names,
        "canary_names": canary_names,
        "scene_counts": dict(Counter(r[2] for r in records)),
        "regime_counts": dict(Counter(regime_names[regime_ids[r[1]]] for r in records)),
        "filtered": dict(failures),
        "labels_sha256": label_hash.hexdigest(),
        "names_sha256": sha_bytes("\n".join(r[0] for r in records)),
        "bdd_root": str(bdd_root),
        "split": "train",
        "img_size": 64,
        "data_min_canary_count": 100,
        "n_images": len(records),
        "decoded_seconds": time.perf_counter() - started,
    }
    write_json(directory / "records.json", meta)
    write_json(
        directory / "complete.json",
        {
            "complete": True,
            "records_sha256": file_sha(directory / "records.json"),
            "images_sha256": file_sha(directory / "images.npy"),
        },
    )
    print(f"DATA COMPLETE {len(records)} images, {len(canary_names)} scenes", flush=True)
    return CachedSource(directory)


def make_config(arm, seed, steps, mode="warm_well"):
    """Compose the existing configuration with explicitly pinned research overrides."""
    overrides = [
        "data=bdd_local",
        "data.split=train",
        "data.min_canary_count=100",
        "backbone=mae",
        "img_size=64",
        "batch_size=32",
        "device=mps",
        "init=from_scratch",
        f"seed={seed}",
        "stream.support_per_canary=25",
        "stream.query_per_canary=25",
        "stream.canary_seed=0",
        "+ssl.aux_weight=0.01",
        *ARMS[arm],
    ]
    if mode == "warm_well":
        overrides += [f"warmup.max_steps={steps}", f"warmup.min_steps={steps + 1}", "warmup.max_passes=50"]
    else:
        overrides += ["gate_arms=false", "filter=accept_all", "eval_every=50"]
    with initialize_config_dir(config_dir=str(ROOT / "cafl4ds/configs"), version_base=None):
        return compose(config_name=mode, overrides=overrides)


def make_method(cfg, seed):
    """Initialize encoder and decoder from independent seeds on the required device."""
    with isolated_rng(10_000 + seed):
        encoder = instantiate(cfg.encoder)
    with isolated_rng(20_000 + seed):
        method = instantiate(cfg.ssl, encoder=encoder)
    return method.to("mps")


def split_membership(source):
    """Reproduce and persist the stream's fixed canary reservation by image identity."""
    gen = torch.Generator().manual_seed(0)
    support, query = [], []
    labels = source.load().canary
    for cls in sorted(set(labels.tolist())):
        ids = (labels == cls).nonzero(as_tuple=True)[0]
        ids = ids[torch.randperm(len(ids), generator=gen)]
        support.extend(ids[:25].tolist())
        query.extend(ids[25:50].tolist())
    held = set(support + query)
    train = [i for i in range(len(labels)) if i not in held]
    assert not set(support) & set(query) and not held & set(train)
    return {
        "support": support,
        "query": query,
        "train": train,
        "membership_sha256": sha_bytes(json.dumps([support, query, train])),
        "chance_accuracy": 1 / source.num_canary_classes,
    }


class Observation:
    """Capture the actual training outputs and isolate only the router-noise draws."""

    def __init__(self, method, seed):
        self.method, self.seed, self.step = method, seed, 0
        self.handles, self.probs = [], {}
        self.capture = False
        self.views, self.ids_restore = None, None
        original = method._masked_recon_loss

        def recon(imgs):
            loss, mask, aux = original(imgs)
            if method.training:
                self.reconstruction = ((loss * mask).sum() / mask.sum()).detach()
                self.aux = aux.detach()
            return loss, mask, aux

        method._masked_recon_loss = recon
        original_masking = method.encoder.random_masking

        def masking(x, ratio):
            result = original_masking(x, ratio)
            if self.capture:
                self.ids_restore = result[2].detach()
            return result

        method.encoder.random_masking = masking

        def capture_views(_m, args):
            if self.capture:
                self.views = args[0].detach()

        self.handles.append(method.encoder.patch_embed.register_forward_pre_hook(capture_views))
        for index, block in enumerate(method.encoder.blocks):
            if not isinstance(block.mlp, MoEMlp):
                continue
            contexts = []

            def before(module, _args, i=index, stack=contexts):
                if module.training:
                    ctx = isolated_rng(30_000_000 + self.seed * 100_000 + self.step * 10 + i)
                    ctx.__enter__()
                    stack.append(ctx)

            def after(module, _args, result, i=index, stack=contexts):
                if module.training:
                    self.probs[i] = result.routing_probs.detach()
                    stack.pop().__exit__(None, None, None)

            self.handles += [block.mlp.register_forward_pre_hook(before), block.mlp.register_forward_hook(after)]

    def counts(self):
        """Counts are read after the timed GPU work, with CLS and patches separate."""
        result = {}
        for i, p in self.probs.items():
            ids = p.argmax(dim=-1).cpu()
            result[i] = {
                "patch": torch.bincount(ids[:, 1:].flatten(), minlength=p.shape[-1]).tolist(),
                "cls": torch.bincount(ids[:, 0], minlength=p.shape[-1]).tolist(),
            }
        return result

    def close(self):
        """Remove hooks before discarding the method."""
        for h in self.handles:
            h.remove()


def entropy(prob):
    """Normalized categorical entropy, preserving zero contributions."""
    return -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1) / math.log(prob.shape[-1])


def routing_readout(method, query, step):
    """Full and masked routing, and expert similarity on shared actual block inputs."""
    if not hasattr(method.encoder, "routing"):
        return {}, []
    inputs = {}
    handles = []
    for index, b in enumerate(method.encoder.blocks):
        if isinstance(b.mlp, MoEMlp):

            def capture(_module, args, i=index):
                flat = args[0][:, 1:].detach().reshape(-1, method.encoder.embed_dim)
                inputs[i] = flat[:: max(1, len(flat) // 512)][:512]

            handles.append(b.mlp.register_forward_pre_hook(capture))
    full = method.encoder.routing(query)
    for h in handles:
        h.remove()
    masked = {}
    handles = []
    for index, b in enumerate(method.encoder.blocks):
        if isinstance(b.mlp, MoEMlp):

            def capture_mask(_module, _args, out, i=index):
                masked[i] = out.routing_probs.detach()

            handles.append(b.mlp.register_forward_hook(capture_mask))
    with isolated_rng(90_000):
        method.encoder.forward_with_aux(query.to("mps"), mask_ratio=0.75)
    for h in handles:
        h.remove()
    metrics, rows = {}, []
    for i, p in full.items():
        p = p.cpu()
        e = p.shape[-1]
        patch_ids = p[:, 1:].argmax(dim=-1)
        hard_counts = torch.bincount(patch_ids.flatten(), minlength=e)
        cls_counts = torch.bincount(p[:, 0].argmax(dim=-1), minlength=e)
        load = hard_counts.float() / hard_counts.sum()
        soft = p[:, 1:].mean(dim=(0, 1))
        pm = masked[i].cpu()[:, 1:]
        masked_counts = torch.bincount(pm.argmax(dim=-1).flatten(), minlength=e)
        pairs = []
        weight_pairs = []
        expert_outputs = [expert(inputs[i]).detach().cpu() for expert in method.encoder.blocks[i].mlp.experts]
        weights = [
            torch.cat([v.detach().flatten().cpu() for v in expert.parameters()])
            for expert in method.encoder.blocks[i].mlp.experts
        ]
        for a in range(e):
            for b in range(a + 1, e):
                pairs.append(measurements.linear_cka(expert_outputs[a], expert_outputs[b]))
                weight_pairs.append(float(torch.nn.functional.cosine_similarity(weights[a], weights[b], dim=0)))
        metrics.update(
            {
                f"route_load_entropy_l{i}": float(entropy(load)),
                f"route_token_entropy_l{i}": float(entropy(p[:, 1:]).mean()),
                f"dead_expert_frac_l{i}": float((hard_counts == 0).float().mean()),
                f"min_expert_load_l{i}": float(load.min()),
                f"near_starved_frac_l{i}": float((load < 0.01).float().mean()),
                f"expert_output_cka_l{i}": float(np.mean(pairs)),
                f"expert_weight_cosine_l{i}": float(np.mean(weight_pairs)),
                f"masked_dead_expert_frac_l{i}": float((masked_counts == 0).float().mean()),
            }
        )
        for expert in range(e):
            metrics[f"route_load_l{i}_e{expert}"] = float(load[expert])
            rows.append(
                {
                    "step": step,
                    "layer": i,
                    "expert": expert,
                    "patch_count": int(hard_counts[expert]),
                    "patch_load": float(load[expert]),
                    "cls_count": int(cls_counts[expert]),
                    "soft_probability": float(soft[expert]),
                    "masked_patch_count": int(masked_counts[expert]),
                    "masked_patch_load": float(masked_counts[expert] / masked_counts.sum()),
                }
            )
    return metrics, rows


def reconstruction_panel(method, images):
    """Average the exact MAE per-image score over eight fixed mask/augmentation draws."""
    values = []
    for draw in range(8):
        chunks = []
        for start in range(0, len(images), 64):
            with isolated_rng(80_000 + draw * 1000 + start):
                chunks.append(method.per_sample_loss(images[start : start + 64].to("mps")).cpu())
        values.append(torch.cat(chunks))
    draws = torch.stack(values)
    return float(draws.mean()), float(draws.mean(dim=1).std()), draws.mean(dim=0).tolist()


def evaluate(method, monitor, query, train_panel, step, detailed=True):
    """Run fixed read-only evaluation and restore model mode and all RNG states."""
    started = time.perf_counter()
    was_training = method.training
    with isolated_rng(70_000), torch.no_grad():
        method.eval()
        monitor.run_knn = monitor.run_linear = detailed
        health = monitor.measure(method, step)
        method.eval()
        held, held_sd, held_per_image = reconstruction_panel(method, query)
        train, train_sd, _ = reconstruction_panel(method, train_panel)
        health.update(
            {
                "heldout_mse": held,
                "heldout_mask_draw_sd": held_sd,
                "train_reference_mse": train,
                "train_mask_draw_sd": train_sd,
                "generalization_gap": held - train,
            }
        )
        routes, route_rows = routing_readout(method, query, step)
        health.update(routes)
        torch.mps.synchronize()
    method.train(was_training)
    return health, route_rows, time.perf_counter() - started, held_per_image


def choose_well(loss_rows, health, method, monitor, source, membership, train_panel, directory):
    """Select a sustained reconstruction plateau with competence, without altering training."""
    losses = [r["reconstruction_loss"] for r in loss_rows]
    streak = 0
    history = []
    candidates = []
    hmap = {int(h["step"]): h for h in health}
    chance = membership["chance_accuracy"]
    for step in sorted(hmap):
        if step < 200:
            continue
        previous = np.mean(losses[step - 100 : step - 50])
        recent = np.mean(losses[step - 50 : step])
        improvement = float((previous - recent) / (abs(previous) + abs(recent) + 1e-8))
        h = hmap[step]
        last = [hmap[s]["heldout_mse"] for s in sorted(hmap) if max(0, step - 150) <= s <= step]
        stable = abs(improvement) < 0.001 and max(last) / min(last) <= 1.02
        streak = streak + 1 if stable else 0
        route_ok = all(v == 0 for k, v in h.items() if k.startswith("dead_expert_frac_l"))
        # Use the trailing 200 *at this checkpoint*, not a future training window.
        counts = {}
        for row in loss_rows[max(0, step - 200) : step]:
            for key, value in row.items():
                if key.startswith("patch_assignments_l"):
                    counts[key] = counts.get(key, 0) + value
        fed = all(v > 0 for v in counts.values())
        history.append(
            {
                "step": step,
                "relative_improvement": improvement,
                "stable": stable,
                "stable_streak": streak,
                "route_ok": route_ok,
                "all_experts_trained": fed,
            }
        )
        if streak >= 3 and route_ok and fed:
            candidates.append(step)
    selected, evidence = None, None
    for step in candidates:
        warmup.load_well(method, directory / "checkpoints" / f"step{step:04d}.pt")
        h, _, _, _ = evaluate(
            method, monitor, source.load().images[membership["query"]], train_panel, step, detailed=True
        )
        if h["knn_acc"] > chance and h["linear_acc"] > chance and h["rankme"] > 0.5:
            selected, evidence = step, h
            break
    result = {
        "selected_step": selected,
        "status": "competent_plateau" if selected else "cap_limited_no_certified_plateau",
        "plateau_tolerance": 0.001,
        "heldout_stability_ratio": 1.02,
        "history": history,
        "candidate_steps": candidates,
        "competence": evidence,
    }
    if selected is not None:
        shutil.copyfile(directory / "checkpoints" / f"step{selected:04d}.pt", directory / "selected_well.pt")
    write_json(directory / "well_selection.json", result)
    return result


def verify_reload(method, checkpoint, query, seed):
    """Strict reload reproduces embeddings and fixed-mask reconstruction exactly on MPS."""
    warmup.load_well(method, checkpoint)
    method.eval()
    with torch.no_grad(), isolated_rng(seed):
        features = method.encoder.embed(query[:16]).cpu()
        scores = method.per_sample_loss(query[:16].to("mps")).cpu()
    warmup.load_well(method, checkpoint)
    with torch.no_grad(), isolated_rng(seed):
        again = method.encoder.embed(query[:16]).cpu()
        again_scores = method.per_sample_loss(query[:16].to("mps")).cpu()
    assert torch.equal(features, again) and torch.equal(scores, again_scores)
    return {
        "strict_reload": True,
        "embeddings_exact": True,
        "fixed_mask_scores_exact": True,
        "checkpoint_sha256": file_sha(checkpoint),
    }


def run_arm(source, membership, output, arm, seed, steps, pilot=False):
    """Run the stock MAE objective with paired random draws and persistent observations."""
    directory = output / ("pilot" if pilot else "runs") / arm / f"s{seed}"
    if (directory / "complete.json").exists():
        print(f"SKIP COMPLETE {directory}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "checkpoints").mkdir(exist_ok=True)
    cfg = make_config(arm, seed, steps)
    cfg.well_out = str(directory / f"step{steps}.pt")
    OmegaConf.save(cfg, directory / "config.yaml", resolve=True)
    stream = RegimeStream(source, batch_size=32, support_per_canary=25, query_per_canary=25, canary_seed=0, seed=seed)
    data = source.load()
    query = data.images[membership["query"]]
    assert torch.equal(query, stream.eval_sets.probe_query.images)
    gen = torch.Generator().manual_seed(60_000)
    train_ids = torch.tensor(membership["train"])
    train_ids = train_ids[torch.randperm(len(train_ids), generator=gen)[: len(query)]]
    train_panel = data.images[train_ids]
    monitor = instantiate(cfg.monitor, eval_sets=stream.eval_sets)
    method = make_method(cfg, seed)
    optimizer = instantiate(cfg.optim, params=method.parameters())
    observer = Observation(method, seed)
    n_encoder = sum(p.numel() for p in method.encoder.parameters())
    n_method = sum(p.numel() for p in method.parameters())
    losses, health, routes, paired = [], [], [], []
    grad_checks = []
    training_seconds, evaluation_seconds, checkpoint_seconds, images_seen = 0.0, 0.0, 0.0, 0
    peak_mps_allocated = peak_mps_driver = 0
    train_order_hash = hashlib.sha256()
    wall_start = time.perf_counter()
    warmup.save_well(method, directory / "checkpoints" / "step0000.pt")
    h, r, seconds, _ = evaluate(method, monitor, query, train_panel, 0)
    health.append(h)
    routes.extend(r)
    evaluation_seconds += seconds
    write_csv(directory / "health.csv", health)
    print(
        f"START {'PILOT' if pilot else 'PRIMARY'} {arm} seed={seed}, params={n_method}, query={len(query)}", flush=True
    )
    step = 0
    method.train()
    for pass_index in range(50):
        indices = torch.tensor([idx for _, idx in stream._order_stream])
        indices = indices[torch.randperm(len(indices), generator=stream._generator)]
        for start in range(0, len(indices), 32):
            ids = indices[start : start + 32]
            step += 1
            observer.step = step
            observer.capture = step <= 5 or step % 100 == 0
            observer.probs = {}
            train_order_hash.update(ids.numpy().tobytes())
            torch.mps.synchronize()
            started = time.perf_counter()
            with isolated_rng(1_000_000 + seed * 100_000 + step):
                optimizer.zero_grad(set_to_none=True)
                loss = method.training_step(data.images[ids].to("mps"))
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(method.parameters(), 1.0)
                if step == 1:
                    for i, b in enumerate(method.encoder.blocks):
                        if isinstance(b.mlp, MoEMlp):
                            router_grad = float(b.mlp.router.weight.grad.abs().sum().cpu())
                            expert_grads = [
                                sum(float(p.grad.abs().sum().cpu()) for p in e.parameters() if p.grad is not None)
                                for e in b.mlp.experts
                            ]
                            assert router_grad > 0 and any(v > 0 for v in expert_grads)
                            grad_checks.append(
                                {"layer": i, "router_grad_abs_sum": router_grad, "expert_grad_abs_sums": expert_grads}
                            )
                optimizer.step()
            torch.mps.synchronize()
            elapsed = time.perf_counter() - started
            training_seconds += elapsed
            images_seen += len(ids)
            row = {
                "step": step,
                "images_seen": images_seen,
                "reconstruction_loss": float(observer.reconstruction.cpu()),
                "aux_loss": float(observer.aux.cpu()),
                "weighted_aux_loss": 0.01 * float(observer.aux.cpu()),
                "total_loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.cpu()),
                "train_seconds": elapsed,
                "cumulative_train_seconds": training_seconds,
                "pass": pass_index,
            }
            assert all(math.isfinite(row[k]) for k in ("reconstruction_loss", "aux_loss", "total_loss", "grad_norm"))
            assert abs(row["total_loss"] - row["reconstruction_loss"] - row["weighted_aux_loss"]) < 2e-6
            for i, counts in observer.counts().items():
                for label, values in counts.items():
                    for e, value in enumerate(values):
                        row[f"{label}_assignments_l{i}_e{e}"] = value
            losses.append(row)
            if observer.capture:
                paired.append(
                    {
                        "step": step,
                        "indices_sha256": sha_bytes(ids),
                        "views_sha256": sha_bytes(observer.views),
                        "mask_restore_sha256": sha_bytes(observer.ids_restore),
                    }
                )
            peak_mps_allocated = max(peak_mps_allocated, torch.mps.current_allocated_memory())
            peak_mps_driver = max(peak_mps_driver, torch.mps.driver_allocated_memory())
            if step % 50 == 0 or step == steps:
                checkpoint_start = time.perf_counter()
                warmup.save_well(method, directory / "checkpoints" / f"step{step:04d}.pt")
                checkpoint_seconds += time.perf_counter() - checkpoint_start
                h, r, seconds, per_image = evaluate(
                    method, monitor, query, train_panel, step, detailed=step in PROBE_STEPS or step == steps
                )
                h.update({"images_seen": images_seen, "train_seconds": training_seconds})
                health.append(h)
                routes.extend(r)
                evaluation_seconds += seconds
                write_csv(directory / "losses.csv", losses)
                write_csv(directory / "health.csv", health)
                write_csv(directory / "routing_detail.csv", routes)
                write_json(directory / "paired_draws.json", paired)
                write_json(
                    directory / "progress.json",
                    {
                        "arm": arm,
                        "seed": seed,
                        "step": step,
                        "steps": steps,
                        "train_seconds": training_seconds,
                        "heldout_mse": h["heldout_mse"],
                    },
                )
                print(
                    f"PROGRESS {arm} s{seed} {step}/{steps} recon={row['reconstruction_loss']:.5f} "
                    f"heldout={h['heldout_mse']:.5f} train_s={training_seconds:.1f}",
                    flush=True,
                )
            if step >= steps:
                break
        if step >= steps:
            break
    assert step == steps, f"Training cap mismatch: {step} vs {steps}"
    endpoint = directory / f"step{steps}.pt"
    shutil.copyfile(directory / "checkpoints" / f"step{step:04d}.pt", endpoint)
    endpoint_reload = verify_reload(method, endpoint, query, 900_000)
    selection = (
        choose_well(losses, health, method, monitor, source, membership, train_panel, directory) if not pilot else None
    )
    selected_reload = (
        verify_reload(method, directory / "selected_well.pt", query, 900_000)
        if selection and selection["selected_step"]
        else None
    )
    tail_counts = {
        k: sum(r.get(k, 0) for r in losses[-200:]) for k in losses[-1] if k.startswith("patch_assignments_l")
    }
    summary = {
        "arm": arm,
        "seed": seed,
        "device": "mps",
        "steps": steps,
        "images_seen": images_seen,
        "passes_used": pass_index + 1,
        "stop_reason": "fixed_step_budget",
        "plateaued": False,
        "encoder_parameters": n_encoder,
        "method_parameters": n_method,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "total_wall_seconds": time.perf_counter() - wall_start,
        "images_per_second": images_seen / training_seconds,
        "seconds_per_update": training_seconds / steps,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "peak_mps_allocated_mib": peak_mps_allocated / 1024**2,
        "peak_mps_driver_mib": peak_mps_driver / 1024**2,
        "checkpoint_bytes": endpoint.stat().st_size,
        "endpoint": health[-1],
        "train_order_sha256": train_order_hash.hexdigest(),
        "data_membership_sha256": membership["membership_sha256"],
        "gradient_checks": grad_checks,
        "final_200_patch_assignments": tail_counts,
        "all_experts_trained_final_200": all(v > 0 for v in tail_counts.values()),
        "endpoint_reload": endpoint_reload,
        "well_selection": selection,
        "selected_reload": selected_reload,
    }
    write_json(directory / "warm_summary.json", summary)
    trust = {
        k: {"status": "candidate", "provenance": "WP-M0 descriptive routing readout"}
        for k in health[-1]
        if any(k.startswith(p) for p in ("route_", "dead_", "min_expert", "near_starved", "expert_", "masked_dead"))
    }
    write_json(
        directory / "trust.json",
        {
            "routing": trust,
            "note": "Existing geometry trust comes from cafl4ds.health_trust; no new calibration claimed.",
        },
    )
    write_json(
        directory / "complete.json",
        {"complete": True, "steps": steps, "summary_sha256": file_sha(directory / "warm_summary.json")},
    )
    observer.close()
    del optimizer, observer, method
    torch.mps.empty_cache()
    print(f"COMPLETE {arm} s{seed}, {steps} steps, wall={summary['total_wall_seconds']:.1f}s", flush=True)


def synthetic_preflight(output):
    """MPS native MoE synthetic MAE warm-up, checkpoint, and strict reload."""
    from cafl4ds.data.attributes import SyntheticAttributeSource

    cfg = make_config("moe_native", 0, 10)
    method = make_method(cfg, 0)
    source = SyntheticAttributeSource(img_size=64, per_cell=8, long_tail=False)
    stream = RegimeStream(source, batch_size=32, support_per_canary=2, query_per_canary=2)
    optimizer = instantiate(cfg.optim, params=method.parameters())
    summary = warmup.warm_until_settled(
        method=method,
        stream=stream,
        optimizer=optimizer,
        max_steps=10,
        plateau=warmup.PlateauCriterion(min_steps=11),
        device="mps",
    )
    path = output / "synthetic" / "well.pt"
    warmup.save_well(method, path)
    check = verify_reload(method, path, stream.eval_sets.probe_query.images, 1234)
    write_json(output / "synthetic" / "preflight.json", {"summary": summary, "reload": check})
    del method, optimizer, stream
    torch.mps.empty_cache()
    print("SYNTHETIC MPS PREFLIGHT PASSED", flush=True)


def verify_pairing(output, phase, seeds):
    """Assert actual sampled image order, augmentations, and masks match across arms."""
    checks = []
    for seed in seeds:
        summaries = [json.loads((output / phase / arm / f"s{seed}" / "warm_summary.json").read_text()) for arm in ARMS]
        draws = [json.loads((output / phase / arm / f"s{seed}" / "paired_draws.json").read_text()) for arm in ARMS]
        assert len({s["train_order_sha256"] for s in summaries}) == 1
        assert draws[0] == draws[1] == draws[2], f"Paired draws mismatch seed {seed}"
        checks.append(
            {
                "seed": seed,
                "train_order_matches": True,
                "sampled_views_and_masks_match": True,
                "n_checked_steps": len(draws[0]),
                "images_seen": summaries[0]["images_seen"],
            }
        )
    write_json(output / f"{phase}_pairing.json", checks)


class RoutingMonitor(HealthMonitor):
    """Minimal full-image per-layer routing extension for the integration corpus."""

    def measure(self, method, step):
        metrics = super().measure(method, step)
        if hasattr(method.encoder, "routing"):
            with torch.no_grad(), isolated_rng(70_000):
                mode = method.training
                method.eval()
                probs = method.encoder.routing(self.eval_sets.probe_query.images)
                for i, p in probs.items():
                    p = p.cpu()[:, 1:]
                    load = torch.bincount(p.argmax(dim=-1).flatten(), minlength=p.shape[-1]).float()
                    load /= load.sum()
                    metrics[f"route_load_entropy_l{i}"] = float(entropy(load))
                    metrics[f"route_token_entropy_l{i}"] = float(entropy(p).mean())
                    for e, v in enumerate(load):
                        metrics[f"route_load_l{i}_e{e}"] = float(v)
                method.train(mode)
        return metrics


def smoke(source, membership, output, arm):
    """Run a full seed-0 accept-all regime walk and publish both validation tiers."""
    directory = output / "smoke" / arm
    if (directory / "complete.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    warm_dir = output / "runs" / arm / "s0"
    selection = json.loads((warm_dir / "well_selection.json").read_text())
    path = warm_dir / ("selected_well.pt" if selection["selected_step"] else "step2000.pt")
    cfg = make_config(arm, 0, 2000, mode="deploy")
    cfg.well = str(path)
    OmegaConf.save(cfg, directory / "config.yaml", resolve=True)
    method = make_method(cfg, 0)
    warmup.load_well(method, path)
    stream = RegimeStream(source, batch_size=32, support_per_canary=25, query_per_canary=25, canary_seed=0, seed=0)
    monitor = RoutingMonitor(eval_sets=stream.eval_sets, knn_k=20)
    started = time.perf_counter()
    live = harness.run_stream_arm(
        name=f"wp_m0_{arm}_live",
        role="live",
        method=method,
        stream=stream,
        optimizer=instantiate(cfg.optim, params=method.parameters()),
        selection_filter=instantiate(cfg.filter),
        monitor=monitor,
        out_dir=directory,
        eval_every=50,
        device="mps",
    )
    report = deploy.build_deploy_report(
        config_header={
            "backbone": "mae",
            "I": "from_scratch",
            "warm": True,
            "diet": "regime",
            "seed": 0,
            "img_size": 64,
            "device": "mps",
            "A": "AcceptAll",
            "gate_arms": False,
        },
        family=Backbone.MAE,
        live=live,
        expected_signals=[*DEFAULT_LABEL_FREE_SIGNALS[Backbone.MAE], *CANARY_SIGNALS],
        canary_chance=membership["chance_accuracy"],
        tier_b_thresholds=dict(cfg.tier_b),
    )
    route_keys = [k for k in live.health[-1] if k.startswith("route_")]
    # The generic registry calls unknown metrics uncalibrated; these exploratory routing columns
    # are explicitly candidate, and this does not change any calibrated geometry entry.
    for key in route_keys:
        report["trust"][key] = [
            {
                "mode": "quality",
                "status": "candidate",
                "caveat": "Descriptive WP-M0 routing; no positive-control calibration",
                "provenance": "WP-M0",
            }
        ]
    deploy_corpus.write_corpus(
        directory / "corpus",
        reports=[(0, report)],
        era_names=stream.era_names,
        composition=stream.era_composition(),
        manifest={
            "study": "WP-M0",
            "arm": arm,
            "well": str(path),
            "certified_well": selection["selected_step"] is not None,
            "data_membership_sha256": membership["membership_sha256"],
        },
    )
    write_json(directory / "report.json", report)
    shutil.copyfile(directory / "config.yaml", directory / "corpus" / "config.yaml")
    write_json(
        directory / "complete.json",
        {
            "complete": True,
            "seconds": time.perf_counter() - started,
            "tier_a": report["validation"]["passed"],
            "tier_b": report["tier_b"]["passed"],
        },
    )
    print(
        f"SMOKE COMPLETE {arm} Tier A={report['validation']['passed']} Tier B={report['tier_b']['passed']}", flush=True
    )
    del method, stream, monitor, live
    torch.mps.empty_cache()


def provenance(output):
    """Pin code, runtime, and the predeclared protocol before training."""
    directory = output / "provenance"
    directory.mkdir(parents=True, exist_ok=True)
    for args, name in [
        (["git", "rev-parse", "HEAD"], "git_sha.txt"),
        (["git", "diff", "HEAD"], "working_tree.patch"),
        (["git", "status", "--short"], "git_status.txt"),
    ]:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
        (directory / name).write_text(result.stdout)
    shutil.copyfile(__file__, directory / Path(__file__).name)
    shutil.copyfile(ROOT / "docs/experiments/WP-M0-warm-well-mae-bdd-plan.md", directory / "plan.md")
    write_json(
        directory / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
            "device": "mps",
            "mps_available": torch.backends.mps.is_available(),
            "precision": "float32",
            "seeds": list(range(5)),
            "steps": 2000,
            "parity_canary_pp": 3,
            "parity_mse_percent": 5,
            "near_starvation_fraction": 0.01,
            "training_timing_includes_rng_isolation": True,
        },
    )


def validate_observer(output):
    """Verify paired real augmentations/masks and auxiliary gradients on MPS before BDD runs."""
    from cafl4ds.data.attributes import SyntheticAttributeSource

    source = SyntheticAttributeSource(img_size=64, per_cell=8, long_tail=False)
    images = source.load().images[:32]
    traces = []
    for arm in ARMS:
        cfg = make_config(arm, 0, 5)
        method = make_method(cfg, 0)
        optimizer = instantiate(cfg.optim, params=method.parameters())
        observed = Observation(method, 0)
        method.train()
        arm_trace = []
        for step in range(1, 6):
            observed.step, observed.capture = step, True
            with isolated_rng(1_000_000 + step):
                optimizer.zero_grad(set_to_none=True)
                loss = method.training_step(images.to("mps"))
                loss.backward()
                assert torch.isfinite(loss)
                for block in method.encoder.blocks:
                    if isinstance(block.mlp, MoEMlp):
                        assert block.mlp.router.weight.grad.abs().sum().item() > 0
                optimizer.step()
            arm_trace.append([sha_bytes(observed.views), sha_bytes(observed.ids_restore)])
        traces.append(arm_trace)
        observed.close()
        del method, optimizer, observed
        torch.mps.empty_cache()
    assert traces[0] == traces[1] == traces[2]
    synthetic_preflight(output)
    write_json(
        output / "synthetic/observer_validation.json",
        {"passed": True, "all_three_arms": True, "paired_augmentation_and_masks": True, "steps": 5, "device": "mps"},
    )
    print("OBSERVER VALIDATION PASSED", flush=True)


def main():
    """Execute phases sequentially with resumable complete-run markers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["validate", "prepare", "pilot", "run", "smoke", "all"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/wp_m0_mae_bdd_mps_2026-10-07")
    parser.add_argument("--bdd-root", type=Path, default=Path("/Users/williammin/cafl4ds_datasets/BDD"))
    args = parser.parse_args()
    assert torch.backends.mps.is_available(), "MPS access is required; no CPU fallback is enabled"
    torch.set_num_threads(8)
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.output / "provenance/environment.json").exists() or args.phase == "all":
        provenance(args.output)
    if args.phase == "validate":
        validate_observer(args.output)
        return
    source = prepare_data(args.output / "data", args.bdd_root)
    membership = split_membership(source)
    write_json(args.output / "data/membership.json", membership)
    if args.phase == "prepare":
        return
    if args.phase in {"pilot", "all"}:
        synthetic_preflight(args.output)
        for arm in ARMS:
            run_arm(source, membership, args.output, arm, 0, 200, pilot=True)
        verify_pairing(args.output, "pilot", [0])
    if args.phase in {"run", "all"}:
        for seed in range(5):
            for arm in ARMS:
                run_arm(source, membership, args.output, arm, seed, 2000)
            verify_pairing(args.output, "runs", list(range(seed + 1)))
    if args.phase in {"smoke", "all"}:
        for arm in ARMS:
            smoke(source, membership, args.output, arm)
    print(f"PHASE {args.phase} COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
