"""Paired, resumable MoE-MAE image admission experiments and their analysis.

Training sees images and stable integer IDs only. Scene/regime attributes are used
for fixed evaluation panels and post-decision audits, never to select training data.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import resource
import shutil
import subprocess
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
from scipy.stats import t as student_t

from cafl4ds import measurements
from cafl4ds.data.attributes import SyntheticAttributeSource
from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.routing_entropy import RoutingEntropyHalf, SelectionRecord
from cafl4ds.health_trust import Backbone, annotate
from cafl4ds.models.vit import FeedForwardOutput, MoEMlp, MoEViTEncoder
from cafl4ds.ssl.mae import MAE

GATE_SEMANTICS = "full_softmax_before_topk_no_renormalization_v1"
ARMS = ("random", "high", "low")


def digest_file(path: Path) -> str:
    """Hash files incrementally without loading a data cache into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    """Hash tensor values for pairing and resume evidence."""
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Hash names and values in a method checkpoint."""
    return hashlib.sha256("".join(k + tensor_hash(v) for k, v in sorted(state.items())).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically write strict JSON; non-finite scientific measurements fail loudly."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def keyed_seed(*parts: object) -> int:
    """Stable independent seed, unaffected by Python's randomized string hash."""
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little") % (2**63 - 1)


@contextmanager
def isolated_rng(seed: int, device: str) -> Iterator[None]:
    """Save/restore CPU and active accelerator RNG, including nested evaluation calls."""
    cpu = torch.get_rng_state()
    accelerator = torch.mps.get_rng_state() if device == "mps" else None
    cuda_states = torch.cuda.get_rng_state_all() if device.startswith("cuda") else None
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(cpu)
        if accelerator is not None:
            torch.mps.set_rng_state(accelerator)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def synchronize(device: str) -> None:
    """Complete device work before timing asynchronous accelerator operations."""
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


class CachedRoutingCorpus:
    """Read and verify the existing identity-preserving BDD cache without modifying it."""

    def __init__(self, directory: str, verify_hashes: bool = True) -> None:
        """Open a WP-M0 cache; verification binds image contents to its recorded provenance."""
        directory_path = Path(directory).resolve()
        complete = json.loads((directory_path / "complete.json").read_text())
        if not complete.get("complete"):
            raise ValueError("Incomplete BDD image cache.")
        if verify_hashes:
            for name, key in (("records.json", "records_sha256"), ("images.npy", "images_sha256")):
                if digest_file(directory_path / name) != complete[key]:
                    raise ValueError(f"Cache content hash mismatch: {name}")
        self.metadata = json.loads((directory_path / "records.json").read_text())
        # Copy-on-write mapping: the original cache cannot be changed through this tensor.
        self.array = np.load(directory_path / "images.npy", mmap_mode="c")
        self.images = torch.from_numpy(self.array)
        self.names = self.metadata["names"]
        self.labels = torch.tensor(self.metadata["canary"], dtype=torch.long)
        self.regimes = torch.tensor(self.metadata["era_key"], dtype=torch.long)
        self.scene_names = {int(k): v for k, v in self.metadata["canary_names"].items()}
        self.regime_names = {int(k): v for k, v in self.metadata["regime_names"].items()}
        self.fingerprint = {"directory": str(directory_path), **complete}
        if len(set(self.names)) != len(self.images) or len(self.labels) != len(self.images):
            raise ValueError("Cache identities must be unique and aligned with images.")


class SyntheticRoutingCorpus:
    """Network-free source for end-to-end and resume verification, never scientific BDD results."""

    def __init__(self, img_size: int = 16, per_cell: int = 12) -> None:
        """Generate two independent label/regime axes with stable sample IDs."""
        data = SyntheticAttributeSource(
            img_size=img_size, num_regimes=2, num_canary_classes=2, per_cell=per_cell, long_tail=False
        ).load()
        self.images, self.labels, self.regimes = data.images, data.canary, data.era_key
        self.names = [f"synthetic-{i}" for i in range(len(self.images))]
        self.scene_names, self.regime_names = data.canary_names, data.regime_names
        self.metadata = {"synthetic": True, "n_images": len(self.images)}
        self.fingerprint = {"synthetic": tensor_hash(self.images)}


Corpus = CachedRoutingCorpus | SyntheticRoutingCorpus


def reserve_panels(corpus: Corpus, cfg: DictConfig) -> dict[str, list[int]]:
    """Reserve balanced support/query and an independent natural-frequency query before ordering."""
    generator = torch.Generator().manual_seed(int(cfg.split_seed))
    support, query = [], []
    for cls in corpus.labels.unique().tolist():
        indices = (corpus.labels == cls).nonzero().flatten()
        indices = indices[torch.randperm(len(indices), generator=generator)]
        n_support, n_query = int(cfg.support_per_class), int(cfg.query_per_class)
        if len(indices) <= n_support + n_query:
            raise ValueError("Every retained class needs support, query, and remaining training images.")
        support.extend(indices[:n_support].tolist())
        query.extend(indices[n_support : n_support + n_query].tolist())
    held = set(support + query)
    remaining = torch.tensor([i for i in range(len(corpus.images)) if i not in held])
    remaining = remaining[torch.randperm(len(remaining), generator=generator)].tolist()
    n_natural = int(cfg.natural_query_size)
    if n_natural < 2 or n_natural >= len(remaining):
        raise ValueError("Natural query must contain at least two images and leave a training pool.")
    return {"support": support, "query": query, "natural": remaining[:n_natural], "train": remaining[n_natural:]}


def epoch_order(train: list[int], seed: int, epoch: int) -> torch.Tensor:
    """Rebuild a full seed/epoch permutation independently of all model and selection RNG."""
    generator = torch.Generator().manual_seed(keyed_seed("data", seed, epoch))
    return torch.tensor(train)[torch.randperm(len(train), generator=generator)]


def paired_views(
    method: MAE, images: torch.Tensor, ids: torch.Tensor, seed: int, epoch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make matching per-image augmentations and masks across policies, restoring CPU RNG."""
    original = torch.get_rng_state()
    views, noises = [], []
    try:
        for image, sample_id in zip(images, ids.tolist(), strict=True):
            torch.random.default_generator.manual_seed(keyed_seed("view", seed, epoch, sample_id))
            views.append(method.augment(image))
            generator = torch.Generator().manual_seed(keyed_seed("mask", seed, epoch, sample_id))
            noises.append(torch.rand(method.encoder.num_patches, generator=generator))
    finally:
        torch.set_rng_state(original)
    return torch.stack(views), torch.stack(noises)


def _quantiles(values: torch.Tensor, prefix: str) -> dict[str, float]:
    """Small CPU quantile summary suitable for strict JSON."""
    values = values.detach().float().cpu()
    return {
        prefix + "_mean": float(values.mean()),
        **{prefix + f"_q{q}": float(torch.quantile(values, q / 100)) for q in (10, 50, 90)},
    }


def _gradient_norm(parameters: Iterable[torch.Tensor]) -> float:
    """Read the unclipped L2 norm, including non-finite values for failure detection."""
    squares = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


class TrainingReadout:
    """Capture gate attenuation and actual masked expert assignments in the training forward."""

    def __init__(self, method: MAE) -> None:
        """Install one small observer per MoE layer; evaluation never uses these hooks."""
        self.rows: dict[str, Any] = {}
        self.counts: dict[str, list[int]] = {}
        self.handles = []
        for layer, block in enumerate(method.encoder.blocks):
            if not isinstance(block.mlp, MoEMlp):
                continue

            def capture(
                _module: torch.nn.Module, _args: tuple[torch.Tensor], output: FeedForwardOutput, index: int = layer
            ) -> None:
                if output.routing_probs is None:
                    raise RuntimeError("MoE training forward omitted routing probabilities.")
                p = output.routing_probs.detach()[:, 1:]
                selected = p.max(-1).values
                weighted = output.features.detach()[:, 1:].norm(dim=-1)
                counts = torch.bincount(p.argmax(-1).flatten(), minlength=p.shape[-1]).cpu().tolist()
                prefix = f"l{index}"
                self.rows.update(
                    {
                        prefix + "_selected_probability": float(selected.mean()),
                        prefix + "_weighted_branch_norm": float(weighted.mean()),
                        prefix + "_expert_output_norm": float((weighted / selected.clamp_min(1e-12)).mean()),
                    }
                )
                self.counts[prefix] = counts

            self.handles.append(block.mlp.register_forward_hook(capture))

    def close(self) -> None:
        """Remove observers even after a failed update."""
        for handle in self.handles:
            handle.remove()


@torch.no_grad()  # type: ignore[misc]
def _encode(method: MAE, images: torch.Tensor, ids: list[int], batch_size: int = 64) -> torch.Tensor:
    """Encode fixed images in bounded batches, keeping features on CPU."""
    return torch.cat(
        [method.encode(images[ids[start : start + batch_size]]).cpu() for start in range(0, len(ids), batch_size)]
    )


@torch.no_grad()  # type: ignore[misc]
def _reconstruction(method: MAE, corpus: Corpus, ids: list[int], cfg: DictConfig) -> tuple[float, list[float]]:
    """Mean per-image reconstruction over fixed per-image mask/augmentation draws."""
    values = []
    for draw in range(int(cfg.mask_draws)):
        current = []
        for start in range(0, len(ids), 64):
            selected = torch.tensor(ids[start : start + 64])
            views, noise = paired_views(method, corpus.images[selected], selected, 910_000, draw)
            losses, mask, _ = method._masked_recon_loss(views.to(cfg.device), augmented=True, mask_noise=noise)
            current.append(((losses * mask).sum(-1) / mask.sum(-1)).cpu())
        values.append(torch.cat(current))
    per_image = torch.stack(values).mean(0)
    return float(per_image.mean()), per_image.tolist()


@torch.no_grad()  # type: ignore[misc]
def _routing(method: MAE, images: torch.Tensor) -> dict[int, torch.Tensor]:
    """Collect deterministic full routing in bounded batches on the CPU."""
    encoder = cast(MoEViTEncoder, method.encoder)
    batches = [encoder.routing(batch) for batch in images.split(64)]
    return {layer: torch.cat([batch[layer].cpu() for batch in batches]) for layer in batches[0]}


def evaluate(
    method: MAE,
    corpus: Corpus,
    panels: dict[str, list[int]],
    cfg: DictConfig,
    step: int,
    reference: torch.Tensor | None,
    previous: dict[int, torch.Tensor] | None,
    initial_routes: dict[int, torch.Tensor] | None,
    probes: bool,
) -> tuple[dict[str, Any], dict[int, torch.Tensor], torch.Tensor, dict[str, np.ndarray]]:
    """Read identical reference panels, restoring modes and RNG regardless of failures."""
    modes = [(module, module.training) for module in method.modules()]
    method.eval()
    try:
        with isolated_rng(900_000, str(cfg.device)), torch.no_grad():
            routing = _routing(method, corpus.images[panels["query"]])
            entropy = measurements.routing_entropy(routing)
            features = _encode(method, corpus.images, panels["query"])
            unit_features = torch.nn.functional.normalize(features, dim=1)
            routes = {layer: p[:, 1:].argmax(-1) for layer, p in routing.items()}
            natural_mse, errors = _reconstruction(method, corpus, panels["natural"], cfg)
            balanced_mse, balanced_errors = _reconstruction(method, corpus, panels["query"], cfg)
            metrics: dict[str, Any] = {
                "step": step,
                "heldout_mse": natural_mse,
                "balanced_mse": balanced_mse,
                "rankme": measurements.rankme(features),
                "mean_feature_var": float(measurements.feature_variance(unit_features).mean()),
                "offdiag_cov": measurements.offdiag_covariance(unit_features),
                **_quantiles(entropy.image_normalized, "entropy"),
                "entropy_nats": float(entropy.image_nats.mean()),
            }
            correlation = float(spearmanr(entropy.image_normalized.numpy(), balanced_errors).statistic)
            metrics["entropy_reconstruction_spearman"] = correlation if math.isfinite(correlation) else None
            metrics["cka_drift"] = 0.0 if reference is None else measurements.cka_drift(reference, features)
            metrics["cosine_drift"] = 0.0 if reference is None else measurements.cosine_drift(reference, features)
            arrays = {
                "token_entropy": entropy.token_normalized.numpy(),
                "image_entropy": entropy.image_normalized.numpy(),
                "natural_reconstruction": np.array(errors),
                "features": features.numpy(),
            }
            for layer, p in routing.items():
                patch = p[:, 1:]
                count = torch.bincount(routes[layer].flatten(), minlength=p.shape[-1]).float()
                load = count / count.sum()
                prefix = f"l{layer}"
                metrics.update(
                    {
                        prefix + "_entropy": float(entropy.layer_image_normalized[layer].mean()),
                        prefix + "_load": load.tolist(),
                        prefix + "_soft_mass": patch.mean((0, 1)).tolist(),
                        prefix + "_load_entropy": float(
                            -(load * load.clamp_min(1e-12).log()).sum() / math.log(len(load))
                        ),
                        prefix + "_load_cv": float(load.std(unbiased=False) / load.mean()),
                        prefix + "_minimum_load": float(load.min()),
                        prefix + "_dead_fraction": float((load == 0).float().mean()),
                        prefix + "_route_change_previous": 0.0
                        if previous is None
                        else float((routes[layer] != previous[layer]).float().mean()),
                        prefix + "_route_change_initial": 0.0
                        if initial_routes is None
                        else float((routes[layer] != initial_routes[layer]).float().mean()),
                    }
                )
                arrays[prefix + "_routes"] = routes[layer].numpy()
            if probes:
                support = _encode(method, corpus.images, panels["support"])
                natural = _encode(method, corpus.images, panels["natural"])
                for name, queries, ids in (
                    ("balanced", features, panels["query"]),
                    ("natural", natural, panels["natural"]),
                ):
                    support_set = (support, corpus.labels[panels["support"]])
                    query_set = (queries, corpus.labels[ids])
                    metrics[name + "_knn"] = measurements.knn_probe(lambda x: x, support_set, query_set, k=20)
                    metrics[name + "_linear"] = measurements.linear_probe(lambda x: x, support_set, query_set)
            return metrics, routes, features, arrays
    finally:
        for module, training in modes:
            module.training = training


def mask_sensitivity(method: MAE, corpus: Corpus, ids: list[int], cfg: DictConfig) -> dict[str, Any]:
    """Compare full-image rankings with repeated clean masked rankings on identical candidates."""
    encoder = cast(MoEViTEncoder, method.encoder)
    positions = torch.linspace(0, len(ids) - 1, min(int(cfg.batch_size), len(ids))).long().tolist()
    chosen = [ids[position] for position in positions]
    images = corpus.images[chosen].to(cfg.device)
    full = measurements.routing_entropy(encoder.routing(images)).image_normalized.cpu()
    masked = []
    for draw in range(max(4, int(cfg.mask_draws))):
        with isolated_rng(keyed_seed("sensitivity", draw), str(cfg.device)), torch.no_grad():
            result = encoder._encode(images, mask_ratio=method.mask_ratio, collect_routing=True, deterministic=True)
            masked.append(measurements.routing_entropy(result.routing_probs).image_normalized.cpu())
    average = torch.stack(masked).mean(0)
    keep = len(chosen) // 2
    correlations = [float(spearmanr(full.numpy(), values.numpy()).statistic) for values in [*masked, average]]
    if not all(math.isfinite(v) for v in correlations):
        return {"defined": False, "reason": "constant entropy ranks", "draws": len(masked), "sample_ids": chosen}
    overlaps = {}
    for high in (True, False):
        selected = set(full.argsort(descending=high)[:keep].tolist())
        overlaps["high" if high else "low"] = [
            len(selected & set(values.argsort(descending=high)[:keep].tolist())) / keep for values in [*masked, average]
        ]
    return {
        "defined": True,
        "sample_ids": chosen,
        "draws": len(masked),
        "spearman_each_then_mean": correlations,
        "half_overlap_each_then_mean": overlaps,
    }


def _selection_json(record: SelectionRecord, corpus: Corpus) -> dict[str, Any]:
    """Attach attributes only after a decision has been made, for later diet audits."""
    scores = record.entropy.image_normalized.cpu()
    accepted = record.accepted_indices
    reject_mask = torch.ones(len(scores), dtype=torch.bool)
    reject_mask[accepted] = False
    accepted_mask = ~reject_mask
    groups: dict[str, Any] = {}
    for name, labels in (("scene", corpus.labels), ("regime", corpus.regimes)):
        candidate = labels[record.sample_ids]
        groups[name] = {
            str(group): {
                "candidate": int((candidate == group).sum()),
                "accepted": int(((candidate == group) & accepted_mask).sum()),
            }
            for group in candidate.unique().tolist()
        }
    return {
        "step": record.step,
        "sample_ids": record.sample_ids.tolist(),
        "accepted_indices": accepted.tolist(),
        "entropy": scores.tolist(),
        "entropy_nats": record.entropy.image_nats.cpu().tolist(),
        **_quantiles(scores, "candidate"),
        **_quantiles(scores[accepted], "accepted"),
        **_quantiles(scores[reject_mask], "rejected"),
        "groups": groups,
    }


def _atomic_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Replace the latest checkpoint only once all state has been serialized."""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run_arm(  # noqa: C901 - one checkpoint transaction owns training, evaluation, and log offsets
    cfg: DictConfig,
    corpus: Corpus,
    panels: dict[str, list[int]],
    seed: int,
    arm: str,
    directory: Path,
    protocol_hash: str,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Train or resume one arm; step-keyed noise plus selector/optimizer state preserve continuation."""
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "complete.json"
    if complete.exists():
        result = json.loads(complete.read_text())
        if result["protocol_hash"] != protocol_hash:
            raise ValueError("Completed run belongs to a different protocol.")
        return cast(dict[str, Any], result)
    device = str(cfg.device)
    with isolated_rng(keyed_seed("initialization", seed), device):
        method = cast(MAE, instantiate(cfg.ssl, encoder=instantiate(cfg.encoder)).to(device))
    if not isinstance(method.encoder, MoEViTEncoder) or any(
        isinstance(m, MoEMlp) and m.top_k != 1 for m in method.encoder.modules()
    ):
        raise ValueError("This experiment requires a top-1 MoE encoder.")
    initial_hash = state_hash(method.state_dict())
    optimizer = instantiate(cfg.optim, params=method.parameters())
    selector_cfg = OmegaConf.create(OmegaConf.to_container(cfg.selectors[arm], resolve=True))
    selector_cfg.seed = keyed_seed("admission", seed)
    selector = cast(RoutingEntropyHalf, instantiate(selector_cfg))
    per_epoch = len(panels["train"]) // int(cfg.batch_size)
    total_steps = int(cfg.epochs) * per_epoch
    if cfg.max_steps is not None:
        total_steps = min(total_steps, int(cfg.max_steps))
    if total_steps < 1:
        raise ValueError("Training budget contains no complete candidate batches.")
    probe_steps = {0, total_steps, round(total_steps / 4), round(total_steps / 2)}
    log_paths = {name: directory / f"{name}.jsonl" for name in ("training", "selection", "evaluation")}
    step, reference, previous, initial_routes = 0, None, None, None
    counts_history: list[dict[str, list[int]]] = []
    unique: set[int] = set()
    elapsed = {"scoring": 0.0, "preparation": 0.0, "update": 0.0, "evaluation": 0.0}
    memory_peaks = {"allocator_sampled_bytes": 0, "driver_sampled_bytes": 0}
    checkpoint_path = directory / "latest.pt"
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state["protocol_hash"] != protocol_hash or state["initial_hash"] != initial_hash:
            raise ValueError("Refusing to resume incompatible data/config/code or initialization.")
        method.load_state_dict(state["method"])
        optimizer.load_state_dict(state["optimizer"])
        selector.load_state_dict(state["selector"])
        step, reference = state["step"], state["reference"].cpu()
        previous = {k: v.cpu() for k, v in state["previous"].items()}
        initial_routes = {k: v.cpu() for k, v in state["initial_routes"].items()}
        counts_history, unique, elapsed = state["counts_history"], set(state["unique"]), state["elapsed"]
        memory_peaks = state["memory_peaks"]
        for name, path in log_paths.items():
            with path.open("r+b") as binary_log:
                binary_log.truncate(state["log_sizes"][name])
    elif any(path.exists() for path in log_paths.values()):
        raise ValueError("Logs exist without a resumable checkpoint; use a fresh output directory.")
    logs = {name: path.open("a") for name, path in log_paths.items()}

    def append(name: str, row: dict[str, Any]) -> None:
        logs[name].write(json.dumps(row, allow_nan=False) + "\n")
        logs[name].flush()

    def save() -> None:
        for log in logs.values():
            log.flush()
        _atomic_checkpoint(
            checkpoint_path,
            {
                "protocol_hash": protocol_hash,
                "gate_semantics": GATE_SEMANTICS,
                "method": method.state_dict(),
                "optimizer": optimizer.state_dict(),
                "selector": selector.state_dict(),
                "step": step,
                "initial_hash": initial_hash,
                "reference": reference,
                "previous": previous,
                "initial_routes": initial_routes,
                "counts_history": counts_history,
                "unique": sorted(unique),
                "elapsed": elapsed,
                "memory_peaks": memory_peaks,
                "log_sizes": {name: path.stat().st_size for name, path in log_paths.items()},
                "cpu_rng": torch.get_rng_state(),
                "accelerator_rng": torch.mps.get_rng_state() if device == "mps" else None,
            },
        )

    def evaluate_step() -> dict[str, Any]:
        nonlocal reference, previous, initial_routes
        synchronize(device)
        started = time.perf_counter()
        metrics, current, features, arrays = evaluate(
            method, corpus, panels, cfg, step, reference, previous, initial_routes, step in probe_steps
        )
        if reference is None:
            reference = features.clone()
            initial_routes = {k: v.clone() for k, v in current.items()}
        previous = current
        if step in probe_steps:
            metrics["mask_sensitivity"] = mask_sensitivity(method, corpus, panels["query"], cfg)
        if counts_history:
            for layer in counts_history[0]:
                rolling = torch.tensor([r[layer] for r in counts_history]).sum(0)
                metrics[layer + "_rolling_starved_fraction"] = float((rolling == 0).float().mean())
        synchronize(device)
        elapsed["evaluation"] += time.perf_counter() - started
        np.savez_compressed(directory / f"panel_{step:06d}.npz", **arrays)
        append("evaluation", metrics)
        logger.info(
            "seed={} arm={} step={}/{} entropy={:.4f} mse={:.4f}",
            seed,
            arm,
            step,
            total_steps,
            metrics["entropy_mean"],
            metrics["heldout_mse"],
        )
        return metrics

    started_wall = time.perf_counter()
    observer = TrainingReadout(method)
    # Initial evaluation must run without the training-only observer.
    observer.close()
    try:
        if not checkpoint_path.exists():
            evaluate_step()
            save()
            shutil.copyfile(checkpoint_path, directory / "initial.pt")
        order = torch.empty(0, dtype=torch.long)
        current_epoch = -1
        while step < total_steps:
            epoch, offset = divmod(step, per_epoch)
            if epoch != current_epoch:
                order, current_epoch = epoch_order(panels["train"], seed, epoch), epoch
            ids = order[offset * int(cfg.batch_size) : (offset + 1) * int(cfg.batch_size)]
            images = corpus.images[ids]
            synchronize(device)
            started = time.perf_counter()
            accepted = selector.select(StreamBatch(images.to(device), 0, step, ids), FilterContext(method, step))
            del accepted
            synchronize(device)
            elapsed["scoring"] += time.perf_counter() - started
            record = selector.last_record
            if record is None:
                raise RuntimeError("Selector omitted its audit record.")
            selected_ids = ids[record.accepted_indices]
            unique.update(selected_ids.tolist())
            started = time.perf_counter()
            views, noise = paired_views(method, corpus.images[selected_ids], selected_ids, seed, epoch)
            elapsed["preparation"] += time.perf_counter() - started
            method.train()
            optimizer.zero_grad(set_to_none=True)
            observer = TrainingReadout(method)
            synchronize(device)
            started = time.perf_counter()
            try:
                with isolated_rng(keyed_seed("router_noise", seed, step), device):
                    losses = method.loss_components(views.to(device), augmented=True, mask_noise=noise)
                    losses.total.backward()
                grad_norm = _gradient_norm(method.parameters())
                if not math.isfinite(float(losses.total.detach())) or not math.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite loss/gradient; optimizer update was not applied.")
                gradients: dict[str, Any] = {}
                for layer, block in enumerate(method.encoder.blocks):
                    if isinstance(block.mlp, MoEMlp):
                        gradients[f"l{layer}_router_grad_norm"] = _gradient_norm(block.mlp.router.parameters())
                        gradients[f"l{layer}_expert_grad_norm"] = [
                            _gradient_norm(e.parameters()) for e in block.mlp.experts
                        ]
                torch.nn.utils.clip_grad_norm_(method.parameters(), float(cfg.grad_clip))
                optimizer.step()
                synchronize(device)
            finally:
                observer.close()
            elapsed["update"] += time.perf_counter() - started
            if device == "mps":
                memory_peaks["allocator_sampled_bytes"] = max(
                    memory_peaks["allocator_sampled_bytes"], torch.mps.current_allocated_memory()
                )
                memory_peaks["driver_sampled_bytes"] = max(
                    memory_peaks["driver_sampled_bytes"], torch.mps.driver_allocated_memory()
                )
            counts_history = (counts_history + [observer.counts])[-200:]
            step += 1
            append("selection", {**_selection_json(record, corpus), "epoch": epoch})
            append(
                "training",
                {
                    "step": step,
                    "epoch": epoch,
                    "candidate_order_hash": tensor_hash(ids),
                    "views_hash": tensor_hash(views),
                    "mask_noise_hash": tensor_hash(noise),
                    "loss": float(losses.total.detach()),
                    "reconstruction": float(losses.reconstruction.detach()),
                    "auxiliary": float(losses.auxiliary.detach()),
                    "weighted_auxiliary": float(losses.auxiliary.detach()) * method.aux_weight,
                    "grad_norm": grad_norm,
                    "lr": optimizer.param_groups[0]["lr"],
                    "candidate_count": len(ids),
                    "accepted_count": len(selected_ids),
                    "train_counts": observer.counts,
                    **observer.rows,
                    **gradients,
                },
            )
            if step % int(cfg.eval_every) == 0 or step in probe_steps or step == total_steps or step == stop_after:
                evaluate_step()
                save()
                if step in probe_steps:
                    shutil.copyfile(checkpoint_path, directory / f"checkpoint_{step:06d}.pt")
            if step == stop_after:
                return {"interrupted": True, "step": step}
        final = json.loads(log_paths["evaluation"].read_text().splitlines()[-1])
        first = json.loads(log_paths["evaluation"].read_text().splitlines()[0])
        result = {
            "protocol_hash": protocol_hash,
            "seed": seed,
            "arm": arm,
            "initial_hash": initial_hash,
            "gate_semantics": GATE_SEMANTICS,
            "steps": step,
            "epochs": float(step / per_epoch),
            "candidate_exposures": step * int(cfg.batch_size),
            "accepted_exposures": step * int(cfg.batch_size) // 2,
            "unique_accepted": len(unique),
            "train_pool": len(panels["train"]),
            "dropped_per_complete_epoch": len(panels["train"]) % int(cfg.batch_size),
            "elapsed_seconds": elapsed,
            "memory_peaks": memory_peaks,
            "this_invocation_wall_seconds": time.perf_counter() - started_wall,
            "rss_process_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
            "initial": first,
            "final": final,
            "entropy_change": final["entropy_mean"] - first["entropy_mean"],
            "peak_memory_note": "RSS is a sequential-process lifetime peak, not an independent per-arm peak",
        }
        if device == "mps":
            result["mps_allocator_bytes_at_end"] = torch.mps.current_allocated_memory()
            result["mps_driver_bytes_at_end"] = torch.mps.driver_allocated_memory()
        write_json(complete, result)
        return result
    finally:
        observer.close()
        for log in logs.values():
            log.close()


def _validate_config(cfg: DictConfig) -> None:
    """Fail before any training on malformed budgets or an unavailable accelerator."""
    if int(cfg.batch_size) < 4 or int(cfg.batch_size) % 2 or int(cfg.epochs) < 1 or int(cfg.eval_every) < 1:
        raise ValueError("Require even candidate batches >=4 and positive epochs/evaluation cadence.")
    seeds = [int(seed) for seed in cfg.seeds]
    if not seeds or len(set(seeds)) != len(seeds) or int(cfg.mask_draws) < 1:
        raise ValueError("Require unique training seeds and at least one fixed evaluation mask draw.")
    if str(cfg.device) == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this execution context; request local GPU access or select CPU.")


def run_experiment(cfg: DictConfig, output: Path) -> dict[str, Any]:
    """Validate/freeze provenance, run all three paired arms, and produce a seed-level comparison."""
    _validate_config(cfg)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(cfg.threads))
    corpus = cast(Corpus, instantiate(cfg.source))
    if tuple(corpus.images.shape[2:]) != (int(cfg.img_size), int(cfg.img_size)):
        raise ValueError("Cached/source resolution does not match the encoder config.")
    panels = reserve_panels(corpus, cfg)
    config = OmegaConf.to_container(cfg, resolve=True)
    source_files = [
        Path(__file__),
        *Path(__file__).parent.joinpath("models").glob("*.py"),
        *Path(__file__).parent.joinpath("ssl").glob("*.py"),
        Path(__file__).with_name("measurements.py"),
        Path(__file__).parent / "filters" / "routing_entropy.py",
        Path(__file__).parent / "filters" / "base.py",
        Path(__file__).parent / "data" / "streams.py",
        Path(__file__).parent / "data" / "attributes.py",
        Path(__file__).with_name("health_trust.py"),
    ]
    source_hashes = {str(path.relative_to(Path(__file__).parent)): digest_file(path) for path in source_files}
    protocol = {
        "config": config,
        "corpus": corpus.fingerprint,
        "panels": panels,
        "gate_semantics": GATE_SEMANTICS,
        "source_hashes": source_hashes,
        "runtime": {"torch": torch.__version__, "python": platform.python_version()},
        "primary_outcomes": ["fixed_panel_entropy_change", "natural_heldout_mse"],
        "practical_thresholds": {"normalized_entropy_absolute": 0.02, "relative_mse": 0.05, "probe_accuracy_pp": 3.0},
        "inference": "paired seed effects; descriptive 95% t intervals, no confirmatory significance claims",
    }
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())["protocol_hash"] != protocol_hash:
        raise ValueError("Output contains a different frozen protocol. Use a fresh directory.")
    write_json(protocol_path, {**protocol, "protocol_hash": protocol_hash})
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    write_json(
        output / "membership.json",
        {
            **panels,
            "names": corpus.names,
            "scene_names": corpus.scene_names,
            "regime_names": corpus.regime_names,
            "labels": corpus.labels.tolist(),
            "regimes": corpus.regimes.tolist(),
        },
    )
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=True)
    for path in source_files:
        dest = provenance / path.relative_to(Path(__file__).parent)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    write_json(
        provenance / "runtime.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(cfg.device),
            "threads": torch.get_num_threads(),
            "git_head": subprocess.check_output(  # noqa: S603 - fixed read-only git command, no user input
                [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
    )
    (provenance / "tracked.patch").write_text(
        subprocess.check_output(  # noqa: S603 - fixed read-only git command, no user input
            [shutil.which("git") or "/usr/bin/git", "diff"], text=True
        )
    )
    trust = annotate(["rankme", "mean_feature_var", "offdiag_cov", "cka_drift", "cosine_drift"], Backbone.MAE)
    trust["routing_entropy"] = [
        {
            "status": "candidate",
            "provenance": "MoE routing entropy filtering study",
            "caveat": "routing ambiguity has no presumed healthy direction",
        }
    ]
    write_json(output / "trust.json", trust)
    results = []
    for seed in cfg.seeds:
        paired = []
        for arm in ARMS:
            result = run_arm(cfg, corpus, panels, int(seed), arm, output / f"seed_{seed}" / arm, protocol_hash)
            paired.append(result)
            results.append(result)
        if len({r["initial_hash"] for r in paired}) != 1:
            raise AssertionError("Arms did not start from identical random encoder/decoder weights.")
        orders = [
            [
                json.loads(line)["candidate_order_hash"]
                for line in (output / f"seed_{seed}" / arm / "training.jsonl").read_text().splitlines()
            ]
            for arm in ARMS
        ]
        if not orders[0] == orders[1] == orders[2]:
            raise AssertionError("Candidate image order differs across paired arms.")
    comparison = summarize(results)
    write_json(output / "comparison.json", comparison)
    make_figures(output, results)
    return comparison


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute paired seed effects, treating seeds as the independent replicates."""
    contrasts = {}
    for arm in ("high", "low"):
        paired = []
        for row in results:
            if row["arm"] != arm:
                continue
            baseline = next(r for r in results if r["arm"] == "random" and r["seed"] == row["seed"])
            paired.append(
                {
                    "seed": row["seed"],
                    "entropy_change_difference": row["entropy_change"] - baseline["entropy_change"],
                    "relative_mse_difference": row["final"]["heldout_mse"] / baseline["final"]["heldout_mse"] - 1,
                    "balanced_knn_pp": 100 * (row["final"]["balanced_knn"] - baseline["final"]["balanced_knn"]),
                    "balanced_linear_pp": 100
                    * (row["final"]["balanced_linear"] - baseline["final"]["balanced_linear"]),
                }
            )
        intervals = {}
        for metric in paired[0]:
            if metric == "seed":
                continue
            values = np.array([r[metric] for r in paired])
            half = (
                float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else None
            )
            mean = float(values.mean())
            intervals[metric] = {"mean": mean, "ci95": [mean - half, mean + half] if half is not None else None}
        contrasts[arm + "_minus_random"] = {"paired_seeds": paired, "effects": intervals}
    return {
        "runs": results,
        "contrasts": contrasts,
        "replicate_unit": "training seed",
        "pairing": "identical initial state and candidate order verified per seed",
        "interval_note": (
            "descriptive paired t intervals; five seeds give limited precision; no multiplicity-adjusted testing"
        ),
    }


def make_figures(output: Path, results: list[dict[str, Any]]) -> None:
    """Save scientific plots and a compact results table without a GUI dependency."""
    import matplotlib  # noqa: PLC0415 - plotting is optional for training/library imports

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    metrics = (
        ("entropy_mean", "Fixed-panel normalized routing entropy"),
        ("heldout_mse", "Natural held-out reconstruction MSE"),
        ("rankme", "Fixed-panel RankMe"),
        ("balanced_linear", "Balanced scene linear-probe accuracy"),
    )
    colors = {"random": "#566573", "high": "#c0392b", "low": "#2471a3"}
    for arm in ARMS:
        rows = [r for r in results if r["arm"] == arm]
        traces = [
            [
                json.loads(line)
                for line in (output / f"seed_{r['seed']}" / arm / "evaluation.jsonl").read_text().splitlines()
            ]
            for r in rows
        ]
        for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
            steps = sorted({row["step"] for trace in traces for row in trace if metric in row})
            values = np.array(
                [[next(row[metric] for row in trace if row["step"] == step) for step in steps] for trace in traces]
            )
            mean = values.mean(0)
            ax.plot(steps, mean, label=arm, color=colors[arm])
            if len(values) > 1:
                sem = values.std(0, ddof=1) / math.sqrt(len(values))
                ax.fill_between(steps, mean - sem, mean + sem, color=colors[arm], alpha=0.15)
            ax.set(title=title, xlabel="Optimizer updates")
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.savefig(output / "trajectories.png", dpi=160)
    plt.close(fig)
    lines = [
        "# MoE routing entropy filtering results",
        "",
        "Means across training seeds; trajectory shading is ±1 SEM.",
        "",
        "| Arm | Seeds | Entropy change | Held-out MSE | kNN (%) | Linear (%) | Scoring/update seconds |",
        "| -- | --: | --: | --: | --: | --: | --: |",
    ]
    for arm in ARMS:
        rows = [r for r in results if r["arm"] == arm]
        mean = {
            key: float(np.mean([r["final"][key] for r in rows]))
            for key in ("heldout_mse", "balanced_knn", "balanced_linear")
        }
        score_seconds = np.mean([r["elapsed_seconds"]["scoring"] for r in rows])
        update_seconds = np.mean([r["elapsed_seconds"]["update"] for r in rows])
        lines.append(
            f"| {arm} | {len(rows)} | {np.mean([r['entropy_change'] for r in rows]):+.4f} | "
            f"{mean['heldout_mse']:.4f} | {100 * mean['balanced_knn']:.1f} | {100 * mean['balanced_linear']:.1f} | "
            f"{score_seconds:.1f} / {update_seconds:.1f} |"
        )
    lines += [
        "",
        "The three arms use identical initialization and candidate order within each seed.",
        "See comparison.json for paired seed outcomes and descriptive uncertainty intervals.",
        "",
        "Entropy and drift are candidate diagnostics. Their direction alone does not establish health or forgetting.",
        "The primary reconstruction panel has natural frequencies; scene probes also report balanced queries.",
        "",
        "![Trajectories](trajectories.png)",
    ]
    (output / "results.md").write_text("\n".join(lines) + "\n")
