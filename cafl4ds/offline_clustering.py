"""WP-M6.0.0: frozen image embeddings and an offline, all-data Gaussian mixture.

This is deliberately a look-ahead baseline: every discovered image, including validation
and test images, contributes to both PCA and GMM fitting. It is independent of the MoE.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import sklearn
import torch
import torchvision
from loguru import logger
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ViT_B_16_Weights, vit_b_16

WEIGHTS = ViT_B_16_Weights.IMAGENET1K_V1
EMBED_DIM = 768


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def discover_images(bdd_root: Path) -> list[dict[str, Any]]:
    """Enumerate all still-image splits without requiring or filtering on annotations."""
    records = []
    for split in ("train", "val", "test"):
        directory = bdd_root / "images" / "100k" / split
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing BDD image split: {directory}")
        paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not paths:
            raise ValueError(f"No images found in {directory}")
        for path in paths:
            stat = path.stat()
            records.append(
                {
                    "path": path.relative_to(bdd_root).as_posix(),
                    "split": split,
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return records


class _Images:
    def __init__(self, root: Path, records: list[dict[str, Any]]) -> None:
        self.root = root
        self.records = records
        self.transform = WEIGHTS.transforms()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.root / self.records[index]["path"]
        with Image.open(path) as image:
            return cast(torch.Tensor, self.transform(image.convert("RGB")))


def build_encoder(device: str) -> nn.Module:
    """Load explicit ImageNet weights and return frozen, evaluation-mode CLS features."""
    model: nn.Module = vit_b_16(weights=WEIGHTS)
    model.heads = nn.Identity()
    return model.requires_grad_(False).eval().to(device)


def _embedding_cache(output: Path, fingerprint: str, n_images: int) -> tuple[np.memmap, int]:
    path = output / "embeddings.npy"
    progress_path = output / "embedding_progress.json"
    completed = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["fingerprint"] != fingerprint:
            raise ValueError("Embedding progress does not match the image manifest.")
        completed = int(progress["completed"])
        if not 0 <= completed <= n_images:
            raise ValueError("Invalid embedding progress.")
    if completed and not path.exists():
        raise FileNotFoundError("Embedding progress exists but the array is missing.")
    features = np.lib.format.open_memmap(
        path, mode="r+" if path.exists() else "w+", dtype="float32", shape=(n_images, EMBED_DIM)
    )
    if features.shape != (n_images, EMBED_DIM) or features.dtype != np.float32:
        raise ValueError("Embedding cache has an unexpected shape or dtype.")
    return features, completed


def extract_embeddings(
    bdd_root: Path,
    output: Path,
    *,
    device: str = "auto",
    batch_size: int = 32,
    workers: int = 4,
) -> dict[str, Any]:
    """Stream original JPEGs into a resumable float32, row-aligned embedding cache.

    Cache identity includes file sizes/mtimes, exact weights, preprocessing and software
    versions. A decoding failure stops extraction rather than silently dropping an image.
    Progress is committed only after the corresponding array rows have been flushed.
    """
    output.mkdir(parents=True, exist_ok=True)
    records = discover_images(bdd_root)
    identity = {
        "images": records,
        "encoder": "torchvision.models.vit_b_16",
        "weights": WEIGHTS.name,
        "weights_url": WEIGHTS.url,
        "transform": repr(WEIGHTS.transforms()),
        "embedding": "final CLS token, before classifier; float32; no L2 normalization",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Embedding cache inputs changed. Use a new output directory.")
    else:
        manifest = {"fingerprint": fingerprint, "bdd_root": str(bdd_root.resolve()), **identity}
        _write_json(manifest_path, manifest)

    progress_path = output / "embedding_progress.json"
    features, completed = _embedding_cache(output, fingerprint, len(records))
    if completed == len(records):
        logger.info("Reusing all {} cached embeddings", completed)
        return manifest

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Metal was requested but MPS is unavailable in this process.")
    model = build_encoder(device)
    dataset = cast(Dataset[torch.Tensor], _Images(bdd_root, records))
    loader = DataLoader(
        Subset(dataset, range(completed, len(records))),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        persistent_workers=workers > 0,
        multiprocessing_context="spawn" if workers else None,
    )
    start, initial, last_log = time.monotonic(), completed, time.monotonic()
    logger.info("Extracting {} images on {} (batch {}, workers {})", len(records), device, batch_size, workers)
    with torch.inference_mode():
        for images in loader:
            encoded = model(images.to(device)).cpu().numpy()
            if encoded.shape != (len(images), EMBED_DIM) or not np.isfinite(encoded).all():
                raise ValueError("Encoder returned invalid embeddings.")
            features[completed : completed + len(images)] = encoded
            features.flush()
            completed += len(images)
            _write_json(
                progress_path,
                {
                    "fingerprint": fingerprint,
                    "completed": completed,
                    "total": len(records),
                    "device": device,
                    "batch_size": batch_size,
                    "workers": workers,
                    "session_seconds": time.monotonic() - start,
                },
            )
            now = time.monotonic()
            if now - last_log >= 30 or completed == len(records):
                rate = (completed - initial) / (now - start)
                logger.info(
                    "{}/{} images; {:.1f} images/s; {:.1f} min remaining",
                    completed,
                    len(records),
                    rate,
                    (len(records) - completed) / rate / 60,
                )
                last_log = now
    return manifest


def fit_clusters(
    output: Path,
    *,
    pca_dim: int = 64,
    n_init: int = 5,
    seed: int = 0,
    max_iter: int = 300,
) -> dict[str, Any]:
    """Fit PCA/GMM to every cached embedding, save inference and inspection artifacts."""
    manifest = json.loads((output / "manifest.json").read_text())
    progress = json.loads((output / "embedding_progress.json").read_text())
    records = manifest["images"]
    if progress["fingerprint"] != manifest["fingerprint"] or progress["completed"] != len(records):
        raise ValueError("Complete image embeddings are required before fitting.")
    features = np.load(output / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if features.shape != (len(records), EMBED_DIM) or not np.isfinite(features).all():
        raise ValueError("The embedding array does not match the manifest or contains non-finite values.")
    if not 2 <= pca_dim <= min(features.shape):
        raise ValueError("PCA dimension must be between 2 and min(n_images, embedding_dim).")

    started = time.monotonic()
    logger.info("Fitting non-whitened PCA to all {} images: {} -> {} dimensions", len(records), EMBED_DIM, pca_dim)
    pca = PCA(n_components=pca_dim, whiten=False, svd_solver="randomized", random_state=seed)
    pca.fit(features)
    reduced = pca.transform(features)
    np.save(output / "embeddings_64.npy", reduced, allow_pickle=False)
    logger.info(
        "PCA retains {:.2%} variance; fitting four full-covariance Gaussians", pca.explained_variance_ratio_.sum()
    )
    gmm = GaussianMixture(
        n_components=4,
        covariance_type="full",
        n_init=n_init,
        max_iter=max_iter,
        reg_covar=1e-6,
        random_state=seed,
        verbose=2,
        verbose_interval=10,
    ).fit(reduced)
    if not gmm.converged_:
        raise RuntimeError("GMM did not converge; no finished model has been saved.")
    probabilities = gmm.predict_proba(reduced)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError("GMM returned invalid component probabilities.")
    labels = probabilities.argmax(axis=1)
    pipeline = Pipeline([("pca", pca), ("gmm", gmm)])
    with (output / "pca_gmm.pkl").open("wb") as stream:
        pickle.dump(pipeline, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(output / "probabilities.npy", probabilities.astype(np.float32), allow_pickle=False)
    np.save(output / "labels.npy", labels.astype(np.uint8), allow_pickle=False)

    rng = np.random.default_rng(seed)
    exemplars = []
    for cluster in range(gmm.n_components):
        members = np.flatnonzero(labels == cluster)
        # Mahalanobis distance in the actual 64-D fit space, restricted to assigned members.
        standardized = (reduced[members] - gmm.means_[cluster]) @ gmm.precisions_cholesky_[cluster]
        distance = np.einsum("ij,ij->i", standardized, standardized)
        central = members[np.argsort(distance, kind="stable")[:6]]
        random = rng.choice(members, size=min(6, len(members)), replace=False)
        exemplars.append({"cluster": cluster, "central": central.tolist(), "random": random.tolist()})
    _write_json(output / "exemplars.json", exemplars)
    entropy = -(probabilities * np.log(np.maximum(probabilities, np.finfo(probabilities.dtype).tiny))).sum(axis=1)
    summary = {
        "experiment": "WP-M6.0.0",
        "scope": "Offline all-data look-ahead baseline; includes train, validation and test. No held-out fit data.",
        "n_images": len(records),
        "splits": dict(Counter(row["split"] for row in records)),
        "embedding_fingerprint": manifest["fingerprint"],
        "embedding_dim": EMBED_DIM,
        "pca_dim": pca_dim,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "pca_2d_explained_variance": float(pca.explained_variance_ratio_[:2].sum()),
        "gmm_parameters": gmm.get_params(),
        "gmm_converged": bool(gmm.converged_),
        "gmm_iterations_best_initialization": int(gmm.n_iter_),
        "mean_log_likelihood": float(gmm.score(reduced)),
        "component_weights": gmm.weights_.tolist(),
        "hard_cluster_counts": np.bincount(labels, minlength=4).tolist(),
        "mean_posterior": probabilities.mean(axis=0).tolist(),
        "mean_entropy_nats": float(entropy.mean()),
        "mean_normalized_entropy": float(entropy.mean() / np.log(4)),
        "max_posterior_quantiles": dict(
            zip(("p05", "p50", "p95"), np.quantile(probabilities.max(axis=1), [0.05, 0.5, 0.95]).tolist(), strict=True)
        ),
        "fraction_max_posterior_above_099": float((probabilities.max(axis=1) > 0.99).mean()),
        "fitting_seconds": time.monotonic() - started,
        "seed": seed,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "plot": "First two coordinates of the fitted PCA; GMM means projected onto the same axes.",
        "exemplar_selection": (
            "Smallest within-component Mahalanobis distance among hard-assigned members; plus seeded random members."
        ),
    }
    _write_json(output / "summary.json", summary)
    logger.info("Finished: {}", json.dumps(summary, indent=2))
    return summary
