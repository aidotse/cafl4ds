"""Final test-split evaluation — did training actually improve the representation?

The per-round health probes read small held-out sets carved from the training split; they track
trends. This is the end-of-run verdict instead: kNN and linear probes fitted on a large labelled
set from the training split (the **support**) and scored on the dataset's **test** split, which
nothing else in a run touches. The same protocol scores the starting and the final model, so the
difference is what training added. The encoder stays frozen and labels are used here only.

Features are encoded once, in batches (a whole training split does not fit through a ViT in one
forward pass), and handed to :func:`~cafl4ds.measurements.knn_probe` /
:func:`~cafl4ds.measurements.linear_probe` as precomputed inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger

from cafl4ds import measurements
from cafl4ds.data.sources import DataSource
from cafl4ds.data.streams import EvalSet
from cafl4ds.jsonio import dumps_valid
from cafl4ds.ssl.base import SSLMethod


def encode_batched(method: SSLMethod, images: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    """Embed ``images`` in eval mode, ``batch_size`` at a time, restoring the training flag.

    Args:
        method: The SSL method whose pooled backbone embedding to read.
        images: Images ``[N, C, H, W]`` (any device).
        batch_size: Images per forward pass.

    Returns:
        The embeddings ``[N, d]`` on the CPU.
    """
    device = next(method.parameters()).device
    was_training = method.training
    method.eval()
    try:
        chunks = [method.encode(images[i : i + batch_size].to(device)).cpu() for i in range(0, len(images), batch_size)]
    finally:
        method.train(was_training)
    return torch.cat(chunks)


def subsample_per_class(labels: torch.Tensor, per_class: int | None, seed: int = 0) -> torch.Tensor:
    """Pick at most ``per_class`` indices of each class, at random but reproducibly.

    Args:
        labels: Integer labels ``[N]``.
        per_class: Cap per class; ``None`` keeps every index.
        seed: RNG seed for the draw.

    Returns:
        Sorted indices into ``labels``.
    """
    if per_class is None:
        return torch.arange(labels.numel())
    generator = torch.Generator().manual_seed(seed)
    picked = []
    for cls in labels.unique():
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        picked.append(idx[torch.randperm(idx.numel(), generator=generator)[:per_class]])
    return torch.cat(picked).sort().values


class FinalEvaluator:
    """kNN + linear probes fitted on a training-split support set, scored on a test split."""

    def __init__(self, support: EvalSet, test: EvalSet, knn_k: int = 20, batch_size: int = 1024) -> None:
        """Configure the evaluator.

        Args:
            support: Labelled images the probes are fitted on (from the training split).
            test: Labelled images the probes are scored on (the test split).
            knn_k: Neighbours for the kNN probe.
            batch_size: Images per forward pass when encoding.
        """
        self.support = support
        self.test = test
        self.knn_k = knn_k
        self.batch_size = batch_size

    @classmethod
    def from_sources(
        cls,
        train: DataSource,
        test: DataSource,
        support_per_class: int | None = None,
        knn_k: int = 20,
        batch_size: int = 1024,
        seed: int = 0,
    ) -> FinalEvaluator:
        """Build the support set from ``train`` and the test set from ``test``.

        Args:
            train: The run's training-split source (the support pool).
            test: The held-out test-split source.
            support_per_class: Cap on support images per class; ``None`` uses all of them.
            knn_k: Neighbours for the kNN probe.
            batch_size: Images per forward pass when encoding.
            seed: RNG seed for the support subsample.

        Returns:
            The configured evaluator.
        """
        train_images, train_labels = train.load()
        test_images, test_labels = test.load()
        keep = subsample_per_class(train_labels, support_per_class, seed)
        # Without a cap, use the training tensors as they are: indexing would copy the whole split.
        support = (
            EvalSet(train_images, train_labels)
            if support_per_class is None
            else EvalSet(train_images[keep], train_labels[keep])
        )
        logger.info(f"final eval: {len(keep)} support images (train split), {len(test_labels)} test images")
        return cls(support, EvalSet(test_images, test_labels), knn_k=knn_k, batch_size=batch_size)

    def evaluate(self, method: SSLMethod) -> dict[str, float]:
        """Score ``method``'s frozen representation on the test split.

        Args:
            method: The model to evaluate.

        Returns:
            ``knn_acc`` and ``linear_acc`` on the test split, in ``[0, 1]``.
        """
        zs = encode_batched(method, self.support.images, self.batch_size)
        zt = encode_batched(method, self.test.images, self.batch_size)
        support, test = (zs, self.support.labels), (zt, self.test.labels)

        def identity(z: torch.Tensor) -> torch.Tensor:
            return z  # features are precomputed

        return {
            "knn_acc": measurements.knn_probe(identity, support, test, k=self.knn_k),
            "linear_acc": measurements.linear_probe(identity, support, test),
        }

    def write_report(self, start: dict[str, float], final: dict[str, float], path: Path) -> dict[str, Any]:
        """Write the start / final / gain comparison to ``path`` as JSON and log it.

        Args:
            start: :meth:`evaluate` on the starting model.
            final: :meth:`evaluate` on the final model.
            path: Destination ``.json`` file.

        Returns:
            The report written.
        """
        gain = {key: final[key] - start[key] for key in final}
        report: dict[str, Any] = {
            "start": start,
            "final": final,
            "gain": gain,
            "support_size": len(self.support.labels),
            "test_size": len(self.test.labels),
        }
        path.write_text(dumps_valid(report), encoding="utf-8")
        logger.info(
            "final eval (test split): "
            + "  ".join(f"{k} {start[k]:.4f} -> {final[k]:.4f} ({gain[k]:+.4f})" for k in final)
            + f"  — wrote {path}"
        )
        return report
