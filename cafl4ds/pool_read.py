"""The tight competence read (P1.1.0): balanced kNN / linear probes over the labelled well and drive pools.

The selection set's 150-image query reads a well's competence to about ±0.04 — too loose to see a
pretrained backbone's warm-up dip and plateau. This read fits frozen-feature probes on a class-balanced
subsample of the **well** pool and scores **balanced accuracy** (mean per-class recall, so the majority
city-street class cannot inflate it) on the whole **drive** pool, ~3,800 frames. It is the read of the
P1.1.0 competence table (``scripts/supervised_ceiling.py``) and the ViT-B warm scout's trace
(``scripts/warm_well.py warmup.drive_read_per_class>0``).

The drive pool is the deployment's future data, so this read *diagnoses* a warm-up; it never stops one.
"""

from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import nn

from cafl4ds import measurements
from cafl4ds.data.streams import EvalSet

_ENCODE_CHUNK = 256  # frozen-encode chunk (a ViT-B/16 at 224 px over ~4k images at once is heavy)


def balanced_accuracy(pred: ArrayLike, truth: ArrayLike) -> float:
    """Mean per-class recall — accuracy that a majority-class guess cannot inflate (chance = 1/#classes)."""
    p, t = np.asarray(pred), np.asarray(truth)
    return float(np.mean([np.mean(p[t == c] == c) for c in np.unique(t)]))


def balanced_subsample(pool: EvalSet, per_class: int, seed: int) -> EvalSet:
    """Up to ``per_class`` images of every class, drawn without replacement (for fitting the probes)."""
    generator = torch.Generator().manual_seed(seed)
    keep = []
    for cls in sorted(set(pool.labels.tolist())):
        idx = (pool.labels == cls).nonzero(as_tuple=True)[0]
        keep.append(idx[torch.randperm(idx.numel(), generator=generator)[:per_class]])
    index = torch.cat(keep)
    return EvalSet(pool.images[index], pool.labels[index])


def frozen(encoder: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """``encoder.embed`` as a frozen read: no gradient, eval mode (training mode restored), in chunks, on CPU."""

    def encode(imgs: torch.Tensor) -> torch.Tensor:
        was_training = encoder.training
        encoder.eval()
        try:
            with torch.no_grad():
                return torch.cat([encoder.embed(chunk).cpu() for chunk in imgs.split(_ENCODE_CHUNK)])
        finally:
            encoder.train(was_training)

    return encode


def probe_read(
    encode: Callable[[torch.Tensor], torch.Tensor], fit: EvalSet, test: EvalSet, knn_k: int
) -> dict[str, float]:
    """Balanced accuracy of frozen-feature kNN and linear probes fit on ``fit``, scored on ``test``."""
    support, query = (fit.images, fit.labels), (test.images, test.labels)
    return {
        "knn_bacc": balanced_accuracy(*measurements.knn_predict(encode, support, query, k=knn_k)),
        "linear_bacc": balanced_accuracy(*measurements.linear_predict(encode, support, query)),
    }
