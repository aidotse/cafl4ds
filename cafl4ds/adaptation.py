"""P1.2 — the readout ladder and the adapted-vs-frozen contrast (leg (a): does adaptation beat B5?).

Leg (a) of the Phase-1 exit gate compares an encoder that kept learning on the stream against the
same encoder frozen (**B5**). This module owns the pieces that comparison needs, as pure functions
the reader (``scripts/read_adaptation.py``) wires together:

* the **held-out pool** — labelled frames no compared encoder trained on, split once by a fixed seed
  into a fit pool (probes and fine-tunes train on class-balanced subsamples of it) and a query set
  (every reading scores it);
* the **tasks** — the scene canary, and binary object-presence tasks from BDD's per-image category
  counts (the secondary, higher-headroom readout);
* the **ladder** — frozen kNN and linear probes, and a few-shot fine-tune of the encoder plus a
  linear head, each returning *per-image correctness* so a contrast can resample images;
* the **contrast** — Δ = adapted − reference in **balanced accuracy** (mean per-class recall), with
  a hierarchical bootstrap over seeds and query images. The image level matters here as it did not
  for P1.1's walk-vs-IID contrasts: B5 and an adapted encoder can disagree on many images, and the
  claim is about the population, not one probe set.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from cafl4ds import measurements

SCENE = "scene"
PRESENCE_PREFIX = "presence:"
_ENCODE_CHUNK = 128  # a ViT-B/16 at 224 px: features are read in chunks, then probed on CPU


@dataclass
class HeldOutPool:
    """Labelled frames for reading encoders — never trained on by any compared encoder.

    Attributes:
        images: ``[N, C, H, W]`` float images in ``[0, 1]``.
        scene: Scene (canary) label per image ``[N]``.
        stratum: Time-of-day stratum per image ``[N]``.
        regime: Regime id per image ``[N]`` (shift-walk order), for regime-relative reads.
        categories: Per-image object-category counts (BDD's labels), or ``None`` when absent.
        stratum_names: Name per stratum id.
        regime_names: Name per regime id.
    """

    images: torch.Tensor
    scene: torch.Tensor
    stratum: torch.Tensor
    regime: torch.Tensor
    categories: list[dict[str, int]] | None = None
    stratum_names: dict[int, str] = field(default_factory=dict)
    regime_names: dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        """The number of frames."""
        return int(self.scene.shape[0])


def task_labels(pool: HeldOutPool, task: str) -> torch.Tensor:
    """Integer labels for ``task``: ``"scene"``, or ``"presence:<category>"`` (1 if any box of it).

    Raises:
        ValueError: If the task is unknown, or a presence task is asked of a pool with no categories.
    """
    if task == SCENE:
        return pool.scene
    if task.startswith(PRESENCE_PREFIX):
        if pool.categories is None:
            raise ValueError(f"task {task!r} needs per-image object categories; this pool has none.")
        category = task[len(PRESENCE_PREFIX) :]
        return torch.tensor([int(c.get(category, 0) > 0) for c in pool.categories], dtype=torch.long)
    raise ValueError(f"unknown task {task!r}; expected {SCENE!r} or '{PRESENCE_PREFIX}<category>'.")


def split_pool(n: int, fit_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split ``range(n)`` once into disjoint, sorted ``(fit, query)`` index arrays by a fixed seed."""
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError(f"fit_fraction must lie in (0, 1); got {fit_fraction}.")
    order = np.random.default_rng(seed).permutation(n)
    cut = int(round(fit_fraction * n))
    return np.sort(order[:cut]), np.sort(order[cut:])


def balanced_indices(labels: np.ndarray, candidates: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Up to ``per_class`` of ``candidates`` per class of ``labels``, drawn without replacement."""
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(labels[candidates]):
        members = candidates[labels[candidates] == cls]
        keep.append(rng.permutation(members)[:per_class])
    return np.sort(np.concatenate(keep))


def balanced_accuracy(correct: np.ndarray, labels: np.ndarray) -> float:
    """Mean per-class recall from per-image correctness (chance = 1 / #classes)."""
    return float(np.mean([correct[labels == c].mean() for c in np.unique(labels)]))


def encode(encoder: nn.Module, images: torch.Tensor, device: str) -> np.ndarray:
    """``encoder.embed`` over ``images`` in chunks — no gradient, eval mode (restored after), on CPU."""
    was_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            parts = [encoder.embed(chunk.to(device)).float().cpu() for chunk in images.split(_ENCODE_CHUNK)]
    finally:
        encoder.train(was_training)
    return torch.cat(parts).numpy()


def frozen_correctness(
    features: np.ndarray, labels: np.ndarray, fit: np.ndarray, query: np.ndarray, knn_k: int
) -> dict[str, np.ndarray]:
    """Per-query correctness of the frozen kNN and linear probes (the canary's recipes) fit on ``fit``."""
    identity: Callable[[torch.Tensor], torch.Tensor] = lambda x: x  # noqa: E731 — features are pre-encoded
    support = (torch.from_numpy(features[fit]), torch.from_numpy(labels[fit]))
    target = (torch.from_numpy(features[query]), torch.from_numpy(labels[query]))
    knn, yq = measurements.knn_predict(identity, support, target, k=knn_k)
    linear, _ = measurements.linear_predict(identity, support, target)
    return {"knn": (knn == yq).astype(np.float32), "linear": (linear == yq).astype(np.float32)}


@dataclass(frozen=True)
class FineTuneRecipe:
    """The few-shot fine-tune: encoder + linear head, AdamW, cosine schedule, horizontal flips."""

    lr: float = 3e-5
    head_lr_mult: float = 10.0
    weight_decay: float = 0.05
    steps: int = 150
    batch_size: int = 32


class _Classifier(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """An encoder with a linear head on its pooled embedding."""

    def __init__(self, encoder: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(int(encoder.embed_dim), num_classes)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.embed(imgs))


def fine_tune_predict(
    encoder: nn.Module,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    query_images: torch.Tensor,
    recipe: FineTuneRecipe,
    *,
    seed: int,
    device: str,
) -> np.ndarray:
    """Fine-tune a *copy* of ``encoder`` with a fresh linear head; return its predictions on the query.

    Batches are drawn class-balanced with replacement (so a 10-shot set and a 1,500-per-class set run
    the same step count), each image flipped horizontally with probability ½. The encoder passed in is
    never modified.
    """
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # the head's init
    num_classes = int(train_labels.max()) + 1
    model = _Classifier(copy.deepcopy(encoder), num_classes).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": recipe.lr},
            {"params": model.head.parameters(), "lr": recipe.lr * recipe.head_lr_mult},
        ],
        weight_decay=recipe.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe.steps)
    by_class = [(train_labels == c).nonzero(as_tuple=True)[0] for c in range(num_classes)]
    per_class = max(1, recipe.batch_size // num_classes)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(recipe.steps):
        batch = torch.cat([idx[torch.randint(idx.numel(), (per_class,), generator=generator)] for idx in by_class])
        imgs = train_images[batch]
        flip = torch.rand(imgs.shape[0], generator=generator) < 0.5
        imgs = torch.where(flip[:, None, None, None], imgs.flip(-1), imgs)
        optimizer.zero_grad()
        loss = loss_fn(model(imgs.to(device)), train_labels[batch].to(device))
        loss.backward()
        optimizer.step()
        schedule.step()
    model.eval()
    with torch.no_grad():
        preds = [model(chunk.to(device)).argmax(dim=1).cpu() for chunk in query_images.split(_ENCODE_CHUNK)]
    return torch.cat(preds).numpy()


def bootstrap_delta(
    adapted: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
    *,
    n_boot: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """Δ = mean-over-seeds balanced accuracy of ``adapted`` minus ``reference``'s, with a bootstrap.

    The bootstrap is hierarchical: each resample draws the seeds with replacement and the query
    images with replacement *within each class* (so every class keeps its size and balanced accuracy
    stays defined), and applies the same image draw to both encoders — the contrast is paired.

    Args:
        adapted: Per-image correctness ``[S, N]`` (one row per seed; few-shot rows already averaged
            over their label draws, so values may be fractional).
        reference: Per-image correctness ``[N]`` (a frozen, deterministic reference) or ``[S_r, N]``
            (averaged over its rows).
        labels: Class label per query image ``[N]``.
        n_boot: Number of bootstrap resamples.
        seed: Bootstrap RNG seed.

    Returns:
        ``delta`` (the point estimate), ``lo`` / ``hi`` (the two-sided 95% percentile interval),
        ``p_one_sided`` (the share of resamples with Δ ≤ 0, add-one corrected — small means
        adapted > reference), ``n_seeds`` and ``n_images``.
    """
    adapted = np.atleast_2d(np.asarray(adapted, dtype=np.float64))
    reference = np.asarray(reference, dtype=np.float64)
    reference = reference.mean(axis=0) if reference.ndim == 2 else reference
    if adapted.shape[1] != labels.shape[0] or reference.shape[0] != labels.shape[0]:
        raise ValueError("adapted, reference and labels must share the query-image axis.")
    rng = np.random.default_rng(seed)
    n_seeds = adapted.shape[0]
    classes = np.unique(labels)
    recall_adapted = np.zeros((n_boot, n_seeds))
    recall_reference = np.zeros(n_boot)
    for cls in classes:
        members = np.flatnonzero(labels == cls)
        # Multinomial image counts per resample: weights[b] @ correctness = that resample's class recall.
        weights = rng.multinomial(members.size, np.full(members.size, 1.0 / members.size), size=n_boot)
        recall_adapted += weights @ adapted[:, members].T / members.size
        recall_reference += weights @ reference[members] / members.size
    bacc_adapted = recall_adapted / classes.size
    bacc_reference = recall_reference / classes.size
    seed_draw = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    deltas = np.take_along_axis(bacc_adapted, seed_draw, axis=1).mean(axis=1) - bacc_reference
    point = float(np.mean([balanced_accuracy(row, labels) for row in adapted]) - balanced_accuracy(reference, labels))
    return {
        "delta": point,
        "lo": float(np.quantile(deltas, 0.025)),
        "hi": float(np.quantile(deltas, 0.975)),
        "p_one_sided": float((1 + np.sum(deltas <= 0)) / (1 + n_boot)),
        "n_seeds": float(n_seeds),
        "n_images": float(labels.shape[0]),
    }


def holm(p_values: dict[str, float], alpha: float) -> dict[str, bool]:
    """Holm's step-down rejections at family-wise ``alpha`` (``True`` = rejected)."""
    ordered = sorted(p_values, key=p_values.__getitem__)
    rejected = dict.fromkeys(p_values, False)
    for rank, key in enumerate(ordered):
        if p_values[key] > alpha / (len(ordered) - rank):
            break
        rejected[key] = True
    return rejected


def scene_overlap(context: np.ndarray, labels: np.ndarray, fit: np.ndarray, query: np.ndarray) -> float:
    """Balanced accuracy of predicting ``labels`` from categorical ``context`` alone (one-hot, logistic).

    The presence-task screen's *overlap* read: how much of an object-presence task the scene and time
    of day already give away. ``context`` is ``[N, K]`` integer codes (e.g. scene and stratum).
    """
    codes = [np.unique(context[:, k]) for k in range(context.shape[1])]
    onehot = np.concatenate(
        [(context[:, [k]] == values[None, :]).astype(np.float32) for k, values in enumerate(codes)], axis=1
    )
    correct = frozen_correctness(onehot, labels, fit, query, knn_k=1)["linear"]
    return balanced_accuracy(correct, labels[query])
