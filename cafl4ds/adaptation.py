"""P1.2 — the readout ladder and the adapted-vs-frozen contrast (leg (a): does adaptation beat B5?).

Leg (a) of the Phase-1 exit gate compares an encoder that kept learning on the stream against the
same encoder frozen (**B5**). This module owns the pieces that comparison needs, as pure functions
the reader (``scripts/read_adaptation.py``) wires together:

* the **held-out pool** — labelled frames no compared encoder trained on, split once by a fixed seed
  into a fit pool (probes and fine-tunes train on class-balanced subsamples of it) and a query set
  (every reading scores it);
* the **tasks** — the scene canary, and binary object-presence tasks from BDD's per-image category
  counts (the secondary, higher-headroom readout); P1.2.5 adds large-object presence from box areas and a
  **per-patch** task (is each patch mostly drivable?), read on patch tokens and scored per patch;
* the **ladder** — frozen kNN and linear probes, and a few-shot fine-tune of the encoder plus a
  linear head, each returning *per-image correctness* so a contrast can resample images;
* the **contrast** — Δ = adapted − reference in **balanced accuracy** (mean per-class recall), with
  a hierarchical bootstrap over seeds and query images. The image level matters here as it did not
  for P1.1's walk-vs-IID contrasts: B5 and an adapted encoder can disagree on many images, and the
  claim is about the population, not one probe set.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from cafl4ds import measurements
from cafl4ds.data.attributes import as_float

SCENE = "scene"
WEATHER = "weather"
WEATHER_CLASSES = ("clear", "rainy", "snowy")  # BDD's unambiguous, well-populated weathers
PRESENCE_PREFIX = "presence:"
PRESENCE_CELL_PREFIX = "presence_cell:"
LARGE_CELL_PREFIX = "large_cell:"
SIZE_CELL_PREFIX = "size_cell:"
DRIVABLE = "drivable"  # the per-patch task: is each patch mostly (> half) drivable area?
_CELL_BALANCED = (PRESENCE_CELL_PREFIX, LARGE_CELL_PREFIX, SIZE_CELL_PREFIX)
_ENCODE_CHUNK = 128  # a ViT-B/16 at 224 px: features are read in chunks, then probed on CPU


@dataclass
class HeldOutPool:
    """Labelled frames for reading encoders — never trained on by any compared encoder.

    Attributes:
        images: ``[N, C, H, W]`` frames — float in ``[0, 1]`` or uint8 in ``[0, 255]`` (converted per batch).
        scene: Scene (canary) label per image ``[N]``.
        stratum: Time-of-day stratum per image ``[N]``.
        regime: Regime id per image ``[N]`` (shift-walk order), for regime-relative reads.
        categories: Per-image object-category counts (BDD's labels), or ``None`` when absent.
        stratum_names: Name per stratum id.
        regime_names: Name per regime id.
        boxes: Per-image ``(category, frame share)`` boxes, or ``None`` when absent (P1.2.5's large objects).
        drivable: Per-image drivable share of each patch-grid cell ``[N, g, g]`` (``NaN`` = no map), or ``None``.
    """

    images: torch.Tensor
    scene: torch.Tensor
    stratum: torch.Tensor
    regime: torch.Tensor
    categories: list[dict[str, int]] | None = None
    stratum_names: dict[int, str] = field(default_factory=dict)
    regime_names: dict[int, str] = field(default_factory=dict)
    boxes: list[list[tuple[str, float]]] | None = None
    drivable: torch.Tensor | None = None

    def __len__(self) -> int:
        """The number of frames."""
        return int(self.scene.shape[0])


def is_patch_task(task: str) -> bool:
    """Whether ``task`` is labelled per patch (read on patch tokens) rather than per frame."""
    return task == DRIVABLE


def task_labels(pool: HeldOutPool, task: str) -> torch.Tensor:
    """Integer labels for ``task``; ``-1`` marks a frame the task does not use.

    * ``"scene"`` — the canary.
    * ``"weather"`` — clear / rainy / snowy, read off the regime name (``"<timeofday>·<weather>"``);
      other weathers are ``-1``.
    * ``"presence:<category>"`` / ``"presence_cell:<category>"`` — 1 if any box of the category. The
      ``_cell`` form uses the same labels; :func:`task_indices` then balances them within each
      (scene × time-of-day) cell, so scene and time of day carry no information about the label.
    * ``"large_cell:<category>[,<category>...]:<min_area>"`` — 1 if any box of those categories covers at
      least ``min_area`` of the frame (e.g. ``large_cell:car,bus,truck:0.02``), cell-balanced as above.
    * ``"size_cell:<category>:<min_area>:<max_area>"`` — P1.2.5's Part F size bins: 1 if the frame's *largest* box
      of the category has a frame share in ``[min_area, max_area)``, 0 if the frame has no box of it, ``-1`` if its
      largest box falls outside the bin; cell-balanced as above.
    * ``"drivable"`` — the per-patch task; per frame, 0 where it has a drivable map and ``-1`` where not
      (its patch labels are :func:`patch_labels`).

    Raises:
        ValueError: If the task is unknown or malformed, or asks for labels the pool does not carry.
    """
    if task == SCENE:
        return pool.scene
    if task == WEATHER:
        weather = {rid: name.split("·")[-1] for rid, name in pool.regime_names.items()}
        code = {w: i for i, w in enumerate(WEATHER_CLASSES)}
        return torch.tensor([code.get(weather.get(int(r), ""), -1) for r in pool.regime], dtype=torch.long)
    if task == DRIVABLE or task.startswith((LARGE_CELL_PREFIX, SIZE_CELL_PREFIX)):
        return _geometry_labels(pool, task)
    for prefix in (PRESENCE_PREFIX, PRESENCE_CELL_PREFIX):
        if task.startswith(prefix):
            if pool.categories is None:
                raise ValueError(f"task {task!r} needs per-image object categories; this pool has none.")
            category = task[len(prefix) :]
            return torch.tensor([int(c.get(category, 0) > 0) for c in pool.categories], dtype=torch.long)
    raise ValueError(
        f"unknown task {task!r}; expected {SCENE!r}, {WEATHER!r}, {DRIVABLE!r}, '{PRESENCE_PREFIX}<category>', "
        f"'{PRESENCE_CELL_PREFIX}<category>', '{LARGE_CELL_PREFIX}<categories>:<min_area>' or "
        f"'{SIZE_CELL_PREFIX}<category>:<min_area>:<max_area>'."
    )


def _geometry_labels(pool: HeldOutPool, task: str) -> torch.Tensor:
    """:func:`task_labels` for P1.2.5's geometry tasks (``drivable``, ``large_cell:…``, ``size_cell:…``)."""
    if task == DRIVABLE:
        if pool.drivable is None:
            raise ValueError(f"task {task!r} needs drivable-area maps; this pool has none.")
        has_map = ~torch.isnan(pool.drivable.flatten(1)).any(dim=1)
        return torch.where(has_map, 0, -1).long()
    if pool.boxes is None:
        raise ValueError(f"task {task!r} needs per-image box areas; this pool has none.")
    if task.startswith(SIZE_CELL_PREFIX):
        return _size_labels(pool.boxes, task)
    spec = task[len(LARGE_CELL_PREFIX) :].rsplit(":", 1)
    if len(spec) != 2:
        raise ValueError(f"task {task!r} must read '{LARGE_CELL_PREFIX}<categories>:<min_area>'.")
    wanted, min_area = set(spec[0].split(",")), float(spec[1])
    return torch.tensor(
        [int(any(c in wanted and a >= min_area for c, a in boxes)) for boxes in pool.boxes], dtype=torch.long
    )


def _size_labels(boxes: list[list[tuple[str, float]]], task: str) -> torch.Tensor:
    """``size_cell:<category>:<min_area>:<max_area>`` labels: the largest box's bin, no box, or neither (``-1``)."""
    spec = task[len(SIZE_CELL_PREFIX) :].rsplit(":", 2)
    if len(spec) != 3:
        raise ValueError(f"task {task!r} must read '{SIZE_CELL_PREFIX}<category>:<min_area>:<max_area>'.")
    category, low, high = spec[0], float(spec[1]), float(spec[2])
    if not 0.0 <= low < high:
        raise ValueError(f"task {task!r} needs 0 <= min_area < max_area.")
    out = []
    for frame in boxes:
        areas = [a for c, a in frame if c == category]
        if not areas:
            out.append(0)
        else:
            out.append(1 if low <= max(areas) < high else -1)
    return torch.tensor(out, dtype=torch.long)


def patch_labels(pool: HeldOutPool) -> torch.Tensor:
    """The per-patch task's labels ``[N, g * g]``: 1 where a patch is mostly (> ½) drivable, ``-1`` without a map."""
    if pool.drivable is None:
        raise ValueError("the per-patch task needs drivable-area maps; this pool has none.")
    share = pool.drivable.flatten(1)
    return torch.where(torch.isnan(share), -1, (share > 0.5).long()).long()


def task_indices(pool: HeldOutPool, task: str, candidates: np.ndarray, seed: int) -> np.ndarray:
    """The frames of ``candidates`` the task uses.

    Those labelled (``>= 0``); for ``presence_cell``, ``large_cell`` and ``size_cell`` tasks, further balanced — equal
    positives and negatives drawn within every (scene × time-of-day) cell.
    """
    labels = task_labels(pool, task).numpy()
    keep = candidates[labels[candidates] >= 0]
    if not task.startswith(_CELL_BALANCED):
        return keep
    rng = np.random.default_rng(seed)
    cells = pool.scene.numpy() * 1000 + pool.stratum.numpy()
    out = []
    for cell in np.unique(cells[keep]):
        members = keep[cells[keep] == cell]
        pos, neg = members[labels[members] == 1], members[labels[members] == 0]
        n = min(pos.size, neg.size)
        out += [rng.permutation(pos)[:n], rng.permutation(neg)[:n]]
    return np.sort(np.concatenate(out)) if out else keep[:0]


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
            parts = [encoder.embed(as_float(chunk).to(device)).float().cpu() for chunk in images.split(_ENCODE_CHUNK)]
    finally:
        encoder.train(was_training)
    return torch.cat(parts).numpy()


def encode_blocks(encoder: nn.Module, images: torch.Tensor, device: str, blocks: list[int]) -> dict[int, np.ndarray]:
    """Pooled features after each of ``blocks`` (1-based, pre final norm) — P1.2.2's per-block probes.

    Pooled as :func:`cafl4ds.block_features.block_features` pools them (patch-token mean, ``cls`` dropped),
    captured by hooks on ``encoder.blocks`` during :func:`encode`'s forward.
    """
    captured: dict[int, list[torch.Tensor]] = {b: [] for b in blocks}
    handles = [
        encoder.blocks[b - 1].register_forward_hook(
            lambda _m, _i, out, b=b: captured[b].append(out[:, 1:, :].mean(dim=1).float().cpu())
        )
        for b in blocks
    ]
    try:
        encode(encoder, images, device)
    finally:
        for handle in handles:
            handle.remove()
    return {b: torch.cat(parts).numpy() for b, parts in captured.items()}


class Readout(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """An encoder read on a chosen pooling of its final tokens: the patch-token ``mean`` or the ``cls`` token.

    P1.2.5 reads every encoder on both. An encoder's own ``embed`` is the mean for the cafl4ds ViTs and the
    ``cls`` token for the torchvision ViT-B, so this wrapper is how each is read on the other. Every encoder it
    wraps exposes ``tokens(imgs) -> [B, 1 + N, d]`` (final-normed, ``cls`` first).
    """

    def __init__(self, encoder: nn.Module, readout: str) -> None:
        """Wrap ``encoder`` to read its ``readout`` (``"mean"`` or ``"cls"``)."""
        super().__init__()
        if readout not in ("mean", "cls"):
            raise ValueError(f"readout must be 'mean' or 'cls'; got {readout!r}.")
        self.encoder = encoder
        self.readout = readout
        self.embed_dim = int(encoder.embed_dim)

    @property
    def blocks(self) -> nn.ModuleList:
        """The wrapped encoder's blocks (for the per-block probes)."""
        return self.encoder.blocks

    def tokens(self, imgs: torch.Tensor) -> torch.Tensor:
        """The wrapped encoder's final tokens."""
        return self.encoder.tokens(imgs)

    def embed(self, imgs: torch.Tensor) -> torch.Tensor:
        """The chosen pooling of the final tokens ``[B, d]``."""
        tokens = self.encoder.tokens(imgs)
        return tokens[:, 0] if self.readout == "cls" else tokens[:, 1:].mean(dim=1)


def encode_readouts(encoder: nn.Module, images: torch.Tensor, device: str) -> dict[str, np.ndarray]:
    """Both poolings of the final tokens, ``{"mean": [N, d], "cls": [N, d]}``, in one pass (as :func:`encode`)."""
    was_training = encoder.training
    encoder.eval()
    parts: dict[str, list[torch.Tensor]] = {"mean": [], "cls": []}
    try:
        with torch.no_grad():
            for chunk in images.split(_ENCODE_CHUNK):
                tokens = encoder.tokens(as_float(chunk).to(device)).float().cpu()
                parts["mean"].append(tokens[:, 1:].mean(dim=1))
                parts["cls"].append(tokens[:, 0])
    finally:
        encoder.train(was_training)
    return {k: torch.cat(v).numpy() for k, v in parts.items()}


def encode_patches(encoder: nn.Module, images: torch.Tensor, device: str) -> np.ndarray:
    """Patch tokens ``[N, P, d]`` (``cls`` dropped, float16 to halve the memory), as :func:`encode`."""
    was_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            parts = [
                encoder.tokens(as_float(chunk).to(device))[:, 1:].half().cpu() for chunk in images.split(_ENCODE_CHUNK)
            ]
    finally:
        encoder.train(was_training)
    return torch.cat(parts).numpy()


def patch_counts(labels: np.ndarray) -> np.ndarray:
    """Patches per class per frame ``[N, 2]`` of per-patch labels ``[N, P]`` (``-1`` patches not counted)."""
    return np.stack([(labels == c).sum(axis=1) for c in (0, 1)], axis=1).astype(np.float32)


def patch_correctness(
    tokens: np.ndarray,
    labels: np.ndarray,
    fit: np.ndarray,
    query: np.ndarray,
    knn_k: int,
    per_class: int,
    seed: int,
    rungs: tuple[str, ...] = ("knn", "linear"),
) -> dict[str, np.ndarray]:
    """The frozen per-patch kNN and linear probes, fit on ``fit``'s patches, as correct-patch counts per query frame.

    The probes fit on up to ``per_class`` patches of each class drawn (``seed``) from the ``fit`` frames, and
    score every labelled patch of the ``query`` frames. ``tokens`` / ``labels`` are ``[N, P, d]`` / ``[N, P]``,
    indexed by ``fit`` and ``query``. ``rungs`` picks the probes (the kNN over every query patch is the slow one).

    Returns:
        ``rung -> [len(query), 2]``, the correct patches of each class (0, 1) per query frame; with
        :func:`patch_counts` they give the patch-level balanced accuracy (:func:`patch_balanced_accuracy`).
    """
    if tokens.shape[1] != labels.shape[1]:
        raise ValueError(
            f"{tokens.shape[1]} patch tokens per frame but {labels.shape[1]} patch labels: the label grid "
            "must be the encoder's patch grid (data.drivable_grid)."
        )
    dim = tokens.shape[-1]
    xs, ys = tokens[fit].reshape(-1, dim), labels[fit].reshape(-1)
    keep = balanced_indices(ys, np.flatnonzero(ys >= 0), per_class, seed)
    xq, yq = tokens[query].reshape(-1, dim), labels[query].reshape(-1)
    valid = yq >= 0
    frame = np.repeat(np.arange(query.size), labels.shape[1])[valid]
    identity: Callable[[torch.Tensor], torch.Tensor] = lambda x: x  # noqa: E731 — features are pre-encoded
    support = (torch.from_numpy(xs[keep].astype(np.float32)), torch.from_numpy(ys[keep]))
    target = (torch.from_numpy(xq[valid].astype(np.float32)), torch.from_numpy(yq[valid]))
    out = {}
    for rung in rungs:
        if rung == "knn":
            pred, truth = measurements.knn_predict(identity, support, target, k=knn_k)
        else:
            pred, truth = measurements.linear_predict(identity, support, target)
        counts = np.zeros((query.size, 2), dtype=np.float32)
        np.add.at(counts, (frame, truth), (pred == truth).astype(np.float32))
        out[rung] = counts
    return out


def patch_balanced_accuracy(correct: np.ndarray, counts: np.ndarray) -> float:
    """Patch-level balanced accuracy from per-frame correct-patch counts ``[N, 2]`` and patch counts ``[N, 2]``."""
    return float(np.mean(correct.sum(axis=0) / counts.sum(axis=0)))


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
    the same step count), each image flipped horizontally with probability ½. The copy is always fully
    trainable, even if ``encoder`` arrives frozen; the encoder passed in is never modified.
    """
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # the head's init
    num_classes = int(train_labels.max()) + 1
    model = _Classifier(copy.deepcopy(encoder), num_classes).to(device)
    model.requires_grad_(True)  # a method may hand over a frozen encoder (DINO's moving-average teacher)
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
        imgs = as_float(train_images[batch])
        flip = torch.rand(imgs.shape[0], generator=generator) < 0.5
        imgs = torch.where(flip[:, None, None, None], imgs.flip(-1), imgs)
        optimizer.zero_grad()
        loss = loss_fn(model(imgs.to(device)), train_labels[batch].to(device))
        loss.backward()
        optimizer.step()
        schedule.step()
    model.eval()
    with torch.no_grad():
        preds = [model(as_float(chunk).to(device)).argmax(dim=1).cpu() for chunk in query_images.split(_ENCODE_CHUNK)]
    return torch.cat(preds).numpy()


class _PatchClassifier(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """An encoder with a linear head on every patch token."""

    def __init__(self, encoder: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(int(encoder.embed_dim), num_classes)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.tokens(imgs)[:, 1:])  # [B, P, classes]


def fine_tune_patch_predict(
    encoder: nn.Module,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    query_images: torch.Tensor,
    recipe: FineTuneRecipe,
    *,
    seed: int,
    device: str,
) -> np.ndarray:
    """The per-patch version of :func:`fine_tune_predict`: a copy of ``encoder`` plus a linear head on every patch.

    ``train_labels`` is ``[n, P]`` (``-1`` patches ignored). Batches are drawn uniformly (both classes share
    frames), the loss is weighted by inverse class frequency (the read is balanced accuracy), and a horizontal
    flip mirrors the label grid with the image.

    Returns:
        Per-patch predictions on the query ``[n_query, P]``.
    """
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # the head's init
    model = _PatchClassifier(copy.deepcopy(encoder), 2).to(device)
    model.requires_grad_(True)
    model.train()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": recipe.lr},
            {"params": model.head.parameters(), "lr": recipe.lr * recipe.head_lr_mult},
        ],
        weight_decay=recipe.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe.steps)
    counts = torch.stack([(train_labels == c).sum() for c in (0, 1)]).float()
    loss_fn = nn.CrossEntropyLoss(weight=(counts.sum() / (2 * counts.clamp(min=1))).to(device), ignore_index=-1)
    side = int(round(train_labels.shape[1] ** 0.5))
    for _ in range(recipe.steps):
        batch = torch.randint(train_labels.shape[0], (recipe.batch_size,), generator=generator)
        imgs, labels = as_float(train_images[batch]), train_labels[batch].view(-1, side, side)
        flip = torch.rand(imgs.shape[0], generator=generator) < 0.5
        imgs = torch.where(flip[:, None, None, None], imgs.flip(-1), imgs)
        labels = torch.where(flip[:, None, None], labels.flip(-1), labels).reshape(imgs.shape[0], -1)
        optimizer.zero_grad()
        loss = loss_fn(model(imgs.to(device)).flatten(0, 1), labels.flatten().to(device))
        loss.backward()
        optimizer.step()
        schedule.step()
    model.eval()
    with torch.no_grad():
        preds = [model(as_float(chunk).to(device)).argmax(dim=2).cpu() for chunk in query_images.split(_ENCODE_CHUNK)]
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


def bootstrap_contrast(
    first: np.ndarray,
    second: np.ndarray,
    labels: np.ndarray,
    *,
    n_boot: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """The two-sample version of :func:`bootstrap_delta`: both arms are adapted, so both carry seed spread.

    Δ = mean-over-seeds balanced accuracy of ``first`` minus ``second``'s (P1.2.4's adapted-vs-adapted
    contrasts, e.g. the BDD IID drive against an off-domain control). Each resample draws the query
    images within each class, shared by both arms (paired), and each arm's seeds with replacement,
    independently (the arms' seeds are unrelated runs).

    Args:
        first: Per-image correctness ``[S1, N]``.
        second: Per-image correctness ``[S2, N]``.
        labels: Class label per query image ``[N]``.
        n_boot: Number of bootstrap resamples.
        seed: Bootstrap RNG seed.

    Returns:
        As :func:`bootstrap_delta`, with ``n_seeds`` for ``first`` and ``n_seeds_second`` for ``second``.
    """
    arms = [np.atleast_2d(np.asarray(a, dtype=np.float64)) for a in (first, second)]
    if any(a.shape[1] != labels.shape[0] for a in arms):
        raise ValueError("first, second and labels must share the query-image axis.")
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    recalls = [np.zeros((n_boot, a.shape[0])) for a in arms]
    for cls in classes:
        members = np.flatnonzero(labels == cls)
        weights = rng.multinomial(members.size, np.full(members.size, 1.0 / members.size), size=n_boot)
        for i, arm in enumerate(arms):
            recalls[i] += weights @ arm[:, members].T / members.size
    means = []
    for recall, arm in zip(recalls, arms, strict=True):
        draw = rng.integers(0, arm.shape[0], size=(n_boot, arm.shape[0]))
        means.append(np.take_along_axis(recall / classes.size, draw, axis=1).mean(axis=1))
    deltas = means[0] - means[1]
    point = float(
        np.mean([balanced_accuracy(r, labels) for r in arms[0]])
        - np.mean([balanced_accuracy(r, labels) for r in arms[1]])
    )
    return {
        "delta": point,
        "lo": float(np.quantile(deltas, 0.025)),
        "hi": float(np.quantile(deltas, 0.975)),
        "p_one_sided": float((1 + np.sum(deltas <= 0)) / (1 + n_boot)),
        "n_seeds": float(arms[0].shape[0]),
        "n_seeds_second": float(arms[1].shape[0]),
        "n_images": float(labels.shape[0]),
    }


def bootstrap_combination(
    terms: list[tuple[float, np.ndarray, np.ndarray]],
    *,
    n_boot: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """A signed sum of balanced accuracies over disjoint image slices, bootstrapped over shared seeds and images.

    P1.2.5's Part F reads change *within a slice* (e.g. the night frames' Δ at the walk's end minus at step 846)
    and differences of such changes between slices (rainy minus clear). Balanced accuracy is linear in per-image
    correctness, so a change between two checkpoints of the same seed is the balanced accuracy of their
    per-image difference; each term carries such a row per seed. The statistic is
    ``mean over seeds of sum_i weight_i * bacc(rows_i[seed])``.

    Each resample draws the seeds once with replacement, shared by every term (the terms' rows are the same
    seeds' runs), and the images with replacement within each class *of each term* (the slices are disjoint
    frames; within a term the draw is shared by everything its row was differenced from).

    Args:
        terms: ``(weight, rows [S, N_i], labels [N_i])`` per slice; every ``rows`` has the same ``S``.
        n_boot: Number of bootstrap resamples.
        seed: Bootstrap RNG seed.

    Returns:
        As :func:`bootstrap_delta` (``n_images`` sums the terms' images).
    """
    if not terms:
        raise ValueError("bootstrap_combination needs at least one term.")
    rows = [np.atleast_2d(np.asarray(r, dtype=np.float64)) for _, r, _ in terms]
    n_seeds = rows[0].shape[0]
    if any(r.shape[0] != n_seeds for r in rows):
        raise ValueError("every term must carry the same seeds.")
    if any(r.shape[1] != lab.shape[0] for r, (_, _, lab) in zip(rows, terms, strict=True)):
        raise ValueError("each term's rows and labels must share the image axis.")
    rng = np.random.default_rng(seed)
    total = np.zeros((n_boot, n_seeds))
    point = 0.0
    for (weight, _, labels), row in zip(terms, rows, strict=True):
        classes = np.unique(labels)
        recall = np.zeros((n_boot, n_seeds))
        for cls in classes:
            members = np.flatnonzero(labels == cls)
            weights = rng.multinomial(members.size, np.full(members.size, 1.0 / members.size), size=n_boot)
            recall += weights @ row[:, members].T / members.size
        total += float(weight) * recall / classes.size
        point += float(weight) * float(np.mean([balanced_accuracy(r, labels) for r in row]))
    draw = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    stats = np.take_along_axis(total, draw, axis=1).mean(axis=1)
    return {
        "delta": point,
        "lo": float(np.quantile(stats, 0.025)),
        "hi": float(np.quantile(stats, 0.975)),
        "p_one_sided": float((1 + np.sum(stats <= 0)) / (1 + n_boot)),
        "n_seeds": float(n_seeds),
        "n_images": float(sum(lab.shape[0] for _, _, lab in terms)),
    }


def bootstrap_delta_patches(
    adapted: np.ndarray,
    reference: np.ndarray,
    counts: np.ndarray,
    *,
    n_boot: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """:func:`bootstrap_delta` for the per-patch task: Δ in patch-level balanced accuracy.

    The resampled unit is still the query frame — patches of one frame are not independent — so each resample
    draws frames with replacement (every frame holds patches of both classes, so there is no class stratum)
    and each class's recall is its correct patches over its patches, summed over the drawn frames. Seeds are
    drawn with replacement as in :func:`bootstrap_delta`; the frame draw is shared by both encoders (paired).

    Args:
        adapted: Correct patches per class per frame ``[S, N, 2]`` (or ``[N, 2]`` for one seed).
        reference: The same for the frozen reference, ``[N, 2]`` (or ``[S_r, N, 2]``, averaged).
        counts: Patches per class per frame ``[N, 2]`` (:func:`patch_counts`).
        n_boot: Number of bootstrap resamples.
        seed: Bootstrap RNG seed.

    Returns:
        As :func:`bootstrap_delta` (``n_images`` counts query frames).
    """
    adapted = np.asarray(adapted, dtype=np.float64)
    adapted = adapted[None] if adapted.ndim == 2 else adapted
    reference = np.asarray(reference, dtype=np.float64)
    reference = reference.mean(axis=0) if reference.ndim == 3 else reference
    counts = np.asarray(counts, dtype=np.float64)
    if adapted.shape[1:] != counts.shape or reference.shape != counts.shape:
        raise ValueError("adapted, reference and counts must share the [frame, class] axes.")
    rng = np.random.default_rng(seed)
    n_seeds, n_frames = adapted.shape[0], counts.shape[0]
    weights = rng.multinomial(n_frames, np.full(n_frames, 1.0 / n_frames), size=n_boot).astype(np.float64)
    totals = weights @ counts  # [B, 2]
    bacc_adapted = (np.einsum("bn,snc->bsc", weights, adapted) / totals[:, None, :]).mean(axis=2)  # [B, S]
    bacc_reference = ((weights @ reference) / totals).mean(axis=1)
    seed_draw = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    deltas = np.take_along_axis(bacc_adapted, seed_draw, axis=1).mean(axis=1) - bacc_reference
    point = float(
        np.mean([patch_balanced_accuracy(row, counts) for row in adapted]) - patch_balanced_accuracy(reference, counts)
    )
    return {
        "delta": point,
        "lo": float(np.quantile(deltas, 0.025)),
        "hi": float(np.quantile(deltas, 0.975)),
        "p_one_sided": float((1 + np.sum(deltas <= 0)) / (1 + n_boot)),
        "n_seeds": float(n_seeds),
        "n_images": float(n_frames),
    }


def _betainc(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b), by its continued fraction (Numerical Recipes 6.4)."""
    if x <= 0.0 or x >= 1.0:
        return 0.0 if x <= 0.0 else 1.0
    if x > (a + 1.0) / (a + b + 2.0):  # the fraction converges fast only below the mean; use the symmetry
        return 1.0 - _betainc(b, a, 1.0 - x)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)) / a
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    result = d
    for m in range(1, 300):
        for numerator in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            result *= c * d
        if abs(c * d - 1.0) < 1e-14:
            break
    return front * result


def t_upper_tail(t: float, df: float) -> float:
    """P(T > t) for Student's t with ``df`` degrees of freedom (``df`` may be fractional, as Welch's)."""
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


def _t_result(mean: float, se: float, df: float, n: float) -> dict[str, float]:
    if se == 0.0:  # every seed agrees: the sign alone decides
        return {
            "mean": mean,
            "t": math.copysign(math.inf, mean) if mean else 0.0,
            "df": df,
            "p_one_sided": float(mean <= 0),
            "n": n,
        }
    t = mean / se
    return {"mean": mean, "t": t, "df": df, "p_one_sided": t_upper_tail(t, df), "n": n}


def seed_ttest(per_seed: np.ndarray, reference: float) -> dict[str, float]:
    """The seed-level test beside :func:`bootstrap_delta`: a one-sample *t*-test that the per-seed Δ exceeds 0.

    Args:
        per_seed: Each seed's balanced accuracy.
        reference: The frozen reference's balanced accuracy.

    Returns:
        The mean Δ, ``t``, ``df``, the one-sided ``p_one_sided`` and the seed count ``n``.
    """
    delta = np.asarray(per_seed, dtype=np.float64) - reference
    if delta.size < 2:
        raise ValueError("a seed-level test needs at least two seeds.")
    return _t_result(
        float(delta.mean()), float(delta.std(ddof=1) / math.sqrt(delta.size)), delta.size - 1.0, delta.size
    )


def seed_welch(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """The seed-level test beside :func:`bootstrap_contrast`: one-sided Welch that ``first``'s seeds read higher.

    Returns:
        As :func:`seed_ttest`, with Welch–Satterthwaite ``df`` and ``n`` the first arm's seeds.
    """
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("Welch's test needs at least two seeds per arm.")
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1)) if va + vb > 0 else a.size + b.size - 2.0
    return _t_result(float(a.mean() - b.mean()), float(math.sqrt(va + vb)), float(df), a.size)


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


PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"


def gate_verdict(
    contrasts: dict[str, dict[str, dict[str, float]]],
    primary_rung: str,
    *,
    delta_floor: float = 0.02,
    alpha: float = 0.05,
) -> dict[str, object]:
    """P1.2.1's pre-registered leg-(a) verdict for one backbone, from its end-of-walk contrasts vs B5.

    * **PASS** — on at least one task, the primary rung's one-sided p clears Holm over the tasks at
      family-wise ``alpha`` (with two tasks: the smaller p ≤ ``alpha`` / 2).
    * **FAIL** — on every task and every rung, the upper bound of Δ's 95% interval is below
      ``delta_floor``: every reading rules out a gain that large.
    * **INCONCLUSIVE** — anything else.

    Args:
        contrasts: ``task -> rung -> bootstrap_delta`` result (Δ = adapted − B5).
        primary_rung: The backbone's pre-registered primary rung (JE ``linear``, MAE ``ft100``).
        delta_floor: The smallest gain a FAIL must rule out.
        alpha: Family-wise one-sided level over the tasks.

    Returns:
        ``verdict``, the per-task primary-rung ``holm`` rejections, and the ``max_upper`` bound seen.

    Raises:
        ValueError: If a task lacks the primary rung.
    """
    missing = [t for t, rungs in contrasts.items() if primary_rung not in rungs]
    if missing:
        raise ValueError(f"tasks {missing} lack the primary rung {primary_rung!r}.")
    rejected = holm({t: rungs[primary_rung]["p_one_sided"] for t, rungs in contrasts.items()}, alpha)
    max_upper = max(c["hi"] for rungs in contrasts.values() for c in rungs.values())
    if any(rejected.values()):
        verdict = PASS
    elif max_upper < delta_floor:
        verdict = FAIL
    else:
        verdict = INCONCLUSIVE
    return {"verdict": verdict, "holm": rejected, "max_upper": max_upper, "primary_rung": primary_rung}


STREAM_SPECIFIC, MIXED, GENERIC = "STREAM-SPECIFIC", "MIXED", "GENERIC"


def attribution_verdict(
    contrasts: dict[str, dict[str, float]], *, generic_ceiling: float = 0.01, alpha: float = 0.05
) -> dict[str, object]:
    """P1.2.4's pre-registered attribution verdict for one off-domain control.

    The contrast per task is D_c = Δ_BDD − Δ_c, the BDD-trained arm's gain over the control's (B5 cancels,
    so it is :func:`bootstrap_contrast` of the two arms).

    * **STREAM-SPECIFIC** — on at least one task, D_c's one-sided p clears Holm over the tasks at ``alpha``.
    * **GENERIC** — on every task, D_c's upper bound is below ``generic_ceiling``: the control gets
      essentially all of the gain.
    * **MIXED** — anything else.

    Args:
        contrasts: ``task -> bootstrap_contrast`` result (BDD arm first, the control second).
        generic_ceiling: The upper bound a GENERIC verdict needs D_c to stay below on every task.
        alpha: Family-wise one-sided level over the tasks.

    Returns:
        ``verdict``, the per-task ``holm`` rejections, and the ``max_upper`` bound seen.
    """
    rejected = holm({t: c["p_one_sided"] for t, c in contrasts.items()}, alpha)
    max_upper = max(c["hi"] for c in contrasts.values())
    if any(rejected.values()):
        verdict = STREAM_SPECIFIC
    elif max_upper < generic_ceiling:
        verdict = GENERIC
    else:
        verdict = MIXED
    return {"verdict": verdict, "holm": rejected, "max_upper": max_upper}


def weakest_attribution(verdicts: list[str]) -> str:
    """The attribution claim over several controls: their weakest verdict (GENERIC < MIXED < STREAM-SPECIFIC)."""
    order = [GENERIC, MIXED, STREAM_SPECIFIC]
    return min(verdicts, key=order.index)
