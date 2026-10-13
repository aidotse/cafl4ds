"""P1.1.0 supervised ceiling — can the canary task be learned from the well pool at all?

The competence scout found SSL wells no better than a randomly initialized encoder on the selection set,
at every resolution. Before buying more data or a bigger model, this asks the cheaper question: with the
*labels*, does the same tiny ViT learn the 3-scene canary from the same ~4,000 well-pool frames? It trains
the encoder plus a linear head on the well pool (class-balanced cross-entropy — the pool is mostly city
street) and scores **balanced accuracy** on the drive pool, ~4,000 labelled frames it never trains on (a
far tighter read than the 150-image selection query, which is scored too).

Two references frame the result, scored the same way (probes fit on a class-balanced subsample of the
well pool, scored on the drive pool):

* ``encoder_at_init`` — frozen-feature kNN / linear probes on the encoder as built: random for the tiny
  ViT (what the SSL wells were indistinguishable from), the pretrained weights for ``encoder=vit_b16_*``;
* ``pixels`` — a linear probe on raw, average-pooled pixels (what is linearly in the frames already).

A ceiling well above both says the frames carry the signal and the SSL recipe is not finding it; a ceiling
near them says the task itself is barely learnable at this scale. Writes ``supervised_ceiling.json`` into
the run directory. ``epochs=0`` skips training: a probe-only read of the encoder as built (the
pretrained-init check), or — with ``well=<path> backbone=<je|mae>`` — of a saved SSL well's encoder,
which reads a well's competence far more tightly than its 150-image selection trace.

Example::

    uv run python scripts/supervised_ceiling.py data=bdd bdd_root=/abs/path img_size=64 device=hpu
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from numpy.typing import ArrayLike
from omegaconf import DictConfig

from cafl4ds import measurements, warmup
from cafl4ds.data.streams import EvalSet

logger.remove()
logger.add(sys.stdout, level="INFO")

_MIN_BATCH = 2
_ENCODE_CHUNK = 256  # frozen-encode chunk (a ViT-B/16 at 224 px over ~4k images at once is heavy)


class Classifier(torch.nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """The encoder with a linear head on its pooled embedding (the probes' representation)."""

    def __init__(self, encoder: torch.nn.Module, num_classes: int) -> None:
        """Wrap ``encoder`` (it must expose ``embed`` and ``embed_dim``)."""
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(int(encoder.embed_dim), num_classes)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Class logits ``[B, num_classes]``."""
        return self.head(self.encoder.embed(imgs))

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """The frozen pooled embedding (no gradient, eval mode, in chunks) — what the probes read."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return torch.cat([self.encoder.embed(chunk).cpu() for chunk in imgs.split(_ENCODE_CHUNK)])
        finally:
            self.train(was_training)


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


def probe_read(
    encode: Callable[[torch.Tensor], torch.Tensor], fit: EvalSet, test: EvalSet, knn_k: int
) -> dict[str, float]:
    """Balanced accuracy of frozen-feature kNN and linear probes fit on ``fit``, scored on ``test``."""
    support, query = (fit.images, fit.labels), (test.images, test.labels)
    return {
        "knn_bacc": balanced_accuracy(*measurements.knn_predict(encode, support, query, k=knn_k)),
        "linear_bacc": balanced_accuracy(*measurements.linear_predict(encode, support, query)),
    }


def head_read(model: Classifier, test: EvalSet, batch_size: int) -> float:
    """Balanced accuracy of the trained head on ``test`` (eval mode, batched)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, test.images.shape[0], batch_size):
            preds.append(model(test.images[start : start + batch_size]).argmax(dim=1).cpu())
    model.train()
    return balanced_accuracy(torch.cat(preds).numpy(), test.labels.numpy())


def train_ceiling(
    model: Classifier, train: EvalSet, tests: dict[str, EvalSet], config: DictConfig
) -> list[dict[str, float]]:
    """Train ``model`` on ``train`` with class-balanced cross-entropy; read every test set each epoch.

    Args:
        model: The classifier (on ``config.device``).
        train: The labelled training pool.
        tests: Named held-out sets scored (head balanced accuracy) after every epoch.
        config: The composed config (``epochs``, ``batch_size``, ``hflip``, ``device``, ``optim``).

    Returns:
        One row per epoch: ``epoch``, ``train_loss``, ``train_bacc``, and ``<name>_bacc`` per test set.
    """
    device = torch.device(config.device)
    counts = torch.bincount(train.labels, minlength=int(train.labels.max()) + 1).float()
    weight = (counts.sum() / (counts.numel() * counts.clamp(min=1))).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = instantiate(config.optim, params=model.parameters())
    generator = torch.Generator().manual_seed(int(config.seed))
    trace: list[dict[str, float]] = []
    for epoch in range(1, int(config.epochs) + 1):
        order = torch.randperm(train.images.shape[0], generator=generator)
        losses = []
        for start in range(0, order.numel(), int(config.batch_size)):
            batch = order[start : start + int(config.batch_size)]
            if batch.numel() < _MIN_BATCH:
                continue
            imgs = train.images[batch]
            if config.hflip:
                flip = torch.rand(imgs.shape[0], generator=generator) < 0.5
                imgs = torch.where(flip[:, None, None, None], imgs.flip(-1), imgs)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs.to(device)), train.labels[batch].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": float(epoch), "train_loss": float(np.mean(losses))}
        row["train_bacc"] = head_read(model, train, int(config.batch_size))
        row.update({f"{name}_bacc": head_read(model, ev, int(config.batch_size)) for name, ev in tests.items()})
        trace.append(row)
        logger.info(f"epoch {epoch}: {row}")
    return trace


def _encoder(config: DictConfig) -> torch.nn.Module:
    """The encoder to read: as configured, or — with ``well=<path>`` — a saved SSL well's encoder."""
    encoder = instantiate(config.encoder)
    if not config.get("well"):
        return encoder
    method = instantiate(config.ssl, encoder=encoder)
    warmup.load_well(method, to_absolute_path(str(config.well)))
    return method.encoder


def run(config: DictConfig, out_dir: Path) -> dict[str, Any]:
    """Build the pools, read the references, train the ceiling, and write the report (Hydra-free core)."""
    torch.manual_seed(int(config.seed))
    stream = instantiate(config.stream)
    well, drive = stream.labelled_pool("well"), stream.labelled_pool("drive")
    selection = stream.selection_sets
    tests = {"drive": drive}
    if selection is not None:
        tests["selection"] = EvalSet(
            torch.cat([selection.probe_support.images, selection.probe_query.images]),
            torch.cat([selection.probe_support.labels, selection.probe_query.labels]),
        )
    fit = balanced_subsample(well, int(config.probe_per_class), int(config.seed))
    logger.info(
        f"well pool {well.labels.shape[0]} (class counts {torch.bincount(well.labels).tolist()}), "
        f"drive pool {drive.labels.shape[0]}, probe fit {fit.labels.shape[0]}"
    )

    num_classes = int(max(well.labels.max(), drive.labels.max())) + 1
    model = Classifier(_encoder(config), num_classes).to(torch.device(config.device))
    knn_k = int(config.knn_k)
    pool = torch.nn.AdaptiveAvgPool2d(int(config.pixel_grid))
    references = {
        "encoder_at_init": probe_read(model.encode, fit, drive, knn_k),
        "pixels": {
            "linear_bacc": balanced_accuracy(
                *measurements.linear_predict(
                    lambda x: pool(x).flatten(1), (fit.images, fit.labels), (drive.images, drive.labels)
                )
            )
        },
    }
    logger.info(f"references (drive pool, balanced accuracy): {references}")

    trace = train_ceiling(model, well, tests, config)
    model.eval()
    report = {
        "config": {
            **{k: config[k] for k in ("img_size", "epochs", "batch_size", "hflip", "probe_per_class", "seed")},
            "encoder": str(config.encoder.get("_target_")),
            "well": str(config.well) if config.get("well") else None,
        },
        "n": {
            "well": int(well.labels.shape[0]),
            "drive": int(drive.labels.shape[0]),
            "probe_fit": int(fit.labels.shape[0]),
        },
        "chance": 1.0 / num_classes,
        "references": references,
        "trained_encoder_probes": probe_read(model.encode, fit, drive, knn_k),
        "best_drive_bacc": max((r["drive_bacc"] for r in trace), default=None),
        "final": trace[-1] if trace else None,
        "trace": trace,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "supervised_ceiling.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        f"supervised ceiling: best drive bacc {report['best_drive_bacc']} | final {report['final']} | "
        f"trained-encoder probes {report['trained_encoder_probes']} | wrote {out}"
    )
    return report


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="supervised_ceiling")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run the supervised ceiling on the configured well/drive split."""
    run(config, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
