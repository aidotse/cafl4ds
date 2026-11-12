"""P1.0.2 warm-the-well step — produce a reusable "generally competent" backbone for the drive.

Trains a config-selected backbone on the regime stream's **stationary** all-regime diet until it
settles — by default the SSL loss plateau; with a selection set (``stream.select_*_per_canary``),
optionally the plateau of its labelled accuracy (``warmup.stop=competence``) — then saves the **whole
method** — encoder and head — as a well the deployment drive resumes intact (:mod:`cafl4ds.warmup`).
``warmup.stop=budget`` runs a fixed step budget instead (the P1.1.0 competence-ceiling scout);
``warmup.drive_read_per_class>0`` adds the tight drive-pool read (:mod:`cafl4ds.pool_read`) to every
selection reading — a scout diagnostic, never a stopping signal. The summary and selection trace land
beside the well as ``<well>.trace.json``. Producing the well once
and pointing many drives at it (``well=<path>``) means every warm drive starts from the *identical*
competent state, so drive-to-drive comparisons carry no per-run warm-up variance.

Examples:
    Warm a synthetic well (fresh clone, no dataset)::

        uv run python scripts/warm_well.py data=attr_synthetic img_size=32 well_out=wells/je_well.pt

    Warm the default BDD well for the joint-embedding backbone::

        uv run python scripts/warm_well.py data=bdd bdd_root=/abs/path backbone=je well_out=wells/je_well.pt
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from cafl4ds import pool_read, warmup
from cafl4ds.data.regime import RegimeStream
from cafl4ds.ssl.base import SSLMethod, apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")


def _drive_read(method: SSLMethod, stream: RegimeStream, config: DictConfig) -> Callable[[], dict[str, float]] | None:
    """The scout's tight read — balanced kNN / linear on the drive pool — or ``None`` when off."""
    per_class = int(config.warmup.drive_read_per_class)
    if per_class <= 0:
        return None
    # Fit only on well frames the drive never delivers, so the read never scores its own fit set.
    well = stream.labelled_pool("well", drive_disjoint=True)
    if well.labels.numel() == 0:
        logger.warning("drive read off: the well shares every frame with the drive (well_overlap=1)")
        return None
    fit = pool_read.balanced_subsample(well, per_class, int(config.seed))
    drive = stream.labelled_pool("drive")
    knn_k = int(config.monitor.knn_k)

    def read() -> dict[str, float]:
        scores = pool_read.probe_read(pool_read.frozen(method.encoder), fit, drive, knn_k)
        return {f"drive_{k}": v for k, v in scores.items()}

    return read


def warm(config: DictConfig) -> Path:
    """Warm a competent well on the stationary diet and save the full method (Hydra-free core).

    Kept separate from :func:`main` so it is unit-testable on the synthetic source.

    Args:
        config: The composed ``warm_well`` config.

    Returns:
        The path the warmed well was saved to.
    """
    torch.manual_seed(int(config.seed))
    method: SSLMethod = instantiate(config.ssl, encoder=instantiate(config.encoder))
    apply_encoder_init(method.encoder, config.init.mode, config.init.checkpoint)
    stream = instantiate(config.stream)
    optimizer = instantiate(config.optim, params=method.parameters())
    drive_read = _drive_read(method, stream, config)

    logger.info(f"warming {method.name} well on the stationary diet ({stream.num_eras} regimes), dev={config.device}")
    summary = warmup.warm_until_settled(
        method=method,
        stream=stream,
        optimizer=optimizer,
        plateau=warmup.PlateauCriterion(
            window=int(config.warmup.window), tol=float(config.warmup.tol), min_steps=int(config.warmup.min_steps)
        ),
        competence_rule=warmup.CompetenceCriterion(**config.warmup.competence),
        max_steps=int(config.warmup.max_steps),
        max_passes=int(config.warmup.max_passes),
        grad_clip=config.warmup.grad_clip,
        device=config.device,
        stop=str(config.warmup.stop),
        select_every=int(config.warmup.select_every),
        knn_k=int(config.monitor.knn_k),
        extra_read=drive_read,
    )

    # A competence read on the reserved canary probes — a sanity line, not a gate.
    health = instantiate(config.monitor, eval_sets=stream.eval_sets).measure(method, summary["steps"])
    logger.info(f"well competence: knn_acc={health.get('knn_acc')}, linear_acc={health.get('linear_acc')}")

    well_path = Path(to_absolute_path(str(config.well_out)))
    warmup.save_well(method, well_path)
    # The drive refuses this well unless it reserves the same probes and well slice.
    warmup.save_split(well_path, stream.split_fingerprint())
    trace = {k: summary[k] for k in ("steps", "plateaued", "init_loss", "final_loss", "competence_trace")}
    trace["measurement"] = {k: health.get(k) for k in ("knn_acc", "linear_acc")}
    well_path.with_suffix(".trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    logger.info(f"realized familiarity (shared share of well / drive): {stream.familiarity}")
    logger.info(f"warm well ready — drive it with `well={well_path}`")
    return well_path


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="warm_well")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Warm a competent well on the stationary diet and save the full method."""
    warm(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
