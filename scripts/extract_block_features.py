r"""P1.1.2 — extract per-block features from saved checkpoints, for the layer-wise lenses.

The lenses (per-block CKA / Procrustes against the well, intrinsic dimension per block) are computed
post hoc on CPU by ``scripts/analyze_lenses.py``. This script does the accelerator part once: for
every saved checkpoint it reads the pooled output of each encoder block, plus the monitor's surfaces
(:func:`cafl4ds.block_features.block_features`), on the drive's fixed probe set — support and query
both, all label-free reads. Each checkpoint's features land in ``features/<run>/s<seed>/<name>.pt``
(``layer -> [N, d]`` float32); an existing file is skipped, so a run resumes where it stopped.

Three sources, all from the run dirs' own records:

* the **well** (``config.well``) — every lens's reference, ``features/well.pt``;
* **drives** (``lens.drives``) — every leg checkpoint in ``checkpoints/s<seed>/index.json`` and
  ``final.pt``;
* **branches** (``lens.branches``) — ``final.pt`` per seed, plus ``post_damage.pt``: the state right
  after the damage, rebuilt from the run's ``resume.checkpoint`` and ``damage`` config exactly as the
  harness applied it (load, move to the device, damage with the drive seed). The pre-damage state is
  the parent drive's ``step00065.pt``, read under ``drives``.

``probe.pt`` records which rows are query images (what the monitor reads) and the canary labels and
strata, kept for analysis only — no lens reads them.

Example (Gaudi node; the drive's diet knobs, so the probes are the drive's)::

    ./scripts/run_gaudi_dev.sh -m $BDD -m /mnt/cifar100 gaudi-env-cafl4ds:latest 0 \\
        python scripts/extract_block_features.py $DIET backbone=mae well=wells/p110_deep/mae_o0.0.pt \\
        'lens.drives=[outputs/p110_deep/warm/mae_o0.0_iid]' 'lens.branches=[outputs/p111/branch/mae_iid_noise1.0]' \\
        hydra.run.dir=outputs/p112/features_mae
"""

import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cafl4ds import branch, warmup
from cafl4ds.block_features import block_features
from cafl4ds.ssl.base import SSLMethod

logger.remove()
logger.add(sys.stdout, level="INFO")


def _save_probe(config: DictConfig, out: Path) -> torch.Tensor:
    """Write ``probe.pt`` (query mask, labels, strata) and return the probe images, support then query."""
    source = instantiate(config.data)
    sets = instantiate(config.stream, source=source).eval_sets
    support, query = sets.probe_support, sets.probe_query
    images = torch.cat([support.images, query.images])
    n_s, n_q = support.images.shape[0], query.images.shape[0]
    record: dict[str, Any] = {
        "is_query": torch.cat([torch.zeros(n_s, dtype=torch.bool), torch.ones(n_q, dtype=torch.bool)]),
        "labels": torch.cat([support.labels, query.labels]),
        "query_groups": query.groups,
        "group_names": dict(sets.group_names),
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save(record, out / "probe.pt")
    logger.info(f"probe set: {n_s} support + {n_q} query images")
    return images


class _Reader:
    """Loads a checkpoint into one reusable method on the device and saves its block features."""

    def __init__(self, config: DictConfig, images: torch.Tensor, out: Path) -> None:
        self.method: SSLMethod = instantiate(config.ssl, encoder=instantiate(config.encoder))
        self.method.to(config.device)
        self.images = images
        self.out = out
        self.batch_size = int(config.lens.batch_size)
        self.index: list[dict[str, Any]] = []

    def read(self, checkpoint: str | Path, dest: Path, damage: dict[str, Any] | None = None) -> None:
        """Read ``checkpoint`` (damaged first if ``damage`` is given) into ``dest`` unless it exists."""
        entry: dict[str, Any] = {"checkpoint": str(checkpoint), "features": str(dest.relative_to(self.out))}
        if not dest.is_file():
            warmup.load_well(self.method, checkpoint)
            if damage is not None:
                entry["damage"] = {
                    **damage,
                    **branch.damage_top_blocks(
                        self.method.encoder, damage["mode"], damage["k"], damage["scale"], seed=damage["seed"]
                    ),
                }
            feats = block_features(self.method, self.images, batch_size=self.batch_size)
            dest.parent.mkdir(parents=True, exist_ok=True)
            torch.save(feats, dest)
            logger.info(f"{entry['features']}: {len(feats)} layers")
        self.index.append(entry)


def _drive(reader: _Reader, run: Path) -> None:
    """Read every indexed leg checkpoint (and ``final.pt``) of each seed of a drive run."""
    for seed_dir in sorted((run / "checkpoints").glob("s*")):
        for item in json.loads((seed_dir / "index.json").read_text(encoding="utf-8")):
            ckpt = seed_dir / str(item["file"])
            reader.read(ckpt, reader.out / "features" / run.name / seed_dir.name / f"{ckpt.stem}.pt")


def _branch(reader: _Reader, run: Path) -> None:
    """Read ``final.pt`` and the rebuilt post-damage state of each seed of a branch run."""
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    mode, k, scale = str(cfg.damage.mode), int(cfg.damage.k), float(cfg.damage.scale)
    for seed_dir in sorted((run / "checkpoints").glob("s*")):
        seed = int(seed_dir.name[1:])
        dest = reader.out / "features" / run.name / seed_dir.name
        reader.read(seed_dir / "final.pt", dest / "final.pt")
        if mode != "none" and k > 0:
            # The harness damages with `damage.seed`, which the drive configs interpolate to the drive seed.
            parent = to_absolute_path(str(cfg.resume.checkpoint).format(seed=seed))
            damage = {"mode": mode, "k": k, "scale": scale, "seed": seed}
            reader.read(parent, dest / "post_damage.pt", damage=damage)


def extract(config: DictConfig, out: Path) -> list[dict[str, Any]]:
    """Extract the well's, the drives' and the branches' per-block features into ``out``; return the index."""
    torch.manual_seed(int(config.seed))
    reader = _Reader(config, _save_probe(config, out), out)
    reader.read(to_absolute_path(str(config.well)), out / "features" / "well.pt")
    for run in config.lens.drives:
        _drive(reader, Path(to_absolute_path(str(run))))
    for run in config.lens.branches:
        _branch(reader, Path(to_absolute_path(str(run))))
    (out / "index.json").write_text(json.dumps(reader.index, indent=2), encoding="utf-8")
    logger.info(f"{len(reader.index)} checkpoints indexed under {out}")
    return reader.index


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="deploy")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Extract per-block features into the Hydra run dir."""
    extract(config, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
