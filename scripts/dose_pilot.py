r"""P1.1.1 dose pilot — read the canary and the instruments on a damaged well, with no training.

The surgical positive control (:func:`cafl4ds.branch.damage_top_blocks`) needs doses that span "no
dent" to "near chance" before any drive is spent on them. This reads each candidate dose once: load
the well (or any saved checkpoint), damage a fresh copy, and run the health monitor on the canary
probes. Minutes, not hours. Writes ``dose_pilot.json`` into the run dir and prints a table.

Example (Gaudi node; the drive's diet knobs, so the probes are the drive's)::

    ./scripts/run_gaudi_dev.sh -m $BDD -m /mnt/cifar100 gaudi-env-cafl4ds:latest 0 python scripts/dose_pilot.py \\
        $DIET backbone=je well=wells/p110_deep/je_o0.0.pt \\
        'doses=[reinit:1,reinit:2,reinit:4,noise:12:0.25,noise:12:0.5]'
"""

import copy
import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from cafl4ds import branch, warmup
from cafl4ds.ssl.base import SSLMethod

logger.remove()
logger.add(sys.stdout, level="INFO")

_SHOW = ("knn_acc", "linear_acc", "rankme", "rankme_proj", "cka_drift", "cka_drift_proj")


def parse_dose(spec: str) -> dict[str, Any]:
    """Parse ``mode:k[:scale]`` (e.g. ``reinit:2``, ``noise:12:0.5``) into damage arguments.

    Raises:
        ValueError: If the spec has the wrong number of fields.
    """
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"dose spec must be mode:k[:scale]; got {spec!r}.")
    return {"mode": parts[0], "k": int(parts[1]), "scale": float(parts[2]) if len(parts) == 3 else 0.0}


def pilot(config: DictConfig, out_dir: Path) -> list[dict[str, Any]]:
    """Read every dose in ``config.doses`` on the well (dose ``none:0`` is read first, as the reference)."""
    torch.manual_seed(int(config.seed))
    source = instantiate(config.data)
    stream = instantiate(config.stream, source=source)
    well = str(config.resume.checkpoint or config.well).format(seed=int(config.seed))
    base: SSLMethod = instantiate(config.ssl, encoder=instantiate(config.encoder))
    warmup.load_well(base, to_absolute_path(well))
    base.to(config.device)
    monitor = instantiate(config.monitor, eval_sets=stream.eval_sets)
    monitor.anchor_drift(base)
    rows = []
    for spec in ["none:0", *config.doses]:
        dose = parse_dose(str(spec))
        method = copy.deepcopy(base)
        dent = branch.damage_top_blocks(method.encoder, dose["mode"], dose["k"], dose["scale"], seed=int(config.seed))
        rows.append({"dose": str(spec), **dose, **dent, "metrics": monitor.measure(method, 0)})
        m = rows[-1]["metrics"]
        logger.info(
            f"{spec:>14}  rel={dent['rel_change']:.3f}  " + "  ".join(f"{k}={m[k]:.3f}" for k in _SHOW if k in m)
        )
    (out_dir / "dose_pilot.json").write_text(json.dumps({"well": well, "rows": rows}, indent=2), encoding="utf-8")
    return rows


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="deploy")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Read the dose ladder on the well."""
    pilot(config, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
