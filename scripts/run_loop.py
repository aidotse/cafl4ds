"""Phase-0 streaming-loop entry point.

Wires the config-instantiated components in dependency order (the standard Hydra recipe when
there are runtime cross-references: the optimizer needs the method's parameters, the monitor
needs the stream's held-out eval sets), then runs the loop. Loss and health land in one run
log as separate series.

Examples:
    Single run (STL-10, MAE, from-scratch)::

        uv run python scripts/run_loop.py

    Exit matrix (C x I) as a multirun::

        uv run python scripts/run_loop.py -m ssl=mae,simsiam init=from_scratch,pretrained

    Fast network-free smoke::

        uv run python scripts/run_loop.py data=synthetic img_size=16 eval_every=3
"""

import sys
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from cafl4ds.run_log import RunLogger
from cafl4ds.ssl.base import apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="loop")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Instantiate and run the Phase-0 streaming loop from the Hydra config."""
    torch.manual_seed(config.seed)

    encoder = instantiate(config.encoder)
    method = instantiate(config.ssl, encoder=encoder)

    checkpoint = config.init.checkpoint
    if config.init.mode == "pretrained" and not checkpoint:
        checkpoint = str(Path(to_absolute_path(config.pretrain_dir)) / f"{method.name}.pt")
    apply_encoder_init(method.encoder, config.init.mode, checkpoint)

    stream = instantiate(config.stream)
    optimizer = instantiate(config.optim, params=method.parameters())
    monitor = instantiate(config.monitor, eval_sets=stream.eval_sets)
    selection_filter = instantiate(config.filter)

    run_name = config.run_name or f"{method.name}_{config.init.mode}"
    # Resolve the run log under Hydra's per-run output dir so multirun jobs never collide
    # (Hydra 1.3 defaults job.chdir=False, so a bare relative path lands in the shared cwd).
    run_log_path = Path(HydraConfig.get().runtime.output_dir) / config.run_log
    run_logger = RunLogger(run_log_path, run_name=run_name)
    logger.info(f"run '{run_name}': {stream.num_eras} eras, {len(stream)} batches, device={config.device}")

    loop = instantiate(
        config.loop,
        stream=stream,
        method=method,
        optimizer=optimizer,
        selection_filter=selection_filter,
        monitor=monitor,
        run_logger=run_logger,
    )
    loop.run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
