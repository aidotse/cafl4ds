"""Federated Phase-0 streaming-loop entry point (FedAvg over per-client streams).

Partitions one data source across ``num_clients`` clients, gives each its own streaming SSL
loop over its shard, and runs synchronous FedAvg: every ``steps_per_round`` stream steps the
server averages the client weights. Clients continue their single-pass streams across rounds
and drop out as they exhaust; the aggregated model's health is logged once per round.

Mirrors :mod:`scripts.run_loop`'s config-instantiation recipe, one level up: the per-client
components (encoder, method, optimizer, filter, monitor) are instantiated fresh for each client
so nothing is shared but the weights the server aggregates.

Examples:
    Network-free smoke::

        uv run python scripts/run_federated.py data=synthetic img_size=16 num_clients=3 \
            support_per_class=3 query_per_class=2 era_eval_per_class=1 partition.scheme=iid

    STL-10 on GPU::

        uv run python scripts/run_federated.py device=cuda data_root=/home/edgelab/stl10 \
            ssl=simsiam num_clients=4 partition.alpha=1.0 batch_size=64
"""

import sys
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from cafl4ds.data.sources import DataSource
from cafl4ds.data.streams import EraStream
from cafl4ds.federated.client import FederatedClient
from cafl4ds.federated.orchestrator import FederatedOrchestrator
from cafl4ds.federated.partition import partition_source
from cafl4ds.run_log import RunLogger
from cafl4ds.ssl.base import apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")


def _build_stream(config: DictConfig, source: DataSource, seed: int) -> EraStream:
    """Build one client's (or the global) stream from the shared sizing config.

    Args:
        config: The composed config (supplies batch size, order, and held-out reservations).
        source: The data source to order (a client shard, or the full dataset for the global set).
        seed: Per-stream RNG seed (offset per client so shards shuffle independently).

    Returns:
        The configured :class:`~cafl4ds.data.streams.EraStream`.
    """
    return EraStream(
        source=source,
        batch_size=config.batch_size,
        order=config.stream.order,
        support_per_class=config.support_per_class,
        query_per_class=config.query_per_class,
        era_eval_per_class=config.era_eval_per_class,
        max_train_per_class=config.max_train_per_class,
        seed=seed,
    )


def _build_client(config: DictConfig, shard: DataSource, client_id: int, out_dir: Path) -> FederatedClient:
    """Instantiate one client's full streaming loop from config, wrapped as a federated client.

    Args:
        config: The composed config.
        shard: This client's data shard.
        client_id: Stable client identifier (also seeds the per-client stream/model).
        out_dir: Hydra run directory the client's run log is written under.

    Returns:
        The assembled :class:`~cafl4ds.federated.client.FederatedClient`.
    """
    torch.manual_seed(config.seed + client_id)
    encoder = instantiate(config.encoder)
    method = instantiate(config.ssl, encoder=encoder)

    checkpoint = config.init.checkpoint
    if config.init.mode == "pretrained" and not checkpoint:
        checkpoint = str(Path(to_absolute_path(config.pretrain_dir)) / f"{method.name}.pt")
    apply_encoder_init(method.encoder, config.init.mode, checkpoint)

    stream = _build_stream(config, shard, seed=config.seed + client_id)
    optimizer = instantiate(config.optim, params=method.parameters())
    monitor = instantiate(config.monitor, eval_sets=stream.eval_sets)
    selection_filter = instantiate(config.filter)
    run_name = f"client{client_id}_{method.name}_{config.init.mode}"
    run_logger = RunLogger(out_dir / f"client{client_id}_{config.run_log}", run_name=run_name)
    loop = instantiate(
        config.loop,
        stream=stream,
        method=method,
        optimizer=optimizer,
        selection_filter=selection_filter,
        monitor=monitor,
        run_logger=run_logger,
    )
    return FederatedClient(client_id, loop)


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="federated")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Instantiate and run the federated Phase-0 streaming loop from the Hydra config."""
    torch.manual_seed(config.seed)
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    source = instantiate(config.data)
    shards = partition_source(
        source,
        num_clients=config.num_clients,
        scheme=config.partition.scheme,
        alpha=config.partition.alpha,
        seed=config.seed,
    )
    clients = [_build_client(config, shard, cid, out_dir) for cid, shard in enumerate(shards)]

    # Global readout: measure the aggregated model on an IID view of the *full* dataset, so the
    # health series is not tied to any one client's skewed local held-out set.
    global_stream = _build_stream(config, source, seed=config.seed)
    global_monitor = instantiate(config.monitor, eval_sets=global_stream.eval_sets)
    global_run_name = f"global_{clients[0].method.name}_{config.init.mode}"
    global_logger = RunLogger(out_dir / f"global_{config.run_log}", run_name=global_run_name)

    logger.info(
        f"federated run: {config.num_clients} clients, {config.partition.scheme} partition "
        f"(alpha={config.partition.alpha}), {config.steps_per_round} steps/round, device={config.device}"
    )
    orchestrator = FederatedOrchestrator(
        clients,
        steps_per_round=config.steps_per_round,
        num_rounds=config.num_rounds,
        global_monitor=global_monitor,
        run_logger=global_logger,
    )
    _, history = orchestrator.run()
    logger.info(f"done: {len(history)} rounds; global health log at {global_logger.path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
