"""Federated Phase-0 streaming-loop entry point (FedAvg over per-client streams).

Partitions one data source across ``num_clients`` clients, gives each its own streaming SSL
loop over its shard, and runs a synchronous federated round loop: every ``steps_per_round``
stream steps the server averages the client weights and applies the result. The FL algorithm is
one config choice — ``strategy`` (FedAvg by default), which sets both the server's rule and any
client-side constraint. Clients continue their single-pass streams across rounds and drop out as
they exhaust; the aggregated model's health is logged once per round.

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

    A different FL strategy (``server_optimizer.lr`` is the *server* rate, not the client one)::

        uv run python scripts/run_federated.py strategy=fedadam strategy.server_optimizer.lr=1e-2
        uv run python scripts/run_federated.py strategy=fedprox strategy.proximal_mu=0.1
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
from cafl4ds.federated.partition import holdout_split, partition_source
from cafl4ds.federated.strategy import FederatedStrategy
from cafl4ds.run_log import RunLogger
from cafl4ds.ssl.base import apply_method_init

logger.remove()
logger.add(sys.stdout, level="INFO")


def _build_stream(
    config: DictConfig, source: DataSource, seed: int, support: int, query: int, era_eval: int
) -> EraStream:
    """Build one client's (or the global) stream from the shared sizing config.

    Reservations are passed in rather than read from ``config``, because the two callers need
    *different* ones: the global monitor reserves a real eval set, while a client reserves
    nothing by default (its local eval sets are never read in a federated run).

    Args:
        config: The composed config (supplies batch size, order, and the train cap).
        source: The data source to order (a client shard, or the global hold-out pool).
        seed: Per-stream RNG seed (offset per client so shards shuffle independently).
        support: Probe-support images reserved per class.
        query: Probe-query images reserved per class.
        era_eval: Per-era held-out images reserved per class.

    Returns:
        The configured :class:`~cafl4ds.data.streams.EraStream`.
    """
    return EraStream(
        source=source,
        batch_size=config.batch_size,
        order=config.stream.order,
        support_per_class=support,
        query_per_class=query,
        era_eval_per_class=era_eval,
        max_train_per_class=config.max_train_per_class,
        seed=seed,
    )


def _build_client(
    config: DictConfig, shard: DataSource, client_id: int, out_dir: Path, strategy: FederatedStrategy
) -> FederatedClient:
    """Instantiate one client's full streaming loop from config, wrapped as a federated client.

    Args:
        config: The composed config.
        shard: This client's data shard.
        client_id: Stable client identifier (also seeds the per-client stream/model).
        out_dir: Hydra run directory the client's run log is written under.
        strategy: The FL strategy, supplying this client's half of the algorithm.

    Returns:
        The assembled :class:`~cafl4ds.federated.client.FederatedClient`.
    """
    torch.manual_seed(config.seed + client_id)
    encoder = instantiate(config.encoder)
    method = instantiate(config.ssl, encoder=encoder)

    checkpoint = config.init.checkpoint
    if config.init.mode == "pretrained" and not checkpoint:
        checkpoint = str(Path(to_absolute_path(config.pretrain_dir)) / f"{method.name}.pt")
    apply_method_init(method, config.init.mode, checkpoint)

    stream = _build_stream(
        config,
        shard,
        seed=config.seed + client_id,
        # Zero by default: a client's local eval sets are never read (health is measured on the
        # global hold-out), so reserving per-client data only starves training — and makes the
        # per-class reservation check fail on sharply skewed shards.
        support=config.client_support_per_class,
        query=config.client_query_per_class,
        era_eval=config.client_era_eval_per_class,
    )
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
        # A fresh term per client: each holds its own per-round anchor, so instances must not
        # be shared. At the strategy's default mu=0 this is an exact no-op.
        proximal=strategy.make_proximal(),
    )
    return FederatedClient(client_id, loop)


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="federated")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Instantiate and run the federated Phase-0 streaming loop from the Hydra config."""
    torch.manual_seed(config.seed)
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    # The FL algorithm, as one object. Built once per run: its server half is stateful across
    # rounds, and each client gets a fresh proximal term from its client half.
    strategy: FederatedStrategy = instantiate(config.strategy)

    source = instantiate(config.data)

    # Carve the global monitor's pool out FIRST, so no client can ever train on an image the
    # aggregated model is scored on. `null` sizes it to the minimum the global stream needs
    # (its own reservations + 1); `0` disables the split, which is only right when the client
    # and global streams coincide anyway (num_clients=1 with matching reservations).
    need = config.support_per_class + config.query_per_class + config.era_eval_per_class
    holdout_per_class = config.global_holdout_per_class
    holdout_per_class = need + 1 if holdout_per_class is None else holdout_per_class
    if holdout_per_class > 0:
        global_pool, client_pool = holdout_split(source, per_class=holdout_per_class, seed=config.seed)
    else:
        logger.warning("global_holdout_per_class=0: the global eval set is NOT held out from client training.")
        global_pool, client_pool = source, source

    shards = partition_source(
        client_pool,
        num_clients=config.num_clients,
        scheme=config.partition.scheme,
        alpha=config.partition.alpha,
        seed=config.seed,
    )
    clients = [_build_client(config, shard, cid, out_dir, strategy) for cid, shard in enumerate(shards)]

    # Global readout: measure the aggregated model on a class-balanced pool that is disjoint from
    # every client's data, so the health series reflects generalization rather than memorization.
    global_stream = _build_stream(
        config,
        global_pool,
        seed=config.seed,
        support=config.support_per_class,
        query=config.query_per_class,
        era_eval=config.era_eval_per_class,
    )
    global_monitor = instantiate(config.monitor, eval_sets=global_stream.eval_sets)
    global_run_name = f"global_{clients[0].method.name}_{config.init.mode}"
    global_logger = RunLogger(out_dir / f"global_{config.run_log}", run_name=global_run_name)

    logger.info(
        f"federated run: strategy={strategy.name}, {config.num_clients} clients, "
        f"{config.partition.scheme} partition (alpha={config.partition.alpha}), "
        f"{config.steps_per_round} steps/round, device={config.device}"
    )
    orchestrator = FederatedOrchestrator(
        clients,
        steps_per_round=config.steps_per_round,
        num_rounds=config.num_rounds,
        global_monitor=global_monitor,
        run_logger=global_logger,
        server_optimizer=strategy.server_optimizer,  # the strategy's server half
    )
    _, history = orchestrator.run()
    logger.info(f"done: {len(history)} rounds; global health log at {global_logger.path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
