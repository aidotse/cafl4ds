"""Unit tests for the federated package — partition, aggregate, client, orchestrator.

Covers the FedAvg simulation end to end on the network-free synthetic source: a non-IID
partition, a resumable per-client streaming loop (true single pass across rounds), weighted
aggregation (including non-float buffer handling), and the synchronous round loop with staggered
client exhaustion.
"""

from pathlib import Path

import pytest
import torch
from torch import optim

from cafl4ds.data.sources import DataSource, SyntheticSource
from cafl4ds.data.streams import EraStream
from cafl4ds.federated.aggregate import federated_average, weights_from_samples
from cafl4ds.federated.client import FederatedClient
from cafl4ds.federated.orchestrator import FederatedOrchestrator
from cafl4ds.federated.partition import dirichlet_partition, iid_partition, partition_source
from cafl4ds.filters.accept_all import AcceptAll
from cafl4ds.loop import StreamingLoop
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.monitor import HealthMonitor
from cafl4ds.run_log import RunLogger
from cafl4ds.ssl.factory import build_simsiam

_RES = {"support_per_class": 3, "query_per_class": 2, "era_eval_per_class": 1}


def _make_loop(source: DataSource, seed: int, log_path: Path, order: str = "class_blocked") -> StreamingLoop:
    """Build a tiny synthetic streaming loop (one client's worth of machinery)."""
    torch.manual_seed(seed)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, in_chans=3, embed_dim=32, depth=2, num_heads=2)
    method = build_simsiam(encoder)
    stream = EraStream(source, batch_size=8, order=order, seed=seed, **_RES)
    return StreamingLoop(
        stream=stream,
        method=method,
        optimizer=optim.AdamW(method.parameters(), lr=1e-3),
        selection_filter=AcceptAll(),
        monitor=HealthMonitor(stream.eval_sets, knn_k=3),
        run_logger=RunLogger(log_path, run_name=log_path.stem),
        device="cpu",
    )


# --- partition ---------------------------------------------------------------


def test_dirichlet_partition_is_a_disjoint_cover() -> None:
    """Dirichlet partition assigns every image to exactly one client."""
    labels = SyntheticSource(num_classes=4, per_class=50).load()[1]
    parts = dirichlet_partition(labels, num_clients=3, alpha=10.0, seed=0)
    assert len(parts) == 3
    all_idx = torch.cat(parts)
    assert all_idx.numel() == labels.numel()  # every image assigned exactly once
    assert set(all_idx.tolist()) == set(range(labels.numel()))


def test_iid_partition_is_even() -> None:
    """IID partition splits images evenly across clients with no overlap."""
    labels = SyntheticSource(num_classes=4, per_class=30).load()[1]  # 120 images
    parts = iid_partition(labels, num_clients=4, seed=1)
    assert [p.numel() for p in parts] == [30, 30, 30, 30]
    assert torch.cat(parts).unique().numel() == 120


def test_partition_source_returns_usable_shards() -> None:
    """partition_source yields loadable shards that keep the global class count."""
    source = SyntheticSource(num_classes=4, per_class=60, img_size=16, seed=0)
    shards = partition_source(source, num_clients=2, scheme="iid", seed=0)
    assert len(shards) == 2
    imgs, lbls = shards[0].load()
    assert imgs.shape[0] == lbls.shape[0] > 0
    assert shards[0].num_classes == 4  # global class count preserved


def test_partition_source_rejects_bad_args() -> None:
    """partition_source validates client count and scheme name."""
    source = SyntheticSource(num_classes=2, per_class=10)
    with pytest.raises(ValueError, match="num_clients"):
        partition_source(source, num_clients=0)
    with pytest.raises(ValueError, match="unknown partition scheme"):
        partition_source(source, num_clients=2, scheme="nope")


# --- aggregate ---------------------------------------------------------------


def test_weights_from_samples() -> None:
    """Sample counts normalize to mixing weights, with a uniform zero-count fallback."""
    assert weights_from_samples([1, 3]) == [0.25, 0.75]
    assert weights_from_samples([0, 0]) == [0.5, 0.5]  # uniform fallback
    assert weights_from_samples([]) == []


def test_federated_average_weights_floats_and_preserves_buffers() -> None:
    """FedAvg weights float tensors and carries non-float buffers from the first state."""
    a = {"w": torch.tensor([2.0, 0.0]), "n": torch.tensor(5)}
    b = {"w": torch.tensor([4.0, 8.0]), "n": torch.tensor(9)}
    out = federated_average([a, b], [0.25, 0.75])
    assert torch.allclose(out["w"], torch.tensor([3.5, 6.0]))  # weighted mean
    assert out["n"].item() == 5  # non-float buffer taken from the first state
    assert a["w"].tolist() == [2.0, 0.0]  # inputs untouched


def test_federated_average_validates_inputs() -> None:
    """federated_average rejects empty input and mismatched lengths."""
    with pytest.raises(ValueError, match="at least one"):
        federated_average([], [])
    with pytest.raises(ValueError, match="length mismatch"):
        federated_average([{"w": torch.zeros(1)}], [0.5, 0.5])


# --- client ------------------------------------------------------------------


def test_set_weights_copies_the_model(tmp_path: Path) -> None:
    """set_weights overwrites the local model with the broadcast weights."""
    src = SyntheticSource(num_classes=3, per_class=30, img_size=16, seed=0)
    a = FederatedClient(0, _make_loop(src, seed=1, log_path=tmp_path / "a.jsonl"))
    b = FederatedClient(1, _make_loop(src, seed=2, log_path=tmp_path / "b.jsonl"))
    # Distinct random inits: at least one parameter differs.
    assert any(not torch.equal(a.method.state_dict()[k], v) for k, v in b.method.state_dict().items())
    a.set_weights(b.get_weights())
    for key, value in b.method.state_dict().items():
        assert torch.equal(a.method.state_dict()[key], value)


def test_client_is_true_single_pass_across_rounds(tmp_path: Path) -> None:
    """A client resumes its stream across rounds for exactly one full pass."""
    src = SyntheticSource(num_classes=3, per_class=40, img_size=16, seed=0)
    client = FederatedClient(0, _make_loop(src, seed=0, log_path=tmp_path / "c.jsonl"))
    total_batches = len(client.loop.stream)
    pulled = 0
    while not client.exhausted:
        pulled += client.train_round(steps_per_round=2).steps_pulled
    # Continues where it left off and never restarts: exactly one pass over the stream.
    assert pulled == total_batches


# --- orchestrator ------------------------------------------------------------


def _make_clients(tmp_path: Path, num_clients: int, per_class: int = 60) -> list[FederatedClient]:
    source = SyntheticSource(num_classes=4, per_class=per_class, img_size=16, seed=0)
    shards = partition_source(source, num_clients=num_clients, scheme="iid", seed=0)
    return [
        FederatedClient(cid, _make_loop(shard, seed=cid, log_path=tmp_path / f"c{cid}.jsonl"))
        for cid, shard in enumerate(shards)
    ]


def test_orchestrator_runs_to_exhaustion_and_logs_health(tmp_path: Path) -> None:
    """The round loop drains every client and logs global health each round."""
    clients = _make_clients(tmp_path, num_clients=3)
    global_stream = EraStream(
        SyntheticSource(num_classes=4, per_class=40, img_size=16, seed=1), batch_size=8, order="iid", seed=1, **_RES
    )
    logger = RunLogger(tmp_path / "global.jsonl", run_name="global")
    orch = FederatedOrchestrator(
        clients, steps_per_round=3, global_monitor=HealthMonitor(global_stream.eval_sets, knn_k=3), run_logger=logger
    )
    final_state, history = orch.run()
    assert history  # at least one round happened
    assert all(c.exhausted for c in clients)  # ran until every client drained
    assert "rankme" in history[0].health  # global health measured each round
    assert set(final_state) == set(clients[0].method.state_dict())  # a full model came out


def test_orchestrator_respects_num_rounds_cap(tmp_path: Path) -> None:
    """num_rounds caps the run before the clients drain."""
    clients = _make_clients(tmp_path, num_clients=2)
    orch = FederatedOrchestrator(clients, steps_per_round=1, num_rounds=2)
    _, history = orch.run()
    assert len(history) == 2
    assert not all(c.exhausted for c in clients)  # capped before draining


def test_orchestrator_single_client_is_centralized(tmp_path: Path) -> None:
    """A single-client federation reduces to a centralized run to exhaustion."""
    clients = _make_clients(tmp_path, num_clients=1)
    _, history = clients and FederatedOrchestrator(clients, steps_per_round=4).run()
    assert history
    assert clients[0].exhausted
    assert all(h.participants == [0] for h in history)  # the lone client every round


def test_orchestrator_validates_inputs(tmp_path: Path) -> None:
    """The orchestrator rejects an empty client list and a non-positive round size."""
    with pytest.raises(ValueError, match="at least one client"):
        FederatedOrchestrator([], steps_per_round=1)
    clients = _make_clients(tmp_path, num_clients=1)
    with pytest.raises(ValueError, match="steps_per_round"):
        FederatedOrchestrator(clients, steps_per_round=0)
