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
from cafl4ds.federated.server_optim import (
    AdaptiveServerOptimizer,
    FedAdagradServer,
    FedAdamServer,
    FedAvgServer,
    FedYogiServer,
)
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


# --- server optimizers -------------------------------------------------------


def _states(old: list[float], new: list[float]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A minimal (old, aggregated) state pair with one float parameter and one int buffer."""
    return (
        {"w": torch.tensor(old), "n": torch.tensor(3)},
        {"w": torch.tensor(new), "n": torch.tensor(7)},
    )


def test_fedavg_server_is_the_identity() -> None:
    """FedAvg's server step returns the aggregate untouched — the Phase-0 baseline."""
    old, aggregated = _states([0.0, 0.0], [1.0, 2.0])
    assert FedAvgServer().step(old, aggregated) is aggregated


@pytest.mark.parametrize("server", [FedAdamServer(), FedYogiServer(), FedAdagradServer()])
def test_adaptive_server_keeps_every_key_and_skips_int_buffers(server: AdaptiveServerOptimizer) -> None:
    """Integer bookkeeping buffers pass through, so the result still loads strictly."""
    old, aggregated = _states([0.0, 0.0], [1.0, -1.0])
    out = server.step(old, aggregated)
    assert set(out) == {"w", "n"}  # a full key set: load_state_dict is strict by default
    assert out["n"].item() == 7  # carried from the aggregate, not optimized
    assert out["n"].dtype == torch.int64  # no float contamination of a step counter
    assert out["w"].dtype == torch.float32


@pytest.mark.parametrize("server", [FedAdamServer(), FedYogiServer(), FedAdagradServer()])
def test_adaptive_server_moves_in_the_pseudo_gradient_direction(server: AdaptiveServerOptimizer) -> None:
    """Each coordinate moves the way the clients collectively moved it."""
    old, aggregated = _states([0.0, 0.0, 0.0], [1.0, -1.0, 0.0])
    out = server.step(old, aggregated)
    assert out["w"][0] > 0.0  # clients pushed up
    assert out["w"][1] < 0.0  # clients pushed down
    assert out["w"][2] == pytest.approx(0.0)  # no signal, no move


def test_adaptive_server_step_is_scale_free_and_bounded_by_lr() -> None:
    """The update is normalized: a huge and a tiny delta both move by about `lr`.

    This is the property that makes the server `lr` a weight-space step per round rather than a
    gradient multiplier, and the reason `tau` matters — see the module docstring.
    """
    server = FedAdamServer(lr=0.01, tau=1e-8)
    out = server.step(*_states([0.0, 0.0], [1000.0, 0.001]))
    assert out["w"][0] == pytest.approx(0.01, rel=0.05)
    assert out["w"][1] == pytest.approx(0.01, rel=0.05)


def test_tau_damps_coordinates_the_clients_barely_moved() -> None:
    """A larger `tau` keeps small deltas small instead of amplifying them to a full step."""
    small = _states([0.0], [1e-4])
    adaptive = FedAdamServer(lr=0.01, tau=1e-8).step(*small)
    damped = FedAdamServer(lr=0.01, tau=1.0).step(*small)
    assert adaptive["w"][0] == pytest.approx(0.01, rel=0.05)  # amplified to the full step
    assert damped["w"][0] < adaptive["w"][0] / 100  # tau dominates the denominator


def test_adaptive_server_carries_state_across_rounds() -> None:
    """Moment estimates and the bias-correction counter persist round to round."""
    server = FedAdamServer()
    server.step(*_states([0.0], [1.0]))
    server.step(*_states([0.0], [1.0]))
    assert server.round == 2
    assert set(server.m) == {"w"} and set(server.v) == {"w"}  # float keys only get moment state
    assert server.m["w"].item() > 0.0


def test_second_moment_rules_differ_as_documented() -> None:
    """Adagrad's second moment only grows; Adam's decays when the deltas shrink."""
    adagrad, adam = FedAdagradServer(), FedAdamServer()
    for aggregated in (1.0, 1.0, 0.0):
        adagrad.step(*_states([0.0], [aggregated]))
        adam.step(*_states([0.0], [aggregated]))
    assert adagrad.v["w"].item() == pytest.approx(2.0 + adagrad.v_init)  # 1 + 1 + 0, a plain sum
    # Adam's EMA forgets: after a zero-delta round its estimate has decayed below Adagrad's sum.
    assert adam.v["w"].item() < adagrad.v["w"].item()


def test_bias_correction_only_changes_the_early_rounds() -> None:
    """Debiasing lifts the first step towards its steady-state size; `False` matches the paper."""
    corrected = FedAdamServer(bias_correction=True).step(*_states([0.0], [1.0]))
    published = FedAdamServer(bias_correction=False).step(*_states([0.0], [1.0]))
    assert corrected["w"][0] > published["w"][0]


def test_adaptive_server_rejects_bad_hyperparameters_and_states() -> None:
    """Hyperparameters are validated up front, and a key-set mismatch is a hard error."""
    with pytest.raises(ValueError, match="server lr"):
        FedAdamServer(lr=0.0)
    with pytest.raises(ValueError, match="beta_1"):
        FedAdamServer(beta_1=1.0)
    with pytest.raises(ValueError, match="beta_2"):
        FedAdamServer(beta_2=-0.1)
    with pytest.raises(ValueError, match="tau"):
        FedAdamServer(tau=0.0)
    with pytest.raises(ValueError, match="v_init"):
        FedAdagradServer(v_init=-1.0)
    with pytest.raises(ValueError, match="key sets differ"):
        FedAdamServer().step({"w": torch.zeros(1)}, {"other": torch.zeros(1)})


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


def test_orchestrator_defaults_to_fedavg(tmp_path: Path) -> None:
    """No server optimizer means plain FedAvg — the pre-existing behaviour, unchanged."""
    clients = _make_clients(tmp_path, num_clients=2)
    orch = FederatedOrchestrator(clients, steps_per_round=1, num_rounds=1)
    assert orch.server_optimizer.name == "fedavg"
    final_state, history = orch.run()
    # The identity server step: after one round the global model IS the weighted client mean.
    # (The clients themselves are not re-synced after the final aggregation, so read their
    # post-round local weights and average them here.)
    expected = federated_average(
        [c.get_weights() for c in clients], weights_from_samples([history[0].samples // 2] * 2)
    )
    for key, value in expected.items():
        assert torch.equal(final_state[key], value)


@pytest.mark.parametrize("server", ["fedadam", "fedyogi", "fedadagrad"])
def test_orchestrator_runs_with_an_adaptive_server(tmp_path: Path, server: str) -> None:
    """Each adaptive server drives the real round loop over a real encoder's state_dict.

    The point of running these end to end is the mixed-dtype state_dict: a live model carries
    integer buffers that the adaptive arithmetic must step over, and `_record_round` then loads
    the result with `strict=True`.
    """
    optimizers = {"fedadam": FedAdamServer, "fedyogi": FedYogiServer, "fedadagrad": FedAdagradServer}
    clients = _make_clients(tmp_path, num_clients=2)
    orch = FederatedOrchestrator(clients, steps_per_round=1, num_rounds=3, server_optimizer=optimizers[server](lr=1e-3))
    final_state, history = orch.run()
    assert len(history) == 3
    assert set(final_state) == set(clients[0].method.state_dict())  # a strictly loadable model
    assert all(torch.isfinite(v).all() for v in final_state.values() if v.is_floating_point())


def test_adaptive_server_leaves_batchnorm_statistics_at_their_mean(tmp_path: Path) -> None:
    """Buffers are measurements, not parameters: an adaptive server must not step them.

    SimSiam's projector/predictor use BatchNorm, so `running_mean`/`running_var` are *float*
    buffers — dtype alone cannot distinguish them from weights. Adam-stepping them moves each
    coordinate by ~`lr` regardless of the true change, which drifts a running variance at a rate
    set by the hyperparameters and can walk it below zero into NaN.
    """
    clients = _make_clients(tmp_path, num_clients=2)
    orch = FederatedOrchestrator(clients, steps_per_round=1, num_rounds=1, server_optimizer=FedAdamServer(lr=1e-2))
    buffer_keys = [k for k in clients[0].method.state_dict() if "running_" in k]
    assert buffer_keys  # the premise: this model really does carry float BN buffers
    final_state, history = orch.run()
    weights = weights_from_samples([history[0].samples // 2] * 2)
    expected = federated_average([c.get_weights() for c in clients], weights)
    for key in buffer_keys:
        assert torch.equal(final_state[key], expected[key])  # the plain FedAvg mean, unstepped
    for key in [k for k in final_state if "running_var" in k]:
        assert (final_state[key] >= 0.0).all()  # a variance cannot go negative


def test_orchestrator_steps_parameters_and_passes_buffers_through(tmp_path: Path) -> None:
    """The parameter/buffer split covers the state_dict exactly, with no key lost either way."""
    clients = _make_clients(tmp_path, num_clients=2)
    orch = FederatedOrchestrator(clients, steps_per_round=1, num_rounds=1)
    state_keys = set(clients[0].method.state_dict())
    assert orch._param_keys < state_keys  # a strict subset: some entries are buffers
    buffers = state_keys - orch._param_keys
    # Every non-parameter is a BatchNorm statistic or a step counter — nothing learnable is
    # accidentally excluded from the server step.
    assert all("running_" in k or "num_batches_tracked" in k for k in buffers), buffers


def test_adaptive_server_diverges_from_fedavg(tmp_path: Path) -> None:
    """The adaptive path actually changes the global model, rather than silently no-op'ing."""
    baseline, _ = FederatedOrchestrator(_make_clients(tmp_path / "a", 2), steps_per_round=1, num_rounds=2).run()
    adaptive, _ = FederatedOrchestrator(
        _make_clients(tmp_path / "b", 2), steps_per_round=1, num_rounds=2, server_optimizer=FedAdamServer(lr=1e-2)
    ).run()
    key = "encoder.blocks.0.norm1.weight"
    assert not torch.allclose(baseline[key], adaptive[key])


def test_orchestrator_validates_inputs(tmp_path: Path) -> None:
    """The orchestrator rejects an empty client list and a non-positive round size."""
    with pytest.raises(ValueError, match="at least one client"):
        FederatedOrchestrator([], steps_per_round=1)
    clients = _make_clients(tmp_path, num_clients=1)
    with pytest.raises(ValueError, match="steps_per_round"):
        FederatedOrchestrator(clients, steps_per_round=0)
