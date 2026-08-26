# Federated — FedAvg over streaming clients

**Not a phase.** Per the [project plan](../../project-plan/index.md#plan), federated learning runs as a **parallel
track** alongside the phased spine — built once Phase 1 confirms degradation, ready by Phase 3 — rather than as one of
the numbered phases itself. This page tracks the FL sub-studies the same way a `phase<ID>/index.md` tracks phase
sub-studies: motivation and status here, routing to detail docs as they're written.

## What's implemented

`cafl4ds/federated/` (entry point [`scripts/run_federated.py`](../../../scripts/run_federated.py)) is a
single-process FedAvg simulation layered on the Phase-0 streaming loop:

- **Partitioning (the `D` factor)** — one data source is split across `num_clients` shards, either `dirichlet`
    (label-skewed non-IID, the headline heterogeneity knob) or `iid` (a uniform control).
- **Per-client loop** — each client owns its own model, optimizer, selection filter, monitor, and single-pass
    `EraStream`, run through the *same* `StreamingLoop.train_step` as the centralized `run_loop.py`. A client's stream
    persists across rounds (true single pass) and it drops out once exhausted.
- **Aggregation** — every `steps_per_round` stream steps, the server FedAvgs client weights, sample-weighted by how
    much each client actually trained this round — the seam where selection-induced skew reaches aggregation
    (novelty claim **N-D**).
- **Global readout** — the aggregated model's health is logged once per round against a global held-out set, distinct
    from any client's skewed local eval set.

## Sub-studies

| ID | Sub-study | What it establishes | Status |
| -- | -- | -- | -- |
| F1 | [Federated harness parity](F1.md) | Confirms the FedAvg harness reproduces the centralized [P0.1](../phase0/P0.1.md) reference under a degenerate/IID partition, before any non-IID skew is introduced — the FL analogue of P0.1's "does the loop run end-to-end" check. | ✅ **Complete** — exact reproduction at `num_clients=1`; multi-client FedAvg adapts sensibly under the IID control, both backbones |

## Open questions (not yet scoped into a sub-study)

- **Non-IID sweep.** How much does label skew alone (`partition.scheme=dirichlet`, varying `alpha`) move global
    health, before any selection filter is introduced?
- **Instrument transfer.** Once the Phase-0 instruments are calibrated (P0.2–P0.4), do they read collapse / forgetting
    / instability the same way on the aggregated global model as on a centralized one?
