# Federated — FedAvg over streaming clients

**Not a phase.** Per the [project plan](../../project-plan/index.md#plan), federated learning runs as a **parallel
track** alongside the phased spine — built once Phase 1 confirms degradation, ready by Phase 3 — rather than as one of
the numbered phases itself. This page tracks the FL sub-studies the same way a `phase<ID>/index.md` tracks phase
sub-studies: motivation and status here, routing to detail docs as they're written.

## What's implemented

`cafl4ds/federated/` (entry point [`scripts/run_federated.py`](../../../scripts/run_federated.py)) is a single-process
FedAvg simulation layered on the Phase-0 streaming loop:

- **Partitioning (the `D` factor)** — one data source is split across `num_clients` shards, either `dirichlet`
    (label-skewed non-IID, the headline heterogeneity knob) or `iid` (a uniform control).
- **Per-client loop** — each client owns its own model, optimizer, selection filter, monitor, and single-pass
    `EraStream`, run through the *same* `StreamingLoop.train_step` as the centralized `run_loop.py`. A client's stream
    persists across rounds (true single pass) and it drops out once exhausted.
- **Aggregation** — every `steps_per_round` stream steps, the server FedAvgs client weights, sample-weighted by how much
    each client actually trained this round — the seam where selection-induced skew reaches aggregation (novelty claim
    **N-D**).
- **Global readout** — the aggregated model's health is logged once per round against a global held-out set, distinct
    from any client's skewed local eval set.
- **Per-client divergence** — alongside it, each participant's representation is compared to the broadcast anchor and to
    every other participant on that same shared probe set (`track_divergence`, on by default). The aggregate can average
    away drift the clients plainly exhibit, so this is the per-client half of the dependent variable; [F2](F2.md) is
    where that ambiguity bit.

## Sub-studies

| ID | Sub-study | What it establishes | Status |
| -- | -- | -- | -- |
| F1 | [Federated harness parity](F1.md) | Confirms the FedAvg harness reproduces the centralized [P0.1](../phase0/P0.1.md) reference under a degenerate/IID partition, before any non-IID skew is introduced — the FL analogue of P0.1's "does the loop run end-to-end" check. | ✅ **Complete** — exact reproduction at `num_clients=1`; multi-client FedAvg adapts sensibly under the IID control, both backbones |
| F2 | [Non-IID label skew on a warm-started MAE](F2.md) | The first *scientific* FL question: does `Dirichlet(α)` label skew degrade global representation health over a single federated pass, with no selection filter? MAE has no classifier head, so supervised FL's dominant non-IID penalty should not apply — the question is whether the remaining client-drift mechanism is large enough to read. | 🟡 **Inconclusive — underpowered by design.** Clients train on images the warm start already saw, so the federated pass is worth ~the noise floor and skew has nothing to degrade. Two warm-start strengths × four arms each **disagree on the sign of every apparent effect**, so no non-IID claim survives; only RankMe moves reproducibly (expands in all eight arms, without reaching the probe). Two solid by-products: the encoder-only warm start's head warm-up confound found and **fixed** in the harness, and MAE recon loss shown to be anti-correlated with skew. Blocked on disjoint pretrain/federated data |

## Open questions (not yet scoped into a sub-study)

- **Instrument transfer.** Once the Phase-0 instruments are calibrated (P0.2–P0.4), do they read collapse / forgetting /
    instability the same way on the aggregated global model as on a centralized one?
- **Feature skew instead of label skew.** Label skew reaches a label-free objective only through the feature
    distribution it induces, which over 100 classes is mild ([F2](F2.md)). Per-client pixel transforms (`grayscale` /
    `phase_scramble`, built for [P0.3.8](../phase0/P0.3.8.md)) would give genuinely different input distributions — the
    axis SSL actually responds to.
- **Healthy SimSiam at scale.** SimSiam point-collapses at the scaled ViT-Tiny F2 uses (RankMe ~1.2 against
    [P0.2.1](../phase0/P0.2.1.md)'s ~1.9 collapse floor) without the warmup+cosine schedule `run_loop.py` never wires,
    so the FL track is MAE-only until this is resolved.
