# Continuous, Active Federated Learning for Data Streams

## Research & Experiment Plan

*Working document for project lead, partners, and potential students. This page is the **always-read core** — the
framing every non-trivial task needs. Detail lives in the sibling files linked below; open one only when the task calls
for it (see the [routing table](#where-to-read-more)).*

**Tags:** `[STD]` standard / re-implement · `[EXT]` extend an existing method to our setting · `[NEW]` genuine
contribution absent from prior art.

## Scope

![](../assets/coupled_loop_adaptive.svg)

**In scope**: learning loop: AL as a label-free streaming filter ↔ trainable SSL backbone, trained using FL over these
transient data streams → learning health monitoring / continual learning health metrics → filter correction. For the AL
filter, analyze the performance-vs-storage Pareto front.

**Out of scope**: solving catastrophic forgetting (CF) end-to-end (instead: demonstrate + opportunistically mitigate),
or FL efficiency or privacy attacks (rely on privacy-by-design).

**Experimental framing:** the *independent variable* (the knob we turn) is the filter selection, together with a choice
of **open-loop** or **closed-loop** setting (i.e. whether the filter is static, or steerable). The health metrics are
the *dependent variable / readout* (the thermometer), and the finding about the coupled system is the contribution, not
any single filter, unless we can show a filter design (e.g. steerable, vs. static) that outperforms others on the health
and downstream (e.g. classifier-level) metrics.

In **open-loop** (Phases 0–2) the base filter is the independent variable: fixed per run, stock knobs, read the
thermometer. In **closed-loop** the independent variable moves up a level to the *control policy* (open vs. closed, and
the controller design), holding the base filter fixed. Closing the loop requires a filter that exposes a steerable
operating point — see [N-G](novelty.md).

## The loop — two feedback edges

The "loop" is two feedback edges, and naming them keeps the experiments clean:

- **Fast (inner): model → filter.** The filter scores informativeness against the *current* backbone, so as the model
    adapts, what counts as novel/surprising shifts. This edge is intrinsic to any streaming filter and is what *creates*
    the coupled dynamics (and the pathologies).

- **Slow (outer): monitor → filter.** The health monitor detects which degradation mode is setting in and re-aims
    selection accordingly. Example: rank dropping (collapse) → push toward **coverage**; probe-on-past falling
    (forgetting) → **retain/replay** under-covered past; overspecialization → **down-weight** the over-selected mode.
    This edge is what *corrects* the pathologies, and it is what makes this a loop rather than a pipeline.

## Strategic ordering — instrument & demonstrate first

The filters are knobs, not the result; the instruments are ours. Lead with the cheap existential experiment, not a
filter design.

1. Build the streaming loop + **measurement apparatus** (effective rank / RankMe, probe-on-past, frozen-backbone
    baseline B5) and a **positive control** known to collapse. Loop is composed of known methods.
1. With cheap knobs only (no-filter / random / dedup / loss), on a correlated stream, establish:

- **(a)** model adaptation beats frozen model
- **(b)** streaming induces measurable degradation
- **(c)** the knob measurably moves the health trajectory.

If (a) or (b) fails, reframe immediately. The steerable filter and the closed loop (N-E, N-G) come **last**, once the
pathologies are shown to exist and the monitor is built. We can't close a loop to correct a problem we haven't
demonstrated.

## Novelty at a glance

*The space is crowded along each axis; the assembly is not. The contribution is the **closed loop as a live system**,
not any single axis. Full statements + the closest prior work each claim stands against are in
**[novelty.md](novelty.md)**.*

- **N-A — The coupling, run as a live loop `[NEW]`.** The two edges operate together, studied live: fast (model→filter)
    - slow (monitor→filter). Prior links selection→collapse *offline* / does streaming selection with *no health loop*.
- **N-B — Does the budget flip survive co-adaptation? `[NEW]`** Coverage wins at small budgets, novelty at large — does
    the flip hold under single-pass causal selection against a model the selection is itself changing?
- **N-C — Self-reinforcing selection (in)stability `[NEW]`.** Selection drives the model, which changes what is selected
    next — may degenerate or be stabilized; and how FL averaging amplifies or damps it.
- **N-D — Selection × aggregation skew in FL `[NEW]`.** Independent clients over-selecting local tails skew the
    aggregate; global-aware selection may correct it.
- **N-E — Online health-monitor → control loop `[NEW]`.** A multivariate health signal as a **live controller** (not a
    diagnostic) — modulating selection, scheduling replay/rollback.
- **N-F — Federation-level health analytics `[NEW]`.** Flag a degraded client via representation health before its
    update poisons the global model — the server-side twin of the on-device monitor.
- **N-G — Health-steerable filter (the actuator) `[NEW]`.** Coverage↔novelty mix `α` + acceptance rate as a **live
    function of the health signal** — the actuator that makes the loop closable.

**The honesty check:** everything else is `[STD]` or `[EXT]`. New work maps to one of N-A…N-G, or it is scope drift.

## Phase spine at a glance

The **scientific spine** is sequential (each phase needs the prior); the **FL** and **monitoring** tracks run in
parallel; **ZOD confirmation** trails the spine. Full plan + parallel tracks + per-phase grids in
**[experiments.md](experiments.md#phased-plan)**.

- **Phase 0 — Instrument `[STD]`.** Streaming loop + health metrics + B5 + positive control on STL-10. **Exit:**
    instruments validated **and PC collapses on RankMe**. → **current phase; progress & latest results:
    [Phase 0 docs](../experiments/phase0/index.md).**
- **Phase 1 — Degradation envelope `[NEW]`.** 1a STL-10 pilot (synthetic correlation) → 1b BDD100K (real,
    make-or-break); sweep pressure incl. init. **Go:** a coupling exists. **No-Go:** reframe to
    selection-for-efficiency.
- **Phase 2 — Open-loop criterion study `[NEW]` (N-B, N-C).** Centralized; F-a/F-b/static-F-c as fixed knobs. Does the
    flip survive co-adaptation; does the loop self-reinforce?
- **Phase 3 — Closed loop `[NEW]` (N-E, N-G).** Centralized; build the monitor→filter controller; **B-open vs.
    closed-loop** on health + downstream.
- **Parallel tracks.** FL (infra from Phase 1; science — N-D, N-F — after the Phase-2 centralized reference) ·
    Monitoring (built once Phase 1 confirms degradation, ready by Phase 3).
- **Phase 4 — ZOD confirmation `[EXT]`** (pretrained-adapt, realistic scale) · **Phase 5 — FL (N-D, N-F).**

**Gates:** Phase 0 — PC collapses on RankMe. Phase 3 — α demonstrably moves health open-loop (F-c is a real actuator)
before building the controller. Full risk register in **[risks.md](risks.md)**.

## Where to read more

| Working on… | Read |
| -- | -- |
| A `[NEW]` contribution / positioning vs. prior art | [novelty.md](novelty.md) |
| SSL backbones, filter families (F-a/F-b/F-c), datasets | [building-blocks.md](building-blocks.md) |
| Setting up / running a phase — baselines, factor levels, grids | [experiments.md](experiments.md) |
| Measurements, health monitor, evaluation protocols | [metrics.md](metrics.md) |
| Citations / lit-review | [reading-list.md](reading-list.md) |
| Phase planning, interpreting a null | [risks.md](risks.md) |
