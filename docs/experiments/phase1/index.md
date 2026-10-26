# Phase 1 — Degradation Envelope `[NEW]`

This is the landing page for Phase 1: it states the goal, the methodology, the exit gate, and indexes the studies the
phase is composed of. Detail lives one level down, in the per-study docs linked from the [Studies](#studies) table.

## Goal (from the [project plan](../../project-plan/index.md#plan))

The relevant part of the project plan architecture is the following open loop, the complement of the project's
[closed loop](../../project-plan/index.md#scope). The forward pipeline (data → filter → backbone → monitor) and the
intrinsic **fast** edge (the filter scores against the current backbone) are live; the **slow** edge (monitor re-aims
selection) is left **open** — health is read out and logged, not fed back. Closing that edge is Phase 3's job.

```
data stream → [selection filter] → SSL backbone (adapts online) → [health monitor]
                   ▲   ▲                                                │
     fast edge ────┘   └ ─ ─ ─ ─ ─ ─ slow edge (monitor → filter) ─ ─ ─ ┘
   (model→filter)
```

**Why this phase is the hinge.** The project's contribution is a *live loop* — a selection filter feeds a
self-supervised backbone, and a health monitor watches the backbone and re-aims the filter — under one governing
discipline: *you cannot close a loop to correct a pathology you have not first shown exists*. That discipline is what
sequences the whole plan. Phase 0 **calibrates** the health instruments, but only on *deliberately-broken* models at toy
scale; Phase 1 **demonstrates the phenomenon** those instruments were built for; Phase 2 studies which selection
criterion wins under the resulting dynamics; Phase 3 finally **closes the loop**. So Phase 1 is the make-or-break pivot,
and everything downstream is conditional on it. The distinction that matters: calibration proved only that a lit match
trips the smoke detector; Phase 1 asks whether the house actually catches fire when a backbone lives on a real,
correlated stream. That makes this a question of **existence, not measurement**.

The concrete job is to **map the degradation envelope of the coupled streaming loop** — to show that running a
self-supervised backbone on a correlated, single-pass stream actually does something measurable to its representation,
and that the pieces we can turn (adapt vs. freeze, which frames we select) have leverage over that outcome. Concretely,
three existential legs, all read against the Phase-0 gates:

- **(a) Adaptation beats frozen.** An online-adapting backbone must beat the frozen-backbone floor (**B5**). If it does
    not, there is nothing to protect.
- **(b) Streaming induces measurable degradation.** A correlated diet must move the health trajectory in a way the
    calibrated instruments catch — and the per-mode positive control (**PC**) must still fire.
- **(c) The selection knob moves health.** Swapping a cheap selection filter must measurably shift the trajectory —
    evidence that the fast (model→filter) edge is a real lever, not a passive readout.

The outcome is a gate based on a `pass` result for each of the gates above:

- **Go** — a coupling exists on real data → proceed to Phase 2 (open-loop criterion study).
- **No-Go** — no coupling survives stress → reframe the project to *selection-for-efficiency* and say so plainly.

## Executive Summary

*Placeholder — written once the runs land. It will read as the Phase-0 index's does: the honest verdict on whether the
coupling exists, which legs passed on which data/init, and the load-bearing caveats Phase 2 must carry.*

## Methodology

**Open loop only.** Phase 1 exercises the *fast* edge (the filter scores informativeness against the continuously
adapting backbone) and *builds* the health monitor, but it does **not** close the *slow* edge (monitor→filter re-aiming)
— that is Phase 3. So the independent variable here is the **base filter**, fixed per run, using stock knobs only
(no-filter / reservoir / dedup / loss). We are not yet designing a filter; we are asking whether the system this loop
creates has pathologies worth correcting. See [novelty.md](../../project-plan/novelty.md) (N-A) for the coupling claim
this phase underwrites.

**Two modes, prioritized — not symmetric.** Phase 1 reads degradation on the two backbone families Phase 0 calibrated,
and the priority between them follows the evidence, not balance.

- **Stability–plasticity (primary, P1.3).** The loop exists to protect what adaptation adds. So forgetting is read on a
    backbone whose adaptation gains something; a backbone that gains nothing has little to forget, and is read for the
    other side of the trade-off instead — plasticity: can it adapt at all once the stream shifts? Phase 1 began with
    forgetting on MAE, Phase 0's open question. P1.2 found that only the joint-embedding backbone (JE) gains on BDD, so
    P1.3 reads forgetting on JE and plasticity on MAE. Either way the novel question is whether **selection** moves the
    trajectory, and how far the envelope must be pushed.
- **Collapse (secondary, P1.4).** Already well calibrated, so the job is to confirm it *shows up* under the diet. It
    shares P1.3's JE backbone, so both modes can be read off the same runs, each with its own reader.

The other two calibrated modes are **carried, not primary vehicles**: instability is a training-recipe *choice* (below),
and MAE representation-*quality* has no label-free internal reader, so Phase 1 only *provisions* for it (a labelled
canary or a JE projector) rather than chasing it.

**Read movement against the gate, never in isolation.** Every run logs the positive control and the frozen floor B5
beside the live arms, because a Phase-0 lesson is load-bearing here: **a signal is only interpretable against its
gate**. This is Phase 1's hardest new problem — the loop's raw instrument *movement* cannot, by itself, tell genuine
degradation from benign reorganization (a healthy backbone reorganizing has, e.g., a U-shaped RankMe dip;
[P0.1.0](../phase0/P0.1.0.md), [P0.2.1](../phase0/P0.2.1.md)). Producing that discrimination — a healthy-movement null
plus a protocol for reading movement against it — is a **named deliverable (P1.1)**, not an assumption the mode studies
get to make.

**Initialization is a factor, not a fixed choice.** A from-scratch backbone is the degradation-sensitive setting and the
natural home for the demonstration; a fully-pretrained one sits in a good basin, adapts gently, and can **mask**
collapse into a false negative. The two carry different senses of "degradation" — *failing to form* good features
(collapse) vs. *eroding* ones it already had (forgetting) — and both are worth eliciting, so init is swept, never
locked.

**The backbones, and why.** Phase 1 keeps Phase 0's two families, because the failure modes are family-specific:
collapse belongs to joint-embedding methods, shortcut-driven quality loss to MAE, and forgetting can hit either. At BDD
scale both need a pretrained ViT-B start, chosen in stages.

- **MAE is pure throughout**: MAE's pretrained weights, continued with MAE's own objective.
- **JE begins as a hybrid.** [P1.1.0](P1.1.0.md) starts it from MAE's weights too, continued with SimSiam. The reasons
    are practical: the pipeline stays label-free end to end (the other ViT-B start on hand was ImageNet-supervised), and
    both families share one start. The cost is a confound: a JE gain could come from switching the objective rather than
    from the data. [P1.2.4](P1.2.4.md)'s off-domain controls exist to test it.
- **A pure JE start replaces it where it can.** [P1.2.5](P1.2.5.md) starts JE from DINO, continued with DINO's own
    objective. If that start passes its gate, it is the JE vehicle from P1.3 on; otherwise the hybrid stays, with its
    confound stated.
- **Which side of the stability–plasticity trade-off each family carries** is set by P1.2's evidence, not in advance
    (see the first bullet of *Two modes* above).

**What Phase 1's own studies changed.** Rules every later study inherits; the evidence is in each study's docs.

- **B5 is the off-the-shelf encoder**, untouched by the domain. A frozen *well* has already trained offline on BDD, so
    it answers a different question (P1.2.0).
- **Settle fresh SSL heads on the frozen encoder before adapting.** Random heads distort a pretrained encoder, and the
    resulting dip looks like a cost of adaptation (P1.2.0).
- **Size MAE's decoder to the encoder.** An undersized decoder damages a pretrained encoder's last blocks (P1.2.2).
- **Read geometry and information separately.** kNN reads the distances the selection filter uses; a linear probe reads
    whether the information survives. They can disagree (P1.2.1).
- **Read MAE's label-free drift in the middle blocks**, not at the output, where natural forgetting does not show
    (P1.1.3).

**What Phase 1 carries in from Phase 0.** The [handover to Phase 1](../phase0/index.md#handover-to-phase-1) is the
operative to-do list; the load-bearing items:

- *Collapse* — anchor the health read to a **fixed, independently-certified healthy reference**, not a co-trained
    "healthy" arm that would drift with the very signal being read; the *principle* ports even though Phase 0's toy
    anchor and its firing numbers do not. Expect the target corner (correlated × single-pass) to **thin the collapse
    suite** — possibly to a single marginal reader — so budget a second reader and re-establish the bar at scale
    ([P0.2](../latest-results/P0.2.md)).
- *Forgetting (MAE)* — MAE **resists**, and no cheap lever made its forgetting permanent
    ([P0.3](../latest-results/P0.3.md)). So the question is whether **selection** and a hard-enough envelope
    (correlation × long horizon) move it, not whether it forgets spontaneously. Two carried cautions: the **LR
    schedule** is itself a forgetting confound (re-warming manufactures it independently of the diet; Beyond Cosine
    Decay, CoLLAs 2025), so pin and sweep it; and label-free **probing can understate** MAE forgetting (fine-tuning
    reflects it better), so corroborate with a fine-tune probe, not kNN / linear alone.
- *Forgetting (joint-embedding)* — read it via representation **drift**, not the reused collapse geometry
    ([P0.6](../latest-results/P0.6.md)).
- *Instability* — the grad-norm early warning is optimizer/loss-contingent (raw MSE + SGD); pick the training recipe
    with that tradeoff in view ([P0.4](../latest-results/P0.4.md)).
- *Representation quality (MAE)* — is **not** readable label-free from MAE internals; provision a labelled canary or a
    joint-embedding projector ([P0.5](../latest-results/P0.5.md)).

**The substrate ladder, and why STL-10 is only a smoke-test.** STL-10 earns its place solely as the engineering
smoke-test (P1.0): Phase 0 showed it can host collapse but *cannot* induce forgetting even under a deliberate control,
so reading dynamics off it would be a class-blocking artifact at best and impossible at worst. Prior art adds a second
reason: the **tunnel effect** is *amplified* by few-class, low-resolution data (Masarczyk 2023), so a rank drop on
STL-10 is especially likely to be benign compression rather than genuine degradation. The scientific claim rests on
BDD's real correlation; the mid rung between them exists to **de-risk BDD** — so that a null there reads as "no
coupling" rather than "underpowered / mis-configured" (see [risks.md](../../project-plan/risks.md),
synthetic-correlation artifact).

## Exit Gate

Phase 1 exits when the degradation envelope has been mapped **on real data (BDD100K)** and the coupling is resolved
either way:

- **Go** requires all three legs on real BDD correlation — (a) adaptation beats B5, (b) streaming induces
    instrument-caught degradation with the PC firing, (c) the selection knob moves the health trajectory — demonstrated
    in at least the primary vehicle (forgetting) or, failing that, cleanly in the secondary (collapse); with the loop's
    apparatus smoke-test passed beforehand, the degradation **corroborated downstream** (a real probe / kNN / few-shot
    drop, not just a moved needle), and the init-dependence of the effect characterized (for leg (a), across start
    quality — pretrained starts of differing strength plus a lightly pretrained one, from-scratch excluded, see P1.2.3;
    for the degradation legs, P1.5's init sweep). We look hardest at forgetting, on the backbone whose adaptation gains
    are worth protecting, but a coupling that manifests as collapse still counts.
- **No-Go** is a legitimate exit: the coupling is shown absent under stress, and the project reframes to
    selection-for-efficiency (the reframe is recorded, not buried).

## Studies

The ordering below follows the scientific dependency: build the loop ([P1.0](../latest-results/P1.0.md)), earn the right
to *read* it ([P1.1](../latest-results/P1.1.md)) and the right to *continue* ([P1.2](../latest-results/P1.2.md), the
existential floor), then run the two prioritized mode-dynamics studies up their substrate ladders to BDD (P1.3
stability–plasticity, primary; P1.4 collapse, secondary), with a cross-cutting regime-and-downstream check (P1.5). The
FL and monitoring tracks (P1.6) run in parallel from the start. The three existential legs map to studies as: **(a) →
P1.2**; **(b) and (c) → read off the *same* runs inside P1.3 / P1.4** (you cannot vary the knob without also reading
degradation), landing on BDD via each study's top rung.

Studies and their sub-studies share **one table**. A study appears as a single `P1.x` row until its work begins; once it
is broken down, that row is **replaced** by its `P1.x.y` sub-study rows (so nothing is stated at both levels). Sub-study
rows follow the Phase-0 conventions (the "What it establishes" cell opens with `Motivation:`; the Status cell is
`<marker> **<label>** (<finding>)`); the sub-study name links to its detail doc `phase1/<id>.md`, and the concise
take-away lives in `latest-results/<id>.md`. So far **P1.0** (`P1.0.0` / `P1.0.1` / `P1.0.2`), **P1.1** (`P1.1.0` /
`P1.1.1` / `P1.1.2` / `P1.1.3`) and **P1.2** (`P1.2.0` … `P1.2.5`) are broken down; the rest still show as single study
rows.

| ID | Study / sub-study | What it establishes | Status |
| -- | -- | -- | -- |
| P1.0.0 | [Loop-integration harness](P1.0.0.md) `[engineering]` | Motivation: nothing else in Phase 1 runs without one *working open loop*. Wire the stream, the selection filter, the SSL backbone, and the health monitor into one loop, consolidating the scattered Phase-0 pieces — B5 measured only at run-end (run_loop.py `adaptation_report`), a separate positive-control script per mode each emitting its own `comparison.json` — into **one multi-arm harness** that co-logs the **live / PC / B5** arms into a single **canonical `comparison.json`** (the base schema is owned here; later studies extend it but can always fall back to it). This goes beyond P0.1.0's ~30-step harness (P0.1.0; stream + backbone + logged instruments) by adding the filter and the monitor as *components* and PC/B5 as loop-integrated baselines; it assembles the known streaming-SSL harness (Purushwalkam's continuous-SSL setup 2022; Memory Storyboard 2025) — the novelty is the coupling, not the substrate. Default to Phase-0's proven **sequential same-seed arms** (each a `StreamingLoop.run()`, reused untouched; B5 = a frozen, measured-only arm), aligned post-hoc on the shared step index. Prove it on the full 2×2 **C×I** STL-10 smoke — the rig runs, instruments move in the right *direction*, the PC fires, nothing NaNs, **no degradation numbers** — plus a **long-horizon** resampled-image run that stays finite with flat memory (the no-blow-up / no-leak check). | ✅ **Built** (full C×I STL-10 smoke green — rig runs, live reproduces P0.1.0, PC fires, B5 frozen, finite + flat-memory) |
| P1.0.1 | [Arm-alignment divergence (same-seed vs shared-iterator)](P1.0.1.md) `[analysis]` | Motivation: the sequential same-seed arms rest on an unverified assumption — that re-seeding each arm reconstructs a **bit-identical** stream and keeps the arms' trajectories aligned for the post-hoc `zip` (the alignment method every Phase-0 PC already inherits). Test it against a **shared-iterator co-stepped** harness (the oracle — provably identical data by construction): measure (1) per-step **data identity** and (2) **health-trajectory divergence**, attributing any drift to stream *construction* vs. *in-arm* RNG (augmentation, MAE masking, dropout, BatchNorm). Retroactively certifies the Phase-0 comparison method; the co-stepped harness is retained as an optional oracle mode. Owns a number (divergence magnitude) — which is *why* it is a sub-study, not folded into the number-free P1.0.0. | ✅ **Complete** (same-seed `zip` certified — rebuild bit-identical, matched trajectory zero, in-arm-RNG control sensitive; oracle retained as optional mode) |
| P1.0.2 | [Deployment-prototype de-risker harness](P1.0.2.md) `[engineering]` | Motivation: the smoke (P1.0.0) and its certificate (P1.0.1) prove the loop *runs* on toy data, but nothing yet de-risks the **real substrate** the exit gate stands on — STL-10 is a smoke-test by construction (can't induce forgetting, amplifies benign tunnel-effect rank drop), so a BDD null is only interpretable once the ingestion + diet + logging are proven to run and read sensibly. Build a **deployment prototype** that takes the certified open loop onto **BDD100K** (images, not video) with a *meaningful, configurable* learning diet (default: correlated `(timeofday, weather)` era blocks walking a `day·clear → day·rain → dusk → night` shift, long-tailed, single-pass — the real-attribute analogue of the synthetic `class_blocked` knob), a **config-selected backbone** (`backbone=mae\|je`, one per run — same seed rebuilds the same regime by P1.0.1's certificate, so no precarious co-step), **optional** PC/B5 gate arms, and **Phase-0-informed multivariate logging**: the full instrument vector each checkpoint, label-free *and* labelled-canary channels (labels allowed — no deployment constraint yet), every signal tagged with its Phase-0 calibrated status via a code trust-map registry stamped into the artifact. Compute-agnostic core (one `device` knob spans cpu/cuda/hpu; `img_size` downscales to fit memory), hardware only at the existing edges (`docker/gaudi.env.Dockerfile`, `run_gaudi_dev.sh`). Reads **no degradation** — a two-tier validation (anyone-can-run wiring check + our sanity-of-readings pass); the diet's scientific adequacy and every degradation read stay P1.1 / P1.3 / P1.4. Primary deliverable = the **config + promoted logged-signal corpus**; the pipeline is the means. New entry point `scripts/run_deploy_harness.py` + `deploy.yaml`. | ✅ **Complete — both tiers green on real BDD100K** (BDD substrate + configurable regime diet + config-selected backbone + warm-well step + seed-ensemble drive + trust-annotated multivariate corpus with a label-tagged leg map; Tier-A green on the synthetic wiring check — no dataset or accelerator, the whole loop runs on a plain CPU in seconds — Tier-B sanity-of-readings green on the BDD100K val split on a Gaudi2 HPU (for throughput — the drive runs on CPU too, just slower) — three promoted 5-seed ensembles (warm/cold JE, warm MAE) over the 17-leg walk across BDD's 18 driving regimes, all finite, RankMe in range, drift accumulating, scene-canary above chance) |
| P1.1.0 | [Healthy-movement null + canary probe power](P1.1.0.md) `[engineering + new runs]` | Motivation: the degradation test (does streaming degrade the backbone?) is uninterpretable without a null — Phase 0 showed a healthy backbone also moves the needles for benign reasons. Runs the BDD regime walk beside a **matched IID control** (same pool and step count, only the order shuffled), warm JE and MAE × 5 seeds, so any gap is down to correlation alone. Records each instrument's healthy seed spread, the canary's probe power, and the effect of the well having seen the drive's frames. Ships the instrument fixes later studies need: a larger probe set, drift anchored to the well, a time-of-day-stratified canary, grad norm, and a checkpoint at every regime boundary. Closes on *deep wells* — a pretrained ViT-B warmed to its accuracy plateau — after the from-scratch tiny ViT could not learn the canary. | ✅ **Complete** (from a settled, competent well the healthy null is quiet; correlation compresses JE's projector with the canary flat → P1.1.1's target; MAE walk ≈ IID; the canary resolves a few points; no large familiarity effect; leg checkpoints kept locally) |
| P1.1.1 | [Reading protocol — degradation vs. benign reorganization](P1.1.1.md) `[analysis + new runs]` | Motivation: a null says how much healthy movement to expect, not how to call one excursion. Turns P1.1.0's null into a **single-trajectory rule** — one drive read against healthy bands, as a deployed monitor would — with the labelled canary as the arbiter of information loss. Validated both ways: quiet on fresh healthy seeds, and firing on **surgical, dose-controlled damage** injected mid-drive into both walk and IID, on both backbones (MAE's first degradation of known size, the screen for P1.1.2). JE collapse gets its own reference, a **warm collapse dial** on the same ViT-B, since a 3-class canary survives heavy rank loss. A **recovery pass** after each drive tests whether the walk's compression undoes. | ✅ **Complete** (false alarms ≲10% per panel; the canary calls a ~6–8-point dent on both backbones, walk ≈ IID, no healing within the drive; JE's compression is benign specialization — it undoes within 10 IID steps at no accuracy cost — while forced collapse also lowers *backbone* RankMe, a candidate discriminator; unplanned: at a higher lr the walk itself degrades JE — its walk-specific excess rests on 1 of 3 seeds (P1.1.3) → P1.4) |
| P1.1.2 | [Candidate label-free lenses — layer-wise CKA + intrinsic dimension](P1.1.2.md) `[engineering + analysis]` | Motivation: the scalar instruments say *that* the representation moved, not *where* in the network or whether the move is low-dimensional compression. Calibrate two candidate lenses against P1.1.0's null and P1.1.1's controls, held to the same bar as any suite member (fire on degradation, stay quiet on benign reorganization) — **here, not via a Phase-0 re-open**. **CKA** (checkpoint-to-reference similarity, Kornblith 2019), read **per transformer block** to localize drift; carry its reliability caveat (sensitive to outliers and function-preserving shifts — use the debiased estimator plus a complementary metric such as Procrustes; Davari 2023). **Intrinsic dimension**, which a 260-model comparison finds the most reliable single label-free downstream proxy (Arputharaj et al., TMLR 2026; estimator: IdEst, ICML 2026), though its reliability too is moderated by architecture/objective. Reuse an existing estimator implementation where one exists; otherwise re-implement (e.g. TwoNN / MLE) under an independent code audit. Widened to the **label-free MAE reader search**: Phase 0 left MAE with no label-free quality reader (P0.5), so a downstream MAE study has no cheap way to tell training is going off the rails. P1.1.1's MAE damage branches log the full label-free vector, so every candidate (these lenses, plus the existing loss / drift / geometry signals) is screened against damage of known size there. The screen is necessary, not sufficient: a reader that tracks surgical damage may only detect scrambled weights (the P0.5.3 trap), so certification waits for a natural MAE failure (P1.3). Both stay **analysis lenses, not Go/No-Go gates**, until they earn promotion; intrinsic dimension then feeds P1.5's downstream-corroboration panel. | ✅ **Complete** (of every lens and logged signal, only CKA drift against the well tracks MAE damage *and* stays quiet on a re-init-top-4 control that moves the weights at no canary cost — a detector of change from a certified anchor, not a quality reader; Phase 0's MAE readers fail again, now for a named reason, as do Procrustes, intrinsic dimension and cosine drift; per-block lenses localize but add no detection on surgical damage (natural forgetting needs them, P1.1.3); CKA drift also fires on most healthy MAE walks and was faint on Phase 0's natural forgetting → P1.1.3) |
| P1.1.3 | [CKA drift deep dive — information loss vs benign adaptation](P1.1.3.md) `[analysis + new runs]` | Motivation: P1.1.2 leaves CKA drift as the one label-free MAE signal that tracks damage without reading the lever — but as a detector of change from a certified anchor it cannot tell information loss from healthy adaptation: it fires on most healthy MAE walks and on optimizer restarts, and on the same ViT-B Phase 0's natural forgetting barely moved it, even read on the old task's images (P0.3.9). Tests whether it can become a usable reader, and under what assumptions: drift read only on the regimes the walk has left behind (where forgetting lives), a null calibrated on healthy walks rather than IID drives, the size and duration of the optimizer-restart jolt, and JE's lr-1e-4 walk as a natural, label-confirmed degradation on BDD; and P0.3.9's seed-stable natural MAE forgetting, re-run with saved states and read per block with the debiased estimator — does natural information loss show anywhere, or is CKA drift a gross-damage detector only? CKA drift is already usable *comparatively* (arm vs matched control), which is how P1.3 reads its legs. | ✅ **Complete** (at the output, none of the single-drive fixes passes — past-regime drift is no better than pooled, a walk-aware null swallows real damage, and JE's lr-1e-4 loss does not move drift beyond the learning rate's own effect; the restart jolt is brief on MAE and absent on JE; but natural forgetting shows: re-running P0.3.9 with saved states, the forgetting arm drifts more than its healthy twin in blocks 4–11 on 5/5 seeds, invisible at the output Phase 0 read; read as a single-drive panel, those blocks pass the BDD damage screen with a wider margin than the output and catch 4/5 forgetting arms without their twin — the fix was where to read, not how to calibrate → P1.3 reads CKA drift in the middle blocks and validates it on its first natural BDD failure) |
| P1.2.0 | [The readout ladder and the frozen floor, read on P1.1's artifacts](P1.2.0.md) `[engineering + analysis]` | Motivation: P1.2's question — does a backbone that keeps learning beat the same backbone frozen? — needs a floor and a measuring stick. The floor (**B5**) is the off-the-shelf MAE ViT-B/16, untouched by BDD, not the harness's frozen *well*, which has already trained offline on BDD. Builds a readout ladder at rising label budgets (kNN → linear → few-shot fine-tune at 10 / 100 labels per class) with a seed × image bootstrap, and reads it on B5, a random ViT-B, P1.1's wells and drive checkpoints, on 14,000 BDD-*train* frames those models never saw. Two side checks: was continued training's familiar early dip caused by bolting random SSL heads onto the pretrained encoder (a head-settle scout), and is there a readout task with more headroom than the scene label? A scout, not the gate. | ✅ **Complete** (training the heads first on the frozen encoder removes JE's early dip and half of MAE's — the rest most likely the undersized MAE decoder P1.2.2 later found; at 224 px objects are sub-patch, so a cell-balanced traffic-light task, which reads intersection context, joins scene as co-primary readout with ~3× its headroom; the squash does not handicap B5, and P1.2.4's COCO control later rules it out as the gain's source; on P1.1's models B5 wins every MAE rung — a recipe effect, not MAE's verdict) |
| P1.2.1 | [The gate read — online adaptation vs B5 over one long pass](P1.2.1.md) `[engineering + new runs]` | Motivation: P1.1's drive is short and starts from a well already trained offline, so it cannot answer whether adapting beats freezing. Drives a pretrained ViT-B over one full pass of BDD *train* (~1,750 steps, every frame new), from the off-the-shelf encoder with its SSL heads settled while frozen — so step 0 *is* B5, and any difference is online adaptation. JE (the MAE-started hybrid) and MAE, correlated walk × 5 seeds plus a shuffled (IID) control × 3; read on the untouched BDD *val*, scene and traffic light co-primary. Bars fixed before data: PASS if the primary rung (JE linear, MAE few-shot fine-tune) beats B5 on either task; FAIL if every rung on both tasks rules out a 2-point gain. | ✅ **Complete** (JE passes, MAE fails: JE's linear probe beats B5 on both tasks, on every seed — scene +0.025, traffic light +0.040 — but on only 2 of 8 rungs, on a fixed domain (val shares train's distribution); MAE loses on every rung of both tasks, most of it within 100 steps → P1.2.2; correlation costs JE's kNN at night while its linear gain holds — geometry, not information — a forgetting lead that P1.2.4's recipe removes against B5) |
| P1.2.2 | [Why does adapting MAE lose to the frozen floor? A recipe diagnosis](P1.2.2.md) `[engineering + new runs]` | Motivation: MAE failed the gate and was P1.3's planned vehicle, so settle whether continued MAE training loses on this data, or the recipe does. One arm per suspect, one seed each: a zero-lr pipeline check (must reproduce B5 exactly), lr ⅓× / 3×, a warmup, fine-tuning "protections" (layer-wise lr decay, no weight decay on norms and biases), and the MAE paper's 512-d × 8 decoder in place of ours; plus per-block probes. The best arm, chosen on a train-side selection set, is confirmed once at 5 seeds on val against P1.2.1's bars. | ✅ **Complete** (pure MAE, not the hybrid; the pre-registered confirmation FAILs: our 64-d × 2 decoder was the culprit, most likely pushing the last blocks toward pixel detail; the paper's decoder + protections closes the loss to +0.003 on few-shot *k* = 100 but gains nothing, and scene kNN and few-shot *k* = 10 stay slightly below B5, on one pass over a fixed domain. MAE stays in the programme as P1.3's plasticity arm, opened by a second MAE pass) |
| P1.2.3 | Lightly pretrained start `[new substudy]` | Motivation: the exit gate asks how the adapt-vs-frozen answer depends on the starting model. **Rescoped (user, 2026-10-25) to a lightly pretrained start.** From-scratch is dropped here: its frozen floor is a random network any training beats, no deployment trains a ViT-B from scratch on its stream, and a from-scratch run of a strong method takes days; it stays in P1.5, where it matters for degradation. A lightly pretrained start models a deployment whose backbone is weak or mismatched; the candidate is a shrink-and-perturb dial between pretrained and random weights, which needs no pretraining run. With the MAE start, the supervised ViT-B and DINO it completes a start-quality axis. Runs last, once P1.2.5's strong-start gate has read; P1.2.0's random-ViT-B read is its first data point. | 🔲 **Not started** (deferred) |
| P1.2.4 | [Is the gain from the stream, and can it grow? Attribution and widening](P1.2.4.md) `[engineering + new runs]` | Motivation: passing the gate shows the gain exists; the loop is only worth building if the gain is worth protecting and comes from the stream. JE's gain could be generic: a joint-embedding objective makes MAE-pretrained features more linearly separable on any images. **Attribution:** JE trained identically on non-driving images — COCO without street content (the matched control) and Places365 without road or BDD-like scenes (the strict skeptic) — read on BDD val; plus stronger frozen floors (DINO, ImageNet-supervised ViT-B). **Widening:** weight interpolation toward B5, a schedule and weight averaging, lr ⅓× / 3×, the protections — chosen on the selection set, confirmed once at 5 seeds. Pre-registered claim levels (Strong / Qualified / Weak). | ✅ **Complete** (Strong claim, scoped: generic images cannot stand in for the domain's own data — COCO recovers a fifth or less of the gain, Places365 about half, a remainder that clears the pre-registered bootstrap but not a seed-only test against Places365; whether any data as far from ImageNet would do is open; the claim is in-domain adaptation, not yet continuous learning; the protections widen the gain, scene linear +0.025 → +0.038, kNN and few-shot turning positive, the night kNN loss gone; it beats frozen DINO (read mean-pooled) on scene (+0.030) but not traffic light (+0.013, n.s.); "JE" here is the MAE-started hybrid) |
| P1.2.5 | [Strong starts — DINO-init adaptation, the time-to-payoff curve and a wider task panel](P1.2.5.md) `[engineering + new runs]` | Motivation: P1.2's evidence rests on one start, the MAE-pretrained ViT-B continued with SimSiam ("MAE-init JE", a hybrid) — weak under linear probes, which flatters adaptation — while frozen DINO and a supervised ViT-B read higher. Asks whether adapting the *best available* start beats keeping it frozen, and after how much stream. **A:** two gated DINO starts — DINO's own objective (the clean, MAE-free JE result) and SimSiam on DINO weights — plus supervised ViT-B + SimSiam (reported), each against its frozen start at 5 seeds on P1.2.1's bars; a collapse control must fire before DINO's objective is read. **B:** the within-pass time-to-payoff curve (~3k–54k frames). **C:** a wider task panel screened on frozen DINO (larger objects, weather, a per-patch drivable-area probe). **D:** P1.2.4's attribution re-run under the protections recipe. The second pass, DINO's healthy bands and a forgetting reader for the hybrid moved to P1.3. | 🔲 **Designed, awaiting runs** (prerequisites flagged: BDD val's box and drivable-area labels and DINO's full checkpoint are not on this node) |
| P1.3 | Stability–plasticity on the adapting backbones (primary) — what does continued learning cost, and can MAE learn at all? | Motivation: the loop exists to protect what adaptation adds. P1.2 found that only the joint-embedding backbone adds something on BDD; MAE, with a repaired recipe, only matches its frozen start, and a backbone that does not move has little to forget. **Reframed (user, 2026-10-25)** from "where does MAE's stability break" to both sides of the trade-off. **Forgetting, on JE:** DINO-started if P1.2.5's DINO gate passes, else the MAE-started hybrid; P1.2.1's lead (the correlated walk costing JE's kNN at night) re-read under the protections recipe from P1.2.5's drives; read both kNN (the geometry the selection filter uses) and linear (information). **Plasticity, on MAE:** can it beat its frozen start once the stream actually shifts? (P1.2's evaluation shared its training distribution.) Opens with a second MAE pass from P1.2.2's repaired recipe — lr with the large decoder, a larger decoder, the mask ratio, a combined MAE + JE objective — read with a fine-tune probe and dormant-unit / weight-norm diagnostics (Dohare 2024). **First obligations, the readers:** P1.1's healthy bands checked on a DINO start; a label-free reader for the hybrid losing its MAE-pretrained information (P1.1.3's per-block CKA drift, repeated); whether a second pass over the same frames still pays. Throughout: the learning-rate schedule as confound and lever (re-warming manufactures forgetting), and a longer, shifting horizon (BDD's driving videos). **Continuous vs adapt-once (a candidate sub-study):** P1.2 shows adapting on in-domain data beats freezing, but on a fixed domain that equals one offline adaptation. Adapt once on a subset of the stream, freeze, and run it into the rest; one clear instance suffices, with the split chosen by a rule stated before reading. On the same runs, the exit gate's other two tests: does the correlated stream degrade the backbone, and does swapping a stock selection filter (reservoir / dedup / loss) move that degradation? | 🔲 **Not started** (reframed 2026-10-25; the JE vehicle waits on P1.2.5's DINO gate) |
| P1.4 | Collapse dynamics (secondary) — does the diet collapse a joint-embedding backbone? | Motivation: the secondary mode — collapse is already well calibrated, so the job is to confirm it *shows up* under the streaming diet. Predicted but not demonstrated: dimensional collapse strikes where augmentation variance exceeds data variance (Jing 2022), so a correlated diet is more susceptible, but no prior work shows collapse from a correlated stream, live, coupled to selection. Controls two confounds that collapse a backbone independent of the diet: streaming's small effective batch and model size (swept with P1.5). Shares its backbone with P1.3's forgetting vehicle, so it may read collapse (projector and backbone rank) off P1.3's runs rather than run its own; on the same runs, the degradation and selection-filter tests. Phase-0 caveat: in the correlated × single-pass corner the collapse suite thins to one marginal reader, so a second is budgeted. Lead from P1.1.1: at lr 1e-4 the walk costs JE 6–7 canary points (twice the IID control's loss) while backbone RankMe falls, as forced collapse does — re-test with that lr from the drive's start. A collapse null does not sink the phase if forgetting clears. | 🔲 **Not started** |
| P1.5 | Initialization × pressure regime map + downstream corroboration | Motivation: two cross-cutting checks the exit gate needs. **Init × pressure:** the coupling may be regime-specific — from-scratch is degradation-sensitive, while a fully-pretrained backbone sits in a good basin and can *mask* collapse. Sweep init {from-scratch, lightly-pretrained, fully-pretrained} against the pressure knobs (correlation, LR, model size, horizon, replay on/off) to locate onset, and report the init-dependence as a finding (failing to form features vs eroding them). Grounded in the tunnel effect (Masarczyk 2023): overparameterization, few classes and low resolution lengthen the compressing "tunnel", so toy regimes amplify benign rank drop. **Downstream corroboration:** confirm that health degradation tracks a real downstream drop (probe / kNN / few-shot), so we have moved the patient, not just the thermometer. Read a metric suite, not one needle: RankMe is necessary-but-not-sufficient (Garrido 2022); intrinsic dimension is a strong cross-model proxy (Arputharaj 2026), but within one run it read weight noise in P1.1.2, so it is not a within-run monitor. For MAE, weight the fine-tune probe. Cross-cuts P1.3 / P1.4. | 🔲 **Not started** |
| P1.6 | Parallel tracks — FL infrastructure + health-monitor seed | Motivation: two tracks the plan starts in parallel from Phase 1. The FL infrastructure (FedAvg + client simulation) is built now so the FL science can slot in right after the Phase-2 centralized reference; the health monitor is seeded once Phase 1 confirms degradation exists (its required input) so it is ready for the Phase-3 closed loop. Enablement, not one of the exit gate's tests. | 🔲 **Not started** |

## Artifacts

*Conventions inherited from Phase 0, to be populated as runs land.* Each study's harness run is expected to write a
`comparison.json`-style artifact (gate verdict + per-`(instrument × surface)` series for the live / PC / B5 arms) into
its gitignored Hydra run dir; the runs that back a study's cited numbers are then promoted, tracked and slimmed, under
`artifacts/<substudy-id>/`, with the per-substudy listing living in each study's own detail doc. The precise per-harness
schema is fixed per study once its harness exists — see the Phase-0 [Artifacts](../phase0/index.md#artifacts) section
for the policy this phase follows.
