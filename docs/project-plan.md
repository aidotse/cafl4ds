# Continuous, Active Federated Learning for Data Streams

## Research & Experiment Plan

*Working document for project lead, partners, and potential students. Quick-skim guide. Method descriptions kept short
for brevity.*

**Tags:**

- `[STD]` standard / re-implement
- `[EXT]` extend an existing method to our setting
- `[NEW]` genuine contribution absent from prior art.

______________________________________________________________________

## Scope

![](assets/coupled_loop_adaptive.svg)

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
operating point — see N-G.

______________________________________________________________________

## The loop — two feedback edges

The "loop" is two feedback edges, and naming them keeps the experiments clean:

- **Fast (inner): model → filter.** The filter scores informativeness against the *current* backbone, so as the model
    adapts, what counts as novel/surprising shifts. This edge is intrinsic to any streaming filter and is what *creates*
    the coupled dynamics (and the pathologies).

- **Slow (outer): monitor → filter.** The health monitor detects which degradation mode is setting in and re-aims
    selection accordingly. Example: rank dropping (collapse) → push toward **coverage**; probe-on-past falling
    (forgetting) → **retain/replay** under-covered past; overspecialization → **down-weight** the over-selected mode.
    This edge is what *corrects* the pathologies, and it is what makes this a loop rather than a pipeline.

______________________________________________________________________

## ★ Novelty & positioning — what is `[NEW]`, and the prior work each claim stands against

*The space is crowded along each axis; the assembly is not. Every claim below is paired with the closest prior work on
that axis — what it covers, and what it leaves open — so the mapping is explicit. Terminology: an **edge** is a literal
connection in the loop (model→filter, or monitor→filter); the claims below are the distinct **axes** of contribution,
only two of which (inside N-A) are literal edges. The contribution is the **closed loop as a live system**, not any
single axis.*

**N-A — The coupling, run as a live loop `[NEW]`.** *Claim:* the two loop **edges** operate together and are studied
live — the **fast edge** (model→filter: the filter scores informativeness against the continuously-adapting backbone, so
what counts as novel shifts as the model learns) and the **slow edge** (monitor→filter: the health monitor re-aims
selection based on the degradation regime). *Closest prior:* DiSF (2025) links selection→collapse but *offline, fixed
budget, no model in the loop*; SOFed/FedCoCo (2022) does selection in streaming SSL but with *one fixed criterion and no
health coupling*. *Open:* nobody runs selection ↔ representation-health as a closed feedback loop with a co-adapting
model.

**N-B — Does the budget flip survive co-adaptation? `[NEW]`** *Terms:* **storage budget** = how many frames the buffer
may keep; **coverage** selection = pick frames that represent/span the data distribution; **novelty** selection = pick
hard/atypical frames. The **budget flip**: at small budgets coverage wins, at large budgets novelty wins — the best rule
"flips" coverage→novelty as the budget grows. *Closest prior:* CCS (Zheng 2023) — coverage beats hard-example selection
at high pruning rates (small budgets); Sorscher (2022) — keep prototypical when data is scarce, hard when abundant; D2
Pruning (2024) — balances diversity vs difficulty. All *offline, fixed model, pool-based*. *Open:* whether the flip
holds, moves, or inverts under single-pass causal selection against a model the selection is itself changing.

**N-C — Self-reinforcing selection (in)stability `[NEW]`.** *Claim:* selection drives the model, which changes what is
selected next — a loop that may self-reinforce toward a degenerate diet, or be stabilized. *Closest prior:*
one-sided-feedback learning (2020), recommender/bandit sampling-bias loops, and LLM self-consuming "model collapse"
(Shumailov et al. 2023) — the same pathology, in *other domains*. *Open:* the streaming-SSL-selection instance, its
damping, and how FL averaging amplifies or damps it.

**N-D — Selection × aggregation skew in FL `[NEW]`.** *Claim:* independent clients each over-selecting their local tail
may skew the aggregate's effective training distribution; global-aware selection may correct it. *Closest prior:* SOFed
(2022) notes overlapping client selections in passing; federated SSL — FedU (2021), Orchestra (2022) — aggregates
unsupervised representations but *without* active per-client selection bias. *Open:* characterizing and correcting the
selection-induced aggregate drift.

**N-E — Online health-monitor → control loop `[NEW]`.** *Claim:* a multivariate health signal used as a **live
controller** — modulating selection (the monitor→filter edge) and scheduling replay/rollback — not merely a diagnostic.
*Closest prior:* RankMe (Garrido 2023), LiDAR (Thilak 2024), dimensional collapse (Jing 2022) supply the health
**metrics**, but as *offline diagnostics / checkpoint selection*; ADWIN (Bifet 2007) detects drift but is not tied to
SSL-health control. *Open:* closing the loop live; payoff is conditional — does closed-loop beat open-loop (B-open)?

**N-F — Federation-level health analytics `[NEW]`.** *Claim:* flag a degraded/corrupted **client** via
representation-health diagnostics before its update poisons the global model — the server-side twin of the on-device
monitor. *Closest prior:* robust-FL aggregation — Krum (Blanchard 2017), trimmed-mean/median (Yin 2018) — detects bad
updates via *update geometry*, not via the client model's representation health. *Open:* using the same health analytics
(effective rank, etc.) as the pre-aggregation gate.

**N-G — Health-steerable filter (the actuator) `[NEW]`.** *Claim:* a selection policy whose coverage↔novelty operating
point (mix `α`) **and** acceptance rate are a **live function of the health signal**, so the monitor can steer it — the
actuator that makes the loop closable. *Closest prior:* the static filters we would otherwise steer — reservoir (Vitter
1985), SemDeDup (Abbas 2023), loss-gate (SOFed 2022), core-set (Sener & Savarese 2018) — expose *no* coverage↔novelty
knob; the hybrid filters that *do* have the mix — D2 Pruning (2024), CCS (2023) — fix it *offline*, never driven by a
live signal. *Open:* making the mix (and acceptance rate) a controlled variable set by the monitor. *Validation:* only
via closed-loop vs. open-loop (B-open) on health + downstream — never a static-benchmark win.

**Summary**

- The loop, not just the leaderboard, is what makes this filter design novel
- This informs our approach: we can implement the loop with existing pieces, and analyze system behavior. This study is
    novel.
- Afterward: work toward novelty along any of the axes above.

______________________________________________________________________

*Everything else is `[STD]` or `[EXT]`. If a claim is not in the list above, we are not claiming it as novel. The full
reference list (including the `[STD]`/`[EXT]` toolbox) is in the Reading List section.*

## Strategic ordering — instrument & demonstrate first

The filters are knobs, not the result; the instruments are ours. Lead with the cheap existential experiment, not a
filter design.

1. Build the streaming loop + **measurement apparatus** (effective rank / RankMe, probe-on-past, frozen-backbone
    baseline B5) and a **positive control** known to collapse. Loop is composed of known methods.
1. With cheap knobs only (no-filter / random / dedup / loss), on a correlated stream, establish:

- **(a)** model adaptation beats frozen model
- **(b)** streaming induces measurable degradation
- **(c)** the knob measurably moves the health trajectory.

If (a) or (b) fails, reframe immedieatly. The steerable filter and the closed loop (N-E, N-G) come **last**, once the
pathologies are shown to exist and the monitor is built. We can't close a loop to correct a problem we haven't
demonstrated.

______________________________________________________________________

## Building blocks

**SSL backbone `[STD]`:** default **MAE** for the pipeline (stable; reconstruction error is a free per-frame
informativeness signal). **MAE is collapse-resistant** (a constant code can't reconstruct diverse pixels), so the
*collapse* demonstration uses a **joint-embedding method** (SimSiam/BYOL/DINO/SimCLR) — MAE's own degradation mode is
forgetting/overspecialization. We start with **SimSiam** as the minimal joint-embedding variant (predictor +
stop-gradient, no EMA target, no negatives — fewest moving parts, and its stop-gradient is *precisely* the
collapse-avoidance mechanism under study), expanding to BYOL then SimCLR once the dynamics are established. Adapt an
ImageNet-pretrained backbone (data too small for from-scratch); ViTDet-compatible if detection is a target. *Methods
used as-is: MAE (He 2022), SimSiam/BYOL/DINO/SimCLR (Chen & He 2021 / Grill 2020 / Caron 2021 / Chen 2020).*

**Filter families (the knobs):**

- **(F-a) Latent-feature tracking — novelty criterion `[EXT]`.** Encode → PCA to 8–32D metric space → score =
    kNN/Mahalanobis distance to a same-domain reference set (frozen reference encoder; not UMAP). *Extends:*
    embedding-space OOD scoring — kNN-OOD (Sun 2022), Mahalanobis (Lee 2018), Deep SVDD (Ruff 2018) — built to *flag*
    OOD at test time; we repurpose them as a *streaming, label-free, single-pass training-selection* signal.
- **(F-b) Streaming core-set — coverage criterion `[EXT]`.** Maintain a buffer that geometrically covers the seen
    embedding distribution. *Extends:* core-set / k-center (Sener & Savarese 2018), herding (Welling 2009 / iCaRL
    Rebuffi 2017), prototypicality (Sorscher 2022), GSS (Aljundi 2019) — offline or supervised-CL coverage methods; we
    run them as streaming, label-free buffer maintenance over the SSL embedding.
- **(F-c) Health-steerable filter `[NEW]` (N-G).** Explicit mix `α·coverage + (1−α)·novelty` + acceptance rate, both set
    **live by the monitor**. The actuator for the closed loop. *Contrasts with:* D2 Pruning (2024) / CCS (2023), which
    have the coverage/difficulty mix but fix it *offline*; here it is a controlled variable. A frozen `α` reduces F-c to
    a hybrid `[EXT]` baseline (= B-open).
- **Temporal-robustness layer `[EXT]`.** Denoise the per-frame signal, separating sustained shifts (true novelty) from
    impulsive spikes (sensor noise). *Extends:* view-disagreement OOD (RSS-MGM) and forgetting/difficulty (FALSE) from
    the label-noise line, plus drift/change-point detection (ADWIN, Page-Hinkley); we transpose to label-free streaming
    SSL and add a sustained-vs-impulsive persistence test.

**Datasets (what we run on, and why):**

- **STL-10 — Phase 0 + Phase 1 pilot.** SSL-purpose-built (100k-image unlabeled split, 96px), small enough for CPU smoke
    tests and fast iteration, clean 10-class labels for linear-probe / kNN eval. Correlation is *synthetic*
    (class-blocked ordering). Role: validate the loop + instruments + positive control cheaply — the harness, not the
    headline.
- **TinyImageNet / ImageNet-100 — optional.** A larger non-driving SSL scale check, if STL-10 proves too small to stress
    the backbone.
- **BDD100K — Phase 1 make-or-break.** Real driving video → *genuine* temporal correlation, the property that makes
    selection meaningful (a make-or-break on shuffled toy data won't convince reviewers). Built-in weather / scene /
    time-of-day attribute labels → ready-made label-efficiency probes and natural era boundaries. GPS/IMU → the optional
    telemetry fused-input ablation. Standard, respected benchmark.
- **ZOD (Zenseact Open Dataset) — Phase 4 confirmation.** In-house partner data; confirms the findings transfer to the
    consortium's own automotive distribution and the use case the project ultimately serves. Multimodal (camera/LiDAR +
    vehicle/GNSS signals).
- **nuScenes — optional alternative.** Use if the telemetry-fusion ablation needs tighter time-synchronized CAN-bus
    signals than BDD provides.

*Ordering rationale:* cheap harness (STL-10) → real-correlation make-or-break (BDD100K) → in-house confirmation (ZOD) —
matching the "instrument & demonstrate first" sequence. Dataset choice is not tagged `[STD]`/`[EXT]`/`[NEW]`: a dataset
isn't a contribution, so the tags (which mark method novelty) don't apply.

______________________________________________________________________

## Baselines

**Backdrop:** the baselines are not all the same kind of thing. Most run the **full streaming loop with the selection
knob swapped** (the loop is the backdrop, the knob is the variable). Two of them **switch a loop component off** to
bound the loop from above and below. Two are **diagnostics**, not training conditions. The "Backdrop" column says which.

| ID          | Baseline                                                                                                              | Backdrop (loop config)                                             | Role / when                                                                                    | Tag     |
| ----------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | ------- |
| **PC**      | Positive control: a config known to collapse                                                                          | diagnostic (deliberately-collapsing)                               | Validates the instruments — **Phase 0 gate, runs first**; without it a null is uninterpretable | `[STD]` |
| **B0**      | Use-everything offline SSL                                                                                            | streaming **off** (full data, shuffled, multi-epoch, no selection) | Ceiling the Pareto approaches as buffer→∞; compute early                                       | `[STD]` |
| **B5**      | Frozen backbone, no adaptation — frozen-**pretrained** (pretrained regime) or frozen-**random** (from-scratch regime) | adaptation **off**                                                 | Existential floor — does adapting beat doing nothing? (RanDumb); compute early                 | `[STD]` |
| **B-floor** | No filter: accept all, no replay, raw order                                                                           | full loop, buffer = 0, no selection                                | Max-degradation reference; Phase 1                                                             | `[STD]` |
| **B1**      | Reservoir sampling                                                                                                    | full loop, random selection + replay                               | "Dumb" bar — replay is partly protective; Phase 1+                                             | `[STD]` |
| **B1.5**    | Reservoir + SemDeDup                                                                                                  | full loop, dedup selection                                         | **The real selection bar**; Phase 1+                                                           | `[STD]` |
| **B2**      | Pool-based AL (storage allowed)                                                                                       | streaming **off**, selection on, large storage                     | "If we could store everything" reference                                                       | `[STD]` |
| **B3**      | SOFed/FedCoCo loss-based importance                                                                                   | full loop (federated), loss-gate selection                         | Closest prior art; Phase 2 + Phase 5                                                           | `[STD]` |
| **B-open**  | Static F-c (fixed α)                                                                                                  | full loop, F-c with frozen α, **loop open**                        | Open-loop control for N-E/N-G — needs F-c; **Phase 3 only**                                    | `[STD]` |
| **B1.6**    | Median filter + hysteresis on the signal                                                                              | diagnostic, over the temporal layer                                | "No fancy temporal model" bar for the denoising layer                                          | `[STD]` |

**Ordering:** PC (gate, first) → {B0, B5, B-floor} (bracket + Phase-1 floor) → {B1, B1.5} (the bars) → B3 (Phase 2 / 5)
→ B-open (Phase 3, once F-c exists).

**Levels at a glance:** *full-loop conditions* = B-floor, B1, B1.5, B-open, B3 (swap the knob) · *loop-bracketing
ablations* = B0 (no streaming, ceiling), B5 (no adaptation, floor) · *stored-selection reference* = B2 · *diagnostics* =
PC, B1.6.

> **Initialization is a pressure knob, not a fixed choice (→ factor P).** A fully-pretrained backbone sits in a good
> basin and adapts gently — it can **mask** collapse and produce a false negative. So sweep init
> `{from-scratch, lightly-pretrained, fully-pretrained}`:
>
> - *From-scratch* (small model, toy/proxy data) is the **degradation-sensitive** setting and the natural home for the
>     Phase-1 demonstration — feasible here because we're eliciting dynamics, not chasing quality features (so "ZOD too
>     small for from-scratch" doesn't apply).
> - *Pretrained-adapt* is the **realistic/deployment** setting where the selection results must ultimately hold, and
>     what compute forces at BDD/ZOD scale.
> - Two senses of "degradation" follow the init: *eroding* good pretrained features (forgetting) vs. *failing to form*
>     them (collapse). Both worth eliciting. Do **not** lock the demonstration to pretrained-only.

______________________________________________________________________

## Phased plan

The **scientific spine** is sequential — each phase needs the prior. The **FL infrastructure** and the **health
monitor** run as **parallel tracks** that start early. **Confirmation** on in-house data trails the spine. Model
initialization threads through: from-scratch (small/toy) to *elicit* the dynamics in Phases 1–3, pretrained-adapt to
*confirm they matter at realistic scale* (ZOD).

### Scientific spine (sequential)

**Phase 0 — Instrument `[STD]`.** *(M1)* Streaming loop + health metrics + B5 + positive control, on STL-10. **Exit:**
instruments validated **and PC collapses on RankMe**. → **Progress & latest results:
[Phase 0 docs](experiments/phase0/index.md).**

**Phase 1 — Degradation envelope `[NEW]`.** *(M2)*

- **1a — STL-10 pilot.** Class-blocked (synthetic) correlation; cheap and fast. Job: confirm the phenomenon *can* appear
    **and** the instruments catch it. Validates the apparatus, **not** the claim.
- **1b — BDD100K subset.** Real video correlation → the publishable make-or-break.
- Sweep pressure including **initialization `{from-scratch, lightly-pretrained, fully-pretrained}`** — from-scratch
    (small) is the degradation-sensitive primary; pretrained checks whether degradation persists or is masked.
    Instrument **both modes** (collapse on joint-embedding, forgetting on MAE); diagnostic = rank-vs-loss divergence;
    **PC must fire.**
- **Go:** a coupling exists. **No-Go:** reframe to selection-for-efficiency.

**Phase 2 — Open-loop criterion study `[NEW]` (N-B, N-C).** **Centralized** (aggregation would confound the dynamics).
F-a / F-b / static-F-c as fixed knobs, in the regime where Phase 1 showed the dynamics clearest. Does the flip survive
co-adaptation? Does the loop self-reinforce, and do damping interventions help?

**Phase 3 — Closed loop `[NEW]` (N-E, N-G).** **Centralized.** Build the monitor→filter controller; close the loop with
the steerable F-c. Compare **B-open (static filter) vs. closed-loop (health-modulated)** on health + downstream — does
closing the loop help, and is it stable?

### Parallel tracks (start early, alongside the spine)

**FL track.** *Infrastructure* (e.g. FedAvg + client simulation) built **in parallel from Phase 1**. *Science* (N-D
selection × aggregation skew + global-aware correction; N-F federation-level health analytics) is gated only on a
**centralized reference (Phase 2)**, so it slots in **right after Phase 2, parallel with Phase 3**. *Why
centralized-first for the science:* aggregation confounds the coupling, so an effect can't be attributed to averaging
vs. streaming selection until the centralized baseline exists. Infra parallelizes; interpretation needs the baseline.

**Monitoring track.** Signal analytics + health monitor built once Phase 1 confirms degradation exists (its required
input), ready by Phase 3 (which consumes it). Proceeds in parallel with Phase 2.

### Confirmation & extensions (trail the spine)

**ZOD confirmation `[EXT]`.** Top configs on a ZOD 2D proxy, **pretrained-adapt** (realistic scale). Confirms transfer
after the relevant centralized result.

**Optional.** Detection use case (mAP); deeper CF study; **federated closed-loop** (the Phase-3 loop studied under
aggregation).

______________________________________________________________________

## Training signal analytics, health, control

**Variables:** the selection policy + steerable filter F-c are the *independent* side (what we turn); the **health
metrics** are the *dependent* readout (the thermometer). The **controller** (monitor→filter) is **not** a readout — once
the loop is closed it is part of the *intervention* (independent) side.

**We probe freely from the start.** The goal is to study the dynamics, not to ship a real-time edge pipeline — so labels
are used wherever they sharpen a metric, from day one. The "label-free vs. needs-probe" tag below is a *recorded
property* (it bears on eventual deployment), not a constraint on the study. Read signals **jointly** (multivariate
change-point / covariance tracking), not per-channel — single-channel thresholds miss correlation flips and subspace
rotations.

### Signal catalogue

*Each tagged with `[architecture]` it applies to and whether it needs a probe.*

**Geometry / collapse (label-free):**

- **Effective rank / RankMe** — # embedding dims actually in use; the core collapse readout. `[general]` (Roy & Vetterli
    2007; Garrido et al. 2023)
- **Per-dimension variance** — flags dimensions collapsing to a constant. `[general]` (VICReg, Bardes et al. 2022)
- **Off-diagonal covariance / decorrelation** — flags informational (redundancy) collapse. `[general]` (VICReg, Bardes
    et al. 2022)
- **Alignment & uniformity** — positive-pair closeness + spread on the hypersphere; uniformity drops as embeddings
    clump. `[joint-embedding]` (uniformity general; alignment needs pairs) (Wang & Isola 2020)
- **Prototype-assignment entropy / usage** — how many prototypes are actually used (collapse = a few dominate); the
    specified version of "entropy." `[prototype: DINO/SwAV]` (Caron et al. 2020, 2021)

**Dynamics / stability (label-free):**

- **Representation drift** — CKA / cosine churn of a *fixed* probe set's embeddings across checkpoints; how fast the
    coordinate frame moves (ties to forgetting + the moving-reference problem). `[general]` (CKA, Kornblith et al. 2019)
- **SSL loss trajectory** — necessary but weak alone; for joint-embedding a *low* loss can be a collapsed trivial
    solution, so read it *against* the collapse signals. `[general]` (collapse caveat: Jing et al. 2022)
- **Gradient norm / gradient diversity** — instability/divergence (norm) and training-diet diversity (diversity); not a
    collapse signal. `[general]` (gradient diversity: GSS, Aljundi et al. 2019)

**Downstream / forgetting (needs a labels / probe):**

- **Linear probe + kNN accuracy, current data** — is the representation useful now (vs. B5). `[general]` (kNN protocol:
    Wu et al. 2018; Caron et al. 2021)
- **Per-era / past-class accuracy → forgetting** — the direct forgetting measure: per-era probe accuracy, **Backward
    Transfer** (Lopez-Paz & Ranzato 2017), and the **Forgetting Measure** (Chaudhry et al. 2018). The ground truth for
    MAE's degradation mode. `[general]`

**Architecture coverage:** MAE exposes loss, the geometry suite, and drift (collapse signals sparse → lean on the
forgetting metrics); joint-embedding adds alignment/uniformity; prototype methods add assignment entropy. This is why
the plan demonstrates collapse on joint-embedding and forgetting on MAE.

### Two questions the labels-allowed freedom lets us separate

1. **Does closing the loop help, given a good signal?** Drive the controller with the best available signal — *including
    a probe-based oracle* (held-out labeled forgetting) — to test the control idea unhandicapped.

1. **Can a label-free signal detect degradation in time to drive the loop?** The leading-indicator / deployability
    question: does a Tier-free signal (rank drop, drift) *predict* the probe-based forgetting before the labels would
    reveal it, with usable lead time?

Keeping these separate makes a null interpretable (control failed, vs. the detector missed).

### Control & continual-learning layer

- **Monitor → filter controller `[NEW]` (N-E):** map the detected health regime to F-c's `α` + acceptance rate and to
    replay/rollback. The closed loop is a control system — watch for oscillation; open-loop (B-open) is the honest
    baseline.

- **The bridge `[NEW]` (N-A):** selection → batch diversity → collapse/forgetting, run live (streaming/dynamic extension
    of DiSF's offline finding).

- **Replay codebook `[EXT]`:** server inverts the global model into a small static latent codebook, shipped with
    weights, mixed in by temporal-distance, scheduled by the monitor (embedding-level, not pixel-MAE; refresh each
    round).

- **Federation health analytics `[NEW]` (N-F)** · telemetry as fused input `[EXT]` · sub-Orin distillation `[STD]` ·
    SPSA forward-only `[EXT]`.

______________________________________________________________________

## Evaluation — diagnostic vs. validating axes

Keep the two axes separate, and **establish the link between them** — that link is part of the contribution.

- **Diagnostic (explanatory).** The health signals, read primarily as the **divergence between the SSL loss and the
    health metrics** (loss flat or improving while effective rank / uniformity / drift degrade — the silent-degradation
    signature). Explains *why* a config degrades. *Metrics are `[EXT]`* (RankMe, VICReg variance/covariance, CKA drift);
    *using them as a live explanatory axis coupled to selection is `[NEW]`.* Necessary but not sufficient: a model can
    forget or overspecialize **without** collapse, so health is **multivariate** (rank + drift + forgetting), never
    collapse alone. Trustworthy only after the positive control (PC) has validated the instruments.

- **Validating (payoff) `[STD]`.** Downstream quality (frozen linear probe + kNN, vs. **B5**) and forgetting (per-era
    probe → **Backward Transfer** / **Forgetting Measure**). A config "wins" only if a diagnostic advantage **translates
    here**. For N-E/N-G: closed-loop must beat **B-open** on this axis.

- **The link `[NEW]` (leading-indicator test).** Does the diagnostic axis *predict* the validating axis with usable lead
    time — a rank drop or rising drift *before* forgetting shows in the probe? This is what licenses a label-free signal
    as a control input at all. Cheap to run early, since both axes are already logged.

**Eval pipeline — three protocols by ascending label budget (not a data flow):**

- **kNN** — zero training: classify eval queries by nearest neighbors in the *frozen* features. The rawest test of the
    geometry.
- **Linear probe** — train *only* a linear head on the *frozen* features. Linear separability of the representation
    (standard SSL eval).
- **Few-shot finetune** — *unfreeze* and finetune on a *small* labeled budget. A different question: label-efficiency /
    how good a starting point the representation is, not frozen quality. (Few-shot, so abundant labels don't wash out
    representation differences.)

All vs. **B5** — frozen-**pretrained** in the pretrained regime, frozen-**random** in the from-scratch regime.

**Figures:** quality vs. buffer size (the flip) · vs. stream samples · vs. label budget — reported **per init regime**,
against B5.

**Signal-quality eval `[EXT]`:** segment-ordered streams with known shift points + injected sensor corruption →
true-shift recall vs. noise rejection vs. confirmation lag. Validates the temporal-robustness layer.

**Monitor / closed-loop `[NEW]` — two separable results:**

1. **Control benefit:** does closing the loop beat **B-open**, driving the controller with the best / oracle signal
    (probes allowed)?
1. **Detection-in-time:** does a *label-free* signal flag degradation with enough lead time (the link above)?

Keeping (1) and (2) apart makes a null interpretable — control failed, vs. the detector missed.

______________________________________________________________________

## Experiment matrix

### Factors

|       | Factor                            | Levels                                                                                                   | Notes                                                                                                                                                            |
| ----- | --------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A** | Selection criterion (the knob)    | B-floor (no filter) · reservoir · dedup · loss-gate · SOFed · F-a novelty · F-b coverage · F-c steerable | Independent variable. First five = baselines; F-a/F-b/F-c = our filters.                                                                                         |
| **L** | Loop                              | open · closed                                                                                            | Closed requires F-c + the monitor→filter controller.                                                                                                             |
| **B** | Storage budget                    | 0 · tiny · small · medium · ∞                                                                            | The Pareto axis; where the flip lives.                                                                                                                           |
| **I** | Initialization                    | from-scratch · lightly-pretrained · fully-pretrained                                                     | Degradation-sensitivity ↔ realism knob. From-scratch elicits the dynamics; pretrained confirms they matter. Sets B5's form (frozen-random vs frozen-pretrained). |
| **C** | SSL method                        | MAE · joint-embedding (SimSiam/BYOL/DINO/SimCLR)                                                         | MAE → forgetting mode; joint-embedding → collapse mode (the collapse demo). SimSiam = minimal JE variant, implemented first.                                     |
| **D** | FL setting                        | centralized · FedAvg · FedProx                                                                           | Centralized first (clean dynamics); FL science gated on the centralized reference.                                                                               |
| **E** | Data                              | STL-10 · BDD100K · ZOD                                                                                   | Harness → real make-or-break → in-house confirmation.                                                                                                            |
| **F** | Stream structure                  | IID (control) · correlated · drift+corruption                                                            | Correlated = headline; drift+corruption for signal-quality + health.                                                                                             |
| **G** | PCA dim (F-a)                     | full · 32 · 16 · 8                                                                                       | "How far can we compress before novelty degrades."                                                                                                               |
| **H** | Reference encoder (F-a)           | frozen · adapting                                                                                        | Tests the stability invariant.                                                                                                                                   |
| **K** | Controller signal (when L=closed) | oracle (probe-based) · label-free                                                                        | Separates *control benefit* from *detection-in-time*.                                                                                                            |
| **P** | Pressure sweep                    | correlation · LR · model size · horizon · replay on/off                                                  | Stress knobs to locate the degradation envelope.                                                                                                                 |

*B5 is always init-matched (frozen-random for from-scratch, frozen-pretrained for pretrained). Probes are used freely
throughout (the study, not an edge pipeline).*

### Grids (anchor a config, vary 1–2 factors per phase — never the full Cartesian)

- **Phase 0 — Instrument (gate).** Validate the apparatus; **PC must collapse on RankMe**; B5 (init-matched) reproduces.
    Not a sweep.
- **Phase 1 — Degradation envelope.** Vary **A**{B-floor,reservoir,dedup,loss} × **I** × **C**{MAE,joint-embedding} ×
    **P**; fixed L=open, D=centralized, F=correlated, E STL-10→BDD; PC + B5 every batch. *Asks:* does degradation appear
    and where does it onset; does adaptation beat B5; does the knob move health? (Both modes instrumented.)
- **Phase 2 — Open-loop dynamics (N-B, N-C).** Vary **A**{F-a,F-b,F-c-static} × **B** over a long horizon; fixed L=open,
    D=centralized, F2 (+F1 control), I = the regime where Ph1 showed the dynamics. *Asks:* does the flip survive
    co-adaptation (N-B); does the loop self-reinforce, do damping interventions help (N-C)?
    - *Ph2b ablations:* top F-a × **G** × **H**.
- **Selection → health + leading indicator (N-A + the link).** Vary **A**{dedup,F-a,F-b} × **B** ×
    **F**{correlated,drift+corruption}; measure the *diagnostic* (rank, drift) and the *validating* (per-era probe /
    forgetting), and test whether the diagnostic **predicts** the validating with lead time.
- **Phase 3 — Closed loop (N-E, N-G).** Vary **L**{open=B-open, closed} × **K**{oracle, label-free} × controller
    variants × **F**{correlated, drift+corruption}; fixed A=F-c, B anchored, D=centralized. *Asks:* (1) control benefit
    — closed vs B-open on health+downstream (oracle K); (2) detection-in-time — does label-free K suffice; (3) stability
    — oscillation?
- **Phase 4 — ZOD confirmation.** Top configs × **B**; E=ZOD, **I**=pretrained-adapt, F2. *Asks:* transfer to in-house
    data at realistic scale?
- **Phase 5 — FL (N-D, N-F).** Top configs × **D**{centralized-ref, FedAvg(+FedProx)} × **B** × non-IID splits ±
    global-aware selection. *Asks:* selection × aggregation skew (N-D); federation-level health analytics (N-F).
- **Monitoring & extension sub-studies (parallel).** Health detector: multivariate vs **B1.6** (median+hysteresis) vs
    per-channel EMA · Replay: {none, codebook} × **B** × scheduler · Telemetry: vision-only vs +telemetry (verdict
    prediction).

______________________________________________________________________

## Reading list

### Position against (work we must distinguish ourselves from) — grouped by the part of our approach each bears on

- **The coupling as a live loop (N-A):** DiSF (2025) — selection→collapse, *offline* · SOFed/FedCoCo (Shi 2022) —
    streaming-SSL selection, *single criterion, no loop* (also the closest prior for N-D).
- **Budget flip under co-adaptation (N-B):** CCS (Zheng 2023) · Sorscher (2022) · D2 Pruning (Maharana 2024) — *offline,
    fixed model*.
- **Selection-loop (in)stability (N-C):** one-sided-feedback (2020) · recsys/bandit sampling-bias loops · LLM
    self-consuming / model collapse (Shumailov 2023) — *other domains*.
- **Selection × aggregation skew in FL (N-D):** FedU (Zhuang 2021) · Orchestra (Lubana 2022) — *federated SSL without
    active selection bias*.
- **Health-monitor → control loop (N-E):** RankMe (Garrido 2023) · LiDAR (Thilak 2024) · dimensional collapse (Jing
    2022\) · ADWIN (Bifet 2007) — *health metrics / drift as offline diagnostics, not live control*.
- **Federation-level health analytics (N-F):** Krum (Blanchard 2017) · trimmed-mean/median (Yin 2018) — *bad-client
    detection by update geometry, not representation health*.
- **Health-steerable filter (N-G):** D2 Pruning (2024) · CCS (2023) — coverage/difficulty mix *fixed offline*, never
    steered live.
- **Collapse prevention (context for N-A/N-E):** VICReg (Bardes 2022) · IConE (2026) · AdaDim (2025) · CMP (2025) — *via
    loss/architecture, not selection, not a loop*.
- **Online-SSL degradation premise (Phase 1 / motivates B5):** continual-SSL line · RanDumb (2024) — *streaming degrades
    SSL; fixed features can rival online-learned ones*.
- **Noise-robust selection (temporal-robustness layer, `[EXT]`):** Co-teaching · DivideMix · FALSE · RSS-MGM —
    *supervised label noise; we transpose to label-free sensor corruption*.
- **Generative replay (replay codebook, `[EXT]`):** CAN (2025) · diffusion-as-replay — *federated generative replay; we
    differ by inversion + embedding-space + temporal scheduling*.
- **Automotive FL-SSL (the use case):** federated SSL for AV depth (2023).

### Toolbox we build on (work we use, does not threaten novelty of our work)

- **SSL backbones `[STD]`:** MAE (He 2022) · DINO (Caron 2021) · SwAV (Caron 2020) · SimSiam (Chen & He 2021) ·
    BYOL/SimCLR/MoCo (2020) · CaSSLe (Fini 2022).
- **Selection `[EXT]`:** Core-Set (Sener & Savarese 2018) · Herding/iCaRL (Welling 2009 / Rebuffi 2017) · SemDeDup
    (Abbas 2023) · GSS/MIR (Aljundi 2019) · reservoir (Vitter 1985).
- **OOD scoring (F-a) `[EXT]`:** kNN-OOD (Sun 2022) · Mahalanobis (Lee 2018) · Deep SVDD (Ruff 2018).
- **Health / collapse metrics `[EXT]`:** effective rank (Roy & Vetterli 2007) · RankMe (Garrido 2023) · LiDAR (Thilak
    2024\) · VICReg variance+covariance (Bardes 2022) · alignment & uniformity (Wang & Isola 2020) · dimensional collapse
    (Jing 2022) · representation drift via CKA (Kornblith 2019) · ADWIN (Bifet 2007).
- **CL & evaluation metrics `[STD]`:** linear probe + kNN eval (Wu 2018) · Backward Transfer (Lopez-Paz & Ranzato 2017)
    · Forgetting Measure (Chaudhry 2018).
- **Control / replay `[EXT]`:** Deep Generative Replay (Shin 2017) · Brain-inspired replay (van de Ven 2020) ·
    DeepInversion (Yin 2020).
- **FL `[STD]`:** FedAvg (McMahan 2017) · FedProx (Li 2020) · Marfoq (2023) · Scaleout FedN.
- **Datasets:** STL-10 (Coates 2011) · BDD100K (Yu 2020) · nuScenes (Caesar 2020) · ZOD.

______________________________________________________________________

## Risks & gates

Ordered: **validity risks** (could make us conclude the wrong thing) first, then **mechanism risks**, then **honest
nulls** (disappointing but still results), then **execution**.

| Risk                                                                             | Signal                                                                  | Response                                                                                                                                                                                    |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **False negative on degradation** (underpowered demo) — collapse *or* forgetting | No degradation seen under stress                                        | Pressure-sweep incl. from-scratch + no-replay + long horizon; collapse on joint-embedding, forgetting on MAE; **PC must fire** first                                                        |
| **Label-free signal can't detect in time** (leading-indicator null)              | Tier-1 signal doesn't precede Tier-2 forgetting, or no usable lead time | Oracle-driven control result still stands (decoupled); report which signals are/aren't leading indicators — a negative result of independent value; deployability becomes the open question |
| **Coupling is regime-specific** (init)                                           | Dynamics present from-scratch, absent pretrained (or vice versa)        | Report the init-dependence as the finding (the regime where adaptation matters); pretrained bounds realism — don't overclaim deployment relevance from a from-scratch-only effect           |
| **Synthetic-correlation artifact** (STL-10)                                      | Degradation on 1a pilot but not 1b BDD                                  | STL-10 validates the apparatus only; the claim rests on BDD's real correlation; non-transfer means class-blocking drove it — say so, lean on BDD                                            |
| Filter wins on collapse metric but not downstream                                | RankMe up, probe flat                                                   | Optimized the thermometer, not the patient — validate on downstream + forgetting                                                                                                            |
| **Actuator null** — F-c can't be steered                                         | Varying α in open-loop doesn't move health                              | The controller has no lever → F-c reduces to a static hybrid; **gated (see below)**                                                                                                         |
| Adaptation doesn't beat frozen (RanDumb)                                         | B5 ≥ adapted                                                            | Existential — reframe to selection-for-efficiency; surface in Phase 1                                                                                                                       |
| Closed loop doesn't beat open, *with oracle signal* (N-E/N-G null)               | Closed ≈ or < B-open                                                    | Still a result (closing this loop is hard/unhelpful); keep the monitor as a passive alarm                                                                                                   |
| Closed loop oscillates / destabilizes                                            | Health metrics ring                                                     | Slow the controller; hysteresis; bound the α excursion — the stability finding is itself reportable                                                                                         |
| Coupling result just confirms DiSF/CCS offline                                   | No surprise                                                             | Hunt loop instability / flip inversion / closed-loop benefit explicitly                                                                                                                     |
| Reservoir+dedup / SOFed hard to beat                                             | Bake-off flat                                                           | Fine — criteria were never the contribution; weight stays on the coupling + control loop                                                                                                    |
| FL infra slips / aggregation confounded                                          | FedN not ready; can't separate skew from non-IID                        | FL science is parallel and deferrable (doesn't block the spine); the centralized reference isolates selection-induced skew from ordinary non-IID                                            |
| Compute / Gaudi bottleneck; Phase-1 grid too large                               | Habana port slow; init × pressure × method blows up                     | CPU + small models for the demonstration; isolate Gaudi as a contained task; **trim rule:** full init×pressure sweep on the STL-10 pilot, carry 1–2 init levels to BDD                      |
| Drift back to the criterion bake-off                                             | Work optimizing filters, not studying the coupling                      | The Novelty-at-a-glance (N-A…N-G) is the honesty check — if work doesn't map there, it's drift                                                                                              |

**Gates (do not proceed until these pass):**

- **Phase 0:** the positive control collapses on RankMe — instruments validated before any conclusion.
- **Phase 3:** α demonstrably moves health in the *open-loop* study — confirm F-c is a real actuator before building the
    closed-loop controller.

**Single most important step:** Phase 0–1 (instrument + demonstrate the degradation envelope, with the positive
control), run first.
