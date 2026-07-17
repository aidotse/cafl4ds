‹ [Project Plan index](index.md)

# Experiments — baselines, phased plan, ablations (factor matrix)

*Tags: `[STD]` re-implement · `[EXT]` extend to our setting · `[NEW]` genuine contribution (see
[index](index.md#novelty-at-a-glance)). The `N-x` novelty claims are stated in full in [novelty.md](novelty.md).*

The building blocks (experimental basis like models, datasets, methods, etc.) to consider are listed in
[building-blocks.md](building-blocks.md)

## Baselines

**Backdrop:** the baselines are not all the same kind of thing. Most run the **full streaming loop with the selection
knob swapped** (the loop is the backdrop, the knob is the variable). Two of them **switch a loop component off** to
bound the loop from above and below. Two are **diagnostics**, not training conditions. The "Backdrop" column says which.

| ID | Baseline | Backdrop (loop config) | Role / when | Tag |
| -- | -- | -- | -- | -- |
| **PC** | Positive control: a config that deliberately induces a failure mode — **one per mode** (collapse, forgetting, instability) | diagnostic (deliberately-degrading) | **Calibrates** the instruments for that mode (must fire under the PC, stay quiet on a healthy baseline) — **Phase 0 gate, runs first**; without it a null is uninterpretable | `[STD]` |
| **B0** | Use-everything offline SSL | streaming **off** (full data, shuffled, multi-epoch, no selection) | Ceiling the Pareto approaches as buffer→∞; compute early | `[STD]` |
| **B5** | Frozen backbone, no adaptation — frozen-**pretrained** (pretrained regime) or frozen-**random** (from-scratch regime) | adaptation **off** | Existential floor — does adapting beat doing nothing? (RanDumb); compute early | `[STD]` |
| **B-floor** | No filter: accept all, no replay, raw order | full loop, buffer = 0, no selection | Max-degradation reference; Phase 1 | `[STD]` |
| **B1** | Reservoir sampling | full loop, random selection + replay | "Dumb" bar — replay is partly protective; Phase 1+ | `[STD]` |
| **B1.5** | Reservoir + SemDeDup | full loop, dedup selection | **The real selection bar**; Phase 1+ | `[STD]` |
| **B2** | Pool-based AL (storage allowed) | streaming **off**, selection on, large storage | "If we could store everything" reference | `[STD]` |
| **B3** | SOFed/FedCoCo loss-based importance | full loop (federated), loss-gate selection | Closest prior art; Phase 2 + Phase 5 | `[STD]` |
| **B-open** | Static F-c (fixed α) | full loop, F-c with frozen α, **loop open** | Open-loop control for N-E/N-G — needs F-c; **Phase 3 only** | `[STD]` |
| **B1.6** | Median filter + hysteresis on the signal | diagnostic, over the temporal layer | "No fancy temporal model" bar for the denoising layer | `[STD]` |

**Ordering:** PC (gate, first) → {B0, B5, B-floor} (bracket + Phase-1 floor) → {B1, B1.5} (the bars) → B3 (Phase 2 / 5)
→ B-open (Phase 3, once F-c exists).

**Levels at a glance:** *full-loop conditions* = B-floor, B1, B1.5, B-open, B3 (swap the knob) · *loop-bracketing
ablations* = B0 (no streaming, ceiling), B5 (no adaptation, floor) · *stored-selection reference* = B2 · *diagnostics* =
PC, B1.6.

## Phased plan - scientific spine (sequential)

The scientific spine is sequential — each phase needs the prior. The FL infrastructure and the health monitor run as
parallel tracks that start early. Confirmation on in-house data trails the spine. Model initialization threads through:
from-scratch (small/toy) to elicit the dynamics in Phases 1–3, pretrained-adapt to confirm they matter at realistic
scale (ZOD).

### Phase 0 — Instrument calibration `[STD]`. (M1)

Streaming loop + health metrics + B5 + a positive control **per failure mode**, on STL-10. Calibration is a two-sided
per-mode test: each metric must **fire** under a PC that induces its mode *and* **stay quiet** under a healthy baseline
— otherwise a Phase-1 signal is uninterpretable. Modes: **collapse** (joint-embedding; PC = predictorless SimSiam) ·
**forgetting** (MAE; PC = train era A then only era B) · **instability** (divergence; gradient-norm). **Exit:** every
mode calibrated.

### Phase 1 — Degradation envelope `[NEW]`. (M2)

- **1a — STL-10 pilot.** Class-blocked (synthetic) correlation; cheap and fast. Job: confirm the phenomenon *can* appear
    **and** the instruments catch it. Validates the apparatus, **not** the claim.
- **1b — BDD100K subset.** Real video correlation → the publishable make-or-break.
- Sweep pressure including **initialization `{from-scratch, lightly-pretrained, fully-pretrained}`** — from-scratch
    (small) is the degradation-sensitive primary; pretrained checks whether degradation persists or is masked.
    Instrument **both modes** (collapse on joint-embedding, forgetting on MAE); diagnostic = rank-vs-loss divergence;
    **PC must fire.**
- **Go:** a coupling exists. **No-Go:** reframe to selection-for-efficiency.

### Phase 2 — Open-loop criterion study `[NEW]` (N-A, N-B, N-C).

F-a / F-b / static-F-c as fixed knobs, in the regime where Phase 1 showed the dynamics clearest. Does the flip survive
co-adaptation? Does the loop self-reinforce, and do damping interventions help?

(N-A): selection → batch diversity → collapse/forgetting, run live (streaming/dynamic extension of DiSF's offline
finding).

**Leading indicator test**: Test whether / how far in advance the diagnostic metrics predict the failure mode, under the
loop dynamics. Example: does a Tier-free signal (rank drop, drift) *predict* the probe-based forgetting before the
labels would reveal it, with usable lead time?

[Evaluation](#evaluation) criteria apply.

### Phase 3 — Closed loop `[NEW]` (N-E, N-G).

Build the monitor→filter controller; close the loop with the steerable F-c (map the detected health regime to F-c's `α`
\+ acceptance rate and to replay/rollback. The closed loop is a control system — watch for oscillation; open-loop
(B-open) is the honest baseline.) Compare **B-open (static filter) vs. closed-loop (health-modulated)** on health +
downstream — does closing the loop help, and is it stable?

**Leading indicator test**: As in open-loop: Test whether / how far in advance the diagnostic metrics predict the
failure mode.

**Replay codebook `[EXT]`**: e.g. in FL case (but also applies to centralized setup): server inverts the global model
into a small static latent codebook, shipped with weights, mixed in by temporal-distance, scheduled by the monitor
(embedding-level, not pixel-MAE; refresh each round).

[Evaluation](#evaluation) criteria apply.

### Phase 4: Confirmation & extensions (trail the spine)

**ZOD confirmation `[EXT]`.** Top configs on a ZOD 2D proxy, **pretrained-adapt** (realistic scale). Confirms transfer
after the relevant centralized result.

**Optional.** Detection use case (mAP); deeper CF study; **federated closed-loop** (the Phase-3 loop studied under
aggregation).

### Parallel tracks (start early, alongside the spine)

**FL track.** *Infrastructure* (e.g. FedAvg + client simulation) built **in parallel from Phase 1**. *Science* (N-D
selection × aggregation skew + global-aware correction; N-F federation-level health analytics) is gated only on a
**centralized reference (Phase 2)**, so it slots in right after Phase 2, parallel with Phase 3. *Why centralized-first
for the science:* aggregation confounds the coupling, so an effect can't be attributed to averaging vs. streaming
selection until the centralized baseline exists.

**Monitoring track.** Signal analytics + health monitor built once Phase 1 confirms degradation exists (its required
input), ready by Phase 3 (which consumes it). Proceeds in parallel with Phase 2.

## Evaluation

Evaluation criteria apply for centralized and decentralized training; the former benchmarks the latter.

### Core SSL model

Three protocols for evaluating the performance of the SSL model, all vs. **B5** (frozen-**pretrained** in the pretrained
regime, frozen-**random** in the from-scratch regime) by ascending label budget (not a data flow):

- **kNN** — zero training: classify eval queries by nearest neighbors in the *frozen* features. The rawest test of the
    geometry.
- **Linear probe** — train *only* a linear head on the *frozen* features. Linear separability of the representation
    (standard SSL eval).
- **Few-shot finetune** — *unfreeze* and finetune on a *small* labeled budget. A different question: label-efficiency /
    how good a starting point the representation is, not frozen quality. (Few-shot, so abundant labels don't wash out
    representation differences.)

**Example figures:** for each model init regime (against B5): quality vs. buffer size (the flip), vs. stream samples,
vs. label budget.

### Loop dynamics

**Signal-quality eval `[EXT]`:** segment-ordered streams with known shift points + injected sensor corruption →
true-shift recall vs. noise rejection vs. confirmation lag. Validates the temporal-robustness layer.

**Monitor / closed-loop `[NEW]` — two separable results:**

1. **Control benefit:** does closing the loop beat **B-open**, driving the controller with the best / oracle signal
    (probes allowed)?
1. **Detection-in-time:** does a *label-free* signal flag degradation with enough lead time (the link above)?

Keeping (1) and (2) apart makes a null interpretable — control failed, vs. the detector missed.

## Experiment matrix

### Factors

|  | Factor | Levels | Notes |
| -- | -- | -- | -- |
| **A** | Selection criterion (the knob) | B-floor (no filter) · reservoir · dedup · loss-gate · SOFed · F-a novelty · F-b coverage · F-c steerable | Independent variable. First five = baselines; F-a/F-b/F-c = our filters. |
| **L** | Loop | open · closed | Closed requires F-c + the monitor→filter controller. |
| **B** | Storage budget | 0 · tiny · small · medium · ∞ | The Pareto axis; where the flip lives. |
| **I** | Initialization | from-scratch · lightly-pretrained · fully-pretrained | Degradation-sensitivity ↔ realism knob. From-scratch elicits the dynamics; pretrained confirms they matter. Sets B5's form (frozen-random vs frozen-pretrained). |
| **C** | SSL method | MAE · joint-embedding (SimSiam/BYOL/DINO/SimCLR) | MAE → forgetting mode; joint-embedding → collapse mode (the collapse demo). SimSiam = minimal JE variant, implemented first. |
| **D** | FL setting | centralized · FedAvg · FedProx | Centralized first (clean dynamics); FL science gated on the centralized reference. |
| **E** | Data | STL-10 · BDD100K · ZOD | Harness → real make-or-break → in-house confirmation. |
| **F** | Stream structure | IID (control) · correlated · drift+corruption | Correlated = headline; drift+corruption for signal-quality + health. |
| **G** | PCA dim (F-a) | full · 32 · 16 · 8 | "How far can we compress before novelty degrades." |
| **H** | Reference encoder (F-a) | frozen · adapting | Tests the stability invariant. |
| **K** | Controller signal (when L=closed) | oracle (probe-based) · label-free | Separates *control benefit* from *detection-in-time*. |
| **P** | Pressure sweep | correlation · LR · model size · horizon · replay on/off | Stress knobs to locate the degradation envelope. |

*B5 is always init-matched (frozen-random for from-scratch, frozen-pretrained for pretrained). Probes are used freely
throughout (the study, not an edge pipeline).*

> **Note on initialization:** its just another factor, not a fixed choice (→ factor I). A fully-pretrained backbone sits
> in a good basin and adapts gently — it can **mask** collapse and produce a false negative. So sweep init
> `{from-scratch, lightly-pretrained, fully-pretrained}`:
>
> - *From-scratch* (small model, toy/proxy data) is the **degradation-sensitive** setting and the natural home for the
>     Phase-1 demonstration — feasible here because we're eliciting dynamics, not chasing quality features (so "ZOD too
>     small for from-scratch" doesn't apply).
> - *Pretrained-adapt* is the **realistic/deployment** setting where the selection results must ultimately hold, and
>     what compute forces at BDD/ZOD scale.
> - Two senses of "degradation" follow the init: *eroding* good pretrained features (forgetting) vs. *failing to form*
>     them (collapse). Both worth eliciting. Do **not** lock the demonstration to pretrained-only.

### Grids (anchor a config, vary 1–2 factors per phase — never the full Cartesian)

- **Phase 0 — Instrument calibration.** Not a factors sweep, but run ablations to figure out the proper baseline for
    each metric (*Asks:* under which operating regime does each metric behave correctly, or unexpectedly?)
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
    — oscillation? — Replay: {none, latent codebook}
- **Phase 4 — ZOD confirmation.** Top configs × **B**; E=ZOD, **I**=pretrained-adapt, F2. *Asks:* transfer to in-house
    data at realistic scale?
- **FL (N-D, N-F).** Top configs × **D**{centralized-ref, FedAvg(+FedProx)} × **B** × non-IID splits ± global-aware
    selection. *Asks:* selection × aggregation skew (N-D); federation-level health analytics (N-F).
- **Monitoring & extension sub-studies (parallel).** Health detector: multivariate vs **B1.6** (median+hysteresis) vs
    per-channel EMA × **B** × scheduler · Telemetry: vision-only vs +telemetry (verdict prediction).
