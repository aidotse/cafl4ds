‹ [Project Plan index](index.md)

# Metrics — signal analytics, health, control, evaluation

*Tags: `[STD]` re-implement · `[EXT]` extend to our setting · `[NEW]` genuine contribution (see
[index](index.md#novelty-at-a-glance)). The `N-x` novelty claims are stated in full in [novelty.md](novelty.md);
baselines (B5, B-open, …) in [experiments.md](experiments.md#baselines).*

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
