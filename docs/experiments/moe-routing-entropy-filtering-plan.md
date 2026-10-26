# MoE-MAE routing entropy filtering: experiment design

Design agreed and implementation executed on 19 October 2026. This document records the protocol; experiment results
belong in [WP-M1](phase1/WP-M1.md). The implementation adds the requested full-softmax top-1 gate, entropy admission,
identity-preserving audit records, and a resumable paired experiment runner.

## Confirmed decisions and remaining questions

The project lead has selected:

- **Scientific target:** how filtering changes routing and representation health.
- **Routing:** top-1, applying softmax across all experts before top-k and retaining the selected probability as its
    output weight, so reconstruction supplies a router gradient.
- **Initialization:** random initialization, with filtering active throughout training.
- **Exposure:** multiple epochs, reshuffling every epoch, under a fixed common training budget.
- **Scoring view:** deterministic full-image scoring, with a masked-view sensitivity check.

Run specifications frozen after the engineering and throughput pilots:

- Two full epochs, 1,904 updates per arm, candidate batch 64 and training batch 32, seeds 0–4.
- CPU with four threads. A measured throughput check found this faster for the small model than the local MPS GPU.
- Existing verified 64 px BDD training-image cache, with its declared four-scene/defined-attribute restrictions.
- Disjoint support, balanced-query and natural-frequency-query panels of 200 images each; 60,955 training images.
- Fixed reconstruction masks: four draws per image. Scene classification is the semantic canary.
- Descriptive paired seed intervals; practical reference boundaries of 0.02 normalized entropy, 5% relative MSE, and
    three percentage points of probe accuracy. No confirmatory significance or equivalence claims from these runs.

The frozen machine-readable protocol is `outputs/moe_routing_entropy_2026-10-19/protocol.json`, hash
`a34eaad9948a5229ce71a95f266d71691a37a3afb19deef99a5613ac4c0871c8`. Pilot and throughput runs are excluded from the
measured comparison. There is no arm-specific tuning, early stopping, or warm-up without filtering.

## Research question and fit

At a fixed acceptance fraction of 50%, how does repeatedly selecting high- versus low-routing-entropy images change
subsequent routing, expert use, and representation quality relative to uniform random selection?

This studies the project's fast feedback edge: model → selection → model. It primarily supports the question of
self-reinforcing selection dynamics, N-C in the [project plan](../project-plan/index.md), and provides a stationary
reference for the healthy-movement protocol in [Phase 1](phase1/index.md). The selection rule stays fixed while its
scores adapt. The health monitor does not control selection.

The shuffled study cannot establish forgetting under distribution shift, the correlated-stream Phase-1 exit gate, the
storage-budget flip N-B, a benefit from closing the health-control loop, or effects of federated aggregation. Selecting
half a batch changes admission to gradient updates; it does not itself define a storage budget.

Routing entropy is already a research signal. Lee, Shin and Park use it for unfamiliar-input detection in autonomous
driving ([paper](https://doi.org/10.1109/ASE63991.2025.00332)). MoCE already combines masked autoencoding and expert
specialization, using cluster-conditional gates ([paper](https://arxiv.org/abs/2402.05382)). These motivate the study;
neither establishes the outcome of live entropy-based image admission. The potential contribution is the measured
feedback between admission, routing, and learning. A broader novelty review is needed before claiming priority.

## Critique and interpretive limits

- **Entropy is routing ambiguity, not established epistemic uncertainty.** High values may reflect weakly differentiated
    experts, nearly uniform router logits, or difficult inputs. Low values may reflect useful specialization, arbitrary
    confidence, or starvation. Neither direction is automatically desirable.
- **The current top-1 implementation must change before this study.** It normalizes selected logits over selected
    experts only, making the top-1 weight identically one. The agreed full-softmax gate preserves reconstruction
    gradients to router parameters. This also scales the expert branch by its selected probability. Apply the same gate
    to every arm and identify the new semantics in configs/checkpoints; WP-M0 used the earlier gate.
- **Entropy can affect update strength as well as the data diet.** More diffuse distributions can have smaller selected
    probabilities, attenuating expert outputs and gradients. Log selected probability and expert-branch norms so an
    effect is not automatically interpreted as selecting more informative images. At initialization, entropy rankings
    may reflect arbitrary random features; their subsequent self-reinforcement is part of the research question.
- **Batch balance differs from token ambiguity.** Uniform aggregate expert use can coexist with very confident per-token
    decisions. A balancing objective is not the same as maximizing each token's entropy.
- **Selected-set entropy is mechanically biased.** Higher entropy in the high-entropy arm's accepted batch is a wiring
    check. Evidence of model change must come from identical reference images across arms and checkpoints.
- **Mean entropy hides structure.** Two images can have equal means but different patch patterns or layer profiles. Keep
    per-layer summaries and sampled token maps even though the selector uses one scalar.
- **Masking changes contextual inputs.** Full-image scoring and visible-token MAE routing need not agree. Masked-out
    patches have no encoder router distribution in that forward pass.
- **Fewer backward examples need not mean less total compute.** Scoring all candidates through the encoder adds work.
    Measure its cost. This experiment also does not imply less image decoding or transient batch storage.

## Metric specification

Let image i contain T patch tokens, and let L denote only the encoder's MoE layers. Dense blocks and the dense decoder
are excluded. At MoE layer l, let E_l be the number of experts and z_ilt the clean router logits for patch t.

Define the full distribution before top-k truncation, with fixed temperature one:

$$
p_{ilte}=\operatorname{softmax}_e(z_{ilt}),\qquad
H_{ilt}=-\sum_{e=1}^{E_l}p_{ilte}\log p_{ilte}.
$$

Compute in float32 with stable log probabilities and the convention 0 log 0 = 0. Exclude CLS from selection; log its
entropy separately if useful. The requested per-token mean over layers is

$$
\overline H_{it}=\frac{1}{|L|}\sum_{l\in L}H_{ilt}.
$$

The raw per-image score is S_i = mean_t(overline H_it). Also report a normalized score:

$$
U_i=\frac{1}{|L|}\sum_{l\in L}\frac{1}{T}\sum_{t=1}^{T}\frac{H_{ilt}}{\log E_l}.
$$

With the same expert count at every layer, raw and normalized scores produce identical rankings. Use U_i for selection
and reporting, and retain raw entropy in nats. If layer expert counts later differ, choosing raw versus normalized
entropy becomes a substantive weighting decision and must be declared. Require E_l >= 2.

Average **entropies**, not distributions. Entropy of the average distribution would mix confident assignments to
different experts with ambiguous assignments. Expert indices also have no shared semantic identity across layers.

Use actual sequential block inputs. Score with no gradients, router noise, stochastic augmentation, or parameter
updates. Restore module modes and RNG state. The existing `encoder.routing(images)` supports the full-image path. Token
identity is the patch position for this view; do not average positions across different crops. For masked sensitivity
checks, use visible patch identities consistently across layers and average image scores over fixed masks.

### Agreed training gate

For a token representation x, let q = softmax(z + epsilon) over all experts, where epsilon is training router noise. Let
e\* = argmax_e q_e. The MoE feedforward output is

$$
y=q_{e^*} f_{e^*}(x).
$$

Do not renormalize q_e\* after selection or detach it. The discrete expert choice is not differentiated, but the
selected probability is. Since that probability depends on the full softmax, its gradient can affect all router logits,
while only the dispatched expert executes for that token. Clean pre-noise p remains the entropy measurement; training q
and selected q_e\* are separate diagnostics. This follows V-MoE's
[router](https://github.com/google-research/vmoe/blob/main/vmoe/nn/routing.py) and
[combine-weight handling](https://github.com/google-research/vmoe/blob/main/vmoe/moe.py). Retain the project's existing
no-capacity-limit/no-token-dropping dispatch; matching gate weights does not imply every V-MoE design choice is adopted.

## Primary experiment and controls

| Arm | Selection within each incoming batch | Accepted count |
| -- | -- | -- |
| Random-half | Uniform sample without replacement, independent of scores | B/2 |
| High-entropy-half | Largest pre-update U_i values | B/2 |
| Low-entropy-half | Smallest pre-update U_i values | B/2 |

Use an even candidate batch size; provisional B=64 gives 32 training images per update, matching the earlier WP-M0
training batch size. Drop the final incomplete candidate batch in this stationary study and report the excluded count.
Resolve entropy ties by a reproducible random permutation from a dedicated tie-breaking generator. Reject non-finite
scores as a run failure rather than silently changing the admission rule.

For each candidate batch: score every image using the current pre-update model, select once, and make one MAE update on
accepted images. Both reconstruction and balancing losses use accepted images only. Discard rejected images without
replay. Keep the reconstruction objective, augmentations, masking ratio, optimizer, LR schedule, and balancing weights
fixed across the three arms.

Use five paired seeds for the measured comparison after a seed-0 engineering pilot. Within a seed, clone exactly the
same randomly initialized encoder and decoder into all three arms, with fresh identical optimizer states. Save the
shared initialization hash. Do not use a warm-up period that bypasses filtering. Use the same candidate order, candidate
batch boundaries, update count, and selected-image count. Each arm then scores with its own evolving model.

Separate random streams for initialization, data order, random admission/ties, augmentation, MAE masks, router noise,
and evaluation. Key augmentation/mask draws by image identity and exposure where feasible so shared images receive
matched draws despite different selected positions. Logging/evaluation must not advance training RNG.

Score the random arm too for common diagnostics. Report this instrumentation cost separately from the cost of a
deployable random selector, which needs no scoring forward. Primary comparisons match exposure and training budget;
equal wall-time or FLOP comparisons require a separately declared efficiency protocol.

Carry the shared frozen initial checkpoint as the no-adaptation reference B5. An accept-all arm is a useful secondary
context control if resources permit; it sees twice as many gradient examples at the same candidate exposure and does not
belong in the matched half-selection contrast.

## Dataset, initialization, and horizon

Use BDD100K still images with a persisted random permutation per seed and epoch, preserving natural attribute
frequencies. Reserve evaluation images before shuffling. Attributes and labels may define evaluation and audit selection
bias, but must not enter the selector or MAE loss. Persist image IDs, content/order hashes, inclusion rules, and split
membership. Keep clips disjoint if using multiple frames per clip. BDD100K provides varied scene, weather, and
time-of-day conditions ([dataset paper](https://arxiv.org/abs/1805.04687)).

The [WP-M0 results](WP-M0-warm-well-mae-bdd-results.md) used 61,555 usable training-split images across four retained
scenes, including 200 reserved support/query images. This is a restricted usable pool, not all 100,000 nominal dataset
images. Recheck counts for this study and declare whether those scene restrictions are retained. Use larger fixed
evaluation panels if feasible; 100 query images give limited precision. Report natural-frequency and balanced scene
performance separately, and audit acceptance by scene, weather, time of day, and their sufficiently populated groups.

WP-M0 supplies engineering references and measurement recipes, not initial weights for this from-scratch study.

Architecture: existing 64 px MAE, patch size 8, four token-routed experts in blocks 2/3, dense decoder, mask ratio 0.75,
existing balancing weights and auxiliary coefficient 0.01. These establish small-scale dynamics; confirm any substantive
result at a larger resolution before interpreting semantic specialization.

Use repeated epochs with an explicit common epoch/update budget and a fresh shuffle each epoch. With N images and
candidate batch size 64, each complete epoch has floor(N/64) updates. Report candidate exposures, accepted exposures,
and unique accepted images separately: an image rejected in one epoch can be accepted in another. Do not silently reuse
WP-M0's 2,000-update cap as an epoch count. Pin the LR recipe across arms without arm-specific tuning or boundary
restarts. Use the pilot to set and freeze a common horizon before measured seeds. Single-pass experiments are a later
restriction with their own interpretation.

## Measurements and analysis

| What | Measurements | Why |
| -- | -- | -- |
| Training | Pure reconstruction, raw and weighted balancing loss, total objective, gradient norms, LR | Separate task learning from balancing |
| Selection | Candidate/accepted/rejected entropy means and quantiles, IDs, acceptance by attribute, retained count | Verify the rule and characterize its induced diet |
| Fixed-reference routing | Layer-mean token entropy, image means/quantiles, per-layer entropy, patch maps on a small fixed panel | Detect model change independently of selection bias |
| Expert use | Per-layer hard assignment shares, soft probability mass, load entropy, load CV, minimum load, starvation over rolling windows | Distinguish ambiguity from concentration and under-training |
| Gate scaling | Selected full-softmax probability, expert output norms before/after weighting, router/expert gradient norms | Reveal confidence-dependent changes in branch/update strength |
| Route stability | Assignment changes on identical patches versus initialization and prior checkpoint; top-k overlap for k>1 | Track routing reorganization |
| Learning quality | Held-out masked reconstruction with fixed masks; fixed-protocol kNN and linear scene probes | Test whether routing movement accompanies useful learning |
| Representation health | Existing RankMe, drift and other calibrated channels with their trust tags | Contextual evidence, not standalone proof of degradation |
| Cost | Scoring/update/evaluation time separately, total elapsed time, images processed, peak memory | Quantify the price of selection |

At every update log scalar losses, selection summaries and cheap counters. Evaluate fixed routing/reconstruction panels
at initialization, every 50 updates, and the endpoint; evaluate probes at initialization, about 25%, 50%, and 100% of
the horizon. Deduplicate coincident checkpoints. Use identical panels, masks, evaluation recipes and timing across arms.
Record deterministic full-image reads separately from noisy masked training-route statistics.

Proposed primary routing outcome: endpoint fixed-panel mean U_i, adjusted by its step-0 value. Compare high-minus-random
and low-minus-random within seed. Proposed primary learning outcome: held-out reconstruction MSE; report scene-probe
accuracy as the semantic corroboration. Freeze this hierarchy and practical effect thresholds after the pilot and before
measured runs. There is no presumed beneficial direction for entropy.

Show per-seed curves and paired effects with uncertainty intervals. Seeds are the independent training replicates;
tokens, checkpoints, mask draws, and images are not extra training replicates. With five seeds, emphasize effect sizes
and interval width. If using formal hypothesis tests, adjust the two planned filter-versus-random contrasts and declare
how multiple primary outcomes are handled. Do not select whichever metric or checkpoint happens to win.

On a fixed diagnostic panel, relate routing entropy to masked reconstruction error and to attribute groups. This tests
what the score tracks without assuming that entropy equals difficulty. Mask sensitivity should report rank correlation
and overlap of selected halves over multiple fixed masks. If rankings are unstable, retain that as a result or revise
the scoring view before measured runs. Interpretation: changed routing with unchanged quality is reorganization; quality
loss plus starvation supports a harmful association; neither alone proves a specific causal mechanism.

## Implementation sequence and verification

1. **Freeze the protocol.** Resolve the questions above, data memberships, initialization policy, horizon, primary
    outcomes, practical thresholds, and compute allowance. Preserve the three requested arms.
1. **Implement the agreed top-1 gate.** In `cafl4ds/models/vit.py`, take the full softmax before selection and gather
    the selected probability without renormalizing. Preserve sparse dispatch and existing balancing semantics. Update
    architecture documentation and tests that encode the old gate. Prove a nonzero router gradient from reconstruction
    with balancing disabled on a controlled nondegenerate example, and verify the forward output equals the selected
    probability times its expert output. This is the first implementation prerequisite.
1. **Add reusable metrics.** Put entropy reduction in `cafl4ds/measurements.py`, using `MoEViTEncoder.routing` rather
    than duplicating the model. Reuse the mathematical convention from WP-M0's runner, but move reusable logic into the
    package. Return per-token, per-image, and per-layer values explicitly. Raise clear errors for dense/image-routed
    encoders when a token-routing study is requested.
1. **Add selectors and audit output.** Implement exact random-half and entropy-rank admission behind `Filter.select`,
    configured through Hydra. The current filter interface returns only image tensors and `StreamBatch` carries no
    sample IDs. Add a backward-compatible way to retain stable sample IDs and emit a detached selection record with
    scores, accepted indices, and metadata. Keep labels outside the selector. Avoid persistent full probability
    tensors.
1. **Wire a dedicated experiment runner.** Reuse `RegimeStream.stationary_batches()` or a persisted manifest adapter for
    shuffled BDD data; clone initial states across arms. Extend logging for separate selection and training records,
    capture loss components from the actual update, and write fixed-panel routing/quality trajectories. Keep business
    logic in the package and the entry point in `scripts/`.
1. **Verify scientifically meaningful invariants.** Test uniform/one-hot entropy, mean-of-entropies versus
    entropy-of-mean, CLS exclusion, layer reduction, expert permutation invariance, exact selection counts/directions,
    reproducible ties/random selection, non-finite failures, and detached scoring with unchanged parameters/modes/RNG.
    Check that only accepted images contribute to update/balancing losses, IDs stay aligned, train/eval are disjoint,
    paired candidate orders match, and checkpoint resume preserves selection.
1. **Pilot, then measure.** Run a short synthetic end-to-end check and a seed-0 BDD pilot including masked/full ranking
    sensitivity, actual router/expert gradient paths, timing, and memory. Validate reload and evaluation
    reproducibility. Freeze any pilot-driven changes before running the five paired seeds. Exclude the exploratory
    pilot from the final summary if it influenced choices.
1. **Produce a reviewable result.** Save resolved configs, initialization/data/code hashes, runtime provenance, seed
    outcomes, selected-ID records, checkpoints including optimizer/RNG/cursor state, strict-JSON summaries, and plots.
    Analyze the three arms together, including null results and failures. Add experiment-index entries when its place
    in the study sequence is agreed; this draft does not revise the existing phase gates.

## Extensions after the shuffled reference

**Separate feedback from static scoring.** Repeat high/low selection with a frozen scoring copy while the learner
adapts. Dynamic-versus-frozen scoring is needed to attribute an effect specifically to selector co-adaptation rather
than simply a different fixed training diet. A router-frozen learner ablation addresses a different mechanism.

**Test the gate-training dependence if needed.** The legacy selected-only-softmax top-1 gate could serve as a targeted
balancing-only ablation, with its output-scale difference explicitly acknowledged. Balancing strength and top-k can also
be studied separately if the primary result motivates them. Avoid changing architecture and selection in a single
uncontrolled comparison.

**Introduce nonstationarity.** Replace the shuffle with BDD scene blocks or time-of-day/weather regimes while holding
image membership and acceptance fraction fixed. BDD object labels are multi-label: class-blocking by object category
requires an explicit assignment/overlap policy. Scene blocking or attribute regimes provide simpler unambiguous eras.
Retain an orthogonal canary target where possible so blocking and probing are not tautological.

**Measure forgetting.** Evaluate fixed held-out panels from every past era at each boundary. Build an accuracy matrix
A_jk for performance on era k after learning through era j. Report backward transfer relative to performance just after
learning each era, and forgetting as the drop from its best observed past performance. Use shared multi-class probe
targets; accuracy on a single-class panel without a defined common classification problem is uninformative. Track
per-era reconstruction, routes, and expert load too. Distinguish a fixed old probe (readout compatibility) from refitted
probes (current separability), and corroborate important forgetting claims with few-shot fine-tuning. Carry the
calibrated positive control and B5 reference.

Keep the LR schedule fixed across era boundaries. Re-warming is itself a forgetting confound in continual pretraining
([Beyond Cosine Decay](https://arxiv.org/abs/2503.02844)). Test whether entropy predicts later quality loss using only
past/current measurements and held-out runs; concurrent correlation is insufficient to establish warning lead time.

Only after harmful dynamics are established should a health-driven selector, replay intervention, retention-budget
sweep, or federated setting be added. Those extensions address separate project claims and need their own controls.
