# WP-M0: MAE/BDD native warm-well comparison

This is an execution and analysis plan, not an experiment result. It is grounded in the
[MoE work packages](https://docs.google.com/document/d/1h1MxKhs0Bz2-21CArZ0lvcU2ZnvEaOoquVsFrt76FzY/edit) and the
current working tree. The objective is a competent, reusable native MoE well and a matched dense comparison. WP-M0's
target is parity and correct wiring, not a claim about superiority or forgetting.

## 1. Scope and current implementation

Run MAE on a stationary shuffle of BDD's training pool, using batch-balanced top-1 token routing in the last two encoder
blocks. Use from-scratch initialization for all primary arms.

| Feature | Current state | Treatment in this experiment |
| -- | -- | -- |
| Native MoE, top-k, token/image routing | Implemented | Top-1 token routing is the primary arm |
| Batch importance/load balancing | Implemented | Both internal weights 1; SSL auxiliary coefficient 0.01 |
| EMA and loss-free bias balancing | Unimplemented; rejected by constructor | Deferred; no placeholder arms |
| Cross-view routing consistency | Unimplemented; proposed for joint-embedding SSL | Deferred; not needed for this single-view MAE comparison |
| Upcycling from a dense well | Unimplemented | Deferred; primary MoE starts natively from scratch |
| Read-only per-layer routing | Implemented | Use for held-out evaluation |
| Routing columns in health corpus | Unimplemented | Add the minimal evaluation readout before measured runs |
| Warm loss/health trajectory export | Warm loop returns loss trace, script discards it and logs final competence | Add persistent instrumentation before measured runs |

Two deliberate differences from the proposal should be recorded: the current implementation dispatches only selected
tokens to each expert, whereas WP-M0 proposes evaluating every expert and masking the result; auxiliary losses are
returned explicitly from each call instead of stored as last-forward state. The comparison below evaluates the current
sparse dispatch. It does not establish compiled-accelerator compatibility or a speedup from sparsity.

## 2. Three primary arms

Hold image size, embedding dimension, depth, attention, patch size, decoder, optimizer, mask ratio, data, and evaluation
protocol fixed. Change only the encoder feedforward architecture.

| Arm | Encoder | MLP ratio | Encoder parameters | Full MAE parameters |
| -- | -- | -- | -: | -: |
| Dense active reference | `tiny` | 2.0 | 324,192 | 414,176 |
| Dense total-capacity reference | `tiny` | 5.0 | 546,528 | 636,512 |
| Native sparse MoE | `moe_tiny`, E=4, k=1, blocks [2,3] | 2.0 per expert | 547,880 | 637,864 |

These counts were verified by instantiating the composed MAE configurations at 64 px. The decoder has 89,984 parameters
in every arm. The total-capacity reference is an approximate match: its encoder has 0.25% fewer parameters than the MoE,
including router and bias overhead. Record that difference. Do not widen the embedding dimension: that would also change
attention, representation shape, and decoder projection.

Top-1 MoE activates the same expert MLP width per token as the plain dense reference, plus routing. The corresponding
per-token encoder parameter count is 324,968 versus 324,192. This is a conceptual active-parameter comparison, not a
FLOP measurement: different tokens in a batch can collectively use all experts.

Use five paired training seeds, 0–4: 15 primary warm runs. First pilot the three arms on seed 0. Optional top-2,
image-routing, or auxiliary-weight ablations come after the primary comparison and must not replace failed primary seeds
or be mixed into its summaries.

## 3. Data and controls

- Use `data=bdd_local data.split=train`. The local config points to `/Users/williammin/cafl4ds_datasets/BDD`, uses
    per-image JSON labels, and both train directories exist. File-level image/label completeness and usable counts still
    need a preflight check.
- Use 64 px images, patch size 8, batch size 32, MAE mask ratio 0.75. There are 64 patch tokens; masked training
    presents 16 visible patches plus CLS to the encoder. Evaluation embeddings and ordinary routing readouts use all 64
    patches plus CLS.
- Reserve 25 support and 25 query images per scene with fixed `stream.canary_seed=0` across every arm and seed. Use
    `data.min_canary_count=100` so every retained scene can reserve 50 images and still contribute training examples.
    Report the actual scene set, class counts, regime counts, training size, and chance accuracy; do not assume there
    are exactly three scenes.
- Use the same post-reservation training pool. Keep `max_images=null` and `max_train_per_regime=null` for measured runs.
    The stationary shuffle preserves natural regime frequency; it is not a regime-balanced sampler. Training is
    label-free; attributes define the existing pool and labels are used only for downstream analysis/probes.
- Persist retained image identities, support/query/train membership, and their hashes. The current attributed-image
    object does not expose filenames, so retaining those identities is a small preflight addition. Verify no overlap and
    identical membership across architectures.
- Pair sample order, augmentation draws, and MAE masks by seed and step. A shared `torch.manual_seed` alone is
    insufficient: architecture initialization and MoE router noise consume different random draws. Separate model
    initialization, data order, augmentation/masking, router-noise, and evaluation RNG streams before claiming exact
    pairing. Evaluation must restore CPU and device RNG states.
- Fix AdamW at lr 0.001, weight decay 0.05, global gradient clipping 1.0, and no learning-rate scheduler, matching the
    warm configuration. Keep the MAE decoder dense and identical in shape.
- Use one device, precision, and thread setting across arms. CPU is the conservative wiring check; select MPS or another
    device only after a forward/backward/checkpoint pilot on that device. Run sequentially, since BDD is decoded
    eagerly. Measure peak memory; do not substitute a smaller pool silently if the full train split does not fit.

## 4. Preparation needed before measured runs

1. Run relevant model/MoE, SSL, warm-up, and script tests. Compose all three MAE configurations and verify parameter
    counts, masking shapes, decoder density, and auxiliary coefficient.
1. Run a short synthetic MAE warm/checkpoint/reload cycle, then a 200-step seed-0 BDD pilot for all three arms. A capped
    BDD subset is acceptable for wiring only; mark it as such. Verify finite losses/gradients, selected-expert
    gradients, nonzero router gradient from balancing, and device support.
1. Add a warm-run observer that logs scalar MAE reconstruction, raw auxiliary loss, weighted auxiliary contribution,
    total objective, gradient norm, image count, time, and per-layer expert assignment counts. Capture losses from the
    actual training call; do not run an extra stochastic forward to reconstruct the auxiliary term. Dense auxiliary
    terms are zero.
1. Persist the returned warm summary, resolved config, full-method checkpoints, periodic held-out health, routing
    metrics, and code provenance. Capture the current Git SHA, tracked diff, and untracked source/config files: the new
    MoE implementation is currently uncommitted.
1. Save evaluation checkpoints at step 0 and every 50 updates through step 2,000. Evaluate lightweight
    geometry/routing/reconstruction every 50 steps; run kNN and linear probes at 0, 200, 500, 1,000, 2,000 and the
    selected plateau checkpoint. Reuse fixed probes and unchanged probe hyperparameters.

`backbone=mae` supplies its own SSL configuration; it does not import `configs/ssl/mae.yaml`. Consequently the current
warm command needs `+ssl.aux_weight=0.01` to make the coefficient explicit. The factory default is already 0.01, but
recording it in the resolved config is necessary.

## 5. Training budget and well selection

Use a **fixed 2,000-update primary comparison**, with identical samples per update. This prevents an early, noisy
plateau from giving one architecture a different training budget. Disable early plateau stopping by setting
`warmup.min_steps=2001`, `warmup.max_steps=2000`, and `warmup.max_passes=50`. Check that the pass limit permits 2,000
steps; record actual samples, passes, and stop reason. Here `plateaued=false` in the stock summary is expected because
its early stopping is disabled.

Identify a warm-well candidate separately from the saved trajectory. Starting at step 200, inspect 50-step
reconstruction-loss windows with tolerance 0.001, require three successive stable comparisons, and check fixed held-out
reconstruction and canary competence. Use reconstruction rather than the combined MAE-plus-balancing objective for this
comparison. The stock criterion treats a loss rise as settled; distinguish a stable plateau from sustained worsening and
do not certify divergence as competence. Treat small stochastic rises as noise only when held-out checks support
stability.

Save both the step-2,000 method and the earliest checkpoint satisfying the predefined well criteria. Compare step-2,000
results as the primary matched-budget outcome; report time/updates to competence as a separate result. If an arm has not
become competent by the cap, label it unresolved/cap-limited. Any longer budget must be a declared follow-up applied to
all arms, not a quiet extension for one model.

The following commands are valid with the current configs. They run the stock warm script and save its endpoint; the
observer/export additions above are still needed for the full analysis protocol. Use `device=cpu` below for preflight,
then use the chosen common device for measured runs.

```bash
for s in 0 1 2 3 4; do
  for arm in dense_active dense_total moe_native; do
    case "$arm" in
      dense_active) architecture=(encoder=tiny encoder.mlp_ratio=2.0) ;;
      dense_total) architecture=(encoder=tiny encoder.mlp_ratio=5.0) ;;
      moe_native) architecture=(encoder=moe_tiny) ;;
    esac
    uv run python scripts/warm_well.py \
      data=bdd_local data.split=train data.min_canary_count=100 \
      backbone=mae init=from_scratch img_size=64 batch_size=32 device=cpu \
      stream.support_per_canary=25 stream.query_per_canary=25 stream.canary_seed=0 \
      seed="$s" +ssl.aux_weight=0.01 "${architecture[@]}" \
      warmup.min_steps=2001 warmup.max_steps=2000 warmup.max_passes=50 \
      well_out="wells/wp_m0_mae_bdd/${arm}/s${s}/step2000.pt" \
      hydra.run.dir="outputs/wp_m0_mae_bdd/${arm}/s${s}"
  done
done
```

The array syntax is for bash/zsh. Do not pass deployment-only `gate_arms` or `eval_every` flags to `warm_well.py`. Pilot
estimates determine total wall time; no current MoE timing estimate is assumed.

## 6. Measurements and analysis

### Reconstruction and representation competence

Read pure masked reconstruction MSE on a fixed held-out query panel and a fixed training reference panel, using the same
evaluation augmentation and eight reproducible mask draws per image. Evaluation runs in eval mode under no-grad; router
noise is off, MAE masks remain stochastic unless explicitly fixed. Average over masks and record the spread. This
separates generalization from router regularization. Do not compare MoE's total training objective directly with dense's
pure reconstruction objective.

Report fixed-budget held-out MSE, train-reference minus held-out gap, kNN accuracy, linear-probe accuracy, RankMe,
normalized feature variance, covariance, uniformity, alignment, and drift from step 0. Lower reconstruction MSE need not
imply better embeddings. Drift is an observation of change, not a damage verdict. Optionally report reconstruction by
regime as descriptive analysis, without tuning training or selecting checkpoints using regime labels.

### Expert utilization and similarity

Read each MoE block separately on the fixed full-image query panel. For top-1, define hard expert load as the fraction
of patch decisions assigned by argmax; report counts and fractions for every expert. Report CLS assignments separately
so CLS cannot hide a dead patch expert. Also report:

- Soft mean router probabilities and mean per-patch probability entropy, normalized by log(E).
- Entropy of the hard load distribution, minimum expert load, and maximum/minimum utilization.
- Strict dead-expert count: zero assignments across the complete fixed query panel. Additionally flag near-starvation
    below 1% as a descriptive threshold, with raw counts and sample size visible.
- Accumulated actual training assignments in the final 200 updates, including visible patches and CLS separately.
    Training router noise can conceal deterministic inference starvation; inspect both.
- A separate fixed masked-query readout matching MAE's visible-token inputs. Ordinary `routing()` is unmasked, so obtain
    this with a temporary forward observer and reproducible masks under no-grad. Do not silently equate the full-image
    routing distribution with training-time routing.
- Pairwise expert-output CKA on the same saved block-input tokens, plus weight cosine similarity. Evaluate every expert
    on that shared diagnostic token panel irrespective of its assignments. This is descriptive redundancy analysis, not
    a calibrated health signal.

No expert may be strictly dead at the candidate well on the full-image patch probe, and no expert may receive zero patch
assignments over the final training window. Report masked-query starvation separately. A pass on this small panel does
not prove every BDD regime uses every expert.

### Cost and statistical comparison

Report training-only synchronized wall time, seconds/update, images/second, peak memory, checkpoint size, and evaluation
time separately. For asynchronous devices synchronize around timed regions. Plot quality against optimizer steps, images
seen, and wall time. Sparse dispatch overhead can erase the expected compute advantage at this model size; measured cost
is the result.

For each MoE-versus-dense comparison report all five paired seed deltas, mean, standard deviation, and a descriptive 95%
interval over seed-level outcomes. Do not treat checkpoint rows or mask draws as independent training replicates. Plot
per-seed curves, not only pooled curves.

Proposed parity targets, to lock before measured runs: canary accuracy no more than 3 percentage points below either
dense reference; held-out reconstruction MSE no more than 5% worse; finite, nondegenerate geometry and no dead experts.
These are experiment-specific practical tolerances, not thresholds supplied by WP-M0. Use them with seed uncertainty: an
interval crossing a parity boundary makes the result inconclusive. Report geometry differences without assuming every
metric has a simple better/worse direction. The stock Tier-B sanity floor is not a parity test.

## 7. Artifacts and acceptance

Each arm/seed should contain:

- `config.yaml`, code/environment provenance, data membership manifest, and checkpoint hashes.
- `warm_summary.json`, `losses.csv`, cost summary, step checkpoints, and selected-well metadata.
- A warm health table and per-layer routing columns in the existing `health.csv`/`trust.json` style; new routing metrics
    remain `candidate`. Dense routing columns are absent/null, not fabricated zeros.
- `routing_detail.csv` with per-expert counts, probabilities, entropy, and similarity readouts.
- A cross-arm summary with paired seed deltas and figures for loss, canary/geometry, load, and cost.

Reload the complete MAE method strictly, including decoder, and confirm deterministic embeddings and reconstruction on
fixed masks reproduce the saved candidate. `save_well()` saves model weights, not optimizer/RNG state; save the latter
separately if training continuation is required.

The warm script does not emit deployment validation tiers or a deploy corpus. A stationary warm trajectory cannot by
itself satisfy WP-M0's existing `tier_a_passed` / `tier_b_passed` manifest requirement. After candidate-well selection,
run one seed-0, accept-all BDD deployment smoke using the exact same train pool, reservation, architecture, and selected
whole-method well. Add the minimal routing reader to the monitor/corpus and check both tiers in the resulting manifest.
Do this for the three primary arms so the sanity comparison is matched. This is an integration check; it does not enter
the warm quality comparison and does not establish forgetting protection.

The planned milestone is a five-seed native MAE warm-well comparison with both dense controls, no-dead-expert checks,
reproducible artifacts, and a one-seed harness acceptance check. Full WP-M0 remains incomplete until the deferred
balancing modes, upcycling comparison, and other required corpora are implemented and assessed. Cross-view consistency
remains a separate JE extension.
