# Positive control — the collapse-instrument calibration gate

The second Phase-0 sub-study (P0.2; see the [Phase 0 overview](index.md)).

## What this is, and why it exists (calibration, not science)

The positive control (PC) is **not a formal scientific measurement** — it is an **instrument calibration**. It is the
training-config analog of P0.1's synthetic unit tests (`tests/unit/test_measurements.py`): a run whose outcome is
**known a priori**, used to prove that the collapse instruments (RankMe / effective rank, alignment/uniformity) actually
**register a real collapse in the regime we measure in**. A thermometer is only trustworthy once you have dipped it in
ice water and in boiling water; the PC is that dip.

Until this gate passes, an ambiguous health reading is uninterpretable. (["P0.1"](instrument.md)) showed RankMe
**falling** under the correlated stream in every configuration — but we could not call those drops *collapse* vs.
*healthy structural adaptation*, because we had never confirmed the instrument responds to a genuine collapse at this
scale. Passing this gate is what **licenses that interpretation**: we now know what a real collapse looks like on these
instruments, in this measurement regime, and can read P0.1's (and Phase 1's) drops against it.

## Scope — one positive control per *failure mode*, not per SSL method

This gate calibrates the **collapse-detecting** instruments **only** (RankMe / effective rank, alignment/uniformity),
using the joint-embedding (SimSiam) collapse mode. It says nothing about MAE; its degradation mode is forgetting /
overspecialization, which is validated by a *different* instrument — the probe-on-past / per-era accuracy (Backward
Transfer, Forgetting Measure). That calibration is a separate positive control planned for a later experiment.

## Why predictorless SimSiam is the vehicle (forced collapse, one toggle)

SimSiam avoids representational collapse with **two coupled mechanisms**: a **predictor** MLP on one branch and a
**stop-gradient** on the target branch (Chen & He 2021). Disable *both* and the objective reduces to `−cosine(z₁, z₂)`
with gradients flowing through both branches — whose **trivial global optimum maps every input to one constant vector**
(cosine → 1, loss → −1). Collapse is then **mathematically forced and scale-independent**: there is no non-degenerate
solution the optimizer can prefer. That makes it the ideal calibration vehicle — a *known-answer* config reachable with
a **single toggle** rather than a separate model.

The toggle lives on the SSL method itself (`cafl4ds/ssl/simsiam.py`), exposed through the factory and config as
`anti_collapse`:

| `anti_collapse`  | Predictor             | Stop-gradient | Role                                       | `method.name`      |
| ---------------- | --------------------- | ------------- | ------------------------------------------ | ------------------ |
| `true` (default) | on                    | on            | **healthy control** (SimSiam as published) | `simsiam`          |
| `false`          | **removed** (`p = z`) | **off**       | **positive control** (forced collapse)     | `simsiam_collapse` |

This is the documented SimSiam collapse ablation: the paper shows removing *either* mechanism collapses the model; we
remove both, the strongest collapse config. The mechanism is unit-tested in `tests/unit/test_ssl.py`
(`test_anti_collapse_off_bypasses_predictor_and_renames` — the predictor leaves the gradient graph;
`test_stop_gradient_toggle_controls_target_branch_gradient` — the target branch gradient is gated by the flag).

## The contrast, and the numeric gate

One curve proves nothing: a lone RankMe drop could be the instrument, the model, or the stream. **The separation is the
artifact.** Both arms are run in **one session**, from-scratch (bit-identical init — the seed is reset before each arm,
so the augmentation RNG also stays in lockstep), over the **same** class-blocked STL-10 stream. The only difference is
the `anti_collapse` toggle. From-scratch is deliberate: a pretrained warm start sits in a good basin that can *mask*
collapse (in P0.1 `simsiam_pretrained` already sat at RankMe ~2.46, possibly pre-collapsed — exactly the false-negative
risk to avoid here).

The pass criterion is **numeric, not eyeball** — and its final form was **driven by what the runs actually did** (see
[What the calibration itself taught us](#what-the-calibration-itself-taught-us) — two of the criteria below exist
because the naive version failed). At the bounded (P0.1) horizon the gate passes iff **all** hold:

1. **PC collapses on RankMe (relative).** PC final RankMe ≤ **45%** of its *own* random-init RankMe. (Relative, not an
    absolute floor: this tiny ViT's projector `BatchNorm` keeps deep collapse off literal rank-1, so the RankMe floor
    is ~1.8, and an absolute `<1.5` is unreachable *at this scale*.)
1. **PC "right reason" (the point-collapse fingerprint).** PC loss floor (min over steps) ≤ **−0.9** — the loss rides to
    its **−1 constant-solution floor** *while* RankMe craters. If RankMe fell but the loss did **not** approach −1, the
    gate does **not** pass — that would mean rank moved for some other reason and the instrument/wiring is suspect.
1. **The intact control holds.** Healthy final RankMe ≥ **60%** of its *own* random-init RankMe.

Thresholds live in `cafl4ds/configs/positive_control.yaml`; the gate is evaluated in code
(`scripts/positive_control.py`), which prints `PASS`/`FAIL`, writes `comparison.json`, and exits non-zero on failure.

## Scale

For this pilot, we want to set the calibration at the smallest / simplest setting that meets all of the criteria above.
Model collapse, though forced, may be scale-independent: experiment results for P0.2 below show that our metric is
reduced for both our collapsed model as well as our healthy model, invalidating a valid baseline at the same complexity
settings as P0.1. The P0.2.1 experiment is about finding this baseline.

## Result — (CPU, canonical)

One session, both arms, class-blocked STL-10, `seed=0`, `eval_every=4`, tiny ViT (same as P0.1). `pc_*` = collapse
ablation, `hc_*` = intact SimSiam.

```
        step           era       pc_loss     pc_rankme       hc_loss     hc_rankme
------------  ------------  ------------  ------------  ------------  ------------
      0.0000        0.0000       -0.4631        8.7402        0.0020        8.5770
      4.0000        1.0000       -0.5025        5.9522       -0.0875        6.4949
      8.0000        2.0000       -0.6707        5.6159       -0.2420        6.2143
     12.0000        4.0000       -0.9590        4.6409       -0.3565        6.1596
     16.0000        5.0000       -0.8238        3.9710       -0.4064        5.4683
     20.0000        6.0000       -0.8663        3.4123       -0.5030        4.5586
     24.0000        8.0000       -0.8801        3.2439       -0.5698        4.0792
     28.0000        9.0000       -0.7530        2.9160       -0.5393        3.5613
     29.0000        9.0000       -0.8933        2.7828       -0.6441        3.4480
```

All required criteria pass, except that the control also appears to collapse, despite the higher loss floor value (-0.64
vs. -0.959). Unclear whether this is adaptation or collapse. This motivates P0.2.1.

## Device portability (Gaudi 2 HPU)

Per the P0.1 milestone convention, the identical run reproduces on a Gaudi 2 HPU (`device=hpu`, inside the container).
Here we see no difference in terms of RankMe trajecteories between the collapsed and healthy models, further motivates
P0.2.1. Loss-floor delta remains as in CPU experiment.

```
        step           era       pc_loss     pc_rankme       hc_loss     hc_rankme
------------  ------------  ------------  ------------  ------------  ------------
      0.0000        0.0000       -0.4663        8.0785       -0.0039        8.5987
     12.0000        4.0000       -0.9573        5.1293       -0.3796        6.1278
     20.0000        6.0000       -0.8730        3.5308       -0.4921        4.4455
     29.0000        9.0000       -0.8511        3.1984       -0.6389        3.3647
```

## Insights

1. The SimSiam projector's terminal `BatchNorm(affine=False)` forbids a literally-constant per-batch output, so even
    total point-collapse bottoms out at RankMe ≈ 1.8.

1. The loss fingerprint is the device-robust discriminator; RankMe separation is not resolvable at 30-step toy scale.
    This is a statement about instrument sensitivity, not a passed collapse-detection gate. Need to question the
    30-step window, be careful not to choose this just because this is where the separation looks good, but in a more
    principled way.

## How to run

```bash
# CPU (canonical, deterministic):
uv run python scripts/positive_control.py

# Fast network-free smoke (synthetic source — wiring only; too short/easy to collapse):
uv run python scripts/positive_control.py data=synthetic img_size=16 eval_every=3

# Gaudi HPU (inside the container; see docs/developing.md). torch is from the Habana base
# image, so use the container's system python, and bind STL-10 via DATA_MOUNT:
DATA_MOUNT=/mnt/stl10 ./scripts/run_gaudi_dev.sh gaudi-env-cafl4ds:latest 0 \
    python scripts/positive_control.py device=hpu
```

Each run writes both arms' `run_log.jsonl` and a `comparison.json` (per-checkpoint series + the gate result) to the
Hydra run dir, and prints the side-by-side table, the RankMe/loss sparklines, and the `PASS`/`FAIL` verdict. Gate
thresholds and the bounded horizon are in `cafl4ds/configs/positive_control.yaml`.
