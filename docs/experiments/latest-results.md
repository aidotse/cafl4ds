# Latest results

## P0.2.1

From the [healthy-baseline sub-study](phase0/positive_control_2.md) — a genuine healthy SimSiam baseline and a gate
recalibrated around it (all numbers, sweeps, and trajectories are in the sub-study). Three findings:

1. **A genuine healthy baseline exists, in the clean long-horizon corner.** The fix P0.2 needed was a *fair training
    regime, not a bigger model* (same tiny ViT; full STL-10 split, IID, 40 epochs, warmup+cosine LR, predictor width
    64→128). Intact SimSiam's RankMe dips then re-expands and holds (~5.5, the healthy U-shape) while the collapse arm
    sinks to the ~1.9 BatchNorm floor; the discriminator flips from loss to **RankMe**, which now separates the arms
    ~2.8× and the gate **PASSES** (CPU + Gaudi HPU). This licenses reading Phase-1 RankMe drops as degradation.
1. **The baseline does not generalize toward the target regime.** Across the 2×2 {IID, class-blocked} × {40 epochs,
    single pass}, only IID × 40 epochs passes: correlated ordering *suppresses* the healthy arm (contrast closes from
    above), and a single pass leaves the *collapse control under-developed* (from below). An LR sweep (annealed + flat)
    rules out learning rate/schedule as the missing lever — the single-pass contrast is structural (~1.3× at best vs
    the 2× bar). So the **40-epoch IID corner is the fixed calibration reference Phase 1 reads against**, and
    correlation's penalty is a multi-epoch effect (ordering barely matters at single pass).
1. **40 epochs is the cheapest *robust* reference and a plateau.** A horizon sweep (5→80 epochs, robustness = min across
    seeds) shows the contrast climbs past 40 (not a local optimum) and is untrustworthy below it — 25–30 straddle the
    2× line within seed noise — while 40 is the first horizon whose worst-of-4-seeds (2.47×) clears 2× comfortably. The
    healthy signal saturates by ~40 epochs; no cheaper reference exists.

## P0.2

From the [positive-control sub-study](phase0/positive_control_1.md) — Goal: calibrate the "collapse" instruments only;
the *forgetting* mode (MAE / probe-on-past) needs its own positive control, later. In **one session**, predictorless /
stop-gradient-off SimSiam (the `anti_collapse=false` toggle) is run side by side with intact SimSiam, both from-scratch
on class-blocked STL-10. Tthe PC drives its loss to the −1 constant-solution floor (`−0.96`) while RankMe craters to
~32% of its random-init value, but values for the control aren't clearly better. Loss seems more of a collapse
discriminator than RankMe - the latter doesn't cleanly read collapse. Conclusion: we can't tell whether the healthy
arm's RankMe decay is (a) SimSiam's general single-pass dynamics — in which case it's a real property that won't vanish
with scale and the baseline is fundamentally compromised — or (b) an artifact of the toy setup (tiny model, 30 steps,
single pass, no schedule) — in which case it disappears as we scale and the baseline is fine. This motivates P0.2.1

## P0.1

From the [instrument sub-study](phase0/instrument.md) — a short CPU run of the full `C × I` matrix (MAE / SimSiam ×
from-scratch / pretrained) on class-blocked STL-10 completes, and each run log shows the SSL loss and the health metrics
(RankMe, drift, probe) side by side over steps. Under the correlated stream, RankMe falls and representation drift rises
in every configuration — the instruments demonstrably respond to a correlated single-pass diet. These are
harness-validation numbers (tiny model, ~30 steps), not a degradation claim: that is Phase 1's job, gated by the PC. See
the sub-study page for the per-config tables.

The same matrix has since been reproduced on a Gaudi 2 HPU (single card, same settings and seed) as an infrastructure
milestone: the loop is device-portable, the instruments produce finite (non-NaN) values, and the numbers move reasonably
— `from-scratch` (bit-identical init) tracks CPU closely, `pretrained` shifts with its HPU-regenerated warm start.
Details in the sub-study's [Porting to the HPU](instrument.md#porting-to-the-hpu-gaudi-2-demonstrator).
