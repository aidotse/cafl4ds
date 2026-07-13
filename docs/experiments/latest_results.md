# Latest results

## P0.2

From the [positive-control sub-study](positive_control.md) — Goal: calibrate the "collapse" instruments only; the
*forgetting* mode (MAE / probe-on-past) needs its own positive control, later. In **one session**, predictorless /
stop-gradient-off SimSiam (the `anti_collapse=false` toggle) is run side by side with intact SimSiam, both from-scratch
on class-blocked STL-10. Tthe PC drives its loss to the −1 constant-solution floor (`−0.96`) while RankMe craters to
~32% of its random-init value, but values for the control aren't clearly better. Loss seems more of a collapse
discriminator than RankMe - the latter doesn't cleanly read collapse. Conclusion: we can't tell whether the healthy
arm's RankMe decay is (a) SimSiam's general single-pass dynamics — in which case it's a real property that won't vanish
with scale and the baseline is fundamentally compromised — or (b) an artifact of the toy setup (tiny model, 30 steps,
single pass, no schedule) — in which case it disappears as we scale and the baseline is fine. This motivates P0.2.1

## P0.1

From the [instrument sub-study](instrument.md) — a short CPU run of the full `C × I` matrix (MAE / SimSiam ×
from-scratch / pretrained) on class-blocked STL-10 completes, and each run log shows the SSL loss and the health metrics
(RankMe, drift, probe) side by side over steps. Under the correlated stream, RankMe falls and representation drift rises
in every configuration — the instruments demonstrably respond to a correlated single-pass diet. These are
harness-validation numbers (tiny model, ~30 steps), not a degradation claim: that is Phase 1's job, gated by the PC. See
the sub-study page for the per-config tables.

The same matrix has since been reproduced on a Gaudi 2 HPU (single card, same settings and seed) as an infrastructure
milestone: the loop is device-portable, the instruments produce finite (non-NaN) values, and the numbers move reasonably
— `from-scratch` (bit-identical init) tracks CPU closely, `pretrained` shifts with its HPU-regenerated warm start.
Details in the sub-study's [Porting to the HPU](instrument.md#porting-to-the-hpu-gaudi-2-demonstrator).
