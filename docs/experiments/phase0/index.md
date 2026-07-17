# Phase 0 — Instrument Calibration

This is the landing page for Phase 0: it tracks the sub-studies and their motivation.

**Goal (from the [project plan](../../project-plan/index.md#strategic-ordering--instrument--demonstrate-first)):** build
the streaming loop + the measurement apparatus (RankMe/effective rank, VICReg variance/covariance, alignment/uniformity,
representation drift, per-era probe, frozen-backbone baseline **B5**) and — the real work — **calibrate** it using a
simple dataset (STL10) and model. Phase 0 tests each health metric against the failure mode it is meant to catch, so
that a Phase-1 signal is interpretable once we scale past the toy model and dataset.

**Calibration is a two-sided, per-failure-mode test.** A metric earns trust only if it (a) **fires** under a **positive
control** that deliberately induces its failure mode, *and* (b) **stays quiet** under a **healthy** baseline (no false
alarm). The failure modes are distinct — a collapse run is *not* a known-forgetting event — so each mode (not metric!)
needs its **own** positive control. This is what drives the sub-studies below:

| Failure mode | Positive control (induces it) + healthy reference | Instruments calibrated | Status |
| -- | -- | -- | -- |
| **Collapse** (joint-embedding) | Predictorless / stop-grad-off SimSiam vs. a genuine healthy SimSiam baseline | RankMe/effective rank, VICReg variance + off-diag covariance, alignment/uniformity, prototype entropy | ✅ Collapse arm craters, healthy arm holds, RankMe separates ~2.8× — [P0.2](P0.2.md) + [P0.2.1](P0.2.1.md) |
| **Forgetting** (MAE) | Deliberately-forgetting run: train hard on era A, then only era B; the past-era probe on A must crater | Per-era probe accuracy, Backward Transfer, Forgetting Measure, and — the label-free leading indicator — representation **drift (CKA)** | ⬜ Not started (P0.3) |
| **Instability** (divergence) | Drive LR / batch pathology until training diverges | Gradient norm (minor mode) | ⬜ Not started (P0.4) |

**Exit gate:** every mode calibrated — its PC fires *and* a healthy baseline stays quiet.

## Sub-studies

| ID | Sub-study | What it establishes | Status |
| -- | -- | -- | -- |
| P0.1 | [Instrument & the streaming loop](P0.1.md) | The instruments (health metrics / measurements) can be queried, and respond to a single-pass diet. The loop + class-blocked STL-10 stream + both SSL backbones (`C`) + both inits (`I`) run end-to-end under the B-floor knob, logging SSL loss and health side by side — on both CPU and the Gaudi HPU. | ✅ **Complete** |
| P0.2 | [Positive control (collapse gate)](P0.2.md) | Motivation: calibration for the *collapse* failure mode only. A configuration known to collapse, to prove the instruments *catch* collapse. Derive a baseline: predictorless / stop-gradient-off SimSiam (one toggle) forces collapse; side-by-side vs. intact SimSiam, from-scratch. | ✅ **Complete** |
| P0.2.1 | [Positive control (collapse gate) - RankMe calibration](P0.2.1.md) | Motivation: P0.2 could not tell a healthy representation from a collapsed one (RankMe did not separate the arms), so the collapse gate had no trustworthy upper reference. P0.2.1 sets out to prove whether RankMe can separate a genuinely-collapsed representation from a genuinely-healthy one by (a) finding a regime where intact SimSiam demonstrably does *not* collapse — a genuine healthy baseline. Then: (b) testing how far that baseline reaches toward the project's target regime (correlated, single-pass), stress-tested with an LR sweep (annealed and flat) to rule out learning rate and schedule as the missing lever. We also want to find the the cheapest *robust* reference in terms of training horizon. | ✅ **Complete** |
| P0.2.2 | [Positive control (collapse gate) - other collapse metric calibration](positive_control_3.md) | Motivation: P0.2.1 gives us a healthy baseline and calibration for RankMe, but not for the remaining collapse metrics. Target: find their operating envelope (still on dummy / STL10 data), i.e. where they activate and where they remain quiet. Essentially repeat of P0.2.1 with the full collapse metric set. | ⬜ Not started |
| P0.3 | [Positive control (forgetting gate)](positive_control_forgetting.md) | Motivation: calibration for the *forgetting* failure mode — the collapse PC in P0.2.1 and P0.2.2 says nothing about whether the forgetting detectors work. Work out a baseline calibrate metrics (per-era probe accuracy, Backward Transfer, Forgetting Measure, and — critically — representation drift (CKA), the label-free leading indicator forgetting is supposed to announce itself through). | ⬜ Not started |
| P0.4 | [Positive control (instability gate)](positive_control_instabilitiy.md) | Motivation: calibration for the *instability / divergence* failure mode (minor). Drive LR / batch pathology until training diverges; the **gradient-norm** instrument must fire, and stay quiet on the healthy baseline. | ⬜ Not started |
