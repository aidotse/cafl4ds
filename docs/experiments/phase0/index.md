# Phase 0 — STL10 Pilot

**Goal (from the [project plan](../../project-plan/index.md)):** build the streaming loop + the measurement apparatus
(effective rank / RankMe, probe-on-past, frozen-backbone baseline **B5**) and a **positive control** known to collapse,
on STL-10. **Exit gate:** instruments validated **and the positive control collapses on RankMe (or similar)** — the
instruments must be trustworthy before any Phase-1 conclusion.

This is the landing page for Phase 0: it tracks the sub-studies and always points at the latest results.

## Sub-studies & status

| ID | Sub-study | What it establishes | Status |
| -- | -- | -- | -- |
| P0.1 | [Instrument & the streaming loop](instrument.md) | The instruments (health metrics / measurements) can be queried, and respond to a single-pass diet. The loop + class-blocked STL-10 stream + both SSL backbones (`C`) + both inits (`I`) run end-to-end under the B-floor knob, logging SSL loss and health side by side — on both CPU and the Gaudi HPU. | ✅ **Complete** |
| P0.2 | [Positive control (collapse gate)](positive_control_1.md) | Motivation: calibration for the *collapse* failure mode only. A configuration known to collapse, to prove the instruments *catch* collapse. Predictorless / stop-gradient-off SimSiam (one toggle) forces collapse; side-by-side vs. intact SimSiam, from-scratch, one session. PC loss rides to the −1 floor while RankMe craters, but also for the control, so we have no healthy baseline yet. Unclear if SimSiam dynamics in general (single-pass), or artifact because of small experiment size. | ✅ **Complete** |
| P0.2.1 | [Healthy baseline + recalibrated gate](positive_control_2.md) | **Motivation:** P0.2 could not tell a healthy representation from a collapsed one (RankMe did not separate the arms), so the collapse gate had no trustworthy upper reference. P0.2.1 (a) finds a regime where intact SimSiam demonstrably does *not* collapse — a genuine healthy baseline — and recalibrates the gate around it, then (b) tests how far that baseline reaches — via a 2×2 {ordering × horizon} matrix toward the project's target regime (correlated, single-pass), stress-tested with an LR sweep (annealed and flat) to rule out learning rate and schedule as the missing lever, and a horizon sweep (5→80 epochs, 4 seeds at the crossover) that fixes 40 epochs as the cheapest *robust* reference and a stable plateau. | ✅ **Complete** |
| P0.3 | [`A` flag knobs](po_knobs.md) | Cheap "knobs": Reservoir sampling, dedup, loss-gate, etc. (non optimized). `B5` baseline. | ⬜ Not started |
