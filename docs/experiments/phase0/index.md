# Phase 0 — Instrument

**Goal (from the [project plan](../../project-plan.md)):** build the streaming loop + the measurement apparatus
(effective rank / RankMe, probe-on-past, frozen-backbone baseline **B5**) and a **positive control** known to collapse,
on STL-10. **Exit gate:** instruments validated **and the positive control collapses on RankMe** — the instruments must
be trustworthy before any Phase-1 conclusion.

This is the landing page for Phase 0: it tracks the sub-studies and always points at the latest results. Each row below
links to its own page as it lands.

## Sub-studies & status

| Sub-study                                              | What it establishes                                                                                                                                                   | Status              |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| [Instrument — the streaming loop (CPU)](instrument.md) | The loop + class-blocked STL-10 stream + both SSL backbones (`C`) + both inits (`I`) run end-to-end under the B-floor knob, logging SSL loss and health side by side. | ✅ **CPU complete** |
| Instrument — HPU scale                                 | Same loop at a realistic model/dataset size on the Gaudi HPUs.                                                                                                        | ⬜ Not started      |
| Positive control (PC)                                  | A configuration known to collapse, to prove the instruments *catch* collapse.                                                                                         | ⬜ Not started      |
| Phase-0 gate                                           | The formal exit: **PC collapses on RankMe**.                                                                                                                          | ⬜ Not started      |

## Latest results

From the [instrument sub-study](instrument.md) — a short CPU run of the full `C × I` matrix (MAE / SimSiam ×
from-scratch / pretrained) on class-blocked STL-10 completes, and each run log shows the SSL loss and the health metrics
(RankMe, drift, probe) **side by side over steps**. Under the correlated stream, **RankMe falls and representation drift
rises in every configuration** — the instruments demonstrably respond to a correlated single-pass diet. These are
harness-validation numbers (tiny model, ~30 steps), **not** a degradation claim: that is Phase 1's job, gated by the PC.
See the sub-study page for the per-config tables.

!!! note "The Phase-0 gate is not yet closed"

    The instruments are validated in isolation (`tests/unit/test_measurements.py`) and now shown to be *drivable* over a
    live stream, but the positive control — and therefore the formal exit gate — is still pending.
