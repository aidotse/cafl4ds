# Phase 0 — STL10 Pilot

**Goal (from the [project plan](../../project-plan.md)):** build the streaming loop + the measurement apparatus
(effective rank / RankMe, probe-on-past, frozen-backbone baseline **B5**) and a **positive control** known to collapse,
on STL-10. **Exit gate:** instruments validated **and the positive control collapses on RankMe** — the instruments must
be trustworthy before any Phase-1 conclusion.

This is the landing page for Phase 0: it tracks the sub-studies and always points at the latest results. Each row below
links to its own page as it lands.

## Sub-studies & status

| ID   | Sub-study                                        | What it establishes                                                                                                                                                                                                                                                       | Status                  |
| ---- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| P0.1 | [Instrument & the streaming loop](instrument.md) | The instruments (health metrics / measurements) can be queried. The loop + class-blocked STL-10 stream + both SSL backbones (`C`) + both inits (`I`) run end-to-end under the B-floor knob, logging SSL loss and health side by side — **on both CPU and the Gaudi HPU**. | ✅ **CPU + HPU (demo)** |
| P0.2 | Positive control (PC).                           | A configuration known to collapse, to prove the instruments *catch* collapse. PC collapses on RankMe, by removing anti-collapse mechanism in SimSiam. Comparison: runs with and without anti-collapse mechanisms.                                                         | ⬜ Not started          |
| P0.3 | Phase-0 Gate                                     | `A` flag knobs: reservoir sampling, dedup, loss-gate, etc. (non optimized). `B5` baseline.                                                                                                                                                                                | ⬜ Not started          |

## Latest results

### P0.1

From the [instrument sub-study](instrument.md) — a short CPU run of the full `C × I` matrix (MAE / SimSiam ×
from-scratch / pretrained) on class-blocked STL-10 completes, and each run log shows the SSL loss and the health metrics
(RankMe, drift, probe) **side by side over steps**. Under the correlated stream, **RankMe falls and representation drift
rises in every configuration** — the instruments demonstrably respond to a correlated single-pass diet. These are
harness-validation numbers (tiny model, ~30 steps), **not** a degradation claim: that is Phase 1's job, gated by the PC.
See the sub-study page for the per-config tables.

The same matrix has since been **reproduced on a Gaudi 2 HPU** (single card, same settings and seed) as an
infrastructure milestone: the loop is device-portable, the instruments produce finite (non-NaN) values, and the numbers
move reasonably — `from-scratch` (bit-identical init) tracks CPU closely, `pretrained` shifts with its HPU-regenerated
warm start. Details in the sub-study's [Porting to the HPU](instrument.md#porting-to-the-hpu-gaudi-2-demonstrator).

!!! note "The Phase-0 gate is not yet closed"

    The instruments are validated in isolation (`tests/unit/test_measurements.py`) and now shown to be *drivable* over a
    live stream, but the positive control — and therefore the formal exit gate — is still pending.
