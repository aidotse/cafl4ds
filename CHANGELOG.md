# Changelog

All notable changes to this project are documented in this file.

This file is managed by [Commitizen](https://commitizen-tools.github.io/commitizen/) and is updated automatically on
each `poe bump` (driven by [Conventional Commits](https://www.conventionalcommits.org/)). Do not edit it by hand.

## v0.34.0 (2026-11-13)

### Feat

- Phase 1 findings deck, covering all of Phase 1 with its audit state

## v0.33.0 (2026-11-13)

### Feat

- promote the Phase 0 traces the deck reads into the P0 artifacts
- Phase 0 findings slide deck, published with the docs

### Fix

- give the P0 deck's data script its own module name for mypy

## v0.32.0 (2026-11-12)

### Feat

- manual "Publish site" workflow; docs nav covers every page
- publish slide decks with the docs site; P1.1 / P1.2 deck

## v0.31.0 (2026-11-12)

### Feat

- P1.2.5 — DINO's objective, P1.2.5's tasks and analyses (Parts A-G), data on the project share
- P1.2.4 attribution verdict and the p124 analysis (per-control attribution, confirmed recipe vs the best floor)
- COCO and Places365 off-domain sources for P1.2.4's attribution controls
- P1.2.2/P1.2.4 recipe levers and arm-choosing reads
- vit_b16_dino encoder config (P1.2.4's stronger frozen floor)
- analyze_adaptation p121 secondary reads - walk vs IID, per stratum, by regime relation
- read_adaptation / analyze_adaptation align subset tasks to their own query frames
- P1.2.0 extended screen - weather and cell-balanced presence tasks, scene whitelist, center-crop
- BDD100KSource cache_dir - reuse a deterministic decode across runs
- P1.2.1 engineering - uint8 BDD frames, chosen-step checkpoints, gate verdict and analysis
- read_adaptation fewshot.tasks restricts the fine-tune rungs to chosen tasks
- P1.2.0 readout ladder, adapted-vs-B5 bootstrap, settled-head warm-up

### Fix

- **ci**: declare threadpoolctl; seed test_p125's fake ladders by name, not temp path
- P1.2.5 Part G results; DINO fine-tunes train the encoder; P1.2 closed (Qualified)

## v0.30.0 (2026-11-12)

### Feat

- P1.1.3 middle-block single-drive read (BDD and P0.3.9)
- P1.1.3 CKA drift reads, save_states for the forgetting harness
- P1.1.2 Stage 1 - label-free reader screen against the damage branches
- P1.1.1 reading protocol - single-trajectory rule, drive branching, damage and collapse-dial reads
- P1.1.0 ViT-B warm scout, two-sided competence stop, deep-well recipe
- P1.1.0 competence study - supervised ceiling, pretrained and BDD-train wells
- P1.1.0 selection set, competence-stopped warm-up, competence scout
- P1.1.0 shallow-well null + familiarity sweep, deep-well handover

### Fix

- cap CPU threads in the P1.1.3 forgetting-lens reader
- P1.1.0 harness review fixes before the deep-well runs

## v0.29.0 (2026-09-16)

### Feat

- P1.0 audit remediation done
- P1.0 audit, pre-remediation
- P1.0 finished, pre-audit
- P1.0.{0,1} done, working on prototype deployment harness P1.0.2

### Fix

- deptry fixes

## v0.28.0 (2026-09-10)

### Feat

- P1 scoping done, re-audit prior art, all synced. Ready for studies

## v0.27.0 (2026-09-09)

### Feat

- finished P0 wrapup
- docs internal consistency check; index.md refactor

## v0.26.0 (2026-09-08)

### Feat

- finish and audit P0.3.10

## v0.25.0 (2026-09-02)

### Feat

- update mkdocs
- P0.6 audit finished

## v0.24.0 (2026-08-31)

### Feat

- refactor phase0/index.md

## v0.23.0 (2026-08-31)

### Feat

- update mkdocs up to P0.6

## v0.22.0 (2026-08-31)

### Feat

- P0.6 review
- P0.6.{0,1} done, pre-audit

### Fix

- docs merge

## v0.21.0 (2026-08-28)

### Feat

- P0.5 representation-degradation — reframe, dose-down, audit, hand-off

## v0.20.0 (2026-08-27)

### Feat

- final P0.4 remediations

## v0.19.0 (2026-08-25)

### Feat

- P0.4 remediation done
- update gitignore
- P0.5 representation-degradation mode — harness, readers, docs, artifacts

## v0.18.0 (2026-08-18)

### Feat

- update phase0 index

## v0.17.0 (2026-08-17)

### Feat

- update mkdocs

## v0.16.0 (2026-08-17)

### Feat

- P0.3 finished

### Fix

- canonicalize non-finite JSON to null

## v0.15.0 (2026-08-15)

### Feat

- P0.3 audit fixes

## v0.14.0 (2026-08-14)

### Feat

- P0.2 audit fixed

## v0.13.0 (2026-08-12)

### Feat

- P0.4 audit done
- P0.4.0 review
- P0.4, still needs review

### Fix

- P0.4 docs review

## v0.12.0 (2026-08-11)

### Feat

- audits

## v0.11.0 (2026-08-10)

### Feat

- P0.3 forgetting-gate calibration — code, results, and docs

### Fix

- update CLAUDE.md

## v0.10.0 (2026-07-30)

### Feat

- P0.2.4 complete
- P0.2.4 false-positive stress on the quiet side

## v0.9.0 (2026-07-29)

### Feat

- P0.2.3 redundancy-collapse vehicle + gate — close offdiag_cov fire side

## v0.8.0 (2026-07-29)

### Feat

- P0.2.2 E1-E6

### Fix

- P0.2.2.md E5 -> P0.2.3.md

## v0.7.0 (2026-07-17)

### Feat

- P0.2.1 wrap up, docs refactor

## v0.6.0 (2026-07-14)

### Feat

- **phase0**: P0.2.1 healthy-baseline calibration, filter/eval blocks, project-plan doc split

### Fix

- **docs**: repo-wide mdformat --compact-tables compliance; carry doc edits

## v0.5.1 (2026-07-13)

### Fix

- docs

## v0.5.0 (2026-07-13)

### Feat

- P0.2, P0.2.1

## v0.4.0 (2026-07-11)

### Feat

- phase 0 loop running on HPU

## v0.3.0 (2026-07-10)

### Feat

- add Phase-0 streaming SSL loop (B-floor) with health logging

## v0.2.0 (2026-07-10)

### Feat

- add phase 0 measurements and correct pytorch dependencies

## v0.1.2 (2026-07-10)

### Fix

- add CLAUDE&Gaudi support
- add CLAUDE&Gaudi support
- add claude + Gaudi support

## v0.1.1 (2026-07-09)

### Fix

- push docs

## v0.1.0 (2026-07-09)

### Feat

- initial generation from PyMaxQ
