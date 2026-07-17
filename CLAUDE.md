# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repo.

## What this project is

- **`cafl4ds`** — **Continuous, Active Federated Learning for Data Streams**. Research on a *coupled active-learning
    loop*: a label-free streaming active-learning filter over a self-supervised (SSL) backbone trained via federated
    learning on transient data streams, with a representation-health monitor that re-aims selection. Full overview:
    `docs/project-plan/index.md`.

- **`docs/project-plan/` is the spec** — split hierarchically for targeted agent reads. Start at
    `docs/project-plan/index.md`: the always-read core plus a routing table to the detail files. Read the index before
    designing anything non-trivial, then open only the detail file the task needs.

- **`docs/experiments/` is the implementation state** - also split hierarchically for targeted agent reads. Start at
    `docs/experiments/latest-results/` The files in this directory correspond to individual sub-studies with the
    following convention "`P<Phase ID>.<Substudy ID>`, where `<Substudy ID>` is also hierarchical (e.g. `3.1` is a
    sub-sub study of sub-study `3`). Read only the sub-studies required for the task. The `docs/experiments/` directory
    also has more specific docs for each project phase. When working in a specific phase, always read the `index.md` of
    that phase (e.g. `docs/experiments/phase0/index.md`), which includes a routing table to the substudies of that
    phase, also in the same directory. Only read the relevant sub-study doc for the required task.

## Environment (uv)

**`uv`**-managed, Python `>=3.10` (`.venv` is 3.12). `uv sync --group dev` to set up; prefix everything with `uv run`.
Versioning is git-tag-based via `hatch-vcs`. For experiments sync `uv` with `--extra cpu` (laptops/CI) or
`--extra cu124` (NVIDIA GPU). For more complex environment setup (e.g. on Gaudi) read: `docs/developing.md`.

### Commands

Tasks are **`poe`** tasks — `uv run poe <task>`; the full list + definitions live in `[tool.poe.tasks]` in
`pyproject.toml`.

## Writing code

### Quality gates

Config for ruff / mypy / pytest is all in `pyproject.toml`. Canonical runner is **pre-commit**
(`uv run pre-commit install` once, then `run --all-files`); hook list is in `.pre-commit-config.yaml`.

### Experiments

- **Layout**: package → `cafl4ds/`, entry points → `scripts/`, tests → `tests/unit/`. Only `cafl4ds` ships in the wheel.
- **Hydra-driven, instantiation-based.** Entry points are `@hydra.main` scripts in `scripts/` reading `cafl4ds/configs/`
    (root `main.yaml`). Build objects with `hydra.utils.instantiate` — plain classes wired via config (`_target_: ...`),
    not hardcoded. Override on the CLI as per Hydra docs.
- Logging via **`loguru`**.

## Claude Code in this repo

Runs behind a token-compression pipeline (RTK shell-hook + Headroom API proxy): bash output may be RTK-filtered.

## Documenting experiments

You can interpret this as a guide for how to add documentation on new studies or sub-studies. See notes above on
`docs/experiments`. There are different levels of granularity at play:

- `docs/experiments/latest-results/<substudy ID>.md` - a *concise* summary of the results / what was achieved in the
    substudy. Avoid too many details; this is a a brief take-away from the study.

- `docs/experiments/phase<ID>/index.md` - summary details of what the *project phase* is about, plus a routing table to
    the substudies. In this table: only the *motivation*, not the *results* of the substudy, kept brief.

- `docs/experiments/phase<ID>/<substudy ID>.md` - a detailed, *but not overly verbose or repetetive* description of the
    sub study, including outputs, quantitative results. Includes (roughly) a brief section for the motivation, the
    methodology and findings, any interesting insights, and how to run the experiments in the substudy. Do not mention
    progress or status here, that goes in `docs/experiments/latest-results/latest-results/`.

## Mandates

- The documentation *must* be internally consistent, contradiction-free, and kept up to date and synchronized with the
    experiments. Immediately flag any inconsistencies you find.
