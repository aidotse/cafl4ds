# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repo.

## What this project is

- **`cafl4ds`** — **Continuous, Active Federated Learning for Data Streams**. Research on a *coupled active-learning
    loop*: a label-free streaming active-learning filter over a self-supervised (SSL) backbone trained via federated
    learning on transient data streams, with a representation-health monitor that re-aims selection. Full overview:
    `docs/project-plan/index.md`.

- **`docs/project-plan/` is the spec** — split hierarchically for targeted agent reads. Start at
    `docs/project-plan/index.md`: the always-read core plus a routing table to the detail files. Read the index only
    when introducing new architectural modules, changing core pipelines, or when explicitly requested.

- **`docs/experiments/` is the implementation state** - also split hierarchically for targeted agent reads. To find
    implementation states, always read `docs/experiments/latest-results/index.md` first. Use its routing table to find
    the exact substudy file you need. Never read an entire directory at once; read only the sub-studies required for the
    task. Additionally, the `docs/experiments/` directory has more specific docs for each project phase. When working in
    a specific phase, always read the `index.md` of that phase (e.g. `docs/experiments/phase0/index.md`), and use the
    routing table in that file to find and read only the relevant sub-study doc for the required task.

## Environment (uv)

- **`uv`**-managed, Python `>=3.10` (`.venv` is 3.12). `uv sync --group dev` to set up; prefix everything with `uv run`.
- Versioning is git-tag-based via `hatch-vcs`.
- For experiments sync `uv` with `--extra cpu` (laptops/CI) or `--extra cu124` (NVIDIA GPU).
- Read `docs/developing.md` ONLY if explicitly asked to run on Gaudi hardware or by user request.

### Commands

Tasks are **`poe`** tasks — `uv run poe <task>`; Run `uv run poe --help` to see available tasks; do not read the full
`pyproject.toml` to view tasks or configs. If you need this information, ask the user.

## Writing code

### Quality gates

- Config for ruff / mypy / pytest is all in `pyproject.toml` (don't read to view tasks or configs; if you need this
    information, ask the user).
- Canonical runner is **pre-commit** (`uv run pre-commit install` once). To check your work, run
    `uv run pre-commit run --files <modified_file>` to avoid terminal flooding from unrelated files. Full hook list is
    in `.pre-commit-config.yaml`.
- When running tests, execute pytest only on the specific test file or directory you are working on to avoid terminal
    flooding.

### Experiments

- **Layout**: package → `cafl4ds/`, entry points → `scripts/`, tests → `tests/unit/`. Only `cafl4ds` ships in the wheel.
- **Hydra-driven, instantiation-based.** Entry points use Hydra reading from `cafl4ds/configs/`. Do not read the entire
    config directory. Use `rg` to find specific `_target_` instantiations in `cafl4ds/configs/`, but target specific
    subfolders (e.g., `configs/model/`) or limit output (e.g., `rg -m 5`) to avoid context flooding. Build objects with
    `hydra.utils.instantiate` — plain classes wired via config (`_target_: ...`), not hardcoded. Override on the CLI as
    per Hydra docs.
- Logging via **`loguru`**.

## Claude Code in this repo

Runs behind a token-compression pipeline (RTK shell-hook + Headroom API proxy): bash output may be RTK-filtered.

## Documenting experiments

You can interpret this as a guide for how to add documentation on new studies or sub-studies. See notes above on
`docs/experiments`. When writing experiment summaries, rely strictly on the active conversation history and terminal
outputs. Do not run `git diff` or read raw source files to figure out what was achieved. If starting a new session with
no history, ask the user to provide the summary points. There are different levels of granularity at play:

- `docs/experiments/latest-results/<substudy ID>.md` - a *concise* summary of the results / what was achieved in the
    substudy. Avoid too many details; this is a a brief take-away from the study.

- `docs/experiments/phase<ID>/index.md` - summary details of what the *project phase* is about, plus a routing table to
    the substudies. In this table: only the *motivation*, not the *results* of the substudy, kept brief.

- `docs/experiments/phase<ID>/<substudy ID>.md` - a detailed, *but not overly verbose or repetetive* description of the
    sub study, including outputs, quantitative results. Includes (roughly) a brief section for the motivation, the
    methodology and findings, any interesting insights, and how to run the experiments in the substudy. Do not mention
    progress or status here, that goes in `docs/experiments/latest-results/<substudy ID>.md`.

## Mandates

- Ensure your updates do not contradict the specific index or substudy files you have been instructed to read. Do not
    read additional documentation files solely to check for global project consistency unless explicitly requested.
