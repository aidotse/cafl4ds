# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repo.

## What this project is

`cafl4ds` — **Continuous, Active Federated Learning for Data Streams**. Research on a *coupled active-learning loop*: a
label-free streaming active-learning filter over a self-supervised (SSL) backbone trained via federated learning on
transient data streams, with a representation-health monitor that re-aims selection.

**Phase 0 (Instrument) underway.** The verified measurement apparatus is `cafl4ds/measurements.py` — standalone health
instruments (RankMe/`effective_rank`, VICReg variance + off-diag covariance, alignment/uniformity, linear-CKA
`cka_drift` + rotation-sensitive `cosine_drift`, kNN/linear probes, frozen baseline **B5**), depending only on embedding
tensors or a passed-in `encoder` callable — no stream/model/training. **Labels enter the probes ONLY.** Known-answer
tests: `tests/unit/test_measurements.py`. Everything else is still PyMaxQ scaffold (`cafl4ds/pipeline.py` +
`scripts/example.py` are throwaway Hydra examples). The architecture is the *research design*:

- **`docs/project-plan.md` is the spec** — streaming loop, filter families (novelty F-a, coverage F-b, steerable F-c),
    SSL backbones, health metrics (RankMe, drift, forgetting), FL track, baselines, phased plan. Read it before
    designing anything non-trivial. Novelty claims are tagged `N-A`…`N-G`; new work maps to one or it is scope drift.

## Environment (uv)

**`uv`**-managed, Python `>=3.10` (`.venv` is 3.12). `uv sync --group dev` to set up; prefix everything with `uv run`.
Private GitLab registry (optional): `source .env` first, where `.env` (git-ignored) holds
`UV_INDEX_<PACKAGE>_USERNAME`/`_PASSWORD` (GitLab PAT, `read`). Versioning is git-tag-based via `hatch-vcs` (fallback
`0.0.0`). **`torch` build is hardware-selected via a mutually-exclusive extra** — sync with `--extra cpu` (laptops/CI)
or `--extra cu124` (NVIDIA GPU); a bare `uv sync` installs no torch. Gaudi uses neither (Habana base image provides
torch; `docker/gaudi.env.Dockerfile` strips it from the export).

## Commands

Tasks are **`poe`** tasks — `uv run poe <task>`; the full list + definitions live in `[tool.poe.tasks]` in
`pyproject.toml`. Notes:

- `uv run poe test [cpus] [paths...]` runs pytest + coverage; run it **before `poe docs`** (docs embed coverage/test
    badges). pytest config (always-on coverage, `docs/exported/pytest.{html,xml}`, DEBUG logging) is in
    `[tool.pytest.ini_options]`.
- Single test: `uv run pytest test/unit/test_dummy.py::test_dummy`.

## Quality gates

Config for ruff / mypy / pytest is all in `pyproject.toml` (line length **120**, strict ruff incl. bandit `S`, Google
docstrings, `docs/` excluded). Canonical runner is **pre-commit** (`uv run pre-commit install` once, then
`run --all-files`); hook list is in `.pre-commit-config.yaml`.

- **`no-commit-to-branch` blocks commits to the default branch — always work on a feature branch.**
- `check-added-large-files` caps additions at 1 MB.

## How code is written here

- **Hydra-driven, instantiation-based.** Entry points are `@hydra.main` scripts (`scripts/example.py`) reading
    `cafl4ds/configs/` (root `main.yaml`). Build objects with `hydra.utils.instantiate` — plain classes wired via config
    (`_target_: ...`), not hardcoded. Override on the CLI: `uv run python scripts/example.py key=value`. Run dirs →
    `outputs/` (ignored).
- Logging via **`loguru`**.
- Layout: package → `cafl4ds/`, entry points → `scripts/`, tests → `test/unit/`. Only `cafl4ds` ships in the wheel.

## Docker

- `docker/Dockerfile` — production image; **requires `--build-arg VERSION=...`** (no `.git` in context). Private deps +
    CA certs via build secrets; see the file's header comment.
- `docker/gaudi.env.Dockerfile` — Gaudi HPU dev environment (default image is `gaudi-env-cafl4ds:latest`). Run things on
    the HPUs with `scripts/run_gaudi_dev.sh <image> <device_id|all> [cmd...]` (smoke test:
    `./scripts/run_gaudi_dev.sh gaudi-env-cafl4ds:latest 0 python scripts/gaudi_simple_test.py`). Only the repo is
    mounted; bind data/models living outside it read-only with the `DATA_MOUNT=<host_path>` env var (e.g.
    `DATA_MOUNT=/mnt/stl10 ./scripts/run_gaudi_dev.sh … python scripts/run_loop.py device=hpu`). Use the container's
    system `python` (torch is from the Habana base image), **not** `uv run`. Host check: `HABANA_LOGS=/tmp/hllog hl-smi`
    (no sudo). Full setup, isolation modes, and RDMA prereqs → `docs/developing.md`.

## Claude Code in this repo

Runs behind a token-compression pipeline (RTK shell-hook + Headroom API proxy): bash output may be RTK-filtered, and
`.claudeignore` excludes `.venv`, `uv.lock`, caches, and generated artifacts from indexing.
