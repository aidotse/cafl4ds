# User Guide

Welcome! This guide walks through the core design principles behind this repo. It is inspired by best practices in the
software engineering community, and is intended to be general enough by making minimal assumptions on *what* you will be
coding, focusing rather on *how* to make this process efficient. Of course there are many tools out there that go
further than what is used here; this repo gathers in essence the key ones, aiming to be as much as infrastructure
agnostic as possible.

The purpose of this repo is to help you design and manage better code faster and better, not to hinder you in the
software engineering process. Keep in mind that the tools here, like anything else, have learning curves and may seem
like hindrances at first, but will undoubtedly show their value once you get an idea of what is going on and why.

## Who is this for?

This guide and repo is meant for anyone that values code quality in their work. If you have a reason to start a new
Python project and want to do this in a principled way, this is a good starting point. This guide is written at a fairly
high level, but with an implicit assumption that you have some basic coding experience already, i.e. know how to use
`git` and have an otherwise high level understanding of how Python projects are structured and work.

## Toolset summary

This is an executive summary of the collection of tools / concepts used in this repository:

- Build system: [pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- Dependency manager: [uv](https://docs.astral.sh/uv/)
- Action runner: [poethepoet](https://github.com/nat-n/poethepoet) — run `uv run poe` to list tasks
- Linting & formatting: [ruff](https://docs.astral.sh/ruff/) (replaces black, isort, and pyupgrade)
- Static type checking: [mypy](https://mypy-lang.org/)
- Unit testing and coverage: [pytest](https://docs.pytest.org/) and
    [coverage](https://coverage.readthedocs.io/en/latest/index.html)
- Multi-version testing: [nox](https://nox.thea.codes/) across Python 3.10–3.13
- Dependency hygiene: [deptry](https://github.com/fpgmaas/deptry) (unused / missing / misplaced deps)
- Security: secret scanning with [gitleaks](https://github.com/gitleaks/gitleaks), `uv audit` for dependency
    vulnerabilities, and [zizmor](https://docs.zizmor.sh/) for GitHub Actions (see [Security](security.md))
- Dependency updates: [Renovate](https://docs.renovatebot.com/) (`renovate.json`)
- Documentation: [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) with API docs via
    [mkdocstrings](https://mkdocstrings.github.io/)
- Gated commits: [pre-commit](https://pre-commit.com/)
- Versioning: [Conventional Commits](https://www.conventionalcommits.org/) driven by
    [commitizen](https://commitizen-tools.github.io/commitizen/), applied to
    [git tags](https://git-scm.com/book/en/v2/Git-Basics-Tagging) (read at build time by hatch-vcs)
- Source code parametrization: [hydra](https://hydra.cc/)

Depending on how you generated the project, CI/CD runs via [GitHub Actions](https://docs.github.com/actions) and/or in
the [GitLab ecosystem](https://docs.gitlab.com/ee/ci/). Documentation is hosted with
[GitHub Pages](https://docs.github.com/pages) and/or [GitLab Pages](https://docs.gitlab.com/ee/user/project/pages/)
accordingly.

## Setup / installation

### 1. Install `uv`

If you don't have `uv` installed locally, install it via curl (macOS/Linux):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

*(For Windows or other methods, see the [official uv docs](https://docs.astral.sh/uv/getting-started/installation/)).*

### 2. Sync the environment

```bash
uv sync
```

This creates the `.venv` and installs the project (editable) plus the `dev` group. Optional agent/AI tooling lives
behind an extra: `uv sync --extra agent`. To reset, delete `.venv` and run `uv sync` again.

### 3. Test your installation

Run the default unit tests with `uv run poe test`, or the example Hydra script with `uv run scripts/example.py`. See all
available tasks with `uv run poe`.

### 4. Install the git hooks

```bash
uv run pre-commit install
```

This installs both the `pre-commit` hooks (see the full list in the pre-commit config) and the `commit-msg` hook that
validates your commit messages (see below).

> **First commit tip.** Run `uv run poe lint` once *before* your initial commit. Auto-fixing hooks (mdformat,
> `end-of-file-fixer`, `trailing-whitespace`, `ruff-format`) normalise the freshly rendered files on their first run and
> report `Failed - files were modified` — that is expected, not an error; just `git add -A` and continue. `poe lint`
> skips the `no-commit-to-branch` guard, so it works even before you branch off `main`. Install the hooks *after* that
> first commit: with them installed, `no-commit-to-branch` would block committing on `main`. From then on, do your work
> on a branch (`git checkout -b feat/initial-setup`) and open a PR — that guard exists to enforce exactly that flow.

### 5. Publishing (optional)

Out of the box, a merge to your default branch carrying a release-worthy commit (`feat:` / `fix:`) will version, tag,
update the changelog, cut a release, and **deploy your docs site** — but it will **not** publish your package or a
Docker image. Those two are **opt-in**, configured per repository so that a project which isn't ready to publish never
sees a red pipeline. You turn them on by adding repository/CI variables (no code changes); leave a variable unset and
its job is simply skipped (a clean, green pipeline).

| What             | Enable it with               | Destination                               | Credentials                     |
| ---------------- | ---------------------------- | ----------------------------------------- | ------------------------------- |
| **Docs site**    | *(on by default)*            | GitHub Pages / GitLab Pages               | GitLab: none. GitHub: see below |
| **Package**      | variable `PUBLISH_TARGET`    | PyPI, a private index, or GitLab registry | see below                       |
| **Docker image** | variable `BUILD_DOCKER=true` | ghcr.io / GitLab container registry       | none (uses the CI token)        |

`PUBLISH_TARGET` values:

- **GitHub** — `pypi-oidc` publishes to PyPI via OIDC Trusted Publishing (no secrets); `private-index` publishes to a
    custom index (also set `PRIVATE_INDEX_URL` + secrets `PRIVATE_INDEX_USERNAME` / `PRIVATE_INDEX_PASSWORD`). Unset →
    package publishing is skipped.
- **GitLab** — set it to any value (e.g. `gitlab-registry`) to publish to the project's built-in PyPI registry via
    `CI_JOB_TOKEN` (no extra credentials). Unset → skipped.

The exact per-platform UI steps and credentials are in [Repository settings](#repository-settings) below.

## Conventional commits & versioning

Versioning is **automatic** and driven by your commit messages, so there are no manual version bumps. Write commits in
the [Conventional Commits](https://www.conventionalcommits.org/) format:

- `fix: ...` → patch release (e.g. `0.1.0` → `0.1.1`)
- `feat: ...` → minor release (e.g. `0.1.1` → `0.2.0`)
- a `feat!: ...` or a `BREAKING CHANGE:` footer → major release

A **release-worthy commit** is any commit whose Conventional-Commit type triggers a version bump: `fix:` (patch),
`feat:` (minor), or a breaking change (`!` or a `BREAKING CHANGE:` footer, major). Other types — `docs:`, `chore:`,
`test:`, `refactor:`, `ci:`, `style:` — do **not** trigger a release on their own; a push containing only those changes
produces no new version.

### Staying pre-1.0 (and how you eventually reach 1.0)

While the project is pre-1.0, `major_version_zero` is enabled: breaking changes are treated as a **minor** bump rather
than a major one, so a `0.x` release never automatically jumps to `1.0`. In other words, `feat!:` / `BREAKING CHANGE:`
on a `0.x` project bumps `0.3.0` → `0.4.0`, not `1.0.0`. This is intentional — pre-1.0 software is expected to break,
and it avoids an accidental "stable" `1.0` signal.

Reaching `1.0` is therefore a **deliberate, manual decision**, never something that happens on its own. When you are
ready to declare the API stable, either:

- run `uv run cz bump --major` to cut the `1.0.0` release explicitly, or
- turn `major_version_zero` off in the commitizen config, after which breaking changes bump the major version normally.

Until you take one of those steps, the project stays in the `0.x` series indefinitely.

The `commit-msg` hook rejects non-conforming messages locally. On merge to the default branch, CI runs
`uv run poe bump`: commitizen computes the next version from the commits since the last tag, updates `CHANGELOG.md`,
creates the git tag, deploys the docs, and — if you've opted in (see [Publishing](#5-publishing-optional)) — publishes
the package and/or Docker image. If there are no release-worthy commits, nothing is released.

You can preview the next changelog locally with `uv run poe changelog`.

## Repository settings

> These are the one-time steps to get **your generated project** to a green first pipeline — do them in your repo's
> hosting UI after the first push.

A few settings live in the hosting platform's UI. Most importantly, **protect the default branch**: the
`no-commit-to-branch` pre-commit hook only blocks direct commits *locally* (in each contributor's clone), so pair it
with server-side protection that requires changes to land via a reviewed PR/MR. The release automation then needs a way
to push the version bump, and **how that plays out differs by platform** (details in the per-platform notes below):

- **GitHub** — the credential is *conditional on protection*. Leave the default branch unprotected and the built-in
    token pushes the release itself with no extra credential (but then `main` isn't protected, defeating the reviewed-PR
    workflow); protect it and you must add a `RELEASE_TOKEN` or a `github-actions[bot]` bypass so the bump can land.
- **GitLab** — the credential is *required either way*. The pipeline authenticates the push as `oauth2:$CI_REPO_ACCESS`
    (the built-in `CI_JOB_TOKEN` can't push commits back), and GitLab protects the default branch out of the box, so
    `CI_REPO_ACCESS` is not optional.

### Enable Automatic Dependency Updates

This project includes a pre-configured [renovate.json](<(https://docs.renovatebot.com/)>) to keep your dependencies,
pre-commit hooks, and CI actions up to date automatically. To activate it on GitHub:

1. Navigate to the [Mend Renovate App](https://github.com/apps/renovate) on the GitHub Marketplace.
1. Click **Install** (or **Configure** if your organization already uses it).
1. Choose **Only select repositories** and select this project's repository.
1. Click **Save**.

> **Note:** Renovate will automatically read the `renovate.json` file at the root of this project. It will bundle
> development tooling updates into a single PR while opening individual PRs for production dependency updates to isolate
> test failures.

*(If you are hosting this project on GitLab, you will need to rely on your organization's internal Renovate bot or
configure a [GitLab Renovate runner](https://docs.renovatebot.com/modules/platform/gitlab/).)*

### If On GitHub...

- **Settings → Actions → General → Workflow permissions**: set **Read and write permissions** so the release workflow
    can push the bump commit/tag and create releases.
- **Branch protection** on `main`: enable **Require a pull request before merging** so collaborators can't push directly
    (the local `no-commit-to-branch` hook only guards each clone). Then let the release workflow push the bump — either
    add a bypass for `github-actions[bot]`, or create a fine-grained PAT with `contents: write`, store it as the
    `RELEASE_TOKEN` secret (the workflow prefers it over `GITHUB_TOKEN`), and allow it to bypass protection.
- **Pages** — **enable once, by hand, before your first release.** Set **Settings → Pages → Source: "GitHub Actions"**.
    CI cannot enable Pages for you: creating the site needs `administration: write`, a permission the default
    `GITHUB_TOKEN` never has. Until you do this the `deploy-docs` job fails; once enabled it stays enabled. Docs are
    then deployed by the official GitHub Pages Actions from the release workflow run — there is no `gh-pages` branch,
    and only the single current version is published.
- **Publishing the package** — *opt-in; unset `PUBLISH_TARGET` means no package is published*:
    - *PyPI via OIDC Trusted Publishing*: set repo **variable** `PUBLISH_TARGET=pypi-oidc` and configure a
        [trusted publisher](https://docs.pypi.org/trusted-publishers/) on PyPI for this repo (workflow file `ci.yml`,
        environment `pypi` if you use one). No secrets required.
    - *Private index instead*: set **variable** `PUBLISH_TARGET=private-index` and **variable** `PRIVATE_INDEX_URL`, plus
        secrets `PRIVATE_INDEX_USERNAME` / `PRIVATE_INDEX_PASSWORD`.
- **Publishing a Docker image** — *opt-in*: set repo **variable** `BUILD_DOCKER=true`. The image is pushed to
    `ghcr.io/<owner>/<repo>` using the built-in `GITHUB_TOKEN` (the workflow already grants `packages: write`), so no
    extra credentials are needed. Newly created packages are private by default — adjust visibility under the repo's
    **Packages** if you want it public.

### If On GitLab...

Configure these under **Settings** for your project:

- **General → Badges** (optional): add a `Pipeline` badge, using your GitLab instance host with the `%{project_path}`
    placeholder for the link and `%{project_path}/badges/%{default_branch}/pipeline.svg` for the image.
- **Access Tokens**: create a project access token named `CI_REPO_ACCESS` with the **`api`** scope and a `Maintainer`
    role. The `api` scope is required (not just `write_repository`): the pipeline both pushes the version-bump commit +
    tag to the protected default branch *and* creates the GitLab Release entry via the Releases API — the latter returns
    `403 insufficient_scope` under a repository-only token.
- **CI/CD → Variables**:
    - Add `CI_REPO_ACCESS` (the token value) as **masked** and **not protected**.
    - Add `GIT_SSL_CAINFO` (type **File**) with any SSL certificates required to reach GitLab.
- **Repository → Protected branches**: protect the default branch and set *Allowed to push* so changes land only via
    merge requests — but allow `CI_REPO_ACCESS` (or Maintainers) to push so the automated bump can land.
- **Merge requests → Merge checks**: enable `Pipelines must succeed` and `All threads must be resolved`.
- **Publishing** — *both are off until you set a CI/CD variable, so an unconfigured project never fails*:
    - *Package*: set variable `PUBLISH_TARGET` (any value, e.g. `gitlab-registry`) to publish to the project's built-in
        [PyPI registry](https://docs.gitlab.com/ee/user/packages/pypi_repository/) via `CI_JOB_TOKEN` — no extra
        credentials.
    - *Docker image*: set variable `BUILD_DOCKER=true` to build and push to the project's built-in
        [container registry](https://docs.gitlab.com/ee/user/packages/container_registry/) via `CI_JOB_TOKEN`.
- **Pages**: no UI toggle is needed — GitLab Pages is served by the pipeline's reserved `pages` job, which publishes the
    `public/` directory. The single current version of the docs is served once that job runs on the default branch. On a
    **private/internal** project the Pages site sits behind Pages access control, so an embedded README `<img>` from the
    Pages host (a different domain than your GitLab instance) can't authenticate and would 404. That is why the coverage
    badge uses GitLab's native, repo-hosted `.../badges/<branch>/coverage.svg` (fed by the `run-tests` job's `coverage:`
    regex) instead of a Pages-hosted SVG. To surface the Pages-hosted test/coverage reports publicly anyway, enable
    **Settings → Deploy → Pages → make the Pages site public**.
