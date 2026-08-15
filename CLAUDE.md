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

You is a guide for how to add documentation on new studies or sub-studies. See notes above on `docs/experiments`. Use
informative, but not terse language. There are different levels of granularity at play (here: most concise to least
concise):

- `docs/experiments/phase<ID>/index.md` - summary details of what the *project phase* is about, plus a routing table to
    the substudies. In this table: only the *motivation*, and an *executive, 1-2 line summary* of the substudy status:
    keep brief.

- `docs/experiments/latest-results/<substudy ID>.md` - a *concise* summary of the results / what was achieved in the
    substudy. Avoid too many details (in particular: NO results numbers); this is a a brief take-away from the study.

- `docs/experiments/phase<ID>/<substudy ID>.md` - a detailed, *but not overly verbose or repetetive* description of the
    sub study, including outputs, and key quantitative results. Includes (roughly) a brief section for the motivation,
    the methodology and findings, any interesting insights, and how to run the experiments in the substudy. Do not
    mention progress or status here, that goes in `docs/experiments/latest-results/<substudy ID>.md`.

### Writing audits

Audits (`docs/experiments/audits/<ID>.md`) are peer-review-style critiques of a completed study or sub-study group. They
review *conclusions, not arithmetic* — spot-check the numbers against the artifacts, then judge whether each claim is
earned. They double as a remediation record: findings get fixed, and the audit tracks where each one landed. Follow this
formula:

- **Language.** Compact but accessible — the same style as *Explaining your work* below (one idea per sentence, signpost
    the logic, unpack a term before leaning on it). Keep every load-bearing number; cut the padding, not the content. No
    wall-of-text paragraphs.

- **Sections, in order.**

    - **Title** — one line, carrying a *single* audit date (when the review was conducted). Do NOT scatter per-item "done
        (date)" lines through the body; the completion signal lives in each point's Conclusion.
    - **Verdict** — the headline judgment, as *found*. End with one sentence on where it lands *after* remediation (e.g.
        "all points remediated; the ✅ ends up stronger").
    - **Strengths** — a short bullet list, so the criticism is calibrated (what the work got right).
    - **Grouped findings** — lettered sections (A, B, C, …) by *kind* — e.g. over-claims, internal tensions, structural
        gaps, conclusions the data supports but the docs missed, and a code-validity check of the premise. Number the
        points within (A1, A2, …).
    - **Disposition** — the final lettered section: the decision of record after every remediation, plus the hand-off
        recommendation. This *is* the bottom line — do not also keep a separate summary table or "bottom line" paragraph
        (that is scaffolding; remove it once the audit is worked through).

- **Each numbered point is a triple** — a three-bullet sub-list under a bold one-line title:

    - **Issue** — what the audit found (the as-found observation; keep it to the claim, tight).
    - **Remediation** — what to do, tagged with the work type: `[wording]` (doc edit) / `[analysis]` (post-hoc, no runs) /
        `[engineering]` (build + confirm) / `[new runs]` (re-run an *existing* config, e.g. seed replication — the results
        land in the substudy docs, usually *in-place* in the existing substudy: confirming results append a line to its
        conclusions, differing results revise its text; a substantial follow-up arc — a new lever plus its own finding —
        is instead promoted to its own doc, as with `[new substudy]`) / `[new substudy]` (a genuinely new vehicle/study
        warranting its own doc) / `[decision]` (needs an architectural call) / `[defer → Phase-1]`.
    - **Conclusion** — the outcome after remediation, led by a bold *verdict label* so a reviewer reads the valence at a
        glance. Use a small, consistent set: *resolved — strengthens the ✅* (addressed, and the result reinforces the
        study), *new supporting finding* (the data supported a favourable conclusion the docs had missed), *corrected* (a
        stated claim was wrong and revised — append "— net favourable" or "— claim downgraded" to state the direction),
        *clarified — no verdict change* (wording sharpened or an overstatement trimmed), *valid concern — carried to
        Phase-1* (a real limitation, honestly deferred — not fixable now). **The audit is not the home of the results:**
        when a remediation produces experimental findings (`[new runs]`, `[new substudy]`, or an `[analysis]` that yields
        new numbers), those findings live in the substudy docs (the `latest-results` distillate + the phase detail doc, or
        a new substudy doc if the arc warrants one); the Conclusion carries only the verdict label, a one-line outcome,
        and a **pointer** to where it landed — never the seed tables or derivations themselves.

## Explaining your work (in your replies to me)

When you explain reasoning, summarize a finding, or walk me through a concept in a reply — this is about your
conversational prose, and documentation language, but not code (this keep its own rules) — write for a reader who wants
to *follow the logic*, not decode it. The failure mode to avoid is dense, compressed prose: sentences that stack three
clauses, piled-up noun-phrase jargon, and caveats crammed into parentheticals. Terse is not the same as efficient — if I
have to re-read a sentence to parse it, it cost me more, not less. Avoid over-repetition.

Concretely, in explanatory passages:

- **One idea per sentence.** Prefer several plain sentences to one clause-stacked sentence. Let the prose breathe.
- **Unpack a concept before you lean on it.** The first time a term or mechanism does real work, say what it means in
    ordinary words, then use it.
- **Signpost the logic.** Small connectives — "The catch is…", "Why this matters…", "The upshot is…" — let me track
    where an argument is heading.
- **Pair the *what* with the *why it matters*.** Don't just state a fact; say what turns on it.
- **Reach for a concrete analogy** when it makes an abstract point land.
- **Give a caveat its own sentence**, rather than folding it into a parenthetical aside.

This is about clarity and flow, **not** length: do not pad, repeat, or add filler, and never drop information content to
hit the style. For a direct factual answer, a status line, or a yes/no, stay short — the guidance above is for the
explanatory paragraphs, where following along matters more than compression.

## Mandates

- Ensure your updates do not contradict the specific index or substudy files you have been instructed to read. Do not
    read additional documentation files solely to check for global project consistency unless explicitly requested.
