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
    implementation states, always read the phase landing page `docs/experiments/phase<X>/index.md` first (e.g.
    `docs/experiments/phase0/index.md`), and use the routing table in that file to find the exact substudy file you
    need. Never read an entire directory at once; read only the sub-studies required for the task. The
    `docs/experiments/latest-results/` directory is the distilled logbook — concise study/substudy summaries reached via
    the phase index, not a router in its own right.

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
- All business logic / code / experiments should be validated with unit tests; be careful not to overfit the tests on
    the logic / code.

## Claude Code in this repo

Runs behind a token-compression pipeline (RTK shell-hook + Headroom API proxy): bash output may be RTK-filtered.

## Documenting experiments

This is a guide for how to add documentation on new studies or sub-studies. See notes above on `docs/experiments`. See
\`Explaining your work' below for how to communicate results. There are different levels of granularity at play (here:
most concise to least concise): A `phase` is composed of `studies`, which are in turn composed of `substudies`. This
corresponds to the following docs hierarchy:

- `docs/experiments/phase<ID>/index.md` - the phase landing page: what the *project phase* is about, plus routing
    tables. Keep each of its three parts at an **at-a-glance** abstraction: very high level, easily scannable. Use
    plain, transparent language. This is meant as an executive summary, and the detail lives in the levels below,
    linked. The individual components of an index file are:

    - **Goal from the project plan** - a short summary of how this project phase fits into the larger project plan.

    - an **Executive Summary** of the whole Phase; it's a placeholder until all studies in the phase are complete, or the
        user requests for it.

    - **Methodology** - a summary of the overarching methodology used, from the perspective of the whole phase rather than
        that of any single study or substudy. Do not dive into too much detail, that's reserved for each individual study.

    - the **Exit Gate**: the criteria by which we consider this phase complete ("what questions must we answer here?")

    - a **Studies** table, effectively an index into all substudies in this phase. All columns stay to a scannable clause
        each. Studies and their sub-studies share this same table. A study appears as a single `P1.x` row until its work
        begins; once it is broken down, that row is **replaced** by its `P1.x.y` sub-study rows (so nothing is stated at
        both levels). Sub-study rows follow the conventions below. The columns are:

        - **ID** - e.g. Px.y.z, the identifier of the substudy.
        - **Study / sub-study** - A short title for the study or sub-study in question, wrapped as a link to its longer
            summary in `latest-results/`. Marked with the type of work done in the study or substudy, one of `engineering`,
            `analysis`, `new substudy`, `wording`, `new runs`, `other`.
        - **What it does** - a summary of the intent ("what are we testing or hoping to achieve?"), not overly verbose or
            technical. Keep the style uniform: cell opens with `Motivation:`.
        - **Status / Outcome** - follows `<marker> **<label>** (<finding>)` — marker ∈ ✅ / ✅\* / 🟡 / 🔲, label ∈ `Complete` /
            `Runs complete` / `Designed, awaiting runs`. A very high level summary of the findings.

    - an **Artifacts** section holding only the *general* conventions (per-harness `comparison.json` schema, the promotion
        / slimming policy). The per-sub-study listings live in each detail doc, see below.

    - **Handover** - a summary of the information that is especially pertinent for future project studies and phases.

- `docs/experiments/latest-results/<study ID>.md` (e.g. `P0.6.md`) - a study-wide **executive summary**, for a reader
    who may not know the project and does not want to dig. Plain, transparent language, ≲ 100 lines, exactly three
    subsections: **The value proposition**, **Results**, **The one-line takeaway**. It sits one level above the
    sub-study summaries.

- `docs/experiments/latest-results/<substudy ID>.md` (e.g. `P0.6.0.md`)- a *concise* summary of the results / what was
    achieved in the particular substudy. Avoid too many details (in particular: NO results numbers); this is a brief
    take-away from the study and is the target of the links from the index above. It, in turn links to the detail doc
    one level down in `phase<ID>/<substudy ID>.md`.

- `docs/experiments/phase<ID>/<substudy ID>.md` - a detailed, *but not overly verbose or repetitive* description of the
    sub study, including outputs, and key quantitative results. Includes (roughly) a brief section for the motivation,
    the methodology and findings, any interesting insights, a dedicated `Artifacts` section (the per-sub-study artifact
    listing: which runs back its numbers, and the slim schema each was promoted to), and how to run the experiments. Do
    not mention progress or status here (unless results change an existing verdict), that goes in the top-level index
    file.

When we say something like "update the docs", this means going through each of these levels for the relevant study /
substudy and checking them for completeness, consistency, and accuracy against the latest results, as well as making
sure that all other sections that depend on these (like the other sections in the index) are also synced and up to date.

### Writing audits

Audits (`docs/experiments/audits/<ID>.md`) are peer-review-style critiques of a completed study or sub-study group. They
judge whether each claim / conclusion is earned. They double as a remediation record: findings get fixed, and the audit
tracks where each one landed. Follow this formula:

- **Language.** Compact but accessible — the same style as *Explaining your work* below (one idea per sentence, signpost
    the logic, unpack a term before leaning on it). Keep every load-bearing number; cut the padding, not the content. No
    wall-of-text paragraphs.

- **Code audit.** First, run a comprehensive audit of the code used in the study or sub-study - if the code is buggy,
    then reviewing conclusions is moot. Conclude this review with an assessment of which claims are affected, if any,
    and a remediation plan to run before the rest of the audit proceeds.

Next, proceed with the main body of the audit:

- **Sections, in order.**

    - **Title** — one line, carrying a *single* audit date (when the review was conducted). Do NOT scatter per-item "done
        (date)" lines through the body; the completion signal lives in each point's Conclusion.
    - **Verdict** — the headline judgment, as *found*. End with one sentence on where it lands *after* remediation (e.g.
        "all points remediated; the ✅ ends up stronger").
    - **Strengths** — a short bullet list, so the criticism is calibrated (what the work got right).
    - **Grouped findings** — lettered sections (A, B, C, …) by *kind*: over-claims, internal tensions, structural gaps,
        conclusions the data supports but the docs missed, and a code-validity check i.e. the output of the code audit
        mentioned above. Number the points within (A1, A2, …).
    - **Disposition** — the final lettered section: the decision of record after every remediation, plus the hand-off
        recommendation. This *is* the bottom line — do not also keep a separate summary table or "bottom line" paragraph
        (that is scaffolding; remove it once the audit is worked through).

- **Each numbered point is a triple** — a three-bullet sub-list under a bold one-line title:

    - **Issue** — what the audit found (the as-found observation; keep it to the claim, tight).
    - **Remediation** — recommendation on how to fix, tagged with the work type: `[wording]` (doc edit) / `[analysis]`
        (post-hoc, no runs) / `[engineering]` (build + confirm) / `[new runs]` (re-run an *existing* config, e.g. seed
        replication — the results land in the substudy docs, usually *in-place* in the existing substudy: confirming
        results append a line to its conclusions, differing results revise its text; a substantial follow-up arc — a new
        lever plus its own finding — is instead promoted to its own doc, as with `[new substudy]`) / `[new substudy]` (a
        genuinely new vehicle/study warranting its own doc) / `[decision]` (needs an architectural call) /
        `[defer → Phase-<ID>]`.
    - **Conclusion** — the outcome after remediation, a placeholder until the remediation is actually implemented and
        agreed upon by the user. Led by a bold *verdict label* so a reviewer reads the valence at a glance. Use a small,
        consistent set: *resolved — strengthens the ✅* (addressed, and the result reinforces the study), *new supporting
        finding* (the data supported a favourable conclusion the docs had missed), *corrected* (a stated claim was wrong
        and revised — append "— net favourable" or "— claim downgraded" to state the direction), *clarified — no verdict
        change* (wording sharpened or an overstatement trimmed), *valid concern — carried to Phase-<ID>* (a real
        limitation, honestly deferred — not fixable now).

- **The audit is not the home of the results:** when a remediation produces experimental findings (`[new runs]`,
    `[new substudy]`, or an `[analysis]` that yields new numbers), those findings are folded back into thethe substudy
    docs (the `latest-results` distillate + the phase detail doc, or a new substudy doc if the arc warrants one); the
    Conclusion carries only the verdict label, a one-line outcome, and a **pointer** to where it landed — never raw
    numbers or results directly (other than in summary).

## Explaining your work

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

### Response Formatting

This applies only to conversations in the terminal.

Most terminal applications have difficulties rendering markdown, hence reading long responses can be tedious if there is
no highlighting to orient the reader. Therefore, please rely on `raw text` whenever you want to highlight something in
your response: section titles, and any boldface or italics text.

## Mandates

We have two general mandates: completeness and accuracy. For completeness, this means e.g. making suggestions to the
user when a study has an evident, open scientific gap; always get permission to go down research tangents before doing
so. For accuracy, this means making sure the docs reflect the latest results, and ensuring your updates do not
contradict the specific index or substudy files you have been instructed to read - if this should happen, notify the
user immediately to remediate before moving on. Do not read additional documentation files solely to check for global
project consistency unless explicitly requested.
