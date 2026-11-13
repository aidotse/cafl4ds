# Slide decks

Presentation decks on the project's results, one directory per deck, built on the
[aidotse/quarto-pres](https://github.com/aidotse/quarto-pres) template. They are published with the docs site:

- [Phase 0 findings](https://aidotse.github.io/cafl4ds/slides/p0-findings/) (`p0-findings/`)
- [Phase 1 findings](https://aidotse.github.io/cafl4ds/slides/p1-findings/) (`p1-findings/`)

## How a deck is built and published

- A deck renders with [Quarto](https://quarto.org/) alone: no Python or Jupyter. Its charts are Apache ECharts, drawn in
    the browser from a generated `assets/data.js`.
- `slides/build.sh [OUT]` renders every deck and copies each to `OUT/<deck>/` (default `public/slides/`).
- The release pipeline's docs job (`.github/workflows/reusable-pipeline.yml`) runs it right after `mkdocs build`, so the
    decks ship in the same GitHub Pages artifact as the docs. A repository has one Pages site, so they cannot deploy on
    their own. A deck therefore goes live with the next release, or when the manual "Publish site" workflow
    (`.github/workflows/publish-site.yml`) is run on `main`.
- Rendered output (`index.html`, `index_files/`, `.quarto/`) is not committed.
- Decks share one copy of ECharts: `p0-findings/assets/vendor/echarts.min.js` is a symlink to the `p1-findings` copy,
    and `build.sh` copies with `-L`, so each published deck gets the real file.

## Working on a deck

```bash
cd slides/<deck> && quarto preview index.qmd
```

Each deck's own README says how to refresh its numbers (in each, a Python script writes `assets/data.js` from the run
artifacts and substudy docs). Give each deck's script its own name (`make_p0_data.py`, `make_p1_data.py`): `mypy` checks
the whole repo, and two files with the same module name stop it. Keep files under 2 MB (the repository's pre-commit
limit): prefer JPEG for photographic backgrounds.
