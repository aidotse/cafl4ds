# Slide decks

Presentation decks on the project's results, one directory per deck, built on the
[aidotse/quarto-pres](https://github.com/aidotse/quarto-pres) template. They are published with the docs site:

- [P1.1 / P1.2 update](https://aidotse.github.io/cafl4ds/slides/p1-findings/) (`p1-findings/`)

## How a deck is built and published

- A deck renders with [Quarto](https://quarto.org/) alone: no Python or Jupyter. Its charts are Apache ECharts, drawn in
    the browser from a generated `assets/data.js`.
- `slides/build.sh [OUT]` renders every deck and copies each to `OUT/<deck>/` (default `public/slides/`).
- The release pipeline's docs job (`.github/workflows/reusable-pipeline.yml`) runs it right after `mkdocs build`, so the
    decks ship in the same GitHub Pages artifact as the docs. A repository has one Pages site, so they cannot deploy on
    their own. A deck therefore goes live with the next release.
- Rendered output (`index.html`, `index_files/`, `.quarto/`) is not committed.

## Working on a deck

```bash
cd slides/<deck> && quarto preview index.qmd
```

Each deck's own README says how to refresh its numbers (for `p1-findings`, `make_data.py` writes `assets/data.js` from
the run artifacts and substudy docs). Keep files under 2 MB (the repository's pre-commit limit): prefer JPEG for
photographic backgrounds.
