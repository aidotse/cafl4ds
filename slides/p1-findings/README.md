# P1.1 / P1.2 update — slide deck

Quarto + reveal.js, on the [aidotse/quarto-pres](https://github.com/aidotse/quarto-pres) template. Published with the
docs site at <https://aidotse.github.io/cafl4ds/slides/p1-findings/> (see `slides/README.md`).

## Preview and render (anywhere with Quarto)

```bash
quarto preview index.qmd
quarto render index.qmd
```

No Python or Jupyter is needed. The charts are [Apache ECharts](https://echarts.apache.org/), drawn in the browser from
the numbers in `assets/data.js` (hover for values). Everything is vendored under `assets/`, so the rendered deck works
offline.

## Layout

- `index.qmd`: the slides. Each chart is a `<div class="chart">` plus a `Deck.chart(id, el => ...)` block.
- `assets/data.js`: the plotted numbers (`window.DeckData`), written by `make_data.py`. Do not edit by hand.
- `assets/js/deck.js`: the template's chart theme and slide-aware drawing (`Deck.chart`, `Deck.echarts`).
- `assets/js/responsive.js`: the template's responsive canvas. It fits the slide to the window (16:10 and 4:3 grow
    taller, ultrawide grows wider) and switches phones and tablets in portrait to one stacked column.
- `assets/js/p1.js`: this deck's chart helpers: interval bands, error whiskers, multi-panel grids, zero lines and shaded
    regions.
- `assets/vendor/echarts.min.js`: Apache ECharts.
- `style.css`: the template's full stylesheet, then a block of this deck's additions (`.side` bullet columns and
    captions, `.lead-in`, `.chart.mid`, the loop diagram, the Phase 0 table, cards with `<h3>` titles).

## Refresh the numbers (on the Gaudi node)

```bash
# from the repo root: pull the plotted numbers out of outputs/ and the substudy docs into assets/data.js
uv run python slides/p1-findings/make_data.py
cd slides/p1-findings && quarto render index.qmd
```

`make_data.py` is the only Python, and it runs once, in the repo's own env, when the numbers change. It names the source
of every table. P1.1.1, P1.2.1, P1.2.2 and P1.2.4 numbers are transcribed from their docs' tables; P1.1.3 and P1.2.5 are
read from the run artifacts (Part F from its promoted `docs/experiments/phase1/artifacts/P1.2.5/`).
