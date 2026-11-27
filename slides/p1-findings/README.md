# Phase 1 findings — slide deck

Quarto + reveal.js, on the [aidotse/quarto-pres](https://github.com/aidotse/quarto-pres) template. Published with the
docs site at <https://aidotse.github.io/cafl4ds/slides/p1-findings/> (see `slides/README.md`).

The deck covers all of Phase 1 and grows with it: one section per study (P1.0 the loop, P1.1 reading degradation, P1.2
adapt vs freeze so far), opened by a status slide and closed by trust / not yet and what comes next.

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
- `assets/data.js`: the plotted numbers (`window.DeckData`), written by `make_p1_data.py`. Do not edit by hand.
- `assets/js/deck.js`: the template's chart theme and slide-aware drawing (`Deck.chart`, `Deck.echarts`).
- `assets/js/responsive.js`: the template's responsive canvas. It fits the slide to the window (16:10 and 4:3 grow
    taller, ultrawide grows wider) and switches phones and tablets in portrait to one stacked column.
- `assets/js/p1.js`: this deck's chart helpers (interval bands, error whiskers, multi-panel grids, zero lines and shaded
    regions), and `P1.audit`, which fills every `<span class="audit" data-study="P1.x">` (section badges, slide footers)
    from the status table.
- `assets/vendor/echarts.min.js`: Apache ECharts.
- `style.css`: the template's full stylesheet, then a block of this deck's additions (`.side` bullet columns and
    captions, `.lead-in`, `.chart.mid`, the loop diagram, the Phase 0 table, cards with `<h3>` titles, audit badges).

## Refresh the numbers

```bash
# from the repo root: read the phase index, the audits, the promoted artifacts and the substudy docs into assets/data.js
uv run python slides/p1-findings/make_p1_data.py
cd slides/p1-findings && quarto render index.qmd
```

`make_p1_data.py` is the only Python, and it runs in the repo's own env when the numbers change. It reads tracked files
only, so it runs on any checkout, and it names the source of every table.

- The status slide and the audit badges parse `docs/experiments/phase1/index.md` (each Studies row's status marker) and
    `docs/experiments/audits/P1.x.md` (an audit is open while its Disposition is a placeholder or a Conclusion is
    pending). The badges read "Audit closed · all remediations implemented", "Audit open · N findings, remediations
    proposed" (none implemented yet), "Audit open · k of N remediations implemented", or "Not yet audited", where a
    remediation counts as implemented once its finding's Conclusion is no longer *Pending*. Re-run after a status or
    audit change and the slide follows.
- The P1.1 audit chart parses the audit's findings and their `[tag]` remediation types; the reader-coverage matrix is
    transcribed from the audit and P1.1.4's design.
- P1.1.1, P1.2.1, P1.2.2 and P1.2.4 numbers are transcribed from their docs' tables; P1.1.3 and P1.2.5 are read from the
    promoted artifacts in `docs/experiments/phase1/artifacts/`.

## Adding a study

Add a section (`# [0N]{.section-num} P1.x · <question> {#p1x}`) with its badge
(`<span class="audit section-badge" data-study="P1.x"></span>`), and end each of its slides with the footer block the
other sections use. Put its numbers in a new function in `make_p1_data.py`, reading promoted artifacts, and update the
trust / not-yet and what-comes-next slides.
