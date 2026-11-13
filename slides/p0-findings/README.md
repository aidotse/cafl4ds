# Phase 0 findings — slide deck

Quarto + reveal.js, on the [aidotse/quarto-pres](https://github.com/aidotse/quarto-pres) template. Published with the
docs site at <https://aidotse.github.io/cafl4ds/slides/p0-findings/> (see `slides/README.md`).

An executive summary of Phase 0 (instrument calibration) for partners and collaborators: one chart per finding, built to
show what the substudy docs can only tabulate.

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
- `assets/js/charts.js`: chart helpers (interval bands, whiskers, multi-panel grids, marks), a copy of the P1 deck's
    `p1.js` exposed as `window.Charts`.
- `assets/js/deck.js`, `assets/js/responsive.js`, `style.css`: the template's files, as in `p1-findings/`.
- `assets/vendor/echarts.min.js`: a symlink to the P1 deck's copy; `slides/build.sh` publishes the real file.

## Refresh the numbers

```bash
# from the repo root: pull the plotted numbers into assets/data.js
uv run python slides/p0-findings/make_data.py
cd slides/p0-findings && quarto render index.qmd
```

`make_data.py` is the only Python, and it runs once, in the repo's own env, when the numbers change. It needs only the
tracked files, and names the source of every table:

- Every table but two reads the artifacts under `docs/experiments/phase0/artifacts/`. That includes the per-step and
    per-checkpoint traces (`P0.2.2/E3_16seed/iid_trajectories.json`, `P0.4.0/*/traces.json`, `P0.4.1/op_traces/`), which
    the substudies' slimmed artifacts drop and which were promoted next to them for this deck. Each trace file records
    its source run and the check tying it to the summary it belongs to.
- The STL-10 and CIFAR-100 forgetting rows are transcribed from P0.3.2 and P0.3.4 (no artifacts exist for them), and the
    scoreboard from the Phase 0 index.
