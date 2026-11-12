/* Chart helpers for the Phase 0 deck (copied from p1-findings/assets/js/p1.js), on top of the template's Deck.echarts().

   Every chart reads window.DeckData (assets/data.js, written by make_data.py) and is
   registered with Deck.chart(id, el => ...) in index.qmd, so it is drawn the first time
   its slide is shown. Values in DeckData are fractions; charts show points (x 100).

   ECharts has no built-in interval bands or error bars, so this file adds both as
   `custom` series, plus a multi-panel layout and the zero / region marks the deck uses. */
(function () {
  "use strict";

  const C = {
    yellow: "#FECB00", blue: "#2196F3", green: "#4CAF50", orange: "#FF9800", red: "#E63946",
    muted: "#8a9bab", text: "#cecece", white: "#ffffff",
  };
  const ARM = {
    "JE (MAE-init)": C.blue,
    "JE (MAE-init), P1.2.1 recipe": "#90CAF9",
    "SimSiam (DINO-init)": C.yellow,
    "DINO (DINO-init)": C.green,
    MAE: C.orange,
  };
  const FONT = { tick: 19, name: 21, legend: 20, title: 22, note: 18 };

  const P = { C, ARM, FONT };

  P.data = (name) => window.DeckData[name];
  P.pts = (v) => 100 * v;
  P.signed = (v, d = 1) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);

  /* Rows grouped by key(row), in first-seen order. */
  P.groupBy = (rows, key) => {
    const m = new Map();
    for (const r of rows) {
      const k = key(r);
      if (!m.has(k)) m.set(k, []);
      m.get(k).push(r);
    }
    return m;
  };
  P.uniq = (xs) => [...new Set(xs)];

  /* A value axis at the deck's font sizes; `extra` overrides. */
  P.axis = (name, extra = {}) => ({
    type: "value", name, nameLocation: "middle", nameGap: 44, scale: true,
    axisLabel: { fontSize: FONT.tick },
    nameTextStyle: { fontSize: FONT.name },
    axisLine: { onZero: false },
    ...extra,
  });

  /* Round axis bounds and tick step covering [lo, hi], for axes shared across panels. */
  P.nice = (lo, hi, ticks = 6) => {
    const raw = (hi - lo) / ticks, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    return { min: Math.floor(lo / step) * step, max: Math.ceil(hi / step) * step, interval: step };
  };

  /* n side-by-side grids, with a title over each. Margins are given in pixels at
     the chart's current width and stored as percentages, so the panels reflow when
     responsive.js resizes the canvas (ECharts' resize() keeps percentages). */
  P.panels = (n, { left = 90, right = 20, top = 70, bottom = 130, gap = 50, width = 1000, titles = [] } = {}) => {
    const pc = (px) => `${(100 * px) / width}%`;
    const w = (width - left - right - gap * (n - 1)) / n;
    const grid = [], title = [];
    for (let i = 0; i < n; i++) {
      const x = left + i * (w + gap);
      grid.push({ left: pc(x), width: pc(w), top, bottom });
      if (titles[i]) title.push({ text: titles[i], left: pc(x + w / 2), top: top - 46, textAlign: "center",
        textStyle: { fontSize: FONT.title - 2, fontWeight: "normal", color: C.text } });
    }
    return { grid, title };
  };

  P.legend = (extra = {}) => ({ bottom: 0, left: "center", textStyle: { fontSize: FONT.legend }, itemGap: 28, ...extra });
  P.title = (text) => ({ text, left: "center", top: 0, textStyle: { fontSize: FONT.title, fontWeight: "normal", color: C.text } });

  /* Shaded interval band between lo and hi (arrays aligned with xs). */
  P.band = (xs, lo, hi, color, idx = {}, alpha = 0.15) => ({
    type: "custom", silent: true, tooltip: { show: false }, z: 1, legendHoverLink: false, ...idx,
    data: xs.map((x, i) => [x, lo[i], hi[i]]),
    encode: { x: 0, y: [1, 2] },
    renderItem: (params, api) => {
      if (params.dataIndex !== 0) return null;
      const top = xs.map((x, i) => api.coord([x, hi[i]]));
      const bot = xs.map((x, i) => api.coord([x, lo[i]])).reverse();
      return { type: "polygon", shape: { points: top.concat(bot) }, style: { fill: Deck.alpha(color, alpha) } };
    },
  });

  /* Error whiskers on points: data rows [x, y, lo, hi]; horizontal draws lo..hi along x. */
  P.whiskers = (rows, color, { horizontal = false, cap = 6, width = 2.5, name, ...idx } = {}) => ({
    type: "custom", silent: true, tooltip: { show: false }, z: 2, name, ...idx,
    data: rows,
    encode: horizontal ? { x: [0, 2, 3], y: 1 } : { x: 0, y: [1, 2, 3] },
    renderItem: (params, api) => {
      const x = api.value(0), y = api.value(1), lo = api.value(2), hi = api.value(3);
      const a = horizontal ? api.coord([lo, y]) : api.coord([x, lo]);
      const b = horizontal ? api.coord([hi, y]) : api.coord([x, hi]);
      const style = { stroke: color, lineWidth: width };
      const capAt = (p) => horizontal
        ? { type: "line", shape: { x1: p[0], y1: p[1] - cap, x2: p[0], y2: p[1] + cap }, style }
        : { type: "line", shape: { x1: p[0] - cap, y1: p[1], x2: p[0] + cap, y2: p[1] }, style };
      return { type: "group", children: [
        { type: "line", shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, style }, capAt(a), capAt(b),
      ] };
    },
  });

  /* Whiskers for bar k of n in a grouped bar chart on a category axis (rows [j, y, lo, hi]). */
  P.barWhiskers = (rows, k, n, { catGap = 0.3, barGap = 0.1, color = "rgba(255,255,255,0.6)", cap = 6, ...idx } = {}) => ({
    type: "custom", silent: true, tooltip: { show: false }, z: 3, ...idx,
    data: rows,
    encode: { x: 0, y: [1, 2, 3] },
    renderItem: (params, api) => {
      const j = api.value(0), lo = api.coord([j, api.value(2)]), hi = api.coord([j, api.value(3)]);
      const band = api.size([1, 0])[0], bar = (band * (1 - catGap)) / (n + (n - 1) * barGap);
      const x = lo[0] - (band * (1 - catGap)) / 2 + k * bar * (1 + barGap) + bar / 2;
      const style = { stroke: color, lineWidth: 2 };
      return { type: "group", children: [
        { type: "line", shape: { x1: x, y1: lo[1], x2: x, y2: hi[1] }, style },
        { type: "line", shape: { x1: x - cap, y1: lo[1], x2: x + cap, y2: lo[1] }, style },
        { type: "line", shape: { x1: x - cap, y1: hi[1], x2: x + cap, y2: hi[1] }, style },
      ] };
    },
  });

  /* An invisible series that carries marks for one grid: a zero line, a shaded
     x- or y-region with a label, a labelled reference line. */
  P.marks = ({ zeroY = false, zeroX = false, xRegion, yRegion, yLine, ...idx } = {}) => {
    const lines = [];
    if (zeroY) lines.push({ yAxis: 0, lineStyle: { color: C.muted, type: "solid", width: 2 }, label: { show: false } });
    if (zeroX) lines.push({ xAxis: 0, lineStyle: { color: C.muted, type: "solid", width: 2 }, label: { show: false } });
    if (yLine) lines.push({ yAxis: yLine.y, lineStyle: { color: yLine.color, type: "dotted", width: 3 },
      label: { show: true, position: "insideEndTop", formatter: yLine.text, color: yLine.color, fontSize: FONT.note } });
    const areas = [];
    for (const [r, axis] of [[xRegion, "xAxis"], [yRegion, "yAxis"]]) {
      if (!r) continue;
      areas.push([
        { [axis]: r.from, name: r.text || "", itemStyle: { color: Deck.alpha(r.color || C.muted, r.alpha ?? 0.15) },
          label: { show: !!r.text, position: r.labelPosition || (axis === "xAxis" ? "insideTop" : "insideRight"),
            color: r.labelColor || r.color || C.muted, fontSize: FONT.note, distance: 10 } },
        { [axis]: r.to },
      ]);
    }
    return {
      type: "line", data: [], silent: true, tooltip: { show: false }, legendHoverLink: false, ...idx,
      markLine: lines.length ? { silent: true, symbol: "none", data: lines, animation: false } : undefined,
      markArea: areas.length ? { silent: true, data: areas } : undefined,
    };
  };

  /* Common option scaffolding: item tooltips, no entry animation surprises. */
  P.draw = (el, option) => Deck.echarts(el, { tooltip: { trigger: "item" }, ...option });

  window.Charts = P;
})();
