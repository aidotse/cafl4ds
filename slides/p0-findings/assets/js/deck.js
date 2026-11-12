/* Deck helpers: shared chart theme, slide-aware chart init, count-up numbers,
   sparklines, typewriter fragments, token highlighting and QR codes.

   Charts are registered from inline <script> blocks in index.qmd and only
   drawn the first time their slide becomes visible (hidden slides have no
   size, so drawing them up front gives squashed charts):

     Deck.chart("my-id", el => Plotly.newPlot(el, data, Deck.layout({...}), Deck.config));

   Fragment-synced charts also get a step callback whenever the slide's
   fragment position changes (i = -1 before the first fragment):

     Deck.steps("my-id", { init(el) {...}, step(el, i) {...} });
*/
(function () {
  "use strict";

  const registry = [];
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const Deck = {
    colors: {},
    chart(id, init) { registry.push({ id, init }); },
    steps(id, { init, step }) { registry.push({ id, init, step }); },
  };

  /* ---------- theme ---------- */

  function readColors() {
    const c = Deck.colors;
    c.series = [1, 2, 3, 4, 5, 6, 7, 8].map((i) => cssVar(`--series-${i}`));
    for (const k of ["accent", "ink", "body", "muted", "grid", "axis", "surface", "green", "red"]) c[k] = cssVar(`--${k}`);
    c.seriesMuted = cssVar("--series-muted");
    c.font = getComputedStyle(document.querySelector(".reveal")).fontFamily;
  }

  const isObj = (v) => v && typeof v === "object" && !Array.isArray(v);
  function merge(base, extra) {
    const out = { ...base };
    for (const [k, v] of Object.entries(extra || {})) out[k] = isObj(v) && isObj(base[k]) ? merge(base[k], v) : v;
    return out;
  }

  function axis() {
    const c = Deck.colors;
    return {
      gridcolor: c.grid,
      linecolor: c.axis,
      zerolinecolor: c.axis,
      showline: true,
      ticks: "",
      tickfont: { size: 20, color: c.muted },
      title: { font: { size: 22, color: c.body }, standoff: 18 },
      automargin: true,
    };
  }

  /* Plotly layout with the deck's dark theme; `extra` is deep-merged on top.
     Extra axes (xaxis2, yaxis2, ...) get the same styling. */
  Deck.layout = function (extra = {}) {
    const c = Deck.colors;
    const base = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: c.font, size: 22, color: c.body },
      colorway: c.series,
      margin: { l: 20, r: 20, t: 20, b: 20 },
      xaxis: axis(),
      yaxis: axis(),
      legend: {
        orientation: "h", x: 0, y: 1.02, xanchor: "left", yanchor: "bottom",
        font: { size: 22, color: c.body }, bgcolor: "rgba(0,0,0,0)",
      },
      hoverlabel: {
        bgcolor: "#0b151e", bordercolor: "rgba(255,255,255,0.18)",
        font: { family: c.font, size: 22, color: "#ffffff" }, align: "left",
      },
      hovermode: "closest",
    };
    for (const k of Object.keys(extra)) if (/^[xy]axis\d+$/.test(k)) base[k] = axis();
    return merge(base, extra);
  };

  /* Mode bar appears on hover: zoom/pan/reset + SVG download for papers. */
  Deck.config = {
    responsive: true,
    displaylogo: false,
    displayModeBar: "hover",
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines", "hoverClosestCartesian", "hoverCompareCartesian"],
    toImageButtonOptions: { format: "svg", filename: "figure" },
  };

  /* Apache ECharts counterpart of Deck.layout(): registers a "deck" theme with
     the same colours, fonts and tooltip look, then draws `option` into `el`
     with the SVG renderer (stays crisp when Reveal scales the slide). */
  Deck.echarts = function (el, option) {
    const c = Deck.colors;
    if (!Deck.echarts.theme) {
      const axisCommon = {
        axisLine: { lineStyle: { color: c.axis } },
        axisTick: { show: false },
        axisLabel: { color: c.muted, fontSize: 17, fontFamily: c.font, margin: 10 },
        splitLine: { lineStyle: { color: c.grid } },
        nameTextStyle: { color: c.body, fontSize: 19, fontFamily: c.font },
      };
      Deck.echarts.theme = {
        color: c.series,
        backgroundColor: "transparent",
        textStyle: { fontFamily: c.font, color: c.body },
        categoryAxis: { ...axisCommon, splitLine: { show: false } },
        valueAxis: axisCommon,
        legend: { textStyle: { color: c.body, fontSize: 18 }, itemWidth: 22, itemHeight: 12 },
        tooltip: {
          backgroundColor: "#0b151e", borderColor: "rgba(255,255,255,0.18)",
          textStyle: { color: "#ffffff", fontSize: 18, fontFamily: c.font },
        },
      };
      echarts.registerTheme("deck", Deck.echarts.theme);
    }
    const chart = echarts.init(el, "deck", { renderer: "svg" });
    // No entry animation in print/PDF export, so the captured frame is final
    if (/print-pdf/i.test(window.location.search)) option = { ...option, animation: false };
    chart.setOption(option);
    return chart;
  };

  /* Hex colour -> rgba() string, for translucent fills (bands, areas). */
  Deck.alpha = function (hex, a) {
    const n = parseInt(hex.replace("#", ""), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  };

  /* Seeded RNG so synthetic demo data is identical on every render. */
  Deck.rng = function (seed = 1) {
    let s = seed >>> 0;
    const uniform = () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const normal = (mu = 0, sd = 1) => mu + sd * Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
    return { uniform, normal };
  };

  /* ---------- count-up numbers ---------- */

  function countUp(slide) {
    for (const el of slide.querySelectorAll(".count[data-to]")) {
      const to = parseFloat(el.dataset.to);
      const dec = parseInt(el.dataset.decimals || "0", 10);
      const fmt = (v) => v.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });
      const t0 = performance.now();
      const dur = 1400;
      const tick = (now) => {
        const p = Math.min(1, (now - t0) / dur);
        el.textContent = fmt(to * (1 - Math.pow(1 - p, 3)));
        if (p < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }
  }

  /* ---------- sparklines  (<span class="spark" data-values="1,3,2,5">) ---------- */

  function sparklines(slide) {
    for (const el of slide.querySelectorAll(".spark[data-values]:empty")) {
      const v = el.dataset.values.split(",").map(Number);
      const w = el.clientWidth || 300, h = el.clientHeight || 70, pad = 7;
      const lo = Math.min(...v), hi = Math.max(...v);
      const pts = v.map((y, i) => [pad + (i * (w - 2 * pad)) / (v.length - 1), h - pad - ((y - lo) / (hi - lo || 1)) * (h - 2 * pad)]);
      const [lx, ly] = pts[pts.length - 1];
      el.innerHTML = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
        <polyline points="${pts.map((p) => p.join(",")).join(" ")}" fill="none" stroke="${Deck.colors.muted}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
        <circle cx="${lx}" cy="${ly}" r="6" fill="${Deck.colors.accent}" stroke="${Deck.colors.surface}" stroke-width="2"/></svg>`;
    }
  }

  /* ---------- before/after sliders ---------- */

  /* img-comparison-slider measures the pointer in unscaled pixels, which is
     off whenever Reveal scales the slide. Drive it from the scaled bounding
     box instead, intercepting events on the parent before the slider sees them. */
  function fixSliders() {
    for (const slider of document.querySelectorAll("img-comparison-slider")) {
      const host = slider.parentElement;
      let dragging = false;
      const move = (e) => {
        const r = slider.getBoundingClientRect();
        slider.value = Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100));
      };
      const inside = (e) => slider.contains(e.target);
      host.addEventListener("pointerdown", (e) => {
        if (!inside(e)) return;
        dragging = true;
        host.setPointerCapture(e.pointerId);
        move(e);
        e.stopPropagation();
        e.preventDefault();
      }, true);
      host.addEventListener("pointermove", (e) => { if (dragging) { move(e); e.stopPropagation(); } }, true);
      host.addEventListener("pointerup", () => (dragging = false), true);
      for (const type of ["mousedown", "mousemove", "touchstart", "touchmove"]) {
        host.addEventListener(type, (e) => { if (inside(e)) e.stopPropagation(); }, true);
      }
    }
  }

  /* ---------- typewriter  (fragment with class "typewriter") ---------- */

  /* Types the fragment's text out when it is revealed. Nested fragments are
     left alone so they can appear on a later click. */
  const typing = new Set();

  function typewrite(el) {
    if (!el._tw) {
      const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
        acceptNode: (n) => (n.parentElement.closest(".fragment") === el && n.textContent.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT),
      });
      el._tw = [];
      for (let n = walker.nextNode(); n; n = walker.nextNode()) el._tw.push([n, n.textContent]);
    }
    const parts = el._tw;
    const total = parts.reduce((s, [, t]) => s + t.length, 0);
    const caret = document.createElement("span");
    caret.className = "caret";
    const job = { el, done: false };
    const finish = () => {
      job.done = true;
      for (const [n, t] of parts) n.textContent = t;
      caret.remove();
      typing.delete(job);
    };
    job.finish = finish;
    typing.add(job);
    el.style.minHeight = `${el.offsetHeight}px`; // keep the final size while typing
    for (const [n] of parts) n.textContent = "";
    const t0 = performance.now(), cps = 110;
    const tick = (now) => {
      if (job.done) return;
      const typed = Math.floor(((now - t0) / 1000) * cps);
      let left = typed;
      for (const [n, t] of parts) {
        const k = Math.min(t.length, Math.max(0, left));
        n.textContent = t.slice(0, k);
        if (k < t.length) { n.after(caret); break; }
        left -= t.length;
      }
      if (typed >= total) finish();
      else requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  /* ---------- token highlighting ---------- */

  /* Deck.tokens(el, [[token, score 0-1], ...]): one chip per token, coloured by
     score once the container gets the "lit" class. A leading space marks a word start. */
  Deck.tokens = function (el, pairs) {
    el.replaceChildren();
    pairs.forEach(([tok, score], i) => {
      if (tok.startsWith(" ")) el.append(" ");
      const span = document.createElement("span");
      span.className = "tok" + (score > 0.55 ? " hi" : "");
      span.textContent = tok.trimStart();
      span.dataset.score = score.toFixed(2);
      span.style.setProperty("--s", score);
      span.style.setProperty("--i", i);
      el.append(span);
    });
  };

  /* ---------- QR codes  (<div class="qr" data-url="..." data-fallback="...">) ---------- */

  /* Encodes data-url, else the deck's own address (when served over http),
     else data-fallback. Fills a sibling .qr-url with the encoded link. */
  function qrCodes() {
    if (!window.qrcode) return;
    for (const el of document.querySelectorAll(".qr")) {
      const here = /^https?:$/.test(location.protocol) ? location.href.split("#")[0] : "";
      const url = el.dataset.url || here || el.dataset.fallback;
      if (!url) continue;
      const qr = qrcode(0, "M");
      qr.addData(url);
      qr.make();
      el.innerHTML = qr.createSvgTag({ cellSize: 10, margin: 0, scalable: true, alt: `QR code for ${url}` });
      const label = (el.closest(".qr-card") || el.parentElement).querySelector(".qr-url");
      if (label) label.textContent = url.replace(/^https?:\/\//, "").replace(/\/$/, "");
    }
  }

  /* ---------- lifecycle ---------- */

  function fragmentIndex(slide) {
    let i = -1;
    for (const f of slide.querySelectorAll(".fragment.visible")) i = Math.max(i, parseInt(f.dataset.fragmentIndex || "0", 10));
    return i;
  }

  function activate(slide, { animate = true } = {}) {
    if (!slide) return;
    // responsive.js can trigger the first draw before boot() runs
    if (!Deck.colors.series) readColors();
    for (const entry of registry) {
      const el = document.getElementById(entry.id);
      if (!el || !slide.contains(el)) continue;
      if (!entry.done) {
        entry.done = true;
        entry.stale = false;
        entry.init(el);
      } else if (entry.stale) {
        entry.stale = false;
        if (el.classList.contains("js-plotly-plot")) Plotly.Plots.resize(el);
        else if (window.echarts && echarts.getInstanceByDom(el)) echarts.getInstanceByDom(el).resize();
      }
      if (entry.step) entry.step(el, fragmentIndex(slide));
    }
    sparklines(slide);
    if (animate) countUp(slide);
  }

  /* Called by responsive.js after the canvas size changes. Charts on hidden
     slides have no size, so they are only marked and resized when shown. */
  Deck.resize = function () {
    for (const entry of registry) entry.stale = true;
    for (const el of document.querySelectorAll(".spark[data-values]")) el.replaceChildren();
    if (window.Reveal && Reveal.isReady()) activate(Reveal.getCurrentSlide(), { animate: false });
  };

  function boot() {
    readColors();
    fixSliders();
    qrCodes();
    const printing = /print-pdf/i.test(window.location.search);
    if (printing) {
      for (const s of Reveal.getSlides()) activate(s, { animate: false });
      return;
    }
    activate(Reveal.getCurrentSlide());
    Reveal.on("slidechanged", (e) => {
      for (const job of [...typing]) job.finish();
      activate(e.currentSlide);
    });
    Reveal.on("fragmentshown", (e) => {
      for (const f of e.fragments) if (f.classList.contains("typewriter")) typewrite(f);
      activate(Reveal.getCurrentSlide(), { animate: false });
    });
    Reveal.on("fragmenthidden", (e) => activate(Reveal.getCurrentSlide(), { animate: false }));
    Reveal.on("overviewshown", () => Reveal.getSlides().forEach((s) => activate(s, { animate: false })));
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (Reveal.isReady()) boot();
    else Reveal.on("ready", boot);
  });

  window.Deck = Deck;
})();
