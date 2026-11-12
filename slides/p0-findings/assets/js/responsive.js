/* Responsive canvas: fit Reveal's logical slide size to the window instead of
   letterboxing a fixed 1920x1080.

   - Landscape up to 16:9 keeps the 1920 width (so type scales consistently)
     and grows taller with the window, down to 4:3.
   - Wider than 16:9 keeps the 1080 height and grows wider, up to 21:9.
   - Portrait switches to a 1080-wide canvas, up to 9:21, and adds
     `body.portrait` so style.css can stack columns and grids.

   Beyond those limits Reveal letterboxes as usual. Slides are flex columns
   (see style.css) that share any extra height between their blocks, and the
   canvas size is exposed as --slide-w / --slide-h for sizing charts.
   Print/PDF export keeps the fixed 1920x1080 layout. */
(function () {
  "use strict";

  const LONG = 1920, SHORT = 1080;

  function canvasSize(vw, vh) {
    const a = vw / vh;
    if (a >= 16 / 9) return [Math.round(Math.min(SHORT * a, (SHORT * 21) / 9)), SHORT];
    if (a >= 1) return [LONG, Math.round(Math.min(LONG / a, (LONG * 3) / 4))];
    return [SHORT, Math.round(Math.min(Math.max(SHORT / a, (SHORT * 4) / 3), (SHORT * 21) / 9))];
  }

  function fit() {
    const [w, h] = canvasSize(window.innerWidth, window.innerHeight);
    document.body.classList.toggle("portrait", h > w);
    document.documentElement.style.setProperty("--slide-w", `${w}px`);
    document.documentElement.style.setProperty("--slide-h", `${h}px`);
    const config = Reveal.getConfig();
    if (config.width !== w || config.height !== h) {
      Reveal.configure({ width: w, height: h });
      if (window.Deck) Deck.resize();
    }
  }

  function boot() {
    if (/print-pdf/i.test(window.location.search)) return;
    // Reveal switches narrow screens to its scroll view by default; keep
    // regular slides so phones get the reflowed portrait layout instead.
    Reveal.configure({ display: "flex", scrollActivationWidth: 0 });
    fit();
    let pending;
    window.addEventListener("resize", () => {
      cancelAnimationFrame(pending);
      pending = requestAnimationFrame(fit);
    });
  }

  // Registered before deck.js's handler, so charts are first drawn at the final size.
  document.addEventListener("DOMContentLoaded", () => {
    if (Reveal.isReady()) boot();
    else Reveal.on("ready", boot);
  });
})();
