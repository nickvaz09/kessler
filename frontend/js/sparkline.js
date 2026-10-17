/* Kessler - trend sparkline (dashboard).
   90x22 CSS px, devicePixelRatio-aware. A 1.5px rounded polyline in --k-spark
   over a faint area wash, with an endpoint dot r=2.5 colored by the row's risk
   level (caller resolves the --k-critical / --k-elevated / --k-text-2 token to
   a concrete color). No axes, no grid. values: plain numbers (log10 of Pc
   recommended for shape) - drawn normalized min..max. Colors are read from
   the theme on every call (KTheme.css when present), so callers redraw on
   KTheme.onChange. */
'use strict';

function drawSparkline(canvas, values, endpointColor) {
  const CSS_W = 90;
  const CSS_H = 22;
  const dpr = window.devicePixelRatio || 1;

  canvas.width = Math.round(CSS_W * dpr);
  canvas.height = Math.round(CSS_H * dpr);
  canvas.style.width = CSS_W + 'px';
  canvas.style.height = CSS_H + 'px';

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, CSS_W, CSS_H);

  if (!Array.isArray(values) || values.length === 0) return;

  const tok = (name) => {
    if (typeof KTheme !== 'undefined' && typeof KTheme.css === 'function') return KTheme.css(name);
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  };
  const lineColor = tok('--k-spark');
  const dotColor = (endpointColor && String(endpointColor).trim()) || tok('--k-text-2');

  /* inset so the r=2.5 endpoint dot never clips at the edges */
  const padX = 3;
  const padY = 3.5;
  const w = CSS_W - padX * 2;
  const h = CSS_H - padY * 2;

  let min = Math.min.apply(null, values);
  let max = Math.max.apply(null, values);
  if (min === max) { min -= 0.5; max += 0.5; }

  const n = values.length;
  function px(i) { return n === 1 ? padX + w / 2 : padX + (i / (n - 1)) * w; }
  function py(i) { return padY + h - ((values[i] - min) / (max - min)) * h; }

  if (n > 1) {
    /* faint area wash under the line */
    ctx.beginPath();
    ctx.moveTo(px(0), CSS_H);
    for (let i = 0; i < n; i++) ctx.lineTo(px(i), py(i));
    ctx.lineTo(px(n - 1), CSS_H);
    ctx.closePath();
    ctx.globalAlpha = 0.12;
    ctx.fillStyle = dotColor;
    ctx.fill();
    ctx.globalAlpha = 1;

    ctx.beginPath();
    ctx.moveTo(px(0), py(0));
    for (let i = 1; i < n; i++) ctx.lineTo(px(i), py(i));
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.stroke();
  }

  ctx.beginPath();
  ctx.arc(px(n - 1), py(n - 1), 2.5, 0, Math.PI * 2);
  ctx.fillStyle = dotColor;
  ctx.fill();
}
