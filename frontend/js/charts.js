/* Kessler shared chart library.
   drawLineChart(canvas, tipEl, opts) - contract in docs/CONTRACTS.md.
   opts = { series:[{label, color, points:[{t:msEpoch, y:Number}]}],
            yScale:'log'|'linear', yFormat:(v)=>string, xTicks:4, yTicks:4,
            thresholds:[{y, label, color}] }
   Optional extras (ignored safely by generic callers):
     tipFormat:(v)=>string        tooltip value formatting (falls back to yFormat)
     thresholds[i].labelColor     stronger ink for the right-edge label
     area:true                    soft flat fill under a single series
     emptyText:string             message when no series has points
     yFloor:Number                log scale only - the axis stops here; points
                                  below it sit on the floor line (their tooltip
                                  still shows the real value via tipFormat)
   Colors: a series/threshold color may be a concrete color or a token name
   ('--k-data-1'); token names are re-resolved on every draw, so charts
   follow the light/dark theme. Every drawn chart re-renders itself on
   KTheme.onChange and on window resize. DPR-aware: backing store sized from
   getBoundingClientRect * devicePixelRatio on every draw. Labels are Inter
   (tabular figures); gridlines are horizontal hairlines only. */
'use strict';

const drawLineChart = (() => {
  const live = new Set();   /* drawn canvases - re-rendered on theme flips */

  function cssVar(name) {
    if (typeof KTheme !== 'undefined' && KTheme.css) return KTheme.css(name);
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function resolveColor(c, fallbackVar) {
    if (typeof c === 'string' && c.slice(0, 2) === '--') return cssVar(c);
    return c || cssVar(fallbackVar || '--k-data-1');
  }

  function withAlpha(c, a) {
    const s = String(c || '').trim();
    let m = /^#([0-9a-f]{6})$/i.exec(s);
    if (m) {
      const n = parseInt(m[1], 16);
      return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
    }
    m = /^rgba?\(([^)]+)\)$/i.exec(s);
    if (m) {
      const p = m[1].split(',').map((x) => x.trim());
      return `rgba(${p[0]}, ${p[1]}, ${p[2]}, ${a})`;
    }
    return s;
  }

  function escHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, (ch) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[ch]));
  }

  function expFmt(v) {
    if (!Number.isFinite(v)) return '—';
    if (v === 0) return '0';
    const exp = Math.floor(Math.log10(Math.abs(v)));
    const mant = v / Math.pow(10, exp);
    return `${mant.toFixed(1)}e${exp >= 0 ? '+' : ''}${exp}`;
  }

  function niceStep(raw) {
    const pow = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / pow;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * pow;
  }

  function linFmt(v, step) {
    const a = Math.abs(v);
    if (a > 0 && (a < 1e-3 || a >= 1e6)) return expFmt(v);
    const dec = Math.max(0, -Math.floor(Math.log10(step)));
    return v.toLocaleString('en-US', { maximumFractionDigits: Math.min(dec, 6) });
  }

  function defaultTipFmt(v, isLog) {
    const a = Math.abs(v);
    if (isLog || (a > 0 && (a < 1e-3 || a >= 1e6))) return expFmt(v);
    return v.toLocaleString('en-US', { maximumFractionDigits: 2 });
  }

  const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  function tickTimeFmt(t, spanMs) {
    const d = new Date(t);
    const hm = `${String(d.getUTCHours()).padStart(2, '0')}:${String(d.getUTCMinutes()).padStart(2, '0')}`;
    const dm = `${d.getUTCDate()} ${MON[d.getUTCMonth()]}`;
    if (spanMs >= 3 * 86400000) return dm;
    if (spanMs >= 86400000) return `${dm} ${hm}`;
    return hm;
  }

  function tipTimeFmt(t) {
    const d = new Date(t);
    const hms = d.toISOString().slice(11, 19);
    return `${d.getUTCDate()} ${MON[d.getUTCMonth()]} ${d.getUTCFullYear()}, ${hms} UTC`;
  }

  function nearestPt(pts, t) {
    let best = null;
    let bd = Infinity;
    for (const p of pts) {
      const d = Math.abs(p.t - t);
      if (d < bd) { bd = d; best = p; }
    }
    return best;
  }

  function render(canvas) {
    const st = canvas.__kChart;
    if (!st || !st.opts) return;
    const opts = st.opts;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 20 || rect.height < 20) return;
    const dpr = window.devicePixelRatio || 1;
    const W = rect.width;
    const H = rect.height;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const sans = cssVar('--k-sans') || 'Inter, system-ui, sans-serif';
    const font = (px, wt) => `${wt || 450} ${px}px ${sans}`;
    const cGrid = cssVar('--k-grid');
    const cText3 = cssVar('--k-text-3');
    const cSurface = cssVar('--k-surface');
    const cCross = cssVar('--k-border-strong');
    const isLog = opts.yScale === 'log';

    const floor = isLog && Number.isFinite(opts.yFloor) && opts.yFloor > 0 ? opts.yFloor : null;
    const series = (opts.series || []).map((s, i) => ({
      label: s.label || '',
      color: resolveColor(s.color, i === 0 ? '--k-data-1' : '--k-data-2'),
      pts: (s.points || [])
        .filter((p) => p && Number.isFinite(p.t) && Number.isFinite(p.y) && (!isLog || p.y > 0))
        .map((p) => (floor && p.y < floor ? { t: p.t, y: floor, raw: p.y } : p))
        .sort((a, b) => a.t - b.t),
    })).filter((s) => s.pts.length);

    ctx.font = font(11);
    if (!series.length) {
      ctx.fillStyle = cText3;
      ctx.font = font(13);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(opts.emptyText || 'No data yet', W / 2, H / 2);
      st.layout = null;
      return;
    }

    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const s of series) {
      for (const p of s.pts) {
        if (p.t < xMin) xMin = p.t;
        if (p.t > xMax) xMax = p.t;
        if (p.y < yMin) yMin = p.y;
        if (p.y > yMax) yMax = p.y;
      }
    }
    const thresholds = (opts.thresholds || [])
      .filter((th) => th && Number.isFinite(th.y) && (!isLog || th.y > 0));
    for (const th of thresholds) {
      if (th.y < yMin) yMin = th.y;
      if (th.y > yMax) yMax = th.y;
    }
    if (xMin === xMax) { xMin -= 1800000; xMax += 1800000; }

    /* y domain + clean ticks (log scale: decade ticks, 1e-4 style) */
    const nY = Math.max(2, opts.yTicks || 4);
    const yTicks = [];
    let yFrac;
    if (isLog) {
      const lo = Math.floor(Math.log10(yMin));
      let hi = Math.ceil(Math.log10(yMax));
      if (hi <= lo) hi = lo + 1;
      const step = Math.max(1, Math.ceil((hi - lo) / nY));
      let lastE = null;
      for (let e = hi; e >= lo; e -= step) {
        const v = Math.pow(10, e);
        yTicks.push({ v, label: opts.yFormat ? opts.yFormat(v) : `1e${e}` });
        lastE = e;
      }
      /* the walk can overshoot lo when (hi-lo) % step != 0 - always tick the
         domain floor so the bottom band keeps a gridline and label */
      if (lastE !== lo) {
        const v = Math.pow(10, lo);
        yTicks.push({ v, label: opts.yFormat ? opts.yFormat(v) : `1e${lo}` });
      }
      yFrac = (v) => (Math.log10(v) - lo) / (hi - lo);
    } else {
      let span = yMax - yMin;
      if (span <= 0) span = Math.abs(yMax) || 1;
      const step = niceStep(span / nY);
      const lo = Math.floor(yMin / step) * step;
      let hi = Math.ceil(yMax / step) * step;
      if (hi <= lo) hi = lo + step;
      for (let v = lo; v <= hi + step / 2; v += step) {
        yTicks.push({ v, label: opts.yFormat ? opts.yFormat(v) : linFmt(v, step) });
      }
      yFrac = (v) => (v - lo) / (hi - lo);
    }

    /* layout - the y-label band and bottom x band live inside the canvas */
    let labelW = 0;
    for (const t of yTicks) labelW = Math.max(labelW, ctx.measureText(t.label).width);
    const padLeft = Math.ceil(labelW) + 14;
    const padRight = 10;
    const padTop = 10;
    const padBottom = 24;
    const plotW = Math.max(10, W - padLeft - padRight);
    const plotH = Math.max(10, H - padTop - padBottom);
    const xPos = (t) => padLeft + ((t - xMin) / (xMax - xMin)) * plotW;
    const yPos = (v) => padTop + (1 - yFrac(v)) * plotH;

    /* gridlines - horizontal hairlines only; the baseline one step stronger */
    ctx.lineWidth = 1;
    yTicks.forEach((t) => {
      const y = Math.round(yPos(t.v)) + 0.5;
      ctx.strokeStyle = cGrid;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    /* axis labels - 11px Inter, --k-text-3, tabular figures */
    /* never crowd the time axis: at most one label per ~96 px */
    const nX = Math.max(2, Math.min(opts.xTicks || 4, Math.floor(plotW / 96) + 1));
    const xTickTs = [];
    for (let i = 0; i < nX; i++) xTickTs.push(xMin + ((xMax - xMin) * i) / (nX - 1));
    ctx.fillStyle = cText3;
    ctx.font = font(11);
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    for (const t of yTicks) ctx.fillText(t.label, padLeft - 8, yPos(t.v));
    ctx.textBaseline = 'top';
    const spanMs = xMax - xMin;
    xTickTs.forEach((t, i) => {
      ctx.textAlign = i === 0 ? 'left' : i === nX - 1 ? 'right' : 'center';
      ctx.fillText(tickTimeFmt(t, spanMs), xPos(t), padTop + plotH + 8);
    });

    /* threshold lines - dashed hairlines with a small right-edge label */
    for (const th of thresholds) {
      const y = yPos(th.y);
      if (y < padTop - 0.5 || y > padTop + plotH + 0.5) continue;
      const yr = Math.round(y) + 0.5;
      ctx.strokeStyle = resolveColor(th.color, '--k-text-3');
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(padLeft, yr);
      ctx.lineTo(padLeft + plotW, yr);
      ctx.stroke();
      ctx.setLineDash([]);
      if (th.label) {
        ctx.font = font(10.5, 550);
        ctx.fillStyle = resolveColor(th.labelColor || th.color, '--k-text-3');
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        /* flip the label under the line when the threshold hugs the top edge */
        const labelY = (yr - 14 < padTop) ? yr + 13 : yr - 5;
        ctx.fillText(th.label, padLeft + 6, labelY);
        ctx.font = font(11);
      }
    }

    /* optional soft area under a single series (flat alpha, not a gradient) */
    if (opts.area && series.length === 1 && series[0].pts.length > 1) {
      const s = series[0];
      ctx.beginPath();
      s.pts.forEach((p, i) => {
        if (i === 0) ctx.moveTo(xPos(p.t), yPos(p.y));
        else ctx.lineTo(xPos(p.t), yPos(p.y));
      });
      ctx.lineTo(xPos(s.pts[s.pts.length - 1].t), padTop + plotH);
      ctx.lineTo(xPos(s.pts[0].t), padTop + plotH);
      ctx.closePath();
      ctx.fillStyle = withAlpha(s.color, 0.10);
      ctx.fill();
    }

    /* series - 2px, round join/cap; endpoint dot r=3.5 ringed in surface */
    for (const s of series) {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      s.pts.forEach((p, i) => {
        const x = xPos(p.t);
        const y = yPos(p.y);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      /* sparse series: small markers so single CDMs read as data points */
      if (s.pts.length <= 24) {
        for (const p of s.pts) {
          ctx.beginPath();
          ctx.arc(xPos(p.t), yPos(p.y), 2, 0, Math.PI * 2);
          ctx.fillStyle = s.color;
          ctx.fill();
        }
      }
      const last = s.pts[s.pts.length - 1];
      const lx = xPos(last.t);
      const ly = yPos(last.y);
      ctx.beginPath();
      ctx.arc(lx, ly, 5.5, 0, Math.PI * 2);
      ctx.fillStyle = cSurface;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(lx, ly, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.fill();
    }

    /* hover crosshair + nearest-point markers */
    if (st.hover) {
      const hx = xPos(st.hover.t);
      if (hx >= padLeft - 0.5 && hx <= padLeft + plotW + 0.5) {
        ctx.strokeStyle = cCross;
        ctx.lineWidth = 1;
        const xr = Math.round(hx) + 0.5;
        ctx.beginPath();
        ctx.moveTo(xr, padTop);
        ctx.lineTo(xr, padTop + plotH);
        ctx.stroke();
        for (const s of series) {
          const p = nearestPt(s.pts, st.hover.t);
          if (!p) continue;
          ctx.beginPath();
          ctx.arc(xPos(p.t), yPos(p.y), 5, 0, Math.PI * 2);
          ctx.fillStyle = cSurface;
          ctx.fill();
          ctx.beginPath();
          ctx.arc(xPos(p.t), yPos(p.y), 3.5, 0, Math.PI * 2);
          ctx.fillStyle = s.color;
          ctx.fill();
        }
      }
    }

    st.layout = {
      series,
      xMin,
      xMax,
      padLeft,
      padTop,
      plotW,
      plotH,
      fmtVal: opts.tipFormat || opts.yFormat || ((v) => defaultTipFmt(v, isLog)),
    };
  }

  function updateTip(canvas, st) {
    const tip = st.tip;
    if (!tip) return;
    const L = st.layout;
    if (!L || !st.hover) {
      tip.style.display = 'none';
      return;
    }
    const rows = [`<span class="tip-label">${tipTimeFmt(st.hover.t)}</span>`];
    for (const s of L.series) {
      const p = nearestPt(s.pts, st.hover.t);
      if (!p) continue;
      const val = escHtml(L.fmtVal(p.raw !== undefined ? p.raw : p.y));
      const sw = `<i style="display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:${escHtml(s.color)}"></i>`;
      rows.push(s.label
        ? `${sw}<span class="tip-label">${escHtml(s.label)}</span> <b>${val}</b>`
        : `<b>${val}</b>`);
    }
    tip.innerHTML = rows.join('<br>');
    tip.style.display = 'block';
    /* position inside the chart-box; flip near the right edge */
    const box = tip.offsetParent || canvas.parentElement;
    const bw = box ? box.clientWidth : canvas.clientWidth;
    const bh = box ? box.clientHeight : canvas.clientHeight;
    const hx = canvas.offsetLeft + L.padLeft
      + ((st.hover.t - L.xMin) / (L.xMax - L.xMin)) * L.plotW;
    let left = hx + 14;
    if (left + tip.offsetWidth > bw - 4) left = hx - tip.offsetWidth - 14;
    if (left < 2) left = 2;
    let top = canvas.offsetTop + st.hover.my + 12;
    if (top + tip.offsetHeight > bh - 2) top = bh - tip.offsetHeight - 2;
    if (top < 2) top = 2;
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }

  function bind(canvas) {
    const st = canvas.__kChart;
    const onMove = (e) => {
      const L = st.layout;
      if (!L) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const tCur = L.xMin + ((mx - L.padLeft) / L.plotW) * (L.xMax - L.xMin);
      let bestT = null;
      let bd = Infinity;
      for (const s of L.series) {
        for (const p of s.pts) {
          const d = Math.abs(p.t - tCur);
          if (d < bd) { bd = d; bestT = p.t; }
        }
      }
      if (bestT === null) return;
      st.hover = { t: bestT, my };
      render(canvas);
      updateTip(canvas, st);
    };
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerdown', onMove);
    canvas.addEventListener('pointerleave', () => {
      st.hover = null;
      if (st.tip) st.tip.style.display = 'none';
      render(canvas);
    });
    const onResize = () => {
      if (!canvas.isConnected) {
        window.removeEventListener('resize', onResize);
        live.delete(canvas);
        return;
      }
      render(canvas);
      updateTip(canvas, st);
    };
    window.addEventListener('resize', onResize);
    live.add(canvas);
  }

  if (typeof KTheme !== 'undefined' && KTheme.onChange) {
    KTheme.onChange(() => {
      for (const c of Array.from(live)) {
        if (!c.isConnected) { live.delete(c); continue; }
        render(c);
      }
    });
  }

  return function drawLineChart(canvas, tipEl, opts) {
    let st = canvas.__kChart;
    if (!st) {
      st = canvas.__kChart = { hover: null, layout: null, tip: null, opts: null };
      bind(canvas);
    }
    st.tip = tipEl || null;
    st.opts = opts || {};
    render(canvas);
    updateTip(canvas, st);
  };
})();
