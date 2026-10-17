/* Kessler - TCA timeline renderer (docs/CONTRACTS.md "TCA timeline").
   drawTimeline(canvas, tipEl, items, opts) - pure renderer over /conjunctions
   summary items: a now->+72 h axis (ticks every 12 h, NOW at the left edge)
   carrying the top 60 events by effective Pc among elevated+critical as 5 px
   diamonds at their TCA x-position, greedy-packed into <=6 lanes, hairline
   drop-lines to the axis, 9 px mono primary-name labels for the top 12 only.
   Hover shows a .chart-tip (pair, Pc, TCA, T-); a hit click invokes
   opts.onClick(item) - the dashboard wires it to KSim.open.

   Risk levels come from the API (risk_level) - never derived from Pc here;
   effective Pc (predicted_final_risk ?? pc_latest ?? 0) is used for ORDERING
   only, mirroring the backend's own sort key. Returns {plotted, eligible} so
   the caller can label the panel meta. DPR-aware, tokens only, no per-frame
   loop - redraws happen on caller feeds, hover moves, and resize.

   sentence case, Inter, round markers, a plain-language tooltip
   (odds + scientific Pc, UTC time, time remaining); colors come from theme
   tokens (KTheme.css when present) on every render, so a theme flip only
   needs the caller to redraw.

   Node (no DOM): exports the pure selection/packing helpers on
   globalThis.__KTIMELINE_TEST__ and degrades drawTimeline to a no-op -
   same headless-harness pattern as js/assistant.js. */
'use strict';

const drawTimeline = (() => {
  const HORIZON_MS = 72 * 3600 * 1000;   // axis span: now -> +72 h
  const TICK_MS = 12 * 3600 * 1000;      // axis ticks every 12 h
  const MAX_EVENTS = 60;                 // top-N cap by effective Pc
  const MAX_LABELS = 12;                 // name labels for the top 12 only
  const MAX_LANES = 6;                   // greedy packing ceiling
  const LANE_MIN_GAP = 12;               // css px between markers sharing a lane
  const DIAMOND_R = 3.5;                 // marker radius (round dot)
  const HIT_R2 = 8 * 8;                  // hover/click pick radius² (css px)

  /* --- pure helpers (exported to the node harness) -------------------------- */

  /* Ordering key only - the same effective-Pc definition the backend sorts
     by. never used to derive a risk level (that arrives as risk_level). */
  function effectivePc(it) {
    const p = it.predicted_final_risk;
    if (typeof p === 'number' && Number.isFinite(p)) return p;
    const q = it.pc_latest;
    if (typeof q === 'number' && Number.isFinite(q)) return q;
    return 0;
  }

  /* Elevated+critical items with a parseable TCA inside [now, now+horizon],
     sorted by effective Pc desc (ties: earlier TCA first), capped at maxN.
     -> [{it, tcaMs, effPc}] in Pc order (label set = first MAX_LABELS). */
  function selectTopEvents(items, nowMs, horizonMs, maxN) {
    const out = [];
    for (const it of (Array.isArray(items) ? items : [])) {
      if (!it) continue;
      if (it.risk_level !== 'elevated' && it.risk_level !== 'critical') continue;
      const tcaMs = Date.parse(it.tca);
      if (!Number.isFinite(tcaMs)) continue;
      if (tcaMs < nowMs || tcaMs > nowMs + horizonMs) continue;
      out.push({ it, tcaMs, effPc: effectivePc(it) });
    }
    out.sort((a, b) => (b.effPc - a.effPc) || (a.tcaMs - b.tcaMs));
    return { top: out.slice(0, maxN), eligible: out.length };
  }

  /* Greedy lane packing over x positions in ASCENDING order: each marker
     takes the lowest lane whose previous occupant sits at least minGap to
     the left; when every lane collides it falls back to the lane with the
     leftmost end (smallest overlap). Returns one lane index (0-based) per
     input x; never exceeds laneCount lanes. */
  function packLanes(xs, minGap, laneCount) {
    const ends = new Array(laneCount).fill(-Infinity);
    const lanes = new Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
      const x = xs[i];
      let lane = -1;
      for (let l = 0; l < laneCount; l++) {
        if (x - ends[l] >= minGap) { lane = l; break; }
      }
      if (lane < 0) {
        lane = 0;
        for (let l = 1; l < laneCount; l++) if (ends[l] < ends[lane]) lane = l;
      }
      ends[lane] = x;
      lanes[i] = lane;
    }
    return lanes;
  }

  /* --- headless harness hook (no DOM) --------------------------------------- */

  if (typeof window === 'undefined' || typeof document === 'undefined') {
    globalThis.__KTIMELINE_TEST__ = {
      effectivePc, selectTopEvents, packLanes,
      MAX_EVENTS, MAX_LABELS, MAX_LANES, LANE_MIN_GAP, HORIZON_MS,
    };
    return () => ({ plotted: 0, eligible: 0 });
  }

  /* --- browser renderer ------------------------------------------------------ */

  function cssVar(name) {
    if (typeof KTheme !== 'undefined' && typeof KTheme.css === 'function') return KTheme.css(name);
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function escHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, (ch) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[ch]));
  }

  /* local formatters (self-contained module - matches FMT's rendering) */
  function pcFmt(p) {
    if (p === null || p === undefined || Number.isNaN(p)) return '—';
    if (p <= 0 || p < 1e-10) return '<1e-10';
    let exp = Math.floor(Math.log10(p));
    let mant = p / Math.pow(10, exp);
    if (mant >= 9.95) { mant /= 10; exp += 1; }
    return `${mant.toFixed(1)}e${exp >= 0 ? '+' : ''}${exp}`;
  }

  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  function utcFmt(ms) {
    const d = new Date(ms);
    const p2 = (n) => String(n).padStart(2, '0');
    return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}, ${p2(d.getUTCHours())}:${p2(d.getUTCMinutes())} UTC`;
  }

  /* "in 5 h 12 min" / "passed" */
  function countdownFmt(ms) {
    const d = ms - Date.now();
    if (d < 0) return 'Passed';
    const m = Math.floor(d / 60000);
    if (m < 1) return 'In under a minute';
    if (m < 60) return `In ${m} min`;
    const h = Math.floor(m / 60);
    if (h < 24) return `In ${h} h${m % 60 ? ` ${m % 60} min` : ''}`;
    return `In ${Math.floor(h / 24)} d${h % 24 ? ` ${h % 24} h` : ''}`;
  }

  /* 1.2e-5 -> "1 in 83K" */
  function oddsFmt(p) {
    if (!(p > 0)) return '—';
    if (p < 1e-10) return '< 1 in 10B';
    const n = 1 / p;
    if (n < 1000) return `1 in ${Math.max(1, Math.round(n))}`;
    for (const [v, u] of [[1e9, 'B'], [1e6, 'M'], [1e3, 'K']]) {
      if (n >= v) {
        const x = n / v;
        return `1 in ${x < 10 ? x.toFixed(1).replace(/\.0$/, '') : Math.round(x)}${u}`;
      }
    }
    return '—';
  }

  function truncName(name, max) {
    const s = String(name || '');
    return s.length > max ? `${s.slice(0, max - 1)}…` : s;
  }

  /* one draw pass - plots into st.marks for the pick path */
  function render(canvas) {
    const st = canvas.__kTimeline;
    if (!st) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 40 || rect.height < 40) return;
    const dpr = window.devicePixelRatio || 1;
    const W = rect.width;
    const H = rect.height;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const sans = cssVar('--k-sans') || 'Inter, system-ui, sans-serif';
    const cGrid = cssVar('--k-grid');
    const cBorder = cssVar('--k-border');
    const cAxis = cssVar('--k-border-strong');
    const cText2 = cssVar('--k-text-2');
    const cText3 = cssVar('--k-text-3');
    const cSurface = cssVar('--k-surface') || 'transparent';
    const cElev = cssVar('--k-elevated');
    const cCrit = cssVar('--k-critical');

    const nowMs = st.nowMs;
    const padLeft = 12;
    const padRight = 16;
    const padTop = 10;
    const axisY = H - 22;                        // tick labels live below
    const plotW = Math.max(10, W - padLeft - padRight);
    const xPos = (tMs) => padLeft + ((tMs - nowMs) / HORIZON_MS) * plotW;

    /* axis + 12 h tick grid */
    ctx.lineWidth = 1;
    ctx.font = `500 11px ${sans}`;
    ctx.textBaseline = 'top';
    const nTicks = Math.round(HORIZON_MS / TICK_MS);          // 6 -> 7 marks
    for (let i = 0; i <= nTicks; i++) {
      const x = Math.round(xPos(nowMs + i * TICK_MS)) + 0.5;
      ctx.strokeStyle = cGrid;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, axisY);
      ctx.stroke();
      ctx.fillStyle = i === 0 ? cText2 : cText3;
      ctx.textAlign = i === 0 ? 'left' : i === nTicks ? 'right' : 'center';
      ctx.fillText(i === 0 ? 'Now' : `+${i * 12} h`, x - 0.5, axisY + 7);
    }
    ctx.strokeStyle = cAxis;
    ctx.beginPath();
    ctx.moveTo(padLeft, Math.round(axisY) + 0.5);
    ctx.lineTo(padLeft + plotW, Math.round(axisY) + 0.5);
    ctx.stroke();

    /* selection: top-N by Pc, then time order for packing */
    const sel = selectTopEvents(st.items, nowMs, HORIZON_MS, MAX_EVENTS);
    st.plotted = sel.top.length;
    st.eligible = sel.eligible;
    st.marks = [];
    if (!sel.top.length) {
      ctx.font = `13px ${sans}`;
      ctx.fillStyle = cText3;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('No elevated or critical close approaches in the next 72 hours.',
        padLeft + plotW / 2, (padTop + axisY) / 2);
      return;
    }
    const labelSet = new Set(sel.top.slice(0, MAX_LABELS).map((e) => e.it));
    const byTime = sel.top.slice().sort((a, b) => a.tcaMs - b.tcaMs);
    const lanes = packLanes(byTime.map((e) => xPos(e.tcaMs)), LANE_MIN_GAP, MAX_LANES);

    const laneH = Math.min(18, (axisY - padTop - 12) / MAX_LANES);
    const laneY = (lane) => axisY - 11 - lane * laneH;

    /* drop-lines first, then elevated diamonds, critical last (on top) */
    ctx.strokeStyle = cBorder;
    ctx.lineWidth = 1;
    for (let i = 0; i < byTime.length; i++) {
      const x = Math.round(xPos(byTime[i].tcaMs)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, laneY(lanes[i]) + DIAMOND_R + 1);
      ctx.lineTo(x, axisY);
      ctx.stroke();
    }

    ctx.font = `500 11px ${sans}`;
    ctx.textBaseline = 'middle';
    /* placed label rects - a label that would overwrite one already drawn
       is flipped to the other side of its marker, or dropped (the diamond
       and its hover tooltip still identify the event) */
    const placedLabels = [];
    /* a label may not cover another event's marker */
    const allMarks = byTime.map((e, i) => ({ x: xPos(e.tcaMs), y: laneY(lanes[i]) }));
    const markClash = (x0, x1, y) => allMarks.some((m) =>
      Math.abs(m.y - y) < 9 && m.x > x0 - 5 && m.x < x1 + 5);
    const labelClash = (x0, x1, y) => placedLabels.some((p) =>
      Math.abs(p.y - y) < 12 && x0 < p.x1 + 6 && x1 > p.x0 - 6);
    for (const pass of ['elevated', 'critical']) {
      ctx.fillStyle = pass === 'critical' ? cCrit : cElev;
      for (let i = 0; i < byTime.length; i++) {
        const e = byTime[i];
        if (e.it.risk_level !== pass) continue;
        const x = xPos(e.tcaMs);
        const y = laneY(lanes[i]);
        const hovered = st.hover === e.it;
        const r = hovered ? DIAMOND_R + 2 : DIAMOND_R;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        if (hovered) {
          ctx.save();
          ctx.globalAlpha = 0.25;
          ctx.beginPath();
          ctx.arc(x, y, r + 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }
        st.marks.push({ x, y, e });
        if (labelSet.has(e.it)) {
          const name = truncName(e.it.primary && e.it.primary.name, 16);
          const w = ctx.measureText(name).width;
          let align = x + 8 + w <= W - 2 ? 'left' : 'right';
          let x0 = align === 'left' ? x + 8 : x - 8 - w;
          if (labelClash(x0, x0 + w, y) && align === 'left'
              && x - 8 - w >= 2 && !labelClash(x - 8 - w, x - 8, y)) {
            align = 'right';
            x0 = x - 8 - w;
          }
          if (!labelClash(x0, x0 + w, y) && !markClash(x0, x0 + w, y)) {
            ctx.textAlign = align;
            /* knock-out halo keeps names legible over drop-lines */
            ctx.lineWidth = 4;
            ctx.lineJoin = 'round';
            ctx.strokeStyle = cSurface;
            ctx.strokeText(name, align === 'left' ? x + 8 : x - 8, y);
            ctx.fillStyle = cText2;
            ctx.fillText(name, align === 'left' ? x + 8 : x - 8, y);
            placedLabels.push({ x0, x1: x0 + w, y });
            ctx.fillStyle = pass === 'critical' ? cCrit : cElev;
          }
        }
      }
    }
  }

  /* nearest mark within the pick radius, or null */
  function pick(st, mx, my) {
    let best = null;
    let bd = HIT_R2;
    for (const m of st.marks) {
      const dx = m.x - mx;
      const dy = m.y - my;
      const d2 = dx * dx + dy * dy;
      if (d2 <= bd) { bd = d2; best = m; }
    }
    return best;
  }

  function updateTip(canvas, st, mark) {
    const tip = st.tip;
    if (!tip) return;
    if (!mark) {
      tip.style.display = 'none';
      return;
    }
    const it = mark.e.it;
    const pair = `${it.primary ? it.primary.name : '—'} and ${it.secondary ? it.secondary.name : '—'}`;
    tip.innerHTML =
      `<strong>${escHtml(pair)}</strong><br>` +
      `<span class="tip-label">Collision chance</span> ${escHtml(oddsFmt(mark.e.effPc))}` +
      ` <span class="tip-label">(${escHtml(pcFmt(mark.e.effPc))})</span><br>` +
      `<span class="tip-label">Closest approach</span> ${escHtml(utcFmt(mark.e.tcaMs))}<br>` +
      `${escHtml(countdownFmt(mark.e.tcaMs))} · <span class="tip-label">select to replay</span>`;
    tip.style.display = 'block';
    const box = tip.offsetParent || canvas.parentElement;
    const bw = box ? box.clientWidth : canvas.clientWidth;
    const bh = box ? box.clientHeight : canvas.clientHeight;
    let left = canvas.offsetLeft + mark.x + 12;
    if (left + tip.offsetWidth > bw - 4) left = canvas.offsetLeft + mark.x - tip.offsetWidth - 12;
    if (left < 2) left = 2;
    let top = canvas.offsetTop + mark.y + 10;
    if (top + tip.offsetHeight > bh - 2) top = bh - tip.offsetHeight - 2;
    if (top < 2) top = 2;
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }

  function bind(canvas) {
    const st = canvas.__kTimeline;
    canvas.addEventListener('mousemove', (ev) => {
      const rect = canvas.getBoundingClientRect();
      const mark = pick(st, ev.clientX - rect.left, ev.clientY - rect.top);
      const over = mark ? mark.e.it : null;
      canvas.style.cursor = over ? 'pointer' : 'default';
      if (over !== st.hover) {
        st.hover = over;
        render(canvas);
      }
      updateTip(canvas, st, mark);
    });
    canvas.addEventListener('mouseleave', () => {
      if (st.hover) {
        st.hover = null;
        render(canvas);
      }
      if (st.tip) st.tip.style.display = 'none';
      canvas.style.cursor = 'default';
    });
    canvas.addEventListener('click', (ev) => {
      const rect = canvas.getBoundingClientRect();
      const mark = pick(st, ev.clientX - rect.left, ev.clientY - rect.top);
      if (mark && st.opts && typeof st.opts.onClick === 'function') {
        st.opts.onClick(mark.e.it);
      }
    });
  }

  return function drawTimeline(canvas, tipEl, items, opts) {
    let st = canvas.__kTimeline;
    if (!st) {
      st = canvas.__kTimeline = {
        items: [], opts: null, tip: null, hover: null,
        marks: [], nowMs: Date.now(), plotted: 0, eligible: 0,
      };
      bind(canvas);
    }
    st.items = Array.isArray(items) ? items : [];
    st.opts = opts || {};
    st.tip = tipEl || null;
    st.nowMs = (st.opts.nowMs != null && Number.isFinite(st.opts.nowMs))
      ? st.opts.nowMs : Date.now();
    st.hover = null;
    if (st.tip) st.tip.style.display = 'none';
    render(canvas);
    return { plotted: st.plotted, eligible: st.eligible };
  };
})();
