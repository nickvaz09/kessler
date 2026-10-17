/* Kessler sky view - polar azimuth/elevation plot of one ground-station pass.
   drawSkyView(canvas, pass) - contract in docs/CONTRACTS.md ("v7 additions").
     pass: one /objects/{norad}/passes item -
       {aos, los, tca, max_el_deg, aos_az_deg, los_az_deg, duration_s,
        track:[[t_offset_s, az_deg, el_deg], ...]}   (10 s samples, AOS -> LOS)
   Geometry: zenith (el 90°) at the center, horizon (el 0°) at the outer
   ring - r = (90 - el)/90 · R. North up, east right, azimuth clockwise:
   x = cx + r·sin(az), y = cy - r·cos(az). Elevation rings at 0/30/60°
   (horizon in --k-border-strong, inner rings hairline --k-grid), N/E/S/W
   cardinals, pass polyline in --k-data-1 (2px) with an arrowhead at the LOS
   end, ringed max-elevation dot labeled '<el>°', AOS/LOS azimuth labels in
   9px mono. Static - one draw per call, DPR-aware like charts.js (backing
   store sized from getBoundingClientRect x devicePixelRatio). All colors
   from --k-* tokens; text labels wear text tokens only.
   Returns true when a track polyline was drawn, false when the base grid
   alone rendered (missing/short track - older backends omit `track`). */
'use strict';

const drawSkyView = (() => {
  const RAD = Math.PI / 180;
  const MARGIN = 22;        /* CSS px kept outside the horizon ring for labels */
  const ARROW_PX = 7;       /* LOS arrowhead barb length */

  const live = new Map();   /* canvas -> pass, re-drawn on theme flips */

  function cssVar(name) {
    if (typeof KTheme !== 'undefined' && KTheme.css) return KTheme.css(name);
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* az/el (deg) -> canvas point. Pure - exercised by the scratchpad harness
     through drawSkyView.__TEST__. */
  function polar(azDeg, elDeg, cx, cy, R) {
    const r = ((90 - elDeg) / 90) * R;
    const a = azDeg * RAD;
    return { x: cx + r * Math.sin(a), y: cy - r * Math.cos(a), r };
  }

  function draw(canvas, pass) {
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(rect.width, 40);
    const h = Math.max(rect.height, 40);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const cGrid = cssVar('--k-grid');
    const cHorizon = cssVar('--k-border-strong');
    const cPath = cssVar('--k-data-1');
    const cSurface = cssVar('--k-surface');
    const cText2 = cssVar('--k-text-2');
    const cText3 = cssVar('--k-text-3');
    const sans = cssVar('--k-sans') || 'Inter, system-ui, sans-serif';
    const mono = `450 11px ${sans}`;
    live.set(canvas, pass);

    const cx = w / 2;
    const cy = h / 2;
    const R = Math.min(w, h) / 2 - MARGIN;
    if (R <= 0) return false;

    /* cardinal spokes (hairline, under everything) */
    ctx.strokeStyle = cGrid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R);
    ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
    ctx.stroke();

    /* elevation rings: 60/30 hairline, horizon strong - all solid */
    for (const el of [60, 30]) {
      ctx.beginPath();
      ctx.arc(cx, cy, ((90 - el) / 90) * R, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.strokeStyle = cHorizon;
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.stroke();

    /* ring labels along the SE diagonal (az 135°, clear of the cardinals) */
    ctx.font = mono;
    ctx.fillStyle = cText3;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    for (const el of [0, 30, 60]) {
      const p = polar(135, el, cx, cy, R);
      ctx.fillText(`${el}°`, p.x + 2, p.y + 1);
    }

    /* cardinals outside the horizon ring - N carries the orientation */
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = cText2;
    ctx.font = `600 11px ${sans}`;
    ctx.fillText('N', cx, cy - R - 9);
    ctx.fillStyle = cText3;
    ctx.fillText('E', cx + R + 9, cy);
    ctx.fillText('S', cx, cy + R + 9);
    ctx.fillText('W', cx - R - 9, cy);
    ctx.font = mono;

    /* --- pass geometry ---------------------------------------------------- */

    const track = pass && Array.isArray(pass.track) ? pass.track : [];
    const pts = [];
    for (const s of track) {
      if (!Array.isArray(s) || s.length < 3) continue;
      const az = Number(s[1]);
      const el = Number(s[2]);
      if (!Number.isFinite(az) || !Number.isFinite(el)) continue;
      pts.push(polar(az, Math.max(el, 0), cx, cy, R));
    }
    if (pts.length < 2) return false;

    /* polyline - 2px, round joins (charts.js series idiom) */
    ctx.strokeStyle = cPath;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    pts.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); });
    ctx.stroke();

    /* arrowhead at the LOS end - barbs against the direction of travel */
    const end = pts[pts.length - 1];
    const prev = pts[pts.length - 2];
    const heading = Math.atan2(end.y - prev.y, end.x - prev.x);
    ctx.beginPath();
    for (const side of [-1, 1]) {
      const a = heading + Math.PI + side * (28 * RAD);
      ctx.moveTo(end.x, end.y);
      ctx.lineTo(end.x + ARROW_PX * Math.cos(a), end.y + ARROW_PX * Math.sin(a));
    }
    ctx.stroke();

    /* max-elevation dot - r=3 with 2px surface ring (charts.js idiom) */
    let maxI = 0;
    for (let i = 1; i < track.length; i++) {
      if (Number(track[i][2]) > Number(track[maxI][2])) maxI = i;
    }
    const maxPt = pts[Math.min(maxI, pts.length - 1)];
    ctx.beginPath();
    ctx.arc(maxPt.x, maxPt.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = cSurface;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(maxPt.x, maxPt.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = cPath;
    ctx.fill();
    const maxEl = Number.isFinite(Number(pass.max_el_deg))
      ? Number(pass.max_el_deg)
      : Number(track[maxI][2]);
    ctx.fillStyle = cssVar('--k-text');
    ctx.font = `600 12px ${sans}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`${Math.round(maxEl)}° high`, maxPt.x + 7, maxPt.y - 4);
    ctx.font = mono;

    /* AOS/LOS azimuth labels - 9px mono, nudged radially outward from the
       endpoints so they sit off the path */
    ctx.fillStyle = cText3;
    const ends = [
      { p: pts[0], az: pass.aos_az_deg, tag: 'Rises' },
      { p: end, az: pass.los_az_deg, tag: 'Sets' },
    ];
    for (const e of ends) {
      const dx = e.p.x - cx;
      const dy = e.p.y - cy;
      const len = Math.hypot(dx, dy) || 1;
      const lx = e.p.x + (dx / len) * 9;
      const ly = e.p.y + (dy / len) * 9;
      ctx.textAlign = dx < 0 ? 'right' : 'left';
      ctx.textBaseline = dy < 0 ? 'bottom' : 'top';
      const az = Number.isFinite(Number(e.az)) ? ` ${Math.round(Number(e.az))}°` : '';
      ctx.fillText(`${e.tag}${az}`, lx, ly);
    }
    return true;
  }

  if (typeof KTheme !== 'undefined' && KTheme.onChange) {
    KTheme.onChange(() => {
      for (const [c, p] of Array.from(live)) {
        if (!c.isConnected) { live.delete(c); continue; }
        draw(c, p);
      }
    });
  }

  draw.__TEST__ = { polar };
  return draw;
})();
