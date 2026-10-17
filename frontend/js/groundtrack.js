/* Kessler ground-track map - equirectangular situational display.
   drawGroundTrack(canvas, data, opts) - contract in docs/CONTRACTS.md.
     data: /objects/{id}/track response -
       {epoch, period_min, samples:[[lon_deg, lat_deg, alt_km]],
        subpoint:{lat_deg, lon_deg, alt_km, velocity_kms}}
     opts: {coastlines: {lines:[[[lon,lat],...],...]}   fetched once by the caller
            label: string                            subpoint annotation name
            date: Date}                              terminator instant (default: data.epoch)
   Static map - no pointer interaction. The caller redraws on data refresh;
   the module re-renders itself on window resize. The one-period track is
   age-faded (alpha 1.0 at the subpoint down to 0.25 at the oldest sample)
   and runs continuously through antimeridian crossings by drawing the
   crossing segment twice with wrapped longitudes under a plot-rect clip;
   coastlines get the same wrap treatment. DPR-aware like charts.js
   (backing store sized from getBoundingClientRect * devicePixelRatio on every
   draw). All colors from --k-* tokens except the contract-mandated flat
   rgba(0,0,0,0.30) night region (a lighter shade in the light theme).
   rounded map card on --k-surface-2, Inter labels, re-renders on
   KTheme.onChange.

   Terminator accuracy: solar declination from the Cooper cosine approximation
   (±0.5°) and equation of time from the three-term sinusoid fit (±0.5 min),
   so the subsolar point lands within about 1° of truth - well inside the 2°
   shading grid used here; the day/night boundary is honest at map scale. */
'use strict';

const drawGroundTrack = (() => {
  const RAD = Math.PI / 180;
  const NIGHT_FILL = 'rgba(0, 0, 0, 0.30)'; /* flat night shade, dark theme */
  const NIGHT_FILL_LIGHT = 'rgba(0, 0, 40, 0.07)'; /* same shade on a light map */
  const SHADE_STEP_DEG = 1;                 /* night-mask sampling resolution */
  const FADE_STEPS = 24;                    /* constant-alpha chunks along the trail */
  const FADE_MIN = 0.25;                    /* trail alpha at the oldest (final) sample */

  const live = new Set();

  function cssVar(name) {
    if (typeof KTheme !== 'undefined' && KTheme.css) return KTheme.css(name);
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function isLight() {
    return document.documentElement.getAttribute('data-theme') === 'light';
  }

  function fmtLat(v, digits) {
    return `${Math.abs(v).toFixed(digits)}°${v < 0 ? 'S' : 'N'}`;
  }

  function fmtLon(v, digits) {
    return `${Math.abs(v).toFixed(digits)}°${v < 0 ? 'W' : 'E'}`;
  }

  /* Subsolar point at a UTC instant: lat = solar declination, lon = where
     apparent solar time is 12:00 (Greenwich noon offset by the equation of
     time, then 15°/h westward). */
  function subsolarPoint(date) {
    const year0 = Date.UTC(date.getUTCFullYear(), 0, 0);
    const doy = (date.getTime() - year0) / 86400000; /* fractional day of year */
    const declDeg = -23.44 * Math.cos(((2 * Math.PI) / 365.24) * (doy + 10));
    const b = (2 * Math.PI * (doy - 81)) / 364;
    const eotMin = 9.87 * Math.sin(2 * b) - 7.53 * Math.cos(b) - 1.5 * Math.sin(b);
    const utcH = date.getUTCHours() + date.getUTCMinutes() / 60 + date.getUTCSeconds() / 3600;
    let lonDeg = -15 * (utcH + eotMin / 60 - 12);
    lonDeg = ((lonDeg + 540) % 360) - 180;
    return { latDeg: declDeg, lonDeg };
  }

  function render(canvas) {
    const st = canvas.__kTrack;
    if (!st || !st.data) return;
    const data = st.data;
    const opts = st.opts || {};
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

    /* layout - lat label band left, lon label band bottom (charts.js idiom) */
    const narrow = W < 520;
    const padLeft = narrow ? 30 : 36;
    const padRight = 6;
    const padTop = 6;
    const padBottom = 22;
    const plotW = Math.max(10, W - padLeft - padRight);
    const plotH = Math.max(10, H - padTop - padBottom);
    const X = (lon) => padLeft + ((lon + 180) / 360) * plotW;
    const Y = (lat) => padTop + ((90 - lat) / 180) * plotH;

    /* map base - one step off the panel surface, rounded like a card */
    const rr = 10;
    const frame = new Path2D();
    frame.moveTo(padLeft + rr, padTop);
    frame.arcTo(padLeft + plotW, padTop, padLeft + plotW, padTop + plotH, rr);
    frame.arcTo(padLeft + plotW, padTop + plotH, padLeft, padTop + plotH, rr);
    frame.arcTo(padLeft, padTop + plotH, padLeft, padTop, rr);
    frame.arcTo(padLeft, padTop, padLeft + plotW, padTop, rr);
    frame.closePath();
    ctx.save();
    ctx.clip(frame);
    ctx.fillStyle = cssVar('--k-surface-2');
    ctx.fillRect(padLeft, padTop, plotW, plotH);

    /* graticule every 30°; equator one step brighter */
    ctx.lineWidth = 1;
    ctx.strokeStyle = cssVar('--k-grid');
    ctx.beginPath();
    for (let lon = -180; lon <= 180; lon += 30) {
      const x = Math.round(X(lon)) + 0.5;
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
    }
    for (let lat = -90; lat <= 90; lat += 30) {
      if (lat === 0) continue;
      const y = Math.round(Y(lat)) + 0.5;
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
    }
    ctx.stroke();
    ctx.strokeStyle = cssVar('--k-border-strong');
    ctx.beginPath();
    const yEq = Math.round(Y(0)) + 0.5;
    ctx.moveTo(padLeft, yEq);
    ctx.lineTo(padLeft + plotW, yEq);
    ctx.stroke();

    /* coastlines - one batched Path2D. Antimeridian crossings are drawn
       twice with wrapped longitudes (prev -> lon±360, prev∓360 -> lon) so
       the outline runs continuously through both map edges instead of
       splitting with a notch; the clip rect cuts the overshoot. */
    const lines = (opts.coastlines && opts.coastlines.lines) || [];
    if (lines.length) {
      const coast = new Path2D();
      for (const line of lines) {
        let pen = false;
        let prevLon = null;
        let prevLat = null;
        for (const pt of line) {
          const lon = pt[0];
          const lat = pt[1];
          if (!Number.isFinite(lon) || !Number.isFinite(lat)) { pen = false; continue; }
          if (!pen) {
            coast.moveTo(X(lon), Y(lat));
            pen = true;
          } else if (Math.abs(lon - prevLon) > 180) {
            const off = lon > prevLon ? -360 : 360;
            coast.lineTo(X(lon + off), Y(lat));
            coast.moveTo(X(prevLon - off), Y(prevLat));
            coast.lineTo(X(lon), Y(lat));
          } else {
            coast.lineTo(X(lon), Y(lat));
          }
          prevLon = lon;
          prevLat = lat;
        }
      }
      ctx.save();
      ctx.beginPath();
      ctx.rect(padLeft, padTop, plotW, plotH);
      ctx.clip();
      ctx.strokeStyle = cssVar('--k-spark');
      ctx.lineWidth = 1;
      ctx.lineJoin = 'round';
      ctx.stroke(coast);
      ctx.restore();
    }

    /* night region - sample a lat/lon grid at SHADE_STEP_DEG, merge each
       row's dark cells into horizontal run rects in a single Path2D, fill
       once with the flat shade (single fill: overlaps never double-darken;
       +0.5 px row overlap kills antialiasing seams). Shading over the
       coastlines dims them honestly; the track is drawn above. */
    const when = (opts.date instanceof Date && !Number.isNaN(opts.date.getTime()))
      ? opts.date
      : (data.epoch && !Number.isNaN(Date.parse(data.epoch)) ? new Date(data.epoch) : new Date());
    const sun = subsolarPoint(when);
    const sinD = Math.sin(sun.latDeg * RAD);
    const cosD = Math.cos(sun.latDeg * RAD);
    const night = new Path2D();
    for (let lat = -90; lat < 90; lat += SHADE_STEP_DEG) {
      const phi = (lat + SHADE_STEP_DEG / 2) * RAD;
      const sinP = Math.sin(phi);
      const cosP = Math.cos(phi);
      const y = Y(lat + SHADE_STEP_DEG);
      const h = Math.min(Y(lat) - y + 0.5, padTop + plotH - y);
      let run = null;
      for (let lon = -180; lon <= 180; lon += SHADE_STEP_DEG) {
        /* sun elevation < 0 at the cell centre -> night */
        const dark = lon < 180
          && sinP * sinD + cosP * cosD
             * Math.cos((lon + SHADE_STEP_DEG / 2 - sun.lonDeg) * RAD) < 0;
        if (dark && run === null) {
          run = lon;
        } else if (!dark && run !== null) {
          night.rect(X(run), y, X(lon) - X(run), h);
          run = null;
        }
      }
    }
    ctx.fillStyle = isLight() ? NIGHT_FILL_LIGHT : NIGHT_FILL;
    ctx.fill(night);
    ctx.restore();

    /* ground track - 1.5 px polyline age-faded along its one period:
       FADE_STEPS constant-alpha chunks stepping linearly from 1.0 at the
       subpoint (t=0, track start) to FADE_MIN at the oldest (final) sample.
       Functional time-along-track encoding, single hue. Antimeridian
       crossings are drawn twice with wrapped longitudes - (prev -> lon±360)
       and (prev∓360 -> lon) at that position's trail alpha - so the trail
       runs continuously off one edge and onto the other; the clip rect cuts
       the overshoot at the map frame. The trail ends with a 5 px hairline
       tick perpendicular to the last segment, at the trail's final alpha. */
    const samples = Array.isArray(data.samples) ? data.samples : [];
    if (samples.length > 1) {
      const span = samples.length - 1;
      const alphaAt = (k) => (FADE_STEPS > 1
        ? 1 - (k / (FADE_STEPS - 1)) * (1 - FADE_MIN)
        : 1);
      ctx.save();
      ctx.beginPath();
      ctx.rect(padLeft, padTop, plotW, plotH);
      ctx.clip();
      ctx.strokeStyle = cssVar('--k-data-1');
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      let open = false;
      let chunk = -1;
      let prev = null; /* previous valid [lon, lat] */
      let dirX = 0;    /* direction of the last drawn segment, px space */
      let dirY = 0;
      for (let i = 0; i < samples.length; i++) {
        const lon = samples[i][0];
        const lat = samples[i][1];
        if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
          if (open) { ctx.stroke(); open = false; }
          prev = null;
          continue;
        }
        if (prev === null) { prev = [lon, lat]; continue; }
        const k = Math.min(FADE_STEPS - 1, Math.floor(((i - 1) / span) * FADE_STEPS));
        if (!open || k !== chunk) {
          if (open) ctx.stroke();
          ctx.globalAlpha = alphaAt(k);
          ctx.beginPath();
          ctx.moveTo(X(prev[0]), Y(prev[1]));
          open = true;
          chunk = k;
        }
        if (Math.abs(lon - prev[0]) > 180) {
          /* continuous crossing - same segment drawn on both map edges */
          const off = lon > prev[0] ? -360 : 360;
          ctx.lineTo(X(lon + off), Y(lat));
          ctx.moveTo(X(prev[0] - off), Y(prev[1]));
          ctx.lineTo(X(lon), Y(lat));
          dirX = X(lon) - X(prev[0] - off);
          dirY = Y(lat) - Y(prev[1]);
        } else {
          ctx.lineTo(X(lon), Y(lat));
          dirX = X(lon) - X(prev[0]);
          dirY = Y(lat) - Y(prev[1]);
        }
        prev = [lon, lat];
      }
      if (open) ctx.stroke();
      const dLen = Math.hypot(dirX, dirY);
      if (prev !== null && dLen > 0) {
        const nx = -dirY / dLen;
        const ny = dirX / dLen;
        const ex = X(prev[0]);
        const ey = Y(prev[1]);
        ctx.globalAlpha = alphaAt(chunk);
        ctx.lineWidth = 1;
        ctx.lineCap = 'butt';
        ctx.beginPath();
        ctx.moveTo(ex - nx * 2.5, ey - ny * 2.5);
        ctx.lineTo(ex + nx * 2.5, ey + ny * 2.5);
        ctx.stroke();
      }
      ctx.restore();
      ctx.globalAlpha = 1;
    }

    /* subpoint - ringed dot + direct label (text in text-token grays) */
    ctx.font = `600 12px ${sans}`;
    const sp = data.subpoint;
    if (sp && Number.isFinite(sp.lat_deg) && Number.isFinite(sp.lon_deg)) {
      const sx = X(sp.lon_deg);
      const sy = Y(sp.lat_deg);
      const cAcc = cssVar('--k-data-1');
      ctx.beginPath();
      ctx.arc(sx, sy, 12, 0, Math.PI * 2);
      ctx.globalAlpha = 0.18;
      ctx.fillStyle = cAcc;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(sx, sy, 6, 0, Math.PI * 2);
      ctx.fillStyle = cssVar('--k-surface');
      ctx.fill();
      ctx.beginPath();
      ctx.arc(sx, sy, 4, 0, Math.PI * 2);
      ctx.fillStyle = cAcc;
      ctx.fill();
      const text = String(opts.label || '').trim() || `${fmtLat(sp.lat_deg, 1)} ${fmtLon(sp.lon_deg, 1)}`;
      const tw = ctx.measureText(text).width;
      let tx = sx + 15;
      if (tx + tw > padLeft + plotW - 6) tx = sx - tw - 15;
      if (tx < padLeft + 6) tx = padLeft + 6;
      let ty = sy + 4;
      if (ty < padTop + 14) ty = padTop + 14;
      if (ty > padTop + plotH - 6) ty = padTop + plotH - 6;
      ctx.fillStyle = cssVar('--k-text');
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(text, tx, ty);
    }

    /* axis labels - lon every 60° (90° when narrow), lat every 30°,
       11 px Inter --k-text-3 */
    ctx.font = `450 ${narrow ? 10 : 11}px ${sans}`;
    ctx.fillStyle = cssVar('--k-text-3');
    ctx.textBaseline = 'top';
    for (let lon = -180; lon <= 180; lon += narrow ? 90 : 60) {
      ctx.textAlign = lon === -180 ? 'left' : lon === 180 ? 'right' : 'center';
      const lbl = Math.abs(lon) === 180 ? '180°' : lon === 0 ? '0°' : fmtLon(lon, 0);
      ctx.fillText(lbl, X(lon), padTop + plotH + 7);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let lat = -60; lat <= 60; lat += 30) {
      const lbl = lat === 0 ? '0°' : fmtLat(lat, 0);
      ctx.fillText(lbl, padLeft - 7, Y(lat));
    }
  }

  function bind(canvas) {
    const onResize = () => {
      if (!canvas.isConnected) {
        window.removeEventListener('resize', onResize);
        live.delete(canvas);
        return;
      }
      render(canvas);
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

  return function drawGroundTrack(canvas, data, opts) {
    let st = canvas.__kTrack;
    if (!st) {
      st = canvas.__kTrack = { data: null, opts: null };
      bind(canvas);
    }
    st.data = data || null;
    st.opts = opts || {};
    render(canvas);
  };
})();
