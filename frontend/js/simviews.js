/* Kessler encounter-simulator views - pure stateless canvas renderers.
   SimViews.drawRelative / drawContext / drawRange (canvas, sim) - v5 contract
   in docs/CONTRACTS.md - plus drawComposite(canvas, sim): the offscreen
   1280x720 video-export frame (relative view in the left 720x720, range strip
   bottom-right, burned-in header). The controller (simulator.js) owns all
   state and re-invokes these from its rAF loop; nothing here binds listeners
   or keeps per-canvas state. DPR-aware like charts.js (backing store sized
   from getBoundingClientRect * devicePixelRatio on every draw - except
   drawComposite, which records at fixed device pixels); every color is a
   --k-* token resolved at draw time, never hardcoded.

   sim (controller-owned): { ephem, tRel, pr, sr, relKm, sepKm, speed,
     playing, zoom:{mode, pxPerKm}, trailRel, basis, fmtDist?, rangeCache?,
     maneuver? }
   drawContext replaces its wireframe disc fill with a
     KEarthTexture.drawHemisphere pass (quality 1) once the shared textures
     have decoded - typeof-guarded, wireframe fallback otherwise; the
     module's polylines/dots/labels always render on top.
   - maneuver: the POST /conjunctions/{id}/maneuver response (or a
     {resp} wrapper), else null/absent. drawRelative adds a post-burn ghost:
     the burn moves the PRIMARY by primary_rel_offset_km at new_tca, so in
     this primary-centred frame the secondary shifts by MINUS that offset -
     ghost = rel(new_tca) - primary_rel_offset_km, drawn as a hollow
     --k-data-1 marker at 50% alpha with a hairline from centre, labelled
     'POST-BURN MISS <dist>'.
   - basis: encounter-plane 2D basis - accepted as [u, v] (two 3-vectors) or
     {u,v} / {e1,e2} / {x,y}. A missing/degenerate basis falls back internally:
     a frame ⊥ tca_state.rel_vel_hat, else the primary's RIC R̂/Ĉ from the
     grid, else TEME x/y. Never throws.
   - fmtDist (optional): meters -> string. Internal FMT.dist-equivalent
     fallback (m under 1 km, km above) so the module stands alone in
     harnesses where format.js is not loaded.
   - rangeCache (optional): per-sample separations in km, grid order, same
     length as the r grids. Absent, drawRange recomputes in O(N) per call.
   - pathCache (optional): {key, pts} where pts is the secondary's FULL
     relative trajectory across the window, decimated to <= 400 rel 3-vectors
     (km), grid order - the controller owns and refreshes it (keyed by
     rounded pxPerKm). drawRelative renders it as a faint flyby-path polyline
     (--k-text-3 at 30%, 1px) BENEATH everything else, so the whole encounter
     shape reads before it happens. Absent/mis-shaped, the renderer decimates
     the grid itself per call (same <= 400 cap); a bare array is accepted.
   - pulse (optional): normalized 0..1 progress of the ONE closest-
     approach ring pulse - an expanding hairline --k-data-2 ring at the TCA
     ghost position (r 4 -> 30 px, alpha 0.9 -> 0), a functional event marker
     fired by the controller when tRel crosses TCA. null/absent = no pulse.
   - header (optional): {primaryName, secondaryName, riskLevel}
     for drawComposite's burned-in header. Names fall back to the ephemeris
     object names; a missing/unknown riskLevel omits the level word. The
     level word is the ONE sanctioned risk-color text use in this module
     (chips elsewhere carry the text; here the word IS the text).
   All draw coordinates are finite-guarded: degenerate input (empty r grids,
   NaN states, zero basis, sep 0) degrades to a partial draw or an empty
   state, never an exception. */
'use strict';

const SimViews = (() => {
  const TAU = Math.PI * 2;
  const RE_KM = 6378;

  /* Tokens resolve against the CANVAS being drawn (set by setup/withScope),
     not <html>: the simulator's space views sit inside .ksim-space, which
     pins the dark space palette in both themes, while chrome-side charts
     (context globe, separation chart) follow the page theme. */
  let scopeStyle = null;
  function cssVar(name) {
    const cs = scopeStyle || getComputedStyle(document.documentElement);
    return cs.getPropertyValue(name).trim();
  }
  function scopeFor(el) {
    try {
      if (el && el.isConnected) return getComputedStyle(el);
    } catch (e) { /* detached / non-element */ }
    const space = typeof document !== 'undefined'
      ? document.querySelector('.ksim-space') : null;
    return getComputedStyle(space || document.documentElement);
  }

  /* ---- 3-vector helpers ([x, y, z], km) ---- */
  function v3ok(p) {
    return Array.isArray(p) && p.length >= 3
      && Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]);
  }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }
  function norm(a) { return Math.hypot(a[0], a[1], a[2]); }
  function unit(a) { const n = norm(a); return [a[0] / n, a[1] / n, a[2] / n]; }
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }

  /* FMT.dist semantics (m under 1 km, km above), uppercased for canvas
     labels; big-km values grouped, clean decades keep no trailing zeros. */
  function fmtDistLocal(m) {
    if (!Number.isFinite(m)) return '—';
    const a = Math.abs(m);
    if (a < 999.5) {
      if (a > 0 && a < 9.95 && Math.abs(a - Math.round(a)) > 1e-9) return `${m.toFixed(1)} m`;
      return `${Math.round(m)} m`;
    }
    const km = m / 1000;
    if (km >= 99.995) return `${Math.round(km).toLocaleString('en-US')} km`;
    return `${km.toFixed(2).replace(/\.?0+$/, '')} km`;
  }
  function distLabel(sim, m) {
    if (sim && typeof sim.fmtDist === 'function') {
      try {
        const s = sim.fmtDist(m);
        if (typeof s === 'string' && s) return s;
      } catch (e) { /* controller formatter failed - fall through */ }
    }
    return fmtDistLocal(m);
  }

  /* Backing-store + context setup, charts.js idiom. Null when unusable. */
  function setup(canvas) {
    if (!canvas || typeof canvas.getContext !== 'function'
      || typeof canvas.getBoundingClientRect !== 'function') return null;
    const rect = canvas.getBoundingClientRect();
    if (!rect || rect.width < 40 || rect.height < 40) return null;
    const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
    const W = rect.width;
    const H = rect.height;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    scopeStyle = scopeFor(canvas);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const mono = cssVar('--k-mono') || "'JetBrains Mono', ui-monospace, monospace";
    return { ctx, W, H, mono };
  }

  function emptyState(ctx, W, H, mono, text) {
    ctx.font = `500 13px ${mono}`;
    ctx.fillStyle = cssVar('--k-text-3');
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, W / 2, H / 2);
  }

  /* Ephemeris grid facts. Null when there is nothing drawable (covers the
     server's propagation-failure degrade: r == [] on either object). */
  function gridOf(ephem) {
    if (!ephem) return null;
    const pr = ephem.primary && Array.isArray(ephem.primary.r) ? ephem.primary.r : [];
    const sr = ephem.secondary && Array.isArray(ephem.secondary.r) ? ephem.secondary.r : [];
    const n = Math.min(pr.length, sr.length);
    if (n < 2) return null;
    const step = Number.isFinite(ephem.step_s) && ephem.step_s > 0 ? ephem.step_s : 1;
    return { pr, sr, n, step, half: ((n - 1) * step) / 2, tcaIdx: (n - 1) >> 1 };
  }

  function relAt(g, j) {
    const a = g.pr[j];
    const b = g.sr[j];
    if (!v3ok(a) || !v3ok(b)) return null;
    return [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  }

  /* Encounter-plane basis: controller-supplied when sane, else rebuilt from
     rel_vel_hat, else the primary's RIC R̂/Ĉ, else TEME x/y. */
  function basisOf(sim, g) {
    const b = sim ? sim.basis : null;
    let u = null;
    let v = null;
    if (Array.isArray(b) && b.length >= 2) { u = b[0]; v = b[1]; }
    else if (b && typeof b === 'object') {
      u = b.u || b.e1 || b.x || null;
      v = b.v || b.e2 || b.y || null;
    }
    if (v3ok(u) && v3ok(v) && norm(u) > 1e-9 && norm(v) > 1e-9) return [unit(u), unit(v)];
    const ts = sim && sim.ephem ? sim.ephem.tca_state : null;
    if (ts && v3ok(ts.rel_vel_hat) && norm(ts.rel_vel_hat) > 1e-9) {
      const h = unit(ts.rel_vel_hat);
      const ref = Math.abs(h[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
      const cu = cross(h, ref);
      if (norm(cu) > 1e-9) {
        const U = unit(cu);
        return [U, unit(cross(h, U))];
      }
    }
    if (g && g.n > 2) {
      const i = g.tcaIdx;
      const r = g.pr[i];
      const r0 = g.pr[Math.max(i - 1, 0)];
      const r2 = g.pr[Math.min(i + 1, g.n - 1)];
      if (v3ok(r) && v3ok(r0) && v3ok(r2) && norm(r) > 1e-9) {
        const c = cross(r, sub(r2, r0));
        if (norm(c) > 1e-9) return [unit(r), unit(c)];
      }
    }
    return [[1, 0, 0], [0, 1, 0]];
  }

  /* Post-burn ghost position in the relative frame (maneuver sandbox).
     rel' = sec - (pri + off) = rel(new_tca) - off - the primary moved by
     +primary_rel_offset_km, so the secondary-relative-to-primary vector
     shifts by MINUS the offset. rel(new_tca) is linearly interpolated on
     the grid; a missing/unparseable new_tca degrades to the stored TCA and
     a junk offset to [0,0,0]. Null when the grid rows are unusable. */
  function maneuverRel(resp, g, ephem) {
    const raw = resp.primary_rel_offset_km;
    const off = v3ok(raw) ? raw : [0, 0, 0];
    let dt = 0;
    if (resp.new_tca && ephem && ephem.tca) {
      const d = (Date.parse(resp.new_tca) - Date.parse(ephem.tca)) / 1000;
      if (Number.isFinite(d)) dt = Math.max(-g.half, Math.min(g.half, d));
    }
    const f = (dt + g.half) / g.step;
    const j0 = Math.max(0, Math.min(g.n - 2, Math.floor(f)));
    const fr = Math.max(0, Math.min(1, f - j0));
    const a = relAt(g, j0);
    const b = relAt(g, j0 + 1);
    if (!a || !b) return null;
    return [
      a[0] + (b[0] - a[0]) * fr - off[0],
      a[1] + (b[1] - a[1]) * fr - off[1],
      a[2] + (b[2] - a[2]) * fr - off[2],
    ];
  }

  /* Every-nth decimation to <= cap points, last point kept (context view). */
  function decimate(arr, cap) {
    const out = [];
    if (!Array.isArray(arr)) return out;
    const step = Math.max(1, Math.ceil(arr.length / cap));
    for (let i = 0; i < arr.length; i += step) if (v3ok(arr[i])) out.push(arr[i]);
    const last = arr[arr.length - 1];
    if (step > 1 && (arr.length - 1) % step !== 0 && v3ok(last)) out.push(last);
    return out;
  }

  /* ================= relative (encounter-plane) view ================= */
  function drawRelative(canvas, sim) {
    const g2 = setup(canvas);
    if (!g2) return;
    renderRelative(g2.ctx, g2.W, g2.H, g2.mono, sim);
  }

  /* core renderer - draws into an already-prepared ctx region (0,0,W,H);
     shared by drawRelative (live canvas) and drawComposite (export frame) */
  function renderRelative(ctx, W, H, mono, sim) {
    const ephem = sim ? sim.ephem : null;
    const g = gridOf(ephem);
    if (!g) { emptyState(ctx, W, H, mono, 'No ephemeris for this event'); return; }

    const cGrid = cssVar('--k-grid');
    const cT2 = cssVar('--k-text-2');
    const cT3 = cssVar('--k-text-3');
    const cSurface = cssVar('--k-surface');
    const cD1 = cssVar('--k-data-1');
    const cD2 = cssVar('--k-data-2');
    const cx = W / 2;
    const cy = H / 2;
    const halfView = Math.min(W, H) / 2;
    const maxPx = Math.hypot(W, H) / 2; /* centre to corner */

    /* px per km - controller AUTO/MANUAL value; fallback puts the current
       separation at 40% of the half-view so a bad zoom never blanks the view */
    let ppk = sim && sim.zoom ? Number(sim.zoom.pxPerKm) : NaN;
    if (!Number.isFinite(ppk) || ppk <= 0) {
      const sep = sim && Number.isFinite(sim.sepKm) && sim.sepKm > 1e-9 ? sim.sepKm : null;
      ppk = sep ? (halfView * 0.4) / sep : 1;
    }
    const [U, V] = basisOf(sim, g);
    const px = (v) => cx + dot(v, U) * ppk;   /* rel km vec -> screen x */
    const py = (v) => cy - dot(v, V) * ppk;   /* screen y grows downward */

    /* full flyby path - the threat's whole relative trajectory across
       the window as a faint polyline BENEATH everything: the moving secondary
       slides along it and the TCA ghost + miss vector sit on it. Controller
       cache preferred (sim.pathCache.{pts} or a bare array); fallback
       decimates the grid to <= 400 points per call. */
    const pcRaw = sim ? sim.pathCache : null;
    let pathPts = null;
    if (pcRaw && Array.isArray(pcRaw.pts)) pathPts = pcRaw.pts;
    else if (Array.isArray(pcRaw)) pathPts = pcRaw;
    if (!pathPts) {
      pathPts = [];
      const pStep = Math.max(1, Math.ceil(g.n / 400));
      for (let j = 0; j < g.n; j += pStep) pathPts.push(relAt(g, j));
      if (pStep > 1 && (g.n - 1) % pStep !== 0) pathPts.push(relAt(g, g.n - 1));
    }
    ctx.strokeStyle = cT3;
    ctx.globalAlpha = 0.3;
    ctx.lineWidth = 1;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let pathPen = false;
    for (const v of pathPts) {
      if (!v3ok(v)) { pathPen = false; continue; }
      const x = px(v);
      const y = py(v);
      if (!Number.isFinite(x) || !Number.isFinite(y)) { pathPen = false; continue; }
      if (!pathPen) { ctx.moveTo(x, y); pathPen = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    /* 1-2-5 decade range rings, labels auto m/km. Exponent-bounded walk:
       cheap and never spins whatever pxPerKm arrives. */
    const minKm = 18 / ppk;
    const maxKm = maxPx / ppk;
    const rings = [];
    if (maxKm > minKm && Number.isFinite(maxKm)) {
      const eTop = Math.floor(Math.log10(maxKm));
      for (let e = eTop; e >= eTop - 10 && rings.length < 12; e--) {
        for (const f of [5, 2, 1]) {
          const rKm = f * Math.pow(10, e);
          if (rKm <= maxKm && rKm >= minKm) rings.push(rKm);
        }
      }
    }
    ctx.lineWidth = 1;
    ctx.strokeStyle = cGrid;
    for (const rKm of rings) {
      ctx.beginPath();
      ctx.arc(cx, cy, rKm * ppk, 0, TAU);
      ctx.stroke();
    }
    ctx.font = `500 11px ${mono}`;
    ctx.fillStyle = cT3;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    /* ring labels sit on the lower-right diagonal - the top of the ring
       is where miss vectors and ghost labels tend to live */
    ctx.textAlign = 'left';
    let lastLy = -Infinity;
    const ringLabels = [];
    for (const rKm of rings.slice().sort((a, b) => a - b)) {
      const rp = rKm * ppk;
      const lx = cx + rp * 0.7071 + 5;
      const ly = cy + rp * 0.7071 + 4;
      if (rp < 22 || lx > W - 50 || ly > H - 8 || ly - lastLy < 16) continue;
      ringLabels.push({ t: distLabel(sim, rKm * 1000), x: lx, y: ly });
      lastLy = ly;
    }

    /* combined 1σ circle - --k-text-3 at 35%. The label is DEFERRED until
       after the MISS label is placed: at close zoom both live at the centre
       and drew on top of each other. */
    let sigmaRingPx = null;
    let missRect = null;
    const labelRects = [];
    /* ghost labels (Miss / Before / After burn) are placed here but DRAWN
       last, over the moving marker and its velocity arrow, with a surface
       halo - and nudged off the marker when it sits on them (QA B5) */
    const ghostLabels = [];
    const sigmaM = ephem ? Number(ephem.sigma_m) : NaN;
    if (Number.isFinite(sigmaM) && sigmaM > 0) {
      const rs = (sigmaM / 1000) * ppk;
      if (rs >= 6 && rs <= maxPx * 4) {
        ctx.globalAlpha = 0.35;
        ctx.strokeStyle = cT3;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(cx, cy, rs, 0, TAU);
        ctx.stroke();
        ctx.globalAlpha = 1;
        sigmaRingPx = rs;
      }
    }

    /* hard-body circle, 10 m fixed - meaningful only zoomed well in */
    if (ppk > 2000) {
      const rh = 0.01 * ppk;
      if (rh <= maxPx * 4) {
        ctx.strokeStyle = cT3;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(cx, cy, rh, 0, TAU);
        ctx.stroke();
        ctx.fillStyle = cT3;
        ctx.textAlign = 'center';
        const hy = cy + rh + 12 < H - 2 ? cy + rh + 12 : cy + rh - 6;
        ctx.fillText('Hard body 10 m', cx, hy);
      }
    }

    /* age-faded secondary trail - groundtrack fade idiom, alpha ramping
       0.25 (oldest) -> 1.0 (newest); monotone by construction */
    const trail = sim && Array.isArray(sim.trailRel) ? sim.trailRel : [];
    const tp = [];
    for (const e of trail) {
      /* controller entries are {t, rel:[x,y,z]} (simulator.js contract);
         bare [x,y,z] vectors are also accepted */
      const v = v3ok(e) ? e : (e && v3ok(e.rel) ? e.rel : null);
      if (!v) continue;
      const x = px(v);
      const y = py(v);
      if (Number.isFinite(x) && Number.isFinite(y)) tp.push([x, y]);
    }
    if (tp.length > 1) {
      ctx.strokeStyle = cD2;
      ctx.lineWidth = 1;
      ctx.lineCap = 'round';
      const m = tp.length - 1;
      for (let i = 0; i < m; i++) {
        ctx.globalAlpha = 0.25 + 0.75 * ((i + 1) / m);
        ctx.beginPath();
        ctx.moveTo(tp[i][0], tp[i][1]);
        ctx.lineTo(tp[i + 1][0], tp[i + 1][1]);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    /* TCA ghost: secondary's TCA-relative position - hollow marker plus a
       hairline miss vector from the centre, labelled with the miss distance */
    const ts = ephem && ephem.tca_state ? ephem.tca_state : null;
    const relTca = relAt(g, g.tcaIdx);
    if (relTca) {
      const gx = px(relTca);
      const gy = py(relTca);
      if (Number.isFinite(gx) && Number.isFinite(gy)) {
        const dx = gx - cx;
        const dy = gy - cy;
        const len = Math.hypot(dx, dy);
        if (len > 0.5) {
          ctx.strokeStyle = cT2;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(gx, gy);
          ctx.stroke();
        }
        /* closest-approach pulse - one expanding ring at the ghost,
           controller-fired on the TCA crossing; sim.pulse is normalized
           0..1 progress (functional event marker, not decoration) */
        const pulseP = sim && Number.isFinite(sim.pulse) ? sim.pulse : null;
        if (pulseP !== null && pulseP >= 0 && pulseP <= 1) {
          ctx.strokeStyle = cD2;
          ctx.globalAlpha = 0.9 * (1 - pulseP);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(gx, gy, 4 + pulseP * 26, 0, TAU);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
        ctx.beginPath();
        ctx.arc(gx, gy, 3.5, 0, TAU);
        ctx.fillStyle = cSurface;
        ctx.fill();
        ctx.strokeStyle = cD2;
        ctx.lineWidth = 1;
        ctx.stroke();
        const missM = ts && Number.isFinite(ts.sep_m) ? ts.sep_m : norm(relTca) * 1000;
        const hasBurn = !!(sim && sim.maneuver);
        const text = `${hasBurn ? 'Before' : 'Miss'} ${distLabel(sim, missM)}`;
        ctx.font = `600 12px ${mono}`;
        const tw = ctx.measureText(text).width;
        let lx;
        let ly;
        if (len > 1) {
          /* along the vector, kept on-canvas even when the ghost is far off */
          const f = Math.min(len * 0.55, halfView - 16) / len;
          lx = cx + dx * f + 6;
          ly = cy + dy * f - 6;
        } else {
          lx = cx + 8;
          ly = cy - 10;
        }
        lx = Math.min(Math.max(lx, 4), W - tw - 4);
        ly = Math.min(Math.max(ly, 12), H - 6);
        missRect = { x: lx, w: tw, y: ly };
        labelRects.push(missRect);
        ghostLabels.push({ text, rect: missRect, ax: gx, ay: gy });
        ctx.font = `500 11px ${mono}`;
      }
    }

    /* post-burn ghost - applied what-if maneuver: hollow --k-data-1
       marker at 50% alpha at rel(new_tca) - primary_rel_offset_km, hairline
       from centre, 'POST-BURN MISS <dist>' label in a text token */
    const mraw = sim ? sim.maneuver : null;
    const mresp = mraw && typeof mraw === 'object'
      ? (mraw.resp && typeof mraw.resp === 'object' ? mraw.resp : mraw) : null;
    if (mresp) {
      /* before/after: the post-burn flyby - the relative path shifted by
         MINUS the burn's primary offset at the new TCA (the offset is taken
         as constant across the window: a drawing aid; the labelled number
         is the server's new_miss_m) - plus a hollow marker riding it. */
      const offRaw = mresp.primary_rel_offset_km;
      const off = v3ok(offRaw) ? offRaw : null;
      if (off && pathPts) {
        ctx.strokeStyle = cD1;
        ctx.globalAlpha = 0.55;
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        let pen = false;
        for (const v of pathPts) {
          if (!v3ok(v)) { pen = false; continue; }
          const q = [v[0] - off[0], v[1] - off[1], v[2] - off[2]];
          const x = px(q);
          const y = py(q);
          if (!Number.isFinite(x) || !Number.isFinite(y)) { pen = false; continue; }
          if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        const cur = sim && v3ok(sim.relKm) ? sim.relKm : null;
        if (cur) {
          const q = [cur[0] - off[0], cur[1] - off[1], cur[2] - off[2]];
          const x = px(q);
          const y = py(q);
          if (Number.isFinite(x) && Number.isFinite(y)) {
            ctx.beginPath();
            ctx.arc(x, y, 4.5, 0, TAU);
            ctx.strokeStyle = cD1;
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        }
      }
      const gRel = maneuverRel(mresp, g, ephem);
      const mgx = gRel ? px(gRel) : NaN;
      const mgy = gRel ? py(gRel) : NaN;
      if (Number.isFinite(mgx) && Number.isFinite(mgy)) {
        ctx.strokeStyle = cD1;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.8;
        if (Math.hypot(mgx - cx, mgy - cy) > 0.5) {
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(mgx, mgy);
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(mgx, mgy, 3.5, 0, TAU);
        ctx.globalAlpha = 1;
        ctx.fillStyle = cSurface;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.stroke();
        const mm = Number.isFinite(mresp.new_miss_m)
          ? Number(mresp.new_miss_m) : norm(gRel) * 1000;
        const mtext = `After burn ${distLabel(sim, mm)}`;
        ctx.font = `600 12px ${mono}`;
        const mtw = ctx.measureText(mtext).width;
        const mlx = Math.min(Math.max(mgx + 7, 4), W - mtw - 4);
        const mly = Math.min(Math.max(mgy + 14, 12), H - 6);
        const mRect = { x: mlx, w: mtw, y: mly };
        labelRects.push(mRect);
        ghostLabels.push({ text: mtext, rect: mRect, ax: mgx, ay: mgy });
        ctx.font = `500 11px ${mono}`;
      }
    }

    /* ring labels, deferred so they yield to the ghost labels */
    ctx.font = `500 11px ${mono}`;
    ctx.fillStyle = cT3;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    /* ...and to the live secondary marker + its velocity arrow */
    const curRel = sim && v3ok(sim.relKm) ? sim.relKm : null;
    const mkx = curRel ? px(curRel) : NaN;
    const mky = curRel ? py(curRel) : NaN;
    for (const rl of ringLabels) {
      const w = ctx.measureText(rl.t).width;
      if (labelRects.some((r) => rl.x < r.x + r.w + 4 && rl.x + w > r.x - 4 && Math.abs(rl.y - r.y) < 13)) continue;
      if (Number.isFinite(mkx) && rl.x < mkx + 24 && rl.x + w > mkx - 24 &&
          rl.y - 11 < mky + 24 && rl.y + 3 > mky - 24) continue;
      ctx.fillText(rl.t, rl.x, rl.y);
      labelRects.push({ x: rl.x, w, y: rl.y });
    }

    /* deferred 1σ label - first placement that stays on-canvas and clears
       the MISS label: inside-top of the ring, below it, above it */
    if (sigmaRingPx !== null) {
      const stext = '1σ assumed';
      ctx.font = `500 11px ${mono}`;
      const sw = ctx.measureText(stext).width;
      const hitsMiss = (y) => missRect !== null &&
        Math.abs(y - missRect.y) < 12 &&
        cx + sw / 2 > missRect.x - 4 &&
        cx - sw / 2 < missRect.x + missRect.w + 4;
      /* lower-left diagonal first (ring labels own the lower right) */
      const dx = sigmaRingPx * 0.7071;
      const ll = { x: cx - dx - 5 - sw, y: cy + dx + 4 };
      const hitsAny = (x, y) => labelRects.some((r) =>
        x < r.x + r.w + 4 && x + sw > r.x - 4 && Math.abs(y - r.y) < 13);
      if (sigmaRingPx > 26 && ll.x > 4 && ll.y < H - 6 && !hitsAny(ll.x, ll.y)) {
        ctx.fillStyle = cT3;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(stext, ll.x, ll.y);
      } else {
        const cand = [cy + sigmaRingPx + 14, cy - sigmaRingPx - 6]
          .find((y) => y > 10 && y < H - 4 && !hitsMiss(y) && !hitsAny(cx - sw / 2, y));
        if (cand !== undefined) {
          ctx.fillStyle = cT3;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'alphabetic';
          ctx.fillText(stext, cx, cand);
        }
      }
    }

    /* current secondary position - controller-interpolated relKm preferred,
       nearest grid sample otherwise */
    const tRel = sim && Number.isFinite(sim.tRel) ? sim.tRel : 0;
    const idx = Math.max(0, Math.min(g.n - 1, Math.round((tRel + g.half) / g.step)));
    let rel = sim && v3ok(sim.relKm) ? sim.relKm : relAt(g, idx);
    let sx = null;
    let sy = null;
    if (rel) {
      sx = px(rel);
      sy = py(rel);
      if (!Number.isFinite(sx) || !Number.isFinite(sy)) { sx = null; sy = null; }
    }

    /* relative-velocity direction arrow - thin 12 px shaft off the marker */
    if (sx !== null && g.n > 2) {
      const i = Math.max(1, Math.min(g.n - 2, idx));
      const a = relAt(g, i - 1);
      const b = relAt(g, i + 1);
      if (a && b) {
        const dv = sub(b, a);
        const ax = dot(dv, U);
        const ay = -dot(dv, V);
        const al = Math.hypot(ax, ay);
        if (al > 1e-12) {
          const ux = ax / al;
          const uy = ay / al;
          const x1 = sx + ux * 19;
          const y1 = sy + uy * 19;
          ctx.strokeStyle = cT2;
          ctx.lineWidth = 1;
          ctx.lineCap = 'butt';
          ctx.beginPath();
          ctx.moveTo(sx + ux * 7, sy + uy * 7);
          ctx.lineTo(x1, y1);
          ctx.moveTo(x1, y1);
          ctx.lineTo(x1 - ux * 4 - uy * 2.5, y1 - uy * 4 + ux * 2.5);
          ctx.moveTo(x1, y1);
          ctx.lineTo(x1 - ux * 4 + uy * 2.5, y1 - uy * 4 - ux * 2.5);
          ctx.stroke();
        }
      }
    }

    /* secondary marker - ringed dot, charts endpoint idiom */
    if (sx !== null) {
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, TAU);
      ctx.fillStyle = cSurface;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(sx, sy, 3, 0, TAU);
      ctx.fillStyle = cD2;
      ctx.fill();
    }

    /* primary - centre-locked 4 px ring on a surface plinth */
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, TAU);
    ctx.fillStyle = cSurface;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, TAU);
    ctx.strokeStyle = cD1;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    /* ghost labels last: keep clear of the live marker + its 19 px arrow */
    if (ghostLabels.length) {
      ctx.font = `600 12px ${mono}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      const R = 24;
      const hitsMarker = (x, y, w) => sx !== null &&
        x < sx + R && x + w > sx - R && y - 12 < sy + R && y + 4 > sy - R;
      const placed = [];
      const hitsPlaced = (x, y, w) => placed.some((r) =>
        x < r.x + r.w + 4 && x + w > r.x - 4 && Math.abs(y - r.y) < 15);
      for (const gl of ghostLabels) {
        let { x, y, w } = gl.rect;
        if (hitsMarker(x, y, w) || hitsPlaced(x, y, w)) {
          const cands = [
            [gl.ax + 8, gl.ay + 18], [gl.ax - w - 10, gl.ay - 8],
            [gl.ax - w - 10, gl.ay + 18], [gl.ax + 8, gl.ay - 12],
            [sx !== null ? sx + R + 4 : x, y], [sx !== null ? sx - R - 4 - w : x, y],
          ].map(([a, b]) => [Math.min(Math.max(a, 4), W - w - 4), Math.min(Math.max(b, 12), H - 6)]);
          const ok = cands.find(([a, b]) => !hitsMarker(a, b, w) && !hitsPlaced(a, b, w));
          if (ok) { x = ok[0]; y = ok[1]; }
        }
        ctx.lineWidth = 3;
        ctx.lineJoin = 'round';
        ctx.strokeStyle = cSurface;
        ctx.strokeText(gl.text, x, y);
        ctx.fillStyle = cT2;
        ctx.fillText(gl.text, x, y);
        placed.push({ x, y, w });
      }
      ctx.font = `500 11px ${mono}`;
    }
  }

  /* ================= context view (wireframe Earth) ================= */
  function circlePts(r, plane, n) {
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const a = (i / n) * TAU;
      if (plane === 'equator') pts.push([r * Math.cos(a), r * Math.sin(a), 0]);
      else pts.push([r * Math.cos(a), 0, r * Math.sin(a)]); /* x-z meridian */
    }
    return pts;
  }

  /* --- textured hemisphere support (earthtex.js, typeof-guarded) -------------
     drawContext's fixed oblique projection is globe.js projEarth with
     spin = -yaw and tilt = -24°: its x1 = x·cosY + y·sinY equals
     x·cos(spin) - y·sin(spin), and its screen-up row z·cosT - y1·sinT equals
     y1·sin(-T) + z·cos(-T) - so KEarthTexture.drawHemisphere (whose rotation
     chain is byte-compatible with projEarth) reproduces this view exactly,
     front hemisphere included. GMST at the displayed instant comes from the
     ephemeris (gmst_start_rad + earth_rot_rad_s·(tRel + window_s)); the
     fallback GMST and the sun direction reuse globe.js's low-precision
     formulas so the day/night blend matches the main globe. */
  const J2000_MS = Date.UTC(2000, 0, 1, 12);
  const DEG = Math.PI / 180;

  function gmstFromMs(ms) {
    const d = (ms - J2000_MS) / 86400000;
    return ((280.46061837 + 360.98564736629 * d) % 360) * DEG;
  }

  function sunDirAt(ms) {
    const d = (ms - J2000_MS) / 86400000;
    const g = (357.529 + 0.98560028 * d) * DEG;
    const L = (280.459 + 0.98564736 * d) * DEG
      + (1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * DEG;
    const eps = (23.439 - 0.00000036 * d) * DEG;
    const sl = Math.sin(L);
    return [Math.cos(L), Math.cos(eps) * sl, Math.sin(eps) * sl];
  }

  /* GMST + sun direction at the sim's displayed instant; null when the
     ephemeris carries neither the gmst fields nor a parseable tca (an
     arbitrarily-rotated Earth would be misleading - stay wireframe). */
  function contextEarthState(ephem, sim) {
    if (!ephem) return null;
    const tRel = sim && Number.isFinite(sim.tRel) ? sim.tRel : 0;
    const tcaMs = ephem.tca ? Date.parse(ephem.tca) : NaN;
    const tMs = Number.isFinite(tcaMs) ? tcaMs + tRel * 1000 : NaN;
    let gmst = NaN;
    if (Number.isFinite(ephem.gmst_start_rad)
        && Number.isFinite(ephem.earth_rot_rad_s)
        && Number.isFinite(ephem.window_s)) {
      gmst = ephem.gmst_start_rad + ephem.earth_rot_rad_s * (tRel + ephem.window_s);
    } else if (Number.isFinite(tMs)) {
      gmst = gmstFromMs(tMs);
    }
    if (!Number.isFinite(gmst)) return null;
    return { gmst, sun: Number.isFinite(tMs) ? sunDirAt(tMs) : null };
  }

  function drawContext(canvas, sim) {
    const g2 = setup(canvas);
    if (!g2) return;
    const { ctx, W, H, mono } = g2;
    const cx = W / 2;
    const cy = H / 2;
    const ephem = sim ? sim.ephem : null;

    const entries = [];
    if (ephem) {
      const pp = decimate(ephem.primary && ephem.primary.r, 300);
      if (pp.length > 1) {
        entries.push({
          path: pp,
          color: cssVar('--k-data-1'),
          label: ephem.primary ? ephem.primary.name : null,
          cur: sim && v3ok(sim.pr) ? sim.pr : null,
          dy: -7,
        });
      }
      const sp = decimate(ephem.secondary && ephem.secondary.r, 300);
      if (sp.length > 1) {
        entries.push({
          path: sp,
          color: cssVar('--k-data-2'),
          label: ephem.secondary ? ephem.secondary.name : null,
          cur: sim && v3ok(sim.sr) ? sim.sr : null,
          dy: 14,
        });
      }
    }

    /* fixed oblique projection, orbit_plot math style (24° elevation) */
    const yaw = -0.6;
    const tilt = (24 * Math.PI) / 180;
    const cosY = Math.cos(yaw);
    const sinY = Math.sin(yaw);
    const cosT = Math.cos(tilt);
    const sinT = Math.sin(tilt);
    let maxR = RE_KM;
    for (const e of entries) {
      for (const p of e.path) {
        const r = Math.hypot(p[0], p[1], p[2]);
        if (r > maxR) maxR = r;
      }
    }
    const scale = Math.max(Math.min(W, H) / 2 - 14, 8) / maxR;
    const proj = (p) => {
      const x1 = p[0] * cosY + p[1] * sinY;
      const y1 = -p[0] * sinY + p[1] * cosY;
      return [cx + x1 * scale, cy - (p[2] * cosT - y1 * sinT) * scale];
    };
    const stroke = (pts, color, width) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      let pen = false;
      for (const p of pts) {
        if (!v3ok(p)) { pen = false; continue; }
        const s = proj(p);
        if (!Number.isFinite(s[0]) || !Number.isFinite(s[1])) { pen = false; continue; }
        if (!pen) { ctx.moveTo(s[0], s[1]); pen = true; } else ctx.lineTo(s[0], s[1]);
      }
      ctx.stroke();
    };

    /* textured hemisphere replaces the wireframe DISC FILL when the
       shared earth-texture module has decoded (orbit polylines, dots and
       labels stay on top; wireframe fallback otherwise - contract). On pages
       without globe.js nothing else kicks the texture load, so an idle
       module is started here (load() is idempotent single-flight). */
    if (typeof KEarthTexture !== 'undefined' && KEarthTexture) {
      try {
        if (typeof KEarthTexture.ready === 'function' && KEarthTexture.ready()
            && typeof KEarthTexture.drawHemisphere === 'function') {
          const es = contextEarthState(ephem, sim);
          if (es) {
            KEarthTexture.drawHemisphere(ctx, {
              cx,
              cy,
              radiusPx: RE_KM * scale,
              spin: -yaw,
              tilt: -tilt,
              gmstRad: es.gmst,
              sunDirTEME: es.sun,
              quality: 1,
            });
          }
        } else if (typeof KEarthTexture.load === 'function') {
          const p = KEarthTexture.load();
          if (p && typeof p.catch === 'function') p.catch(() => {});
        }
      } catch (e) { /* texture fault - the wireframe carries the frame */ }
    }

    /* wireframe Earth: spin axis, meridian, equator, disc outline */
    const cGrid = cssVar('--k-grid');
    stroke([[0, 0, -RE_KM * 1.2], [0, 0, RE_KM * 1.2]], cGrid, 1);
    stroke(circlePts(RE_KM, 'meridian', 72), cGrid, 1);
    stroke(circlePts(RE_KM, 'equator', 72), cssVar('--k-border'), 1);
    ctx.strokeStyle = cssVar('--k-border-strong');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, RE_KM * scale, 0, TAU);
    ctx.stroke();

    ctx.font = `500 11px ${mono}`;
    if (!entries.length) {
      ctx.fillStyle = cssVar('--k-text-3');
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText('No orbit data', cx, cy - RE_KM * scale - 8);
      return;
    }

    const cSurface = cssVar('--k-surface');
    const cLabel = cssVar('--k-text-2');
    for (const e of entries) {
      stroke(e.path, e.color, 1.5);
      /* moving dot at the interpolated current position */
      let anchor = null;
      if (e.cur) {
        const s = proj(e.cur);
        if (Number.isFinite(s[0]) && Number.isFinite(s[1])) {
          ctx.beginPath();
          ctx.arc(s[0], s[1], 4, 0, TAU);
          ctx.fillStyle = cSurface;
          ctx.fill();
          ctx.beginPath();
          ctx.arc(s[0], s[1], 2.5, 0, TAU);
          ctx.fillStyle = e.color;
          ctx.fill();
          anchor = s;
        }
      }
      if (!anchor) anchor = proj(e.path[0]);
      if (e.label && Number.isFinite(anchor[0]) && Number.isFinite(anchor[1])) {
        const text = String(e.label);
        const tw = ctx.measureText(text).width;
        const tx = Math.min(Math.max(anchor[0] + 7, 3), W - tw - 3);
        const ty = Math.min(Math.max(anchor[1] + e.dy, 10), H - 4);
        ctx.fillStyle = cLabel;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(text, tx, ty);
      }
    }
  }

  /* ================= range (log-y separation) chart ================= */
  function fmtRelT(t) {
    if (Math.abs(t) < 0.5) return 'Closest';
    const a = Math.abs(Math.round(t));
    const mm = String(Math.floor(a / 60)).padStart(2, '0');
    const ss = String(a % 60).padStart(2, '0');
    return `${t < 0 ? '−' : '+'}${mm}:${ss}`;
  }

  function drawRange(canvas, sim) {
    const g2 = setup(canvas);
    if (!g2) return;
    renderRange(g2.ctx, g2.W, g2.H, g2.mono, sim);
  }

  /* core renderer - region-relative like renderRelative */
  function renderRange(ctx, W, H, mono, sim) {
    const ephem = sim ? sim.ephem : null;
    const g = gridOf(ephem);
    if (!g) { emptyState(ctx, W, H, mono, 'No data'); return; }

    /* per-sample separations, km - controller cache or O(N) recompute */
    let seps = null;
    const rc = sim ? sim.rangeCache : null;
    if (Array.isArray(rc) && rc.length === g.n) seps = rc;
    if (!seps) {
      seps = new Array(g.n);
      for (let j = 0; j < g.n; j++) {
        const r = relAt(g, j);
        seps[j] = r ? norm(r) : NaN;
      }
    }
    let minVal = Infinity;
    let minIdx = -1;
    let minPos = Infinity;
    let maxSep = 0;
    for (let j = 0; j < g.n; j++) {
      const v = seps[j];
      if (!Number.isFinite(v) || v < 0) continue;
      if (v < minVal) { minVal = v; minIdx = j; }
      if (v > 0 && v < minPos) minPos = v;
      if (v > maxSep) maxSep = v;
    }
    if (minIdx < 0) { emptyState(ctx, W, H, mono, 'No data'); return; }
    if (!Number.isFinite(minPos)) { minPos = 0.001; maxSep = Math.max(maxSep, 1); }
    let lo = Math.floor(Math.log10(minPos));
    let hi = Math.ceil(Math.log10(Math.max(maxSep, minPos)));
    if (hi <= lo) hi = lo + 1;

    /* decade y ticks - charts.js walk with a guaranteed floor tick */
    const stepE = Math.max(1, Math.ceil((hi - lo) / 4));
    const tickEs = [];
    let lastE = null;
    for (let e = hi; e >= lo; e -= stepE) { tickEs.push(e); lastE = e; }
    if (lastE !== lo) tickEs.push(lo);
    ctx.font = `500 11px ${mono}`;
    const tickLabels = tickEs.map((e) => distLabel(sim, Math.pow(10, e) * 1000));
    let labelW = 0;
    for (const l of tickLabels) labelW = Math.max(labelW, ctx.measureText(l).width);

    const padLeft = Math.ceil(labelW) + 12;
    const padRight = 8;
    const padTop = 8;
    const padBottom = 18;
    const plotW = Math.max(10, W - padLeft - padRight);
    const plotH = Math.max(10, H - padTop - padBottom);
    const floorKm = Math.pow(10, lo);
    const yPos = (v) => {
      const lv = Math.log10(Math.max(v, floorKm));
      return padTop + (1 - (lv - lo) / (hi - lo)) * plotH;
    };
    const xPos = (t) => padLeft + ((t + g.half) / (2 * g.half)) * plotW;

    /* gridlines - solid hairlines */
    ctx.lineWidth = 1;
    ctx.strokeStyle = cssVar('--k-grid');
    for (const e of tickEs) {
      const y = Math.round(yPos(Math.pow(10, e))) + 0.5;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    }
    const xFracs = [0, 0.25, 0.5, 0.75, 1];
    const xTickTs = xFracs.map((f) => -g.half + f * 2 * g.half);
    for (const t of xTickTs) {
      const x = Math.round(xPos(t)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();
    }

    /* TCA vertical hairline, one step stronger than the grid */
    ctx.strokeStyle = cssVar('--k-border-strong');
    const xTca = Math.round(xPos(0)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(xTca, padTop);
    ctx.lineTo(xTca, padTop + plotH);
    ctx.stroke();

    /* axis labels - 10 px mono --k-text-3 */
    const cT3 = cssVar('--k-text-3');
    ctx.fillStyle = cT3;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    for (let i = 0; i < tickEs.length; i++) {
      ctx.fillText(tickLabels[i], padLeft - 6, yPos(Math.pow(10, tickEs[i])));
    }
    ctx.textBaseline = 'top';
    xTickTs.forEach((t, i) => {
      ctx.textAlign = i === 0 ? 'left' : i === xTickTs.length - 1 ? 'right' : 'center';
      ctx.fillText(fmtRelT(t), xPos(t), padTop + plotH + 5);
    });

    /* separation curve - pen-up over gaps; zeros sit on the domain floor */
    const cD1 = cssVar('--k-data-1');
    ctx.strokeStyle = cD1;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    let pen = false;
    for (let j = 0; j < g.n; j++) {
      const v = seps[j];
      if (!Number.isFinite(v) || v < 0) { pen = false; continue; }
      const x = xPos(-g.half + j * g.step);
      const y = yPos(v);
      if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();

    /* minimum point - ringed dot + MIN label */
    const cSurface = cssVar('--k-surface');
    const cT2 = cssVar('--k-text-2');
    const mx = xPos(-g.half + minIdx * g.step);
    const my = yPos(minVal);
    ctx.beginPath();
    ctx.arc(mx, my, 5, 0, TAU);
    ctx.fillStyle = cSurface;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(mx, my, 3, 0, TAU);
    ctx.fillStyle = cD1;
    ctx.fill();
    const minText = `Min ${distLabel(sim, minVal * 1000)}`;
    ctx.font = `600 11px ${mono}`;
    const mtw = ctx.measureText(minText).width;
    let lx = mx + 7;
    if (lx + mtw > padLeft + plotW - 2) lx = mx - mtw - 7;
    if (lx < padLeft + 2) lx = padLeft + 2;
    let ly = my - 7;
    if (ly < padTop + 9) ly = my + 14;
    ctx.fillStyle = cT2;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(minText, lx, ly);
    ctx.font = `500 11px ${mono}`;

    /* applied what-if burn: the post-burn minimum as a dashed accent line,
       so the before/after gap reads on the same log axis */
    const mr = sim ? sim.maneuver : null;
    const mresp = mr && typeof mr === 'object' ? (mr.resp || mr) : null;
    const newMissKm = mresp && Number.isFinite(Number(mresp.new_miss_m))
      ? Number(mresp.new_miss_m) / 1000 : NaN;
    if (Number.isFinite(newMissKm) && newMissKm > 0) {
      const yb = Math.round(yPos(newMissKm)) + 0.5;
      ctx.strokeStyle = cD1;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(padLeft, yb);
      ctx.lineTo(padLeft + plotW, yb);
      ctx.stroke();
      ctx.setLineDash([]);
      const bt = `After burn ${distLabel(sim, newMissKm * 1000)}`;
      ctx.font = `600 11px ${mono}`;
      ctx.fillStyle = cD1;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(bt, padLeft + 6, Math.max(padTop + 10, yb - 5));
      ctx.font = `500 11px ${mono}`;
    }

    /* cursor at tRel + current-separation dot on the curve */
    const t = sim && Number.isFinite(sim.tRel)
      ? Math.max(-g.half, Math.min(g.half, sim.tRel)) : 0;
    const xc = Math.round(xPos(t)) + 0.5;
    ctx.strokeStyle = cT2;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xc, padTop);
    ctx.lineTo(xc, padTop + plotH);
    ctx.stroke();
    let cur = sim && Number.isFinite(sim.sepKm) && sim.sepKm >= 0 ? sim.sepKm : null;
    if (cur === null) {
      const f = (t + g.half) / g.step;
      const j0 = Math.max(0, Math.min(g.n - 2, Math.floor(f)));
      const a = seps[j0];
      const b = seps[j0 + 1];
      if (Number.isFinite(a) && Number.isFinite(b)) cur = a + (b - a) * (f - j0);
    }
    if (cur !== null && Number.isFinite(cur)) {
      ctx.beginPath();
      ctx.arc(xPos(t), yPos(cur), 2.5, 0, TAU);
      ctx.fillStyle = cD1;
      ctx.fill();
    }
  }

  /* ================= composite export frame ======================
     One 1280x720 (device px - no DPR scaling: MediaRecorder captures the
     backing store) video frame: relative view fills the left HxH square,
     range strip sits bottom-right, and the burned-in header (KESSLER
     wordmark, risk-level word, pair names, SEP / T-vs-TCA readouts)
     occupies the right column's top. The level word is drawn in its risk
     color - the sanctioned use; every other label stays in text tokens. */
  function pad2(n) { return String(n).padStart(2, '0'); }
  function fmtVsTca(t) {
    if (!Number.isFinite(t)) return '—';
    const a = Math.abs(t);
    const h = Math.floor(a / 3600);
    const m = Math.floor((a % 3600) / 60);
    const s = Math.floor(a % 60);
    return `${t >= 0 ? 'T+' : 'T−'}${pad2(h)}:${pad2(m)}:${pad2(s)}`;
  }

  /* truncate to maxW px with an ellipsis (measureText-driven) */
  function fitText(ctx, text, maxW) {
    let t = String(text);
    if (ctx.measureText(t).width <= maxW) return t;
    while (t.length > 1 && ctx.measureText(`${t}…`).width > maxW) t = t.slice(0, -1);
    return `${t}…`;
  }

  function drawComposite(canvas, sim) {
    if (!canvas || typeof canvas.getContext !== 'function') return;
    let W = canvas.width;
    let H = canvas.height;
    if (!(W > 0) || !(H > 0)) {
      /* controller presets 1280x720; fall back to layout size, then default */
      const r = typeof canvas.getBoundingClientRect === 'function'
        ? canvas.getBoundingClientRect() : null;
      W = r && r.width > 0 ? Math.round(r.width) : 1280;
      H = r && r.height > 0 ? Math.round(r.height) : 720;
      canvas.width = W;
      canvas.height = H;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    scopeStyle = scopeFor(null);   /* offscreen: the dark space palette */
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const mono = cssVar('--k-mono') || "'JetBrains Mono', ui-monospace, monospace";

    /* opaque background - a transparent canvas encodes to black in webm */
    ctx.fillStyle = cssVar('--k-bg');
    ctx.fillRect(0, 0, W, H);

    /* left square - relative view, clipped to its region */
    const side = Math.min(W, H);
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, side, side);
    ctx.clip();
    renderRelative(ctx, side, side, mono, sim);
    ctx.restore();

    /* hairline separator between the view and the right column */
    ctx.strokeStyle = cssVar('--k-border');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(side + 0.5, 0);
    ctx.lineTo(side + 0.5, H);
    ctx.stroke();

    if (W - side < 120) return; /* no room for the column (square canvas) */
    const colX = side + 24;
    const colR = W - 24;
    const colW = colR - colX;
    const cText = cssVar('--k-text');
    const cT2 = cssVar('--k-text-2');
    const cT3 = cssVar('--k-text-3');

    /* burned-in header: wordmark + risk-level word, hairline, pair names */
    const hdr = sim && sim.header && typeof sim.header === 'object' ? sim.header : {};
    const ephem = sim ? sim.ephem : null;
    const pName = hdr.primaryName
      || (ephem && ephem.primary && ephem.primary.name) || '—';
    const sName = hdr.secondaryName
      || (ephem && ephem.secondary && ephem.secondary.name) || '—';
    const lvl = typeof hdr.riskLevel === 'string' ? hdr.riskLevel.toLowerCase() : '';
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'left';
    ctx.font = `650 17px ${mono}`;
    ctx.fillStyle = cText;
    ctx.fillText('Kessler', colX, 44);
    if (lvl === 'critical' || lvl === 'elevated' || lvl === 'nominal') {
      /* the level WORD in its risk color - sanctioned (nominal has no risk
         color; it stays in a text token) */
      ctx.font = `600 13px ${mono}`;
      ctx.fillStyle = lvl === 'critical' ? cssVar('--k-critical')
        : lvl === 'elevated' ? cssVar('--k-elevated') : cT2;
      ctx.textAlign = 'right';
      ctx.fillText(lvl.charAt(0).toUpperCase() + lvl.slice(1), colR, 44);
      ctx.textAlign = 'left';
    }
    ctx.strokeStyle = cssVar('--k-border');
    ctx.beginPath();
    ctx.moveTo(colX, 58.5);
    ctx.lineTo(colR, 58.5);
    ctx.stroke();
    ctx.font = `600 15px ${mono}`;
    ctx.fillStyle = cText;
    ctx.fillText(fitText(ctx, String(pName), colW), colX, 88);
    ctx.font = `15px ${mono}`;
    ctx.fillStyle = cT2;
    ctx.fillText(fitText(ctx, `and ${String(sName)}`, colW), colX, 110);

    /* SEP / T-vs-TCA readouts */
    const sepM = sim && Number.isFinite(sim.sepKm) ? sim.sepKm * 1000 : NaN;
    ctx.font = `500 13px ${mono}`;
    ctx.fillStyle = cT3;
    ctx.fillText('Separation', colX, 168);
    ctx.font = `650 34px ${mono}`;
    ctx.fillStyle = cText;
    ctx.fillText(distLabel(sim, sepM), colX, 198);
    ctx.font = `500 13px ${mono}`;
    ctx.fillStyle = cT3;
    ctx.fillText('Time from closest approach', colX, 240);
    ctx.font = `600 24px ${mono}`;
    ctx.fillStyle = cText;
    ctx.fillText(fmtVsTca(sim && Number.isFinite(sim.tRel) ? sim.tRel : NaN), colX, 268);

    /* range strip, bottom-right */
    const rgX = side + 16;
    const rgW = W - rgX - 16;
    const rgH = Math.min(220, Math.floor(H * 0.3));
    const rgY = H - rgH - 20;
    if (rgW >= 40 && rgH >= 40) {
      ctx.font = `500 13px ${mono}`;
      ctx.fillStyle = cT3;
      ctx.fillText('Separation over time', rgX + 8, rgY - 8);
      ctx.save();
      ctx.translate(rgX, rgY);
      ctx.beginPath();
      ctx.rect(0, 0, rgW, rgH);
      ctx.clip();
      renderRange(ctx, rgW, rgH, mono, sim);
      ctx.restore();
    }
  }

  /* ================= space camera (chase) view =======================
     A perspective chase camera: the view axis runs from Earth's center
     through the pair midpoint, with the camera standing off BEYOND the
     pair looking back - Earth fills the background, the two objects ride
     in front of it with trails, markers and a live separation line.

     The rotation chain (yaw about z, then tilt) is byte-identical to
     drawContext's proj / globe.js projEarth, with yaw/tilt derived from
     the view axis - so KEarthTexture.drawHemisphere({spin:-yaw, tilt:-tilt})
     paints the exact hemisphere this camera faces, and markers projected
     here land on the right geography. Earth's disk uses the orthographic
     hemisphere radius (f·RE/z) - a knowing simplification at LEO
     distances; this is a situational display, not an optical model. */

  function drawCamera(canvas, sim) {
    const g2 = setup(canvas);
    if (!g2) return;
    const { ctx, W, H, mono } = g2;
    const ephem = sim ? sim.ephem : null;
    const g = gridOf(ephem);
    if (!g) {
      emptyState(ctx, W, H, mono, 'Camera view unavailable: no ephemeris');
      return;
    }
    /* current positions; a transient zero state (pre-interpolation frame)
       falls back to the grid's TCA row rather than an empty frame */
    let pr = sim && v3ok(sim.pr) && norm(sim.pr) > 1 ? sim.pr : g.pr[g.tcaIdx];
    let sr = sim && v3ok(sim.sr) && norm(sim.sr) > 1 ? sim.sr : g.sr[g.tcaIdx];
    if (!v3ok(pr) || !v3ok(sr)) {
      emptyState(ctx, W, H, mono, 'Camera view unavailable: no ephemeris');
      return;
    }
    /* chase framing rides the PRIMARY: away from TCA the pair can sit on
       opposite sides of the planet, so a pair-midpoint axis would fall
       inside Earth for most of the window. The primary's radius never
       does. */
    const midR = norm(pr);
    if (!(midR > RE_KM * 0.5)) {
      emptyState(ctx, W, H, mono, 'Camera view unavailable for this geometry');
      return;
    }
    const wq = [pr[0] / midR, pr[1] / midR, pr[2] / midR];
    /* Camera basis. drawContext's chain gives screen-right
       x1 = (cosY, sinY, 0) and screen-up (sinY·sinT, -cosY·sinT, cosT); the
       right-handed toward-viewer axis is right x up = -(y1·cosT + z·sinT),
       which is also the side globe.js projEarth / drawHemisphere treat as
       front (pjD = -y2). The first camera put the eye on +(y1·cosT + z·sinT):
       a left-handed view (mirror image) looking at the far hemisphere. So
       yaw/tilt are chosen to put wq on the -y2 axis:
         -(-sinY·cosT) = wq0,  -(cosY·cosT) = wq1,  -sinT = wq2
       -> tilt = -asin(wq2), yaw = atan2(wq0, -wq1); depth = -y2. */
    const tiltA = -Math.asin(Math.max(-1, Math.min(1, wq[2])));
    const yawA = Math.atan2(wq[0], -wq[1]);
    const cosY = Math.cos(yawA);
    const sinY = Math.sin(yawA);
    const cosT = Math.cos(tiltA);
    const sinT = Math.sin(tiltA);
    /* view space: [right, up, depth-toward-camera], right-handed */
    const view = (p) => {
      const x1 = p[0] * cosY + p[1] * sinY;
      const y1 = -p[0] * sinY + p[1] * cosY;
      return [x1, p[2] * cosT - y1 * sinT, -(y1 * cosT + p[2] * sinT)];
    };
    const D = midR + RE_KM * 0.35;   /* camera ~2200 km behind the pair */
    const cx = W / 2;
    const cy = H / 2;
    const f = Math.min(W, H) / 2;    /* 90° field of view */
    const project = (p) => {
      const v = view(p);
      const z = D - v[2];
      if (!(z > 1)) return null;     /* behind the camera */
      return [cx + (f * v[0]) / z, cy - (f * v[1]) / z, z];
    };

    /* starfield - deterministic (seeded LCG; Math.random would shimmer) */
    let seed = 20260724;
    const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
    ctx.fillStyle = cssVar('--k-text-3');
    for (let i = 0; i < 140; i++) {
      const sx = rnd() * W;
      const sy = rnd() * H;
      const b = rnd();
      ctx.globalAlpha = 0.2 + b * 0.45;
      const s = b > 0.9 ? 1.5 : 1;
      ctx.fillRect(sx, sy, s, s);
    }
    ctx.globalAlpha = 1;

    /* Earth - perspective horizon radius (f·RE/√(D²-RE²)), not the
       orthographic f·RE/D: the horizon circle a camera at distance D
       actually sees is wider, which kills the pasted-lens look. */
    const ec = project([0, 0, 0]);
    const eR = ec && D > RE_KM * 1.02
      ? (f * RE_KM) / Math.sqrt(D * D - RE_KM * RE_KM) : 0;
    /* soft atmosphere limb - a few fading strokes in place of a hard ring */
    if (ec && eR > 4) {
      ctx.strokeStyle = cssVar('--k-data-1');
      for (let i = 5; i >= 1; i--) {
        ctx.globalAlpha = 0.028 * (6 - i);
        ctx.lineWidth = i * 2.2;
        ctx.beginPath();
        ctx.arc(ec[0], ec[1], eR + i * 1.1, 0, TAU);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    let textured = false;
    if (ec && eR > 4 && typeof KEarthTexture !== 'undefined' && KEarthTexture) {
      try {
        if (typeof KEarthTexture.ready === 'function' && KEarthTexture.ready()
            && typeof KEarthTexture.drawHemisphere === 'function') {
          const es = contextEarthState(ephem, sim);
          if (es) {
            /* The hemisphere renderer is orthographic: asked to fill the
               horizon disc it would squeeze 90° of geography into a cap
               that really spans acos(RE/D) - the 'zoomed patch in a
               circle' artifact. Render the sphere at the cap-matched
               orthographic radius f·RE·D/(D²-RE²) and CLIP to the true
               horizon circle, so the disc holds exactly the geography a
               camera at this distance sees, out to grazing angle. */
            const eROrtho = (f * RE_KM * D) / (D * D - RE_KM * RE_KM);
            ctx.save();
            ctx.beginPath();
            ctx.arc(ec[0], ec[1], eR, 0, TAU);
            ctx.clip();
            KEarthTexture.drawHemisphere(ctx, {
              cx: ec[0],
              cy: ec[1],
              radiusPx: eROrtho,
              spin: -yawA,
              tilt: -tiltA,
              gmstRad: es.gmst,
              sunDirTEME: es.sun,
              quality: 1,
            });
            ctx.restore();
            textured = true;
          }
        } else if (typeof KEarthTexture.load === 'function') {
          const p = KEarthTexture.load();
          if (p && typeof p.catch === 'function') p.catch(() => {});
        }
      } catch (e) { /* texture fault - flat disc carries the frame */ }
    }
    if (ec && !textured) {
      ctx.fillStyle = cssVar('--k-surface');
      ctx.beginPath();
      ctx.arc(ec[0], ec[1], eR, 0, TAU);
      ctx.fill();
    }

    /* true sphere occlusion: P is hidden when the eye->P segment passes
       through the globe (slightly shrunk so the limb doesn't flicker) */
    const eye = [wq[0] * D, wq[1] * D, wq[2] * D];
    const RE2 = RE_KM * RE_KM * 0.996;
    const occluded = (p) => {
      const ex = p[0] - eye[0];
      const ey = p[1] - eye[1];
      const ez = p[2] - eye[2];
      const L = Math.hypot(ex, ey, ez);
      if (!(L > 1e-6)) return false;
      const t = -(eye[0] * ex + eye[1] * ey + eye[2] * ez) / L;
      if (t <= 0 || t >= L) return false;
      const qx = eye[0] + (ex / L) * t;
      const qy = eye[1] + (ey / L) * t;
      const qz = eye[2] + (ez / L) * t;
      return qx * qx + qy * qy + qz * qz < RE2;
    };

    /* stroke a 3-D polyline in two passes: hidden-behind-Earth segments
       first at whisper alpha (the ring reads COMPLETE), visible on top */
    const strokePath3 = (pts, color, base, closed) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      for (const pass of [{ hide: true, a: base * 0.22 }, { hide: false, a: base }]) {
        ctx.globalAlpha = pass.a;
        ctx.beginPath();
        let pen = false;
        const n2 = pts.length + (closed ? 1 : 0);
        for (let i = 0; i < n2; i++) {
          const p = pts[i % pts.length];
          const s = v3ok(p) ? project(p) : null;
          if (!s || occluded(p) !== pass.hide) { pen = false; continue; }
          if (!pen) { ctx.moveTo(s[0], s[1]); pen = true; } else ctx.lineTo(s[0], s[1]);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    };

    /* full orbit rings - the ±window ephemeris only covers part of an
       orbit, so close the loop with the in-plane circle through the
       current radius (exact for circular orbits, honest context for
       near-circular LEO); the true ephemeris arc draws brighter on top */
    const j = Math.max(1, Math.min(g.n - 2,
      Math.round((sim.tRel + g.half) / g.step)));
    const orbitRing = (rows) => {
      const r0 = rows[j];
      const ra = rows[j - 1];
      const rb = rows[j + 1];
      if (!v3ok(r0) || !v3ok(ra) || !v3ok(rb)) return null;
      const v = sub(rb, ra);
      const h = cross(r0, v);
      if (!(norm(h) > 1e-6)) return null;
      const rad = norm(r0);
      const e1 = unit(r0);
      const e2 = unit(cross(h, r0));
      const pts = [];
      for (let k = 0; k < 180; k++) {
        const a = (k / 180) * TAU;
        const ca = Math.cos(a);
        const sa = Math.sin(a);
        pts.push([
          rad * (e1[0] * ca + e2[0] * sa),
          rad * (e1[1] * ca + e2[1] * sa),
          rad * (e1[2] * ca + e2[2] * sa),
        ]);
      }
      return pts;
    };
    const ringP = orbitRing(g.pr);
    const ringS = orbitRing(g.sr);
    if (ringP) strokePath3(ringP, cssVar('--k-data-1'), 0.4, true);
    if (ringS) strokePath3(ringS, cssVar('--k-data-2'), 0.4, true);
    strokePath3(decimate(ephem.primary.r, 240), cssVar('--k-data-1'), 0.9, false);
    strokePath3(decimate(ephem.secondary.r, 240), cssVar('--k-data-2'), 0.9, false);

    /* the pair - markers, labels, live separation line. Behind-the-globe
       markers dim instead of vanishing, so the pair is never lost. */
    const P = project(pr);
    const S = project(sr);
    ctx.font = `600 12px ${mono}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    if (P && S && !occluded(pr) && !occluded(sr)) {
      const dx = S[0] - P[0];
      const dy = S[1] - P[1];
      if (dx * dx + dy * dy > 4) {
        ctx.strokeStyle = cssVar('--k-text-3');
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(P[0], P[1]);
        ctx.lineTo(S[0], S[1]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = cssVar('--k-text-2');
        ctx.fillText(fmtDistLocal(sim.sepKm * 1000),
          (P[0] + S[0]) / 2 + 6, (P[1] + S[1]) / 2 - 6);
      }
    }
    /* near TCA both markers share a pixel: stack the labels at fixed rows
       (P above, S below) so names and the separation text never collide */
    const tight = P && S &&
      (P[0] - S[0]) * (P[0] - S[0]) + (P[1] - S[1]) * (P[1] - S[1]) < 30 * 30;
    const marker = (s, p3, color, name, dy) => {
      if (!s) return;
      ctx.globalAlpha = occluded(p3) ? 0.3 : 1;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(s[0], s[1], 3, 0, TAU);
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(s[0], s[1], 7, 0, TAU);
      ctx.stroke();
      if (name) {
        ctx.fillStyle = cssVar('--k-text-2');
        let ly = tight ? (dy < 0 ? -14 : 14) : dy;
        const txt = String(name);
        const tw = ctx.measureText(txt).width;
        /* QA B4: never clip at the stage edge - flip above/left when the
           marker hugs the bottom/right, and stay off the corner meta row */
        if (s[1] + ly > H - 36) ly = -Math.max(14, Math.abs(ly));
        if (s[1] + ly < 12) ly = Math.abs(ly) || 14;
        let lx = s[0] + 11;
        if (lx + tw > W - 8) lx = s[0] - 11 - tw;
        lx = Math.max(8, lx);
        ctx.fillText(txt, lx, Math.min(Math.max(s[1] + ly, 12), H - 36));
      }
      ctx.globalAlpha = 1;
    };
    marker(P, pr, cssVar('--k-data-1'), ephem.primary && ephem.primary.name, -6);
    marker(S, sr, cssVar('--k-data-2'), ephem.secondary && ephem.secondary.name, 10);

    /* corner meta - chase subject altitude, camera standoff. Phones: the
       separation card owns the bottom edge, so the meta line is dropped */
    if (W < 560) return;
    ctx.fillStyle = cssVar('--k-text-3');
    ctx.font = `500 11px ${mono}`;
    ctx.textAlign = 'right';
    ctx.fillText(
      `Following ${ephem.primary && ephem.primary.name ? ephem.primary.name : 'the primary'}`
      + ` at ${Math.round(midR - RE_KM).toLocaleString('en-US')} km · camera`
      + ` ${Math.round(D - midR).toLocaleString('en-US')} km above`,
      W - 18, H - 22);
  }

  return { drawRelative, drawContext, drawRange, drawComposite, drawCamera };
})();
