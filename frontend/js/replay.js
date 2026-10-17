/* ===========================================================================
   KESSLER - Time Machine (replay.html)

   A story-driven replay of the 10 February 2009 Iridium 33 / Cosmos 2251
   collision, rendered with its own Canvas2D engine (the textured Earth comes
   from earthtex.js, read-only). Five chapters on one story clock:

     1  72 hours before   the replayed screening run flags the pair
     2  The encounter     SGP4 ephemeris, camera closes on the crossing
     3  Impact            flash + the modeled fragments leaving the point
     4  The cascade       fragment orbits spread: point -> ring -> shell
     5  Why Kessler exists

   Every number shown comes from the backend (/scenarios/{id}, /ephemeris,
   /debris). The client only interpolates the 1 s SGP4 grid and advances the
   backend's fragment elements with the same two-body + J2 + decay formulas
   the backend documents (breakup.py) for drawing.

   Projection: the globe engine's orthographic convention (docs/CONTRACTS.md)
   p1 = Rz(spin) p, p2 = Rx(tilt) p1, screen = (x1, -z2), depth = -y2, so
   KEarthTexture.drawHemisphere lines up with every overlay.

   Frame state is a pure function of story time T, so scrubbing, chapter
   jumps and ?t= / ?chapter= deep links all render identical frames.
   ======================================================================== */
'use strict';

(() => {
  const SID = 'iridium-cosmos-2009';
  const DEG = Math.PI / 180;
  const TWO_PI = Math.PI * 2;
  const RE = 6378.137;
  const MU = 398600.4418;
  const J2000_MS = Date.UTC(2000, 0, 1, 12, 0, 0);
  const TWO_YEARS_S = 2 * 365.25 * 86400;
  const CHECKPOINT_HOURS = [1, 24, 24 * 7, 24 * 30, 24 * 182, 24 * 365, 24 * 730];

  const mqReduce = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
  const reduceMotion = () => !!(mqReduce && mqReduce.matches);

  const CHAPTERS = [
    { key: 'before', title: '72 hours before', dur: 14 },
    { key: 'encounter', title: 'The encounter', dur: 18 },
    { key: 'impact', title: 'Impact', dur: 9 },
    { key: 'cascade', title: 'The cascade', dur: 22 },
    { key: 'why', title: 'Why Kessler exists', dur: 7 },
  ];
  let acc = 0;
  for (const c of CHAPTERS) { c.start = acc; acc += c.dur; }
  const TOTAL = acc;
  /* representative still per chapter (reduced motion + chapter previews) */
  const KEY_U = [0.82, 0.955, 0.42, 0.8, 0.6];

  const $ = (id) => document.getElementById(id);
  const el = {
    stage: $('tm-stage'), canvas: $('tm-canvas'),
    caption: $('tm-caption'), capNum: $('tm-cap-num'), capDate: $('tm-cap-date'),
    capTitle: $('tm-cap-title'), capBody: $('tm-cap-body'), capSource: $('tm-cap-source'),
    prov: $('tm-prov'), provText: $('tm-prov-text'),
    readout: $('tm-readout'),
    intro: $('tm-intro'), start: $('tm-start'), startLabel: $('tm-start-label'),
    introError: $('tm-intro-error'), introLede: $('tm-intro-lede'),
    end: $('tm-end'), replay: $('tm-replay'),
    transport: $('tm-transport'), play: $('tm-play'),
    clockMain: $('tm-clock-main'), clockSub: $('tm-clock-sub'),
    scrub: $('tm-scrub'), segs: $('tm-segs'), chapters: $('tm-chapters'),
    notes: $('tm-notes'),
  };
  const ctx = el.canvas.getContext('2d');
  const ecv = document.createElement('canvas');           // offscreen Earth layer
  const ectx = ecv.getContext('2d');

  /* ------------------------------------------------------------------------
     Formatting (numbers are backend values; this only renders them)
     ---------------------------------------------------------------------- */
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const MONTHS_LONG = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
    'August', 'September', 'October', 'November', 'December'];
  const pad = (n, w = 2) => String(n).padStart(w, '0');
  const nf = (n) => Number(n).toLocaleString('en-US');

  function fmtDate(ms, long) {
    const d = new Date(ms);
    return long
      ? `${d.getUTCDate()} ${MONTHS_LONG[d.getUTCMonth()]} ${d.getUTCFullYear()}`
      : `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
  }
  function fmtClock(ms, tenths) {
    const d = new Date(ms);
    let s = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
    if (tenths !== undefined) {
      s += `:${pad(d.getUTCSeconds())}`;
      if (tenths) s += `.${Math.floor(d.getUTCMilliseconds() / 100)}`;
    }
    return s;
  }
  function fmtHms(sec, tenths) {
    const neg = sec < 0;
    let s = Math.abs(sec);
    const h = Math.floor(s / 3600); s -= h * 3600;
    const m = Math.floor(s / 60); s -= m * 60;
    const whole = Math.floor(s);
    let out = `${pad(h)}:${pad(m)}:${pad(whole)}`;
    if (tenths) out += `.${Math.min(9, Math.floor((s - whole) * 10))}`;
    return { neg, text: out };
  }
  function fmtPcHtml(p) {
    if (p === null || p === undefined || !(p > 0)) return '—';
    let exp = Math.floor(Math.log10(p));
    let mant = p / Math.pow(10, exp);
    if (mant >= 9.95) { mant /= 10; exp += 1; }
    return `${mant.toFixed(1)} × 10<sup>${exp < 0 ? '−' : ''}${Math.abs(exp)}</sup>`;
  }
  function fmtDistHtml(m) {
    if (m === null || m === undefined || !Number.isFinite(m)) return '—';
    if (m < 10) return `${m.toFixed(1)}<small>m</small>`;
    if (m < 1000) return `${Math.round(m)}<small>m</small>`;
    const km = m / 1000;
    if (km < 10) return `${km.toFixed(2)}<small>km</small>`;
    if (km < 100) return `${km.toFixed(1)}<small>km</small>`;
    return `${nf(Math.round(km))}<small>km</small>`;
  }
  function fmtSince(sec) {
    if (sec < 90) return { v: Math.round(sec), u: 'seconds' };
    if (sec < 5400) return { v: Math.round(sec / 60), u: 'minutes' };
    if (sec < 2 * 86400) return { v: Math.round(sec / 3600), u: 'hours' };
    if (sec < 61 * 86400) return { v: Math.round(sec / 86400), u: 'days' };
    if (sec < 700 * 86400) return { v: Math.round(sec / (30.44 * 86400)), u: 'months' };
    return { v: +(sec / (365.25 * 86400)).toFixed(1), u: 'years' };
  }
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));

  /* ------------------------------------------------------------------------
     Math: easing, astro, orbits
     ---------------------------------------------------------------------- */
  const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
  const lerp = (a, b, t) => a + (b - a) * t;
  const smooth = (e0, e1, x) => { const t = clamp01((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t); };
  const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const logLerp = (a, b, t) => Math.exp(lerp(Math.log(a), Math.log(b), t));

  function gmstRad(ms) {
    const d = (ms - J2000_MS) / 86400000;
    return ((280.46061837 + 360.98564736629 * d) % 360) * DEG;
  }
  /* low-precision solar direction in ECI (same model as the globe engine) */
  function sunDirEci(ms, out) {
    const d = (ms - J2000_MS) / 86400000;
    const g = (357.529 + 0.98560028 * d) * DEG;
    const L = (280.459 + 0.98564736 * d) * DEG
      + (1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * DEG;
    const eps = (23.439 - 0.00000036 * d) * DEG;
    out[0] = Math.cos(L);
    out[1] = Math.cos(eps) * Math.sin(L);
    out[2] = Math.sin(eps) * Math.sin(L);
    return out;
  }

  /* Keplerian elements + J2 secular rates -> TEME position (km), written
     into out[o..o+2]. Same model as backend/app/scenarios/breakup.py. */
  function kepPos(a, e, inc, raan, argp, M, out, o) {
    let E = M;
    for (let k = 0; k < 6; k++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    const xp = a * (Math.cos(E) - e);
    const yp = a * Math.sqrt(1 - e * e) * Math.sin(E);
    const co = Math.cos(raan), so = Math.sin(raan);
    const cw = Math.cos(argp), sw = Math.sin(argp);
    const ci = Math.cos(inc), si = Math.sin(inc);
    out[o] = (co * cw - so * sw * ci) * xp + (-co * sw - so * cw * ci) * yp;
    out[o + 1] = (so * cw + co * sw * ci) * xp + (-so * sw + co * cw * ci) * yp;
    out[o + 2] = (sw * si) * xp + (cw * si) * yp;
  }
  function elPos(elm, t, out, o) {
    kepPos(elm.a_km, elm.e, elm.i_rad, elm.raan_rad + elm.raan_dot * t,
      elm.argp_rad + elm.argp_dot * t, (elm.m0_rad + elm.m_dot * t) % TWO_PI, out, o || 0);
    return out;
  }

  /* ------------------------------------------------------------------------
     Data
     ---------------------------------------------------------------------- */
  const D = {
    ready: false, scn: null, eph: null, deb: null, checkpoints: new Map(),
    tcaMs: 0, runAtMs: 0, P: [0, 0, 0], latP: 0, raP: 0,
    ephStartMs: 0, ephN: 0, ephP: null, ephS: null,
    parents: [], frag: null, nFrag: 0, orbitIdx: [],
  };

  async function getJSON(path) {
    const res = await fetch(path, { headers: { Accept: 'application/json' } });
    if (!res.ok) {
      let detail = res.statusText;
      try { const b = await res.json(); if (typeof b.detail === 'string') detail = b.detail; } catch (_) { /* keep */ }
      throw new Error(`${res.status} ${detail}`);
    }
    return res.json();
  }

  function ingest(scn, eph, deb) {
    D.scn = scn; D.eph = eph; D.deb = deb;
    const det = scn.screening && scn.screening.detection;
    D.tcaMs = Date.parse(eph.tca_precise || eph.tca);
    D.runAtMs = det ? Date.parse(det.run_at) : D.tcaMs - 72 * 3600e3;
    D.ephStartMs = Date.parse(eph.start_precise || eph.start);
    const n = Math.min(eph.primary.r.length, eph.secondary.r.length);
    D.ephN = n;
    D.ephP = new Float64Array(n * 3);
    D.ephS = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < 3; k++) {
        D.ephP[3 * i + k] = eph.primary.r[i][k];
        D.ephS[3 * i + k] = eph.secondary.r[i][k];
      }
    }
    const P = ephAt(D.ephP, 0, [0, 0, 0]);
    D.P = P;
    /* public TLEs leave the pair D.missKm apart at TCA, though the real
       objects collided: the drawing closes that gap at the contact point M
       (see contactW); every readout keeps the SGP4 numbers */
    const S0 = ephAt(D.ephS, 0, [0, 0, 0]);
    D.M = [(P[0] + S0[0]) / 2, (P[1] + S0[1]) / 2, (P[2] + S0[2]) / 2];
    D.shift = [[D.M[0] - P[0], D.M[1] - P[1], D.M[2] - P[2]],
      [D.M[0] - S0[0], D.M[1] - S0[1], D.M[2] - S0[2]]];
    D.missKm = Math.hypot(P[0] - S0[0], P[1] - S0[1], P[2] - S0[2]);
    const rP = Math.hypot(P[0], P[1], P[2]);
    D.latP = Math.asin(P[2] / rP);
    D.raP = Math.atan2(P[1], P[0]);

    const tr = scn.tracks;
    D.trk = null;
    if (tr && tr.primary.length > 2 && tr.secondary.length > 2) {
      const m = Math.min(tr.primary.length, tr.secondary.length);
      D.trk = { startMs: Date.parse(tr.start), step: tr.step_s, n: m,
        P: new Float64Array(m * 3), S: new Float64Array(m * 3) };
      for (let i = 0; i < m; i++) {
        for (let k = 0; k < 3; k++) {
          D.trk.P[3 * i + k] = tr.primary[i][k];
          D.trk.S[3 * i + k] = tr.secondary[i][k];
        }
      }
    }
    D.parents = deb.parents.map((p) => ({ norad: p.norad_id, name: p.name, el: p.elements }));
    const rows = deb.fragments;
    const f = deb.fragment_fields;
    const ix = (k) => f.indexOf(k);
    const N = rows.length;
    const cols = ['parent', 'length_m', 'a_km', 'e', 'i_rad', 'raan_rad', 'argp_rad', 'm0_rad',
      'raan_dot', 'argp_dot', 'm_dot', 'tau_s', 'scale_h_km'];
    const F = {};
    for (const c of cols) F[c] = new Float64Array(N);
    for (let r = 0; r < N; r++) {
      for (const c of cols) {
        const v = rows[r][ix(c)];
        F[c][r] = v === null ? Infinity : v;
      }
    }
    D.frag = F;
    D.nFrag = N;
    /* drawn size follows each fragment's modeled characteristic length
       (0.1 m -> 1 px, 2.5 m and up -> 2.7 px); glint phase and rate are a
       fixed per-fragment spin, so frames stay a pure function of T */
    D.fragSize = new Float32Array(N);
    D.fragPh = new Float32Array(N);
    D.fragW = new Float32Array(N);
    for (let r = 0; r < N; r++) {
      const L = Number.isFinite(F.length_m[r]) ? F.length_m[r] : 0.1;
      D.fragSize[r] = 1 + 1.7 * clamp01(Math.log10(Math.max(L, 0.1) / 0.1) / Math.log10(25));
      D.fragPh[r] = (r * 2.39996323) % TWO_PI;
      D.fragW[r] = 1.2 + ((r * 0.618034) % 1) * 2.2;
    }
    D.fragPos = new Float64Array(N * 3);
    D.fragPos2 = new Float64Array(N * 3);
    D.fragAlive = new Uint8Array(N);
    const step = Math.max(1, Math.round(N / 300));
    D.orbitIdx = [];
    for (let r = 0; r < N; r += step) D.orbitIdx.push(r);
    D.ready = true;
  }

  /* SGP4 track at the detecting run (30 s grid); times past the end wrap
     back one orbital period so a full ring can start anywhere */
  function trkAt(arr, ms, periodS, out) {
    const tk = D.trk;
    let x = (ms - tk.startMs) / 1000 / tk.step;
    const span = tk.n - 1;
    const per = periodS / tk.step;
    while (x > span && per > 0) x -= per;
    if (x < 0) x = 0;
    let i = Math.floor(x);
    if (i > span - 1) i = span - 1;
    const t = x - i;
    for (let k = 0; k < 3; k++) out[k] = arr[3 * i + k] + (arr[3 * i + 3 + k] - arr[3 * i + k]) * t;
    return out;
  }

  function ephAt(arr, offS, out) {
    /* offS: seconds from TCA; linear interpolation on the 1 s SGP4 grid */
    const x = (D.tcaMs + offS * 1000 - D.ephStartMs) / 1000;
    let i = Math.floor(x);
    if (i < 0) i = 0;
    if (i > D.ephN - 2) i = D.ephN - 2;
    const t = x - i;
    for (let k = 0; k < 3; k++) out[k] = arr[3 * i + k] + (arr[3 * i + 3 + k] - arr[3 * i + k]) * t;
    return out;
  }

  /* Contact. Over the last CONTACT_S seconds before TCA the DRAWN position
     of each object is pulled onto the contact point M, and the correction
     relaxes just as fast afterwards: the dots touch, the flash and both
     fragment streams start at one point, and the streams then follow their
     parents' real orbits. Readouts always use the uncorrected SGP4 states. */
  const CONTACT_S = 2.5;
  function contactW(off) {
    if (off <= -CONTACT_S) return 0;
    if (off <= 0) { const x = 1 + off / CONTACT_S; return x * x * (3 - 2 * x); }
    return Math.exp(-off / 2);
  }
  function ephAtC(arr, i, off, out) {
    ephAt(arr, off, out);
    const w = contactW(off);
    if (w > 0) { const sh = D.shift[i]; out[0] += w * sh[0]; out[1] += w * sh[1]; out[2] += w * sh[2]; }
    return out;
  }

  /* fragment positions at t seconds after the collision (NaN-free: dead
     fragments flagged in fragAlive) */
  function fragPositions(t, out) {
    const F = D.frag;
    const w = t >= 0 ? contactW(t) : 0;
    for (let r = 0; r < D.nFrag; r++) {
      const tau = F.tau_s[r];
      let drop = 0;
      if (Number.isFinite(tau)) {
        const x = 1 - t / tau;
        if (x <= 0) { D.fragAlive[r] = 0; continue; }
        drop = F.scale_h_km[r] * Math.log(x);
      }
      const a = F.a_km[r] + drop;
      const e = F.e[r];
      if (a * (1 - e) - RE < 120) { D.fragAlive[r] = 0; continue; }
      D.fragAlive[r] = 1;
      kepPos(a, e, F.i_rad[r], F.raan_rad[r] + F.raan_dot[r] * t,
        F.argp_rad[r] + F.argp_dot[r] * t, (F.m0_rad[r] + F.m_dot[r] * t) % TWO_PI, out, 3 * r);
      if (w > 0) {
        const sh = D.shift[F.parent[r] ? 1 : 0];
        out[3 * r] += w * sh[0]; out[3 * r + 1] += w * sh[1]; out[3 * r + 2] += w * sh[2];
      }
    }
  }

  /* ------------------------------------------------------------------------
     Palette (space layer reads tokens from .tm-stage; chrome from KTheme)
     ---------------------------------------------------------------------- */
  const C = {};
  function readPalette() {
    const cs = getComputedStyle(el.stage);
    const g = (n, fb) => (cs.getPropertyValue(n).trim() || fb);
    C.space = g('--tm-space', '#000');
    C.text = g('--tm-space-text', '#f5f5f7');
    C.text2 = g('--tm-space-text-2', 'rgba(245,245,247,.72)');
    C.text3 = g('--tm-space-text-3', 'rgba(245,245,247,.52)');
    C.line = g('--tm-space-line', 'rgba(245,245,247,.16)');
    C.accent = g('--tm-space-accent', '#2997ff');
    C.second = g('--tm-space-secondary', '#d9dde3');
    C.fragA = g('--tm-space-frag-a', '#8cc4ff');      // Iridium 33's stream
    C.fragB = g('--tm-space-frag-b', '#f0e3cf');      // Cosmos 2251's stream
    C.elevated = g('--tm-space-elevated', '#ff9f0a');
    C.critical = g('--tm-space-critical', '#ff453a');
    C.atmo = g('--tm-space-atmo', '120, 170, 255');
    const rs = getComputedStyle(document.documentElement);
    C.font = (rs.getPropertyValue('--k-sans').trim() || 'Inter, system-ui, sans-serif');
  }

  /* ------------------------------------------------------------------------
     Canvas sizing
     ---------------------------------------------------------------------- */
  let W = 0, H = 0, dpr = 1, mobile = false;
  function resize() {
    const r = el.stage.getBoundingClientRect();
    W = Math.max(1, Math.round(r.width));
    H = Math.max(1, Math.round(r.height));
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    el.canvas.width = Math.round(W * dpr);
    el.canvas.height = Math.round(H * dpr);
    mobile = W < 761;
    introAx = introClearAx();
    buildStars();
    dirty = true;
  }

  /* Intro framing (desktop): push the Earth right just far enough that its
     limb clears the title and lede blocks, instead of a fixed offset that
     lets the paragraph run over the planet. Capped so the planet stays on
     screen; RE / fov mirror setProjection for the chapter-0 camera. */
  let introAx = 0.64;
  function introClearAx() {
    if (mobile || W <= 1000 || !el.intro || el.intro.hidden) return 0.64;
    const sr = el.stage.getBoundingClientRect();
    const R = (Math.min(W, H * 1.25) / 21500) * RE;
    const cy = 0.5 * H;
    let cx = 0.64 * W;
    for (const node of el.intro.querySelectorAll('.tm-intro__title, .tm-intro__lede, .tm-intro__actions')) {
      const r = node.getBoundingClientRect();
      if (!r.width) continue;
      /* the text's actual right edge (block boxes span their max-width) */
      const range = document.createRange();
      range.selectNodeContents(node);
      const tr = range.getBoundingClientRect();
      const right = (tr.width ? tr.right : r.right) - sr.left;
      const top = r.top - sr.top, bot = r.bottom - sr.top;
      const dy = cy < top ? top - cy : cy > bot ? cy - bot : 0;
      if (dy >= R) continue;
      cx = Math.max(cx, right + 28 + Math.sqrt(R * R - dy * dy));
    }
    return Math.min(0.74, cx / W);
  }

  /* ------------------------------------------------------------------------
     Stars (fixed celestial directions; perspective-projected so they drift
     with the camera orientation)
     ---------------------------------------------------------------------- */
  let stars = null;
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function buildStars() {
    if (stars) return;
    const rnd = mulberry32(20090210);
    const n = 900;
    stars = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      const z = rnd() * 2 - 1, ph = rnd() * TWO_PI, s = Math.sqrt(1 - z * z);
      stars[4 * i] = s * Math.cos(ph);
      stars[4 * i + 1] = s * Math.sin(ph);
      stars[4 * i + 2] = z;
      const m = rnd();
      stars[4 * i + 3] = m * m * m;                     // brightness, mostly faint
    }
  }

  /* ------------------------------------------------------------------------
     Camera + projection
     ---------------------------------------------------------------------- */
  const cam = { spin: 0, tilt: 0, fov: 18000, fx: 0, fy: 0, fz: 0, ax: 0.5, ay: 0.5, earthA: 1 };
  let cS = 1, sS = 0, cT = 1, sT = 0, sc = 1, ecx = 0, ecy = 0;
  let pjX = 0, pjY = 0, pjD = 0, pjR2 = 0;
  function setProjection() {
    cS = Math.cos(cam.spin); sS = Math.sin(cam.spin);
    cT = Math.cos(cam.tilt); sT = Math.sin(cam.tilt);
    sc = Math.min(W, H * (mobile ? 1.05 : 1.25)) / cam.fov;
    const x1 = cam.fx * cS - cam.fy * sS;
    const y1 = cam.fx * sS + cam.fy * cS;
    const z2 = y1 * sT + cam.fz * cT;
    ecx = cam.ax * W - sc * x1;
    ecy = cam.ay * H + sc * z2;
  }
  /* project ECI km -> screen px; pjD > 0 is toward the viewer */
  function proj(x, y, z) {
    const x1 = x * cS - y * sS;
    const y1 = x * sS + y * cS;
    const y2 = y1 * cT - z * sT;
    const z2 = y1 * sT + z * cT;
    pjX = ecx + sc * x1;
    pjY = ecy - sc * z2;
    pjD = -y2;
    pjR2 = x1 * x1 + z2 * z2;
  }
  const visible = () => pjD > 0 || pjR2 > RE * RE;

  /* ------------------------------------------------------------------------
     Story: frame state as a pure function of story time T
     ---------------------------------------------------------------------- */
  function chapterAt(T) {
    for (let i = CHAPTERS.length - 1; i >= 0; i--) if (T >= CHAPTERS[i].start) return i;
    return 0;
  }

  /* orientation presets (placeholders until the scenario loads, so the
     intro shows the planet while data is in flight) */
  const O = {
    c0a: { spin: -2.2, tilt: 18 * DEG, fov: 21500 },
    c0: { spin: -2.2, tilt: 18 * DEG, fov: 21500 },
  };
  const wrapPi = (a) => { a = (a + Math.PI) % TWO_PI; if (a < 0) a += TWO_PI; return a - Math.PI; };
  function presets() {
    const face = -Math.PI / 2 - D.raP;
    /* chapter 1 opens on the two satellites where they are at the run
       (SGP4), then turns to the predicted crossing */
    let spinObj = face, tiltObj = 20 * DEG;
    if (D.trk) {
      const ms = D.runAtMs + 150 * 1000 * 2;
      trkAt(D.trk.P, ms, periodS(0), tmpA);
      trkAt(D.trk.S, ms, periodS(1), tmpB);
      const mx = tmpA[0] + tmpB[0], my = tmpA[1] + tmpB[1], mz = tmpA[2] + tmpB[2];
      spinObj = -Math.PI / 2 - Math.atan2(my, mx);
      tiltObj = Math.max(-5 * DEG, Math.min(30 * DEG, Math.atan2(mz, Math.hypot(mx, my))));
    }
    const spinP = face - 0.35;
    O.c0a = { spin: spinP + wrapPi(spinObj - spinP), tilt: tiltObj, fov: 21500 };
    O.c0 = { spin: spinP, tilt: 30 * DEG, fov: 21500 };
    O.ce = { spin: face - 0.12, tilt: D.latP - 36 * DEG };
    O.c3 = { spin: face - 0.9, tilt: 31 * DEG, fov: 24500 };
  }
  function periodS(i) { return TWO_PI / D.parents[i].el.m_dot; }

  const tmpA = [0, 0, 0], tmpB = [0, 0, 0];
  function centroid(offS, out) {
    elPos(D.parents[0].el, offS, tmpA, 0);
    elPos(D.parents[1].el, offS, tmpB, 0);
    out[0] = (tmpA[0] + tmpB[0]) / 2; out[1] = (tmpA[1] + tmpB[1]) / 2; out[2] = (tmpA[2] + tmpB[2]) / 2;
    return out;
  }

  const F = { ch: 0, u: 0, tl: 0, off: 0, scnMs: 0 };
  const cen = [0, 0, 0];
  function frameAt(T) {
    const ch = chapterAt(T);
    const c = CHAPTERS[ch];
    const tl = T - c.start;
    const u = clamp01(tl / c.dur);
    F.ch = ch; F.u = u; F.tl = tl;
    cam.ax = 0.5;
    cam.ay = mobile ? (started ? 0.52 : 0.3) : 0.5;
    cam.earthA = 1;
    if (ch === 0) {
      F.scnMs = D.runAtMs + tl * 150 * 1000;
      F.off = (F.scnMs - D.tcaMs) / 1000;
      const b = easeInOut(smooth(0.1, 0.56, u));
      cam.spin = lerp(O.c0a.spin, O.c0.spin, b) + 0.05 * u;
      cam.tilt = lerp(O.c0a.tilt, O.c0.tilt, b);
      cam.fov = O.c0.fov;
      cam.fx = cam.fy = cam.fz = 0;
      if (!started && !mobile && W > 1000) cam.ax = introAx;
    } else if (ch === 1) {
      F.off = -600 * Math.pow(1 - u, 3);
      F.scnMs = D.tcaMs + F.off * 1000;
      const b = easeInOut(smooth(0, 0.5, u));
      const endSpin = O.c0.spin + 0.05;
      cam.spin = lerp(endSpin, O.ce.spin, b);
      cam.tilt = lerp(O.c0.tilt, O.ce.tilt, b);
      const target = 11 + 16500 * Math.pow(1 - u, 2.4);
      cam.fov = Math.min(O.c0.fov, logLerp(O.c0.fov, target, smooth(0, 0.5, u)));
      cam.fx = D.M[0] * b; cam.fy = D.M[1] * b; cam.fz = D.M[2] * b;
      cam.earthA = lerp(0.07, 1, smooth(Math.log(260), Math.log(2600), Math.log(cam.fov)));
    } else if (ch === 2) {
      F.off = 300 * (Math.exp(6 * u) - 1) / (Math.exp(6) - 1);
      F.scnMs = D.tcaMs + F.off * 1000;
      centroid(F.off, cen);
      const b = easeInOut(smooth(0.45, 1, u));
      cam.spin = lerp(O.ce.spin, O.c3.spin, b);
      cam.tilt = lerp(O.ce.tilt, O.c3.tilt, b);
      cam.fov = 11 + 3.4 * 11.7 * F.off;
      cam.fx = cen[0]; cam.fy = cen[1]; cam.fz = cen[2];
      cam.earthA = lerp(0.07, 1, smooth(Math.log(260), Math.log(2600), Math.log(cam.fov)));
    } else if (ch === 3) {
      F.off = 300 * Math.pow(TWO_YEARS_S / 300, u);
      F.scnMs = D.tcaMs + F.off * 1000;
      centroid(300, cen);
      const b = easeInOut(smooth(0, 0.14, u));
      cam.spin = O.c3.spin + 0.55 * u;
      cam.tilt = O.c3.tilt;
      if (!mobile && W > 1100) cam.ax = lerp(0.5, 0.56, b);
      cam.fov = logLerp(11 + 3.4 * 11.7 * 300, O.c3.fov, b);
      cam.fx = cen[0] * (1 - b); cam.fy = cen[1] * (1 - b); cam.fz = cen[2] * (1 - b);
    } else {
      F.off = TWO_YEARS_S;
      F.scnMs = D.tcaMs + F.off * 1000;
      cam.spin = O.c3.spin + 0.55 + 0.06 * tl;
      cam.tilt = O.c3.tilt;
      if (!mobile && W > 1100) cam.ax = 0.56;
      cam.fov = O.c3.fov;
      cam.fx = cam.fy = cam.fz = 0;
    }
    return F;
  }

  /* Earth's displayed rotation: true GMST until the cascade, where months
     pass per second - there the planet turns slowly for legibility instead
     of strobing (fragments stay in the inertial frame either way). */
  function displayGmst(f) {
    if (f.ch <= 2) return gmstRad(f.scnMs);
    const base = gmstRad(D.tcaMs + 300e3);
    const tl = f.ch === 3 ? f.tl : CHAPTERS[3].dur + f.tl;
    return base + 0.05 * tl;
  }
  const sunV = [1, 0, 0];

  /* ------------------------------------------------------------------------
     Rendering
     ---------------------------------------------------------------------- */
  let lastCamKey = '';
  let earthRes = 1;
  function drawEarth(f, moving) {
    let want = moving ? 0.6 : Math.min(dpr, 1.5);
    /* deep zoom magnifies the (nearest-sampled) night texture into blocks:
       render at about one texel per pixel and let the upscale smooth it */
    const texelPx = (TWO_PI * RE / 3600) * sc;
    if (texelPx > 1.2) want = Math.max(0.02, Math.min(want, 1.2 / texelPx));
    want = Math.round(want * 1000) / 1000;
    if (want !== earthRes || ecv.width !== Math.round(W * want) || ecv.height !== Math.round(H * want)) {
      earthRes = want;
      ecv.width = Math.round(W * want);
      ecv.height = Math.round(H * want);
    }
    const rpx = RE * sc;
    ectx.setTransform(earthRes, 0, 0, earthRes, 0, 0);
    ectx.clearRect(0, 0, W, H);
    let ok = false;
    if (typeof KEarthTexture !== 'undefined' && KEarthTexture.ready()) {
      const sunMs = f.ch >= 3 ? D.tcaMs : f.scnMs;
      ok = KEarthTexture.drawHemisphere(ectx, {
        cx: ecx, cy: ecy, radiusPx: rpx, spin: cam.spin, tilt: cam.tilt,
        gmstRad: displayGmst(f), sunDirTEME: sunDirEci(sunMs, sunV),
      });
    }
    ctx.save();
    ctx.globalAlpha = cam.earthA;
    if (ok) {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(ecv, 0, 0, ecv.width, ecv.height, 0, 0, W, H);
    } else {
      ctx.fillStyle = '#0b0d12';
      ctx.beginPath();
      ctx.arc(ecx, ecy, rpx, 0, TWO_PI);
      ctx.fill();
    }
    ctx.restore();
    /* atmosphere rim: a thin glow just outside the limb */
    if (rpx < 40000) {
      const w = Math.max(3, rpx * 0.028);
      const g = ctx.createRadialGradient(ecx, ecy, rpx - 1, ecx, ecy, rpx + w);
      g.addColorStop(0, `rgba(${C.atmo}, ${0.34 * cam.earthA})`);
      g.addColorStop(0.35, `rgba(${C.atmo}, ${0.12 * cam.earthA})`);
      g.addColorStop(1, `rgba(${C.atmo}, 0)`);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(ecx, ecy, rpx + w, 0, TWO_PI);
      ctx.arc(ecx, ecy, rpx - 1, 0, TWO_PI, true);
      ctx.fill();
    }
  }

  function drawStars(alpha) {
    const f = Math.max(W, H) * 0.9;
    ctx.fillStyle = C.text;
    for (let i = 0; i < stars.length; i += 4) {
      const x = stars[i], y = stars[i + 1], z = stars[i + 2];
      const x1 = x * cS - y * sS;
      const y1 = x * sS + y * cS;
      const y2 = y1 * cT - z * sT;
      if (y2 < 0.25) continue;                         // stars behind the viewer
      const z2 = y1 * sT + z * cT;
      const sx = W / 2 + f * x1 / y2;
      const sy = H / 2 - f * z2 / y2;
      if (sx < 0 || sy < 0 || sx > W || sy > H) continue;
      const b = stars[i + 3];
      ctx.globalAlpha = alpha * (0.12 + 0.6 * b);
      const s = b > 0.7 ? 1.6 : 1.05;
      ctx.fillRect(sx, sy, s, s);
    }
    ctx.globalAlpha = 1;
  }

  /* one orbit of SGP4 track ahead of `ms`, drawn to `frac`; a brighter
     sweep runs along it while the screening scan is in progress */
  function trackRing(arr, ms, perS, color, alpha, frac, u) {
    if (frac <= 0) return;
    const n = 160;
    const upto = Math.round(n * frac);
    const pts = [];
    for (let k = 0; k <= upto; k++) {
      trkAt(arr, ms + (k / n) * perS * 1000, perS, ringPt);
      proj(ringPt[0], ringPt[1], ringPt[2]);
      pts.push(visible() ? [pjX, pjY, pjD > 0] : null);
    }
    ctx.lineWidth = 1.3;
    ctx.strokeStyle = color;
    for (let pass = 0; pass < 2; pass++) {
      ctx.globalAlpha = alpha * (pass === 0 ? 1 : 0.4);
      ctx.beginPath();
      let pen = false;
      for (const p of pts) {
        if (!p || p[2] !== (pass === 0)) { pen = false; continue; }
        if (!pen) { ctx.moveTo(p[0], p[1]); pen = true; } else ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
    }
    /* scan sweep: three laps while the look-ahead runs */
    const scan = clamp01((u - 0.12) / 0.4);
    if (scan > 0 && scan < 1 && !reduceMotion()) {
      const head = (easeInOut(scan) * 3) % 1;
      const len = 0.12;
      ctx.lineWidth = 2.4;
      for (let k = 0; k < 24; k++) {
        const a0 = head - len * (k + 1) / 24, a1 = head - len * k / 24;
        trkAt(arr, ms + ((a0 + 1) % 1) * perS * 1000, perS, pa);
        proj(pa[0], pa[1], pa[2]);
        if (!visible()) continue;
        const x0 = pjX, y0 = pjY;
        trkAt(arr, ms + ((a1 + 1) % 1) * perS * 1000, perS, pa);
        proj(pa[0], pa[1], pa[2]);
        if (!visible()) continue;
        ctx.globalAlpha = (1 - k / 24) * 0.9;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(pjX, pjY); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  /* orbit ring from elements at time t (seconds from TCA) */
  const ringPt = [0, 0, 0];
  function strokeOrbit(elm, t, color, alpha, width, drawFrac, dash) {
    const n = 180;
    const raan = elm.raan_rad + elm.raan_dot * t;
    const argp = elm.argp_rad + elm.argp_dot * t;
    const upto = Math.max(2, Math.round(n * (drawFrac === undefined ? 1 : drawFrac)));
    /* start the ring at the object's current anomaly so draw-in follows it */
    const m0 = (elm.m0_rad + elm.m_dot * t) % TWO_PI;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash || []);
    let pen = false, front = true;
    const paths = [[], []];
    for (let k = 0; k <= upto; k++) {
      const M = m0 + (k / n) * TWO_PI;
      kepPos(elm.a_km, elm.e, elm.i_rad, raan, argp, M, ringPt, 0);
      proj(ringPt[0], ringPt[1], ringPt[2]);
      const vis = visible();
      if (!vis) { pen = false; continue; }
      const isFront = pjD > 0;
      if (!pen || isFront !== front) {
        paths[isFront ? 0 : 1].push([pjX, pjY]);
        paths[isFront ? 0 : 1].push(null);
      }
      paths[isFront ? 0 : 1].push([pjX, pjY]);
      pen = true; front = isFront;
    }
    for (let side = 0; side < 2; side++) {
      ctx.globalAlpha = alpha * (side === 0 ? 1 : 0.45);
      ctx.beginPath();
      let move = true;
      for (const p of paths[side]) {
        if (p === null) { move = true; continue; }
        if (move) { ctx.moveTo(p[0], p[1]); move = false; } else ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
  }

  function glowDot(x, y, color, r, glow) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r * glow);
    g.addColorStop(0, color);
    g.addColorStop(0.25, color);
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.globalAlpha = 0.35;
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, r * glow, 0, TWO_PI);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, TWO_PI);
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.55)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  function label(x, y, name, sub, color, align) {
    ctx.font = `600 13px ${C.font}`;
    ctx.textBaseline = 'middle';
    const right = align !== 'left';
    const dx = right ? 12 : -12;
    ctx.textAlign = right ? 'left' : 'right';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 4;
    ctx.strokeStyle = 'rgba(0,0,0,0.72)';
    ctx.strokeText(name, x + dx, y - 7);
    ctx.fillStyle = color || C.text;
    ctx.fillText(name, x + dx, y - 7);
    if (sub) {
      ctx.font = `500 12px ${C.font}`;
      ctx.strokeText(sub, x + dx, y + 9);
      ctx.fillStyle = C.text2;
      ctx.fillText(sub, x + dx, y + 9);
    }
  }

  function drawMarker(x, y, pulse, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(x, y, 9, 0, TWO_PI);
    ctx.stroke();
    if (pulse > 0) {
      ctx.globalAlpha = (1 - pulse) * 0.8;
      ctx.beginPath();
      ctx.arc(x, y, 9 + pulse * 26, 0, TWO_PI);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 2.4, 0, TWO_PI);
    ctx.fill();
  }

  const pa = [0, 0, 0], pb = [0, 0, 0];
  function drawTrail(arr, i, offS, spanS, color, width, alpha) {
    /* SGP4 grid samples from off-span to off as a fading polyline */
    const steps = 48;
    let px = null, py = null;
    ctx.lineCap = 'round';
    for (let k = 0; k <= steps; k++) {
      const t = offS - spanS * (1 - k / steps);
      ephAtC(arr, i, t, pa);
      proj(pa[0], pa[1], pa[2]);
      if (!visible()) { px = null; continue; }
      if (px !== null) {
        ctx.globalAlpha = alpha * Math.pow(k / steps, 1.6);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(pjX, pjY);
        ctx.stroke();
      }
      px = pjX; py = pjY;
    }
    ctx.globalAlpha = 1;
  }

  function drawPath(arr, i, fromS, toS, color, alpha, dash) {
    const steps = 60;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.setLineDash(dash || []);
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    let pen = false;
    for (let k = 0; k <= steps; k++) {
      ephAtC(arr, i, lerp(fromS, toS, k / steps), pa);
      proj(pa[0], pa[1], pa[2]);
      if (!visible()) { pen = false; continue; }
      if (!pen) { ctx.moveTo(pjX, pjY); pen = true; } else ctx.lineTo(pjX, pjY);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  /* Debris, drawn the way the cloud would look: each fragment sized by its
     modeled length, tinted by the parent it came from (the two streams keep
     their parents' momentum), glinting as it tumbles in sunlight and dimmed
     inside Earth's shadow. xray > 0 draws fragments hidden behind Earth
     faintly through it, so the young streams stay readable while they round
     the far side. Glints run on story time, so a scrub never strobes. */
  const sunF = [1, 0, 0];
  function drawFragments(t, alpha, streakFrac, size, xray) {
    fragPositions(t, D.fragPos);
    const hasStreak = streakFrac > 0;
    if (hasStreak) fragPositions(t * (1 - streakFrac), D.fragPos2);
    sunDirEci(D.tcaMs + t * 1000, sunF);
    const T = CHAPTERS[F.ch].start + F.tl;
    const k = size / 1.5;
    const tint = [C.fragA, C.fragB];
    const still = reduceMotion();
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineWidth = 1;
    const P = D.fragPos, Q = D.fragPos2, Fr = D.frag;
    for (let side = 0; side < 2; side++) {
      ctx.fillStyle = tint[side];
      ctx.strokeStyle = tint[side];
      for (let r = 0; r < D.nFrag; r++) {
        if (!D.fragAlive[r] || (Fr.parent[r] ? 1 : 0) !== side) continue;
        const X = P[3 * r], Y = P[3 * r + 1], Z = P[3 * r + 2];
        proj(X, Y, Z);
        const hidden = !visible();
        if (hidden && !(xray > 0)) continue;
        const x = pjX, y = pjY;
        if (x < -20 || y < -20 || x > W + 20 || y > H + 20) continue;
        /* Earth's shadow: the cylinder behind the planet, away from the Sun */
        const along = X * sunF[0] + Y * sunF[1] + Z * sunF[2];
        const lit = along >= 0 || (X * X + Y * Y + Z * Z - along * along) > RE * RE;
        const glint = still ? 0.8
          : 0.62 + 0.38 * Math.pow(Math.max(0, Math.sin(D.fragPh[r] + T * D.fragW[r])), 8);
        const op = alpha * (hidden ? xray : pjD > 0 ? 1 : 0.5) * (lit ? glint : 0.45);
        if (hasStreak) {
          proj(Q[3 * r], Q[3 * r + 1], Q[3 * r + 2]);
          ctx.globalAlpha = op * 0.3;
          ctx.beginPath();
          ctx.moveTo(pjX, pjY);
          ctx.lineTo(x, y);
          ctx.stroke();
        }
        ctx.globalAlpha = op;
        const sz = D.fragSize[r] * k;
        if (sz >= 1.9) {
          ctx.beginPath();
          ctx.arc(x, y, sz / 2, 0, TWO_PI);
          ctx.fill();
        } else {
          ctx.fillRect(x - sz / 2, y - sz / 2, sz, sz);
        }
      }
    }
    ctx.restore();
  }

  const RING_N = 72;
  const RING_COS = new Float64Array(RING_N + 1);
  const RING_SIN = new Float64Array(RING_N + 1);
  for (let k = 0; k <= RING_N; k++) {
    RING_COS[k] = Math.cos((k / RING_N) * TWO_PI);
    RING_SIN[k] = Math.sin((k / RING_N) * TWO_PI);
  }
  function drawFragmentOrbits(t, alpha) {
    if (alpha <= 0.002) return;
    const Fr = D.frag;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.strokeStyle = C.text;
    ctx.lineWidth = 0.7;
    ctx.globalAlpha = alpha;
    const n = RING_N;
    ctx.beginPath();
    for (const r of D.orbitIdx) {
      const tau = Fr.tau_s[r];
      let drop = 0;
      if (Number.isFinite(tau)) {
        const x = 1 - t / tau;
        if (x <= 0) continue;
        drop = Fr.scale_h_km[r] * Math.log(x);
      }
      const a = Fr.a_km[r] + drop;
      const e = Fr.e[r];
      if (a * (1 - e) - RE < 120) continue;
      /* a closed ring needs no Kepler solve: sweep the eccentric anomaly
         directly through the perifocal -> TEME rotation */
      const inc = Fr.i_rad[r];
      const raan = Fr.raan_rad[r] + Fr.raan_dot[r] * t;
      const argp = Fr.argp_rad[r] + Fr.argp_dot[r] * t;
      const co = Math.cos(raan), so = Math.sin(raan);
      const cw = Math.cos(argp), sw = Math.sin(argp);
      const ci = Math.cos(inc), si = Math.sin(inc);
      const px = co * cw - so * sw * ci, qx = -co * sw - so * cw * ci;
      const py = so * cw + co * sw * ci, qy = -so * sw + co * cw * ci;
      const pz = sw * si, qz = cw * si;
      const b = a * Math.sqrt(1 - e * e);
      let pen = false;
      for (let k = 0; k <= n; k++) {
        const xp = a * (RING_COS[k] - e), yp = b * RING_SIN[k];
        proj(px * xp + qx * yp, py * xp + qy * yp, pz * xp + qz * yp);
        if (!visible()) { pen = false; continue; }
        if (!pen) { ctx.moveTo(pjX, pjY); pen = true; } else ctx.lineTo(pjX, pjY);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawShell(alpha, withLabel) {
    if (alpha <= 0.01 || !D.deb) return;
    const band = D.deb.altitude_band_km;
    ctx.save();
    ctx.strokeStyle = C.text;
    ctx.setLineDash([2, 5]);
    ctx.lineWidth = 1;
    for (const h of band) {
      ctx.globalAlpha = alpha * 0.45;
      ctx.beginPath();
      ctx.arc(ecx, ecy, (RE + h) * sc, 0, TWO_PI);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    if (!withLabel || mobile) { ctx.restore(); return; }
    const rOut = (RE + band[1]) * sc;
    const ang = Math.PI * 0.74;
    const lx = ecx + Math.cos(ang) * rOut, ly = ecy + Math.sin(ang) * rOut;
    ctx.globalAlpha = alpha;
    ctx.font = `600 12.5px ${C.font}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = C.text2;
    ctx.fillText(`${nf(band[0])}–${nf(band[1])} km shell`, lx - 10, ly + 6);
    ctx.restore();
  }

  function drawScaleBar() {
    const kmPerPx = 1 / sc;
    const target = 120 * kmPerPx;
    const choices = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
    let km = choices[0];
    for (const c of choices) if (c <= target) km = c;
    const px = km * sc;
    const x = mobile ? W - 16 - px : 32;
    const y = mobile ? H * 0.6 : H - 112;
    ctx.save();
    ctx.strokeStyle = C.text2;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, y - 4); ctx.lineTo(x, y); ctx.lineTo(x + px, y); ctx.lineTo(x + px, y - 4);
    ctx.stroke();
    ctx.font = `500 12px ${C.font}`;
    ctx.fillStyle = C.text2;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(km < 1 ? `${km * 1000} m` : `${nf(km)} km`, x, y - 7);
    ctx.restore();
  }

  function render(moving) {
    const f = F;
    setProjection();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = C.space;
    ctx.fillRect(0, 0, W, H);
    drawStars(cam.fov > 3000 ? 1 : smooth(300, 3000, cam.fov));
    drawEarth(f, moving);
    if (!D.ready) return;

    const ip = D.parents[0], sp = D.parents[1];
    const P = D.P;
    if (f.ch === 0) {
      /* current orbits (SGP4 at the run), drawn ahead of each satellite
         as the screening propagates forward */
      const ring = smooth(0.04, 0.5, f.u);
      const lock = smooth(0.52, 0.58, f.u);
      if (D.trk) {
        trackRing(D.trk.P, f.scnMs, periodS(0), C.accent, 0.8, ring, f.u);
        trackRing(D.trk.S, f.scnMs, periodS(1), C.second, 0.65, ring, f.u);
      }
      if (lock > 0) {
        /* the same orbits 72 h later, when their paths cross */
        strokeOrbit(ip.el, 0, C.accent, 0.5 * lock, 1, 1, [3, 5]);
        strokeOrbit(sp.el, 0, C.second, 0.4 * lock, 1, 1, [3, 5]);
      }
      if (D.trk) {
        trkAt(D.trk.P, f.scnMs, periodS(0), pa);
        trkAt(D.trk.S, f.scnMs, periodS(1), pb);
        proj(pa[0], pa[1], pa[2]);
        if (visible()) { glowDot(pjX, pjY, C.accent, 3.6, 5); label(pjX, pjY, 'Iridium 33', 'active', C.text); }
        proj(pb[0], pb[1], pb[2]);
        if (visible()) { glowDot(pjX, pjY, C.second, 3.4, 5); label(pjX, pjY, 'Cosmos 2251', 'defunct', C.text); }
      }
      if (lock > 0) {
        proj(D.M[0], D.M[1], D.M[2]);
        if (visible()) {
          ctx.globalAlpha = lock;
          const pulse = reduceMotion() ? 0 : ((f.tl * 0.8) % 1);
          drawMarker(pjX, pjY, pulse, C.elevated);
          label(pjX, pjY, 'Predicted closest approach', `${fmtDate(D.tcaMs)}, ${fmtClock(D.tcaMs)} UTC`, C.elevated);
          ctx.globalAlpha = 1;
        }
      }
    } else if (f.ch === 1) {
      const fade = 1 - smooth(0.25, 0.6, f.u);
      if (fade > 0) {
        strokeOrbit(ip.el, f.off, C.accent, 0.75 * fade, 1.3);
        strokeOrbit(sp.el, f.off, C.second, 0.6 * fade, 1.3);
      }
      /* the predicted tracks through the crossing, then the SGP4 objects */
      const pathA = smooth(0.2, 0.5, f.u);
      if (pathA > 0) {
        const span = Math.max(30, -f.off * 1.6);
        drawPath(D.ephP, 0, f.off, Math.min(span, 890), C.accent, 0.4 * pathA, [3, 5]);
        drawPath(D.ephS, 1, f.off, Math.min(span, 890), C.second, 0.35 * pathA, [3, 5]);
      }
      proj(D.M[0], D.M[1], D.M[2]);
      const px = pjX, py = pjY;
      drawMarker(px, py, 0, C.elevated);
      const trail = Math.max(1.5, Math.min(60, -f.off * 0.6));
      drawTrail(D.ephP, 0, f.off, trail, C.accent, 2, 0.9);
      drawTrail(D.ephS, 1, f.off, trail, C.second, 2, 0.8);
      ephAtC(D.ephP, 0, f.off, pa);
      ephAtC(D.ephS, 1, f.off, pb);
      proj(pa[0], pa[1], pa[2]);
      const ax = pjX, ay = pjY;
      proj(pb[0], pb[1], pb[2]);
      const bx = pjX, by = pjY;
      glowDot(ax, ay, C.accent, 4, 5);
      glowDot(bx, by, C.second, 3.8, 5);
      const sep = Math.hypot(ax - bx, ay - by);
      if (sep > 28) {
        label(ax, ay, 'Iridium 33', null, C.text, ax < bx ? 'left' : 'right');
        label(bx, by, 'Cosmos 2251', null, C.text, ax < bx ? 'right' : 'left');
      }
      if (cam.fov < 900) drawScaleBar();
    } else if (f.ch === 2) {
      /* the parents' paths carry on through the point: each debris stream
         keeps its parent's momentum */
      const trailFade = 1 - smooth(0.3, 0.7, f.u);
      if (trailFade > 0) {
        const span = Math.max(40, f.off + 40);
        drawTrail(D.ephP, 0, Math.min(f.off, 890), span, C.accent, 1.6, 0.55 * trailFade);
        drawTrail(D.ephS, 1, Math.min(f.off, 890), span, C.second, 1.6, 0.45 * trailFade);
      }
      const flash = reduceMotion() ? 0 : Math.exp(-f.tl * 3.2);
      drawFragments(f.off, 0.95, f.off > 0 ? 0.22 : 0, 1.5);
      proj(D.M[0], D.M[1], D.M[2]);
      if (flash > 0.01) {
        /* 11.6 km/s: the impact vaporises metal, so the flash runs white-hot
           to orange, with one shock ring racing out ahead of the fragments */
        const r = 30 + 220 * (1 - flash);
        const g = ctx.createRadialGradient(pjX, pjY, 0, pjX, pjY, r);
        g.addColorStop(0, `rgba(255,255,255,${0.97 * flash})`);
        g.addColorStop(0.12, `rgba(255,236,200,${0.8 * flash})`);
        g.addColorStop(0.4, `rgba(255,150,70,${0.32 * flash})`);
        g.addColorStop(1, 'rgba(255,120,40,0)');
        ctx.save();
        ctx.globalCompositeOperation = 'lighter';
        ctx.fillStyle = g;
        ctx.fillRect(pjX - r, pjY - r, 2 * r, 2 * r);
        const ring = 1 - Math.exp(-f.tl * 1.6);
        ctx.strokeStyle = `rgba(255,214,170,${0.6 * Math.exp(-f.tl * 1.4)})`;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(pjX, pjY, 10 + 0.42 * Math.min(W, H) * ring, 0, TWO_PI);
        ctx.stroke();
        ctx.fillStyle = `rgba(255,240,225,${0.06 * flash})`;
        ctx.fillRect(0, 0, W, H);
        ctx.restore();
      }
      if (reduceMotion() && f.tl < 2.5) drawMarker(pjX, pjY, 0, C.text);
      if (cam.fov < 900) drawScaleBar();
    } else {
      const t = f.off;
      const orbitA = f.ch === 3 ? smooth(0.18, 0.5, f.u) * 0.022 : 0.022;
      strokeOrbit(ip.el, 0, C.accent, 0.35 * (f.ch === 3 ? 1 - smooth(0.3, 0.6, f.u) : 0), 1, 1, [4, 6]);
      strokeOrbit(sp.el, 0, C.second, 0.3 * (f.ch === 3 ? 1 - smooth(0.3, 0.6, f.u) : 0), 1, 1, [4, 6]);
      drawFragmentOrbits(t, orbitA);
      drawFragments(t, f.ch === 3 ? lerp(0.95, 0.75, smooth(0.3, 0.7, f.u)) : 0.75, 0, 1.8,
        f.ch === 3 ? 0.3 * (1 - smooth(0.35, 0.5, f.u)) : 0);
      drawShell(f.ch === 3 ? smooth(0.62, 0.8, f.u) : 1, f.ch === 3);
      if (f.ch === 4) {
        ctx.fillStyle = `rgba(0,0,0,${0.35 * smooth(0, 0.3, f.u)})`;
        ctx.fillRect(0, 0, W, H);
      }
    }
  }

  /* ------------------------------------------------------------------------
     Chrome: captions, readout, clock, transport
     ---------------------------------------------------------------------- */
  function captionFor(ch) {
    const scn = D.scn, det = scn.screening.detection, ev = scn.event;
    const loc = det && det.location;
    const src = esc(ev.source_short || ev.source);
    const model = esc(D.deb ? D.deb.model : 'illustrative model');
    const orbits = scn.provenance === 'archived'
      ? 'Orbits: archived Space-Track element sets.'
      : `Orbits reconstructed from published parameters (${src}); they are built to meet at the published point, so the near-zero miss distance is by construction — replay archived TLEs for the true forecast.`;
    if (ch === 0) {
      return {
        date: `${fmtDate(D.runAtMs, true)}, ${fmtClock(D.runAtMs)} UTC`,
        body: `Replaying Kessler's screening on the catalog as it stood that afternoon, the 72-hour look-ahead flags <strong>Iridium 33</strong> against <strong>Cosmos 2251</strong>: a crossing ${loc ? `${Math.round(loc.alt_km)} km` : ''} above ${esc(ev.location_label.replace(/^above /, ''))}${det ? `, ${det.hours_before_tca.toFixed(0)} hours out` : ''}.`,
        source: `${orbits} Screening and probability: Kessler's live pipeline.`,
      };
    }
    if (ch === 1) {
      return {
        date: `${fmtDate(D.tcaMs, true)}, ${fmtClock(D.tcaMs)} UTC`,
        body: `Iridium 33, an active communications satellite, and Cosmos 2251, a defunct Russian satellite, close on each other at ${esc(ev.relative_speed_label)}.`,
        source: `Source: ${src}. Positions: SGP4, 1-second grid.`,
      };
    }
    if (ch === 2) {
      return {
        date: `${fmtDate(D.tcaMs, true)}, ${esc(ev.time_label.split(', ').pop())}`,
        body: `Both satellites are destroyed in ${esc(ev.significance.charAt(0).toLowerCase() + ev.significance.slice(1))}`,
        source: `Source: ${src}. Fragments: ${model}.`,
      };
    }
    if (ch === 3) {
      const real = ev.debris_label.replace(/\.$/, '');
      return {
        date: 'Over the next two years',
        body: 'Each fragment keeps its parent\'s orbit, give or take its kick. Within a day the cloud stretches into rings; over months, Earth\'s equatorial bulge fans them into a shell.',
        source: `Fragments: ${model}. In reality: ${esc(real.charAt(0).toLowerCase() + real.slice(1))}, per ${src}.`,
      };
    }
    return { date: '', body: '', source: '' };
  }

  let shownCh = -1;
  let rd = {};                     // readout value nodes by key
  let swapTimer = 0;
  function setChapterChrome(ch, instant) {
    if (ch === shownCh) return;
    shownCh = ch;
    const apply = () => {
      const cap = captionFor(ch);
      el.capNum.textContent = `Chapter ${ch + 1} of ${CHAPTERS.length}`;
      el.capDate.textContent = cap.date;
      el.capTitle.textContent = CHAPTERS[ch].title;
      el.capBody.innerHTML = cap.body;
      el.capSource.innerHTML = cap.source;
      el.caption.hidden = ch === 4;
      buildReadout(ch);
      el.caption.classList.remove('is-swapping');
      el.readout.classList.remove('is-swapping');
      dirty = true;
    };
    clearTimeout(swapTimer);
    if (instant || reduceMotion()) { apply(); } else {
      el.caption.classList.add('is-swapping');
      el.readout.classList.add('is-swapping');
      swapTimer = setTimeout(apply, 220);
    }
    el.end.hidden = ch !== 4;
    $('tm-track-now').textContent = `${ch + 1} of ${CHAPTERS.length} · ${CHAPTERS[ch].title}`;
    [...el.chapters.querySelectorAll('li')].forEach((li, i) => {
      li.classList.toggle('is-current', i === ch);
      const b = li.querySelector('button');
      if (i === ch) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
    });
  }

  function kv(key, lab, sub, cls, keepSub) {
    return `<div class="tm-kv"><div class="tm-kv__label">${lab}</div>
      <div class="tm-kv__value ${cls || ''}" data-k="${key}">—</div>${sub ? `<div class="tm-kv__sub${keepSub ? ' tm-kv__sub--keep' : ''}" data-k="${key}-sub">${sub}</div>` : ''}</div>`;
  }

  function buildReadout(ch) {
    const scn = D.scn, det = scn.screening.detection;
    let html = '';
    if (ch === 0) {
      const riskCls = det.risk_level === 'critical' ? 'is-critical' : det.risk_level === 'elevated' ? 'is-elevated' : '';
      html = `<div class="tm-readout__hd"><div><div class="tm-readout__title">Screening run</div>
          <div class="tm-readout__meta">${fmtDate(D.runAtMs)}, ${fmtClock(D.runAtMs)} UTC</div></div>
          <span class="chip chip--${esc(det.risk_level)}" data-k="chip" hidden>${esc(det.risk_level)}</span></div>
        <div data-k="scan"><div class="tm-kv__label">Propagating the catalog ahead</div>
          <div class="tm-kv__value" data-k="ahead">+0 h</div>
          <div class="tm-progress"><span data-k="bar"></span></div></div>
        <div class="tm-readout__grid" data-k="found" hidden>
          ${kv('lead', 'Flagged ahead', 'before closest approach')}
          ${kv('miss', 'Miss distance', 'predicted')}
          ${kv('pc', 'Probability', 'of collision', riskCls)}
          ${kv('vrel', 'Relative speed', 'at closest approach')}
        </div>
        <div data-k="more" hidden>
          <div class="tm-readout__rule"></div>
          <div class="tm-kv__label">Probability as the encounter nears</div>
          <div class="tm-spark"><canvas data-k="spark" height="64"></canvas>
            <div class="tm-spark__legend"><span data-k="spark-l"></span><span data-k="spark-r"></span></div></div>
          <p class="tm-readout__note">Replayed screening runs, same code as live. Probability uses Kessler's assumed TLE-class uncertainty, which shrinks as the encounter nears.</p>
        </div>`;
    } else if (ch === 1) {
      const st = D.eph.tca_state;
      html = `<div class="tm-readout__hd"><div><div class="tm-readout__title">Closest approach</div>
          <div class="tm-readout__meta">${fmtDate(D.tcaMs)}, ${fmtClock(D.tcaMs, false)} UTC</div></div></div>
        <div class="tm-readout__grid tm-readout__grid--1">${kv('cd', 'Time to closest approach', null, 'tm-kv__value--xl')}</div>
        <div class="tm-readout__rule"></div>
        <div class="tm-readout__grid">
          ${kv('sep', 'Separation', 'SGP4, 1 s grid')}
          ${kv('close', 'Closing speed', 'relative')}
          ${kv('angle', 'Crossing angle', 'between velocities')}
          ${kv('alt', 'Altitude', 'at closest approach')}
        </div>`;
      queueMicrotask(() => {
        setVal('close', st.rel_vel_ms ? `${(st.rel_vel_ms / 1000).toFixed(2)}<small>km/s</small>` : '—');
        setVal('angle', st.approach_angle_deg ? `${Math.round(st.approach_angle_deg)}°` : '—');
        const enc = D.scn.encounter;
        setVal('alt', enc ? `${Math.round(enc.alt_km)}<small>km</small>` : '—');
      });
    } else if (ch === 2) {
      const en = D.deb.energy;
      html = `<div class="tm-readout__hd"><div><div class="tm-readout__title">Breakup</div>
          <div class="tm-readout__meta">Illustrative model</div></div></div>
        <div class="tm-readout__grid tm-readout__grid--1">${kv('nfrag', 'Fragments 10 cm and larger', 'NASA Standard Breakup Model, simplified', 'tm-kv__value--xl')}</div>
        <div class="tm-readout__rule"></div>
        <div class="tm-readout__grid">
          ${kv('since', 'Since impact', 'scenario time')}
          ${kv('energy', 'Impact energy', `catastrophic above ${en.catastrophic_threshold_j_per_g} J/g`)}
        </div>
        <p class="tm-readout__note">Masses are approximate public figures: ${D.scn.objects.map((o) => `${esc(o.name)} ${nf(o.mass_kg)} kg`).join(', ')}.</p>`;
      queueMicrotask(() => {
        setVal('energy', `${(en.specific_energy_j_per_g / 1000).toFixed(1)}<small>kJ/g</small>`);
      });
    } else if (ch === 3) {
      const live = D.deb.live_catalog;
      const band = D.deb.altitude_band_km;
      html = `<div class="tm-readout__hd"><div><div class="tm-readout__title">Debris cloud</div>
          <div class="tm-readout__meta">Illustrative model</div></div></div>
        <div class="tm-readout__grid tm-readout__grid--1">${kv('since', 'Since the collision', null, 'tm-kv__value--xl')}</div>
        <div class="tm-readout__rule"></div>
        <div class="tm-readout__grid">
          ${kv('alive', 'Still in orbit', `of ${nf(D.deb.counts.modeled_total_ge_10cm)} modeled`)}
          ${kv('real', 'Cataloged in reality', 'fragments, per NASA')}
        </div>
        <div class="tm-readout__rule"></div>
        <div class="tm-readout__grid tm-readout__grid--1">
          ${kv('band', 'Tracked objects crossing this shell today', `${nf(band[0])}–${nf(band[1])} km, from Kessler's live catalog. A band-overlap count, not a risk.`, '', true)}
        </div>`;
      queueMicrotask(() => {
        setVal('real', '2,000+');
        setVal('band', live ? nf(live.objects_in_band) : '—');
      });
    }
    el.readout.innerHTML = html;
    el.readout.hidden = !html;
    rd = {};
    el.readout.querySelectorAll('[data-k]').forEach((n) => { rd[n.dataset.k] = n; });
    rd._cache = {};
    if (ch === 0) {
      queueMicrotask(() => {
        if (!det) return;   /* never detected: the readout keeps its dashes */
        setVal('lead', `${det.hours_before_tca.toFixed(1)}<small>h</small>`);
        setVal('miss', fmtDistHtml(det.miss_distance_m));
        setVal('pc', fmtPcHtml(det.pc));
        setVal('vrel', `${(det.relative_velocity_ms / 1000).toFixed(2)}<small>km/s</small>`);
        drawSpark();
      });
    }
  }

  function setVal(k, html) {
    const n = rd[k];
    if (!n || rd._cache[k] === html) return;
    rd._cache[k] = html;
    n.innerHTML = html;
  }
  function setHidden(k, hidden) {
    const n = rd[k];
    if (n && n.hidden !== hidden) n.hidden = hidden;
  }

  function updateReadout(f) {
    if (f.ch === 0) {
      const scan = clamp01((f.u - 0.12) / 0.4);
      const e = easeInOut(scan);
      const aheadH = e * ((D.tcaMs - D.runAtMs) / 3600e3);
      const locked = f.u >= 0.53;
      setVal('ahead', `+${Math.floor(aheadH)}<small>h of ${Math.round(D.scn.screening.method.window_hours)} h</small>`);
      if (rd.bar) rd.bar.style.width = `${(locked ? 1 : e) * 100}%`;
      setHidden('scan', locked);
      setHidden('chip', !locked);
      setHidden('found', !locked);
      setHidden('more', f.u < 0.62);
    } else if (f.ch === 1) {
      const t = fmtHms(f.off, true);
      setVal('cd', Math.abs(f.off) < 10
        ? `${Math.abs(f.off).toFixed(2)}<small>s</small>` : t.text);
      ephAt(D.ephP, f.off, pa);
      ephAt(D.ephS, f.off, pb);
      const d = Math.hypot(pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]) * 1000;
      setVal('sep', fmtDistHtml(d));
      const sub = rd['sep-sub'];
      const near = f.off > -CONTACT_S;
      if (sub && sub.dataset.near !== String(near)) {
        sub.dataset.near = String(near);
        sub.textContent = near ? 'public-TLE gap · in reality: contact' : 'SGP4, 1 s grid';
      }
    } else if (f.ch === 2) {
      setVal('since', f.off < 60 ? `${f.off.toFixed(f.off < 10 ? 2 : 1)}<small>s</small>`
        : `${Math.floor(f.off / 60)}<small>min</small> ${Math.round(f.off % 60)}<small>s</small>`);
      setVal('nfrag', nf(D.deb.counts.modeled_total_ge_10cm));
    } else if (f.ch === 3) {
      const s = fmtSince(f.off);
      setVal('since', `${nf(s.v)}<small>${s.u}</small>`);
      let alive = D.deb.counts.in_orbit_modeled;
      for (const h of CHECKPOINT_HOURS) {
        if (f.off >= h * 3600 && D.checkpoints.has(h)) alive = D.checkpoints.get(h);
      }
      setVal('alive', nf(alive));
    }
  }

  function drawSpark() {
    const cv = rd.spark;
    if (!cv) return;
    const hist = (D.scn.screening.history || []).filter((h) => h.detected && h.pc > 0);
    if (hist.length < 2) { cv.parentElement.hidden = true; return; }
    const r = cv.getBoundingClientRect();
    const w = Math.max(200, Math.round(r.width || 264));
    const h = 64;
    const pr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = w * pr; cv.height = h * pr;
    const c = cv.getContext('2d');
    c.setTransform(pr, 0, 0, pr, 0, 0);
    c.clearRect(0, 0, w, h);
    const css = (n, fb) => (window.KTheme ? KTheme.css(n, fb) : fb);
    const xs = hist.map((p) => p.hours_before_tca);
    const ys = hist.map((p) => Math.log10(p.pc));
    const crit = Math.log10(D.scn.screening.method.risk_thresholds.critical);
    const x0 = Math.max(...xs), x1 = 0;
    const yMin = Math.min(...ys, crit) - 0.3, yMax = Math.max(...ys, crit) + 0.3;
    const X = (v) => 6 + (w - 12) * (x0 - v) / (x0 - x1);
    const Y = (v) => h - 8 - (h - 16) * (v - yMin) / (yMax - yMin);
    c.strokeStyle = css('--k-critical', '#ff453a');
    c.globalAlpha = 0.5;
    c.setLineDash([3, 4]);
    c.beginPath(); c.moveTo(0, Y(crit)); c.lineTo(w, Y(crit)); c.stroke();
    c.setLineDash([]);
    c.globalAlpha = 1;
    c.font = `500 11px ${C.font}`;
    c.fillStyle = css('--k-critical', '#ff453a');
    c.textAlign = 'left';
    c.fillText('Critical', 4, Y(crit) - 5);
    c.strokeStyle = css('--k-text-2', '#a1a1a6');
    c.lineWidth = 1.5;
    c.beginPath();
    hist.forEach((p, i) => { const x = X(xs[i]), y = Y(ys[i]); if (i) c.lineTo(x, y); else c.moveTo(x, y); });
    c.stroke();
    hist.forEach((p, i) => {
      c.fillStyle = p.risk_level === 'critical' ? css('--k-critical', '#ff453a')
        : p.risk_level === 'elevated' ? css('--k-elevated', '#ff9f0a') : css('--k-text-3', '#86868b');
      c.beginPath(); c.arc(X(xs[i]), Y(ys[i]), 3, 0, TWO_PI); c.fill();
    });
    const last = hist[hist.length - 1];
    if (rd['spark-l']) rd['spark-l'].textContent = `${Math.round(x0)} h before`;
    if (rd['spark-r']) rd['spark-r'].innerHTML = `${Math.round(last.hours_before_tca)} h before · ${fmtPcHtml(last.pc)}`;
  }

  /* scenario seconds per story second - shown so slow motion and
     fast-forward are never mistaken for real time */
  function rateAt(f) {
    const dur = CHAPTERS[f.ch].dur;
    if (f.ch === 0) return 150;
    if (f.ch === 1) return 1800 * Math.pow(1 - f.u, 2) / dur;
    if (f.ch === 2) return 1800 * Math.exp(6 * f.u) / (Math.exp(6) - 1) / dur;
    if (f.ch === 3) return f.off * Math.log(TWO_YEARS_S / 300) / dur;
    return 0;
  }
  function fmtRate(r) {
    if (!(r > 0)) return '';
    if (r < 0.67) return `${Math.round(1 / r)}× slower`;
    if (r < 1.5) return 'real time';
    if (r < 1e6) {
      /* two significant figures read calmly while the rate sweeps: 1,000x · 45,000x */
      const mag = Math.pow(10, Math.max(0, Math.floor(Math.log10(r)) - 1));
      return `${nf(Math.round(r / mag) * mag)}× speed`;
    }
    return `${nf(Math.round(r / 1e6))} million× speed`;
  }

  let lastClock = '';
  function updateClock(f) {
    let main, sub;
    if (f.ch === 0) {
      main = `${fmtDate(f.scnMs)} · ${fmtClock(f.scnMs)} UTC`;
      const h = (D.tcaMs - f.scnMs) / 3600e3;
      sub = `${Math.floor(h)} h ${pad(Math.floor((h % 1) * 60))} min before`;
    } else if (f.ch <= 2) {
      main = `${fmtDate(f.scnMs)} · ${fmtClock(f.scnMs, true)} UTC`;
      const t = fmtHms(f.off, true);
      sub = f.off < 0 ? `T−${t.text}` : `T+${t.text}`;
    } else {
      main = fmtDate(f.scnMs);
      const s = fmtSince(f.off);
      sub = `${nf(s.v)} ${s.u} after`;
    }
    const rate = fmtRate(rateAt(f));
    if (rate) sub += ` · ${rate}`;
    const key = main + sub;
    if (key !== lastClock) {
      lastClock = key;
      el.clockMain.textContent = main;
      el.clockSub.textContent = sub;
      el.scrub.setAttribute('aria-valuetext', `${CHAPTERS[f.ch].title}: ${main}, ${sub}`);
    }
  }

  function buildTransport() {
    el.segs.innerHTML = CHAPTERS.map((c) => `<div class="tm-seg" style="flex:${c.dur} 1 0"><span></span></div>`).join('');
    el.chapters.innerHTML = CHAPTERS.map((c, i) => `<li style="flex:${c.dur} 1 0"><button class="tm-chapter" type="button" data-ch="${i}">${esc(c.title)}</button></li>`).join('');
    el.chapters.addEventListener('click', (e) => {
      const b = e.target.closest('[data-ch]');
      if (b) jumpChapter(+b.dataset.ch);
    });
  }
  let lastSeg = '';
  function updateTransport(T) {
    const v = Math.round((T / TOTAL) * 1000);
    if (+el.scrub.value !== v && !scrubbing) el.scrub.value = String(v);
    const fills = CHAPTERS.map((c) => clamp01((T - c.start) / c.dur));
    const key = fills.map((x) => x.toFixed(3)).join();
    if (key !== lastSeg) {
      lastSeg = key;
      el.segs.querySelectorAll('.tm-seg > span').forEach((s, i) => { s.style.width = `${fills[i] * 100}%`; });
    }
  }

  /* ------------------------------------------------------------------------
     Playback
     ---------------------------------------------------------------------- */
  let T = 0, playing = false, started = false, dirty = true, scrubbing = false;
  let lastTs = 0, idleSpin = 0, stillTimer = 0;

  function setPlaying(p) {
    playing = p && T < TOTAL;
    el.play.setAttribute('aria-pressed', playing ? 'true' : 'false');
    el.play.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    if (playing && reduceMotion()) scheduleStill();
    dirty = true;
  }

  /* Reduced motion: playback becomes a slideshow of each chapter's key
     frame instead of continuous camera motion. */
  function scheduleStill() {
    clearTimeout(stillTimer);
    const ch = chapterAt(T);
    T = CHAPTERS[ch].start + CHAPTERS[ch].dur * KEY_U[ch];
    dirty = true;
    stillTimer = setTimeout(() => {
      if (!playing) return;
      if (ch + 1 < CHAPTERS.length) { T = CHAPTERS[ch + 1].start; scheduleStill(); } else setPlaying(false);
    }, 6000);
  }

  function jumpChapter(i) {
    const ch = Math.max(0, Math.min(CHAPTERS.length - 1, i));
    T = CHAPTERS[ch].start + (reduceMotion() ? CHAPTERS[ch].dur * KEY_U[ch] : 0.001);
    if (!started) begin(false);
    if (playing && reduceMotion()) scheduleStill();
    dirty = true;
  }

  function begin(play) {
    started = true;
    el.intro.classList.add('is-leaving');
    setTimeout(() => { el.intro.hidden = true; }, reduceMotion() ? 0 : 520);
    el.transport.hidden = false;
    el.prov.hidden = false;
    el.caption.hidden = false;
    setPlaying(play);
  }

  function tick(ts) {
    requestAnimationFrame(tick);
    const dt = lastTs ? Math.min(0.1, (ts - lastTs) / 1000) : 0;
    lastTs = ts;
    let moving = false;
    if (playing && !reduceMotion()) {
      T += dt;
      if (T >= TOTAL) { T = TOTAL; setPlaying(false); }
      moving = true;
      dirty = true;
    }
    if (!started && !reduceMotion()) {
      idleSpin += dt * 0.02;
      dirty = true;
      moving = true;
    }
    if (!dirty) return;
    dirty = false;
    frameAt(started ? T : 0);
    if (!started) cam.spin += idleSpin;
    const key = `${cam.fov.toFixed(3)}|${cam.tilt.toFixed(5)}|${cam.fx.toFixed(1)}|${W}x${H}`;
    const camMoving = key !== lastCamKey;
    lastCamKey = key;
    render(moving && camMoving);
    if (camMoving && !moving) dirty = true;           // settle at full res
    if (started) {
      setChapterChrome(F.ch);
      updateReadout(F);
      updateClock(F);
      updateTransport(T);
    }
  }

  /* ------------------------------------------------------------------------
     Input
     ---------------------------------------------------------------------- */
  function wireInput() {
    el.start.addEventListener('click', () => { T = 0; begin(true); });
    el.play.addEventListener('click', () => {
      if (T >= TOTAL - 0.01) T = 0;
      setPlaying(!playing);
    });
    el.replay.addEventListener('click', () => { T = 0; shownCh = -1; setPlaying(true); });
    let wasPlaying = false;
    el.scrub.addEventListener('pointerdown', () => { scrubbing = true; wasPlaying = playing; playing = false; });
    const endScrub = () => {
      if (!scrubbing) return;
      scrubbing = false;
      if (wasPlaying) setPlaying(true);
    };
    el.scrub.addEventListener('pointerup', endScrub);
    el.scrub.addEventListener('pointercancel', endScrub);
    el.scrub.addEventListener('input', () => {
      T = (+el.scrub.value / 1000) * TOTAL;
      if (!started) begin(false);
      dirty = true;
    });
    document.addEventListener('keydown', (e) => {
      if (!D.ready || e.metaKey || e.ctrlKey || e.altKey) return;
      const tg = e.target;
      const tag = tg && tg.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (tg && tg.isContentEditable)) return;
      if (e.key === ' ' || e.code === 'Space') {
        if (tag === 'BUTTON' || tag === 'A') return;
        e.preventDefault();
        if (!started) { T = 0; begin(true); return; }
        if (T >= TOTAL - 0.01) T = 0;
        setPlaying(!playing);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        jumpChapter(chapterAt(T) + 1);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        /* previous chapter (Home restarts from the top); at the very end the
           last chapter counts as current, so Left always moves */
        jumpChapter(chapterAt(Math.min(T, TOTAL - 0.01)) - 1);
      } else if (e.key === 'Home') {
        e.preventDefault();
        jumpChapter(0);
      }
    });
    if (window.KTheme) KTheme.onChange(() => { readPalette(); drawSpark(); dirty = true; });
    if (mqReduce && mqReduce.addEventListener) mqReduce.addEventListener('change', () => { dirty = true; });
    if (window.ResizeObserver) new ResizeObserver(resize).observe(el.stage);
    else window.addEventListener('resize', resize);
  }

  /* ------------------------------------------------------------------------
     Notes section (chrome, themed)
     ---------------------------------------------------------------------- */
  function fillNotes() {
    const scn = D.scn, det = scn.screening.detection, deb = D.deb;
    const archived = scn.provenance === 'archived';
    $('tm-note-prov-chip').textContent = archived ? 'Archived' : 'Reconstructed';
    $('tm-note-prov').textContent = scn.provenance_note;
    const hint = $('tm-note-hint');
    if (scn.replay_hint) {
      /* the API hint is operator-facing (a script path) - users get plain words */
      hint.textContent = 'Archived element sets can replace the reconstruction when they’re available.';
    } else hint.hidden = true;
    $('tm-note-objects').innerHTML = scn.objects.map((o) => {
      const e = o.elements || {};
      return `<tr><th>${esc(o.name)} <span class="faint">· ${esc(o.status)}</span></th><td>${e.inclination_deg != null ? `${e.inclination_deg.toFixed(1)}° inclination` : '—'} · <span class="code">NORAD ${o.norad_id}</span></td></tr>`;
    }).join('');
    const m = scn.screening.method;
    $('tm-note-screen-meta').textContent = `${m.window_hours} h window · ${m.threshold_km} km threshold`;
    $('tm-note-screen').textContent = `Each screening run is replayed on an hourly cadence with only the element sets published before it, through the same screening, refinement and probability code as the live pipeline. ` + (det
      ? `The first run to see the pair was ${fmtDate(Date.parse(det.run_at), true)}, ${fmtClock(Date.parse(det.run_at))} UTC, ${det.hours_before_tca.toFixed(1)} hours before closest approach.`
      : 'No run in the replay window flagged the pair — with these element sets, the screen would not have warned.');
    const rows = scn.screening.history.map((h) => `<tr>
        <td>${fmtDate(Date.parse(h.run_at))}, ${fmtClock(Date.parse(h.run_at))}</td>
        <td class="num">${h.hours_before_tca != null ? `${h.hours_before_tca.toFixed(1)} h` : '—'}</td>
        <td class="num">${h.detected ? FMT.dist(h.miss_distance_m) : '—'}</td>
        <td class="num">${h.detected ? FMT.pc(h.pc) : '—'}</td>
        <td>${h.detected ? `<span class="chip chip--${esc(h.risk_level)}">${esc(h.risk_level)}</span>` : ''}</td></tr>`).join('');
    $('tm-note-history').innerHTML = `<thead><tr><th>Run (UTC)</th><th class="num">Ahead</th><th class="num">Miss</th><th class="num">Pc</th><th>Risk</th></tr></thead><tbody>${rows}</tbody>`;
    $('tm-note-debris').textContent = `The fragment cloud is an ${deb.model}. It is drawn to show how a single collision spreads through an orbital shell, not to reproduce the cataloged debris.`;
    const live = deb.live_catalog;
    const items = [
      `<strong>${nf(deb.counts.modeled_total_ge_10cm)}</strong> modeled fragments 10 cm and larger (${Object.entries(deb.counts.by_parent).map(([n, c]) => `${nf(c)} from ${esc((scn.objects.find((o) => String(o.norad_id) === n) || {}).name || n)}`).join(', ')}), split by mass.`,
      `Impact energy <strong>${nf(Math.round(deb.energy.specific_energy_j_per_g))} J/g</strong>, far above the ${deb.energy.catastrophic_threshold_j_per_g} J/g catastrophic threshold.`,
      ...deb.model_notes.map(esc),
    ];
    if (live) {
      items.push(`Today <strong>${nf(live.objects_in_band)}</strong> of ${nf(live.catalog_objects)} tracked objects in Kessler's catalog (${nf(live.payloads_in_band)} payloads) cross the ${nf(live.band_km[0])}–${nf(live.band_km[1])} km shell. ${esc(live.label.split('. ').slice(1).join('. '))}`);
      items.push(`<strong>${nf(live.collision_fragments_cataloged)}</strong> objects in the catalog are named as fragments of this collision.`);
    }
    $('tm-note-debris-list').innerHTML = items.map((x) => `<li>${x}</li>`).join('');
    $('tm-note-sources').innerHTML = (scn.sources || []).map((s) => `<li>${s.url ? `<a href="${esc(s.url)}" rel="noopener">${esc(s.label)}</a>` : esc(s.label)}</li>`).join('');
    el.notes.hidden = false;
  }

  /* ------------------------------------------------------------------------
     Boot
     ---------------------------------------------------------------------- */
  function applyDeepLink() {
    const q = new URLSearchParams(location.search);
    let t = null;
    if (q.has('chapter')) {
      const ch = Math.max(1, Math.min(CHAPTERS.length, +q.get('chapter') || 1)) - 1;
      const u = q.has('u') ? clamp01(+q.get('u')) : KEY_U[ch];
      t = CHAPTERS[ch].start + CHAPTERS[ch].dur * u;
    } else if (q.has('t')) {
      t = Math.max(0, Math.min(TOTAL, +q.get('t') || 0));
    }
    if (t !== null) { T = t; begin(false); }
  }

  async function boot() {
    readPalette();
    resize();
    /* intro text metrics settle once the web fonts land */
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => { introAx = introClearAx(); dirty = true; });
    }
    buildTransport();
    wireInput();
    if (typeof FMT !== 'undefined') FMT.startUtcClock($('utc-clock'));
    requestAnimationFrame(tick);
    if (typeof KEarthTexture !== 'undefined') {
      KEarthTexture.load().then(() => { dirty = true; }).catch(() => { /* flat Earth fallback */ });
    }
    try {
      const [scn, eph, deb] = await Promise.all([
        getJSON(`/scenarios/${SID}`),
        getJSON(`/scenarios/${SID}/ephemeris?window_s=900`),
        getJSON(`/scenarios/${SID}/debris?hours=0&limit=1500`),
      ]);
      if (!scn.screening || !scn.screening.detected) throw new Error('the replayed screening did not detect the encounter');
      ingest(scn, eph, deb);
      presets();
      el.provText.textContent = scn.provenance_label || (scn.provenance === 'archived' ? 'Archived element sets' : 'Reconstructed');
      el.prov.title = scn.replay_hint || scn.provenance_label || '';
      el.start.disabled = false;
      el.startLabel.textContent = 'Play the replay';
      fillNotes();
      applyDeepLink();
      dirty = true;
      /* backend in-orbit counts at the cascade's checkpoints (tiny payloads) */
      CHECKPOINT_HOURS.forEach((h) => {
        getJSON(`/scenarios/${SID}/debris?hours=${h}&limit=1`)
          .then((b) => { D.checkpoints.set(h, b.counts.in_orbit_modeled); dirty = true; })
          .catch(() => { /* the readout keeps the last known count */ });
      });
    } catch (err) {
      el.startLabel.textContent = 'Replay unavailable';
      el.introError.hidden = false;
      el.introError.textContent = `The Time Machine could not load its scenario (${err.message}).`;
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
