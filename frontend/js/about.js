/* Kessler - About (/about.html).

   - Hero: a slowly rotating dot globe drawn on canvas (no libraries). Land
     dots come from a mask rasterized from data/coastlines.json (Natural
     Earth 110 m, public domain); orbits and satellites are illustrative
     (the page says nothing about them being real objects). Colors are theme
     tokens, re-read on KTheme.onChange.
   - The problem: live numbers from GET /status with honest empty / failed /
     stale / offline states - never a number the server did not produce.
   - VIGIL: held-out metrics from GET /model/card, with the own-split caveat;
     the heuristic fallback is said out loud.
   - Kessler AI: the tool names are a documented constant mirrored from
     backend/app/assistant/tools.py (TOOLS); refreshed from GET /assistant/tools.
   - Scroll reveals via IntersectionObserver; prefers-reduced-motion shows
     everything at once and draws a still globe. */
'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
  const nf = (n) => Number(n).toLocaleString('en-US');
  const esc = FMT.esc;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* backend/app/assistant/tools.py TOOLS, in order. UI-action tools drive
     the interface; the rest read data or run analyses. */
  const AI_TOOLS = [
    'get_status', 'list_conjunctions', 'get_conjunction', 'get_object', 'search_objects',
    'get_risk', 'explain_risk', 'run_pipeline_refresh', 'get_changes', 'get_anomalies',
    'get_passes', 'simulate_maneuver', 'optimize_maneuver', 'plan_avoidance', 'list_user_objects',
    'screen_user_object', 'navigate_user', 'select_on_globe', 'set_globe_filter',
    'set_globe_speed', 'set_table_filter', 'fill_field', 'show_trajectory',
    'show_encounter', 'open_simulator', 'create_report',
  ];
  const UI_TOOLS = new Set(['navigate_user', 'select_on_globe', 'set_globe_filter',
    'set_globe_speed', 'set_table_filter', 'fill_field', 'show_trajectory',
    'show_encounter', 'open_simulator']);

  FMT.startUtcClock($('utc-clock'));

  /* --- scroll reveals ------------------------------------------------------ */

  const revealCbs = new Map();
  function onReveal(el, fn) { revealCbs.set(el, fn); }
  const revealIo = 'IntersectionObserver' in window ? new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      e.target.classList.add('is-in');
      revealIo.unobserve(e.target);
      const fn = revealCbs.get(e.target);
      if (fn) { revealCbs.delete(e.target); fn(); }
    }
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.12 }) : null;
  function observeAll() {
    document.querySelectorAll('.ab-reveal').forEach((el) => {
      if (revealIo && !reduce) revealIo.observe(el);
      else el.classList.add('is-in');
    });
  }

  /* count a number up once it scrolls into view (real values only) */
  function countUp(el, value) {
    if (reduce || value < 20) { el.textContent = nf(value); return; }
    const t0 = performance.now();
    const dur = 1100;
    const step = (t) => {
      const p = Math.min(1, (t - t0) / dur);
      const e = 1 - Math.pow(1 - p, 3);
      el.textContent = nf(Math.round(value * e));
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* --- live numbers (/status) ---------------------------------------------------- */

  function ago(iso) {
    const m = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 60000));
    if (m < 1) return 'just now';
    if (m < 60) return `${m} min ago`;
    const h = Math.round(m / 60);
    if (h < 48) return `${h} h ago`;
    return `${Math.round(h / 24)} days ago`;
  }

  async function loadLive() {
    const live = $('live');
    const meta = $('live-meta');
    const note = $('live-note');
    const nums = { catalog: $('n-catalog'), active: $('n-active'), attn: $('n-attn') };
    const setNote = (html) => { note.innerHTML = html; note.hidden = !html; };
    let st;
    try {
      st = await API.status();
    } catch (_) {
      meta.textContent = 'This server is not answering';
      live.classList.add('is-empty');
      setNote('The Kessler server is not answering, so there are no live numbers to show. Nothing here is filled in by hand.');
      return;
    }
    const win = st.window_hours || 72;
    $('n-active-label').textContent = `close approaches in the next ${win} hours`;
    const hasData = st.catalog_size > 0 && st.last_ingest_at;
    const notes = [];
    if (!hasData) {
      meta.textContent = 'No catalog ingested yet';
      live.classList.add('is-empty');
      notes.push('This server has not ingested a catalog yet, so there is nothing live to show. The numbers appear after the first refresh.');
      if (st.last_run_error) notes.push(`The last refresh failed: ${esc(st.last_run_error)}`);
      setNote(notes.join(' '));
      return;
    }
    const ageH = (Date.now() - Date.parse(st.last_ingest_at)) / 3600000;
    if (st.last_run_error) {
      notes.push(`The most recent catalog refresh failed, so these numbers may be out of date (${esc(st.last_run_error)}).`);
    }
    meta.textContent = `From this server · catalog updated ${ago(st.last_ingest_at)}` +
      (ageH > 6 && !st.last_run_error ? ' · not recently refreshed' : '');
    if (st.has_demo_data) notes.push('This server also holds labeled demo message sequences used for trend charts; the counts above come from screening, not from those sequences.');
    setNote(notes.join(' '));
    const attn = (st.elevated_count || 0) + (st.critical_count || 0);
    const fill = () => {
      countUp(nums.catalog, st.catalog_size);
      countUp(nums.active, st.active_conjunctions || 0);
      countUp(nums.attn, attn);
    };
    if (live.classList.contains('is-in') || reduce || !revealIo) fill();
    else onReveal(live, fill);
  }

  /* --- VIGIL (/model/card) ---------------------------------------------------------- */

  /* VIGIL: the share of the carry-forward baseline's error the model
     removes, as columns rising from that baseline (the win points up; lower-
     is-better error bars were misread). Data comes live from /model/card. */
  async function loadVigil() {
    const fig = $('vigil');
    const body = $('vigil-body');
    let mc;
    try {
      mc = await (await fetch('/model/card', { headers: { Accept: 'application/json' } })).json();
      if (!mc || typeof mc !== 'object') throw new Error('bad card');
    } catch (_) {
      body.innerHTML = '<p class="vg-empty">The server did not return the model card, so no metrics are shown here.</p>';
      return;
    }
    const m = mc.metrics;
    const pos = (v) => typeof v === 'number' && v > 0;
    if (!m || mc.heuristic || !pos(m.mse_test_all) || !pos(m.baseline_mse_all)) {
      body.innerHTML = mc.heuristic
        ? `<p class="vg-empty">This server is running the heuristic fallback (${esc(mc.version || 'heuristic')}) because no trained model is installed, so there are no held-out metrics to show.</p>`
        : '<p class="vg-empty">The model card has no held-out metrics, so none are shown.</p>';
      return;
    }
    const cut = (mine, base) => (1 - mine / base) * 100;
    const cols = [{ name: 'All held-out events', n: m.n_test, gain: cut(m.mse_test_all, m.baseline_mse_all) }];
    if (pos(m.mse_test_high_risk) && pos(m.baseline_mse_high_risk)) {
      cols.push({ name: 'High-risk events', n: m.n_test_high_risk,
                  gain: cut(m.mse_test_high_risk, m.baseline_mse_high_risk) });
    }
    const top = Math.max(...cols.map((c) => c.gain), 1);
    const pct = (g) => (g <= 0.5 ? '±0%' : `${g < 10 ? g.toFixed(1) : Math.round(g)}%`);
    body.innerHTML =
      `<div class="vg-plot" role="img" aria-label="${esc(cols.map((c) => `${c.name}: ${pct(c.gain)} less error than carry-forward`).join('; '))}">` +
      cols.map((c, i) => `<div class="vg-col${c.gain <= 0.5 ? ' is-flat' : ''}"><span class="vg-val">${pct(c.gain)}</span>` +
        `<span class="vg-bar" style="--d:${150 + i * 180}ms" data-h="${Math.max(0, c.gain / top * 78).toFixed(1)}"></span></div>`).join('') +
      '</div>' +
      '<div class="vg-zero"><span>Carry-forward baseline</span><span>less error ↑</span></div>' +
      '<div class="vg-labels">' + cols.map((c) =>
        `<span>${esc(c.name)}${c.n ? `<small>${nf(c.n)} events</small>` : ''}</span>`).join('') + '</div>' +
      `<p class="vg-cap"><strong>${esc(mc.version || 'VIGIL')}</strong>` +
      `${mc.trained_at ? ` · trained ${esc(String(mc.trained_at).slice(0, 10))}` : ''}` +
      ' · prediction error removed vs. carry-forward</p>' +
      '<p class="vg-note">Mean squared error of the predicted final risk (log<sub>10</sub>), measured on our own held-out split of the ESA Kelvins training data' +
      `${m.n_test && m.n_events ? ` (${nf(m.n_test)} of ${nf(m.n_events)} events)` : ''}, not the competition’s hidden test set.</p>`;
    const grow = () => body.querySelectorAll('.vg-bar').forEach((b) => { b.style.height = `${b.dataset.h}%`; });
    if (fig.classList.contains('is-in') || reduce || !revealIo) requestAnimationFrame(grow);
    else onReveal(fig, grow);
  }

  /* Time Machine teaser: two satellites ride their exact orbits (the same
     ellipses the SVG draws) toward the crossing, touching as the scrubber
     reaches the closest-approach tick; then the replay rewinds. */
  function tmScene() {
    const a = $('tm-a');
    const b = $('tm-b');
    const knob = $('tm-knob');
    const burst = $('tm-burst');
    const played = $('tm-played');
    const card = document.querySelector('.ab-tm');
    if (!a || !b || !knob || !played || !card || reduce) return;  // markup holds the meeting pose
    const C = [180, 86];
    const O1 = { rx: 165, ry: 36, rot: -14 * Math.PI / 180, t: 0.9255 };  // crossing on orbit 1
    const O2 = { rx: 150, ry: 50, rot: 24 * Math.PI / 180, t: 5.4061 };   // ... and on orbit 2
    const ARC = 2.5;    // radians of approach (~40% of an orbit)
    const KISS = 0.02;  // orbit-2 stops a hair short: the dots overlap (they collide), both still visible
    const at = (o, t) => {
      const x = o.rx * Math.cos(t);
      const y = o.ry * Math.sin(t);
      return [C[0] + x * Math.cos(o.rot) - y * Math.sin(o.rot), C[1] + x * Math.sin(o.rot) + y * Math.cos(o.rot)];
    };
    const place = (el, p) => { el.setAttribute('cx', p[0].toFixed(2)); el.setAttribute('cy', p[1].toFixed(2)); };
    const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
    const CYCLE = 12000;
    let raf = 0;
    let running = false;
    const frame = (now) => {
      const u = (now % CYCLE) / CYCLE;
      const tri = u < 0.5 ? u * 2 : 2 - u * 2;           // out, then rewind
      const p = ease(Math.min(1, tri * 1.12));           // a short hold at the meeting
      place(a, at(O1, O1.t - (1 - p) * ARC));
      place(b, at(O2, O2.t - KISS - (1 - p) * ARC));
      if (burst) {                                       // contact flash across the hold
        const h = (u - 0.4464) / 0.1072;                 // the hold spans u 0.4464..0.5536
        const on = h >= 0 && h <= 1;
        burst.setAttribute('r', (on ? 4 + 13 * h : 4).toFixed(2));
        burst.setAttribute('opacity', (on ? 0.75 * (1 - h) : 0).toFixed(3));
      }
      const kx = 70 + p * 200;                           // tick sits at x = 270
      knob.setAttribute('cx', kx.toFixed(2));
      played.setAttribute('d', `M40 176H${kx.toFixed(2)}`);
      raf = requestAnimationFrame(frame);
    };
    const start = () => { if (!running) { running = true; raf = requestAnimationFrame(frame); } };
    const stop = () => { running = false; cancelAnimationFrame(raf); };
    let inView = !('IntersectionObserver' in window);
    if (!inView) {
      new IntersectionObserver((es) => {
        inView = es[0].isIntersecting;
        (inView ? start : stop)();
      }).observe(card);
    } else start();
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) stop();
      else if (inView) start();
    });
  }

  /* --- Kessler AI tools ---------------------------------------------------------------- */

  function renderTools() {
    $('ai-tools').textContent = String(AI_TOOLS.length);
    /* the live catalog wins over the bundled copy when the server answers */
    fetch('/assistant/tools').then((r) => (r.ok ? r.json() : null)).then((cat) => {
      if (!cat || !Array.isArray(cat.items) || !cat.items.length) return;
      $('ai-tools').textContent = String(cat.count);
      $('tool-list').innerHTML = cat.items.map((t) =>
        `<li${t.family === 'ui' ? ' class="is-ui"' : ''} title="${esc(t.summary || '')}">${esc(t.name)}</li>`).join('') +
        '<li class="ab-tools__legend" aria-hidden="true">Highlighted: the tools that drive the interface.</li>';
    }).catch(() => { /* keep the bundled list */ });
    $('tool-list').innerHTML = AI_TOOLS.map((t) => `<li${UI_TOOLS.has(t) ? ' class="is-ui"' : ''}>${esc(t)}</li>`).join('') +
      '<li class="ab-tools__legend" aria-hidden="true">Highlighted: the tools that drive the interface.</li>';
  }

  /* --- Time Machine teaser animation starts when visible ------------------------------ */

  const tm = document.querySelector('.ab-tm');
  if (tm && !reduce && revealIo) onReveal(tm, () => {});

  /* --- the dot globe -------------------------------------------------------------------- */

  const globe = (() => {
    const cv = $('globe-canvas');
    const wrap = cv.parentElement;
    const ctx = cv.getContext('2d');
    let W = 0, H = 0, dpr = 1;
    let land = null;          // Float32Array [lat, lon, ...] (radians)
    let landAlpha = 0;        // fades in once the mask is ready
    let colors = null;
    let raf = 0;
    let visible = true;
    let t0 = performance.now();
    const TILT = 0.36;        // view tilt (rad)
    const SPIN = 0.00006;     // Earth rotation (rad/ms) - about one turn in 105 s

    /* illustrative orbits: inclination, node, radius (x R), phase speed */
    const ORBITS = [
      { inc: 0.9, raan: 0.2, r: 1.2, w: 0.00042, sats: [0, 2.1, 4.4] },
      { inc: 1.72, raan: 1.4, r: 1.28, w: 0.00036, sats: [0.8, 3.6] },
      { inc: 0.35, raan: 2.6, r: 1.42, w: 0.00028, sats: [1.5, 5.2] },
    ];
    const MINE = { orbit: 0, phase: 1.05 };
    let seed = 11;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
    const DEBRIS = Array.from({ length: 150 }, () => ({
      inc: rnd() * Math.PI, raan: rnd() * Math.PI * 2, r: 1.06 + rnd() * 0.3, ph: rnd() * Math.PI * 2, w: 0.00018 + rnd() * 0.0003,
    }));

    function readColors() {
      colors = {
        land: KTheme.isLight() ? KTheme.css('--k-text-2') : KTheme.css('--k-text'),
        ocean: KTheme.css('--k-text-4'),
        disk: KTheme.css('--k-surface-2'),
        bg: KTheme.css('--k-bg'),
        rim: KTheme.css('--k-border-strong'),
        orbit: KTheme.css('--k-text-4'),
        sat: KTheme.css('--k-text'),
        debris: KTheme.css('--k-text-3'),
        accent: KTheme.css('--k-accent'),
        light: KTheme.isLight(),
      };
    }

    function resize() {
      const r = wrap.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = Math.max(1, r.width);
      H = Math.max(1, r.height);
      cv.width = Math.round(W * dpr);
      cv.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (reduce) draw(performance.now());
    }

    /* --- land mask from coastline polylines ------------------------------ */
    function stitch(lines) {
      const closed = [];
      const open = [];
      for (const l of lines) {
        if (l.length < 2) continue;
        const a = l[0], b = l[l.length - 1];
        if (a[0] === b[0] && a[1] === b[1]) closed.push(l);
        else open.push(l.slice());
      }
      const wrapD = (p, q) => {
        let dl = Math.abs(p[0] - q[0]);
        if (dl > 180) dl = 360 - dl;
        return Math.hypot(dl, p[1] - q[1]);
      };
      while (open.length) {
        let cur = open.shift();
        for (let guard = 0; guard < 64; guard++) {
          const end = cur[cur.length - 1];
          if (wrapD(end, cur[0]) < 0.01) break;
          let best = -1, bestD = Infinity, rev = false;
          open.forEach((l, i) => {
            const ds = wrapD(end, l[0]);
            const de = wrapD(end, l[l.length - 1]);
            if (ds < bestD) { bestD = ds; best = i; rev = false; }
            if (de < bestD) { bestD = de; best = i; rev = true; }
          });
          const selfD = wrapD(end, cur[0]);
          if (best < 0 || bestD > 8 || selfD <= bestD) break;
          const next = open.splice(best, 1)[0];
          cur = cur.concat(rev ? next.slice().reverse() : next);
        }
        closed.push(cur);
      }
      return closed;
    }

    function buildMask(data) {
      const MW = 720, MH = 360;
      const oc = document.createElement('canvas');
      oc.width = MW; oc.height = MH;
      const g = oc.getContext('2d', { willReadFrequently: true });
      g.fillStyle = '#fff';
      for (const ring of stitch(data.lines || [])) {
        /* unwrap longitudes so a ring never jumps across the map */
        const pts = [[ring[0][0], ring[0][1]]];
        for (let i = 1; i < ring.length; i++) {
          let lon = ring[i][0];
          const prev = pts[i - 1][0];
          while (lon - prev > 180) lon -= 360;
          while (lon - prev < -180) lon += 360;
          pts.push([lon, ring[i][1]]);
        }
        const span = pts[pts.length - 1][0] - pts[0][0];
        if (Math.abs(span) > 300) {   // circumpolar (Antarctica): close via the pole
          const meanLat = pts.reduce((s, p) => s + p[1], 0) / pts.length;
          const pole = meanLat < 0 ? -90 : 90;
          pts.push([pts[pts.length - 1][0], pole], [pts[0][0], pole]);
        }
        for (const off of [-360, 0, 360]) {
          g.beginPath();
          pts.forEach((p, i) => {
            const x = (p[0] + off + 180) / 360 * MW;
            const y = (90 - p[1]) / 180 * MH;
            if (i) g.lineTo(x, y); else g.moveTo(x, y);
          });
          g.closePath();
          g.fill();
        }
      }
      const img = g.getImageData(0, 0, MW, MH).data;
      const isLand = (lat, lon) => {
        const x = Math.min(MW - 1, Math.max(0, Math.floor((lon + 180) / 360 * MW)));
        const y = Math.min(MH - 1, Math.max(0, Math.floor((90 - lat) / 180 * MH)));
        return img[(y * MW + x) * 4 + 3] > 127;
      };
      /* latitude rows of evenly spaced dots */
      const STEP = 1.55;
      const out = [];
      for (let lat = -88; lat <= 88; lat += STEP) {
        const n = Math.max(1, Math.round(360 / STEP * Math.cos(lat * Math.PI / 180)));
        for (let k = 0; k < n; k++) {
          const lon = -180 + (k + 0.5) * 360 / n;
          if (isLand(lat, lon)) out.push(lat * Math.PI / 180, lon * Math.PI / 180);
        }
      }
      return new Float32Array(out);
    }

    function fallbackDots() {
      /* no coastlines: a uniform sphere of faint dots, no fake continents */
      const out = [];
      const n = 2600;
      const ga = Math.PI * (3 - Math.sqrt(5));
      for (let i = 0; i < n; i++) {
        const y = 1 - (i / (n - 1)) * 2;
        out.push(Math.asin(y), ((i * ga) % (Math.PI * 2)) - Math.PI);
      }
      return new Float32Array(out);
    }

    /* --- projection -------------------------------------------------------- */
    const cT = Math.cos(TILT), sT = Math.sin(TILT);
    function project(x, y, z, cx, cy, R, out) {
      /* tilt the view about the screen x axis */
      const y2 = y * cT - z * sT;
      const z2 = y * sT + z * cT;
      out[0] = cx + R * x;
      out[1] = cy - R * y2;
      out[2] = z2;
      return out;
    }
    /* point on a circular orbit: p = r (cos u N + sin u (cos i T + sin i P)),
       with the ascending-node vector N, T 90 deg ahead on the equator and
       the pole P = world y. Orbits are inertial; only the Earth spins. */
    function orbitPoint(o, u) {
      const cu = Math.cos(u), su = Math.sin(u);
      const ci = Math.cos(o.inc), si = Math.sin(o.inc);
      const cO = Math.cos(o.raan), sO = Math.sin(o.raan);
      return [
        o.r * (cu * cO + su * ci * sO),
        o.r * (su * si),
        o.r * (-cu * sO + su * ci * cO),
      ];
    }

    const tmp = [0, 0, 0];
    function drawOrbitHalf(o, cx, cy, R, front) {
      ctx.beginPath();
      let pen = false;
      for (let k = 0; k <= 160; k++) {
        const a = k / 160 * Math.PI * 2;
        const p = orbitPoint(o, a);
        project(p[0], p[1], p[2], cx, cy, R, tmp);
        const isFront = tmp[2] >= 0;
        if (isFront === front) {
          if (pen) ctx.lineTo(tmp[0], tmp[1]); else { ctx.moveTo(tmp[0], tmp[1]); pen = true; }
        } else pen = false;
      }
      ctx.stroke();
    }

    function drawBodies(cx, cy, R, t, front) {
      /* debris */
      ctx.fillStyle = colors.debris;
      for (const d of DEBRIS) {
        const p = orbitPoint(d, d.ph + t * d.w);
        project(p[0], p[1], p[2], cx, cy, R, tmp);
        if ((tmp[2] >= 0) !== front) continue;
        if (!front && Math.hypot(tmp[0] - cx, tmp[1] - cy) < R) continue;
        ctx.globalAlpha = front ? 0.75 : 0.3;
        ctx.fillRect(tmp[0] - 0.9, tmp[1] - 0.9, 1.8, 1.8);
      }
      /* satellites on the drawn orbits */
      ORBITS.forEach((o, oi) => {
        for (const ph of o.sats) {
          const mine = oi === MINE.orbit && ph === o.sats[0];
          const a = (mine ? MINE.phase : ph) + t * o.w;
          const p = orbitPoint(o, a);
          project(p[0], p[1], p[2], cx, cy, R, tmp);
          if ((tmp[2] >= 0) !== front) continue;
          if (!front && Math.hypot(tmp[0] - cx, tmp[1] - cy) < R) continue;
          ctx.globalAlpha = front ? 1 : 0.35;
          if (mine) {
            const pulse = reduce ? 0.4 : (t % 2400) / 2400;
            ctx.strokeStyle = colors.accent;
            ctx.lineWidth = 1.5;
            ctx.globalAlpha = (front ? 0.6 : 0.2) * (1 - pulse);
            ctx.beginPath();
            ctx.arc(tmp[0], tmp[1], 5 + pulse * 16, 0, Math.PI * 2);
            ctx.stroke();
            ctx.globalAlpha = front ? 1 : 0.35;
            ctx.fillStyle = colors.accent;
            ctx.beginPath();
            ctx.arc(tmp[0], tmp[1], 4.6, 0, Math.PI * 2);
            ctx.fill();
          } else {
            ctx.fillStyle = colors.sat;
            ctx.beginPath();
            ctx.arc(tmp[0], tmp[1], 2.6, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      });
      ctx.globalAlpha = 1;
    }

    function draw(now) {
      if (!colors) readColors();
      const t = reduce ? 42000 : now - t0;
      ctx.clearRect(0, 0, W, H);
      const R = Math.min(W * (W < 700 ? 0.36 : 0.26), H * 0.6);
      const cx = W / 2;
      const cy = R * 1.34 + 8;

      /* back halves of orbits + far bodies */
      ctx.lineWidth = 1.1;
      ctx.strokeStyle = colors.orbit;
      ctx.globalAlpha = 0.5;
      ORBITS.forEach((o) => drawOrbitHalf(o, cx, cy, R, false));
      ctx.globalAlpha = 1;
      drawBodies(cx, cy, R, t, false);

      /* the disk */
      const grd = ctx.createRadialGradient(cx - R * 0.35, cy - R * 0.45, R * 0.1, cx, cy, R);
      grd.addColorStop(0, colors.disk);
      grd.addColorStop(1, colors.bg);
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = colors.rim;
      ctx.lineWidth = 1;
      ctx.stroke();

      /* land dots */
      if (land) {
        if (landAlpha < 1) landAlpha = reduce ? 1 : Math.min(1, landAlpha + 0.03);
        const rot = t * SPIN;
        const ds = Math.max(1.8, R / 135);
        ctx.fillStyle = colors.land;
        for (let i = 0; i < land.length; i += 2) {
          const lat = land[i], lon = land[i + 1] + rot;
          const cl = Math.cos(lat);
          const x = cl * Math.sin(lon);
          const y = Math.sin(lat);
          const z = cl * Math.cos(lon);
          const z2 = y * sT + z * cT;
          if (z2 <= 0.02) continue;
          const y2 = y * cT - z * sT;
          ctx.globalAlpha = landAlpha * (colors.light ? 0.2 + 0.8 * z2 : 0.12 + 0.68 * z2);
          const s = ds * (0.55 + 0.45 * z2);
          ctx.fillRect(cx + R * x - s / 2, cy - R * y2 - s / 2, s, s);
        }
        ctx.globalAlpha = 1;
      }

      /* front halves + near bodies */
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = colors.orbit;
      ctx.globalAlpha = 0.9;
      ORBITS.forEach((o) => drawOrbitHalf(o, cx, cy, R, true));
      ctx.globalAlpha = 1;
      drawBodies(cx, cy, R, t, true);

      if (!reduce && visible) raf = requestAnimationFrame(draw);
    }

    function start() {
      cancelAnimationFrame(raf);
      if (reduce) { draw(performance.now()); return; }
      raf = requestAnimationFrame(draw);
    }

    return {
      init() {
        readColors();
        resize();
        if ('ResizeObserver' in window) new ResizeObserver(resize).observe(wrap);
        else window.addEventListener('resize', resize);
        if ('IntersectionObserver' in window) {
          new IntersectionObserver((es) => {
            const v = es.some((e) => e.isIntersecting);
            if (v === visible) return;
            visible = v;
            if (visible) start(); else cancelAnimationFrame(raf);
          }).observe(wrap);
        }
        document.addEventListener('visibilitychange', () => {
          if (document.hidden) cancelAnimationFrame(raf);
          else if (visible) start();
        });
        KTheme.onChange(() => { readColors(); if (reduce) draw(performance.now()); });
        fetch('data/coastlines.json')
          .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
          .then((data) => { land = buildMask(data); })
          .catch(() => { land = fallbackDots(); })
          .finally(() => { if (reduce) draw(performance.now()); });
        start();
      },
    };
  })();

  /* --- boot ----------------------------------------------------------------------------- */

  renderTools();
  observeAll();
  globe.init();
  loadLive();
  loadVigil();
  tmScene();
})();
