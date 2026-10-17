/* Kessler - space weather on the Overview (docs/v2/spaceweather.md).

   Space weather is not a separate page: it is one more input to the same
   stage, the same risk table and the same sheet.
   - The globe: live aurora (NOAA OVATION) on the Earth, an atmosphere rim
     that warms with the storm, and a 'Storm' color mode binning every object
     by how much denser the storm makes the air at its perigee.
   - The sheet's "Space weather" section: the storm level, what it does to
     low-orbit air, the flare, the wind, Kp (two days + NOAA's forecast), the
     poles right now, and the close approaches whose predictions it widens.
   - The "Space weather" lens (top of the stage): the stage seen from the Sun
     (js/sunview.js), drawn from NOAA's live readings only - a tool, not a
     simulator: no scenarios, no staged events. ?view=spaceweather opens it;
     the header pill on every page (js/theme.js) links here.
   Every surface degrades quietly: no data, no section; stale data, flagged. */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const stage = $('stage');
  const panel = $('sw-panel');
  if (!stage || !panel) return;

  const TWO_PI = Math.PI * 2;
  const DEG = Math.PI / 180;
  const J2000_MS = Date.UTC(2000, 0, 1, 12, 0, 0);
  const POLL_MS = 5 * 60 * 1000;
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const esc = (v) => (typeof FMT !== 'undefined' ? FMT.esc(v) : String(v));
  const fix = (v, d = 1) => (Number.isFinite(v) ? Number(v).toFixed(d) : '—');

  const els = {
    canvas: $('sun-canvas'), cap: $('sun-cap'),
    capEyebrow: $('sun-cap-eyebrow'), capTitle: $('sun-cap-title'),
    capText: $('sun-cap-text'), capFacts: $('sun-cap-facts'),
    lensGroup: $('lens-group'),
    meta: $('sw-meta'), level: $('sw-level'), kp: $('sw-kp'), lede: $('sw-lede'),
    facts: $('sw-facts'), kpchart: $('sw-kpchart'), widened: $('sw-widened'),
    open: $('sw-open'), src: $('sw-src'), north: $('sw-north'), south: $('sw-south'),
    side: $('stage-side'), dock: $('stage-dock'),
    stormMode: document.querySelector('#globe-mode-group button[data-mode="storm"]'),
  };

  let data = null;            // last /spaceweather payload
  let impact = null;          // last /spaceweather/impact
  let view = null;            // KSunView instance (lazy)
  let lens = 'orbits';
  let capTimer = 0;

  /* --- small helpers ----------------------------------------------------------- */
  function setPressed(btn, on) {
    if (!btn) return;
    btn.classList.toggle('is-active', on);
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  }
  function relTime(iso) {
    const t = Date.parse(iso);
    if (!Number.isFinite(t)) return '';
    const m = Math.max(0, Math.round((Date.now() - t) / 60000));
    return m < 1 ? 'just now' : m < 60 ? `${m} min ago` : `${Math.round(m / 60)} h ago`;
  }
  function interp(x, xs, ys) {
    if (!xs || !ys || !xs.length) return null;
    if (x <= xs[0]) return ys[0];
    for (let k = 1; k < xs.length; k++) {
      if (x <= xs[k]) return ys[k - 1] + (ys[k] - ys[k - 1]) * (x - xs[k - 1]) / (xs[k] - xs[k - 1]);
    }
    return ys[ys.length - 1];
  }
  function ratioAt(km) {
    const th = data && data.thermosphere;
    return th && th.ratio_now ? interp(km, th.alts_km, th.ratio_now) : null;
  }
  function levelClass(sum) {
    if (!sum) return '';
    if (sum.g >= 3) return 'is-severe';
    if (sum.g >= 1) return 'is-storm';
    if (sum.level === 'active' || sum.r >= 1) return 'is-active';
    return 'is-quiet';
  }
  function gmstRad(ms) {
    const d = (ms - J2000_MS) / 86400000;
    return ((280.46061837 + 360.98564736629 * d) % 360) * DEG;
  }
  function sunEci(ms) {
    const d = (ms - J2000_MS) / 86400000;
    const g = (357.529 + 0.98560028 * d) * DEG;
    const L = (280.459 + 0.98564736 * d) * DEG + (1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * DEG;
    const eps = (23.439 - 0.00000036 * d) * DEG;
    return [Math.cos(L), Math.cos(eps) * Math.sin(L), Math.sin(eps) * Math.sin(L)];
  }

  /* live conditions for the Sun -> Earth view */
  function liveParams() {
    const w = (data && data.solar_wind) || {};
    const kp = data && data.kp ? data.kp.now : null;
    return {
      speed: Number.isFinite(w.speed_kms) ? w.speed_kms : 420,
      density: Number.isFinite(w.density_cc) ? w.density_cc : 5,
      bz: Number.isFinite(w.bz_nt) ? w.bz_nt : 0,
      kp: Number.isFinite(kp) ? kp : 1.5,
      ratio: ratioAt(550) || 1,
    };
  }

  /* --- the sentence a person reads first -------------------------------------- */
  function liveStory(sum) {
    const w = (data && data.solar_wind) || {};
    const r550 = ratioAt(550);
    const shield = Number.isFinite(w.standoff_re) ? `${fix(w.standoff_re)} Earth radii` : null;
    const flare = sum.flare_6h_peak && /^[MX]/.test(sum.flare_6h_peak)
      ? ` An ${sum.flare_6h_peak.startsWith('X') ? 'X' : 'M'}-class flare (${sum.flare_6h_peak}) erupted in the last six hours.` : '';
    if (sum.g >= 1) {
      return `A ${sum.label.replace(' storm', '')} geomagnetic storm. The solar wind at ${Math.round(w.speed_kms || 0)} km/s` +
        (shield ? ` has pushed Earth’s magnetic shield in to ${shield} (about 10 on a calm day)` : ' is pressing on Earth’s magnetic shield') +
        `, energy is pouring into both poles, and the air at 550 km is ${fix(r550)}× denser than on a quiet day.${flare}`;
    }
    if (sum.level === 'active') {
      return `Unsettled. The solar wind is ${Math.round(w.speed_kms || 0)} km/s and the aurora is active; low-orbit air is ${fix(r550)}× a quiet day’s.${flare}`;
    }
    return `Quiet. The solar wind is a calm ${Math.round(w.speed_kms || 0)} km/s` +
      (shield ? `, the shield sits ${shield} out` : '') + `, and low orbits feel their usual drag.${flare}`;
  }

  /* ==========================================================================
     The sheet section
     ======================================================================== */
  function renderPanel() {
    const sum = data && data.summary;
    if (!sum) {
      if (data && data.error) {
        panel.hidden = false;
        els.level.className = 'sw-level';
        els.level.textContent = 'Unavailable';
        els.kp.textContent = '';
        els.lede.textContent = 'NOAA’s space-weather feeds are not answering right now. Kessler keeps screening as usual; storm effects return when the feeds do.';
        els.facts.innerHTML = ''; els.kpchart.innerHTML = ''; els.widened.hidden = true;
        els.src.textContent = '';
      }
      return;
    }
    panel.hidden = false;
    const cls = levelClass(sum);
    els.meta.textContent = sum.g >= 1 ? `G${sum.g}` : sum.level === 'quiet' ? 'Quiet' : 'Active';
    els.meta.className = `sheet-sec__meta sw-meta ${cls}`;
    els.level.className = `sw-level ${cls}`;
    els.level.textContent = sum.label;
    els.kp.textContent = Number.isFinite(sum.kp) ? `Kp ${fix(sum.kp)}` : '';
    els.lede.textContent = liveStory(sum);

    const w = data.solar_wind || {};
    const r550 = ratioAt(550);
    const counts = (typeof KGlobe !== 'undefined' && KGlobe.stormCounts) ? KGlobe.stormCounts() : null;
    const heated = counts ? counts[2] + counts[3] + counts[4] : null;
    const facts = [
      ['Air at 550 km', r550 ? `×${fix(r550)}` : '—', 'vs a quiet day', r550 > 1.25],
      ['Storm-heated air', heated ? heated.toLocaleString('en-US') : '—',
        'tracked objects fly through it', heated > 0],
      ['Solar flare', sum.flare || '—', sum.flare_6h_peak ? `6 h peak ${sum.flare_6h_peak}` : 'GOES X-rays',
        /^[MX]/.test(sum.flare_6h_peak || '')],
      ['Solar wind', Number.isFinite(w.speed_kms) ? `${Math.round(w.speed_kms)} km/s` : '—',
        Number.isFinite(w.standoff_re) ? `shield at ${fix(w.standoff_re)} Earth radii` : 'DSCOVR · L1',
        w.speed_kms > 600],
    ];
    els.facts.innerHTML = facts.map(([k, v, sub, hot]) =>
      `<div class="sw-fact${hot ? ' is-hot' : ''}"><dt>${esc(k)}</dt><dd>${esc(v)}</dd><span>${esc(sub)}</span></div>`).join('');
    els.kpchart.innerHTML = kpChart();
    renderWidened();
    const stale = sum.stale ? ' · <span class="sw-stale">delayed — NOAA feeds slow</span>' : '';
    els.src.innerHTML = `NOAA SWPC · NASA/NRL MSIS 2.1 · ${esc(relTime(sum.updated))}${stale}`;
    drawPoles();
  }

  function renderWidened() {
    const im = impact;
    if (!im || !im.available || !im.widened) { els.widened.hidden = true; return; }
    els.widened.hidden = false;
    const n = im.flagged;
    const rows = (im.items || []).slice(0, 3).map((it) => {
      const sw = it.space_weather || {};
      const pair = `${it.primary ? it.primary.name : '—'} × ${it.secondary ? it.secondary.name : '—'}`;
      const why = sw.flag === 'watch' ? `${it.risk_level} · ±${FMT.dist(sw.drift_m)}` : `could reach elevated · ±${FMT.dist(sw.drift_m)}`;
      return `<button type="button" class="sw-wrow" data-id="${esc(String(it.id))}" title="${esc(pair)}">` +
        `<span class="sw-wrow__pair">${esc(pair)}</span>` +
        `<span class="sw-wrow__drift">${esc(why)}</span></button>`;
    }).join('');
    const share = im.active ? Math.round((100 * im.widened) / im.active) : 0;
    els.widened.innerHTML =
      `<div class="sw-wlabel"><span class="storm-tag">Storm</span>` +
      (n ? `${esc(n.toLocaleString('en-US'))} close approach${n === 1 ? '' : 'es'} to recheck` : 'No close approach needs a recheck') +
      `</div><p class="sw-wnote">The extra drag widens ${share}% of predictions (${esc(im.widened.toLocaleString('en-US'))} of ` +
      `${esc(im.active.toLocaleString('en-US'))}). Flagged: already risky, or close enough that the storm could make them elevated. Risk levels are unchanged.</p>` +
      rows;
  }

  /* Kp: the last two days (observed) and NOAA's forecast, G1/G3/G5 marked */
  function kpChart() {
    const series = (data && data.kp_series) || [];
    if (!series.length) return '';
    const now = Date.now();
    const rows = series.filter((r) => { const t = Date.parse(r[0]); return t >= now - 48 * 3600e3 && t <= now + 72 * 3600e3; });
    if (!rows.length) return '';
    const W = 300, H = 92, padL = 20, padB = 16, padT = 8;
    const t0 = now - 48 * 3600e3, t1 = now + 72 * 3600e3;
    const x = (t) => padL + (W - padL - 2) * (t - t0) / (t1 - t0);
    const y = (kp) => padT + (H - padT - padB) * (1 - kp / 9);
    const bw = Math.max(1.5, (W - padL) * (3 * 3600e3) / (t1 - t0) - 1.2);
    let out = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">`;
    for (const [kp, lab] of [[5, 'G1'], [7, 'G3'], [9, 'G5']]) {
      out += `<line class="kp-grid" x1="${padL}" x2="${W}" y1="${y(kp)}" y2="${y(kp)}"/>` +
        `<text class="kp-lab" x="0" y="${y(kp) + 3.5}">${lab}</text>`;
    }
    for (const [tag, kp, kind] of rows) {
      const t = Date.parse(tag);
      const cls = kp >= 7 ? 'kp-bar is-severe' : kp >= 5 ? 'kp-bar is-storm' : 'kp-bar';
      const fut = kind === 'predicted' ? ' is-fc' : '';
      out += `<rect class="${cls}${fut}" x="${x(t).toFixed(1)}" y="${y(kp).toFixed(1)}" width="${bw.toFixed(1)}" height="${(y(0) - y(kp)).toFixed(1)}" rx="1"/>`;
    }
    out += `<line class="kp-now" x1="${x(now)}" x2="${x(now)}" y1="${padT - 4}" y2="${H - padB}"/>` +
      `<text class="kp-tick" x="${padL}" y="${H - 3}">−2 d</text>` +
      `<text class="kp-tick is-now" x="${x(now)}" y="${H - 3}" text-anchor="middle">now</text>` +
      `<text class="kp-tick" x="${W}" y="${H - 3}" text-anchor="end">+3 d forecast</text></svg>`;
    return out;
  }

  /* the poles right now: the real Earth from above each pole, NOAA's live
     aurora on it (soft additive sprites, dimmer where it is daylight) */
  let sprite = null;
  const poleOff = document.createElement('canvas');
  function auroraSprite() {
    if (sprite) return sprite;
    sprite = document.createElement('canvas');
    sprite.width = sprite.height = 24;
    const g = sprite.getContext('2d');
    const gr = g.createRadialGradient(12, 12, 0, 12, 12, 12);
    gr.addColorStop(0, 'rgba(70, 255, 140, 0.9)');
    gr.addColorStop(0.45, 'rgba(50, 235, 125, 0.38)');
    gr.addColorStop(1, 'rgba(30, 210, 110, 0)');
    g.fillStyle = gr; g.fillRect(0, 0, 24, 24);
    return sprite;
  }
  function drawPoles() {
    const pts = data && data.aurora && data.aurora.points;
    if (!pts || !els.north || panel.classList.contains('is-collapsed')) return;
    const now = Date.now();
    const gm = gmstRad(now);
    const sun = sunEci(now);
    const sx = Math.cos(gm) * sun[0] + Math.sin(gm) * sun[1];
    const sy = -Math.sin(gm) * sun[0] + Math.cos(gm) * sun[1];
    const sz = sun[2];
    const spr = auroraSprite();
    for (const [c, hemi] of [[els.north, 1], [els.south, -1]]) {
      const size = Math.round(c.clientWidth || 130);
      if (!size) continue;
      const d = Math.min(window.devicePixelRatio || 1, 2);
      c.width = poleOff.width = Math.round(size * d);
      c.height = poleOff.height = Math.round(size * d);
      const ctx = c.getContext('2d');
      const R = size * 0.47, cx = size / 2, cy = size / 2;
      ctx.setTransform(d, 0, 0, d, 0, 0);
      ctx.clearRect(0, 0, size, size);
      const tilt = hemi * Math.PI / 2;
      let ok = false;
      if (typeof KEarthTexture !== 'undefined' && KEarthTexture.ready()) {
        const o = poleOff.getContext('2d');
        o.setTransform(d, 0, 0, d, 0, 0);
        o.clearRect(0, 0, size, size);
        ok = KEarthTexture.drawHemisphere(o, { cx, cy, radiusPx: R, spin: 0, tilt, gmstRad: gm, sunDirTEME: sun });
        if (ok) ctx.drawImage(poleOff, 0, 0, poleOff.width, poleOff.height, 0, 0, size, size);
      }
      if (!ok) { ctx.fillStyle = '#0b1424'; ctx.beginPath(); ctx.arc(cx, cy, R, 0, TWO_PI); ctx.fill(); }
      const cT = Math.cos(tilt), sT = Math.sin(tilt);
      const s = Math.max(5, R * 0.1);
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      for (const [lon, lat, p] of pts) {
        if ((hemi > 0) !== (lat > 0)) continue;
        const la = lat * DEG, lo = lon * DEG;
        const ex = Math.cos(la) * Math.cos(lo), ey = Math.cos(la) * Math.sin(lo), ez = Math.sin(la);
        const a = lo + gm;
        const x = Math.cos(la) * Math.cos(a), y = Math.cos(la) * Math.sin(a), z = ez;
        const y2 = y * cT - z * sT, z2 = y * sT + z * cT;
        if (-y2 <= 0) continue;
        const elev = ex * sx + ey * sy + ez * sz;
        const night = elev < -0.1 ? 1 : elev > 0.1 ? 0.35 : 0.35 + 0.65 * (0.1 - elev) / 0.2;
        ctx.globalAlpha = clamp(p / 60, 0, 1) * 0.16 * night;
        ctx.drawImage(spr, cx + R * x - s / 2, cy - R * z2 - s / 2, s, s);
      }
      ctx.restore();
      ctx.strokeStyle = 'rgba(140, 180, 255, 0.28)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, TWO_PI); ctx.stroke();
    }
  }

  /* ==========================================================================
     The globe
     ======================================================================== */
  function feedGlobe() {
    if (typeof KGlobe === 'undefined' || typeof KGlobe.setSpaceWeather !== 'function' || !data) return;
    const th = data.thermosphere || {};
    try {
      KGlobe.setSpaceWeather({
        aurora: data.aurora ? data.aurora.points : [],
        alts: th.alts_km, ratio: th.ratio_now,
      });
    } catch (_) { return; /* engine fault: the layer just stays off */ }
    if (els.stormMode) els.stormMode.hidden = !(th.ratio_now && th.ratio_now.length);
  }

  /* ==========================================================================
     The Sun -> Earth lens
     ======================================================================== */
  function ensureView() {
    if (view || typeof KSunView === 'undefined' || !els.canvas) return view;
    view = KSunView.mount(els.canvas);
    return view;
  }

  function syncInsets() {
    if (!view) return;
    const sr = stage.getBoundingClientRect();
    const box = (el) => {
      if (!el || el.hidden) return null;
      const cs = getComputedStyle(el);
      if (cs.position !== 'absolute' || cs.display === 'none') return null;
      return el.getBoundingClientRect();
    };
    const side = box(els.side), dock = box(els.dock);
    const cap = els.cap && !els.cap.hidden ? els.cap.getBoundingClientRect() : null;
    const capOn = cap && cap.width > 0 && getComputedStyle(els.cap.parentElement).position === 'absolute';
    view.setInsets({
      left: 0, top: 0,
      right: side ? Math.max(0, sr.right - side.left + 8) : 0,
      bottom: dock ? Math.max(0, sr.bottom - dock.top + 6) : 0,
      capRight: capOn ? cap.right - sr.left : 0,
      capBottom: capOn ? cap.bottom - sr.top : 0,
    });
  }

  /* the view and its caption always show NOAA's latest readings */
  let snapped = false;
  function applyLive() {
    if (!view) return;
    const live = data && data.summary;
    view.set(liveParams(), live && !snapped);       // the first real reading: no easing in
    if (live) snapped = true;
    renderCaption();
  }

  function renderCaption() {
    if (!els.cap) return;
    const sum = data && data.summary;
    const age = sum && sum.updated ? ` · ${relTime(sum.updated)}` : '';
    els.capEyebrow.textContent = `Space weather · live from NOAA${age}`;
    els.capTitle.textContent = sum ? sum.label : (data && data.error ? 'NOAA is not answering' : 'Waiting for NOAA…');
    els.capText.textContent = sum ? liveStory(sum)
      : data && data.error ? 'The live feeds are unreachable right now. Screening and risk carry on as usual.'
        : 'Live solar-wind data is on its way.';
    renderCapFacts();
    syncInsets();
  }

  /* the caption's numbers are NOAA's readings, never the animation's */
  function renderCapFacts() {
    const sum = data && data.summary;
    if (!els.capFacts) return;
    if (lens !== 'weather' || !sum) { els.capFacts.innerHTML = ''; return; }
    const w = data.solar_wind || {};
    const r = ratioAt(550);
    const facts = [
      ['Kp', fix(sum.kp)],
      ['Solar wind', Number.isFinite(w.speed_kms) ? `${Math.round(w.speed_kms)} km/s` : '—'],
      ['Shield', Number.isFinite(w.standoff_re) ? `${fix(w.standoff_re)} radii` : '—'],
      ['Air density', r ? `×${fix(r)}` : '—'],
    ];
    els.capFacts.innerHTML = facts.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('');
  }

  function setLens(next, opts) {
    const on = next === 'weather';
    if (on && !ensureView()) return;
    lens = on ? 'weather' : 'orbits';
    stage.classList.toggle('is-weather', on);
    els.canvas.hidden = !on;
    els.cap.hidden = !on;
    for (const b of els.lensGroup.querySelectorAll('button[data-lens]')) setPressed(b, b.dataset.lens === lens);
    els.open.textContent = on ? 'Back to the globe' : 'See it from the Sun';
    if (typeof KGlobe !== 'undefined' && typeof KGlobe.setPaused === 'function') {
      try { KGlobe.setPaused(on); } catch (_) { /* engine fault */ }
    }
    if (window.KesslerUI && typeof KesslerUI.relayout === 'function') KesslerUI.relayout();
    if (on) {
      syncInsets();
      applyLive();
      view.start();
      /* bring the sheet's space-weather section into view */
      if (els.side && getComputedStyle(els.side).position === 'absolute' && !panel.hidden) {
        els.side.scrollTo({ top: Math.max(0, panel.offsetTop - 6), behavior: 'smooth' });
      }
      clearInterval(capTimer);
      capTimer = setInterval(renderCaption, 60000);       // keeps "n min ago" honest
    } else {
      if (view) view.stop();
      clearInterval(capTimer);
    }
    if (!(opts && opts.silent)) {
      try {
        const u = new URL(location.href);
        if (on) u.searchParams.set('view', 'spaceweather'); else u.searchParams.delete('view');
        history.replaceState(history.state, '', u.pathname + (u.search ? u.search : '') + u.hash);
      } catch (_) { /* old browsers: the lens still works */ }
    }
  }

  /* ==========================================================================
     Data
     ======================================================================== */
  async function load() {
    try {
      const res = await fetch('/spaceweather', { headers: { Accept: 'application/json' } });
      if (!res.ok) return;                         // a pre-space-weather backend
      data = await res.json();
    } catch (_) {
      return;                                      // offline: keep what we have
    }
    feedGlobe();
    try {
      const r = await fetch('/spaceweather/impact?limit=3');
      impact = r.ok ? await r.json() : null;
    } catch (_) { impact = null; }
    renderPanel();
    if (lens === 'weather') applyLive();
  }

  /* --- wiring ------------------------------------------------------------------ */
  els.lensGroup.addEventListener('click', (ev) => {
    const b = ev.target.closest('button[data-lens]');
    if (b) setLens(b.dataset.lens);
  });
  els.open.addEventListener('click', () => setLens(lens === 'weather' ? 'orbits' : 'weather'));
  els.widened.addEventListener('click', (ev) => {
    const b = ev.target.closest('.sw-wrow');
    if (!b || !window.KesslerUI) return;
    if (lens === 'weather') setLens('orbits');
    KesslerUI.focusConjunction(Number(b.dataset.id));
  });
  /* the sheet section expands: the poles need a size to draw at */
  panel.addEventListener('click', (ev) => {
    if (ev.target.closest('.sheet-sec__toggle')) setTimeout(drawPoles, 0);
  });
  /* the header pill (js/theme.js) on this page switches the lens in place */
  document.addEventListener('click', (ev) => {
    const a = ev.target.closest('a.sw-pill');
    if (!a || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.button !== 0) return;
    ev.preventDefault();
    setLens('weather');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
  window.addEventListener('resize', () => { syncInsets(); drawPoles(); });
  if (typeof ResizeObserver === 'function') {
    const ro = new ResizeObserver(() => syncInsets());
    if (els.side) ro.observe(els.side);
    if (els.dock) ro.observe(els.dock);
  }
  if (typeof KEarthTexture !== 'undefined') KEarthTexture.onReady(drawPoles);
  /* keep the 'Storm-heated air' count current once the fleet has loaded */
  setTimeout(() => { if (data) renderPanel(); }, 6000);

  window.KSpaceWeather = {
    open() { setLens('weather'); },
    close() { setLens('orbits'); },
    simulate(seconds) { if (view) view.simulate(seconds); },
    data: () => data,
  };

  const params = new URLSearchParams(location.search);
  if (params.get('view') === 'spaceweather') setLens('weather', { silent: true });
  load();
  setInterval(load, POLL_MS);
})();
