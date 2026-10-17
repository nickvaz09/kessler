/* Kessler - ops wall controller (contract in docs/CONTRACTS.md "Ops wall").
   Big-screen display, always dark. KGlobe fills the left region (surface
   mode + 0.5°/s autospin when the engine additions exist, typeof-
   guarded); the right column carries a large UTC clock, three counters
   (active / elevated / critical from /status), "Now on the globe" - the
   top 6 events auto-cycled every 12 s (KGlobe.showEncounter watch:false;
   any globe interaction pauses the cycle for 60 s) - the highest-risk
   list with the current one highlighted, and the latest /changes rows.
   Polls /status + /conjunctions every 60 s and never blanks: fetch
   failures keep the last data and the footer says so. Click anywhere
   (gesture-gated; links and drags excluded) requests fullscreen. */
'use strict';

/* the wall is always dark (a big-screen display in a dim room). The
   saved appearance still applies everywhere else - keep the attribute
   pinned here without touching localStorage. */
(() => {
  const root = document.documentElement;
  const pin = () => { if (root.getAttribute('data-theme') !== 'dark') root.setAttribute('data-theme', 'dark'); };
  pin();
  new MutationObserver(pin).observe(root, { attributes: true, attributeFilter: ['data-theme'] });
})();

(() => {
  const POLL_MS = 60000;       // /status + /conjunctions cadence
  const CYCLE_MS = 12000;      // focus panel advance
  const HOLD_MS = 60000;       // cycle pause after user globe interaction
  const FOCUS_N = 6;           // top events in the cycle
  const CHANGES_N = 6;         // change rows shown
  const KNOWN_LEVELS = { critical: true, elevated: true, nominal: true };
  const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
    'August', 'September', 'October', 'November', 'December'];
  const LEVEL_LABELS = { critical: 'Critical', elevated: 'Elevated', nominal: 'Low' };
  const TOP_N = 6;             // rows in the highest-risk list

  const els = {
    globeWrap: document.getElementById('wall-globe-wrap'),
    globeCanvas: document.getElementById('globe'),
    globeTip: document.getElementById('globe-tip'),
    legend: document.getElementById('wall-legend'),
    clock: document.getElementById('wall-clock'),
    date: document.getElementById('wall-date'),
    active: document.getElementById('wall-active'),
    elevated: document.getElementById('wall-elevated'),
    critical: document.getElementById('wall-critical'),
    sw: document.getElementById('wall-sw'),
    focusMeta: document.getElementById('focus-meta'),
    focusBody: document.getElementById('focus-body'),
    changesPanel: document.getElementById('wall-changes-panel'),
    changesMeta: document.getElementById('wall-changes-meta'),
    changesBody: document.getElementById('wall-changes-body'),
    foot: document.getElementById('wall-foot'),
    topList: document.getElementById('wall-top-list'),
  };

  let globeReady = false;      // KGlobe present and init() returned
  let hasData = false;         // at least one successful poll rendered
  let lastOkIso = null;        // UTC of the last successful poll
  let conjItems = [];          // latest /conjunctions items (API risk order)
  let focusList = [];          // top FOCUS_N of conjItems
  let focusIdx = 0;
  let lastCycleAt = 0;         // wall-clock ms of the last focus advance
  let holdUntil = 0;           // cycle paused until this wall-clock ms
  let encounterApplied = false; // showEncounter accepted for the current focus
  let downX = 0, downY = 0;    // pointerdown origin - drag vs click gate

  function safeLevel(level) {
    return KNOWN_LEVELS[level] ? level : 'nominal';
  }

  /* --- globe ----------------------------------------------------------------- */

  function showGlobeFallback() {
    globeReady = false;
    const d = document.createElement('div');
    d.className = 'wall-empty wall-empty--globe';
    d.textContent = 'The globe can’t be drawn on this display.';
    els.globeWrap.replaceChildren(d);
  }

  function initGlobe() {
    /* globe.js + earthtex.js build in parallel against the KGlobe contract -
       a dead engine must never leave a blank wall (same guard as dashboard). */
    if (typeof KGlobe === 'undefined') { showGlobeFallback(); return; }
    try {
      KGlobe.init({ canvas: els.globeCanvas, tipEl: els.globeTip, onSelect: null });
    } catch (_) {
      showGlobeFallback();
      return;
    }
    globeReady = true;
    /* engine additions, typeof-guarded: textured surface + slow autospin.
       Color mode is left at the engine's current default (contract:
       "setColorMode current" - nothing to override). */
    try {
      if (typeof KGlobe.setSurface === 'function') KGlobe.setSurface(true);
      if (typeof KGlobe.setAutoSpin === 'function') KGlobe.setAutoSpin(0.5);
    } catch (_) { /* engine fault - wireframe wall still works */ }
    /* any globe interaction holds the focus cycle for 60 s */
    els.globeWrap.addEventListener('pointerdown', holdCycle);
    els.globeWrap.addEventListener('wheel', holdCycle, { passive: true });
  }

  /* --- counters ---------------------------------------------------------------- */

  function renderStats(st) {
    els.active.textContent = FMT.num(st.active_conjunctions);
    els.elevated.textContent = FMT.num(st.elevated_count);
    els.elevated.classList.toggle('is-elevated', Number(st.elevated_count) > 0);
    els.critical.textContent = FMT.num(st.critical_count);
    els.critical.classList.toggle('is-critical', Number(st.critical_count) > 0);
    renderSpaceWeather(st.space_weather);
  }

  /* space weather: the storm level and what it does, in one line (hidden
     without live NOAA data) */
  function renderSpaceWeather(sw) {
    if (!els.sw) return;
    if (!sw) { els.sw.hidden = true; return; }
    const g = Number(sw.g) || 0;
    const bits = [];
    const r = sw.density_ratio && sw.density_ratio['550'];
    if (Number.isFinite(r) && r >= 1.1) bits.push(`air at 550 km ×${r.toFixed(1)}`);
    if (Number.isFinite(sw.flagged_conjunctions) && sw.flagged_conjunctions > 0) {
      bits.push(`${FMT.num(sw.flagged_conjunctions)} approaches to recheck`);
    }
    els.sw.hidden = false;
    els.sw.className = `wall-sw ${g >= 3 ? 'is-severe' : g >= 1 ? 'is-storm' : 'is-quiet'}`;
    els.sw.innerHTML = `<span class="wall-sw__lvl">${FMT.esc(sw.label)}</span>` +
      (bits.length ? `<span class="wall-sw__sub">${FMT.esc(bits.join(' · '))}</span>` : '');
  }

  /* the live aurora on the wall globe (NOAA OVATION, /spaceweather) */
  async function feedAurora() {
    if (!globeReady || typeof KGlobe.setSpaceWeather !== 'function') return;
    try {
      const res = await fetch('/spaceweather');
      if (!res.ok) return;
      const d = await res.json();
      const th = d.thermosphere || {};
      KGlobe.setSpaceWeather({ aurora: d.aurora ? d.aurora.points : [], alts: th.alts_km, ratio: th.ratio_now });
    } catch (_) { /* no aurora this round */ }
  }

  /* --- focus cycle ---------------------------------------------------------------- */

  function holdCycle() {
    holdUntil = Date.now() + HOLD_MS;
    renderFocusMeta();
  }

  function renderFocusMeta() {
    if (!focusList.length) { els.focusMeta.textContent = ''; return; }
    const pos = `${focusIdx + 1} of ${focusList.length}`;
    els.focusMeta.textContent =
      Date.now() < holdUntil ? `${pos} · paused` : pos;
  }

  function pairOf(it) {
    return [it.primary ? it.primary.name : '—', it.secondary ? it.secondary.name : '—'];
  }

  function renderFocus() {
    renderFocusMeta();
    renderTopList();
    if (!focusList.length) {
      els.focusBody.innerHTML = hasData
        ? '<div class="wall-empty">No active close approaches right now.</div>'
        : '<div class="wall-empty">Waiting for conjunction data…</div>';
      return;
    }
    if (focusIdx >= focusList.length) focusIdx = 0;
    const it = focusList[focusIdx];
    const lvl = safeLevel(it.risk_level);
    const effPc = it.predicted_final_risk != null ? it.predicted_final_risk : it.pc_latest;
    const [a, b] = pairOf(it);
    els.focusBody.innerHTML =
      `<div class="focus k-rise">` +
        `<span class="chip chip--${lvl}">${LEVEL_LABELS[lvl]}</span>` +
        `<div class="focus-names" title="${FMT.esc(`${a} and ${b}`)}">${FMT.esc(a)}<span class="focus-and">and</span>${FMT.esc(b)}</div>` +
        `<div class="focus-k">Closest approach in</div>` +
        `<div class="focus-count" data-tca="${FMT.esc(it.tca)}">${FMT.esc(FMT.countdown(it.tca))}</div>` +
        `<div class="focus-facts">` +
          `<div><span>Miss distance</span><b>${FMT.esc(FMT.dist(it.miss_distance_m))}</b></div>` +
          `<div><span>Probability</span><b class="is-${lvl}">${FMT.esc(FMT.pc(effPc))}</b></div>` +
          `<div><span>At</span><b>${FMT.esc(FMT.utc(it.tca).slice(11, 16))} UTC</b></div>` +
        `</div>` +
      `</div>`;
  }

  function renderTopList() {
    if (!els.topList) return;
    const rows = conjItems.slice(0, TOP_N);
    if (!rows.length) {
      els.topList.innerHTML = `<li class="wall-empty">${hasData ? 'Nothing to show.' : 'Waiting for data…'}</li>`;
      return;
    }
    const cur = focusList[focusIdx] ? focusList[focusIdx].id : null;
    els.topList.innerHTML = rows.map((it) => {
      const lvl = safeLevel(it.risk_level);
      const [a, b] = pairOf(it);
      return `<li class="wl${it.id === cur ? ' is-current' : ''}">` +
        `<span class="wl-dot wl-dot--${lvl}" aria-label="${LEVEL_LABELS[lvl]}"></span>` +
        `<span class="wl-pair">${FMT.esc(a)} <span class="wl-and">and</span> ${FMT.esc(b)}</span>` +
        `<span class="wl-miss">${FMT.esc(FMT.dist(it.miss_distance_m))}</span>` +
        `<span class="wl-when" data-tca="${FMT.esc(it.tca)}">${FMT.esc(FMT.countdown(it.tca))}</span>` +
        `</li>`;
    }).join('');
  }

  /* Arm the globe overlay for the current focus event. Returns false (and
     leaves the retry flag down) before the fleet snapshot loads - the 1 s
     tick keeps retrying until the engine accepts. watch stays false: the
     wall never yanks the sim clock around. */
  function tryShowEncounter() {
    encounterApplied = false;
    if (!globeReady || typeof KGlobe.showEncounter !== 'function') return;
    const it = focusList[focusIdx];
    if (!it || !it.primary || !it.secondary) return;
    try {
      encounterApplied = KGlobe.showEncounter({
        primaryNorad: it.primary.norad_id,
        secondaryNorad: it.secondary.norad_id,
        tcaIso: it.tca,
        watch: false,
      }) === true;
    } catch (_) { encounterApplied = false; }
  }

  function advanceFocus() {
    if (!focusList.length) return;
    focusIdx = (focusIdx + 1) % focusList.length;
    renderFocus();
    tryShowEncounter();
    lastCycleAt = Date.now();
  }

  /* --- changes (top 6, flat) ------------------------------------------------------- */

  function lvlWord(level) {
    const lvl = safeLevel(level);
    return `<span class="chg-lvl chg-lvl--${lvl}">${LEVEL_LABELS[lvl]}</span>`;
  }

  function chgPair(text) {
    const t = String(text || '').replace(/\s*↔\s*/g, ' and ');
    return `<span class="chg-pair" title="${FMT.esc(t)}">${FMT.esc(t)}</span>`;
  }

  /* level shifts first (the interesting rows), then new, then resolved -
     one flat list capped at CHANGES_N, no scroll animation */
  function changeRows(data) {
    const rows = [];
    for (const c of (Array.isArray(data.level_changes) ? data.level_changes : [])) {
      rows.push('<div class="chg-row">' +
        `<span class="chg-tag">${lvlWord(c.from_level)}<span class="chg-arrow">→</span>${lvlWord(c.to_level)}</span>` +
        chgPair(c.pair) + '</div>');
    }
    for (const it of (Array.isArray(data.new_conjunctions) ? data.new_conjunctions : [])) {
      const [a, b] = pairOf(it);
      rows.push('<div class="chg-row"><span class="chg-tag">New · ' +
        `${lvlWord(it.risk_level)}</span>${chgPair(`${a} and ${b}`)}</div>`);
    }
    for (const r of (Array.isArray(data.resolved) ? data.resolved : [])) {
      rows.push('<div class="chg-row"><span class="chg-tag">Passed · ' +
        `${lvlWord(r.last_level)}</span>${chgPair(r.pair)}</div>`);
    }
    return rows.slice(0, CHANGES_N);
  }

  /* Mirrors dashboard.js: 404 (older backend) keeps the panel hidden and
     retries next poll; other failures keep the last rendered diff. */
  async function refreshChanges() {
    let data = null;
    try {
      const res = await fetch('/changes');
      if (res.status === 404) { els.changesPanel.hidden = true; return; }
      if (!res.ok) return;
      data = await res.json();
    } catch (_) { return; /* network fault - keep last state */ }
    if (!data || typeof data !== 'object') return;
    const since = data.since && data.since.finished_at;
    els.changesMeta.textContent = since ? `Since ${String(since).slice(11, 16)} UTC` : '';
    const rows = changeRows(data);
    els.changesBody.innerHTML = rows.length
      ? rows.join('')
      : '<div class="wall-empty">Nothing changed since the last refresh.</div>';
    els.changesPanel.hidden = false;
  }

  /* --- poll (never-blank) ------------------------------------------------------------ */

  async function poll() {
    try {
      /* limit 1000 like the dashboard - the globe's risk styling needs the
         full elevated/critical population, not the first page */
      const [st, conj] = await Promise.all([
        API.status(),
        API.conjunctions({ limit: 1000 }),
      ]);
      renderStats(st);
      conjItems = Array.isArray(conj.items) ? conj.items : [];
      const prevKey = focusList[focusIdx] ? focusList[focusIdx].id : null;
      focusList = conjItems.slice(0, FOCUS_N);
      if (focusIdx >= focusList.length) focusIdx = 0;
      if (!hasData) lastCycleAt = Date.now(); // first data: hold focus 1/N a full cycle
      hasData = true;
      lastOkIso = new Date().toISOString().slice(11, 16) + ' UTC';
      renderFocus();
      /* re-arm the overlay only when the event under the cursor changed and
         the operator is not mid-interaction - never fight a live drag */
      const curKey = focusList[focusIdx] ? focusList[focusIdx].id : null;
      if (curKey !== prevKey && Date.now() >= holdUntil) tryShowEncounter();
      if (globeReady) {
        try { KGlobe.setConjunctions(conjItems); } catch (_) { /* engine fault */ }
      }
      els.foot.textContent = `Live · updated ${lastOkIso}` +
        (st.last_run_error ? ' · last catalog refresh failed, data may be stale' : '');
      els.foot.classList.toggle('is-stale', !!st.last_run_error);
    } catch (_) {
      /* never blank - last data stays up, footer flags the staleness */
      els.foot.textContent = lastOkIso
        ? `Connection lost — showing data from ${lastOkIso}, retrying`
        : 'No data yet — retrying';
      els.foot.classList.add('is-stale');
    } finally {
      refreshChanges();
    }
  }

  /* --- 1 s tick: date, countdown, cycle, encounter retry ------------------------------ */

  function tick() {
    const now = new Date();
    els.date.textContent = `UTC · ${DAYS[now.getUTCDay()]}, ${now.getUTCDate()} ${MONTHS[now.getUTCMonth()]}`;
    for (const cd of document.querySelectorAll('.wall-side [data-tca]')) {
      cd.textContent = FMT.countdown(cd.getAttribute('data-tca'));
    }
    if (!focusList.length) return;
    const t = Date.now();
    if (t < holdUntil) return;               // operator owns the globe
    if (t - lastCycleAt >= CYCLE_MS) {
      advanceFocus();
    } else if (!encounterApplied) {
      tryShowEncounter();                    // fleet snapshot may have just landed
    }
    renderFocusMeta();                       // HOLD label clears when the hold expires
  }

  /* --- fullscreen on click (gesture-gated) --------------------------------------------- */

  function onPointerDown(ev) {
    downX = ev.clientX;
    downY = ev.clientY;
  }

  function onDocClick(ev) {
    if (ev.target.closest('a')) return;                     // links navigate
    if (Math.abs(ev.clientX - downX) > 4 ||
        Math.abs(ev.clientY - downY) > 4) return;           // drag, not click
    if (document.fullscreenElement) return;                 // ESC exits normally
    const root = document.documentElement;
    if (root.requestFullscreen) root.requestFullscreen().catch(() => { /* denied */ });
  }

  /* --- init ------------------------------------------------------------------------------ */

  (function clock() {
    const tick = () => { els.clock.textContent = new Date().toISOString().slice(11, 16); };
    tick();
    setInterval(tick, 1000);
  })();
  initGlobe();
  document.addEventListener('pointerdown', onPointerDown);
  document.addEventListener('click', onDocClick);
  setInterval(tick, 1000);
  tick();
  setInterval(poll, POLL_MS);
  poll();
  feedAurora();
  setInterval(feedAurora, 5 * 60 * 1000);
})();
