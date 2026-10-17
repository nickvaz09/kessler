/* Kessler - "Protect a satellite" (/start.html).

   A three-step guided flow a CubeSat team can finish in about a minute:
     1. Which satellite is yours?  catalog typeahead (GET /objects?search=),
        a pasted TLE (checked here first, then strictly by the server), or a
        planned circular orbit turned into a clearly labeled HYPOTHETICAL TLE.
     2. Screening.  POST /user-objects/from-catalog (catalog pick) or
        POST /user-objects (TLE / hypothetical), then
        POST /user-objects/{id}/screen. The progress copy only states real
        facts: catalog size and window from /status, a real elapsed timer.
     3. Your satellite.  A one-sentence verdict that is never an all-clear
        when the screen failed, the catalog is empty/failed/stale, or the
        element set is too old to trust; the close approaches in human
        numbers; links to the simulator (/?sim= when the same pair and TCA
        exist as a catalog event, else /?sel= on the globe) and the fleet;
        "Brief me" asks Kessler AI (POST /assistant/chat) with a graceful
        card when the server has no API key (503).

   js/api.js is a frozen shared client, so the POST helpers live here and
   mirror its error shape (err.status + readable detail). */
'use strict';

(() => {
  /* Documented constants (backend/app/config.py). Shown in copy; never
     used to re-derive risk levels, which come from the API. */
  const SCREEN_THRESHOLD_KM = 15;        // config.SCREENING_THRESHOLD_KM
  const MAX_LEO_APOGEE_KM = 3000;        // config.SCREENING_MAX_LEO_ALT_KM
  /* honesty gates for an all-clear (frontend policy, stated in the copy) */
  const MAX_TLE_AGE_DAYS = 14;           // older element sets drift by km
  const MAX_CATALOG_AGE_H = 72;          // catalog older than the window
  const MIN_STEP2_MS = 1400;             // keep the progress screen readable

  const MU = 398600.4418;                // km^3/s^2 (WGS-84)
  const RE = 6378.137;                   // km

  const $ = (id) => document.getElementById(id);
  const els = {
    stage: $('st-stage'),
    dots: document.querySelectorAll('.st-dot'),
    steps: { 1: $('step-1'), 2: $('step-2'), 3: $('step-3') },
    s1Lede: $('s1-lede'),
    tabs: document.querySelectorAll('.st-modes [role="tab"]'),
    modes: { search: $('mode-search'), tle: $('mode-tle'), plan: $('mode-plan') },
    search: $('search-input'),
    spin: $('search-spin'),
    list: $('search-list'),
    hint: $('search-hint'),
    picked: $('picked'),
    pickedName: $('picked-name'),
    pickedMeta: $('picked-meta'),
    pickedClear: $('picked-clear'),
    tleName: $('tle-name'),
    tleText: $('tle-text'),
    tleMsg: $('tle-msg'),
    planAlt: $('plan-alt'),
    planInc: $('plan-inc'),
    planMsg: $('plan-msg'),
    go: $('s1-go'),
    s2Name: $('s2-name'),
    s2Lede: $('s2-lede'),
    chkAdd: $('chk-add'),
    chkScreen: $('chk-screen'),
    chkScore: $('chk-score'),
    chkTime: $('chk-time'),
    canvas: $('scan-canvas'),
    result: $('result'),
  };

  const state = {
    step: 1,
    mode: 'search',
    status: null,          // last /status payload (null = unknown)
    statusError: false,
    options: [],           // current typeahead items
    active: -1,            // keyboard-highlighted option
    picked: null,          // catalog item chosen in search mode
    tle: null,             // {name, line1, line2, norad, epoch, ageDays}
    plan: null,            // {alt, inc, line1, line2, name}
    busy: false,
    run: null,             // {source, item, screen, conjByPair, created}
  };

  /* --- small fetch helpers (api.js has no POST bodies) ------------------ */

  function detailText(detail) {
    if (detail == null) return '';
    if (typeof detail === 'string') return detail;
    const first = Array.isArray(detail) ? detail[0] : detail;
    if (first && typeof first === 'object' && typeof first.msg === 'string') return first.msg;
    try { return JSON.stringify(detail); } catch (_) { return String(detail); }
  }

  async function req(method, path, body, signal) {
    const opts = { method, headers: { Accept: 'application/json' }, signal };
    if (body !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = detailText((await res.json()).detail) || detail; } catch (_) { /* keep */ }
      const err = new Error(detail || `HTTP ${res.status}`);
      err.status = res.status;
      err.detail = detail;
      throw err;
    }
    return res.status === 204 ? null : res.json();
  }

  /* --- human formatting -------------------------------------------------- */

  const esc = FMT.esc;
  const nf = (n) => Number(n).toLocaleString('en-US');

  function typeLabel(t) {
    const s = String(t || 'UNKNOWN').toUpperCase();
    if (s === 'ROCKET BODY' || s === 'RB' || s === 'R/B') return 'Rocket body';
    return s.charAt(0) + s.slice(1).toLowerCase();
  }

  function titleCase(name) {
    /* Catalog names are upper case; keep them as the catalog spells them
       (they are identifiers) - only the fallback names get softened. */
    return String(name || '');
  }

  function distHtml(m) {
    if (m == null || !Number.isFinite(m)) return '—';
    if (m < 1000) return `${Math.round(m)}<small>m</small>`;
    return `${(m / 1000).toFixed(m < 10000 ? 2 : 1)}<small>km</small>`;
  }
  function distText(m) {
    if (m == null || !Number.isFinite(m)) return '—';
    if (m < 1000) return `${Math.round(m)} m`;
    return `${(m / 1000).toFixed(m < 10000 ? 2 : 1)} km`;
  }
  function speedText(ms) {
    if (ms == null || !Number.isFinite(ms)) return '—';
    return `${(ms / 1000).toFixed(1)} km/s`;
  }

  function roundSig(x) {
    if (x < 100) return Math.round(x);
    const p = Math.pow(10, Math.floor(Math.log10(x)) - 1);
    return Math.round(x / p) * p;
  }
  /* collision probability as "1 in N" - the scientific value goes in title */
  function pcHuman(pc) {
    if (pc == null || !Number.isFinite(pc)) return 'Not available';
    if (pc < 1e-10) return 'Negligible';
    if (pc >= 0.5) return 'Better than even';
    const inv = 1 / pc;
    if (inv >= 1e9) return `1 in ${(roundSig(inv) / 1e9).toLocaleString('en-US', { maximumFractionDigits: 1 })} billion`;
    if (inv >= 1e6) return `1 in ${(roundSig(inv) / 1e6).toLocaleString('en-US', { maximumFractionDigits: 1 })} million`;
    return `1 in ${nf(roundSig(inv))}`;
  }

  /* same relative-time ladder as every other page (FMT.rel) */
  function relTime(iso) {
    const t = Date.parse(iso);
    if (!Number.isFinite(t)) return '—';
    if (Math.abs(t - Date.now()) < 60000) return t >= Date.now() ? 'in under a minute' : 'just now';
    return FMT.rel(iso);
  }
  function whenUtc(iso) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    const day = d.toLocaleDateString('en-US', { weekday: 'short', timeZone: 'UTC' });
    return `${day} ${d.toISOString().slice(11, 16)} UTC`;
  }
  function ageText(hours) {
    if (hours < 1) return `${Math.max(1, Math.round(hours * 60))} min`;
    if (hours < 48) return `${Math.round(hours)} h`;
    return `${Math.round(hours / 24)} days`;
  }
  function orbitText(o) {
    const parts = [];
    if (o.perigee_km != null && o.apogee_km != null) {
      parts.push(`${nf(Math.round(o.perigee_km))} × ${nf(Math.round(o.apogee_km))} km`);
    }
    if (o.inclination_deg != null) parts.push(`${o.inclination_deg.toFixed(1)}°`);
    return parts.join(' · ');
  }

  const ICON = {
    sat: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9.2" y="9.2" width="5.6" height="5.6" rx="1" transform="rotate(45 12 12)"/><path d="M8.6 8.6 5.2 5.2M15.4 15.4l3.4 3.4"/><path d="M3.2 7.2l4-4 2 2-4 4zM16.8 20.8l4-4-2-2-4 4z"/></svg>',
    debris: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" aria-hidden="true"><path d="M7 6.5 13.5 4l4.5 5-2 7.5-6.5 2L5 13z"/></svg>',
    rocket: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" aria-hidden="true"><rect x="8.5" y="3" width="7" height="15" rx="3.5"/><path d="M8.5 14 6 18.5h3M15.5 14l2.5 4.5h-3M10.5 18v2.5h3V18"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><path d="M12 7v6.5"/><circle cx="12" cy="17.2" r="0.6" fill="currentColor"/></svg>',
    question: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9.3 9.2a2.8 2.8 0 1 1 3.9 2.6c-.8.4-1.2 1-1.2 1.8v.6"/><circle cx="12" cy="17.4" r="0.6" fill="currentColor"/></svg>',
    play: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5.8v12.4a.8.8 0 0 0 1.2.7l9.8-6.2a.8.8 0 0 0 0-1.4L9.2 5.1a.8.8 0 0 0-1.2.7z"/></svg>',
    fleet: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="2.2"/><ellipse cx="12" cy="12" rx="9" ry="4" transform="rotate(-25 12 12)"/></svg>',
    mark: '<svg viewBox="0 0 24 24" aria-hidden="true"><g fill="none"><ellipse cx="12" cy="12" rx="9" ry="3.4" transform="rotate(30 12 12)" stroke="currentColor" stroke-opacity=".9" stroke-width="1.6"/><ellipse cx="12" cy="12" rx="9" ry="3.4" transform="rotate(-30 12 12)" stroke="currentColor" stroke-opacity=".5" stroke-width="1.6"/></g><circle cx="12" cy="12" r="1.5" fill="currentColor"/></svg>',
  };
  function glyphFor(type) {
    const t = String(type || '').toUpperCase();
    if (t === 'DEBRIS') return ICON.debris;
    if (t === 'ROCKET BODY' || t === 'RB' || t === 'R/B') return ICON.rocket;
    return ICON.sat;
  }

  /* --- step navigation ----------------------------------------------------- */

  function setDots(step) {
    els.dots.forEach((d) => {
      const n = Number(d.dataset.step);
      d.classList.toggle('is-current', n === step);
      d.classList.toggle('is-done', n < step);
      if (n === step) d.setAttribute('aria-current', 'step');
      else d.removeAttribute('aria-current');
    });
  }

  function showStep(step, { back = false } = {}) {
    const from = els.steps[state.step];
    const to = els.steps[step];
    if (from && from !== to) {
      from.hidden = true;
      from.classList.remove('st-enter', 'st-enter--back');
    }
    to.hidden = false;
    to.classList.remove('st-enter', 'st-enter--back');
    void to.offsetWidth;   // restart the entrance animation
    to.classList.add(back ? 'st-enter--back' : 'st-enter');
    state.step = step;
    els.stage.classList.toggle('is-wide', step === 3);
    setDots(step);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    if (step === 2) scan.start(); else scan.stop();
  }

  /* --- /status -------------------------------------------------------------- */

  async function loadStatus() {
    try {
      state.status = await API.status();
      state.statusError = false;
    } catch (_) {
      state.status = null;
      state.statusError = true;
    }
    const st = state.status;
    if (st && st.catalog_size > 0) {
      els.s1Lede.innerHTML =
        `Find it among the <strong class="tnum">${nf(st.catalog_size)}</strong> objects Kessler tracks, ` +
        'or paste its two-line element set.';
    } else if (st) {
      els.s1Lede.textContent = 'The tracked catalog is empty right now, so search will not find anything. You can still paste a TLE.';
    } else {
      els.s1Lede.textContent = 'The Kessler server is not answering right now. Check that it is running, then reload.';
    }
    return st;
  }

  /* --- mode tabs ------------------------------------------------------------- */

  function setMode(mode, focus) {
    state.mode = mode;
    els.tabs.forEach((t) => {
      const on = t.dataset.mode === mode;
      t.classList.toggle('is-active', on);
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      if (on && focus) t.focus();
    });
    for (const [k, el] of Object.entries(els.modes)) el.hidden = k !== mode;
    closeList();
    updateGo();
    if (!focus) return;
  }

  els.tabs.forEach((t, i) => {
    t.addEventListener('click', () => {
      setMode(t.dataset.mode);
      const target = { search: els.search, tle: els.tleText, plan: els.planAlt }[t.dataset.mode];
      if (target && !matchMedia('(pointer: coarse)').matches) target.focus();
    });
    t.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault();
      const n = els.tabs.length;
      const next = els.tabs[(i + (e.key === 'ArrowRight' ? 1 : n - 1)) % n];
      setMode(next.dataset.mode, true);
    });
  });

  function updateGo() {
    let ok = false;
    if (state.mode === 'search') ok = !!state.picked;
    else if (state.mode === 'tle') ok = !!state.tle;
    else if (state.mode === 'plan') ok = !!state.plan;
    els.go.disabled = !ok || state.busy;
  }

  /* --- catalog typeahead ------------------------------------------------------- */

  let searchTimer = null;
  let searchCtl = null;
  let searchSeq = 0;
  const LIMIT = 8;

  function closeList() {
    els.list.hidden = true;
    els.search.setAttribute('aria-expanded', 'false');
    els.search.removeAttribute('aria-activedescendant');
    state.active = -1;
  }

  function highlight(name, q) {
    const s = String(name || '');
    const i = q && !/^\d+$/.test(q) ? s.toLowerCase().indexOf(q.toLowerCase()) : -1;
    if (i < 0) return esc(s);
    return esc(s.slice(0, i)) + '<mark>' + esc(s.slice(i, i + q.length)) + '</mark>' + esc(s.slice(i + q.length));
  }

  function renderOptions(q, count) {
    const items = state.options;
    if (!items.length) {
      els.list.innerHTML =
        `<li class="st-opt st-opt--empty" role="option" aria-disabled="true">No tracked object matches “${esc(q)}”. ` +
        'Try another spelling, a NORAD id, or paste a TLE.</li>';
    } else {
      els.list.innerHTML = items.map((o, i) => {
        const orbit = orbitText(o);
        return `<li class="st-opt${i === state.active ? ' is-active' : ''}" role="option" id="opt-${i}" data-i="${i}"` +
          ` aria-selected="${i === state.active}">` +
          `<span class="st-opt__glyph">${glyphFor(o.object_type)}</span>` +
          '<span class="st-opt__body">' +
            `<span class="st-opt__name" style="display:block">${highlight(o.name, q)}</span>` +
            `<span class="st-opt__meta" style="display:block">NORAD <span class="code">${esc(o.norad_id)}</span> · ${esc(typeLabel(o.object_type))}${orbit ? ' · ' + esc(orbit) : ''}</span>` +
          '</span><span class="st-opt__go" aria-hidden="true">›</span></li>';
      }).join('') + (count > items.length
        ? `<li class="st-results__foot" role="presentation">Showing ${items.length} of ${nf(count)} matches. Keep typing to narrow it down.</li>`
        : '');
    }
    els.list.hidden = false;
    els.search.setAttribute('aria-expanded', 'true');
  }

  function setActive(i) {
    const n = state.options.length;
    if (!n) return;
    state.active = (i + n) % n;
    els.list.querySelectorAll('.st-opt').forEach((li) => {
      const on = Number(li.dataset.i) === state.active;
      li.classList.toggle('is-active', on);
      li.setAttribute('aria-selected', String(on));
      if (on) li.scrollIntoView({ block: 'nearest' });
    });
    els.search.setAttribute('aria-activedescendant', `opt-${state.active}`);
  }

  async function runSearch(q) {
    const seq = ++searchSeq;
    if (searchCtl) searchCtl.abort();
    searchCtl = new AbortController();
    els.spin.hidden = false;
    try {
      const url = `/objects?search=${encodeURIComponent(q)}&limit=${LIMIT}`;
      const data = await req('GET', url, undefined, searchCtl.signal);
      if (seq !== searchSeq) return;
      state.options = data.items || [];
      state.active = state.options.length ? 0 : -1;
      renderOptions(q, data.count || 0);
      if (state.active >= 0) els.search.setAttribute('aria-activedescendant', 'opt-0');
    } catch (err) {
      if (err.name === 'AbortError' || seq !== searchSeq) return;
      state.options = [];
      els.list.innerHTML = '<li class="st-opt st-opt--empty" role="option" aria-disabled="true">Search is unavailable: the server did not answer.</li>';
      els.list.hidden = false;
    } finally {
      if (seq === searchSeq) els.spin.hidden = true;
    }
  }

  els.search.addEventListener('input', () => {
    const q = els.search.value.trim();
    clearTimeout(searchTimer);
    if (state.picked && q !== state.picked.name) unpick(false);
    if (!q) { closeList(); els.spin.hidden = true; return; }
    searchTimer = setTimeout(() => runSearch(q), 140);
  });

  els.search.addEventListener('keydown', (e) => {
    const open = !els.list.hidden && state.options.length;
    if (e.key === 'ArrowDown' && open) { e.preventDefault(); setActive(state.active + 1); }
    else if (e.key === 'ArrowUp' && open) { e.preventDefault(); setActive(state.active - 1); }
    else if (e.key === 'Enter') {
      e.preventDefault();
      if (open && state.active >= 0) pick(state.options[state.active]);
      else if (state.picked) go();
    } else if (e.key === 'Escape') {
      if (!els.list.hidden) { e.preventDefault(); closeList(); }
    }
  });
  els.search.addEventListener('focus', () => {
    if (!state.picked && els.search.value.trim() && state.options.length) renderOptions(els.search.value.trim(), 0);
  });
  els.list.addEventListener('mousedown', (e) => e.preventDefault()); // keep focus in the input
  els.list.addEventListener('click', (e) => {
    const li = e.target.closest('.st-opt[data-i]');
    if (li) pick(state.options[Number(li.dataset.i)]);
  });
  document.addEventListener('click', (e) => {
    if (!e.target.closest('#st-search')) closeList();
  });
  els.hint.addEventListener('click', (e) => {
    const b = e.target.closest('.st-try');
    if (!b) return;
    unpick(false);
    els.search.value = b.dataset.q;
    els.search.focus();
    runSearch(b.dataset.q);
  });

  function pick(o) {
    if (!o) return;
    state.picked = o;
    closeList();
    els.search.value = o.name;
    els.pickedName.textContent = titleCase(o.name);
    const orbit = orbitText(o);
    els.pickedMeta.innerHTML =
      `NORAD <span class="code">${esc(o.norad_id)}</span> · ${esc(typeLabel(o.object_type))}` +
      (orbit ? ` · ${esc(orbit)}` : '') +
      (o.epoch ? ` · elements from ${esc(relAgeFromIso(o.epoch))}` : '');
    els.picked.hidden = false;
    els.hint.hidden = true;
    updateGo();
    els.go.focus({ preventScroll: true });
  }
  function unpick(clearInput = true) {
    state.picked = null;
    els.picked.hidden = true;
    els.hint.hidden = false;
    if (clearInput) { els.search.value = ''; els.search.focus(); }
    updateGo();
  }
  els.pickedClear.addEventListener('click', () => unpick(true));

  function relAgeFromIso(iso) {
    const h = (Date.now() - Date.parse(iso)) / 3600000;
    if (!Number.isFinite(h)) return 'an unknown date';
    return h < 0 ? 'the future' : `${ageText(h)} ago`;
  }

  /* --- TLE paste -------------------------------------------------------------- */

  function tleChecksum(line) {
    let sum = 0;
    for (const ch of line.slice(0, 68)) {
      if (ch >= '0' && ch <= '9') sum += Number(ch);
      else if (ch === '-') sum += 1;
    }
    return sum % 10;
  }

  function tleEpoch(line1) {
    const yy = Number(line1.slice(18, 20));
    const doy = Number(line1.slice(20, 32));
    if (!Number.isFinite(yy) || !Number.isFinite(doy)) return null;
    const year = yy < 57 ? 2000 + yy : 1900 + yy;
    return new Date(Date.UTC(year, 0, 1) + (doy - 1) * 86400000);
  }

  /* Returns {ok, msg, level, tle?}. Mirrors the server's strict gates with
     friendlier words; the server still has the final say (422s surface). */
  function parseTle(raw) {
    const lines = raw.split(/\r?\n/).map((l) => l.replace(/\s+$/, '')).filter((l) => l.trim() !== '');
    if (!lines.length) return { ok: false, level: '', msg: 'Paste lines 1 and 2. A name line above them is fine too.' };
    const i1 = lines.findIndex((l) => l.trimStart().startsWith('1 '));
    const i2 = lines.findIndex((l) => l.trimStart().startsWith('2 '));
    if (i1 < 0 && i2 < 0) {
      return { ok: false, level: 'error', msg: 'That does not look like a TLE yet: line 1 starts with “1 ” and line 2 with “2 ”.' };
    }
    if (i1 < 0) return { ok: false, level: 'error', msg: 'Line 1 is missing. It starts with “1 ” and comes before line 2.' };
    if (i2 < 0) return { ok: false, level: 'error', msg: 'Line 2 is missing. It starts with “2 ” and comes right after line 1.' };
    if (i2 < i1) return { ok: false, level: 'error', msg: 'The lines look swapped: line 1 (starting “1 ”) must come first.' };
    const l1 = lines[i1].trim();
    const l2 = lines[i2].trim();
    let name = '';
    if (i1 > 0) name = lines[i1 - 1].trim().replace(/^0\s+/, '');
    for (const [label, l] of [['Line 1', l1], ['Line 2', l2]]) {
      if (l.length !== 69) {
        const hint = l.length < 69 ? 'Did a character get lost when copying?' : 'Is there extra text on the line?';
        return { ok: false, level: 'error', msg: `${label} has ${l.length} characters; a TLE line has exactly 69. ${hint}` };
      }
      const want = tleChecksum(l);
      const got = Number(l[68]);
      if (got !== want) {
        return { ok: false, level: 'error', msg: `${label} fails its checksum (the last digit should be ${want}, not ${l[68]}). A digit may have been mistyped.` };
      }
    }
    const n1 = l1.slice(2, 7).trim();
    const n2 = l2.slice(2, 7).trim();
    if (n1 !== n2) return { ok: false, level: 'error', msg: `The two lines describe different objects (NORAD ${n1} and ${n2}).` };
    const epoch = tleEpoch(l1);
    if (!epoch) return { ok: false, level: 'error', msg: 'Line 1 has an unreadable epoch (columns 19–32).' };
    const ageDays = (Date.now() - epoch.getTime()) / 86400000;
    const tle = { name, line1: l1, line2: l2, norad: n1, epoch: epoch.toISOString(), ageDays };
    const when = epoch.toISOString().slice(0, 10);
    if (ageDays > MAX_TLE_AGE_DAYS) {
      return {
        ok: true, level: 'warn', tle,
        msg: `Readable, but this element set is from ${when} (${nf(Math.round(ageDays))} days old). ` +
          'Old TLEs drift by kilometres, so Kessler will not give an all-clear from it. Paste a fresh one if you can.',
      };
    }
    if (ageDays < -2) {
      return { ok: true, level: 'warn', tle, msg: `Readable, but its epoch (${when}) is in the future. Check the date columns.` };
    }
    return {
      ok: true, level: 'ok', tle,
      msg: `Looks good · NORAD ${n1}${name ? ` · ${name}` : ''} · epoch ${when} (${ageText(Math.max(ageDays, 0) * 24)} old)`,
    };
  }

  function onTleInput() {
    const r = parseTle(els.tleText.value);
    state.tle = r.ok ? r.tle : null;
    els.tleMsg.textContent = r.msg;
    els.tleMsg.className = 'st-msg' + (r.level ? ` is-${r.level}` : '');
    if (r.tle && r.tle.name && !els.tleName.value.trim()) els.tleName.placeholder = r.tle.name;
    updateGo();
  }
  els.tleText.addEventListener('input', onTleInput);
  els.tleText.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && state.tle) { e.preventDefault(); go(); }
  });
  els.tleName.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && state.tle) { e.preventDefault(); go(); }
  });

  /* --- planned orbit -> hypothetical TLE ---------------------------------------- */

  function fmtField(v, width, digits) {
    return v.toFixed(digits).padStart(width, ' ');
  }

  /* Circular orbit at `alt` km, inclination `inc`. RAAN, argument of perigee
     and mean anomaly are 0 - one arbitrary slot in that orbit, which is why
     every surface labels it a hypothetical orbit. */
  function hypotheticalTle(alt, inc, now = new Date()) {
    const a = RE + alt;
    const revPerDay = Math.sqrt(MU / (a * a * a)) * 86400 / (2 * Math.PI);
    const year = now.getUTCFullYear();
    const doy = (now.getTime() - Date.UTC(year, 0, 1)) / 86400000 + 1;
    const yy = String(year % 100).padStart(2, '0');
    const norad = '99999';
    let l1 = `1 ${norad}U          ${yy}${doy.toFixed(8).padStart(12, '0')}  .00000000  00000-0  00000-0 0  999`;
    l1 += tleChecksum(l1);
    let l2 = `2 ${norad} ${fmtField(inc, 8, 4)} ${fmtField(0, 8, 4)} 0001000 ${fmtField(0, 8, 4)} ` +
      `${fmtField(0, 8, 4)} ${fmtField(revPerDay, 11, 8)}    0`;
    l2 += tleChecksum(l2);
    return { line1: l1, line2: l2 };
  }

  function onPlanInput() {
    const alt = Number(els.planAlt.value);
    const inc = Number(els.planInc.value);
    let msg = '';
    if (!els.planAlt.value || !Number.isFinite(alt) || alt < 160 || alt > 2000) {
      msg = 'Altitude must be between 160 and 2,000 km.';
    } else if (!els.planInc.value || !Number.isFinite(inc) || inc < 0 || inc > 180) {
      msg = 'Inclination must be between 0° and 180°.';
    }
    if (msg) {
      state.plan = null;
      els.planMsg.textContent = msg;
      els.planMsg.className = 'st-msg is-error';
    } else {
      const t = hypotheticalTle(alt, inc);
      const period = 2 * Math.PI * Math.sqrt(Math.pow(RE + alt, 3) / MU) / 60;
      state.plan = {
        alt, inc, ...t,
        name: `Hypothetical orbit · ${nf(Math.round(alt))} km, ${inc.toFixed(1)}°`,
      };
      els.planMsg.textContent = `Hypothetical circular orbit · one lap every ${period.toFixed(1)} minutes.`;
      els.planMsg.className = 'st-msg';
    }
    updateGo();
  }
  els.planAlt.addEventListener('input', onPlanInput);
  els.planInc.addEventListener('input', onPlanInput);
  [els.planAlt, els.planInc].forEach((el) => el.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && state.plan) { e.preventDefault(); go(); }
  }));

  /* --- step 2: create + screen --------------------------------------------------- */

  function setCheck(el, cls, text) {
    el.classList.remove('is-active', 'is-done', 'is-failed');
    if (cls) el.classList.add(cls);
    if (text) el.querySelector('.st-check__text').textContent = text;
  }

  let timer = null;
  function startTimer() {
    const t0 = performance.now();
    els.chkTime.textContent = '';
    clearInterval(timer);
    timer = setInterval(() => {
      const s = Math.floor((performance.now() - t0) / 1000);
      els.chkTime.textContent = s >= 1 ? `${s} s` : '';
    }, 250);
    return () => { clearInterval(timer); return (performance.now() - t0) / 1000; };
  }

  function sourceLabel() {
    if (state.mode === 'search') return state.picked.name;
    if (state.mode === 'tle') return (els.tleName.value.trim() || state.tle.name || `NORAD ${state.tle.norad}`);
    return state.plan.name;
  }

  async function findOrCreate() {
    if (state.mode === 'search') {
      const item = await req('POST', '/user-objects/from-catalog', { norad_id: state.picked.norad_id });
      return { item, created: item.created !== false };
    }
    if (state.mode === 'tle') {
      /* same element set already in the fleet? reuse it (no duplicate rows) */
      try {
        const list = await req('GET', '/user-objects');
        const want = Number(state.tle.norad);
        const epochIso = state.tle.epoch.slice(0, 19);
        const hit = (list.items || []).find((u) => u.norad_id === want && u.epoch &&
          Math.abs(Date.parse(u.epoch) - Date.parse(epochIso + 'Z')) < 1500);
        if (hit) return { item: hit, created: false };
      } catch (_) { /* fall through to create */ }
      const name = els.tleName.value.trim() || state.tle.name || undefined;
      const item = await req('POST', '/user-objects', {
        name: name ? name.slice(0, 64) : undefined, line1: state.tle.line1, line2: state.tle.line2,
        note: 'Added with Protect a satellite (pasted TLE)',
      });
      return { item, created: true };
    }
    const item = await req('POST', '/user-objects', {
      name: state.plan.name.slice(0, 64), line1: state.plan.line1, line2: state.plan.line2,
      note: 'Hypothetical orbit generated from altitude and inclination. Not a real satellite.',
    });
    return { item, created: true };
  }

  /* catalog events for the same object: {partnerNorad: [summaries]} */
  async function catalogEvents(norad) {
    try {
      const obj = await req('GET', `/objects/${encodeURIComponent(norad)}`);
      const map = new Map();
      for (const c of obj.conjunctions || []) {
        if (c.status && c.status !== 'active') continue;
        const partner = c.primary && c.primary.norad_id === norad ? c.secondary : c.primary;
        if (!partner) continue;
        if (!map.has(partner.norad_id)) map.set(partner.norad_id, []);
        map.get(partner.norad_id).push(c);
      }
      return map;
    } catch (_) {
      return new Map();
    }
  }

  /* the catalog event for this exact encounter: same pair, TCA within 1 h */
  function matchConj(run, ev) {
    if (!run.conjByPair) return null;
    const list = run.conjByPair.get(ev.secondary.norad_id) || [];
    const t = Date.parse(ev.tca);
    let best = null;
    for (const c of list) {
      const dt = Math.abs(Date.parse(c.tca) - t);
      if (dt <= 3600000 && (!best || dt < best.dt)) best = { dt, c };
    }
    return best ? best.c : null;
  }

  async function go() {
    if (els.go.disabled || state.busy) return;
    state.busy = true;
    updateGo();
    const source = state.mode;
    const label = sourceLabel();
    const st = state.status;

    els.s2Name.textContent = label;
    const n = st && st.catalog_size > 0 ? nf(st.catalog_size) : null;
    const win = st && st.window_hours ? st.window_hours : 72;
    els.s2Lede.innerHTML = n
      ? `Checking against <strong class="tnum">${n}</strong> tracked objects over the next <strong class="tnum">${win}</strong> hours…`
      : `Checking against the tracked catalog over the next <strong class="tnum">${win}</strong> hours…`;
    setCheck(els.chkAdd, 'is-active', 'Adding it to your fleet');
    setCheck(els.chkScreen, null, 'Propagating every orbit with SGP4 and searching for close approaches');
    setCheck(els.chkScore, null, 'Preparing your results');
    els.chkTime.textContent = '';
    clearStepError();
    showStep(2);
    const t0 = performance.now();

    let phase = 'add';
    try {
      const { item, created } = await findOrCreate();
      setCheck(els.chkAdd, 'is-done', created ? 'Added to your fleet' : 'Already in your fleet');
      phase = 'screen';
      setCheck(els.chkScreen, 'is-active');
      const stop = startTimer();
      const screen = await req('POST', `/user-objects/${item.id}/screen`);
      rememberScreen(item.id, screen);
      const secs = stop();
      setCheck(els.chkScreen, 'is-done',
        `Screened ${n ? `against ${n} objects` : 'against the catalog'} in ${Math.max(1, Math.round(secs))} s`);
      els.chkTime.textContent = '';
      phase = 'prepare';
      setCheck(els.chkScore, 'is-active');
      const [conjByPair] = await Promise.all([
        source === 'search' ? catalogEvents(state.picked.norad_id) : Promise.resolve(null),
        loadStatus(),
      ]);
      setCheck(els.chkScore, 'is-done', 'Results ready');
      const wait = MIN_STEP2_MS - (performance.now() - t0);
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      state.run = {
        source, label, item, created, screen, conjByPair,
        catalogNorad: source === 'search' ? state.picked.norad_id : null,
        catalogType: source === 'search' ? state.picked.object_type : null,
        tleAgeDays: item.epoch ? (Date.now() - Date.parse(item.epoch)) / 86400000 : null,
      };
      renderResult();
      showStep(3);
      try { history.pushState({ step: 3 }, '', window.location.pathname + window.location.search); } catch (_) { /* ignore */ }
      $('s3-title') && $('s3-title').focus({ preventScroll: true });
    } catch (err) {
      clearInterval(timer);
      els.chkTime.textContent = '';
      failStep2(phase, err);
    } finally {
      state.busy = false;
      updateGo();
    }
  }
  els.go.addEventListener('click', go);

  /* Fleet shows the last screen of each satellite: same per-browser
     summary fleet.js keeps in localStorage 'kessler.screens'. */
  function rememberScreen(id, resp) {
    try {
      const events = Array.isArray(resp && resp.events) ? resp.events : [];
      const counts = { critical: 0, elevated: 0, nominal: 0 };
      for (const e of events) {
        const l = e && e.risk_level;
        counts[l === 'critical' || l === 'elevated' ? l : 'nominal'] += 1;
      }
      const saved = JSON.parse(localStorage.getItem('kessler.screens') || '{}') || {};
      saved[id] = {
        count: events.length,
        worst: counts.critical ? 'critical' : counts.elevated ? 'elevated' : 'nominal',
        critical: counts.critical, elevated: counts.elevated,
        hours: resp && resp.window_hours != null ? Math.round(resp.window_hours) : 72,
        at: (resp && resp.screened_at) || new Date().toISOString(),
      };
      localStorage.setItem('kessler.screens', JSON.stringify(saved));
    } catch (_) { /* storage blocked - Fleet just shows "Not screened yet" */ }
  }

  function clearStepError() {
    const old = $('s2-error');
    if (old) old.remove();
  }

  function failStep2(phase, err) {
    const el = phase === 'add' ? els.chkAdd : phase === 'screen' ? els.chkScreen : els.chkScore;
    setCheck(el, 'is-failed');
    scan.stop();
    let msg;
    if (phase === 'add' && err.status === 422) {
      msg = `The server could not use this element set: ${err.detail}.`;
    } else if (phase === 'add' && err.status === 413) {
      msg = 'Your fleet is full. Remove a satellite on the Fleet page, then try again.';
    } else if (phase === 'add' && err.status === 404) {
      msg = 'That object is no longer in the tracked catalog. Search again or paste its TLE.';
    } else if (err.status) {
      msg = `The ${phase === 'screen' ? 'screen' : 'request'} failed (${err.status}: ${err.detail}). No verdict is shown, because we could not check.`;
    } else {
      msg = 'The Kessler server did not answer. No verdict is shown, because we could not check.';
    }
    els.s2Lede.textContent = msg;
    const box = document.createElement('div');
    box.className = 'st-actions';
    box.id = 's2-error';
    box.style.justifyContent = 'center';
    box.innerHTML =
      '<button class="btn btn--primary btn--lg" type="button" data-act="retry">Try again</button>' +
      '<button class="btn btn--lg" type="button" data-act="back">Back</button>';
    els.steps[2].appendChild(box);
    box.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      if (b.dataset.act === 'retry') go();
      else { clearStepError(); showStep(1, { back: true }); }
    });
    box.querySelector('button').focus();
  }

  /* --- step 3: verdict + results ------------------------------------------------ */

  const LEO_SCREEN_MAX_KM = 3000; /* config.SCREENING_MAX_LEO_ALT_KM */

  function verdictFor(run) {
    const st = state.status;
    const events = run.screen.events || [];
    const win = run.screen.window_hours || 72;
    const critical = events.filter((e) => e.risk_level === 'critical');
    const elevated = events.filter((e) => e.risk_level === 'elevated');
    const attention = critical.length + elevated.length;

    /* reasons we cannot call it clear, whatever the screen found */
    const doubts = [];
    if (!st) doubts.push('we could not read the catalog status');
    else if (st.last_run_error) doubts.push('the last catalog refresh failed');
    else if (!st.catalog_size || !st.last_ingest_at) doubts.push('the tracked catalog is empty');
    else {
      const ageH = (Date.now() - Date.parse(st.last_ingest_at)) / 3600000;
      if (ageH > MAX_CATALOG_AGE_H) doubts.push(`the catalog was last updated ${ageText(ageH)} ago`);
    }
    if (run.tleAgeDays != null && run.tleAgeDays > MAX_TLE_AGE_DAYS) {
      doubts.push(`your element set is ${nf(Math.round(run.tleAgeDays))} days old`);
    }
    if (run.item.perigee_km == null) doubts.push('we could not derive an orbit from the element set');
    /* screening only covers catalog objects with apogee below 3,000 km -
       above that band an empty result says nothing about your neighbours */
    if (run.item.apogee_km != null && run.item.apogee_km > LEO_SCREEN_MAX_KM) {
      doubts.push(`your orbit reaches ${nf(Math.round(run.item.apogee_km))} km, above the low-Earth-orbit band (below ${nf(LEO_SCREEN_MAX_KM)} km) that Kessler screens`);
    }

    if (attention) {
      const lvl = critical.length ? 'critical' : 'elevated';
      const title = attention === 1 ? '1 close approach needs attention' : `${attention} close approaches need attention`;
      const top = (critical[0] || elevated[0]);
      const sub = `The riskiest is ${esc(top.secondary.name)}, ${esc(relTime(top.tca))}: ` +
        `${esc(distText(top.miss_distance_m))} apart with a collision probability of about ${esc(pcHuman(top.pc).toLowerCase())}.` +
        (doubts.length ? ` Also note: ${esc(doubts.join('; '))}.` : '');
      return { lvl, title, sub, icon: ICON.alert };
    }
    if (doubts.length) {
      const found = events.length
        ? ` The screen found ${events.length === 1 ? '1 pass' : `${events.length} passes`} within ${SCREEN_THRESHOLD_KM} km, none flagged, but`
        : ' The screen found nothing, but';
      return {
        lvl: 'unknown', icon: ICON.question,
        title: 'We can’t give an all-clear yet',
        sub: `${found} ${esc(doubts.join(' and '))}, so an empty list is not proof of safety.`,
      };
    }
    const title = `All clear for the next ${win} hours`;
    if (!events.length) {
      return { lvl: 'ok', icon: ICON.check, title, sub: `No tracked object comes within ${SCREEN_THRESHOLD_KM} km of ${esc(run.label)}.` };
    }
    const closest = events.reduce((a, b) => (b.miss_distance_m < a.miss_distance_m ? b : a));
    const n = events.length;
    return {
      lvl: 'ok', icon: ICON.check, title,
      sub: `${n === 1 ? '1 object passes' : `${n} objects pass`} within ${SCREEN_THRESHOLD_KM} km. ` +
        `The closest, ${esc(closest.secondary.name)}, comes to ${esc(distText(closest.miss_distance_m))}` +
        `${closest.pc != null && closest.pc < 1e-10 ? ' with a negligible chance of collision.' : ', well below the 1-in-a-million attention threshold.'}`,
    };
  }

  function simTarget(run) {
    const events = run.screen.events || [];
    for (const ev of events) {
      const c = matchConj(run, ev);
      if (c) return { href: `/?sim=${encodeURIComponent(c.id)}`, label: 'Open in simulator', title: `Replay the encounter with ${ev.secondary.name}` };
    }
    if (run.catalogNorad != null) {
      return { href: `/?sel=${encodeURIComponent(run.catalogNorad)}`, label: 'Show on the globe', title: 'Open the live globe with your satellite selected' };
    }
    return null;
  }

  function eventRow(run, ev) {
    const lvl = ev.risk_level === 'critical' || ev.risk_level === 'elevated' ? ev.risk_level : 'nominal';
    const c = matchConj(run, ev);
    const sec = ev.secondary;
    const pcTitle = ev.pc != null ? `Pc ${FMT.pc(ev.pc)}` : 'Pc not available';
    const levelChip = lvl === 'nominal'
      ? '<span class="chip chip--ok">Low risk</span>'
      : `<span class="chip chip--${lvl}">${lvl === 'critical' ? 'Critical' : 'Elevated'}</span>`;
    return `<li class="st-ev st-ev--${lvl}">` +
      '<div class="st-ev__who">' +
        `<div class="st-ev__name"><a href="/object.html?id=${encodeURIComponent(sec.norad_id)}">${esc(sec.name)}</a></div>` +
        `<div class="st-ev__sub">${levelChip}<span>${esc(typeLabel(sec.object_type))}</span>` +
          `<span class="code">${esc(sec.norad_id)}</span></div>` +
      '</div>' +
      `<div><div class="st-ev__k">Closest approach</div><div class="st-ev__v" title="${esc(FMT.utc(ev.tca))}">${esc(relTime(ev.tca))}</div>` +
        `<div class="st-ev__k">${esc(whenUtc(ev.tca))}</div></div>` +
      `<div><div class="st-ev__k">Miss distance</div><div class="st-ev__v">${distHtml(ev.miss_distance_m)}</div>` +
        `<div class="st-ev__k">at ${esc(speedText(ev.relative_velocity_ms))}</div></div>` +
      `<div><div class="st-ev__k">Collision probability</div><div class="st-ev__v${lvl !== 'nominal' ? ' is-' + lvl : ''}" title="${esc(pcTitle)}">${esc(pcHuman(ev.pc))}</div>` +
        `<div class="st-ev__k">${esc(pcTitle)}</div></div>` +
      '<div class="st-ev__act">' +
        (c ? `<a class="btn btn--xs" href="/?sim=${encodeURIComponent(c.id)}">Simulate</a>` : '') +
      '</div>' +
    '</li>';
  }

  const EV_SHOW = 6;

  function renderResult() {
    const run = state.run;
    const st = state.status;
    const events = run.screen.events || [];
    const v = verdictFor(run);
    const win = run.screen.window_hours || 72;
    const item = run.item;
    const orbit = orbitText(item);
    const hypothetical = run.source === 'plan';
    const idPart = hypothetical
      ? '<span class="chip chip--accent chip--plain">Hypothetical orbit</span>'
      : (item.norad_id != null ? `<span class="st-sat__meta">NORAD <span class="code">${esc(item.norad_id)}</span></span>` : '');
    const closest = events.length ? events.reduce((a, b) => (b.miss_distance_m < a.miss_distance_m ? b : a)) : null;
    const sim = simTarget(run);
    const verdictCls = v.lvl === 'unknown' ? '' : ` st-verdict--${v.lvl}`;

    let html = '';
    html += '<div class="st-sat">' +
      `<span class="st-sat__name">${esc(item.name)}</span>${idPart}` +
      (orbit ? `<span class="st-sat__meta">${esc(orbit)}${item.period_min ? ` · ${item.period_min.toFixed(1)} min` : ''}</span>` : '') +
      '</div>';
    html += `<div class="st-verdict${verdictCls}">` +
      `<div class="st-verdict__icon">${v.icon}</div>` +
      '<div class="st-verdict__text">' +
        `<h1 class="st-verdict__title" id="s3-title" tabindex="-1">${esc(v.title)}</h1>` +
        `<p class="st-verdict__sub">${v.sub}</p>` +
      '</div></div>';

    html += '<div class="st-facts">' +
      '<div class="st-fact"><div class="st-fact__label">Checked against</div>' +
        `<div class="st-fact__value">${st && st.catalog_size ? nf(st.catalog_size) : '—'}</div>` +
        '<div class="st-fact__sub">tracked objects in the catalog</div></div>' +
      '<div class="st-fact"><div class="st-fact__label">Looking ahead</div>' +
        `<div class="st-fact__value">${esc(win)}<small>hours</small></div>` +
        `<div class="st-fact__sub">screened ${esc(whenUtc(run.screen.screened_at))}</div></div>` +
      '<div class="st-fact"><div class="st-fact__label">Closest approach</div>' +
        `<div class="st-fact__value">${closest ? distHtml(closest.miss_distance_m) : 'None'}</div>` +
        `<div class="st-fact__sub">${closest ? esc(`${closest.secondary.name}, ${relTime(closest.tca)}`) : `nothing within ${SCREEN_THRESHOLD_KM} km`}</div></div>` +
      '</div>';

    html += '<div class="st-cta-row">' +
      (sim ? `<a class="btn btn--primary" href="${sim.href}" title="${esc(sim.title)}">${ICON.play}${esc(sim.label)}</a>` : '') +
      `<a class="btn" href="/fleet.html">${ICON.fleet}Watch in my fleet</a>` +
      `<button class="btn" type="button" id="brief-btn">${ICON.mark}Brief me</button>` +
      '</div>';
    html += '<div id="brief-slot"></div>';

    html += '<div class="st-list-head">' +
      `<h2>${events.length ? `Close approaches (${events.length})` : 'Close approaches'}</h2>` +
      `<span>within ${SCREEN_THRESHOLD_KM} km · riskiest first</span></div>`;
    if (events.length) {
      html += '<ul class="st-list" id="ev-list">' + events.slice(0, EV_SHOW).map((ev) => eventRow(run, ev)).join('') + '</ul>';
      if (events.length > EV_SHOW) {
        html += `<button class="btn btn--ghost st-more" type="button" id="ev-more">Show all ${events.length}</button>`;
      }
    } else {
      html += `<div class="st-empty-card">No close approaches in the next ${esc(win)} hours.</div>`;
    }

    const notes = [];
    notes.push(`Screened with SGP4 on public TLEs against catalog objects with apogee below ${nf(MAX_LEO_APOGEE_KM)} km; approaches under ${SCREEN_THRESHOLD_KM} km are reported.`);
    notes.push('Collision probability is a screening estimate that assumes TLE-class position uncertainty, not operator covariance. Risk levels come from the server (attention at 1 in a million and above).');
    if (hypothetical) notes.push('This is a hypothetical orbit: one arbitrary position in a circular orbit at the altitude and inclination you entered. Your real launch slot will meet different objects.');
    if (st && st.has_demo_data) notes.push('This server also holds labeled demo CDM sequences for trend charts; they do not affect this screen.');
    notes.push(`Results are computed on demand and not stored. ${run.created ? 'Your satellite was added to your fleet' : 'Your satellite was already in your fleet'}, where you can re-screen it any time.`);
    html += `<p class="st-fine">${notes.map(esc).join(' ')}</p>`;
    html += '<div class="st-again"><button class="btn btn--ghost" type="button" id="again-btn">Protect another satellite</button></div>';

    els.result.innerHTML = html;

    $('brief-btn').addEventListener('click', brief);
    $('again-btn').addEventListener('click', restart);
    const more = $('ev-more');
    if (more) {
      more.addEventListener('click', () => {
        $('ev-list').innerHTML = events.map((ev) => eventRow(run, ev)).join('');
        more.remove();
      });
    }
  }

  function restart() {
    state.run = null;
    unpick(false);
    els.search.value = '';
    showStep(1, { back: true });
    setMode(state.mode === 'search' ? 'search' : state.mode);
    setTimeout(() => { if (state.mode === 'search') els.search.focus(); }, 60);
  }

  window.addEventListener('popstate', () => {
    if (state.step !== 1 && !state.busy) {
      state.run = null;
      showStep(1, { back: true });
    }
  });

  /* --- Kessler AI brief ------------------------------------------------------- */

  function inlineMd(escaped) {
    const codes = [];
    let s = escaped.replace(/\u0000/g, '').replace(/`([^`\n]+)`/g, (m, c) => {
      codes.push(c);
      return `\u0000${codes.length - 1}\u0000`;
    });
    s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/\[([^\]\n]+)\]\((\/(?!\/)[^)\s]*)\)/g, '<a href="$2">$1</a>');
    return s.replace(/\u0000(\d+)\u0000/g, (m, i) => `<code>${codes[Number(i)]}</code>`);
  }

  function renderMarkdown(raw) {
    const s = inlineMd(esc(raw));
    const out = [];
    let list = null;
    let para = [];
    const flushPara = () => { if (para.length) { out.push(`<p>${para.join('<br>')}</p>`); para = []; } };
    const flushList = () => { if (list) { out.push(`<${list.tag}>${list.items.join('')}</${list.tag}>`); list = null; } };
    for (const line of s.split('\n')) {
      const t = line.trim();
      const ol = /^(\d+)\. (.*)$/.exec(t);
      const h = /^#{1,4} (.*)$/.exec(t);
      if (/^[-*] /.test(t)) {
        flushPara();
        if (!list || list.tag !== 'ul') { flushList(); list = { tag: 'ul', items: [] }; }
        list.items.push(`<li>${t.slice(2)}</li>`);
      } else if (ol) {
        flushPara();
        if (!list || list.tag !== 'ol') { flushList(); list = { tag: 'ol', items: [] }; }
        list.items.push(`<li>${ol[2]}</li>`);
      } else if (h) { flushPara(); flushList(); out.push(`<div class="st-brief__h">${h[1]}</div>`); }
      else if (/^---+$/.test(t)) { flushPara(); flushList(); out.push('<hr>'); }
      else if (t === '') { flushPara(); flushList(); }
      else { flushList(); para.push(t); }
    }
    flushPara();
    flushList();
    return out.join('');
  }

  function briefPrompt(run) {
    const it = run.item;
    const events = (run.screen.events || []).slice(0, 10).map((e) => ({
      partner: e.secondary.name,
      partner_norad: e.secondary.norad_id,
      partner_type: e.secondary.object_type,
      tca: e.tca,
      miss_distance_m: e.miss_distance_m,
      relative_velocity_ms: e.relative_velocity_ms,
      pc: e.pc,
      risk_level: e.risk_level,
    }));
    const who = run.source === 'plan'
      ? `a HYPOTHETICAL circular orbit (${state.plan ? `${state.plan.alt} km, ${state.plan.inc}°` : it.name}), not a real satellite`
      : `"${it.name}"${it.norad_id != null ? ` (NORAD ${it.norad_id})` : ''}`;
    return [
      `Brief our student satellite team on ${who}. It is user object #${it.id} in my fleet;`,
      `orbit ${orbitText(it) || 'unknown'}${it.epoch ? `, element set epoch ${it.epoch}` : ''}.`,
      `I just screened it against the live catalog for the next ${run.screen.window_hours} hours`,
      `(POST /user-objects/${it.id}/screen at ${run.screen.screened_at}). Those results, riskiest first:`,
      JSON.stringify(events),
      events.length ? '' : '(no close approaches were reported)',
      'In plain language and under 170 words: 1) one sentence on whether anything needs attention;',
      '2) the closest approaches in human terms (when, how close, how fast, how likely);',
      '3) what we should do next, if anything. Use only these numbers or ones you fetch.',
      'Answer in text only; do not navigate or change the interface.',
    ].filter(Boolean).join(' ');
  }

  let briefing = false;
  async function brief() {
    const run = state.run;
    if (!run || briefing) return;
    briefing = true;
    const btn = $('brief-btn');
    btn.disabled = true;
    const slot = $('brief-slot');
    slot.innerHTML =
      '<section class="st-brief" aria-live="polite" aria-busy="true">' +
        `<div class="st-brief__hd"><span class="st-brief__mark">${ICON.mark}</span>` +
        '<span class="st-brief__title">Kessler AI brief</span><span class="st-brief__meta">Reading your screen…</span></div>' +
        '<div class="st-brief__lines"><div class="skeleton" style="width:92%"></div><div class="skeleton" style="width:84%"></div>' +
        '<div class="skeleton" style="width:88%"></div><div class="skeleton" style="width:56%"></div></div>' +
      '</section>';
    slot.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    const card = slot.firstElementChild;
    const meta = card.querySelector('.st-brief__meta');
    const context = { page: 'start', url: '/start.html' };
    if (run.catalogNorad != null) context.selected_norad = run.catalogNorad;
    try {
      const res = await req('POST', '/assistant/chat', {
        messages: [{ role: 'user', content: briefPrompt(run) }], context,
      });
      const body = document.createElement('div');
      body.className = 'st-brief__body';
      body.innerHTML = renderMarkdown(String(res.reply || '').trim() || 'Kessler AI returned an empty answer.');
      card.querySelector('.st-brief__lines').replaceWith(body);
      meta.textContent = res.model ? `Written by ${res.model} from the numbers above` : 'Written from the numbers above';
      if (res.report_url && /^\/(?!\/)/.test(res.report_url)) {
        const a = document.createElement('p');
        a.style.marginTop = '12px';
        a.innerHTML = `<a href="${esc(res.report_url)}" target="_blank" rel="noopener">Open the printable report</a>`;
        body.appendChild(a);
      }
    } catch (err) {
      const off = document.createElement('div');
      off.className = 'st-brief__off';
      if (err.status === 503) {
        meta.textContent = 'Not configured';
        off.innerHTML =
          '<p><strong>Kessler AI isn’t switched on for this server.</strong> It needs an Anthropic API key in the server environment ' +
          '(<code>ANTHROPIC_API_KEY</code>), then a restart.</p>' +
          '<p style="margin-top:8px">Everything above — the screen, the numbers and the verdict — comes straight from Kessler and works without it.</p>';
      } else {
        meta.textContent = 'Unavailable';
        off.innerHTML = `<p><strong>Kessler AI couldn’t answer just now.</strong> ${esc(err.status ? `${err.status}: ${err.detail}` : 'The server did not answer.')}</p>` +
          '<p style="margin-top:10px"><button class="btn btn--xs" type="button" data-act="retry">Try again</button></p>';
        off.querySelector('[data-act="retry"]').addEventListener('click', () => { briefing = false; brief(); });
      }
      card.querySelector('.st-brief__lines').replaceWith(off);
    } finally {
      card.setAttribute('aria-busy', 'false');
      briefing = false;
      btn.disabled = false;
      if (card.querySelector('.st-brief__body')) btn.lastChild.textContent = 'Brief me again';
    }
  }

  /* --- screening animation (canvas; colors from theme tokens) ------------------ */

  const scan = (() => {
    const cv = els.canvas;
    const ctx = cv.getContext('2d');
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    let raf = 0;
    let running = false;
    let colors = null;
    const dots = [];
    let seed = 7;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
    for (let i = 0; i < 90; i++) {
      dots.push({ r: 0.46 + rnd() * 0.46, a: rnd() * Math.PI * 2, tilt: (rnd() - 0.5) * 1.6, w: 0.00012 + rnd() * 0.00022, s: 0.8 + rnd() * 0.9 });
    }
    function readColors() {
      colors = {
        text: KTheme.css('--k-text'), dim: KTheme.css('--k-text-4'), accent: KTheme.css('--k-accent'),
        surface: KTheme.css('--k-surface-3'), border: KTheme.css('--k-border-strong'),
      };
    }
    function size() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const css = 200;
      if (cv.width !== css * dpr) { cv.width = css * dpr; cv.height = css * dpr; }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return css;
    }
    function frame(t) {
      const S = size();
      const c = S / 2;
      const R = S * 0.46;
      ctx.clearRect(0, 0, S, S);
      /* Earth */
      ctx.beginPath();
      ctx.arc(c, c, R * 0.34, 0, Math.PI * 2);
      ctx.fillStyle = colors.surface;
      ctx.fill();
      /* catalog: faint dots on tilted rings */
      ctx.fillStyle = colors.dim;
      for (const d of dots) {
        const a = d.a + t * d.w;
        const x = Math.cos(a) * d.r * R;
        const y = Math.sin(a) * d.r * R * 0.42;
        const ct = Math.cos(d.tilt), stl = Math.sin(d.tilt);
        const px = c + x * ct - y * stl;
        const py = c + x * stl + y * ct;
        const behind = Math.sin(a) < 0 && Math.hypot(px - c, py - c) < R * 0.34;
        if (behind) continue;
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.arc(px, py, d.s, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      /* the user's orbit + sweep */
      const tilt = -0.42;
      const rx = R * 0.72, ry = R * 0.3;
      ctx.save();
      ctx.translate(c, c);
      ctx.rotate(tilt);
      ctx.strokeStyle = colors.accent;
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
      ctx.stroke();
      const a = reduce ? 0.6 : t * 0.0011;
      /* trailing arc */
      ctx.globalAlpha = 1;
      for (let k = 0; k < 24; k++) {
        const aa = a - k * 0.045;
        ctx.globalAlpha = (1 - k / 24) * 0.9;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.ellipse(0, 0, rx, ry, 0, aa - 0.05, aa);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      const sx = Math.cos(a) * rx, sy = Math.sin(a) * ry;
      ctx.fillStyle = colors.accent;
      ctx.beginPath();
      ctx.arc(sx, sy, 4.2, 0, Math.PI * 2);
      ctx.fill();
      /* pulse ring around the satellite */
      const p = reduce ? 0.5 : ((t % 1800) / 1800);
      ctx.globalAlpha = (1 - p) * 0.55;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(sx, sy, 4 + p * 18, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
      ctx.globalAlpha = 1;
      if (running && !reduce) raf = requestAnimationFrame(frame);
    }
    return {
      start() {
        readColors();
        running = true;
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(frame);
      },
      stop() { running = false; cancelAnimationFrame(raf); },
      recolor() { if (colors) { readColors(); if (!running) return; } },
    };
  })();
  KTheme.onChange(() => scan.recolor());

  /* --- global keys ------------------------------------------------------------- */

  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' || state.step !== 1 || e.defaultPrevented) return;
    const t = e.target;
    if (t === els.go || (t && (t.tagName === 'BUTTON' || t.tagName === 'A' || t.tagName === 'TEXTAREA'))) return;
    if (!els.go.disabled) { e.preventDefault(); go(); }
  });

  /* --- boot ---------------------------------------------------------------------- */

  FMT.startUtcClock(document.getElementById('utc-clock'));
  onPlanInput();
  loadStatus().then(async () => {
    /* ?norad=<id> pre-selects a catalog object (deep link from other pages) */
    const q = new URLSearchParams(window.location.search).get('norad');
    if (q && /^\d{1,9}$/.test(q)) {
      try {
        const data = await req('GET', `/objects?search=${q}&limit=1`);
        if (data.items && data.items[0]) pick(data.items[0]);
      } catch (_) { /* no preselect */ }
    }
  });
  if (!matchMedia('(pointer: coarse)').matches) els.search.focus({ preventScroll: true });
})();
