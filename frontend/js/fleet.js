/* Kessler - my fleet page.
   User-owned TLEs via /user-objects:
   - "Add a satellite" sheet (server-validated TLE; a 422 detail is shown
     in plain words inside the sheet). A name line above the two TLE lines
     (three-line format) is used as the name when the name field is blank.
   - Fleet list: one row per satellite with its orbit summary, a status
     line (last screen summary -> result, else "Not screened yet"), and
     actions: Screen, 14-day scan, Delete (two-step confirm, 3 s arm).
   - Screen: POST /user-objects/{id}/screen (72 h). Results render below
     with server-computed risk chips; nothing is stored server-side.
   - 14-day scan: POST /user-objects/{id}/window-scan (~90 s coarse pass)
     as a day strip shaded by event count, scaled to this scan's own
     busiest/quietest days (the key names the range), accent tints.
   Last-screen summaries persist per user object in localStorage
   'kessler.screens' (also written by /start.html) - a per-browser
   convenience; the server stores nothing about screens.
   User objects never touch the real event tables. js/api.js has no
   /user-objects wrappers - the local request() mirrors its error shape
   plus JSON bodies and 204 handling. */
'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
  const els = {
    hdrStatus: $('hdr-status'),
    errorBanner: $('error-banner'),
    demoFlag: $('demo-flag'),
    slAuto: $('sl-auto'),
    addOpen: $('add-open'),
    addSheet: $('add-sheet'),
    addScrim: $('add-scrim'),
    addClose: $('add-close'),
    addCancel: $('add-cancel'),
    addErr: $('add-err'),
    addForm: $('add-form'),
    addName: $('add-name'),
    addTle: $('add-tle'),
    addNote: $('add-note'),
    addBtn: $('add-btn'),
    fleetMeta: $('fleet-meta'),
    fleetBody: $('fleet-body'),
    screenPanel: $('screen-panel'),
    screenName: $('screen-name'),
    screenMeta: $('screen-meta'),
    screenSummary: $('screen-summary'),
    screenBody: $('screen-body'),
    screenMore: $('screen-more'),
    wscanPanel: $('wscan-panel'),
    wscanName: $('wscan-name'),
    wscanMeta: $('wscan-meta'),
    wscanRow: $('wscan-row'),
  };

  const KNOWN_LEVELS = { critical: true, elevated: true, nominal: true };
  const LEVEL_LABELS = { critical: 'Critical', elevated: 'Elevated', nominal: 'Low' };
  const TYPE_LABELS = { PAYLOAD: 'Payload', DEBRIS: 'Debris', 'ROCKET BODY': 'Rocket body' };
  const WSCAN_DAYS = 14;
  const COARSE = !!(window.matchMedia && window.matchMedia('(pointer: coarse)').matches);

  let items = [];
  let screening = false;
  let screenedId = null;
  let scanning = false;
  let scannedId = null;
  let armTimer = null;
  let armedId = null;
  /* id -> {count, worst, critical, elevated, hours, at}: last screen summary,
     persisted per browser (start.js writes the same key after its screen) */
  const SCREENS_KEY = 'kessler.screens';
  const results = new Map();
  try {
    const saved = JSON.parse(localStorage.getItem(SCREENS_KEY) || '{}');
    for (const [k, v] of Object.entries(saved || {})) {
      if (v && typeof v === 'object' && Number.isFinite(Number(v.count))) results.set(Number(k), v);
    }
  } catch (_) { /* storage blocked or corrupt - session-only */ }
  function saveResults() {
    try {
      const o = {};
      for (const [k, v] of results) o[k] = v;
      localStorage.setItem(SCREENS_KEY, JSON.stringify(o));
    } catch (_) { /* storage blocked */ }
  }

  FMT.startUtcClock($('utc-clock'));

  function safeLevel(level) { return KNOWN_LEVELS[level] ? level : 'nominal'; }
  function typeLabel(t) {
    const k = String(t || '').toUpperCase();
    return TYPE_LABELS[k] || (k ? k.charAt(0) + k.slice(1).toLowerCase() : 'Object');
  }

  /* --- local API client ------------------------------------------------------ */

  function flattenDetail(detail) {
    if (Array.isArray(detail)) {
      return detail.map((d) => (d && d.msg) ? d.msg : JSON.stringify(d)).join(' · ');
    }
    return String(detail);
  }

  async function request(method, path, params, body) {
    const url = new URL(path, window.location.origin);
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
      }
    }
    const res = await fetch(url, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const payload = await res.json();
        if (payload && payload.detail != null) detail = flattenDetail(payload.detail);
      } catch (_) { /* keep statusText */ }
      const err = new Error(`${res.status}: ${detail}`);
      err.status = res.status;
      err.detail = detail;
      throw err;
    }
    if (res.status === 204) return null;
    return res.json();
  }

  /* --- errors ----------------------------------------------------------------- */

  function showError(msg, detail) {
    els.errorBanner.textContent = msg;
    els.errorBanner.title = detail || '';
    els.errorBanner.hidden = false;
  }
  function clearError() {
    els.errorBanner.hidden = true;
    els.errorBanner.textContent = '';
  }

  /* --- header status ------------------------------------------------------------ */

  API.status().then((s) => {
    /* #hdr-status is filled by theme.js (one format on every page) */
    els.demoFlag.hidden = !s.has_demo_data;
    const autoH = Number(s.auto_refresh_interval_h);
    if (Number.isFinite(autoH) && autoH > 0) {
      els.slAuto.textContent = `Refreshes every ${autoH} h`;
      els.slAuto.hidden = false;
    }
  }).catch(() => { /* header status unavailable */ });

  /* --- fleet list ------------------------------------------------------------------ */

  function orbitLine(it) {
    const parts = [];
    if (it.apogee_km != null && it.perigee_km != null) {
      const lo = Math.round(it.perigee_km);
      const hi = Math.round(it.apogee_km);
      parts.push(hi - lo < 15 ? `${FMT.num(Math.round((lo + hi) / 2))} km up` : `${FMT.num(lo)}–${FMT.num(hi)} km up`);
    }
    if (it.inclination_deg != null) parts.push(`${Number(it.inclination_deg).toFixed(1)}° inclination`);
    if (it.period_min != null) parts.push(`${Number(it.period_min).toFixed(1)} min orbit`);
    return parts.length ? parts.join(' · ') : 'Orbit unavailable';
  }

  const LEO_SCREEN_MAX_KM = 3000; /* config.SCREENING_MAX_LEO_ALT_KM */

  function statusHtml(it) {
    const r = results.get(it.id);
    if (!r) return '<span class="fl-status"><span class="chip chip--nominal">Not screened yet</span></span>';
    let chip;
    const aboveLeo = it.apogee_km != null && it.apogee_km > LEO_SCREEN_MAX_KM;
    if (r.count === 0 && aboveLeo) chip = '<span class="chip chip--nominal">Outside screened band</span>';
    else if (r.count === 0) chip = '<span class="chip chip--ok">All clear</span>';
    else if (r.worst === 'critical') chip = `<span class="chip chip--critical">${FMT.num(r.critical)} critical</span>`;
    else if (r.worst === 'elevated') chip = `<span class="chip chip--elevated">${FMT.num(r.elevated)} elevated</span>`;
    else chip = `<span class="chip chip--nominal">${FMT.num(r.count)} low-risk</span>`;
    const what = r.count === 0 && aboveLeo
      ? `Kessler screens below ${FMT.num(LEO_SCREEN_MAX_KM)} km; this orbit reaches ${FMT.num(Math.round(it.apogee_km))} km`
      : r.count === 0
      ? `No close approaches in ${r.hours} h`
      : `${FMT.num(r.count)} close approach${r.count === 1 ? '' : 'es'} in ${r.hours} h`;
    return `<span class="fl-status">${chip}<span class="fl-status__txt">${what} · screened <span data-rel="${FMT.esc(r.at)}">${FMT.esc(FMT.rel(r.at))}</span></span></span>`;
  }

  function fleetRow(it) {
    const name = FMT.esc(it.name);
    const norad = it.norad_id != null ? String(it.norad_id) : null;
    return `
      <li class="fl" data-id="${FMT.esc(String(it.id))}">
        <div class="fl-main">
          <div class="fl-name">${name}</div>
          <div class="fl-meta">
            ${norad ? `<span>NORAD <span class="code">${FMT.esc(norad)}</span></span>` : ''}
            <span>${FMT.esc(orbitLine(it))}</span>
            <span title="${FMT.esc(FMT.utc(it.added_at))}">Added ${FMT.esc(FMT.rel(it.added_at))}</span>
          </div>
          ${it.note ? `<div class="fl-note">${FMT.esc(it.note)}</div>` : ''}
          ${statusHtml(it)}
        </div>
        <div class="fl-actions">
          <button class="btn btn--primary btn--xs js-screen" type="button"
                  aria-label="Screen ${name} against the catalog for the next 72 hours">Screen 72 h</button>
          <button class="btn btn--xs js-wscan" type="button"
                  title="A coarse scan of the next ${WSCAN_DAYS} days. Takes about 90 seconds."
                  aria-label="Scan the next ${WSCAN_DAYS} days for ${name} (takes about 90 seconds)">Scan 14 days</button>
          <button class="btn btn--xs btn--ghost fl-del js-del" type="button"
                  aria-label="Remove ${name} from the fleet">Remove</button>
        </div>
      </li>`;
  }

  function renderFleet() {
    disarmDelete();
    els.fleetMeta.textContent = items.length
      ? `${FMT.num(items.length)} satellite${items.length === 1 ? '' : 's'}` : '';
    if (!items.length) {
      els.fleetBody.innerHTML = `
        <li class="fleet-empty">
          <div class="fleet-empty__art" aria-hidden="true"><svg viewBox="0 0 64 64" width="64" height="64" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="24" y="24" width="16" height="16" rx="3"/><path d="M24 32H8M40 32h16M11 26v12M53 26v12M32 24v-6M29 18h6"/></svg></div>
          <strong>No satellites yet</strong>
          <p>Add your satellite’s two-line elements and Kessler will check it for close approaches with anything in the catalog.</p>
          <button class="btn btn--primary btn--lg js-add" type="button">Add your first satellite</button>
        </li>`;
      return;
    }
    els.fleetBody.innerHTML = items.map(fleetRow).join('');
  }

  function rerenderRow(id) {
    const li = els.fleetBody.querySelector(`.fl[data-id="${CSS.escape(String(id))}"]`);
    const it = items.find((x) => x.id === id);
    if (!li || !it) return;
    const busy = li.querySelectorAll('button[disabled]');
    if (busy.length) return; /* don't clobber a running job's button state */
    const tmp = document.createElement('ul');
    tmp.innerHTML = fleetRow(it).trim();
    li.replaceWith(tmp.firstElementChild);
  }

  async function loadFleet() {
    let data;
    try {
      data = await request('GET', '/user-objects');
    } catch (err) {
      showError(err.status === 404
        ? 'The fleet needs a newer server build.'
        : 'We couldn’t load your fleet. Check your connection and reload.', err.message);
      els.fleetMeta.textContent = '';
      els.fleetBody.innerHTML = '<li class="empty">Your fleet isn’t available right now.</li>';
      return;
    }
    clearError();
    items = Array.isArray(data.items) ? data.items : [];
    /* forget summaries of satellites no longer in the fleet */
    const live = new Set(items.map((it) => Number(it.id)));
    let pruned = false;
    for (const k of Array.from(results.keys())) if (!live.has(k)) { results.delete(k); pruned = true; }
    if (pruned) saveResults();
    renderFleet();
  }

  /* --- add sheet ----------------------------------------------------------------- */

  let lastFocus = null;

  function openSheet() {
    lastFocus = document.activeElement;
    els.addErr.hidden = true;
    els.addScrim.hidden = false;
    els.addSheet.hidden = false;
    document.body.style.overflow = 'hidden';
    setTimeout(() => els.addTle.focus(), 30);
  }

  function closeSheet() {
    if (els.addSheet.hidden) return;
    els.addScrim.hidden = true;
    els.addSheet.hidden = true;
    document.body.style.overflow = '';
    if (lastFocus && lastFocus.isConnected) lastFocus.focus();
  }

  function showAddError(msg) {
    els.addErr.textContent = msg;
    els.addErr.hidden = false;
  }

  /* server 422 details name the failing line - keep them, drop the jargon */
  function humanTleError(detail) {
    const s = String(detail || '').trim();
    if (!s) return 'That TLE didn’t pass validation. Check that both lines are complete.';
    const lead = /^[A-Z][a-z]/.test(s) ? s.charAt(0).toLowerCase() + s.slice(1) : s;
    return `That TLE didn’t pass validation: ${lead}${/[.!]$/.test(s) ? '' : '.'}`;
  }

  async function onAdd(ev) {
    ev.preventDefault();
    let lines = els.addTle.value.split('\n').map((l) => l.trim()).filter((l) => l.length > 0);
    let nameFromTle = '';
    if (lines.length === 3 && !/^1 /.test(lines[0]) && /^1 /.test(lines[1])) {
      nameFromTle = lines[0].replace(/^0 /, '').slice(0, 64);
      lines = lines.slice(1);
    }
    if (lines.length !== 2) {
      showAddError(lines.length < 2
        ? 'Paste both lines of the TLE — line 1 and line 2.'
        : 'That looks like more than one TLE. Paste just the two lines for one satellite.');
      els.addTle.focus();
      return;
    }
    const body = { line1: lines[0], line2: lines[1] };
    const name = els.addName.value.trim() || nameFromTle;
    const note = els.addNote.value.trim();
    if (name) body.name = name;
    if (note) body.note = note;
    els.addBtn.disabled = true;
    els.addBtn.textContent = 'Adding…';
    try {
      await request('POST', '/user-objects', null, body);
      els.addErr.hidden = true;
      els.addForm.reset();
      closeSheet();
      await loadFleet();
    } catch (err) {
      if (err.status === 422) showAddError(humanTleError(err.detail));
      else if (err.status === 413) showAddError('Your fleet is full. Remove a satellite to add another.');
      else showAddError('We couldn’t add it right now. Try again in a moment.');
    } finally {
      els.addBtn.disabled = false;
      els.addBtn.textContent = 'Add to fleet';
    }
  }

  /* --- screening ----------------------------------------------------------------- */

  function screenRow(ev) {
    const lvl = safeLevel(ev.risk_level);
    const sec = ev.secondary || {};
    const secHtml = sec.norad_id != null
      ? `<a href="object.html?id=${encodeURIComponent(String(sec.norad_id))}">${FMT.esc(sec.name || '—')}</a>` +
        `<span class="td-sub">${FMT.esc(typeLabel(sec.object_type))} · <span class="code">${FMT.esc(String(sec.norad_id))}</span></span>`
      : '<span class="faint">—</span>';
    return `
      <tr class="${lvl === 'critical' ? 'row--critical' : lvl === 'elevated' ? 'row--elevated' : ''}">
        <td><span class="chip chip--${lvl}">${LEVEL_LABELS[lvl]}</span></td>
        <td class="td-obj">${secHtml}</td>
        <td>${FMT.esc(FMT.when(ev.tca))}<span class="td-sub" data-rel="${FMT.esc(ev.tca)}">${FMT.esc(FMT.rel(ev.tca))}</span></td>
        <td class="num td-big">${FMT.esc(FMT.dist(ev.miss_distance_m))}</td>
        <td class="num col-speed">${FMT.esc(FMT.vel(ev.relative_velocity_ms))}</td>
        <td class="num" title="about ${FMT.esc(FMT.pcOdds(ev.pc))}">${FMT.esc(FMT.pc(ev.pc))}</td>
      </tr>`;
  }

  function renderScreenResults(item, resp) {
    const events = Array.isArray(resp.events) ? resp.events : [];
    const hours = resp.window_hours != null ? Math.round(resp.window_hours) : 72;
    screenedId = item.id;
    const counts = { critical: 0, elevated: 0, nominal: 0 };
    for (const e of events) counts[safeLevel(e.risk_level)] += 1;
    const worst = counts.critical ? 'critical' : counts.elevated ? 'elevated' : 'nominal';
    results.set(item.id, {
      count: events.length, worst, critical: counts.critical, elevated: counts.elevated,
      hours, at: resp.screened_at || new Date().toISOString(),
    });
    saveResults();
    rerenderRow(item.id);
    els.screenName.textContent = item.name;
    els.screenMeta.textContent = `Next ${hours} h · screened ${FMT.rel(resp.screened_at || new Date().toISOString())}`;
    const s = (n, w) => `<div class="ss"><div class="ss__v${n && w !== 'nominal' ? ` is-${w}` : ''}">${FMT.num(n)}</div><div class="ss__k">${LEVEL_LABELS[w]}</div></div>`;
    els.screenSummary.innerHTML = events.length
      ? s(counts.critical, 'critical') + s(counts.elevated, 'elevated') + s(counts.nominal, 'nominal') +
        `<p class="ss__txt">${FMT.num(events.length)} close approach${events.length === 1 ? '' : 'es'} with catalog objects, highest probability first.</p>`
      : (item.apogee_km != null && item.apogee_km > LEO_SCREEN_MAX_KM)
        ? `<div class="empty"><strong>Outside the screened band</strong>Kessler screens against objects below ${FMT.num(LEO_SCREEN_MAX_KM)} km; this orbit reaches ${FMT.num(Math.round(item.apogee_km))} km, so an empty result is not an all-clear.</div>`
        : `<div class="empty"><strong>All clear</strong>No close approaches with any catalog object in the next ${hours} hours.</div>`;
    let shown = Math.min(events.length, 10);
    const drawRows = () => {
      els.screenBody.innerHTML = events.slice(0, shown).map(screenRow).join('');
      const left = events.length - shown;
      els.screenMore.hidden = left <= 0;
      els.screenMore.textContent = `Show all ${FMT.num(events.length)} · ${FMT.num(left)} more`;
    };
    els.screenMore.onclick = () => { shown = events.length; drawRows(); };
    drawRows();
    els.screenBody.closest('.tbl-scroll').hidden = !events.length;
    els.screenPanel.hidden = false;
    els.screenPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  async function onScreen(id, btn) {
    if (screening || scanning) return; /* one heavy SGP4 job at a time */
    const item = items.find((it) => it.id === id);
    if (!item) return;
    screening = true;
    btn.disabled = true;
    btn.textContent = 'Screening…';
    if (!els.screenPanel.hidden) els.screenPanel.classList.add('is-refetching');
    try {
      const resp = await request('POST',
        `/user-objects/${encodeURIComponent(String(id))}/screen`, { window_hours: 72 });
      clearError();
      btn.disabled = false;
      renderScreenResults(item, resp);
    } catch (err) {
      showError('Screening didn’t finish. Try again in a moment.', err.message);
    } finally {
      screening = false;
      els.screenPanel.classList.remove('is-refetching');
      if (btn.isConnected) {
        btn.disabled = false;
        btn.textContent = 'Screen 72 h';
      }
    }
  }

  /* --- 14-day scan ----------------------------------------------------------------- */

  /* shade 0 (no events) ... 5, scaled between this scan's quietest and
     busiest non-empty days so a uniformly busy fortnight still shows its
     relative peaks; wscanRange is set per render */
  let wscanRange = { lo: 0, hi: 0 };
  function wscanBin(count) {
    if (!(count > 0)) return 0;
    const { lo, hi } = wscanRange;
    if (hi <= lo) return 3;
    return 1 + Math.round((4 * (count - lo)) / (hi - lo));
  }

  const DOW = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  function wscanCell(day) {
    const count = Number(day.events) || 0;
    const missKm = Number(day.min_miss_km);
    const miss = Number.isFinite(missKm) ? FMT.dist(missKm * 1000) : '—';
    const maxPc = day.max_pc != null ? FMT.pc(day.max_pc) : '—';
    const date = String(day.date || '');
    const d = new Date(`${date}T00:00:00Z`);
    const valid = !Number.isNaN(d.getTime());
    const title = `${date} · ${FMT.num(count)} close approach${count === 1 ? '' : 'es'}` +
      (count ? ` · closest ${miss} · highest probability ${maxPc}` : '');
    const step = wscanBin(count);
    return (
      `<div class="wscan-day" title="${FMT.esc(title)}">` +
      `<div class="wscan-date">${valid ? `${DOW[d.getUTCDay()]} <b>${d.getUTCDate()}</b>` : FMT.esc(date)}</div>` +
      `<div class="wscan-cell${step === 0 ? ' is-zero' : ''}${step >= 4 ? ' is-strong' : ''}" style="--s: ${step}">` +
      `<span class="wscan-cell__count">${FMT.num(count)}</span>` +
      `<span class="wscan-cell__sub">${count > 0 ? FMT.esc(miss) : 'quiet'}</span>` +
      `</div>` +
      `</div>`
    );
  }

  function renderWindowScan(item, resp) {
    const days = Array.isArray(resp.days) ? resp.days : [];
    scannedId = item.id;
    els.wscanName.textContent = item.name;
    const dur = Number(resp.duration_s);
    els.wscanMeta.textContent = Number.isFinite(dur) ? `Scan took ${Math.round(dur)} s` : '';
    const shown = days.slice(0, WSCAN_DAYS);
    const counts = shown.map((d) => Number(d.events) || 0).filter((n) => n > 0);
    wscanRange = counts.length
      ? { lo: Math.min(...counts), hi: Math.max(...counts) } : { lo: 0, hi: 0 };
    const key = document.querySelector('.wscan-key');
    if (key && key.firstElementChild && key.lastElementChild) {
      const { lo, hi } = wscanRange;
      key.firstElementChild.textContent = counts.length ? `Fewest (${FMT.num(lo)})` : 'Quiet';
      key.lastElementChild.textContent = counts.length ? `Most (${FMT.num(hi)})` : 'Busy';
    }
    els.wscanRow.innerHTML = days.length
      ? shown.map(wscanCell).join('')
      : '<div class="empty">The scan came back empty.</div>';
    els.wscanPanel.hidden = false;
    els.wscanPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  async function onWindowScan(id, btn) {
    if (screening || scanning) return;
    const item = items.find((it) => it.id === id);
    if (!item) return;
    scanning = true;
    btn.disabled = true;
    btn.textContent = 'Scanning… about 90 s';
    if (!els.wscanPanel.hidden) els.wscanPanel.classList.add('is-refetching');
    try {
      const resp = await request('POST',
        `/user-objects/${encodeURIComponent(String(id))}/window-scan`, null, { days: WSCAN_DAYS });
      clearError();
      renderWindowScan(item, resp);
    } catch (err) {
      showError(err.status === 404 || err.status === 405
        ? 'The 14-day scan needs a newer server build.'
        : 'The 14-day scan didn’t finish. Try again in a moment.', err.message);
    } finally {
      scanning = false;
      els.wscanPanel.classList.remove('is-refetching');
      if (btn.isConnected) {
        btn.disabled = false;
        btn.textContent = 'Scan 14 days';
      }
    }
  }

  /* --- two-step remove (confirm state, 3 s arm timeout) ------------------------------ */

  function disarmDelete() {
    clearTimeout(armTimer);
    armTimer = null;
    armedId = null;
    for (const b of els.fleetBody.querySelectorAll('.js-del.is-arm')) {
      b.classList.remove('is-arm');
      b.textContent = 'Remove';
    }
  }

  async function onDelete(id, btn) {
    if (armedId !== id) {
      disarmDelete();
      armedId = id;
      btn.classList.add('is-arm');
      btn.textContent = COARSE ? 'Tap again to remove' : 'Click again to remove';
      armTimer = setTimeout(disarmDelete, 3000);
      return;
    }
    disarmDelete();
    btn.disabled = true;
    try {
      await request('DELETE', `/user-objects/${encodeURIComponent(String(id))}`);
      clearError();
      results.delete(id);
      saveResults();
      if (screenedId === id) { els.screenPanel.hidden = true; screenedId = null; }
      if (scannedId === id) { els.wscanPanel.hidden = true; scannedId = null; }
      await loadFleet();
    } catch (err) {
      showError('We couldn’t remove it. Try again in a moment.', err.message);
      if (btn.isConnected) btn.disabled = false;
    }
  }

  /* --- events / init ------------------------------------------------------------------- */

  els.addForm.addEventListener('submit', onAdd);
  els.addOpen.addEventListener('click', openSheet);
  els.addClose.addEventListener('click', closeSheet);
  els.addCancel.addEventListener('click', closeSheet);
  els.addScrim.addEventListener('click', closeSheet);
  document.addEventListener('keydown', (e) => {
    if (els.addSheet.hidden) return;
    if (e.key === 'Escape') { e.preventDefault(); closeSheet(); return; }
    if (e.key === 'Tab') { /* keep focus inside the sheet */
      const f = els.addSheet.querySelectorAll('button:not([disabled]), input, textarea');
      if (!f.length) return;
      const first = f[0];
      const last = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  });

  els.fleetBody.addEventListener('click', (ev) => {
    if (ev.target.closest('.js-add')) { openSheet(); return; }
    const li = ev.target.closest('[data-id]');
    if (!li) return;
    const id = Number(li.dataset.id);
    if (!Number.isFinite(id)) return;
    const screenBtn = ev.target.closest('.js-screen');
    if (screenBtn) { onScreen(id, screenBtn); return; }
    const wscanBtn = ev.target.closest('.js-wscan');
    if (wscanBtn) { onWindowScan(id, wscanBtn); return; }
    const delBtn = ev.target.closest('.js-del');
    if (delBtn) onDelete(id, delBtn);
  });

  setInterval(() => {
    document.querySelectorAll('[data-rel]').forEach((el) => { el.textContent = FMT.rel(el.dataset.rel); });
  }, 15000);

  loadFleet();
})();
