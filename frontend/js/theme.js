/* ===========================================================================
   KESSLER - theme + shell

   Loaded first on every page (before format.js). Owns:
   - the light/dark theme: data-theme on <html>, from localStorage
     'kessler.theme' ('light' | 'dark'), else the system preference. The
     tiny inline boot snippet in each page's <head> applies it before first
     paint; this module keeps it in sync and wires the header toggle.
   - KTheme.css(name): resolved value of a CSS custom property, so canvas
     renderers (globe, charts, simulator) draw with the same tokens as CSS.
   - KTheme.onChange(fn): subscribe to theme flips (canvases re-read colors).
   - the header live pill (#live-pill): polls /status, shows data freshness,
     and never claims "live" when the last refresh failed.
   Canvas colors that depict space (the globe's night sky, the simulator)
   stay dark in both themes by design - only chrome follows the theme.
   ======================================================================== */
(function () {
  'use strict';

  const KEY = 'kessler.theme';
  const root = document.documentElement;
  const listeners = [];
  const mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: light)') : null;

  function stored() {
    try { const v = localStorage.getItem(KEY); return v === 'light' || v === 'dark' ? v : null; }
    catch (_) { return null; }
  }
  function resolved() {
    return stored() || (mq && mq.matches ? 'light' : 'dark');
  }
  function apply(theme) {
    if (root.getAttribute('data-theme') === theme) return;
    root.setAttribute('data-theme', theme);
    cache = Object.create(null);
    for (const fn of listeners) { try { fn(theme); } catch (_) { /* listener fault */ } }
    syncToggle();
  }
  function set(theme) {
    try { localStorage.setItem(KEY, theme); } catch (_) { /* storage blocked */ }
    apply(theme);
  }

  let cache = Object.create(null);
  function css(name, fallback) {
    if (name in cache) return cache[name];
    const v = getComputedStyle(root).getPropertyValue(name).trim();
    cache[name] = v || fallback || '';
    return cache[name];
  }

  const SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.2M12 19.3v2.2M4.6 4.6l1.6 1.6M17.8 17.8l1.6 1.6M2.5 12h2.2M19.3 12h2.2M4.6 19.4l1.6-1.6M17.8 6.2l1.6-1.6"/></svg>';
  const MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20.2 14.6A8.3 8.3 0 0 1 9.4 3.8a8.3 8.3 0 1 0 10.8 10.8z"/></svg>';

  function syncToggle() {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    const light = root.getAttribute('data-theme') === 'light';
    btn.innerHTML = light ? MOON : SUN;
    btn.setAttribute('aria-label', light ? 'Switch to dark appearance' : 'Switch to light appearance');
    btn.title = light ? 'Dark appearance' : 'Light appearance';
  }

  function relAge(iso) {
    const t = Date.parse(iso);
    if (!Number.isFinite(t)) return null;
    const m = Math.max(0, Math.round((Date.now() - t) / 60000));
    if (m < 1) return 'just now';
    if (m < 60) return `${m} min ago`;
    const h = Math.round(m / 60);
    if (h < 48) return `${h} h ago`;
    return `${Math.round(h / 24)} days ago`;
  }

  /* One header status line on every page: "9,747 objects · vigil-0.4".
     theme.js owns #hdr-status (pages no longer write their own variants);
     the observer re-asserts the canonical line if a page script still does. */
  let statusLine = '';
  let statusObs = null;
  function syncHdrStatus(st) {
    const el = document.getElementById('hdr-status');
    if (!el) return;
    const parts = [];
    const n = Number(st.catalog_size);
    if (Number.isFinite(n) && n > 0) parts.push(`${n.toLocaleString('en-US')} objects`);
    if (st.model_version) parts.push(String(st.model_version));
    statusLine = parts.join(' · ');
    if (el.textContent !== statusLine) el.textContent = statusLine;
    if (!statusObs && window.MutationObserver) {
      statusObs = new MutationObserver(() => {
        if (statusLine && el.textContent !== statusLine) el.textContent = statusLine;
      });
      statusObs.observe(el, { childList: true, characterData: true, subtree: true });
    }
  }

  /* Space weather in the header of every page: the live storm level from
     /status (app.spaceweather.summary), linking to the Overview's
     space-weather view. Absent data removes the pill - never a stale guess. */
  const SW_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="2.9" fill="currentColor"/>' +
    '<path d="M8 1.2v1.9M8 12.9v1.9M1.2 8h1.9M12.9 8h1.9M3.2 3.2l1.35 1.35M11.45 11.45l1.35 1.35M3.2 12.8l1.35-1.35M11.45 4.55l1.35-1.35" ' +
    'stroke="currentColor" stroke-width="1.35" stroke-linecap="round"/></svg>';
  function syncSwPill(sw) {
    const right = document.querySelector('.topbar .topbar__right');
    let pill = document.getElementById('sw-pill');
    if (!sw || !right) { if (pill) pill.remove(); return; }
    if (!pill) {
      pill = document.createElement('a');
      pill.id = 'sw-pill';
      pill.className = 'sw-pill';
      pill.href = '/?view=spaceweather';
      const live = document.getElementById('live-pill');
      right.insertBefore(pill, live || right.firstChild);
    }
    const g = Number(sw.g) || 0;
    const level = g >= 3 ? 'is-severe' : g >= 1 ? 'is-storm' : (sw.level === 'active' || sw.r >= 1) ? 'is-active' : 'is-quiet';
    const text = g >= 1 ? `G${g} storm` : sw.r >= 1 ? `R${sw.r} blackout` : sw.level === 'active' ? 'Sun active' : 'Sun quiet';
    pill.className = `sw-pill ${level}${sw.stale ? ' is-stale' : ''}`;
    pill.innerHTML = `${SW_ICON}<span>${text}</span>`;
    const bits = [sw.label];
    if (Number.isFinite(sw.kp)) bits.push(`Kp ${Number(sw.kp).toFixed(1)}`);
    const r = sw.density_ratio && sw.density_ratio['550'];
    if (Number.isFinite(r)) bits.push(`air at 550 km ×${r.toFixed(1)}`);
    if (sw.flare_6h_peak) bits.push(`flare peak ${sw.flare_6h_peak} (6 h)`);
    if (Number.isFinite(sw.widened_conjunctions) && sw.widened_conjunctions > 0) {
      bits.push(`${sw.widened_conjunctions} close approach${sw.widened_conjunctions === 1 ? '' : 'es'} widened`);
    }
    pill.title = `Space weather: ${bits.join(' · ')}${sw.stale ? ' (delayed)' : ''}`;
    pill.setAttribute('aria-label', pill.title);
  }

  async function pollLive() {
    const pill = document.getElementById('live-pill');
    try {
      const res = await fetch('/status', { headers: { Accept: 'application/json' } });
      if (!res.ok) throw new Error(String(res.status));
      const st = await res.json();
      syncHdrStatus(st);
      syncSwPill(st.space_weather);
      if (!pill) return;
      const text = pill.querySelector('.live-pill__text');
      pill.classList.remove('is-stale', 'is-down');
      let label;
      if (st.snapshot && st.snapshot.path) {
        pill.classList.add('is-stale');
        label = st.last_ingest_at ? `Offline snapshot · ${relAge(st.last_ingest_at)}` : 'Offline snapshot';
        pill.title = `Serving a saved snapshot (${st.snapshot.path}); data as of ${st.last_ingest_at || 'unknown'}`;
      } else if (st.last_run_error) {
        pill.classList.add('is-down');
        label = 'Refresh failed';
        pill.title = st.last_run_error;
      } else if (!st.last_ingest_at) {
        pill.classList.add('is-stale');
        label = 'No data yet';
        pill.title = 'No catalog ingested yet';
      } else {
        const ageMin = (Date.now() - Date.parse(st.last_ingest_at)) / 60000;
        if (ageMin > 360) pill.classList.add('is-stale');
        label = `Updated ${relAge(st.last_ingest_at)}`;
        pill.title = `Catalog ingested ${st.last_ingest_at}`;
      }
      if (text) text.textContent = label;
    } catch (_) {
      if (!pill) return;
      const text = pill.querySelector('.live-pill__text');
      pill.classList.remove('is-stale');
      pill.classList.add('is-down');
      if (text) text.textContent = 'Offline';
      pill.title = 'The Kessler server is not answering';
    }
  }

  /* phones: the primary nav collapses into a menu button + sheet built
     from the page's own .topnav links (no per-page markup to keep in sync) */
  function initCompactNav() {
    const nav = document.querySelector('.topbar .topnav');
    const right = document.querySelector('.topbar .topbar__right');
    if (!nav || !right || document.getElementById('nav-toggle')) return;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.id = 'nav-toggle';
    btn.className = 'icon-btn nav-toggle';
    btn.setAttribute('aria-label', 'Menu');
    btn.setAttribute('aria-expanded', 'false');
    btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>';
    right.insertBefore(btn, right.firstChild);
    let sheet = null;
    function close() {
      if (!sheet) return;
      sheet.remove(); sheet = null;
      btn.setAttribute('aria-expanded', 'false');
      document.removeEventListener('click', outside, true);
      document.removeEventListener('keydown', onKey);
    }
    function outside(ev) { if (sheet && !sheet.contains(ev.target) && ev.target !== btn && !btn.contains(ev.target)) close(); }
    function onKey(ev) { if (ev.key === 'Escape') { close(); btn.focus(); } }
    btn.addEventListener('click', () => {
      if (sheet) { close(); return; }
      sheet = document.createElement('nav');
      sheet.className = 'nav-sheet';
      sheet.setAttribute('aria-label', 'Primary');
      sheet.innerHTML = nav.innerHTML + '<hr><a href="/wall.html">Ops wall</a><a href="/start.html">Protect a satellite</a>';
      document.body.appendChild(sheet);
      btn.setAttribute('aria-expanded', 'true');
      const first = sheet.querySelector('a');
      if (first) first.focus();
      document.addEventListener('click', outside, true);
      document.addEventListener('keydown', onKey);
    });
  }

  function init() {
    apply(resolved());
    initCompactNav();
    const btn = document.getElementById('theme-toggle');
    if (btn) {
      syncToggle();
      btn.addEventListener('click', () => {
        set(root.getAttribute('data-theme') === 'light' ? 'dark' : 'light');
      });
    }
    if (mq && mq.addEventListener) {
      mq.addEventListener('change', () => { if (!stored()) apply(resolved()); });
    }
    pollLive();
    setInterval(pollLive, 60000);
  }

  window.KTheme = {
    css,
    get: () => root.getAttribute('data-theme') || 'dark',
    set,
    onChange(fn) { if (typeof fn === 'function') listeners.push(fn); },
    isLight: () => root.getAttribute('data-theme') === 'light',
  };

  apply(resolved());
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
