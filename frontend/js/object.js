/* Kessler object detail page. ?id=<norad>[&conj=<id>]
   Hero (name, type, ids, element age, actions) + headline stats, ground
   track, orbit plot, altitude profile, orbital elements, element (TLE)
   history, close approaches + probability chart, and ground-station passes.
   Countdowns tick locally; only the ground track/altitude data refetches
   (every 60 s).
   - Unknown object / load failure: a friendly card with catalog search and
     a way back - never a red error bar.
   - Close-approach rows carry Simulate (KSim.open, typeof-guarded).
   - Passes: ground site persisted in localStorage 'kessler.site' as
     {lat, lon, alt_m}; set from the browser's geolocation or typed
     coordinates; /objects/{norad}/passes (48 h, min el 10°) refetches
     whenever the site changes. Pass rows expand to a polar sky view
     (skyview.js) from the pass's `track` samples; backends without `track`
     degrade to a note.
   - Statusline auto-refresh flag from /status.auto_refresh_interval_h. */
'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(window.location.search);
  const noradParam = (params.get('id') || '').trim();
  const conjParam = params.get('conj');

  FMT.startUtcClock($('utc-clock'));

  const TYPE_LABELS = {
    PAYLOAD: 'Payload',
    DEBRIS: 'Debris',
    'ROCKET BODY': 'Rocket body',
    'R/B': 'Rocket body',
    UNKNOWN: 'Unknown type',
  };
  const LEVEL_LABELS = { critical: 'Critical', elevated: 'Elevated', nominal: 'Low' };

  function typeLabel(t) {
    const k = String(t || 'UNKNOWN').toUpperCase();
    if (TYPE_LABELS[k]) return TYPE_LABELS[k];
    const s = String(t).toLowerCase();
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  function safeLevel(l) {
    return l === 'critical' || l === 'elevated' ? l : 'nominal';
  }

  function partnerOf(conj, noradId) {
    return Number(conj.primary.norad_id) === Number(noradId) ? conj.secondary : conj.primary;
  }

  function fmtLat(v, d = 2) { return `${Math.abs(v).toFixed(d)}° ${v < 0 ? 'S' : 'N'}`; }
  function fmtLon(v, d = 2) { return `${Math.abs(v).toFixed(d)}° ${v < 0 ? 'W' : 'E'}`; }

  /* mirrors API.request's error shape for endpoints js/api.js has no
     wrapper for (track, passes) */
  async function getJson(url) {
    const res = await fetch(url);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) { /* keep statusText */ }
      const err = new Error(`${res.status}: ${detail}`);
      err.status = res.status;
      throw err;
    }
    return res.json();
  }

  /* --- not found / failure card ----------------------------------------- */

  function fail(kind) {
    $('obj-loading').hidden = true;
    $('content').hidden = true;
    const card = $('err');
    const idText = noradParam ? FMT.esc(noradParam) : '';
    if (kind === 'missing-param') {
      $('err-title').textContent = 'Which object would you like to see?';
      $('err-text').textContent = 'Search the catalog by name or NORAD id.';
    } else if (kind === 'not-found') {
      $('err-title').textContent = 'We couldn’t find that object';
      $('err-text').innerHTML =
        `There’s no object with NORAD id <span class="code">${idText}</span> in the catalog. ` +
        'It may have re-entered, or the number may be mistyped.';
    } else {
      $('err-title').textContent = 'Kessler can’t reach the server right now';
      $('err-text').textContent =
        'Your connection or the Kessler server may be down. Try again in a moment.';
      $('err-retry').hidden = false;
    }
    card.hidden = false;
    document.title = 'Kessler · Object not found';
    if (kind !== 'unreachable') {
      const q = $('find-q');
      if (noradParam && kind === 'not-found') q.value = '';
      q.focus({ preventScroll: true });
    }
  }

  function initFind() {
    const form = $('find-form');
    const out = $('find-results');
    $('err-retry').addEventListener('click', () => window.location.reload());
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const q = $('find-q').value.trim();
      if (!q) return;
      if (/^\d{1,9}$/.test(q)) {
        window.location.href = `object.html?id=${encodeURIComponent(q)}`;
        return;
      }
      out.innerHTML = '<div class="faint find-msg">Searching…</div>';
      try {
        const data = await API.objects({ search: q, limit: 8 });
        const items = Array.isArray(data.items) ? data.items : [];
        if (!items.length) {
          out.innerHTML = `<div class="faint find-msg">No objects match “${FMT.esc(q)}”.</div>`;
          return;
        }
        out.innerHTML = '<ul class="find-list">' + items.map((it) =>
          `<li><a href="object.html?id=${encodeURIComponent(it.norad_id)}">` +
          `<span class="find-name">${FMT.esc(it.name)}</span>` +
          `<span class="find-meta">${FMT.esc(typeLabel(it.object_type))} · <span class="code">${FMT.esc(it.norad_id)}</span></span>` +
          '</a></li>').join('') + '</ul>';
      } catch (_) {
        out.innerHTML = '<div class="faint find-msg">Search isn’t available right now.</div>';
      }
    });
  }

  /* --- hero + stats -------------------------------------------------------- */

  const ICON_GLOBE = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.6 2.7 3.9 5.7 3.9 9s-1.3 6.3-3.9 9c-2.6-2.7-3.9-5.7-3.9-9S9.4 5.7 12 3z"/></svg>';
  const ICON_COPY = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linejoin="round" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2.5"/><path d="M16 8V6.5A2.5 2.5 0 0 0 13.5 4h-7A2.5 2.5 0 0 0 4 6.5v7A2.5 2.5 0 0 0 6.5 16H8"/></svg>';

  function renderHead(obj) {
    const hasTle = Array.isArray(obj.tle_history) && obj.tle_history.length > 0;
    const ageH = (Date.now() - Date.parse(obj.epoch)) / 3600000;
    const stale = Number.isFinite(ageH) && ageH > 72;
    $('obj-head').innerHTML = `
      <div class="obj-hero__main k-rise">
        <nav class="obj-crumb" aria-label="Breadcrumb"><a href="/">Overview</a><span aria-hidden="true">›</span><span>Object</span></nav>
        <h1 class="obj-hero__name">${FMT.esc(obj.name)}</h1>
        <div class="obj-hero__meta">
          <span class="chip chip--type">${FMT.esc(typeLabel(obj.object_type))}</span>
          <span class="obj-id" title="NORAD catalog number"><span class="obj-id__k">NORAD</span> <span class="code">${FMT.esc(obj.norad_id)}</span></span>
          <span class="obj-id" title="International designator (COSPAR)"><span class="obj-id__k">Intl</span> <span class="code">${FMT.esc(obj.intl_designator || '—')}</span></span>
          <span class="obj-age${stale ? ' is-stale' : ''}" title="Element epoch ${FMT.esc(FMT.utc(obj.epoch))}">Elements ${FMT.esc(FMT.ageHours(obj.epoch))} old</span>
        </div>
      </div>
      <div class="obj-hero__actions k-rise" style="--d: 40ms">
        <a class="btn btn--primary" href="/?sel=${encodeURIComponent(obj.norad_id)}">${ICON_GLOBE}Show on globe</a>
        ${hasTle ? `<button class="btn" type="button" id="copy-tle">${ICON_COPY}<span>Copy TLE</span></button>` : ''}
      </div>`;
    const copy = $('copy-tle');
    if (copy) {
      copy.addEventListener('click', async () => {
        const t = obj.tle_history[0];
        const label = copy.querySelector('span');
        try {
          await navigator.clipboard.writeText(`${obj.name}\n${t.line1}\n${t.line2}`);
          label.textContent = 'Copied';
        } catch (_) {
          label.textContent = 'Copy unavailable';
        }
        setTimeout(() => { label.textContent = 'Copy TLE'; }, 1800);
      });
    }
  }

  function stat(label, value, sub, cls) {
    return `<div class="stat k-rise" style="--d: ${stat.n++ * 40}ms">
      <div class="stat__label">${label}</div>
      <div class="stat__value${cls ? ` ${cls}` : ''}">${value}</div>
      <div class="stat__sub">${sub}</div>
    </div>`;
  }
  stat.n = 0;

  function unit(v, u) { return `${v}<span class="stat__unit${u === '°' ? ' stat__unit--deg' : ''}">${u}</span>`; }

  function renderStats(obj) {
    const conjs = (obj.conjunctions || []).filter((c) => c.status !== 'past');
    const hasAlt = obj.perigee_km != null && obj.apogee_km != null;
    const altVal = hasAlt
      ? (Math.abs(obj.apogee_km - obj.perigee_km) < 15
        ? unit(FMT.num(Math.round((obj.apogee_km + obj.perigee_km) / 2)), 'km')
        : unit(`${FMT.num(Math.round(obj.perigee_km))}–${FMT.num(Math.round(obj.apogee_km))}`, 'km'))
      : '—';
    const period = obj.period_min != null ? unit(Number(obj.period_min).toFixed(1), 'min') : '—';
    const revs = obj.mean_motion_rev_day != null
      ? `${Number(obj.mean_motion_rev_day).toFixed(2)} orbits a day` : 'Period unavailable';
    const inc = obj.inclination_deg != null ? unit(Number(obj.inclination_deg).toFixed(1), '°') : '—';
    const incSub = obj.inclination_deg == null ? 'Unavailable'
      : obj.inclination_deg > 96 && obj.inclination_deg < 101 ? 'Near sun-synchronous'
        : obj.inclination_deg > 90 ? 'Retrograde orbit'
          : `Reaches ${Math.round(obj.inclination_deg)}° N and S`;
    let worst = 'nominal';
    for (const c of conjs) {
      const l = safeLevel(c.risk_level);
      if (l === 'critical') worst = 'critical';
      else if (l === 'elevated' && worst !== 'critical') worst = 'elevated';
    }
    const next = conjs
      .filter((c) => Date.parse(c.tca) > Date.now())
      .sort((a, b) => Date.parse(a.tca) - Date.parse(b.tca))[0];
    const total = Number(obj.conjunction_count);
    const count = Number.isFinite(total) && total >= conjs.length ? total : conjs.length;
    const conjSub = next
      ? `Next <span data-rel="${FMT.esc(next.tca)}">${FMT.esc(FMT.rel(next.tca))}</span>`
      : 'None in the screening window';
    const conjCls = worst === 'critical' ? 'is-critical' : worst === 'elevated' ? 'is-elevated' : '';
    stat.n = 0;
    $('obj-stats').innerHTML =
      stat('Altitude', altVal, hasAlt ? 'Perigee to apogee' : 'Unavailable') +
      stat('Orbital period', period, revs) +
      stat('Inclination', inc, incSub) +
      stat('Close approaches', FMT.num(count), conjSub, conjCls);
  }

  /* --- space weather: the current storm, through this object's own orbit data
     (api/objects.py -> app.spaceweather.object_effects) ------------------------- */

  function renderSpaceWeather(obj) {
    const sw = obj.space_weather;
    const box = $('sw-obj');
    if (!box || !sw) return;
    box.hidden = false;
    const km = (m) => FMT.dist(m);
    if (!sw.applies) {
      $('sw-obj-lede').textContent = (sw.perigee_km != null && sw.perigee_km > 1000)
        ? 'This orbit stays above about 1,000 km, where the air is too thin for a storm to change it noticeably. '
          + 'Storms still matter up here for electronics and charging, which Kessler does not model.'
        : 'This object’s orbit data shows no measurable drag, so the storm has no effect Kessler can estimate.';
      $('sw-obj-grid').innerHTML = '';
      $('sw-obj-note').textContent = '';
      return;
    }
    const r = sw.density_ratio;
    const storm = r >= 1.25;
    $('sw-obj-lede').textContent =
      `This orbit dips to ${FMT.num(Math.round(sw.perigee_km))} km, where the air is ` +
      (storm ? `${r.toFixed(1)}× denser than on a quiet day right now` : 'about as dense as on a quiet day') +
      `. It is losing about ${FMT.num(Math.round(sw.decay_now_m_day))} m of altitude a day` +
      (storm ? ` (${FMT.num(Math.round(sw.decay_quiet_m_day))} m on a quiet day).` : '.');
    const cell = (label, value, sub, hot) =>
      `<div class="obj-sw__cell${hot ? ' is-hot' : ''}"><div class="obj-sw__k">${label}</div>` +
      `<div class="obj-sw__v">${value}</div><div class="obj-sw__s">${sub}</div></div>`;
    $('sw-obj-grid').innerHTML =
      cell('Air density', `×${r.toFixed(1)}`, 'vs a quiet day', storm) +
      cell('Altitude loss', `${FMT.num(Math.round(sw.decay_now_m_day))} m/day`,
        `quiet day: ${FMT.num(Math.round(sw.decay_quiet_m_day))} m`, storm) +
      cell('Storm drift', km(sw.drift_24h_m), `by tomorrow · ${km(sw.drift_72h_m)} in 3 days`,
        sw.drift_24h_m >= 1000);
    $('sw-obj-note').textContent =
      'How far the storm’s extra drag can move this object from its predicted position, from its own '
      + 'orbit data and NOAA’s storm forecast through NASA/NRL MSIS 2.1. Kessler widens this object’s '
      + 'close-approach uncertainty by the same amount; the risk levels themselves are unchanged.';
  }

  /* --- orbital elements ------------------------------------------------------ */

  function renderElements(obj) {
    const sma = (obj.apogee_km != null && obj.perigee_km != null)
      ? 6378.137 + (obj.apogee_km + obj.perigee_km) / 2
      : null;
    const n = (v, d) => (v != null && Number.isFinite(Number(v)) ? Number(v).toFixed(d) : null);
    const rows = [
      ['Inclination', n(obj.inclination_deg, 2), '°', 'Tilt of the orbit against the equator'],
      ['Ascending node', n(obj.raan_deg, 2), '°', 'Right ascension of the ascending node (RAAN)'],
      ['Eccentricity', n(obj.eccentricity, 6), '', '0 is a perfect circle'],
      ['Argument of perigee', n(obj.arg_perigee_deg, 2), '°', 'Where the lowest point sits in the orbit'],
      ['Mean motion', n(obj.mean_motion_rev_day, 4), 'rev/day', 'Orbits per day'],
      ['Period', n(obj.period_min, 2), 'min', 'Time for one orbit'],
      ['Apogee', obj.apogee_km != null ? FMT.num(Number(obj.apogee_km).toFixed(1)) : null, 'km', 'Highest altitude'],
      ['Perigee', obj.perigee_km != null ? FMT.num(Number(obj.perigee_km).toFixed(1)) : null, 'km', 'Lowest altitude'],
      ['Semi-major axis', sma != null ? FMT.num(sma.toFixed(1)) : null, 'km', 'Derived from apogee and perigee'],
    ];
    $('elements').innerHTML = rows.map(([k, v, u, hint]) => `
      <div class="obj-kv__row" title="${FMT.esc(hint)}">
        <dt>${k}</dt>
        <dd>${v == null ? '<span class="faint">—</span>'
          : `${FMT.esc(v)}${u ? `<span class="obj-kv__u">${u === '°' ? '°' : ` ${u}`}</span>` : ''}`}</dd>
      </div>`).join('');
    $('el-meta').textContent = obj.epoch ? `Epoch ${FMT.when(obj.epoch, { day: false })}` : '';
  }

  /* --- element (TLE) history ------------------------------------------------ */

  function renderTles(obj) {
    const tles = obj.tle_history || [];
    $('tle-meta').textContent = `${tles.length} set${tles.length === 1 ? '' : 's'}`;
    const list = $('tle-body');
    if (!tles.length) {
      list.innerHTML = '<li class="faint tle-none">No element sets stored for this object yet.</li>';
      return;
    }
    list.innerHTML = tles.map((t, i) => `
      <li class="tle-item">
        <button class="tle-row" type="button" data-i="${i}" aria-expanded="false" aria-controls="tle-raw-${i}">
          <span class="tle-row__when">${FMT.esc(FMT.when(t.epoch, { day: false }))}</span>
          <span class="tle-row__age">${i === 0 ? '<span class="chip chip--accent chip--plain">Latest</span>' : ''}${FMT.esc(FMT.ageHours(t.epoch))} old</span>
          <svg class="tle-row__chev" viewBox="0 0 12 12" width="12" height="12" aria-hidden="true"><path d="M4 2.5 7.5 6 4 9.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </button>
        <pre class="tle-raw code" id="tle-raw-${i}" hidden>${FMT.esc(t.line1)}\n${FMT.esc(t.line2)}</pre>
      </li>`).join('');
    list.addEventListener('click', (e) => {
      const btn = e.target.closest('.tle-row');
      if (!btn) return;
      const raw = $(btn.getAttribute('aria-controls'));
      const open = btn.getAttribute('aria-expanded') === 'true';
      btn.setAttribute('aria-expanded', String(!open));
      raw.hidden = open;
    });
  }

  /* --- orbital plot ----------------------------------------------------------- */

  async function renderOrbit(obj) {
    const canvas = $('orbit-canvas');
    const primary = { label: obj.name, path: obj.orbit_path || [] };
    OrbitPlot.draw(canvas, { primary, secondary: null });
    if (!conjParam) return;
    try {
      const conj = await API.conjunction(conjParam);
      const partner = partnerOf(conj, obj.norad_id);
      const pobj = await API.object(partner.norad_id);
      const secondary = { label: pobj.name, path: pobj.orbit_path || [] };
      $('orbit-meta').textContent = `With ${pobj.name} · closest ${FMT.when(conj.tca)}`;
      const lg = $('orbit-legend');
      lg.innerHTML = `
        <span class="key key--1">${FMT.esc(obj.name)}</span>
        <span class="key key--2">${FMT.esc(pobj.name)}</span>
        <span class="faint">Drag to rotate</span>`;
      lg.hidden = false;
      OrbitPlot.draw(canvas, { primary, secondary });
    } catch (_) {
      /* conjunction or partner unavailable - primary orbit already drawn */
    }
  }

  /* --- ground track + altitude profile ------------------------------------- */

  async function loadCoastlines() {
    try {
      const res = await fetch('data/coastlines.json');
      if (!res.ok) return { lines: [] };
      return await res.json();
    } catch (_) {
      return { lines: [] }; /* map still renders - graticule + terminator */
    }
  }

  function renderTrackPanels(obj, data, coastlines) {
    const samples = Array.isArray(data.samples) ? data.samples : [];
    if (!samples.length) {
      $('track-box').hidden = true;
      $('track-empty').hidden = false;
      $('alt-box').hidden = true;
      $('alt-empty').hidden = false;
      $('track-meta').textContent = '';
      $('alt-meta').textContent = '';
      return;
    }
    $('track-empty').hidden = true;
    $('alt-empty').hidden = true;
    $('track-box').hidden = false; /* unhide before drawing - canvases need size */
    $('alt-box').hidden = false;

    const sp = data.subpoint;
    $('track-meta').textContent =
      (sp && Number.isFinite(sp.lat_deg) && Number.isFinite(sp.lon_deg))
        ? `Now over ${fmtLat(sp.lat_deg, 1)}, ${fmtLon(sp.lon_deg, 1)} · ${FMT.km(sp.alt_km, 0)} up`
        : '';
    drawGroundTrack($('track-canvas'), data, { coastlines, label: obj.name });

    const epochMs = Date.parse(data.epoch);
    const periodMs = Number(data.period_min) * 60000;
    const points = samples.map((s, i) => ({
      t: epochMs + (i / samples.length) * periodMs,
      y: s[2],
    }));
    let lo = Infinity;
    let hi = -Infinity;
    for (const p of points) { if (p.y < lo) lo = p.y; if (p.y > hi) hi = p.y; }
    $('alt-meta').textContent = Number.isFinite(lo)
      ? `${FMT.num(Math.round(lo))}–${FMT.num(Math.round(hi))} km` : '';
    drawLineChart($('alt-canvas'), $('alt-tip'), {
      series: [{ label: '', color: '--k-data-1', points }],
      yScale: 'linear',
      xTicks: 4,
      yTicks: 4,
      area: true,
      yFormat: (v) => FMT.km(v, 0),
      tipFormat: (v) => FMT.km(v),
    });
  }

  async function initGroundTrack(obj) {
    const coastlines = await loadCoastlines();
    let drawn = false;
    async function poll() {
      let data;
      try {
        data = await getJson(`/objects/${encodeURIComponent(obj.norad_id)}/track`);
      } catch (err) {
        if (!drawn) { /* first fetch failed - calm note; later polls may recover */
          $('track-err').hidden = false;
          $('track-box').hidden = true;
          $('alt-box').hidden = true;
          $('alt-empty').hidden = false;
        }
        return; /* refetch failure - keep the last drawing */
      }
      $('track-err').hidden = true;
      renderTrackPanels(obj, data, coastlines);
      drawn = true;
    }
    await poll();
    setInterval(poll, 60000);
  }

  /* --- ground-station passes --------------------------------------------------- */

  const SITE_KEY = 'kessler.site';

  /* stored shape {lat, lon, alt_m} - null when unset or malformed */
  function loadSite() {
    let raw = null;
    try { raw = JSON.parse(localStorage.getItem(SITE_KEY)); } catch (_) { return null; }
    if (!raw || typeof raw !== 'object') return null;
    const lat = Number(raw.lat);
    const lon = Number(raw.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;
    const altM = Number(raw.alt_m);
    return { lat, lon, alt_m: Number.isFinite(altM) ? altM : 0 };
  }

  function saveSite(site) {
    try { localStorage.setItem(SITE_KEY, JSON.stringify(site)); } catch (_) { /* private mode */ }
  }

  async function fetchPasses(noradId, site) {
    const url = new URL(`/objects/${encodeURIComponent(noradId)}/passes`, window.location.origin);
    url.searchParams.set('lat', String(site.lat));
    url.searchParams.set('lon', String(site.lon));
    url.searchParams.set('alt_m', String(site.alt_m));
    url.searchParams.set('hours', '48');
    url.searchParams.set('min_el', '10');
    return getJson(url);
  }

  function passesState(msg) {
    $('passes-wrap').hidden = true;
    $('passes-empty').innerHTML = msg;
    $('passes-empty').hidden = !msg;
  }

  let passesData = [];

  function hm(iso) { return FMT.utc(iso).slice(11, 16); }

  function renderPasses(data) {
    const passes = Array.isArray(data.passes) ? data.passes : [];
    passesData = passes;
    $('passes-meta').textContent = passes.length
      ? `${passes.length} in the next 48 h` : '';
    if (!passes.length) {
      passesState('<strong>No passes in the next 48 hours</strong>It never climbs more than 10° above your horizon in that time.');
      return;
    }
    $('passes-body').innerHTML = passes.map((p, i) => {
      const el = Number(p.max_el_deg);
      const elTxt = Number.isFinite(el) ? `${Math.round(el)}°` : '—';
      const good = Number.isFinite(el) && el >= 45;
      return `
      <li class="pass">
        <button class="pass-row" type="button" data-i="${i}" aria-expanded="false">
          <span class="pass-row__when">
            <span class="pass-row__time">${FMT.esc(FMT.when(p.aos))}</span>
            <span class="pass-row__sub"><span data-rel="${FMT.esc(p.aos)}">${FMT.esc(FMT.rel(p.aos))}</span> · ${FMT.esc(FMT.dur(p.duration_s))}</span>
          </span>
          <span class="pass-row__el${good ? ' is-good' : ''}" title="Highest point above your horizon">${FMT.esc(elTxt)}<small>max</small></span>
          <svg class="pass-row__chev" viewBox="0 0 12 12" width="12" height="12" aria-hidden="true"><path d="M4 2.5 7.5 6 4 9.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </button>
      </li>`;
    }).join('');
    $('passes-empty').hidden = true;
    $('passes-wrap').hidden = false;
  }

  function skyReadout(p) {
    const az = (v) => (Number.isFinite(Number(v)) ? ` in the ${compass(Number(v))}` : '');
    const el = Number.isFinite(Number(p.max_el_deg)) ? `${Math.round(Number(p.max_el_deg))}°` : '—';
    return `Rises ${hm(p.aos)}${az(p.aos_az_deg)} · ` +
           `peaks at ${el} at ${hm(p.tca)} · ` +
           `sets ${hm(p.los)} UTC${az(p.los_az_deg)}`;
  }

  function compass(az) {
    const names = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest'];
    return names[Math.round((((az % 360) + 360) % 360) / 45) % 8];
  }

  function onPassClick(e) {
    const btn = e.target.closest('.pass-row');
    if (!btn) return;
    const li = btn.parentElement;
    const open = li.querySelector('.pass-sky');
    if (open) {
      open.remove();
      btn.setAttribute('aria-expanded', 'false');
      return;
    }
    const p = passesData[Number(btn.dataset.i)];
    if (!p) return;
    const hasTrack = Array.isArray(p.track) && p.track.length >= 2;
    const box = document.createElement('div');
    box.className = 'pass-sky';
    box.innerHTML = (hasTrack
      ? '<div class="sky-box"><canvas class="sky-canvas" role="img" aria-label="Path across your sky"></canvas></div>'
      : '<div class="sky-note faint">A sky map needs a newer server build — the times below are still accurate.</div>') +
      `<div class="sky-readout">${FMT.esc(skyReadout(p))}</div>`;
    li.appendChild(box);
    btn.setAttribute('aria-expanded', 'true');
    if (hasTrack) drawSkyView(box.querySelector('.sky-canvas'), p);
  }

  function initPasses(obj) {
    const latEl = $('site-lat');
    const lonEl = $('site-lon');
    const errEl = $('passes-err');
    const form = $('site-form');
    const bar = $('site-bar');
    let fetching = false;

    $('site-lede').innerHTML =
      `See when <strong>${FMT.esc(obj.name)}</strong> rises above your horizon. ` +
      'Set your ground station to predict passes for the next 48 hours.';

    function note(msg) {
      errEl.textContent = msg;
      errEl.hidden = !msg;
    }

    function showForm(editing) {
      form.hidden = false;
      bar.hidden = true;
      if (!editing) {
        $('passes-meta').textContent = '';
        passesState('');
      }
    }

    function showBar(site) {
      form.hidden = true;
      bar.hidden = false;
      $('site-val').textContent = `${fmtLat(site.lat)}, ${fmtLon(site.lon)}`;
    }

    async function refreshPasses() {
      const site = loadSite();
      if (!site) { showForm(false); return; }
      showBar(site);
      if (fetching) return;
      fetching = true;
      $('passes-wrap').classList.add('is-refetching');
      $('passes-meta').textContent = 'Predicting…';
      if ($('passes-wrap').hidden) passesState('<span class="faint">Predicting passes…</span>');
      try {
        const data = await fetchPasses(obj.norad_id, site);
        renderPasses(data);
      } catch (_) {
        $('passes-meta').textContent = '';
        passesState('<strong>Pass predictions aren’t available right now</strong>Try again in a minute.');
      } finally {
        fetching = false;
        $('passes-wrap').classList.remove('is-refetching');
      }
    }

    function setSite(lat, lon) {
      const prev = loadSite();
      saveSite({ lat, lon, alt_m: prev ? prev.alt_m : 0 });
      note('');
      refreshPasses();
    }

    function onSetSite() {
      const la = latEl.value.trim();
      const lo = lonEl.value.trim();
      const lat = Number(la);
      const lon = Number(lo);
      if (!la || !lo || !Number.isFinite(lat) || !Number.isFinite(lon)) {
        note('Enter both a latitude and a longitude in degrees, like 40.71 and -74.01.');
        (la && Number.isFinite(lat) ? lonEl : latEl).focus();
        return;
      }
      if (lat < -90 || lat > 90) { note('Latitude must be between -90 and 90.'); latEl.focus(); return; }
      if (lon < -180 || lon > 180) { note('Longitude must be between -180 and 180.'); lonEl.focus(); return; }
      setSite(lat, lon);
    }

    function onGeo() {
      const btn = $('site-geo');
      if (!('geolocation' in navigator)) {
        note('This browser can’t share its location. Enter coordinates instead.');
        return;
      }
      btn.disabled = true;
      const label = btn.lastChild;
      label.textContent = ' Locating…';
      navigator.geolocation.getCurrentPosition((pos) => {
        btn.disabled = false;
        label.textContent = ' Use my location';
        const lat = Math.round(pos.coords.latitude * 1000) / 1000;
        const lon = Math.round(pos.coords.longitude * 1000) / 1000;
        latEl.value = String(lat);
        lonEl.value = String(lon);
        setSite(lat, lon);
      }, (err) => {
        btn.disabled = false;
        label.textContent = ' Use my location';
        note(err && err.code === 1
          ? 'Location access was blocked. You can enter coordinates instead.'
          : 'We couldn’t get your location. Enter coordinates instead.');
      }, { timeout: 10000, maximumAge: 600000 });
    }

    const stored = loadSite();
    if (stored) {
      latEl.value = String(stored.lat);
      lonEl.value = String(stored.lon);
    }
    $('site-set').addEventListener('click', onSetSite);
    $('site-geo').addEventListener('click', onGeo);
    $('site-change').addEventListener('click', () => {
      showForm(true);
      latEl.focus();
    });
    const onEnter = (e) => { if (e.key === 'Enter') onSetSite(); };
    latEl.addEventListener('keydown', onEnter);
    lonEl.addEventListener('keydown', onEnter);
    $('passes-body').addEventListener('click', onPassClick);
    refreshPasses();
  }

  /* --- probability chart ---------------------------------------------------- */

  async function renderRiskChart(obj) {
    const conjs = obj.conjunctions || [];
    if (!conjs.length) {
      $('risk-box').hidden = true;
      $('risk-empty').hidden = false;
      return;
    }
    /* the two most severe (API order: active first, risk desc) */
    const chosen = conjs.slice(0, 2);
    const details = await Promise.all(chosen.map((c) => API.conjunction(c.id).catch(() => null)));
    const series = [];
    details.forEach((d, i) => {
      if (!d) return;
      const pts = (d.cdms || [])
        .filter((m) => m.probability_of_collision != null && m.probability_of_collision > 0)
        .map((m) => ({ t: new Date(m.message_time).getTime(), y: m.probability_of_collision }));
      if (!pts.length) return;
      const partner = partnerOf(chosen[i], obj.norad_id);
      series.push({
        label: `${partner.name} · ${FMT.when(chosen[i].tca, { day: false })}`,
        color: series.length === 0 ? '--k-data-1' : '--k-data-2',
        points: pts,
      });
    });
    if (!series.length) {
      $('risk-box').hidden = true;
      $('risk-empty').hidden = false;
      return;
    }
    $('risk-meta').textContent = conjs.length > 2
      ? `Top 2 of ${FMT.num(conjs.length)} by risk · log scale`
      : 'Log scale';
    drawLineChart($('risk-canvas'), $('risk-tip'), {
      series,
      yScale: 'log',
      xTicks: 4,
      yTicks: 4,
      yFloor: 1e-10,
      thresholds: [
        { y: 1e-4, label: 'Critical', color: '--k-critical-brd', labelColor: '--k-critical' },
        { y: 1e-6, label: 'Elevated', color: '--k-elevated-brd', labelColor: '--k-elevated' },
      ],
      tipFormat: FMT.pc,
    });
    const lg = $('risk-legend');
    lg.innerHTML = series
      .map((s, i) => `<span class="key key--${i + 1}">${FMT.esc(s.label)}</span>`)
      .join('');
    lg.hidden = false;
  }

  /* --- close approaches table ------------------------------------------------ */

  /* the Overview's storm tag (api/conjunctions.py flags), same wording */
  function stormTag(c) {
    const sw = c && c.space_weather;
    if (!sw || !sw.flag) return '';
    const title = `Storm: the extra drag could shift these objects about ${FMT.dist(sw.drift_m)} by closest ` +
      `approach (uncertainty ${FMT.dist(sw.sigma_m)} → ${FMT.dist(sw.sigma_storm_m)})` +
      (sw.flag === 'may_escalate' ? ` — enough to reach elevated (Pc up to ${FMT.pc(sw.pc_storm)}).` : '. Recheck after the next orbit update.');
    return `<span class="storm-tag" title="${FMT.esc(title)}">Storm</span>`;
  }

  function renderConjTable(obj) {
    const conjs = obj.conjunctions || [];
    const total = Number(obj.conjunction_count);
    $('conj-meta').textContent =
      (Number.isFinite(total) && total > conjs.length)
        ? `Showing ${FMT.num(conjs.length)} of ${FMT.num(total)}`
        : conjs.length ? `${conjs.length} event${conjs.length === 1 ? '' : 's'}` : '';
    if (!conjs.length) {
      $('conj-wrap').hidden = true;
      $('conj-empty').hidden = false;
      return;
    }
    const rowHtml = (c) => {
      const partner = partnerOf(c, obj.norad_id);
      const pcEff = c.predicted_final_risk ?? c.pc_latest;
      const rl = safeLevel(c.risk_level);
      const href = `object.html?id=${encodeURIComponent(partner.norad_id)}&conj=${encodeURIComponent(c.id)}`;
      const past = c.status === 'past';
      return `
        <tr class="row--${rl}${past ? ' is-past' : ''}" data-conj-id="${FMT.esc(c.id)}">
          <td><span class="chip chip--${rl}">${LEVEL_LABELS[rl]}</span>${stormTag(c)}</td>
          <td class="td-with"><a href="${href}">${FMT.esc(partner.name)}</a>
            <span class="td-sub">${FMT.esc(typeLabel(partner.object_type))} · <span class="code">${FMT.esc(partner.norad_id)}</span></span></td>
          <td class="td-when">${FMT.esc(FMT.when(c.tca))}
            <span class="td-sub" data-rel="${FMT.esc(c.tca)}">${FMT.esc(FMT.rel(c.tca))}</span></td>
          <td class="num td-big">${FMT.esc(FMT.dist(c.miss_distance_m))}</td>
          <td class="num col-speed">${FMT.esc(FMT.vel(c.relative_velocity_ms))}</td>
          <td class="num" title="${FMT.esc(FMT.pcOdds(pcEff))} · model ${FMT.esc(c.model_version || '—')}">${FMT.esc(FMT.pc(pcEff))}</td>
          <td class="td-act"><button class="btn btn--xs js-sim" type="button" data-conj="${FMT.esc(c.id)}"
              aria-label="Simulate the encounter with ${FMT.esc(partner.name)}">Simulate</button></td>
        </tr>`;
    };

    const PAGE = 8;
    let shown = 0;
    const more = $('conj-more');
    function showMore() {
      const next = conjs.slice(shown, shown + (shown ? 25 : PAGE));
      $('conj-body').insertAdjacentHTML('beforeend', next.map(rowHtml).join(''));
      shown += next.length;
      const left = conjs.length - shown;
      more.hidden = left <= 0;
      more.textContent = `Show ${FMT.num(Math.min(left, 25))} more · ${FMT.num(left)} not shown`;
    }
    more.addEventListener('click', showMore);
    showMore();
    /* ?conj=<id> (Alerts "Details", drawer links): reveal, scroll to and
       highlight that event's row in the close-approach list */
    if (conjParam) {
      const idx = conjs.findIndex((c) => String(c.id) === String(conjParam));
      if (idx >= 0) {
        while (shown <= idx) showMore();
        const tr = Array.from($('conj-body').rows)
          .find((r) => r.dataset.conjId === String(conjParam));
        if (tr) {
          tr.classList.add('is-target');
          setTimeout(() => tr.scrollIntoView({ block: 'center', behavior: 'smooth' }), 400);
        }
      }
    }
    $('conj-body').addEventListener('click', (e) => {
      const sim = e.target.closest('.js-sim');
      if (!sim) return;
      e.preventDefault();
      const id = Number(sim.dataset.conj);
      if (!Number.isFinite(id)) return;
      if (typeof KSim !== 'undefined' && typeof KSim.open === 'function') {
        try { KSim.open({ conjunctionId: id }); return; } catch (_) { /* module fault */ }
      }
      window.location.href = `/?sim=${encodeURIComponent(id)}`;
    });
  }

  /* relative times tick locally */
  setInterval(() => {
    document.querySelectorAll('[data-rel]').forEach((el) => {
      el.textContent = FMT.rel(el.dataset.rel);
    });
  }, 15000);

  /* --- boot ------------------------------------------------------------------- */

  API.status().then((s) => {
    /* #hdr-status is filled by theme.js (one format on every page) */
    if (s.has_demo_data) $('demo-flag').hidden = false;
    const autoH = Number(s.auto_refresh_interval_h);
    if (Number.isFinite(autoH) && autoH > 0) {
      const auto = $('sl-auto');
      auto.textContent = `Refreshes every ${autoH} h`;
      auto.hidden = false;
    }
  }).catch(() => { /* header status unavailable */ });

  initFind();

  (async () => {
    if (!noradParam) {
      fail('missing-param');
      return;
    }
    if (!/^\d{1,9}$/.test(noradParam)) {   /* not a catalog number - don't ask */
      fail('not-found');
      return;
    }
    let obj;
    try {
      obj = await API.object(noradParam);
    } catch (err) {
      fail(err && (err.status === 404 || err.status === 422) ? 'not-found' : 'unreachable');
      return;
    }
    document.title = `${obj.name} · Kessler`;
    $('obj-loading').hidden = true;
    $('content').hidden = false; /* unhide before drawing so canvases have size */
    renderHead(obj);
    renderStats(obj);
    renderElements(obj);
    renderSpaceWeather(obj);
    renderTles(obj);
    renderConjTable(obj);
    renderOrbit(obj);
    renderRiskChart(obj);
    initGroundTrack(obj);
    initPasses(obj);
  })();
})();
