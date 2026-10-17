/* Kessler - dashboard controller.
   Stat strip from /status, active-conjunction table from /conjunctions
   (API pre-sorts by effective risk desc, TCA asc - rendered in API order).
   Polls both every 60 s; countdown cells tick on a single 1 s interval.
   Row click expands an inline drawer (tr.drawer) with the CDM sequence
   table and two mini charts via drawLineChart (js/charts.js).
   Orbital situational display: KGlobe (js/globe.js) drives the #globe canvas;
   this file owns init/speed/sim-clock wiring, feeds it conjunction items after
   every poll, and renders the SELECTED OBJECT panel from onSelect payloads.
   globe typeahead search + type/RISK ONLY filters (KGlobe.list/setFilter,
   guarded so an older engine degrades gracefully), client-side conjunction
   table search/risk filter + sortable columns re-rendered through the same
   renderTable path (drawer re-attach + countdown keep working), and the
   ELEVATED/CRITICAL stat tiles double as risk-filter shortcuts.
   ?sel=/?risk=/?search= URL params applied once on load (then cleaned via
   history.replaceState) and a tiny window.KesslerUI hook surface consumed by
   js/assistant.js - see the URL-param section below.
   encounter mode - the drawer footer's SHOW ON GLOBE -> and the ?conj=<id>
   URL param hand a conjunction pair to KGlobe.showEncounter (watch mode), and
   KesslerUI gains a showEncounter wrapper for the assistant executor. Table
   power: LOAD MORE appends the next API page (200) to conjItems through the
   same applyTableView path, EXPORT CSV downloads the currently rendered
   (post-filter/sort) rows as RFC-4180 CSV via a Blob.
   the drawer footer gains SIMULATE -> (KSim.open, typeof-guarded and
   hidden when the encounter-simulator module is absent).
   globe color-mode group (RISK/TYPE/CONST/ALT) + HEAT toggle wired to
   KGlobe.setColorMode/setHeatmap with the legend re-rendered from
   KGlobe.legend() when mode != risk (all typeof-guarded - the engine's
   surface builds in parallel); time-scrub strip under the canvas (±24 h,
   drag -> KGlobe.setSimTime, live cursor from status().simMs, double-click
   snaps back to real now); CHANGES panel in the right column fed by
   GET /changes on load + after every poll (hidden until the endpoint
   answers - pre-restart backends 404 it); statusline AUTO-REFRESH flag
   from /status.auto_refresh_interval_h.
   TCA timeline panel between the globe row and the table - drawTimeline
   (js/timeline.js, typeof-guarded) is fed the UNFILTERED conjItems through
   the applyTableView path (every poll/filter change redraws; the timeline
   always shows the true picture), a marker click opens the encounter
   simulator (KSim.open), and the debounced resize redraws the canvas.
   Also: SURFACE toggle in the globe filter row -> KGlobe.setSurface
   (typeof-guarded; initial state mirrors the engine's persisted
   localStorage 'kessler.surface' default-ON), and the CHANGES panel gains a
   MANEUVER CANDIDATES · ELEMENT-DELTA HEURISTIC subsection fed by
   GET /anomalies (top 5, ΔSMA signed km, object links; 404 = pre-restart
   backend keeps the subsection absent).
   WATCH LOG panel in the right column (GET /watchlog on the poll
   cadence - hidden while the endpoint 404s, empty state on an empty log;
   summary_md renders through an escape-first mini-markdown copied from the
   assistant.js idiom, never raw innerHTML) and WHY THIS NUMBER in the
   conjunction drawer (GET /risk/{id}/explain - carry_forward renders the
   one-line deference note, learned_residual the top-5 SHAP contribution
   bars; 404/unavailable/pre-restart backends omit the block silently).
   design (Overview hero): the globe is a full-bleed "space" stage with
   floating glass UI - summary card (stat-* ids unchanged; the elevated /
   critical tiles still filter the table, and whichever matters more leads),
   a details sheet (selected object / changes / watch log, collapsible) and
   a control dock (search, filters, color modes, time scrub, speed, zoom).
   All copy is sentence case and plain language; machine precision lives in
   tooltips and secondary text. Layout insets are pushed to the engine
   (KGlobe.setViewInset) so the Earth centers between the floating panels,
   a star field is painted once per resize under the globe canvas, and the
   risk filter is a segmented control mirroring the hidden #conj-risk select
   (still the single source of truth for KesslerUI and the tiles). */
'use strict';

(() => {
  const POLL_MS = 60000;
  const KNOWN_LEVELS = { critical: true, elevated: true, nominal: true };

  const els = {
    hdrStatus: document.getElementById('hdr-status'),
    errorBanner: document.getElementById('error-banner'),
    statCatalog: document.getElementById('stat-catalog'),
    statActive: document.getElementById('stat-active'),
    statActiveLabel: document.getElementById('stat-active-label'),
    statElevated: document.getElementById('stat-elevated'),
    statCritical: document.getElementById('stat-critical'),
    statIngest: document.getElementById('stat-ingest'),
    statIngestSub: document.getElementById('stat-ingest-sub'),
    statModel: document.getElementById('stat-model'),
    conjMeta: document.getElementById('conj-meta'),
    tableWrap: document.getElementById('conj-table-wrap'),
    tbody: document.getElementById('conj-tbody'),
    demoFlag: document.getElementById('demo-flag'),
    globeWrap: document.getElementById('globe-wrap'),
    encBar: document.getElementById('enc-bar'),
    encPair: document.getElementById('enc-pair'),
    encSub: document.getElementById('enc-sub'),
    encNote: document.getElementById('enc-note'),
    encSim: document.getElementById('enc-sim'),
    encClose: document.getElementById('enc-close'),
    globeCanvas: document.getElementById('globe'),
    globeTip: document.getElementById('globe-tip'),
    globeCount: document.getElementById('globe-count'),
    globeSim: document.getElementById('globe-sim'),
    globeSpeed: document.getElementById('globe-speed'),
    selBody: document.getElementById('sel-body'),
    statTileElevated: document.getElementById('stat-tile-elevated'),
    statTileCritical: document.getElementById('stat-tile-critical'),
    conjThead: document.getElementById('conj-thead'),
    conjSearch: document.getElementById('conj-search'),
    conjRisk: document.getElementById('conj-risk'),
    globeFilters: document.getElementById('globe-filters'),
    globeSearch: document.getElementById('globe-search'),
    globeTypeahead: document.getElementById('globe-typeahead'),
    globeTypeGroup: document.getElementById('globe-type-group'),
    globeRiskOnly: document.getElementById('globe-risk-only'),
    loadMore: document.getElementById('conj-load-more'),
    conjExport: document.getElementById('conj-export'),
    globeModeGroup: document.getElementById('globe-mode-group'),
    globeHeat: document.getElementById('globe-heat'),
    globeSurface: document.getElementById('globe-surface'),
    globeLegend: document.getElementById('globe-legend'),
    timeScrub: document.getElementById('time-scrub'),
    scrubTrack: document.getElementById('scrub-track'),
    scrubCursor: document.getElementById('scrub-cursor'),
    changesPanel: document.getElementById('changes-panel'),
    changesMeta: document.getElementById('changes-meta'),
    changesBody: document.getElementById('changes-body'),
    slAuto: document.getElementById('sl-auto'),
    timelinePanel: document.getElementById('timeline-panel'),
    timelineCanvas: document.getElementById('timeline-canvas'),
    timelineTip: document.getElementById('timeline-tip'),
    timelineMeta: document.getElementById('timeline-meta'),
    watchlogPanel: document.getElementById('watchlog-panel'),
    watchlogBody: document.getElementById('watchlog-body'),
    stage: document.getElementById('stage'),
    stageSide: document.getElementById('stage-side'),
    stageDock: document.getElementById('stage-dock'),
    stageLeft: document.querySelector('.stage__left'),
    stageHint: document.getElementById('stage-hint'),
    globeStars: document.getElementById('globe-stars'),
    globeNow: document.getElementById('globe-now'),
    heroWindow: document.getElementById('hero-window'),
    scrubFill: document.getElementById('scrub-fill'),
    selClear: document.getElementById('sel-clear'),
    selPanel: document.getElementById('sel-panel'),
    conjRiskSeg: document.getElementById('conj-risk-seg'),
    zoomGroup: document.querySelector('.zoom-group'),
  };

  let expandedId = null;   /* conjunction id of the open drawer, or null */
  let hasData = false;     /* at least one successful poll rendered */
  const drawerCache = new Map(); /* id -> /conjunctions/{id} payload (per poll) */
  const whyCache = new Map();    /* id -> /risk/{id}/explain payload | null (per poll) */
  let conjItems = [];      /* latest /conjunctions items (globe + selected panel) */
  let lastSel = null;      /* last KGlobe onSelect payload; null = nothing selected */
  let globeReady = false;  /* KGlobe present and initialized without throwing */
  let conjTotal = 0;       /* API count from the last successful poll */
  let currentView = [];    /* rows as last rendered (post-filter/sort) - CSV export */
  let loadingMore = false; /* a LOAD MORE page fetch is in flight */
  const LOAD_PAGE = 200;   /* LOAD MORE page size (poll's first fetch stays 1000) */
  let visibleRows = LOAD_PAGE; /* table render cap - LOAD MORE raises it */

  /* conjunction table view state (client-side over conjItems) */
  let tableSearch = '';
  let tableRisk = '';      /* '' | 'critical' | 'elevated' | 'nominal' */
  let riskFetched = null;  /* {level, items, count} - server-side page for the
                              active risk filter (the 1000-row poll page is
                              risk-desc, so low levels may have zero rows in it) */
  let riskFetching = null; /* level currently being fetched, or null */
  let sortKey = null;      /* null = API order (risk desc, TCA asc) */
  let sortDir = 'desc';

  /* globe filter state (mirrors KGlobe.setFilter) */
  const GLOBE_TYPES = ['PAYLOAD', 'DEBRIS', 'RB', 'UNKNOWN'];
  const typesOn = { PAYLOAD: true, DEBRIS: true, RB: true, UNKNOWN: true };
  let riskOnly = false;

  /* globe typeahead state */
  let taMatches = [];
  let taActive = -1;       /* keyboard-highlighted match index, -1 = none */

  /* globe color mode / heatmap / time scrub state */
  let colorMode = 'type';  /* engine default - the static legend applies */
  let heatOn = false;
  /* SURFACE|WIREFRAME - mirrors globe.js's persisted default (ON unless
     localStorage 'kessler.surface' === '0'); the ENGINE owns persistence,
     this flag only keeps the button state in sync. */
  let surfaceOn = true;
  try {
    if (typeof localStorage !== 'undefined'
        && localStorage.getItem('kessler.surface') === '0') surfaceOn = false;
  } catch (_) { /* storage disabled - session-only default ON */ }
  let scrubbing = false;   /* pointer drag on the time-scrub track in flight */
  const SCRUB_RANGE_MS = 24 * 3600 * 1000; /* ±24 h around real now */

  /* --- tokens ------------------------------------------------------------ */

  function tok(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function safeLevel(level) {
    return KNOWN_LEVELS[level] ? level : 'nominal';
  }

  function riskColor(level) {
    if (level === 'critical') return tok('--k-critical');
    if (level === 'elevated') return tok('--k-elevated');
    return tok('--k-text-2');
  }

  /* toggle buttons: .is-active and aria-pressed always move together */
  function setPressed(btn, on) {
    btn.classList.toggle('is-active', !!on);
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  }

  /* --- plain-language formatting ------------------------------------------ */

  const LEVEL_LABEL = { critical: 'Critical', elevated: 'Elevated', nominal: 'Nominal' };
  const DAY_MS = 86400000;
  const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  function pad2(n) { return String(n).padStart(2, '0'); }

  /* 83333 -> "83K", 2.94e6 -> "2.9M" */
  function compactNum(n) {
    if (n < 1000) return String(Math.max(1, Math.round(n)));
    for (const [v, u] of [[1e9, 'B'], [1e6, 'M'], [1e3, 'K']]) {
      if (n >= v) {
        const x = n / v;
        return (x < 10 ? x.toFixed(1).replace(/\.0$/, '') : String(Math.round(x))) + u;
      }
    }
    return String(Math.round(n));
  }

  /* collision probability as odds: 1.2e-5 -> "1 in 83K" (the sci value stays
     alongside - this is presentation of the same backend number) */
  function odds(p) {
    if (p === null || p === undefined || !Number.isFinite(p) || p < 0) return '—';
    if (p < 1e-10) return '< 1 in 10B';
    return `1 in ${compactNum(1 / p)}`;
  }

  /* "412 m" -> 412<small unit>; anything without a trailing unit passes */
  function unitHtml(str) {
    const m = /^(.*\d)\s?([a-zA-Z/°]+)$/.exec(String(str));
    if (!m) return FMT.esc(str);
    return `${FMT.esc(m[1])}<span class="u">${FMT.esc(m[2])}</span>`;
  }

  /* 42 min / 5 h 12 min / 2 d 5 h - the FMT.span ladder (same as FMT.rel) */
  function durText(ms) {
    const m = Math.floor(Math.abs(ms) / 60000);
    if (m < 1) return 'under a minute';
    if (m < 60) return `${m} min`;
    return FMT.span(m * 60000);
  }

  /* time until an instant, for countdown cells */
  function humanIn(iso) {
    const t = Date.parse(iso);
    if (!Number.isFinite(t)) return '—';
    const d = t - Date.now();
    return d < 0 ? 'Passed' : durText(d);
  }

  /* "Today 13:22" / "Tomorrow 04:10" / "Sat 09:15" - UTC */
  function whenText(iso) {
    const t = Date.parse(iso);
    if (!Number.isFinite(t)) return '—';
    const dt = new Date(t);
    const dayDiff = Math.floor(t / DAY_MS) - Math.floor(Date.now() / DAY_MS);
    const day = dayDiff === 0 ? 'Today' : dayDiff === 1 ? 'Tomorrow'
      : dayDiff === -1 ? 'Yesterday'
        : Math.abs(dayDiff) < 6 ? WEEKDAYS[dt.getUTCDay()]
          : `${dt.getUTCDate()} ${MONTHS[dt.getUTCMonth()]}`;
    return `${day} ${pad2(dt.getUTCHours())}:${pad2(dt.getUTCMinutes())}`;
  }

  /* "24 Sep, 00:59 UTC" */
  function shortUtc(iso) {
    const t = Date.parse(iso);
    if (!Number.isFinite(t)) return '—';
    const dt = new Date(t);
    return `${dt.getUTCDate()} ${MONTHS[dt.getUTCMonth()]}, ` +
      `${pad2(dt.getUTCHours())}:${pad2(dt.getUTCMinutes())} UTC`;
  }

  /* "11 min ago" */
  function agoText(iso) {
    const t = Date.parse(iso);
    if (!Number.isFinite(t)) return '—';
    const d = Date.now() - t;
    return d < 60000 ? 'Just now' : `${durText(d)} ago`;
  }

  /* --- summary card / header ---------------------------------------------- */

  function riskLabel(tile, n) {
    const el = tile.querySelector('.hero-risk__label');
    if (!el) return;
    el.textContent = n === 1 ? el.dataset.one : el.dataset.many;
  }

  function renderStatus(st) {
    const model = String(st.model_version || '—');
    /* #hdr-status is filled by theme.js (one format on every page) */
    const hours = st.window_hours != null ? st.window_hours : 72;
    els.heroWindow.textContent = `Next ${hours} hours`;
    els.statCatalog.textContent = FMT.num(st.catalog_size);
    els.statActive.textContent = FMT.num(st.active_conjunctions);
    els.statActive.title = `Active conjunctions in the ${hours}-hour screening window`;
    els.statActiveLabel.textContent = 'All close approaches';
    const elev = Number(st.elevated_count) || 0;
    const crit = Number(st.critical_count) || 0;
    /* fresh = live data under 6 h old; a snapshot or stale catalog never
       earns the green all-clear */
    const ageMs = st.last_ingest_at ? Date.now() - Date.parse(st.last_ingest_at) : Infinity;
    const fresh = !!st.last_ingest_at && !st.last_run_error && !st.snapshot && ageMs < 6 * 3600e3;
    els.statElevated.textContent = FMT.num(st.elevated_count);
    els.statElevated.classList.toggle('is-elevated', elev > 0);
    els.statCritical.textContent = FMT.num(st.critical_count);
    els.statCritical.classList.toggle('is-critical', crit > 0);
    /* whichever matters more leads the card: critical when there is any */
    const critLead = crit > 0;
    els.statTileCritical.classList.toggle('is-lead', critLead);
    els.statTileElevated.classList.toggle('is-lead', !critLead);
    /* an honest all-clear only when the data is fresh */
    els.statElevated.classList.toggle('is-ok', !critLead && elev === 0 && fresh);
    riskLabel(els.statTileElevated, elev);
    riskLabel(els.statTileCritical, crit);
    /* tiles act as risk-filter shortcuts only while their count is non-zero */
    els.statTileElevated.disabled = !(elev > 0);
    els.statTileCritical.disabled = !(crit > 0);
    els.statIngest.textContent = st.last_ingest_at ? agoText(st.last_ingest_at) : 'Never';
    els.statIngest.title = st.last_ingest_at ? `Catalog ingested ${FMT.utc(st.last_ingest_at)}` : '';
    els.statIngestSub.textContent = st.last_run_error
      ? 'Last refresh failed'
      : (st.last_ingest_at ? shortUtc(st.last_ingest_at) : 'No catalog yet');
    els.statIngestSub.classList.toggle('is-error', !!st.last_run_error);
    els.statIngestSub.title = st.last_run_error || '';
    els.statModel.textContent = model;
    els.statModel.title = 'Version of the risk model that produced these numbers';
    els.demoFlag.hidden = !st.has_demo_data;
    /* scheduler flag - key absent on older backends, null when off */
    const autoH = Number(st.auto_refresh_interval_h);
    if (Number.isFinite(autoH) && autoH > 0) {
      els.slAuto.textContent = `Refreshes every ${autoH} h`;
      els.slAuto.hidden = false;
    } else {
      els.slAuto.hidden = true;
    }
  }

  /* --- conjunction table -------------------------------------------------- */

  function sparkValues(trend) {
    if (!Array.isArray(trend)) return [];
    return trend
      .filter((p) => typeof p === 'number' && p > 0)
      .map((p) => Math.log10(p));
  }

  function objectCell(o) {
    if (!o) return '<span class="faint">—</span>';
    const norad = encodeURIComponent(String(o.norad_id));
    return `<a href="object.html?id=${norad}" title="${FMT.esc(o.name)}">${FMT.esc(o.name)}</a>` +
      `<span class="code">${FMT.esc(String(o.norad_id))}</span>`;
  }

  /* space weather (api/conjunctions.py attach_space_weather): a small tag
     on the pairs worth a second look during a storm - 'watch' (already
     elevated/critical, and the storm widens it) or 'may_escalate' (nominal,
     but the widened uncertainty allows an elevated Pc). The risk level
     itself is unchanged, so the tag sits beside it. */
  function stormTag(it) {
    const sw = it && it.space_weather;
    if (!sw || !sw.flag) return '';
    const shift = `the storm's extra drag could shift these objects about ${FMT.dist(sw.drift_m)} ` +
      `by closest approach (uncertainty ${FMT.dist(sw.sigma_m)} → ${FMT.dist(sw.sigma_storm_m)})`;
    const title = sw.flag === 'watch'
      ? `Storm: ${shift}. Recheck after the next orbit update.`
      : `Storm: ${shift} — enough for this approach to reach elevated (Pc up to ${FMT.pc(sw.pc_storm)}).`;
    return `<span class="storm-tag" title="${FMT.esc(title)}">Storm</span>`;
  }

  /* drawer: the same storm effect in a sentence */
  function stormNote(detail) {
    const sw = detail && detail.space_weather;
    if (!sw || !(sw.drift_m >= 50)) return '';
    const lead = sw.flag === 'may_escalate'
      ? 'The storm could make this approach riskier.'
      : sw.widened ? 'Space weather is widening this prediction.' : 'Space weather is nudging this prediction.';
    const tail = sw.flag === 'may_escalate'
      ? ` At that uncertainty the chance of collision could reach ${FMT.pc(sw.pc_storm)} — elevated.`
      : '';
    return '<div class="storm-note' + (sw.flag ? ' is-widened' : '') + '">' +
      '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="3" fill="currentColor"/>' +
      '<path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>' +
      `<p><strong>${FMT.esc(lead)}</strong> The storm's extra drag could move these objects about ` +
      `${FMT.esc(FMT.dist(sw.drift_m))} along their orbits by closest approach, so the 1-sigma ` +
      `uncertainty grows from ${FMT.esc(FMT.dist(sw.sigma_m))} to ${FMT.esc(FMT.dist(sw.sigma_storm_m))}.` +
      `${FMT.esc(tail)} Expect the next orbit update to move this approach; the risk level shown is unchanged.</p></div>`;
  }

  function soonClass(iso) {
    const d = Date.parse(iso) - Date.now();
    return d >= 0 && d < 6 * 3600000 ? ' is-soon' : '';
  }

  function buildRow(it) {
    const lvl = safeLevel(it.risk_level);
    const effPc = it.predicted_final_risk != null ? it.predicted_final_risk : it.pc_latest;
    const tr = document.createElement('tr');
    tr.dataset.id = String(it.id);
    tr.className = 'clickable' +
      (lvl === 'critical' ? ' row--critical' : lvl === 'elevated' ? ' row--elevated' : '');
    tr.tabIndex = 0;                     /* keyboard: Enter/Space toggle the drawer */
    tr.setAttribute('role', 'button');
    const pcTitle = `Pc ${FMT.pc(effPc)} — ` +
      (it.predicted_final_risk != null ? 'predicted at closest approach' : 'latest message') +
      (it.pc_source === 'covariance' ? ', computed from delivered covariance' : '');
    tr.innerHTML =
      `<td><span class="chip chip--${lvl}">${LEVEL_LABEL[lvl]}</span>${stormTag(it)}</td>` +
      /* tiny covariance tag when the Pc comes from a real delivered
         covariance (pc_source on the summary item; absent shows nothing). */
      `<td class="num" title="${FMT.esc(pcTitle)}">` +
        `<span class="pc-odds">${FMT.esc(odds(effPc))}</span>` +
        `<span class="pc-sci">${FMT.esc(FMT.pc(effPc))}` +
        (it.pc_source === 'covariance' ? '<span class="cov-tag">Cov</span>' : '') +
        '</span></td>' +
      '<td class="td-spark"></td>' +
      `<td class="num td-num">${unitHtml(FMT.dist(it.miss_distance_m))}</td>` +
      `<td class="num td-num">${unitHtml(FMT.vel(it.relative_velocity_ms))}</td>` +
      `<td title="${FMT.esc(FMT.utc(it.tca))}"><span class="td-when">${FMT.esc(whenText(it.tca))}</span></td>` +
      `<td class="num td-in${soonClass(it.tca)}" data-tca="${FMT.esc(it.tca)}" ` +
        `title="Countdown ${FMT.esc(FMT.countdown(it.tca))}">${FMT.esc(humanIn(it.tca))}</td>` +
      `<td class="td-obj">${objectCell(it.primary)}</td>` +
      `<td class="td-obj">${objectCell(it.secondary)}</td>` +
      `<td class="num td-num">${FMT.esc(FMT.num(it.cdm_count))}</td>`;

    const sparkCell = tr.querySelector('.td-spark');
    const values = sparkValues(it.pc_trend);
    if (values.length) {
      const canvas = document.createElement('canvas');
      canvas.className = 'spark';
      sparkCell.appendChild(canvas);
      drawSparkline(canvas, values, riskColor(lvl));
    } else {
      sparkCell.innerHTML = '<span class="faint">—</span>';
    }
    return tr;
  }

  function renderTable(items, emptyMsg) {
    const frag = document.createDocumentFragment();
    if (!items.length) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 10;
      td.className = 'empty';
      td.textContent = emptyMsg || 'No close approaches in the next 72 hours.';
      tr.appendChild(td);
      frag.appendChild(tr);
    }
    for (const it of items) frag.appendChild(buildRow(it));
    els.tbody.replaceChildren(frag);

    /* re-attach the open drawer after a poll re-render */
    if (expandedId != null) {
      const anchor = els.tbody.querySelector(`tr[data-id="${expandedId}"]`);
      if (!anchor) { expandedId = null; return; }
      anchor.classList.add('row--expanded');
      const drawer = buildDrawerRow();
      anchor.after(drawer);
      populateDrawer(expandedId, drawer);
    }
  }

  /* --- table view: client-side filter + sort over stored items -------------- */

  const SORT_GETTERS = {
    pc: (it) => (it.predicted_final_risk != null ? it.predicted_final_risk : it.pc_latest),
    miss: (it) => it.miss_distance_m,
    rel: (it) => it.relative_velocity_ms,
    tca: (it) => { const t = Date.parse(it.tca); return Number.isNaN(t) ? null : t; },
    cdms: (it) => it.cdm_count,
  };
  const SORT_LABELS = {
    pc: 'collision chance', miss: 'miss distance', rel: 'speed',
    tca: 'closest approach', cdms: 'messages',
  };

  /* q is lowercased: name substring or norad prefix, primary or secondary */
  function matchesConj(it, q) {
    const hit = (o) => !!o && (
      String(o.name || '').toLowerCase().includes(q) ||
      String(o.norad_id != null ? o.norad_id : '').startsWith(q));
    return hit(it.primary) || hit(it.secondary);
  }

  /* null-safe comparator: missing values sink to the bottom either direction */
  function makeComparator(get, dir) {
    return (a, b) => {
      const va = get(a), vb = get(b);
      const aBad = va == null || (typeof va === 'number' && Number.isNaN(va));
      const bBad = vb == null || (typeof vb === 'number' && Number.isNaN(vb));
      if (aBad && bBad) return 0;
      if (aBad) return 1;
      if (bBad) return -1;
      return va < vb ? -dir : va > vb ? dir : 0;
    };
  }

  /* Single re-render path for filter/sort/poll - flows through renderTable so
     drawer re-attach and the countdown ticker keep working unchanged. */
  function applyTableView() {
    if (!hasData) return; /* keep the loading / error placeholder untouched */
    let view = (tableRisk && riskFetched && riskFetched.level === tableRisk)
      ? riskFetched.items : conjItems;
    const q = tableSearch.trim().toLowerCase();
    if (q) view = view.filter((it) => matchesConj(it, q));
    if (tableRisk) view = view.filter((it) => safeLevel(it.risk_level) === tableRisk);
    if (sortKey && SORT_GETTERS[sortKey]) {
      view = view.slice().sort(
        makeComparator(SORT_GETTERS[sortKey], sortDir === 'asc' ? 1 : -1));
    }
    currentView = view; /* exactly what EXPORT CSV serializes (uncapped) */
    /* render at most visibleRows rows - the 1000-item poll fetch feeds the
       globe/timeline, but a thousand-row TABLE makes the page unscrollable;
       LOAD MORE reveals further rows before fetching further pages */
    renderTable(view.slice(0, visibleRows),
      (q || tableRisk)
        ? (riskFetching === tableRisk && tableRisk
            ? 'Loading close approaches at this level…'
            : 'No close approaches match this filter.')
        : null);
    const sortTxt = sortKey
      ? `sorted by ${SORT_LABELS[sortKey]}, ${sortDir === 'desc' ? 'high to low' : 'low to high'}`
      : 'highest risk first';
    const filtered = !!(q || tableRisk);
    els.conjMeta.textContent =
      `Showing ${FMT.num(Math.min(visibleRows, view.length))} of ` +
      `${FMT.num(filtered ? view.length : conjTotal)}` +
      `${filtered ? ' matching' : ''} · ${sortTxt}`;
    renderSortIndicators();
    els.conjExport.disabled = false;
    updateLoadMore();
    renderTimeline(); /* single feed point - every poll/filter re-render */
  }

  /* --- TCA timeline (js/timeline.js) ------------------------------------------ */

  /* Always the UNFILTERED conjItems - the timeline shows the true picture no
     matter what the table filters hide. drawTimeline builds in parallel:
     typeof-guarded, and a renderer fault never breaks the table render path. */
  function renderTimeline() {
    if (!hasData || els.timelinePanel.hidden) return;
    let res = null;
    try {
      res = drawTimeline(els.timelineCanvas, els.timelineTip, conjItems, {
        onClick: onTimelineClick,
      });
    } catch (_) { return; /* renderer fault - panel keeps its last frame */ }
    if (res && typeof res.plotted === 'number') {
      els.timelineMeta.textContent = res.plotted
        ? (res.eligible > res.plotted
            ? `${FMT.num(res.plotted)} riskiest of ${FMT.num(res.eligible)}`
            : `${FMT.num(res.plotted)} close approach${res.plotted === 1 ? '' : 'es'}`)
        : '';
    }
  }

  /* marker click -> encounter simulator (typeof-guarded: parallel module) */
  function onTimelineClick(it) {
    if (!it || it.id == null) return;
    if (typeof KSim !== 'undefined' && typeof KSim.open === 'function') {
      try { KSim.open({ conjunctionId: it.id }); } catch (_) { /* module fault */ }
    }
  }

  /* LOAD MORE stays visible while the table hides rendered rows (the
     visibleRows cap) or fewer items are loaded than the API holds -
     revealing is free and runs first; fetching appends pages that flow
     through the same applyTableView re-filter. */
  function updateLoadMore() {
    const hidden = hasData && currentView.length > visibleRows;
    /* with a risk filter active the view is its own server-side page -
       fetching further unfiltered pages would not grow it */
    const more = hasData && (hidden || (!tableRisk && conjItems.length < conjTotal));
    els.loadMore.hidden = !more;
    if (more && !loadingMore) {
      const next = hidden
        ? Math.min(LOAD_PAGE, currentView.length - visibleRows)
        : Math.min(LOAD_PAGE, conjTotal - conjItems.length);
      els.loadMore.textContent = `Show ${FMT.num(next)} more`;
    }
  }

  /* Reveal hidden rows first (free); only then fetch the NEXT API page in
     API order (offset = loaded count) and append. A 60 s poll still
     re-fetches only the first 1000 (the API limit cap) - extra pages drop
     then and the control reappears. */
  async function onLoadMore() {
    if (loadingMore || !hasData) return;
    if (currentView.length > visibleRows) {
      visibleRows += LOAD_PAGE;
      applyTableView();
      return;
    }
    if (tableRisk || conjItems.length >= conjTotal) return;
    visibleRows += LOAD_PAGE;   /* the fetched page must actually appear */
    loadingMore = true;
    els.loadMore.disabled = true;
    els.loadMore.textContent = 'Loading…';
    try {
      const page = await API.conjunctions({ offset: conjItems.length, limit: LOAD_PAGE });
      const items = Array.isArray(page.items) ? page.items : [];
      /* dedupe on id - the feed can shift between the poll and this fetch */
      const seen = new Set(conjItems.map((it) => it.id));
      for (const it of items) if (!seen.has(it.id)) conjItems.push(it);
      conjTotal = page.count != null ? page.count : conjTotal;
      if (!items.length) conjTotal = conjItems.length; /* server has no more */
      applyTableView();
      if (globeReady) {
        try { KGlobe.setConjunctions(conjItems); } catch (_) { /* engine fault */ }
      }
      clearError();
    } catch (err) {
      showError(err);
    } finally {
      loadingMore = false;
      els.loadMore.disabled = false;
      updateLoadMore();
    }
  }

  /* --- CSV export (currently rendered rows) --------------------------------- */

  /* RFC 4180: quote fields containing comma, quote, or CR/LF; double quotes. */
  function csvField(v) {
    if (v === null || v === undefined) return '';
    const s = String(v);
    return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }

  const CSV_COLUMNS = ['risk', 'pc_pred', 'pc_latest', 'miss_m', 'rel_vel_ms',
    'tca', 'primary_name', 'primary_norad', 'secondary_name', 'secondary_norad',
    'cdms', 'status'];

  function csvRow(it) {
    return [
      it.risk_level,
      it.predicted_final_risk,
      it.pc_latest,
      it.miss_distance_m,
      it.relative_velocity_ms,
      it.tca,
      it.primary ? it.primary.name : null,
      it.primary ? it.primary.norad_id : null,
      it.secondary ? it.secondary.name : null,
      it.secondary ? it.secondary.norad_id : null,
      it.cdm_count,
      it.status,
    ].map(csvField).join(',');
  }

  /* Serializes currentView - exactly the rows on screen (post-filter/sort). */
  function onExportCsv() {
    if (!hasData) return;
    const lines = [CSV_COLUMNS.join(',')];
    for (const it of currentView) lines.push(csvRow(it));
    const stamp = new Date().toISOString()
      .replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');
    const blob = new Blob([lines.join('\r\n') + '\r\n'],
      { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `kessler-conjunctions-${stamp}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function renderSortIndicators() {
    for (const th of els.conjThead.querySelectorAll('th[data-sort]')) {
      const on = th.dataset.sort === sortKey;
      th.classList.toggle('is-sorted', on);
      if (on) th.setAttribute('aria-sort', sortDir === 'desc' ? 'descending' : 'ascending');
      else th.removeAttribute('aria-sort');
      const ind = th.querySelector('.sort-ind');
      if (ind) ind.textContent = on ? (sortDir === 'desc' ? '↓' : '↑') : '';
    }
  }

  /* desc -> asc -> back to API order; a new column resets the others */
  function onSortClick(ev) {
    const th = ev.target.closest('th[data-sort]');
    if (!th) return;
    const key = th.dataset.sort;
    if (sortKey !== key) { sortKey = key; sortDir = 'desc'; }
    else if (sortDir === 'desc') sortDir = 'asc';
    else { sortKey = null; sortDir = 'desc'; }
    applyTableView();
  }

  /* keyboard: sortable headers - Enter/Space run the same onSortClick logic */
  function onSortKeydown(ev) {
    if (ev.key !== 'Enter' && ev.key !== ' ') return;
    if (!ev.target.closest('th[data-sort]')) return;
    if (ev.key === ' ') ev.preventDefault(); /* Space must not scroll the page */
    onSortClick(ev);
  }

  /* stat-tile shortcut: point the risk select at a level. The dispatched
     change event is the SINGLE path - it keeps the custom dropdown's label
     in sync (a bare .value write updates neither the facade nor the view). */
  function setRiskFilter(level) {
    els.conjRisk.value = level;
    els.conjRisk.dispatchEvent(new Event('change', { bubbles: true }));
    const panel = els.tbody.closest('.panel');
    if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  /* the visible risk control is a segmented control mirroring the select */
  function syncRiskSeg() {
    if (!els.conjRiskSeg) return;
    for (const b of els.conjRiskSeg.querySelectorAll('button[data-risk]')) {
      setPressed(b, b.dataset.risk === (els.conjRisk.value || ''));
    }
  }

  /* fetch the active risk level server-side once per selection, so a level
     with zero rows in the loaded risk-desc page never shows a lying empty
     state (nominal always did; elevated once the elevated+critical count
     topped the 1000-row poll cap) */
  async function ensureRiskPage() {
    const level = tableRisk;
    if (!level || riskFetching === level) return;
    if (riskFetched && riskFetched.level === level) return;
    riskFetching = level;
    try {
      const page = await API.conjunctions({ risk: level, limit: 1000 });
      if (tableRisk !== level) return; /* filter changed while in flight */
      riskFetched = {
        level,
        items: Array.isArray(page.items) ? page.items : [],
        count: page.count != null ? page.count : null,
      };
      applyTableView();
    } catch (_) {
      /* keep the client-filtered view - partial but never wrong */
    } finally {
      if (riskFetching === level) riskFetching = null;
      applyTableView();
    }
  }

  /* --- drawer -------------------------------------------------------------- */

  function buildDrawerRow() {
    const tr = document.createElement('tr');
    tr.className = 'drawer';
    const td = document.createElement('td');
    td.colSpan = 10;
    /* the content box is pinned to the scroller's visible width (sticky
       left + --drawer-w) so on a phone it never scrolls away sideways with
       the wide table */
    td.innerHTML = '<div class="drawer-inner"><div class="drawer-loading">Loading messages…</div></div>';
    tr.appendChild(td);
    syncDrawerWidth();
    return tr;
  }

  function syncDrawerWidth() {
    const w = els.tableWrap ? els.tableWrap.clientWidth : 0;
    if (w > 0) els.tableWrap.style.setProperty('--drawer-w', `${w}px`);
  }

  function closeDrawer() {
    expandedId = null;
    const d = els.tbody.querySelector('tr.drawer');
    if (d) d.remove();
    const ex = els.tbody.querySelector('tr.row--expanded');
    if (ex) ex.classList.remove('row--expanded');
  }

  function toggleDrawer(id, anchorTr) {
    if (expandedId === id) { closeDrawer(); return; }
    closeDrawer();
    expandedId = id;
    anchorTr.classList.add('row--expanded');
    const drawer = buildDrawerRow();
    anchorTr.after(drawer);
    populateDrawer(id, drawer);
  }

  async function populateDrawer(id, drawerTr) {
    const td = drawerTr.querySelector('.drawer-inner') || drawerTr.firstElementChild;
    let detail = drawerCache.get(id);
    if (!detail) {
      try {
        detail = await API.conjunction(id);
        drawerCache.set(id, detail);
      } catch (err) {
        if (expandedId === id && drawerTr.isConnected) {
          td.innerHTML =
            `<div class="drawer-loading">Couldn’t load the messages for this approach (${FMT.esc(err.message)}).</div>`;
        }
        return;
      }
    }
    if (expandedId !== id || !drawerTr.isConnected) return;
    renderDrawerContent(td, detail);
  }

  function renderDrawerContent(td, detail) {
    const cdms = Array.isArray(detail.cdms) ? detail.cdms : [];
    const cdmRows = cdms.map((c) =>
      '<tr>' +
      `<td class="dim" title="${FMT.esc(FMT.utc(c.message_time))}">${FMT.esc(shortUtc(c.message_time))}</td>` +
      `<td class="num">${unitHtml(FMT.dist(c.miss_distance_m))}</td>` +
      `<td class="num">${unitHtml(FMT.vel(c.relative_velocity_ms))}</td>` +
      `<td class="num" title="Pc ${FMT.esc(FMT.pc(c.probability_of_collision))}">` +
        `${FMT.esc(odds(c.probability_of_collision))} ` +
        `<span class="faint">${FMT.esc(FMT.pc(c.probability_of_collision))}</span></td>` +
      `<td><span class="chip chip--type">${FMT.esc(String(c.source || '—'))}</span></td>` +
      '</tr>').join('');

    const detailLink = detail.primary
      ? `<a class="drawer-link" href="object.html?id=${encodeURIComponent(String(detail.primary.norad_id))}` +
        `&conj=${encodeURIComponent(String(detail.id))}">Object details</a>`
      : '';

    /* hand the pair to the globe's encounter mode - rendered only when
       the engine is up and exposes showEncounter */
    const globeBtn = (canShowEncounter() && detail.primary && detail.secondary)
      ? '<button type="button" class="drawer-link drawer-link--primary drawer-globe-btn">Show on globe</button>'
      : '';

    /* encounter simulator launch - rendered only when the KSim module
       is loaded (typeof-guarded), hidden when absent. */
    const simBtn = (typeof KSim !== 'undefined' && typeof KSim.open === 'function')
      ? '<button type="button" class="drawer-link drawer-sim-btn">Replay the encounter</button>'
      : '';

    td.innerHTML =
      '<div class="drawer-grid">' +
        '<div class="drawer-col">' +
          '<div class="drawer-label" title="Conjunction data messages (CDMs)">Messages ' +
            `<span class="faint">${FMT.esc(FMT.num(cdms.length))}</span></div>` +
          '<div class="tbl-scroll"><table class="tbl tbl--mini">' +
            '<thead><tr>' +
              '<th>Received</th>' +
              '<th class="num">Miss distance</th>' +
              '<th class="num">Speed</th>' +
              '<th class="num">Collision chance</th>' +
              '<th>Source</th>' +
            '</tr></thead>' +
            `<tbody>${cdmRows || '<tr><td colspan="5" class="empty">No messages recorded yet.</td></tr>'}</tbody>` +
          '</table></div>' +
          /* glass-box risk block - stays hidden unless /risk/{id}/explain
             answers with an explainable mode (populateWhy fills it async) */
          '<div class="why-block" data-why hidden></div>' +
        '</div>' +
        '<div class="drawer-col drawer-charts">' +
          '<div class="drawer-chart">' +
            '<div class="drawer-label" title="Probability of collision (Pc), log scale">Collision chance over time</div>' +
            '<div class="chart-box" data-chart="pc"><canvas class="mini-chart"></canvas><div class="chart-tip"></div></div>' +
          '</div>' +
          '<div class="drawer-chart">' +
            '<div class="drawer-label">Miss distance over time</div>' +
            '<div class="chart-box" data-chart="md"><canvas class="mini-chart"></canvas><div class="chart-tip"></div></div>' +
          '</div>' +
        '</div>' +
      '</div>' +
      stormNote(detail) +
      `<div class="drawer-foot">${detailLink}${globeBtn}${simBtn}</div>`;

    const showBtn = td.querySelector('.drawer-globe-btn');
    if (showBtn) {
      showBtn.addEventListener('click', () => {
        showEncounterOnGlobe(
          detail.primary.norad_id, detail.secondary.norad_id, detail.tca, detail);
      });
    }

    const simLaunch = td.querySelector('.drawer-sim-btn');
    if (simLaunch) {
      simLaunch.addEventListener('click', () => {
        try { KSim.open({ conjunctionId: detail.id }); } catch (_) { /* module fault */ }
      });
    }

    drawDrawerCharts(td, cdms);
    populateWhy(detail.id, td.querySelector('[data-why]'));
  }

  /* --- WHY THIS NUMBER (glass-box risk, GET /risk/{id}/explain) -------------- */

  /* js/api.js is a frozen shared foundation - same direct-fetch idiom as
     /changes. Failures of any kind (404 = unknown id or pre-restart backend
     without the route, network faults, 'unavailable' mode) leave the block
     hidden: the drawer renders exactly as it did before. */
  async function populateWhy(id, box) {
    if (!box) return;
    let data = whyCache.get(id);
    if (data === undefined) {
      try {
        const res = await fetch(`/risk/${encodeURIComponent(String(id))}/explain`);
        if (!res.ok) { whyCache.set(id, null); return; }
        data = await res.json();
      } catch (_) { return; /* network fault - omit silently, retry on next open */ }
      whyCache.set(id, data && typeof data === 'object' ? data : null);
    }
    if (!data || expandedId !== id || !box.isConnected) return;
    renderWhy(box, data);
  }

  /* One bar: |contribution| scaled against the block's max, signed value
     %+.2f (contribution units: log10-Pc residual per the explain contract). */
  function whyBarRow(c, maxAbs) {
    const v = Number(c.contribution);
    const up = v >= 0;
    const w = Math.max(2, Math.round((Math.abs(v) / maxAbs) * 100));
    const signed = `${up ? '+' : '−'}${Math.abs(v).toFixed(2)}`;
    return '<div class="why-row">' +
      `<span class="why-name" title="${FMT.esc(String(c.name || '—'))}">${FMT.esc(String(c.name || '—'))}</span>` +
      `<span class="why-track"><span class="why-bar${up ? ' why-bar--up' : ''}"` +
        ` style="width:${w}%"></span></span>` +
      `<span class="why-val">${signed}</span>` +
      '</div>';
  }

  /* carry_forward -> the model's one-line deference note; learned_residual ->
     top 5 |contribution| SHAP bars (the API pre-sorts desc); any other mode
     (unavailable / unknown future values) keeps the block hidden. */
  function renderWhy(box, data) {
    const mode = data.mode;
    if (mode !== 'carry_forward' && mode !== 'learned_residual') return;
    const model = String(data.model_version || '');
    let html = '<div class="drawer-label">Why this number' +
      (model ? ` <span class="faint">· ${FMT.esc(model)}</span>` : '') + '</div>';
    if (mode === 'carry_forward') {
      const note = String(data.note ||
        'This event is already above the high-risk line, so VIGIL keeps the latest observed probability rather than adjusting it.');
      html += `<div class="why-note">${FMT.esc(note)}</div>`;
    } else {
      const rows = (Array.isArray(data.contributions) ? data.contributions : [])
        .filter((c) => c && Number.isFinite(Number(c.contribution)))
        .slice(0, 5);
      if (!rows.length) return;
      const maxAbs = Math.max(...rows.map((c) => Math.abs(Number(c.contribution)))) || 1;
      html += rows.map((c) => whyBarRow(c, maxAbs)).join('') +
        '<div class="why-units">What moved the prediction most (SHAP contributions to the log₁₀ Pc residual). Positive values push risk up.</div>';
    }
    box.innerHTML = html;
    box.hidden = false;
  }

  function drawDrawerCharts(td, cdms) {
    const pcPoints = cdms
      .filter((c) => typeof c.probability_of_collision === 'number' && c.probability_of_collision > 0)
      .map((c) => ({ t: Date.parse(c.message_time), y: c.probability_of_collision }));
    const mdPoints = cdms
      .filter((c) => typeof c.miss_distance_m === 'number')
      .map((c) => ({ t: Date.parse(c.message_time), y: c.miss_distance_m }));

    renderMiniChart(td.querySelector('.chart-box[data-chart="pc"]'), 'Pc', pcPoints, {
      yScale: 'log', yFormat: FMT.pc, color: tok('--k-data-1'),
    });
    renderMiniChart(td.querySelector('.chart-box[data-chart="md"]'), 'Miss distance', mdPoints, {
      yScale: 'linear', yFormat: FMT.dist, color: tok('--k-data-2'),
    });
  }

  function renderMiniChart(box, label, points, cfg) {
    if (!box) return;
    if (!points.length) {
      box.innerHTML = '<div class="chart-nodata">Not enough messages to chart yet.</div>';
      return;
    }
    /* charts.js is loaded as a separate script; show a visible fallback if
       a load-order regression ever leaves the global undefined. */
    if (typeof drawLineChart !== 'function') {
      box.innerHTML = '<div class="chart-nodata">Chart unavailable.</div>';
      return;
    }
    drawLineChart(box.querySelector('canvas'), box.querySelector('.chart-tip'), {
      series: [{ label, color: cfg.color, points }],
      yScale: cfg.yScale,
      yFormat: cfg.yFormat,
      xTicks: 4,
      yTicks: 4,
    });
  }

  /* --- errors -------------------------------------------------------------- */

  function showError(err) {
    els.errorBanner.textContent = `Can’t reach the Kessler server (${err.message}).` +
      (hasData ? ' Showing the last data received.' : '');
    els.errorBanner.hidden = false;
    if (!hasData) {
      els.tbody.replaceChildren();
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 10;
      td.className = 'empty';
      td.textContent = 'Close approach data is unavailable right now.';
      tr.appendChild(td);
      els.tbody.appendChild(tr);
    }
  }

  function clearError() {
    els.errorBanner.hidden = true;
    els.errorBanner.textContent = '';
  }

  /* --- situational display (globe) ------------------------------------------ */

  const TYPE_NAMES = ['PAYLOAD', 'DEBRIS', 'ROCKET BODY', 'UNKNOWN'];
  const TYPE_WORDS = {
    PAYLOAD: 'Payload', DEBRIS: 'Debris', 'ROCKET BODY': 'Rocket body', RB: 'Rocket body',
    'R-B': 'Rocket body', UNKNOWN: 'Unknown type',
  };

  function typeName(t) {
    if (typeof t === 'number') return TYPE_NAMES[t] || 'UNKNOWN';
    if (t === null || t === undefined || t === '') return 'UNKNOWN';
    return String(t).toUpperCase();
  }

  function typeWord(t) {
    return TYPE_WORDS[typeName(t)] || 'Unknown type';
  }

  function showGlobeFallback() {
    globeReady = false;
    const d = document.createElement('div');
    d.className = 'chart-nodata chart-nodata--globe';
    d.textContent = 'The globe can’t be shown in this browser.';
    els.globeWrap.replaceChildren(d);
    els.globeCount.textContent = '';
    els.globeSim.textContent = '';
    for (const b of els.globeSpeed.querySelectorAll('button')) b.disabled = true;
    if (els.globeNow) els.globeNow.disabled = true;
    els.globeSearch.disabled = true;
    /* mode group + HEAT live inside .globe-filters - this loop covers them */
    for (const b of els.globeFilters.querySelectorAll('button')) b.disabled = true;
    els.timeScrub.hidden = true;
  }

  /* engine surface is built in parallel - controls whose KGlobe methods
     are missing stay visibly disabled instead of silently doing nothing. */
  function gateV6GlobeControls() {
    const hasModes = globeReady && typeof KGlobe.setColorMode === 'function';
    for (const b of els.globeModeGroup.querySelectorAll('button[data-mode]')) {
      b.disabled = !hasModes;
    }
    els.globeHeat.disabled = !(globeReady && typeof KGlobe.setHeatmap === 'function');
    els.timeScrub.hidden = !(globeReady && typeof KGlobe.setSimTime === 'function');
    /* SURFACE toggle - engine surface builds in parallel */
    els.globeSurface.disabled = !(globeReady && typeof KGlobe.setSurface === 'function');
    setPressed(els.globeSurface, surfaceOn);
  }

  function initGlobe() {
    /* never leave a dead blank canvas if the globe engine is absent or its
       init throws. */
    if (typeof KGlobe === 'undefined') { showGlobeFallback(); return; }
    try {
      KGlobe.init({
        canvas: els.globeCanvas,
        tipEl: els.globeTip,
        onSelect: renderSelected,
        /* the hero sits in a scrolling page: plain wheel scrolls the page,
           ctrl/⌘ + wheel (and trackpad pinch) zooms */
        wheelZoom: 'modifier',
        onWheelHint: showWheelHint,
      });
      KGlobe.setSpeed(60); /* contract default x60 - keep button state in sync */
    } catch (err) {
      showGlobeFallback();
      return;
    }
    globeReady = true;
    els.globeSpeed.addEventListener('click', onSpeedClick);
    if (els.globeNow) els.globeNow.addEventListener('click', onSpeedClick);
    gateV6GlobeControls();
    layoutStage();
    setTimeout(checkGlassBudget, 5000);
    setInterval(tickSim, 500);
    tickSim();
  }

  /* --- stage layout, star field, zoom ---------------------------------------- */

  /* Keep --dock-h in sync with the real dock and tell the engine which part
     of the canvas is uncovered, so the Earth centers between the floating
     panels (desktop). Narrow layouts stack the UI below - no inset. */
  function layoutStage() {
    if (!els.stage || !els.stageDock) return;
    const floating = getComputedStyle(els.stageDock).position === 'absolute';
    if (floating) els.stage.style.setProperty('--dock-h', `${els.stageDock.offsetHeight}px`);
    else els.stage.style.removeProperty('--dock-h');
    if (!globeReady || typeof KGlobe.setViewInset !== 'function') return;
    if (!floating) {
      /* stacked (phone) layout: frame the Earth above the legend caption so
         the dots never run under its text */
      const cap = els.globeWrap.querySelector('.stage__caption');
      const capH = cap ? cap.offsetHeight + 18 : 0;
      KGlobe.setViewInset({ top: 0, right: 0, bottom: capH, left: 0 });
      return;
    }
    const W = els.stage.clientWidth;
    const H = els.stage.clientHeight;
    const gap = 28;
    const bottom = els.stageDock.offsetHeight + gap;
    let left = els.stageLeft ? els.stageLeft.offsetWidth + gap : 0;
    let right = (els.stageSide && !els.stageSide.hidden) ? els.stageSide.offsetWidth + gap : 0;
    const vh = H - bottom;
    /* prefer the window between both panels; fall back to centering in the
       whole width when that window gets too narrow for a good-size Earth */
    if (W - left - right < vh * 0.9) { left = 0; right = 0; }
    KGlobe.setViewInset({ top: 0, right, bottom, left });
  }

  /* A calm, static star field painted once per size (seeded, so it never
     shimmers between resizes). Pure decoration under the globe canvas. */
  function drawStars() {
    const c = els.globeStars;
    if (!c) return;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    let seed = 1337;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
    const n = Math.round((w * h) / 2300);
    for (let i = 0; i < n; i++) {
      const x = rnd() * w, y = rnd() * h;
      const k = rnd();
      const r = k > 0.985 ? 1.1 : k > 0.9 ? 0.8 : 0.5;
      const a = k > 0.985 ? 0.85 : 0.12 + rnd() * 0.42;
      const tint = rnd();
      g.fillStyle = tint > 0.86 ? `rgba(200, 216, 255, ${a})`
        : tint > 0.76 ? `rgba(255, 236, 214, ${a})` : `rgba(255, 255, 255, ${a})`;
      g.beginPath();
      g.arc(x, y, r, 0, Math.PI * 2);
      g.fill();
    }
  }

  /* Frosted glass over a live canvas is re-blurred every frame. On machines
     without GPU compositing that can halve the globe's frame rate, so after
     the scene settles we sample the frame rate once; if it is poor the stage
     switches to solid (unblurred) panels. Capable machines keep the glass. */
  function checkGlassBudget() {
    if (!els.stage || !globeReady) return;
    let n = 0;
    const t0 = performance.now();
    const step = (t) => {
      n++;
      if (t - t0 < 1500) { requestAnimationFrame(step); return; }
      const fps = n / ((t - t0) / 1000);
      if (fps < 45 && !document.hidden) els.stage.classList.add('stage--lite');
    };
    requestAnimationFrame(step);
  }

  let hintTimer = null;
  let hintShownAt = 0;
  function showWheelHint() {
    if (!els.stageHint) return;
    const now = Date.now();
    if (now - hintShownAt < 15000 && !els.stageHint.classList.contains('is-on')) return;
    if (!els.stageHint.classList.contains('is-on')) hintShownAt = now;
    els.stageHint.classList.add('is-on');
    clearTimeout(hintTimer);
    hintTimer = setTimeout(() => els.stageHint.classList.remove('is-on'), 1100);
  }

  function onZoomClick(ev) {
    const b = ev.target.closest('button[data-zoom]');
    if (!b || !globeReady || typeof KGlobe.zoomBy !== 'function') return;
    KGlobe.zoomBy(b.dataset.zoom === 'in' ? 1.4 : 1 / 1.4);
  }

  function tickSim() {
    if (!globeReady) return;
    let st = null;
    try { st = KGlobe.status(); } catch (_) { /* engine still loading its snapshot */ }
    if (!st) return;
    let countTxt = '';
    if (st.count) {
      const filterOn = riskOnly || GLOBE_TYPES.some((t) => !typesOn[t]);
      countTxt = (filterOn && st.visible != null)
        ? `Showing ${FMT.num(st.visible)} of ${FMT.num(st.count)} objects`
        : `${FMT.num(st.count)} objects in orbit`;
    }
    if (els.globeCount.textContent !== countTxt) els.globeCount.textContent = countTxt;
    renderClock(st);
    /* live scrub cursor from sim time (a drag in flight owns the cursor) */
    if (!scrubbing && !els.timeScrub.hidden && st.simMs != null) {
      const frac = 0.5 + (st.simMs - Date.now()) / (2 * SCRUB_RANGE_MS);
      setScrubCursor(Math.max(0, Math.min(1, frac)));
    }
  }

  /* the dock's clock: simulated time plus how far it is from now, in words */
  let clockKey = '';
  function renderClock(st) {
    if (st.simMs == null) {
      if (clockKey !== 'loading') {
        clockKey = 'loading';
        els.globeSim.innerHTML = '<span class="dock-clock__time">—</span>' +
          '<span class="dock-clock__rel">Loading orbits…</span>';
      }
      return;
    }
    const d = new Date(st.simMs);
    const time = `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}:${pad2(d.getUTCSeconds())} UTC`;
    const diff = st.simMs - Date.now();
    const live = Math.abs(diff) < 90000;
    let rel;
    if (live) rel = st.speed === 1 ? 'Live' : 'Now';
    else rel = diff > 0 ? `${durText(diff)} ahead` : `${durText(diff)} ago`;
    if (st.speed && st.speed !== 1) rel += ` · ${st.speed}×`;
    const key = `${time}|${rel}`;
    if (key === clockKey) return;
    clockKey = key;
    els.globeSim.innerHTML = `<span class="dock-clock__time">${FMT.esc(time)}</span>` +
      `<span class="dock-clock__rel${live && st.speed === 1 ? ' is-live' : ''}">${FMT.esc(rel)}</span>`;
    els.globeSim.title = `Simulated time ${st.simIso || ''}`;
  }

  function onSpeedClick(ev) {
    /* NOW: snap the sim clock to real time at live rate (x1) */
    const nowBtn = ev.target.closest('button[data-now]');
    if (nowBtn && globeReady) {
      if (typeof KGlobe.setSimTime === 'function') {
        try { KGlobe.setSimTime(Date.now()); } catch (_) { /* engine fault */ }
      }
      KGlobe.setSpeed(1);
      for (const b of els.globeSpeed.querySelectorAll('button[data-speed]')) {
        setPressed(b, Number(b.dataset.speed) === 1);
      }
      if (els.scrubCursor) setScrubCursor(0.5);
      return;
    }
    const btn = ev.target.closest('button[data-speed]');
    if (!btn || !globeReady) return;
    KGlobe.setSpeed(Number(btn.dataset.speed));
    for (const b of els.globeSpeed.querySelectorAll('button[data-speed]')) {
      setPressed(b, b === btn);
    }
  }

  /* --- encounter mode --------------------------------------------------------- */

  /* showEncounter is a KGlobe addition - existence-checked so the drawer
     button and ?conj= degrade to no-ops against an older engine. */
  function canShowEncounter() {
    return globeReady && typeof KGlobe.showEncounter === 'function';
  }

  /* Watch mode (sim jumps to TCA-2min at x60 inside the engine), then
     smooth-scroll the situational display panel into view. Returns success
     so callers (assistant hook) can fall back to a navigation. */
  function showEncounterOnGlobe(primaryNorad, secondaryNorad, tcaIso, ctx) {
    if (!canShowEncounter()) return false;
    encCtx = encContext(ctx, primaryNorad, secondaryNorad, tcaIso);
    try {
      KGlobe.showEncounter({
        primaryNorad: Number(primaryNorad),
        secondaryNorad: Number(secondaryNorad),
        tcaIso: tcaIso || null,
        watch: true,
      });
    } catch (_) { return false; /* engine fault */ }
    const stage = els.stage || els.globeWrap;
    if (stage.getBoundingClientRect().top < -40 || stage.getBoundingClientRect().top > 80) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    /* speed buttons: watch mode sets x60 inside the engine - mirror it */
    for (const b of els.globeSpeed.querySelectorAll('button[data-speed]')) {
      setPressed(b, Number(b.dataset.speed) === 60);
    }
    setTimeout(syncEncBar, 0);
    return true;
  }

  /* --- encounter bar: pair, Simulate, geometry note ---------------------------- */

  let encCtx = null; /* {id, pNorad, sNorad, pName, sName, missM, tca} of the shown pair */

  /* ctx: a conjunction detail/list item when the caller has one; otherwise
     the loaded table page is searched for the pair (assistant path). */
  function encContext(ctx, pNorad, sNorad, tcaIso) {
    let c = ctx;
    if (!c) {
      const p = Number(pNorad), s = Number(sNorad);
      const t = Date.parse(tcaIso);
      c = conjItems.find((it) => it.primary && it.secondary &&
        ((it.primary.norad_id === p && it.secondary.norad_id === s) ||
         (it.primary.norad_id === s && it.secondary.norad_id === p)) &&
        (!Number.isFinite(t) || Math.abs(Date.parse(it.tca) - t) < 60000)) || null;
    }
    if (!c || !c.primary || !c.secondary) {
      return { id: null, pNorad: Number(pNorad), sNorad: Number(sNorad), pName: null,
        sName: null, missM: null, tca: tcaIso || null };
    }
    return {
      id: c.id != null ? Number(c.id) : null,
      pNorad: Number(c.primary.norad_id), sNorad: Number(c.secondary.norad_id),
      pName: c.primary.name || null, sName: c.secondary.name || null,
      missM: Number.isFinite(Number(c.miss_distance_m)) ? Number(c.miss_distance_m) : null,
      tca: c.tca || tcaIso || null,
    };
  }

  function simAvailable() {
    return typeof KSim !== 'undefined' && typeof KSim.open === 'function';
  }

  /* Honest captions for the globe's encounter markers. The globe propagates
     simplified (two-body) orbits, so its markers rarely coincide; the
     refined miss comes from GET /conjunctions/{id}/ephemeris (SGP4 on the
     newest element sets, the simulator's source). Two cases:
     - the newest tracking data disagree with the event's last message (same
       spirit as the avoidance plan's geometry_mismatch: refined miss >= 10x
       the listed miss and > 5 km more) -> say so plainly;
     - the markers sit visibly apart only because of the simplified orbits
       -> say that, with the precise miss, instead of implying they meet. */
  const ephemSep = new Map(); /* conjunction id -> refined miss (m) | null */
  function loadRefSep(id) {
    if (id == null || ephemSep.has(id)) return;
    ephemSep.set(id, undefined);
    fetch(`/conjunctions/${encodeURIComponent(String(id))}/ephemeris?window_s=60`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        const v = d && d.tca_state ? Number(d.tca_state.sep_m) : NaN;
        ephemSep.set(id, Number.isFinite(v) ? v : null);
      })
      .catch(() => ephemSep.set(id, null))
      .finally(() => { if (encCtx && encCtx.id === id) syncEncBar(); });
  }

  function geometryNote(globeSepKm, missM, refM) {
    if (refM != null && missM != null && refM >= 10 * missM && refM - missM > 5000) {
      return `The newest tracking data put them ${FMT.dist(refM)} apart at closest approach, ` +
        `not the ${FMT.dist(missM)} in the last message. The data have changed since that message.`;
    }
    if (!Number.isFinite(globeSepKm) || globeSepKm < 100) return '';
    const precise = refM != null ? `The refined miss is ${FMT.dist(refM)}.`
      : missM != null ? `The listed miss is ${FMT.dist(missM)}.` : '';
    return `The globe draws simplified orbits, so the markers sit about ${FMT.km(sepRound(globeSepKm), 0)} ` +
      `apart here. ${precise}`.trim();
  }
  function sepRound(km) {
    const mag = Math.pow(10, Math.max(0, Math.floor(Math.log10(km)) - 1));
    return Math.round(km / mag) * mag;
  }

  function syncEncBar() {
    if (!els.encBar) return;
    let st = null;
    try { st = globeReady && KGlobe.status ? KGlobe.status().encounter : null; } catch (_) { st = null; }
    const c = encCtx;
    const same = st && c && ((st.primaryNorad === c.pNorad && st.secondaryNorad === c.sNorad) ||
      (st.primaryNorad === c.sNorad && st.secondaryNorad === c.pNorad));
    if (!same) {
      /* encCtx survives a queued encounter (engine still loading); a new
         showEncounterOnGlobe always replaces it */
      els.encBar.hidden = true;
      return;
    }
    const pName = c.pName || `NORAD ${c.pNorad}`;
    const sName = c.sName || `NORAD ${c.sNorad}`;
    els.encPair.innerHTML = `<span>${FMT.esc(pName)}</span>` +
      '<span class="enc-bar__and"> and </span>' + `<span>${FMT.esc(sName)}</span>`;
    const bits = [];
    if (c.tca) bits.push(FMT.when(c.tca, { day: false }));
    if (c.missM != null) bits.push(`listed miss ${FMT.dist(c.missM)}`);
    els.encSub.textContent = bits.join(' · ');
    loadRefSep(c.id);
    const ref = c.id != null ? ephemSep.get(c.id) : null;
    const note = ref === undefined && c.id != null ? '' : geometryNote(st.sepKm, c.missM, ref);
    els.encNote.textContent = note;
    els.encNote.hidden = !note;
    els.encSim.hidden = !(c.id != null && simAvailable());
    els.encBar.hidden = false;
  }

  function onEncSim() {
    if (!encCtx || encCtx.id == null || !simAvailable()) return;
    try { KSim.open({ conjunctionId: encCtx.id }); } catch (_) { /* module fault */ }
  }

  /* --- globe type / risk filters --------------------------------------------- */

  /* setFilter is a KGlobe addition - existence-checked so an older engine
     leaves the toggles inert instead of throwing. */
  function pushGlobeFilter() {
    if (!globeReady || typeof KGlobe.setFilter !== 'function') return;
    try {
      KGlobe.setFilter({ types: GLOBE_TYPES.filter((t) => typesOn[t]), riskOnly });
    } catch (_) { /* engine fault - filters stay cosmetic */ }
  }

  function onTypeToggle(ev) {
    const btn = ev.target.closest('button[data-type]');
    if (!btn || !globeReady) return;
    const t = btn.dataset.type;
    typesOn[t] = !typesOn[t];
    setPressed(btn, typesOn[t]);
    pushGlobeFilter();
  }

  function onRiskOnlyToggle() {
    if (!globeReady) return;
    riskOnly = !riskOnly;
    setPressed(els.globeRiskOnly, riskOnly);
    pushGlobeFilter();
  }

  /* --- globe color modes + heatmap + legend swap ------------------------------- */

  /* The static markup (risk mode's dot key) is captured once so switching
     back from an engine-described legend restores it byte-identically. */
  const staticLegendHtml = els.globeLegend.innerHTML;

  /* mode != risk: render from the KGlobe.legend() descriptor
     {mode, entries:[{label, color}]}; any missing/faulty descriptor keeps
     the static legend (risk styling still overrides in every mode). */
  function renderGlobeLegend() {
    if (colorMode === 'type' || !globeReady || typeof KGlobe.legend !== 'function') {
      els.globeLegend.innerHTML = staticLegendHtml;
      return;
    }
    let desc = null;
    try { desc = KGlobe.legend(); } catch (_) { /* engine fault */ }
    const entries = desc && Array.isArray(desc.entries) ? desc.entries : [];
    if (!entries.length) {
      els.globeLegend.innerHTML = staticLegendHtml;
      return;
    }
    els.globeLegend.innerHTML = entries.map((e) =>
      `<span class="key" style="--key-color:${FMT.esc(String(e.color || ''))}">` +
        `${FMT.esc(String(e.label || ''))}</span>`).join('');
  }

  function onModeClick(ev) {
    const btn = ev.target.closest('button[data-mode]');
    if (!btn || btn.disabled || !globeReady ||
        typeof KGlobe.setColorMode !== 'function') return;
    const mode = btn.dataset.mode;
    if (mode === colorMode) return;
    try { KGlobe.setColorMode(mode); } catch (_) { return; /* engine fault */ }
    colorMode = mode;
    for (const b of els.globeModeGroup.querySelectorAll('button[data-mode]')) {
      setPressed(b, b === btn);
    }
    renderGlobeLegend();
  }

  function onHeatToggle() {
    if (els.globeHeat.disabled || !globeReady ||
        typeof KGlobe.setHeatmap !== 'function') return;
    try { KGlobe.setHeatmap(!heatOn); } catch (_) { return; /* engine fault */ }
    heatOn = !heatOn;
    setPressed(els.globeHeat, heatOn);
  }

  /* SURFACE|WIREFRAME toggle - the engine persists the preference in
     localStorage 'kessler.surface' inside setSurface. */
  function onSurfaceToggle() {
    if (els.globeSurface.disabled || !globeReady ||
        typeof KGlobe.setSurface !== 'function') return;
    try { KGlobe.setSurface(!surfaceOn); } catch (_) { return; /* engine fault */ }
    surfaceOn = !surfaceOn;
    setPressed(els.globeSurface, surfaceOn);
  }

  /* --- time-scrub strip (±24 h, drag re-anchors sim time) ---------------------- */

  function scrubFracToMs(frac) {
    return Date.now() + (frac - 0.5) * 2 * SCRUB_RANGE_MS;
  }

  /* cursor position and the slider's aria-valuenow (hours, ±24) move together */
  function setScrubCursor(frac) {
    els.scrubCursor.style.left = `${(frac * 100).toFixed(2)}%`;
    if (els.scrubFill) {
      els.scrubFill.style.left = `${(Math.min(frac, 0.5) * 100).toFixed(2)}%`;
      els.scrubFill.style.width = `${(Math.abs(frac - 0.5) * 100).toFixed(2)}%`;
    }
    const hours = (frac - 0.5) * 48;
    els.scrubTrack.setAttribute('aria-valuenow', hours.toFixed(2));
    els.scrubTrack.setAttribute('aria-valuetext', Math.abs(hours) < 0.05 ? 'Now'
      : `${durText(Math.abs(hours) * 3600000)} ${hours > 0 ? 'ahead' : 'ago'}`);
  }

  function applyScrub(ev) {
    const rect = els.scrubTrack.getBoundingClientRect();
    if (!rect.width) return;
    const frac = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    setScrubCursor(frac);
    try { KGlobe.setSimTime(scrubFracToMs(frac)); } catch (_) { /* engine fault */ }
  }

  function onScrubDown(ev) {
    if (!globeReady || typeof KGlobe.setSimTime !== 'function') return;
    scrubbing = true;
    els.scrubTrack.classList.add('is-dragging');
    try { els.scrubTrack.setPointerCapture(ev.pointerId); } catch (_) { /* older engines */ }
    applyScrub(ev);
  }

  function onScrubMove(ev) {
    if (scrubbing) applyScrub(ev);
  }

  function onScrubUp() {
    scrubbing = false;
    els.scrubTrack.classList.remove('is-dragging');
  }

  /* double-click: snap the sim clock back to real now */
  function onScrubDblClick() {
    if (!globeReady || typeof KGlobe.setSimTime !== 'function') return;
    try { KGlobe.setSimTime(Date.now()); } catch (_) { /* engine fault */ }
    setScrubCursor(0.5);
  }

  /* keyboard scrub: <-/-> = ±15 min of sim time, Shift steps ±2 h */
  function onScrubKeydown(ev) {
    if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
    if (!globeReady || typeof KGlobe.setSimTime !== 'function') return;
    ev.preventDefault();
    let st = null;
    try { st = KGlobe.status(); } catch (_) { /* engine still loading its snapshot */ }
    const cur = (st && st.simMs != null) ? st.simMs : Date.now();
    const step = (ev.shiftKey ? 2 * 3600 * 1000 : 15 * 60 * 1000) *
      (ev.key === 'ArrowRight' ? 1 : -1);
    const frac = Math.max(0, Math.min(1,
      0.5 + (cur + step - Date.now()) / (2 * SCRUB_RANGE_MS)));
    setScrubCursor(frac);
    try { KGlobe.setSimTime(scrubFracToMs(frac)); } catch (_) { /* engine fault */ }
  }

  /* --- globe typeahead search ------------------------------------------------- */

  function globeList() {
    if (!globeReady || typeof KGlobe.list !== 'function') return [];
    try {
      const l = KGlobe.list();
      return Array.isArray(l) ? l : [];
    } catch (_) { return []; }
  }

  function taShortType(t) {
    return typeWord(t);
  }

  function taClose() {
    taMatches = [];
    taActive = -1;
    els.globeTypeahead.hidden = true;
    els.globeTypeahead.innerHTML = '';
  }

  function taRender() {
    if (!taMatches.length) {
      const q = els.globeSearch.value.trim();
      if (q && globeList().length) {
        els.globeTypeahead.innerHTML =
          `<div class="ta-none">Nothing matches “${FMT.esc(q)}”.</div>`;
        els.globeTypeahead.hidden = false;
      } else taClose();
      return;
    }
    els.globeTypeahead.innerHTML = taMatches.map((o, i) =>
      `<button type="button" class="ta-item${i === taActive ? ' is-active' : ''}"` +
        ` data-norad="${FMT.esc(String(o.norad))}">` +
        `<span class="ta-name">${FMT.esc(o.name)}</span>` +
        `<span class="ta-norad">${FMT.esc(String(o.norad))}</span>` +
        `<span class="ta-type">${FMT.esc(taShortType(o.type))}</span>` +
      '</button>').join('');
    els.globeTypeahead.hidden = false;
  }

  /* name substring or norad prefix, case-insensitive, max 8 matches */
  function taCompute() {
    const q = els.globeSearch.value.trim().toLowerCase();
    if (!q) { taClose(); return; }
    const out = [];
    for (const o of globeList()) {
      if (String(o.name || '').toLowerCase().includes(q) ||
          String(o.norad).startsWith(q)) {
        out.push(o);
        if (out.length >= 8) break;
      }
    }
    taMatches = out;
    taActive = -1;
    taRender();
  }

  function taSelect(norad) {
    if (globeReady) {
      try { KGlobe.select(Number(norad)); } catch (_) { /* engine fault */ }
    }
    els.globeSearch.value = '';
    taClose();
    els.globeSearch.blur();
  }

  function onTaKeydown(ev) {
    if (ev.key === 'Escape') { taClose(); return; }
    if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
      if (!taMatches.length) taCompute();
      if (!taMatches.length) return;
      ev.preventDefault();
      const n = taMatches.length;
      taActive = ev.key === 'ArrowDown'
        ? (taActive + 1) % n
        : (taActive - 1 + n) % n;
      taRender();
      return;
    }
    if (ev.key === 'Enter') {
      if (!taMatches.length) return;
      ev.preventDefault();
      taSelect(taMatches[taActive >= 0 ? taActive : 0].norad);
    }
  }

  /* pointerdown (not click) + preventDefault: select before the input can blur */
  function onTaPointerDown(ev) {
    const btn = ev.target.closest('.ta-item');
    if (!btn) return;
    ev.preventDefault();
    taSelect(btn.dataset.norad);
  }

  /* --- selected object panel ------------------------------------------------- */

  /* zoom needs Ctrl/⌘ + wheel (a plain wheel scrolls the page) or a pinch */
  const COARSE = !!(window.matchMedia && window.matchMedia('(pointer: coarse)').matches);
  const SEL_EMPTY_HTML =
    '<div class="sel-empty">' +
      '<p>Select any object to see its orbit and close approaches.</p>' +
      `<p class="sel-empty__hint">${COARSE
        ? 'Drag to rotate · pinch to zoom'
        : 'Drag to rotate · hold Ctrl or ⌘ and scroll to zoom'}</p>` +
    '</div>';

  function selStat(label, valueStr, title) {
    return `<div class="sel-stat"${title ? ` title="${FMT.esc(title)}"` : ''}>` +
      `<div class="sel-stat__label">${label}</div>` +
      `<div class="sel-stat__value">${unitHtml(valueStr)}</div></div>`;
  }

  /* Loaded close approaches involving this norad (API order = risk desc,
     TCA asc). Click shows that encounter on the globe. */
  function selConjRows(norad) {
    const rows = [];
    for (const it of conjItems) {
      let partner = null;
      if (it.primary && it.primary.norad_id === norad) partner = it.secondary;
      else if (it.secondary && it.secondary.norad_id === norad) partner = it.primary;
      if (!partner) continue;
      rows.push({ it, partner });
    }
    return rows;
  }

  /* Honest "none": the loaded page is the riskiest slice of all approaches,
     so an empty list only proves "no elevated/critical" when the page
     reaches into nominal rows (or holds everything). */
  function selNoneText() {
    if (conjItems.length >= conjTotal) return 'No close approaches in the next 72 hours.';
    if (conjItems.some((it) => safeLevel(it.risk_level) === 'nominal')) {
      return 'No elevated or critical close approaches in the next 72 hours.';
    }
    return 'None among the riskiest close approaches loaded.';
  }

  const SEL_MAX_ROWS = 6;

  function renderSelected(sel) {
    setTimeout(syncEncBar, 0); /* encounter overlay arms right after its onSelect */
    const changed = (lastSel ? lastSel.norad : null) !== (sel ? sel.norad : null);
    lastSel = sel || null;
    if (els.selClear) els.selClear.hidden = !sel;
    if (!sel) {
      els.selBody.innerHTML = SEL_EMPTY_HTML;
      return;
    }
    const elems = sel.elements || {};
    const noradStr = String(sel.norad);
    const href = `object.html?id=${encodeURIComponent(noradStr)}`;
    const rows = selConjRows(sel.norad);
    const period = elems.period_min != null ? `${elems.period_min.toFixed(1)} min` : '—';
    const list = rows.slice(0, SEL_MAX_ROWS).map(({ it, partner }) => {
      const lvl = safeLevel(it.risk_level);
      return `<button type="button" class="sel-conj" data-conj="${FMT.esc(String(it.id))}"` +
          ` data-partner="${FMT.esc(String(partner.norad_id))}"` +
          ` title="Show this close approach on the globe">` +
        `<span class="chip chip--${lvl}">${LEVEL_LABEL[lvl]}</span>` +
        `<span class="sel-conj__name">${FMT.esc(partner.name)}</span>` +
        `<span class="sel-conj__when">in ${FMT.esc(humanIn(it.tca))}</span>` +
        '</button>';
    }).join('');

    els.selBody.innerHTML =
      `<div class="${changed ? 'sel-body-in' : ''}">` +
        `<a class="sel-name" href="${href}">${FMT.esc(sel.name)}</a>` +
        '<div class="sel-sub">' +
          `<span class="chip chip--type">${FMT.esc(typeWord(sel.type))}</span>` +
          `<span>NORAD <span class="code">${FMT.esc(noradStr)}</span></span>` +
        '</div>' +
        '<div class="sel-grid">' +
          selStat('Altitude', FMT.km(sel.alt_km, 0), 'Current altitude above the Earth') +
          selStat('Speed', sel.speed_kms != null ? FMT.vel(sel.speed_kms * 1000) : '—') +
          selStat('Inclination', FMT.deg(elems.inclination_deg, 1)) +
          selStat('Period', period, 'Time for one orbit') +
          selStat('Apogee', FMT.km(elems.apogee_km, 0), 'Highest point of the orbit') +
          selStat('Perigee', FMT.km(elems.perigee_km, 0), 'Lowest point of the orbit') +
        '</div>' +
        `<div class="sel-label">Close approaches <span class="faint">${FMT.esc(FMT.num(rows.length))}</span></div>` +
        (rows.length
          ? `<div class="sel-conj-list">${list}</div>` +
            (rows.length > SEL_MAX_ROWS
              ? `<div class="sel-conj-more">and ${FMT.num(rows.length - SEL_MAX_ROWS)} more in the table</div>` : '')
          : `<div class="sel-none">${FMT.esc(selNoneText())}</div>`) +
        '<div class="sel-foot">' +
          `<a class="btn btn--primary btn--xs" href="${href}">Object details</a>` +
          (rows.length ? '<button type="button" class="btn btn--xs" data-act="table">Show in table</button>' : '') +
        '</div>' +
      '</div>';
    if (changed) {
      /* a new selection opens its section and brings it into view */
      setSectionCollapsed(els.selPanel, false);
      if (els.stageSide && getComputedStyle(els.stageSide).position === 'absolute') {
        els.stageSide.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }
  }

  /* collapsible sheet sections */
  function setSectionCollapsed(sec, collapsed) {
    if (!sec) return;
    sec.classList.toggle('is-collapsed', collapsed);
    const t = sec.querySelector('.sheet-sec__toggle');
    if (t) t.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
  }

  function onSheetClick(ev) {
    const t = ev.target.closest('.sheet-sec__toggle');
    if (!t) return;
    const sec = t.closest('.sheet-sec');
    setSectionCollapsed(sec, !sec.classList.contains('is-collapsed'));
  }

  function onSelConjClick(ev) {
    const act = ev.target.closest('[data-act="table"]');
    if (act && lastSel) {
      window.KesslerUI.setTableFilter('', String(lastSel.norad));
      const panel = els.tbody.closest('.panel');
      if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    const btn = ev.target.closest('.sel-conj');
    if (!btn || !globeReady) return;
    const it = conjItems.find((c) => String(c.id) === btn.dataset.conj);
    if (it && it.primary && it.secondary && canShowEncounter()) {
      showEncounterOnGlobe(it.primary.norad_id, it.secondary.norad_id, it.tca, it);
      return;
    }
    KGlobe.select(Number(btn.dataset.partner));
  }

  /* --- CHANGES panel (run-over-run diff from /changes) ------------------------ */

  const LEVEL_RANK = { nominal: 0, elevated: 1, critical: 2 };

  function lvlWord(level) {
    const lvl = safeLevel(level);
    return `<span class="chg-lvl chg-lvl--${lvl}">${LEVEL_LABEL[lvl]}</span>`;
  }

  /* full names, never truncated: first object on one line, "and ..." on the next */
  function chgPair(text) {
    const t = String(text).replace(/\s*(↔|<->)\s*/g, ' and ');
    const k = t.indexOf(' and ');
    const body = k > 0
      ? `<span class="chg-pair__a">${FMT.esc(t.slice(0, k))}</span>` +
        `<span class="chg-pair__b"><span class="chg-pair__and">and </span>${FMT.esc(t.slice(k + 5))}</span>`
      : FMT.esc(t);
    return `<span class="chg-pair">${body}</span>`;
  }

  /* every change row is a link to its event: click/Enter focuses the
     conjunction (globe encounter + table row) via focusChangeConj */
  function chgRowOpen(id) {
    const cid = Number(id);
    return Number.isFinite(cid) && cid > 0
      ? `<div class="chg-row chg-row--link" data-conj="${cid}" tabindex="0" role="button">`
      : '<div class="chg-row">';
  }

  function chgLevelRow(c) {
    const up = (LEVEL_RANK[safeLevel(c.to_level)] || 0) >=
               (LEVEL_RANK[safeLevel(c.from_level)] || 0);
    return chgRowOpen(c.conjunction_id) +
      `<span class="chg-arrow">${up ? '↑' : '↓'}</span>` +
      `<span class="chg-shift">${lvlWord(c.from_level)}<span aria-hidden="true">→</span>` +
        `<span class="sr-only"> to </span>${lvlWord(c.to_level)}</span>` +
      chgPair(String(c.pair || '')) +
      '</div>';
  }

  function chgNewRow(it) {
    const pair = `${it.primary ? it.primary.name : '—'} and ${it.secondary ? it.secondary.name : '—'}`;
    return chgRowOpen(it.id) + `${lvlWord(it.risk_level)}${chgPair(pair)}</div>`;
  }

  function chgResolvedRow(r) {
    return chgRowOpen(r.conjunction_id) +
      `${lvlWord(r.last_level)}${chgPair(String(r.pair || ''))}</div>`;
  }

  /* Focus one conjunction from a CHANGES click: encounter on the globe
     (applyConjParam also pre-caches the drawer detail), then - when the
     event is in the loaded table page - clear any filter hiding it, lift
     the render cap to reach it, scroll to the row and open its drawer.
     Events beyond the loaded page still get the globe encounter. */
  async function focusChangeConj(id) {
    await applyConjParam(id);
    const item = conjItems.find((it) => it.id === id);
    if (!item) return;
    const q = tableSearch.trim().toLowerCase();
    const hidden = (q && !matchesConj(item, q)) ||
      (tableRisk && safeLevel(item.risk_level) !== tableRisk);
    if (hidden) {
      tableRisk = null;
      tableSearch = '';
      els.conjRisk.value = '';
      els.conjRisk.dispatchEvent(new Event('change', { bubbles: true }));
      els.conjSearch.value = '';
    }
    applyTableView();
    const pos = currentView.findIndex((it) => it.id === id);
    if (pos >= visibleRows) {
      visibleRows = Math.ceil((pos + 1) / LOAD_PAGE) * LOAD_PAGE;
      applyTableView();
    }
    const tr = els.tbody.querySelector(`tr[data-id="${id}"]`);
    if (!tr) return;
    tr.scrollIntoView({ block: 'center', behavior: 'smooth' });
    if (expandedId !== id) toggleDrawer(id, tr);
  }

  function renderChanges(data) {
    const levels = Array.isArray(data.level_changes) ? data.level_changes : [];
    const fresh = Array.isArray(data.new_conjunctions) ? data.new_conjunctions : [];
    const gone = Array.isArray(data.resolved) ? data.resolved : [];
    const counts = data.counts || {};
    const since = data.since && data.since.finished_at;
    els.changesMeta.textContent = since ? `Since ${String(since).slice(11, 16)} UTC` : '';
    /* a full-catalog screening can diff THOUSANDS of events - the panel
       shows the top few per section and states the honest total (a first
       full run once put 8,420 rows here and made the page a minute-long
       scroll) */
    const CHG_ROWS = 8;
    const section = (list, total, label, rowFn) => {
      if (!list.length) return '';
      let out = `<div class="chg-label">${label} <span class="faint">${FMT.num(total)}</span></div>` +
        list.slice(0, CHG_ROWS).map(rowFn).join('');
      if (total > CHG_ROWS) {
        out += `<div class="chg-row chg-more">and ${FMT.num(total - CHG_ROWS)} more</div>`;
      }
      return out;
    };
    let html = '';
    if (!levels.length && !fresh.length && !gone.length) {
      html += '<div class="chg-empty">Nothing changed since the last refresh.</div>';
    } else {
      html += section(levels, counts.level_changes ?? levels.length,
        'Risk level changed', chgLevelRow);
      html += section(fresh, counts.new ?? fresh.length, 'New', chgNewRow);
      html += section(gone, counts.resolved ?? gone.length,
        'No longer a concern', chgResolvedRow);
    }
    /* maneuver-candidate subsection - absent when /anomalies has nothing
       (or the backend predates it); heuristic label carried in the header */
    if (Array.isArray(lastAnomalies) && lastAnomalies.length) {
      html += '<div class="chg-label">Possible maneuvers</div>' +
        '<div class="chg-note">Estimated from changes in orbital elements — a heuristic, not a confirmation.</div>' +
        lastAnomalies.map(anomalyRow).join('');
    }
    els.changesBody.innerHTML = html;
  }

  /* js/api.js is a frozen shared foundation without a /changes wrapper - this
     mirrors its error handling for the one endpoint the dashboard needs.
     404 = pre-restart backend: keep the panel hidden and try again next poll
     (a mid-session restart then makes it appear). Other failures keep the
     last rendered diff. */
  async function refreshChanges() {
    let data = null;
    try {
      const res = await fetch('/changes');
      if (res.status === 404) { els.changesPanel.hidden = true; return; }
      if (!res.ok) return;
      data = await res.json();
    } catch (_) { return; /* network fault - keep last state */ }
    if (!data || typeof data !== 'object') return;
    lastChanges = data;
    renderChanges(data);
    els.changesPanel.hidden = false;
  }

  /* --- MANEUVER CANDIDATES (element-delta heuristic, GET /anomalies) ----------- */

  let lastChanges = null;   /* last good /changes payload (re-render anchor) */
  let lastAnomalies = null; /* last good /anomalies items, top 5; null = none */

  /* ΔSMA signed km (the contract-mandated number) + a terse kind tag; the
     object name links to its detail page. Level colors are not used here -
     an element delta is not a risk severity (amber/red reservation). */
  function anomalyRow(a) {
    const isInc = a.kind === 'inclination';
    const dv = Number(isInc ? a.delta_inc_deg : a.delta_sma_km);
    const unit = isInc ? '°' : ' km';
    const label = isInc ? 'Change in inclination' : 'Change in semi-major axis';
    const mag = Number.isFinite(dv) ? `${Math.abs(dv).toFixed(isInc ? 3 : 2)}${unit}` : '—';
    const signed = Number.isFinite(dv) ? `${dv < 0 ? '−' : '+'}${mag}` : '—';
    const kind = isInc ? 'Plane change'
      : a.kind === 'sma_drop' ? 'Lowered' : 'Raised';
    const norad = encodeURIComponent(String(a.norad_id));
    const name = String(a.name || a.norad_id || '—');
    return `<div class="chg-row" title="${FMT.esc(`${label} ${signed}`)}">` +
      `<span class="chg-shift">${kind}</span>` +
      `<span class="chg-pair"><a href="object.html?id=${norad}" ` +
        `title="${FMT.esc(name)}">${FMT.esc(name)}</a></span>` +
      `<span class="chg-dsma">${FMT.esc(isInc ? signed : mag)}</span>` +
      '</div>';
  }

  /* Top 5 by |Δ| (the API pre-sorts desc). 404 = pre-restart backend - the
     subsection stays absent; other failures keep the last rendered list.
     Renders through renderChanges so the subsection lives inside the
     CHANGES panel and survives its re-renders. */
  async function refreshAnomalies() {
    try {
      const res = await fetch('/anomalies');
      if (res.status === 404) { lastAnomalies = null; return; }
      if (!res.ok) return;
      const data = await res.json();
      const items = data && Array.isArray(data.items) ? data.items : [];
      lastAnomalies = items.slice(0, 5);
    } catch (_) { return; /* network fault - keep last state */ }
    if (lastChanges) renderChanges(lastChanges);
  }

  /* --- WATCH LOG panel (AI change-of-watch entries, GET /watchlog) ----------- */

  /* Minimal safe markdown for watch summaries - the assistant.js idiom:
     escape first (FMT.esc), then transform. Code spans are protected with
     NUL-index placeholders (NULs stripped from the source first so entry
     text can never forge one); internal links only (single leading '/'). */
  function wlInline(escaped) {
    const codes = [];
    let s = escaped.replace(/\u0000/g, '').replace(/`([^`\n]+)`/g, (m, c) => {
      codes.push(c);
      return `\u0000${codes.length - 1}\u0000`;
    });
    s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/\[([^\]\n]+)\]\((\/(?!\/)[^)\s]*)\)/g, '<a href="$2">$1</a>');
    return s.replace(/\u0000(\d+)\u0000/g, (m, i) => `<code>${codes[Number(i)]}</code>`);
  }

  /* Bullets + paragraphs only - watch entries are terse bullet logs. */
  function wlMarkdown(raw) {
    const out = [];
    let list = [];
    let para = [];
    const flushList = () => {
      if (list.length) { out.push(`<ul>${list.join('')}</ul>`); list = []; }
    };
    const flushPara = () => {
      if (para.length) { out.push(`<p>${para.join('<br>')}</p>`); para = []; }
    };
    for (const line of wlInline(FMT.esc(raw)).split('\n')) {
      const t = line.trim();
      const h = /^#{1,4} (.*)$/.exec(t);   /* the model may head with '## ...' */
      if (h) { flushPara(); flushList(); out.push(`<p><strong>${h[1]}</strong></p>`); }
      else if (/^[-*] /.test(t)) { flushPara(); list.push(`<li>${t.slice(2)}</li>`); }
      else if (t === '') { flushPara(); flushList(); }
      else { flushList(); para.push(t); }
    }
    flushPara();
    flushList();
    return out.join('');
  }

  /* "2026-08-17T06:00:12Z" -> "2026-08-17 06:00Z" (short UTC, minutes) */
  function wlTime(iso) {
    const s = String(iso || '');
    return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(s) ? shortUtc(s) : '—';
  }

  function renderWatchlog(entries) {
    if (!entries.length) {
      els.watchlogBody.innerHTML =
        '<div class="wl-empty">No notes yet. The AI writes one at the end of each scheduled shift.</div>';
      return;
    }
    /* only the newest entry carries its body (scrollable via CSS max-height)
       - earlier shifts collapse to header lines so the panel stays short */
    els.watchlogBody.innerHTML = entries.map((e, i) =>
      `<div class="wl-entry${i ? ' wl-entry--older' : ''}">` +
        '<div class="wl-hd">' +
          `<span class="wl-time">${FMT.esc(wlTime(e.created_at))}</span>` +
          (e.model ? `<span class="wl-model">${FMT.esc(String(e.model))}</span>` : '') +
          (i ? '<span class="wl-model">· earlier shift</span>' : '') +
        '</div>' +
        (i ? '' : `<div class="wl-body">${wlMarkdown(String(e.summary_md || ''))}</div>`) +
      '</div>').join('');
  }

  /* Same degrade contract as refreshChanges: 404 = pre-restart backend -
     keep the panel hidden and try again next poll; other failures keep the
     last rendered entries. An answering endpoint with an empty log shows
     the panel with its empty state. */
  async function refreshWatchlog() {
    let data = null;
    try {
      const res = await fetch('/watchlog?limit=5');
      if (res.status === 404) { els.watchlogPanel.hidden = true; return; }
      if (!res.ok) return;
      data = await res.json();
    } catch (_) { return; /* network fault - keep last state */ }
    if (!data || !Array.isArray(data.entries)) return;
    renderWatchlog(data.entries.slice(0, 5));
    els.watchlogPanel.hidden = false;
  }

  /* --- polling ------------------------------------------------------------- */

  async function poll() {
    els.tableWrap.classList.add('is-refetching');
    try {
      /* limit=1000 (the API cap): the default page of 100 starved the globe's
         risk styling (~700+ elevated/critical events live) and left the
         NOMINAL filter permanently empty. The meta line still shows
         `<shown> of <total>` when the catalog exceeds the page. */
      const [st, conj] = await Promise.all([
        API.status(),
        API.conjunctions({ limit: 1000 }),
      ]);
      const items = Array.isArray(conj.items) ? conj.items : [];
      renderStatus(st);
      drawerCache.clear();
      whyCache.clear(); /* explain payloads follow the CDM cache lifetime */
      conjItems = items;
      conjTotal = conj.count != null ? conj.count : items.length;
      hasData = true;
      if (tableRisk) { riskFetched = null; ensureRiskPage(); } /* keep filter fresh */
      applyTableView(); /* renders through renderTable with filters/sort applied */
      clearError();
      if (globeReady) {
        /* isolated so an engine fault never reads as an API error */
        try { KGlobe.setConjunctions(conjItems); } catch (_) { /* engine fault */ }
      }
      if (lastSel) renderSelected(lastSel); /* refresh partner rows in place */
    } catch (err) {
      showError(err);
    } finally {
      els.tableWrap.classList.remove('is-refetching');
      refreshChanges();   /* on load + after every poll, success or not */
      refreshAnomalies(); /* maneuver-candidate subsection, same cadence */
      refreshWatchlog();  /* AI watch-log entries, same cadence */
    }
  }

  /* --- countdown tick (single interval for every cell) ---------------------- */

  function tickCountdowns() {
    const cells = els.tbody.querySelectorAll('td[data-tca]');
    for (const c of cells) {
      const iso = c.getAttribute('data-tca');
      const txt = humanIn(iso);
      if (c.textContent !== txt) c.textContent = txt;
      c.classList.toggle('is-soon', soonClass(iso) !== '');
    }
  }

  /* --- URL-param application + window.KesslerUI hooks ------------------------- */

  /* Applies ?sel=<norad> (globe selection), ?risk=<level> and ?search=<text>
     (conjunction-table filter controls), and ?conj=<id> (encounter mode)
     once on load, then strips exactly those params via history.replaceState.
     KGlobe.select queues internally until the fleet snapshot arrives, so
     calling it right after initGlobe is safe; the filter state is picked up
     by the first poll's applyTableView. Params may combine (?sel=&conj=...). */
  function applyUrlParams() {
    const params = new URLSearchParams(window.location.search);
    if (!params.has('sel') && !params.has('risk') && !params.has('search') &&
        !params.has('conj') && !params.has('sim')) return;
    const sel = Number(params.get('sel'));
    if (params.has('sel') && Number.isFinite(sel) && globeReady) {
      try { KGlobe.select(sel); } catch (_) { /* engine fault */ }
    }
    const conjId = Number(params.get('conj'));
    const simId = Number(params.get('sim'));
    if (params.has('sim') && Number.isFinite(simId) && simId > 0) {
      /* the encounter shows on the globe behind the simulator */
      if (!(params.has('conj') && conjId === simId)) applyConjParam(simId);
      openSimWhenReady(simId);
    }
    if (params.has('conj') && Number.isFinite(conjId) && conjId > 0) {
      applyConjParam(conjId); /* async - param already stripped below */
    }
    const risk = params.get('risk');
    if (risk && KNOWN_LEVELS[risk]) {
      els.conjRisk.value = risk;
      els.conjRisk.dispatchEvent(new Event('change', { bubbles: true }));
    }
    const search = params.get('search');
    if (search) {
      tableSearch = search;
      els.conjSearch.value = search;
    }
    params.delete('sel');
    params.delete('risk');
    params.delete('search');
    params.delete('conj');
    params.delete('sim');
    const qs = params.toString();
    history.replaceState(null, '',
      window.location.pathname + (qs ? `?${qs}` : '') + window.location.hash);
    applyTableView(); /* no-op before first poll - poll renders the same path */
  }

  /* ?conj=<id>: fetch the conjunction, then hand its pair to encounter
     mode. The engine queues showEncounter internally until its snapshot is
     ready (same as select); a fetch failure or absent engine is a silent
     no-op - the dashboard still loads normally. */
  async function applyConjParam(id) {
    let detail = null;
    try { detail = await API.conjunction(id); } catch (_) { return; }
    if (!detail || !detail.primary || !detail.secondary) return;
    drawerCache.set(detail.id, detail); /* free drawer open until next poll */
    showEncounterOnGlobe(
      detail.primary.norad_id, detail.secondary.norad_id, detail.tca, detail);
  }

  /* ?sim=<id> (links from Start, Alerts, Object, early warnings): show the
     pair on the globe, then open the encounter simulator as soon as its
     module (loaded after this script) is up. Gives up quietly after 20 s. */
  function openSimWhenReady(id) {
    const t0 = Date.now();
    (function tryOpen() {
      if (simAvailable() && document.readyState === 'complete') {
        try { KSim.open({ conjunctionId: id }); } catch (_) { /* module fault */ }
        return;
      }
      if (Date.now() - t0 < 20000) setTimeout(tryOpen, 120);
    })();
  }

  /* Minimal cross-script surface consumed by js/assistant.js (Kessler AI).
     Keep this tiny - it is the only sanctioned coupling point:
       setTableFilter(risk, search) - drive the conjunction-table controls;
         null/undefined leaves a field unchanged, '' clears it, unknown risk
         values fall back to ALL RISK.
       setGlobeFilter({types, riskOnly}) - drive the globe type / RISK ONLY
         toggles; types uses ['PAYLOAD','DEBRIS','RB','UNKNOWN'] ('R-B' and
         'ROCKET BODY' accepted as aliases), omitted keys keep current state.
       setGlobeSpeed(mult) - sim speed 1|60|600 with button-state sync.
       selectedNorad() - current globe selection (number) or null.
       globeReady() - engine initialized and usable.
       showEncounter(opts) - encounter mode for the assistant's
         show_encounter action; accepts the action payload keys
         {primary_norad, secondary_norad, tca} (camelCase primaryNorad/
         secondaryNorad/tcaIso also accepted), runs watch mode and scrolls
         the globe into view. Returns true on success, false when the
         engine is absent/older - caller then folds into /?conj=<id>. */
  const UI_TYPE_ALIASES = { 'R-B': 'RB', 'ROCKET BODY': 'RB' };
  window.KesslerUI = {
    /* scroll to a close approach in the table and open its drawer */
    focusConjunction(id) {
      const n = Number(id);
      if (Number.isFinite(n)) focusChangeConj(n);
    },
    /* the dock / insets changed shape (the space-weather lens) */
    relayout() { layoutStage(); },
    setTableFilter(risk, search) {
      if (search !== null && search !== undefined) {
        tableSearch = String(search);
        els.conjSearch.value = tableSearch;
      }
      if (risk !== null && risk !== undefined) {
        els.conjRisk.value = KNOWN_LEVELS[risk] ? risk : '';
        /* change event keeps the custom dropdown label in sync and re-renders */
        els.conjRisk.dispatchEvent(new Event('change', { bubbles: true }));
      } else {
        applyTableView();
      }
    },
    setGlobeFilter(opts) {
      const o = opts || {};
      if (Array.isArray(o.types)) {
        const on = {};
        for (const t of o.types) {
          const key = String(t).toUpperCase();
          on[UI_TYPE_ALIASES[key] || key] = true;
        }
        for (const t of GLOBE_TYPES) typesOn[t] = !!on[t];
        for (const b of els.globeTypeGroup.querySelectorAll('button[data-type]')) {
          setPressed(b, !!typesOn[b.dataset.type]);
        }
      }
      if (o.riskOnly !== undefined) {
        riskOnly = !!o.riskOnly;
        setPressed(els.globeRiskOnly, riskOnly);
      }
      pushGlobeFilter();
    },
    setGlobeSpeed(mult) {
      const m = Number(mult);
      if (!globeReady || (m !== 1 && m !== 60 && m !== 600)) return;
      KGlobe.setSpeed(m);
      for (const b of els.globeSpeed.querySelectorAll('button[data-speed]')) {
        setPressed(b, Number(b.dataset.speed) === m);
      }
    },
    selectedNorad() { return lastSel ? lastSel.norad : null; },
    globeReady() { return globeReady; },
    showEncounter(opts) {
      const o = opts || {};
      const p = Number(o.primary_norad !== undefined ? o.primary_norad : o.primaryNorad);
      const s = Number(o.secondary_norad !== undefined ? o.secondary_norad : o.secondaryNorad);
      const tca = o.tca !== undefined ? o.tca : o.tcaIso;
      if (!Number.isFinite(p) || !Number.isFinite(s)) return false;
      return showEncounterOnGlobe(p, s, tca);
    },
  };

  /* --- events / init --------------------------------------------------------- */

  function onRowClick(ev) {
    if (ev.target.closest('a')) return; /* let object links navigate */
    const tr = ev.target.closest('tr[data-id]');
    if (!tr) return;
    toggleDrawer(Number(tr.dataset.id), tr);
  }

  let resizeTimer = null;
  function onResize() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      drawStars();
      layoutStage();
      renderTimeline(); /* canvas backing store follows the new width */
      if (expandedId == null) return;
      const drawer = els.tbody.querySelector('tr.drawer');
      const detail = drawerCache.get(expandedId);
      if (drawer && detail) renderDrawerContent(drawer.firstElementChild, detail);
    }, 150);
  }

  FMT.startUtcClock(document.getElementById('utc-clock'));
  els.tbody.addEventListener('click', onRowClick);
  els.tbody.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Enter' && ev.key !== ' ') return;
    const tr = ev.target.closest('tr[data-id]');
    if (!tr) return;
    ev.preventDefault(); /* Space must not scroll the page */
    toggleDrawer(Number(tr.dataset.id), tr);
  });
  els.changesBody.addEventListener('click', (ev) => {
    const row = ev.target.closest('[data-conj]');
    if (row) focusChangeConj(Number(row.dataset.conj));
  });
  els.changesBody.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Enter' && ev.key !== ' ') return;
    const row = ev.target.closest('[data-conj]');
    if (row) { ev.preventDefault(); focusChangeConj(Number(row.dataset.conj)); }
  });
  els.selBody.addEventListener('click', onSelConjClick);
  els.conjThead.addEventListener('click', onSortClick);
  els.conjThead.addEventListener('keydown', onSortKeydown);
  els.conjSearch.addEventListener('input', () => {
    tableSearch = els.conjSearch.value;
    applyTableView();
  });
  els.conjRisk.addEventListener('change', () => {
    tableRisk = els.conjRisk.value;
    syncRiskSeg();
    applyTableView();
    ensureRiskPage();
  });
  if (els.conjRiskSeg) {
    els.conjRiskSeg.addEventListener('click', (ev) => {
      const b = ev.target.closest('button[data-risk]');
      if (!b) return;
      els.conjRisk.value = b.dataset.risk;
      els.conjRisk.dispatchEvent(new Event('change', { bubbles: true }));
    });
  }
  if (els.stageSide) els.stageSide.addEventListener('click', onSheetClick);
  if (els.selClear) {
    els.selClear.addEventListener('click', () => {
      if (globeReady) { try { KGlobe.select(null); } catch (_) { /* engine fault */ } }
    });
  }
  if (els.zoomGroup) els.zoomGroup.addEventListener('click', onZoomClick);
  if (els.encSim) els.encSim.addEventListener('click', onEncSim);
  window.addEventListener('resize', syncDrawerWidth);
  if (els.encClose) {
    els.encClose.addEventListener('click', () => {
      if (globeReady) { try { KGlobe.select(null); } catch (_) { /* engine fault */ } }
      encCtx = null;
      syncEncBar();
    });
  }
  document.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Escape' || !lastSel || !globeReady) return;
    const t = ev.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    if (document.querySelector('.k-sheet, .k-scrim')) return; /* a modal owns Esc */
    try { KGlobe.select(null); } catch (_) { /* engine fault */ }
  });
  /* theme flips: canvases re-read their tokens (the globe does it itself) */
  if (typeof KTheme !== 'undefined' && typeof KTheme.onChange === 'function') {
    KTheme.onChange(() => {
      applyTableView();   /* sparklines + drawer charts */
      renderGlobeLegend();
    });
  }
  if (typeof ResizeObserver === 'function' && els.stageDock) {
    const ro = new ResizeObserver(() => layoutStage());
    ro.observe(els.stageDock);
    if (els.stageSide) ro.observe(els.stageSide);
  }
  els.statTileElevated.addEventListener('click', () => setRiskFilter('elevated'));
  els.statTileCritical.addEventListener('click', () => setRiskFilter('critical'));
  els.loadMore.addEventListener('click', onLoadMore);
  els.conjExport.addEventListener('click', onExportCsv);
  els.globeTypeGroup.addEventListener('click', onTypeToggle);
  els.globeRiskOnly.addEventListener('click', onRiskOnlyToggle);
  els.globeModeGroup.addEventListener('click', onModeClick);
  els.globeHeat.addEventListener('click', onHeatToggle);
  els.globeSurface.addEventListener('click', onSurfaceToggle);
  els.scrubTrack.addEventListener('pointerdown', onScrubDown);
  els.scrubTrack.addEventListener('pointermove', onScrubMove);
  els.scrubTrack.addEventListener('pointerup', onScrubUp);
  els.scrubTrack.addEventListener('pointercancel', onScrubUp);
  els.scrubTrack.addEventListener('dblclick', onScrubDblClick);
  els.scrubTrack.addEventListener('keydown', onScrubKeydown);
  els.globeSearch.addEventListener('input', taCompute);
  els.globeSearch.addEventListener('focus', () => {
    if (els.globeSearch.value.trim()) taCompute();
  });
  els.globeSearch.addEventListener('blur', taClose);
  els.globeSearch.addEventListener('keydown', onTaKeydown);
  els.globeTypeahead.addEventListener('pointerdown', onTaPointerDown);
  window.addEventListener('resize', onResize);
  setInterval(tickCountdowns, 1000);
  setInterval(poll, POLL_MS);
  /* timeline renderer builds in parallel - a missing module hides the
     panel instead of leaving a dead canvas */
  if (typeof drawTimeline !== 'function') els.timelinePanel.hidden = true;
  drawStars();
  initGlobe();
  if (!lastSel) els.selBody.innerHTML = SEL_EMPTY_HTML; /* pointer-aware zoom hint */
  applyUrlParams(); /* after initGlobe - ?sel= needs the engine (it queues) */
  poll();
})();
