/* Kessler - alerts page (an inbox for triage).
   Elevated + critical active conjunctions as clear alert rows: risk pill,
   the two objects, closest approach as a live human countdown, miss
   distance, probability (with plain-language odds), a trend sparkline of
   the probability history, and actions (Globe, Simulate, Details).
   - Two risk-filtered /conjunctions?risk= fetches per poll (CONTRACTS.md)
     so the feed is never silently truncated by an unfiltered page limit;
     the summary counts come from /status (authoritative).
   - The list renders at most PAGE rows at a time; "Show more" reveals
     +PAGE from the already-fetched buckets, and an honest "Showing X of Y"
     line closes the list only when the fetch cap is below the server count.
   - A client-side All / Critical / Elevated filter narrows the inbox.
   - Polls every 60 s with .is-refetching; on failure shows the banner and
     keeps the last data. Empty states stay honest: an empty catalog or a
     failed refresh is "no data", never an all-clear (render()).
   - Globe -> /?conj=<id> (dashboard watch mode); Simulate -> KSim.open in
     place (typeof-guarded).
   - Decaying orbits: /reentry?perigee_km=250, 10 min poll (server cache);
     404 = backend without the route -> quiet in-panel note. */
'use strict';

(() => {
  const POLL_MS = 60000;
  const REENTRY_POLL_MS = 600000;
  const REENTRY_PERIGEE_KM = 250;
  const PAGE = 25;

  const els = {
    list: document.getElementById('alert-list'),
    error: document.getElementById('error-banner'),
    hdrStatus: document.getElementById('hdr-status'),
    sumActive: document.getElementById('sum-active'),
    sumCritical: document.getElementById('sum-critical'),
    sumElevated: document.getElementById('sum-elevated'),
    sumAsOf: document.getElementById('sum-asof'),
    sumWindow: document.getElementById('sum-window'),
    queueMeta: document.getElementById('queue-meta'),
    filter: document.getElementById('alert-filter'),
    slWindow: document.getElementById('sl-window'),
    slDemo: document.getElementById('sl-demo'),
    slAuto: document.getElementById('sl-auto'),
    reentryWrap: document.getElementById('reentry-wrap'),
    reentryBody: document.getElementById('reentry-body'),
    reentryEmpty: document.getElementById('reentry-empty'),
    reentryCount: document.getElementById('reentry-count'),
    reentryMore: document.getElementById('reentry-more'),
  };

  const TYPE_LABELS = { PAYLOAD: 'Payload', DEBRIS: 'Debris', 'ROCKET BODY': 'Rocket body', UNKNOWN: 'Unknown' };

  let hasData = false;
  let alertItems = [];      /* fetched alerts (<=1000/bucket), critical first */
  let alertTotal = 0;       /* API count across both risk buckets */
  let visibleRows = PAGE;
  let filter = 'all';
  let lastStatus = null;
  let lastOkAt = null;
  let reentryAll = false;

  FMT.startUtcClock(document.getElementById('utc-clock'));

  function typeLabel(t) {
    const k = String(t || 'UNKNOWN').toUpperCase();
    return TYPE_LABELS[k] || (k.charAt(0) + k.slice(1).toLowerCase());
  }

  /* --- rendering ---------------------------------------------------------- */

  function objectLink(obj) {
    return `<a class="ar-obj" href="object.html?id=${encodeURIComponent(obj.norad_id)}">${FMT.esc(obj.name)}</a>`;
  }

  function trendOf(values) {
    if (values.length < 2) return { word: values.length ? 'One message' : '', cls: '' };
    const d = values[values.length - 1] - values[0];   /* decades (log10) */
    if (d > 0.5) return { word: 'Rising', cls: 'is-up' };
    if (d < -0.5) return { word: 'Falling', cls: 'is-down' };
    return { word: 'Steady', cls: '' };
  }

  function sparkValues(trend) {
    if (!Array.isArray(trend)) return [];
    return trend.filter((p) => typeof p === 'number' && p > 0).map((p) => Math.log10(p));
  }

  const ICON_GLOBE = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.6 2.7 3.9 5.7 3.9 9s-1.3 6.3-3.9 9c-2.6-2.7-3.9-5.7-3.9-9S9.4 5.7 12 3z"/></svg>';
  const ICON_PLAY = '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.2-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z"/></svg>';

  function alertRow(c) {
    const level = c.risk_level === 'critical' ? 'critical' : 'elevated';
    const effectivePc = c.predicted_final_risk != null ? c.predicted_final_risk : c.pc_latest;
    const p = c.primary;
    const s = c.secondary;
    const detailHref = `object.html?id=${encodeURIComponent(p.norad_id)}&conj=${encodeURIComponent(c.id)}`;
    const globeHref = `/?conj=${encodeURIComponent(c.id)}`;
    const values = sparkValues(c.pc_trend);
    const tr = trendOf(values);
    const cdms = Number(c.cdm_count) || 0;
    return (
      `<article class="ar ar--${level}" data-id="${FMT.esc(c.id)}">` +
        `<div class="ar-main">` +
          `<div class="ar-top"><span class="chip chip--${level}">${level === 'critical' ? 'Critical' : 'Elevated'}</span>` +
          `<span class="ar-id">Event <span class="code">${FMT.esc(c.id)}</span></span></div>` +
          `<div class="ar-pair">${objectLink(p)}<span class="ar-and" aria-label="and">↔</span>${objectLink(s)}</div>` +
          `<div class="ar-sub">${FMT.esc(typeLabel(p.object_type))} <span class="code">${FMT.esc(p.norad_id)}</span>` +
          ` · ${FMT.esc(typeLabel(s.object_type))} <span class="code">${FMT.esc(s.norad_id)}</span></div>` +
        `</div>` +
        `<div class="ar-cell ar-when">` +
          `<div class="ar-k">Closest approach in</div>` +
          `<div class="ar-v js-count" data-tca="${FMT.esc(c.tca)}">${FMT.esc(FMT.countdown(c.tca))}</div>` +
          `<div class="ar-s">${FMT.esc(FMT.when(c.tca))}</div>` +
        `</div>` +
        `<div class="ar-cell">` +
          `<div class="ar-k">Miss distance</div>` +
          `<div class="ar-v">${FMT.esc(FMT.dist(c.miss_distance_m))}</div>` +
          `<div class="ar-s">at ${FMT.esc(FMT.vel(c.relative_velocity_ms))}</div>` +
        `</div>` +
        `<div class="ar-cell">` +
          `<div class="ar-k" title="Probability of collision (Pc), ${FMT.esc(c.model_version || 'model')}">Probability</div>` +
          `<div class="ar-v ar-v--${level}">${FMT.esc(FMT.pc(effectivePc))}</div>` +
          `<div class="ar-s">about ${FMT.esc(FMT.pcOdds(effectivePc))}</div>` +
        `</div>` +
        `<div class="ar-cell ar-trend">` +
          `<div class="ar-k">Trend</div>` +
          `<div class="ar-spark">${values.length ? '<canvas class="spark" aria-hidden="true"></canvas>' : '<span class="faint">—</span>'}</div>` +
          `<div class="ar-s ${tr.cls}">${FMT.esc(tr.word)}${cdms ? ` · ${FMT.num(cdms)} message${cdms === 1 ? '' : 's'}` : ''}</div>` +
        `</div>` +
        `<div class="ar-actions">` +
          `<a class="btn btn--xs" href="${globeHref}" aria-label="Show event ${FMT.esc(c.id)} on the globe">${ICON_GLOBE}Globe</a>` +
          `<button class="btn btn--xs js-sim" type="button" data-conj="${FMT.esc(c.id)}" aria-label="Simulate event ${FMT.esc(c.id)}">${ICON_PLAY}Simulate</button>` +
          `<a class="btn btn--xs btn--ghost" href="${detailHref}">Details</a>` +
        `</div>` +
      `</article>`
    );
  }

  function drawSparks() {
    if (typeof drawSparkline !== 'function') return;
    const rows = els.list.querySelectorAll('.ar');
    const byId = new Map(alertItems.map((c) => [String(c.id), c]));
    for (const row of rows) {
      const cv = row.querySelector('canvas.spark');
      const c = byId.get(row.dataset.id);
      if (!cv || !c) continue;
      const color = KTheme.css(c.risk_level === 'critical' ? '--k-critical' : '--k-elevated');
      drawSparkline(cv, sparkValues(c.pc_trend), color);
    }
  }

  function emptyHtml(title, text) {
    return `<div class="empty alerts-empty"><strong>${title}</strong>${text}</div>`;
  }

  function filtered() {
    return filter === 'all' ? alertItems : alertItems.filter((c) => c.risk_level === filter);
  }

  function render() {
    if (alertItems.length === 0) {
      /* an empty catalog or a failed refresh is "no data", never all-clear */
      const st = lastStatus || {};
      const noData = !(Number(st.catalog_size) > 0);
      els.queueMeta.textContent = '';
      els.list.innerHTML = noData
        ? emptyHtml('No data yet',
          'The satellite catalog hasn’t been ingested, so there’s nothing to screen.' +
          (st.last_run_error ? ' The last refresh failed.' : ''))
        : st.last_run_error
          ? emptyHtml('No active alerts — but the last refresh failed',
            'What you see may be out of date. Kessler will try again on the next refresh.')
          : st.snapshot
          ? emptyHtml('No active alerts in this snapshot',
            'Kessler is serving a saved offline snapshot, so this is not a live all-clear.')
          : (!st.last_ingest_at || Date.now() - Date.parse(st.last_ingest_at) > 6 * 3600e3)
          ? emptyHtml('No active alerts — but the data is out of date',
            'The catalog hasn’t been refreshed in over 6 hours, so this is not a current all-clear.')
          : emptyHtml('No alerts right now',
            'Every tracked close approach is below the elevated threshold.');
      return;
    }
    const items = filtered();
    if (!items.length) {
      els.queueMeta.textContent = '';
      els.list.innerHTML = emptyHtml(`No ${filter} alerts`, 'Switch to All to see the rest of the inbox.');
      return;
    }
    const shown = items.slice(0, visibleRows);
    let html = shown.map(alertRow).join('');
    if (items.length > shown.length) {
      html += `<button type="button" class="load-more js-load-more">Show more · ${FMT.num(shown.length)} of ${FMT.num(items.length)} shown</button>`;
    } else if (filter === 'all' && alertTotal > alertItems.length) {
      html += `<div class="alerts-end">Showing the ${FMT.num(alertItems.length)} highest-risk of ${FMT.num(alertTotal)} alerts.</div>`;
    }
    els.list.innerHTML = html;
    els.queueMeta.textContent = `${FMT.num(items.length)} ${items.length === 1 ? 'alert' : 'alerts'}`;
    drawSparks();
  }

  /* Counts come from /status - the item lists are pages and must never drive
     the headline numbers. */
  function renderSummary(st) {
    const critical = Number(st.critical_count) || 0;
    const elevated = Number(st.elevated_count) || 0;
    els.sumActive.textContent = FMT.num(critical + elevated);
    els.sumCritical.textContent = FMT.num(critical);
    els.sumElevated.textContent = FMT.num(elevated);
    els.sumCritical.classList.toggle('is-critical', critical > 0);
    els.sumElevated.classList.toggle('is-elevated', elevated > 0);
    lastOkAt = new Date().toISOString();
    tickAsOf();
  }

  /* C8: the tile says when the CATALOG was last updated (/status
     last_ingest_at - same source as the header pill); when this page last
     checked the server is secondary text. */
  function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }
  function tickAsOf() {
    const label = els.sumAsOf.previousElementSibling;
    if (label && label.textContent !== 'Catalog updated') label.textContent = 'Catalog updated';
    const st = lastStatus || {};
    if (st.last_ingest_at) {
      els.sumAsOf.textContent = cap(FMT.rel(st.last_ingest_at));
      els.sumAsOf.title = FMT.when(st.last_ingest_at);
    } else if (lastStatus) {
      els.sumAsOf.textContent = 'Not yet';
    }
    const hours = st.window_hours != null ? st.window_hours : 72;
    const checked = lastOkAt ? `Checked ${FMT.rel(lastOkAt)}` : 'Checking…';
    els.sumWindow.textContent = `${checked} · next ${hours} hours`;
  }

  function renderStatus(st) {
    lastStatus = st;
    tickAsOf();
    /* #hdr-status is filled by theme.js (one format on every page) */
    if (st.window_hours != null) {
      els.slWindow.textContent = st.window_hours;
    }
    els.slDemo.hidden = !st.has_demo_data;
    const autoH = Number(st.auto_refresh_interval_h);
    if (Number.isFinite(autoH) && autoH > 0) {
      els.slAuto.textContent = `Refreshes every ${autoH} h`;
      els.slAuto.hidden = false;
    } else {
      els.slAuto.hidden = true;
    }
  }

  /* --- interactions -------------------------------------------------------- */

  els.list.addEventListener('click', (ev) => {
    const more = ev.target.closest('.js-load-more');
    if (more) {
      visibleRows += PAGE;
      render();
      return;
    }
    const sim = ev.target.closest('.js-sim');
    if (!sim) return;
    ev.preventDefault();
    const id = Number(sim.dataset.conj);
    if (!Number.isFinite(id)) return;
    if (typeof KSim !== 'undefined' && typeof KSim.open === 'function') {
      try { KSim.open({ conjunctionId: id }); return; } catch (_) { /* module fault */ }
    }
    window.location.href = `/?sim=${encodeURIComponent(id)}`; /* dashboard opens it */
  });

  els.filter.addEventListener('click', (ev) => {
    const b = ev.target.closest('button[data-f]');
    if (!b) return;
    filter = b.dataset.f;
    visibleRows = PAGE;
    for (const x of els.filter.querySelectorAll('button')) {
      x.setAttribute('aria-pressed', String(x === b));
    }
    render();
  });

  KTheme.onChange(drawSparks);

  /* --- decaying orbits ------------------------------------------------------ */

  async function fetchReentry() {
    const res = await fetch(`/reentry?perigee_km=${REENTRY_PERIGEE_KM}`);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) { /* keep statusText */ }
      const err = new Error(`${res.status}: ${detail}`);
      err.status = res.status;
      throw err;
    }
    return res.json();
  }

  function reentryState(msg) {
    els.reentryWrap.hidden = true;
    els.reentryMore.hidden = true;
    els.reentryEmpty.textContent = msg;
    els.reentryEmpty.hidden = false;
  }

  /* "<30 D" style server windows -> "Within 30 days" */
  function windowText(w) {
    const s = String(w || '').trim();
    const m = /^([<>])\s*(\d+)\s*D$/i.exec(s);
    if (m) return `${m[1] === '<' ? 'Within' : 'More than'} ${m[2]} days`;
    const r = /^(\d+)\s*[-–]\s*(\d+)\s*D$/i.exec(s);
    if (r) return `${r[1]}–${r[2]} days`;
    return s || '—';
  }

  function reentryRow(it) {
    const href = `object.html?id=${encodeURIComponent(it.norad_id)}`;
    const decay = Number(it.decay_rate_km_day);
    const age = Number(it.epoch_age_h);
    const epochTitle = `Elements from ${FMT.when(it.epoch)}` +
      (Number.isFinite(age) ? ` (${age.toFixed(1)} h old)` : '');
    return (
      `<tr>` +
      `<td class="td-obj" title="${FMT.esc(epochTitle)}">` +
      `<a href="${href}">${FMT.esc(it.name)}</a>` +
      ` <span class="faint code">${FMT.esc(it.norad_id)}</span></td>` +
      `<td class="num">${FMT.esc(FMT.km(it.perigee_km, 0))}</td>` +
      `<td class="num">${Number.isFinite(decay) ? `${FMT.esc(decay.toFixed(1))} km/day` : '—'}</td>` +
      `<td>${FMT.esc(windowText(it.est_window))}</td>` +
      `<td class="td-link"><a class="btn btn--xs btn--ghost" href="${href}">Details</a></td>` +
      `</tr>`
    );
  }

  function renderReentry(items, total) {
    els.reentryCount.textContent = `${FMT.num(total)} below ${REENTRY_PERIGEE_KM} km`;
    if (!items.length) {
      reentryState(`No objects have a perigee below ${REENTRY_PERIGEE_KM} km right now.`);
      return;
    }
    const cap = reentryAll ? items.length : Math.min(items.length, 10);
    els.reentryBody.innerHTML = items.slice(0, cap).map(reentryRow).join('');
    els.reentryMore.hidden = cap >= items.length;
    els.reentryMore.textContent = `Show all ${FMT.num(items.length)}`;
    els.reentryMore.onclick = () => { reentryAll = true; renderReentry(items, total); };
    els.reentryEmpty.hidden = true;
    els.reentryWrap.hidden = false;
  }

  async function refreshReentry() {
    els.reentryWrap.classList.add('is-refetching');
    try {
      const data = await fetchReentry();
      const items = Array.isArray(data.items) ? data.items : [];
      const total = Number.isFinite(Number(data.count)) ? Number(data.count) : items.length;
      renderReentry(items, total);
    } catch (err) {
      if (els.reentryWrap.hidden) {
        els.reentryCount.textContent = '';
        reentryState(err.status === 404
          ? 'The decay watch needs a newer server build.'
          : 'The decay watch isn’t available right now. Retrying shortly.');
      }
    } finally {
      els.reentryWrap.classList.remove('is-refetching');
    }
  }

  /* --- ticks ----------------------------------------------------------------- */

  setInterval(() => {
    for (const node of els.list.querySelectorAll('.js-count')) {
      node.textContent = FMT.countdown(node.dataset.tca);
    }
  }, 1000);
  setInterval(tickAsOf, 15000);

  /* --- polling ---------------------------------------------------------------- */

  async function refresh() {
    els.list.classList.add('is-refetching');
    try {
      const [st, crit, elev] = await Promise.all([
        API.status(),
        API.conjunctions({ status: 'active', risk: 'critical', limit: 1000 }),
        API.conjunctions({ status: 'active', risk: 'elevated', limit: 1000 }),
      ]);
      /* critical first, each bucket in API order (effective risk desc, TCA asc) */
      alertItems = [...(crit.items || []), ...(elev.items || [])];
      alertTotal = (Number(crit.count) || 0) + (Number(elev.count) || 0);
      renderStatus(st);
      render();
      renderSummary(st);
      hasData = true;
      els.error.hidden = true;
    } catch (err) {
      els.error.textContent = hasData
        ? 'We couldn’t refresh the alert feed. Showing the last alerts we received.'
        : 'We couldn’t load the alert feed. Kessler will keep trying every minute.';
      els.error.title = err.message;
      els.error.hidden = false;
      if (!hasData) {
        els.list.innerHTML = emptyHtml('Alerts unavailable', 'The Kessler server isn’t answering right now.');
      }
    } finally {
      els.list.classList.remove('is-refetching');
    }
  }

  refresh();
  setInterval(refresh, POLL_MS);
  refreshReentry();
  setInterval(refreshReentry, REENTRY_POLL_MS);
})();
