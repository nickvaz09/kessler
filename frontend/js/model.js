/* Kessler - model page ("how it works").
   Renders GET /model/card: VIGIL's provenance, held-out metrics as big
   honest figures (with model-vs-baseline bars), the ship-gate arithmetic,
   feature importance (div bars, human names + raw names) and live
   prediction counts. The page hardcodes no metric numbers - every figure,
   including the vs-baseline percentages, is computed from the card, so a
   retrained model re-renders with zero HTML edits. Without the trained
   artifact the page degrades cleanly: neutral heuristic notice, em-dash
   figures, empty-state bars. Polls every 60 s (.is-refetching, never
   blanking). #vigil-e-slot is left untouched for another module. */
'use strict';

(() => {
  const POLL_MS = 60000;
  const $ = (id) => document.getElementById(id);

  const els = {
    error: $('error-banner'),
    banner: $('heuristic-banner'),
    hdrStatus: $('hdr-status'),
    heroVersion: $('hero-version'),
    heroTrained: $('hero-trained'),
    heroHigh: $('hero-high'),
    heroHighBase: $('hero-high-base'),
    heroHighPct: $('hero-high-pct'),
    heroAll: $('hero-all'),
    heroAllBase: $('hero-all-base'),
    heroAllPct: $('hero-all-pct'),
    barsAll: $('bars-all'),
    barsHigh: $('bars-high'),
    heroEvents: $('hero-events'),
    heroSplit: $('hero-split'),
    heroHighN: $('hero-high-n'),
    proseEvents: $('prose-events'),
    proseFeat: $('prose-feat'),
    gate1: $('gate-1'),
    gate2: $('gate-2'),
    fiList: $('fi-list'),
    fiMeta: $('fi-meta'),
    livePreds: $('live-preds'),
    liveConj: $('live-conj'),
    liveLast: $('live-last'),
    liveAsOf: $('live-asof'),
    slWindow: $('sl-window'),
    slDemo: $('sl-demo'),
    slAuto: $('sl-auto'),
    hero: document.querySelector('.model-hero'),
  };

  /* plain-language names for the model's features (raw name shown too) */
  const FEATURE_NAMES = {
    pc_latest: 'Latest probability',
    pc_log_delta: 'Last change in probability',
    pc_log_max: 'Peak probability so far',
    pc_log_slope: 'Probability trend',
    time_to_tca_h: 'Time to closest approach',
    msg_gap_last_h: 'Gap between the last two messages',
    cdm_count: 'Number of messages',
    relative_velocity_ms: 'Relative speed',
    miss_distance_m: 'Miss distance',
    md_slope: 'Miss distance trend',
    md_delta: 'Last change in miss distance',
  };
  const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen'];

  let hasData = false;

  FMT.startUtcClock($('utc-clock'));

  /* --- helpers ------------------------------------------------------------- */

  function finite(v) { return v !== null && v !== undefined && Number.isFinite(Number(v)); }
  function mse(v) { return finite(v) ? Number(v).toFixed(2) : '—'; }

  /* improvement vs baseline as a positive "x% lower" (or "x% higher") */
  function improvement(model, baseline) {
    if (!finite(model) || !finite(baseline) || Number(baseline) === 0) return null;
    return (1 - Number(model) / Number(baseline)) * 100;
  }

  function pctText(p) {
    if (p === null) return '—';
    const a = Math.abs(p);
    return `${a < 10 ? a.toFixed(1) : a.toFixed(0)}%`;
  }

  async function fetchCard() {
    const res = await fetch('/model/card');
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) { /* keep statusText */ }
      const err = new Error(`${res.status}: ${detail}`);
      err.status = res.status;
      throw err;
    }
    return res.json();
  }

  /* --- rendering ---------------------------------------------------------- */

  function renderStatus(st) {
    /* the header status line is owned by theme.js on every page */
    if (st.window_hours != null) els.slWindow.textContent = st.window_hours;
    els.slDemo.hidden = !st.has_demo_data;
    const autoH = Number(st.auto_refresh_interval_h);
    if (Number.isFinite(autoH) && autoH > 0) {
      els.slAuto.textContent = `Refreshes every ${autoH} h`;
      els.slAuto.hidden = false;
    } else {
      els.slAuto.hidden = true;
    }
  }

  function bars(el, model, baseline) {
    if (!finite(model) || !finite(baseline)) { el.innerHTML = ''; return; }
    const max = Math.max(Number(model), Number(baseline)) || 1;
    const row = (label, v, cls) =>
      `<div class="bar ${cls}"><span class="bar__l">${label}</span>` +
      `<span class="bar__t"><span class="bar__f" style="width:${((Number(v) / max) * 100).toFixed(1)}%"></span></span>` +
      `<span class="bar__v">${mse(v)}</span></div>`;
    el.innerHTML = row('VIGIL', model, 'bar--model') + row('Carry-forward', baseline, 'bar--base');
  }

  function renderHero(card) {
    const m = card.metrics;
    els.heroVersion.textContent = String(card.version || 'Unversioned');
    els.heroTrained.textContent = card.trained_at
      ? `Trained ${FMT.when(card.trained_at, { day: false, year: true }).replace(/, \d\d:\d\d UTC$/, '')}`
      : 'Not trained on this server';
    if (!m) {
      for (const k of ['heroHigh', 'heroAll', 'heroEvents', 'heroHighN', 'heroAllPct', 'heroHighPct']) {
        els[k].textContent = '—';
      }
      els.heroHighBase.textContent = '';
      els.heroAllBase.textContent = '';
      els.heroSplit.textContent = 'No held-out metrics available';
      els.barsAll.innerHTML = '';
      els.barsHigh.innerHTML = '';
      return;
    }
    const allImp = improvement(m.mse_test_all, m.baseline_mse_all);
    const highImp = improvement(m.mse_test_high_risk, m.baseline_mse_high_risk);
    els.heroAllPct.textContent = pctText(allImp);
    els.heroHighPct.textContent = pctText(highImp);
    els.heroAllPct.nextElementSibling.textContent =
      allImp !== null && allImp < 0 ? 'higher than carry-forward' : 'lower than carry-forward';
    els.heroHighPct.nextElementSibling.textContent =
      highImp !== null && highImp < 0 ? 'higher than carry-forward' : 'lower than carry-forward';
    /* the bars carry the raw errors; these lines carry the reading */
    els.heroAll.textContent = 'Mean squared error, lower is better.';
    els.heroAllBase.textContent = '';
    els.heroHigh.textContent = 'Seed-sensitive —';
    els.heroHighBase.textContent = 'read the note below.';
    bars(els.barsAll, m.mse_test_all, m.baseline_mse_all);
    bars(els.barsHigh, m.mse_test_high_risk, m.baseline_mse_high_risk);
    els.heroEvents.textContent = FMT.num(m.n_events);
    els.heroSplit.textContent = `${FMT.num(m.n_train)} to train on · ${FMT.num(m.n_test)} held out to test`;
    els.heroHighN.textContent = FMT.num(m.n_test_high_risk);
    if (finite(m.n_events)) els.proseEvents.textContent = FMT.num(m.n_events);
  }

  /* the two train.py conditions, evaluated from the card's own numbers:
     1. mse_test_high_risk <= baseline_mse_high_risk * 1.02
     2. mse_test_all       <= baseline_mse_all       * 0.90 */
  function renderGate(metrics) {
    const m = metrics || {};
    const rows = [
      [els.gate1, m.mse_test_high_risk, m.baseline_mse_high_risk, 1.02],
      [els.gate2, m.mse_test_all, m.baseline_mse_all, 0.90],
    ];
    for (const [el, value, baseline, factor] of rows) {
      const v = Number(value);
      const limit = Number(baseline) * factor;
      if (!finite(value) || !Number.isFinite(limit)) {
        el.innerHTML = '<span class="faint">—</span>';
        continue;
      }
      const ok = v <= limit;
      el.innerHTML =
        `<span class="gate-row__math">${mse(v)} ${ok ? '≤' : '>'} ${mse(limit)}</span>` +
        `<span class="chip ${ok ? 'chip--ok' : 'chip--nominal'}">${ok ? 'Passed' : 'Failed'}</span>`;
    }
  }

  function renderImportance(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      els.fiMeta.textContent = '';
      els.fiList.innerHTML =
        '<div class="empty">Feature importance appears once the trained model is loaded.</div>';
      return;
    }
    const total = rows.reduce((acc, r) => acc + (Number(r.gain) || 0), 0);
    const max = rows.reduce((acc, r) => Math.max(acc, Number(r.gain) || 0), 0);
    els.fiMeta.textContent = 'Share of model gain';
    if (els.proseFeat) els.proseFeat.textContent = WORDS[rows.length] || FMT.num(rows.length);
    els.fiList.innerHTML = rows.map((r) => {
      const gain = Number(r.gain) || 0;
      const width = max > 0 ? Math.max(1, (gain / max) * 100) : 0;
      const share = total > 0 ? (gain / total) * 100 : 0;
      const nice = FEATURE_NAMES[r.name] || String(r.name).replace(/_/g, ' ');
      return (
        `<div class="fi-row" title="Gain ${FMT.esc(gain.toFixed(1))}">` +
        `<div class="fi-name">${FMT.esc(nice)}<span class="fi-raw code">${FMT.esc(r.name)}</span></div>` +
        `<div class="fi-track"><div class="fi-bar" style="width:${width.toFixed(1)}%"></div></div>` +
        `<div class="fi-gain">${FMT.esc(share.toFixed(0))}%</div>` +
        `</div>`
      );
    }).join('');
  }

  function renderLive(live) {
    els.livePreds.textContent = FMT.num(live.predictions_total);
    els.liveConj.textContent = FMT.num(live.conjunctions_scored);
    if (live.last_prediction_at) {
      const r = FMT.rel(live.last_prediction_at);
      els.liveLast.textContent = r.charAt(0).toUpperCase() + r.slice(1);
      els.liveAsOf.textContent = FMT.when(live.last_prediction_at);
    } else {
      els.liveLast.textContent = 'Not yet';
      els.liveAsOf.textContent = 'No predictions stored';
    }
  }

  function renderCard(card) {
    els.banner.hidden = !card.heuristic;
    renderHero(card);
    renderGate(card.metrics);
    renderImportance(card.feature_importance);
    renderLive(card.live || {});
  }

  /* --- polling -------------------------------------------------------------- */

  async function refresh() {
    els.hero.classList.add('is-refetching');
    try {
      const [st, card] = await Promise.all([API.status(), fetchCard()]);
      renderStatus(st);
      renderCard(card);
      hasData = true;
      els.error.hidden = true;
    } catch (err) {
      els.error.textContent = hasData
        ? 'We couldn’t refresh the model card. Showing the last figures we received.'
        : err.status === 404
          ? 'The model card needs a newer server build.'
          : 'We couldn’t load the model card. Kessler will keep trying every minute.';
      els.error.title = err.message;
      els.error.hidden = false;
      if (!hasData) {
        els.fiList.innerHTML = '<div class="empty">The model card isn’t available right now.</div>';
      }
    } finally {
      els.hero.classList.remove('is-refetching');
    }
  }

  /* --- the official ESA benchmark (GET /model/benchmark, generated from the
     committed VIGIL Lab reports): the 2019 leaderboard with VIGIL and ESA's
     own model placed by score. Real teams keep their real rank; the two
     inserted rows say where they WOULD rank. Absent card -> section hidden. */
  async function loadBenchmark() {
    const sec = $('bench');
    let card;
    try {
      const res = await fetch('/model/benchmark');
      if (!res.ok) return;
      card = await res.json();
    } catch (_) { return; }
    const v = card.vigil || {};
    const esa = card.esa_model || {};
    const lb = Array.isArray(card.leaderboard) ? card.leaderboard : [];
    if (!finite(v.L) || !lb.length) return;
    const fix3 = (x) => (finite(x) ? Number(x).toFixed(3) : '—');
    const wouldRank = (L) => 1 + lb.filter((r) => r.L < L).length;
    const rows = lb.map((r) => ({ L: r.L, rank: String(r.rank), name: r.team, cls: '' }))
      .concat([
        { L: v.L, rank: `→ ${wouldRank(v.L)}`, cls: 'lad--vigil',
          name: 'VIGIL', note: 'locked before testing' },
        { L: esa.L, rank: `→ ${wouldRank(esa.L)}`, cls: 'lad--esa',
          name: 'ESA’s own AI model', note: `chosen from ${FMT.num(esa.models)}` },
      ])
      .filter((r) => finite(r.L))
      .sort((a, b) => a.L - b.L);
    $('bench-ladder').innerHTML = rows.map((r) =>
      `<li class="lad ${r.cls}"><span class="lad__r">${FMT.esc(r.rank)}</span>` +
      `<span class="lad__n">${FMT.esc(r.name)}${r.note ? ` <small>${FMT.esc(r.note)}</small>` : ''}</span>` +
      `<span class="lad__v">${fix3(r.L)}</span></li>`).join('');
    const b = card.benchmark || {};
    const ew = (card.early_warning || {}).research || {};
    const fa = card.false_alarms || {};
    const fid = (card.fidelity || {}).L || {};
    $('bench-n').textContent = FMT.num(b.test_events);
    $('bench-teams').textContent = FMT.num(b.teams);
    $('bench-hr').textContent = FMT.num(b.test_high_risk);
    $('bench-vigil').textContent = fix3(v.L);
    $('bench-esa').textContent = fix3(esa.L);
    $('bench-esa-n').textContent = FMT.num(esa.models);
    $('bench-fa').textContent = finite(fa.removed_pct) ? `${Math.round(fa.removed_pct)}%` : '—';
    $('bench-fa-lat').textContent = finite(fa.latest_risk_removed_pct)
      ? `${Math.round(fa.latest_risk_removed_pct)}%` : '—';
    $('bench-ew-nums').textContent = `${fix3(ew.pr_auc)} vs ${fix3(ew.latest_risk_pr_auc)} for the latest risk itself`;
    $('bench-fid').textContent = `L ${fix3(fid.lab)} vs ${fix3(fid.published)} published`;
    $('bench-fid-big').textContent = (card.fidelity || {}).reproduces_published
      ? `${fix3(fid.lab)} = ${fix3(fid.published)}` : fix3(fid.lab);
    $('bench-delta').textContent = finite(v.delta_pct_vs_baseline)
      ? `${Math.abs(v.delta_pct_vs_baseline).toFixed(1)}%` : '—';
    const ci = Array.isArray(v.ci975_pct) ? v.ci975_pct : [];
    $('bench-ci').textContent = ci.length === 2
      ? `${ci[0] < 0 ? '−' : '+'}${Math.abs(ci[0]).toFixed(1)}% to ${ci[1] < 0 ? '−' : '+'}${Math.abs(ci[1]).toFixed(1)}%`
      : '—';
    sec.hidden = false;
  }

  refresh();
  loadBenchmark();
  setInterval(refresh, POLL_MS);
})();
