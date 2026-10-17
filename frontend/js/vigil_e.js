/* Kessler - VIGIL-E early-warning card for the Model page.

   Renders GET /model/card -> early_warning into <section id="vigil-e-slot">.
   Every figure comes from that payload (the training meta json the ship
   gate judged, plus live counts); nothing is hardcoded. When the section is
   null - no trained, gate-passing artifact - the card says so plainly and
   shows the command that trains it. No slot on the page: does nothing.

   Self-contained: one small scoped stylesheet built from main.css tokens,
   no dependencies beyond the browser (FMT.esc is used when present). */
'use strict';

(() => {
  const slot = document.getElementById('vigil-e-slot');
  if (!slot) return;


  const esc = (window.FMT && FMT.esc) ? FMT.esc : (s) => String(s ?? '')
    .replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const finite = (v) => v !== null && v !== undefined && Number.isFinite(Number(v));
  const dash = '—';
  const fix = (v, d = 2) => (finite(v) ? Number(v).toFixed(d) : dash);
  const pct = (v) => (finite(v) ? `${Math.round(Number(v) * 100)}%` : dash);
  const hours = (v) => (finite(v) ? `${Math.round(Number(v))}` : dash);
  const num = (v) => (finite(v) ? Number(v).toLocaleString('en-US') : dash);

  function day(iso) {
    if (!iso) return dash;
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return dash;
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
  }

  /* "1 in a million" for -6; generic 10^n otherwise */
  function thresholdWords(logPc) {
    if (!finite(logPc)) return 'the high-risk threshold';
    const n = Number(logPc);
    if (n === -6) return '1 in a million';
    return `10<sup>${n}</sup>`;
  }

  function injectStyle() {
    if (document.getElementById('vigil-e-style')) return;
    const css = `
#vigil-e-slot { margin: 16px 0; }
.ve-lede { font-size: 15px; line-height: 1.55; color: var(--k-text-2); max-width: 72ch; margin: 2px 0 18px; }
.ve-lede strong { color: var(--k-text); font-weight: 600; }
.ve-hero { display: flex; align-items: baseline; flex-wrap: wrap; gap: 6px 14px; margin-bottom: 18px; }
.ve-hero__num { font-size: clamp(40px, 5vw, 56px); font-weight: 700; letter-spacing: -0.035em; line-height: 1;
  font-variant-numeric: tabular-nums; color: var(--k-text); }
.ve-hero__unit { font-size: 20px; font-weight: 600; color: var(--k-text-2); letter-spacing: -0.01em; }
.ve-hero__txt { font-size: 15px; color: var(--k-text-2); max-width: 46ch; }
.ve-stats { margin-bottom: 20px; }
.ve-stats .stat { box-shadow: none; background: var(--k-surface-2); border-color: transparent; }
.ve-stats .stat__label, .ve-stats .stat__sub { white-space: normal; }
.ve-block { margin-top: 22px; }
.ve-block__title { font-size: 13px; font-weight: 600; color: var(--k-text); margin-bottom: 8px; }
.ve-tbl-wrap { overflow-x: auto; margin: 0 -18px; }
.ve-tbl-wrap .tbl td.is-model { font-weight: 600; }
.ve-gate { list-style: none; display: grid; gap: 0; }
.ve-gate li { display: flex; align-items: baseline; gap: 12px; padding: 9px 0; border-bottom: 1px solid var(--k-grid);
  font-size: 13.5px; color: var(--k-text-2); }
.ve-gate li:last-child { border-bottom: 0; }
.ve-gate__mark { flex: none; width: 58px; font-size: 12px; font-weight: 600; color: var(--k-text-3); }
.ve-gate__mark.is-pass { color: var(--k-text); }
.ve-seeds { display: flex; flex-wrap: wrap; gap: 8px; }
.ve-foot { margin-top: 18px; font-size: 12.5px; color: var(--k-text-3); line-height: 1.55; max-width: 90ch; }
.ve-cmd { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 14px 0 6px; }
.ve-cmd code { font-family: var(--k-code); font-size: 13px; color: var(--k-text); background: var(--k-surface-2);
  border-radius: var(--k-radius-sm); padding: 8px 12px; overflow-x: auto; max-width: 100%; white-space: nowrap; }
.ve-empty { padding-top: 6px; }
.ve-empty__title { font-size: 20px; font-weight: 650; letter-spacing: -0.018em; margin-bottom: 6px; }
@media (max-width: 700px) { .ve-hero__num { font-size: 40px; } }`;
    const el = document.createElement('style');
    el.id = 'vigil-e-style';
    el.textContent = css;
    document.head.appendChild(el);
  }

  function header(meta) {
    return `<div class="panel__hd">
      <h2 class="panel__title">Early warning</h2>
      <span class="panel__meta">${meta}</span>
    </div>`;
  }

  function renderUntrained(reason) {
    slot.innerHTML = `<div class="panel k-rise">
      ${header('VIGIL-E')}
      <div class="panel__bd ve-empty">
        <div class="ve-empty__title">Not in the live console yet</div>
        <p class="ve-lede">VIGIL-E estimates, for events still below 1 in a million,
        the chance that the final message crosses it — before the risk number itself does.
        ${esc(reason)}</p>
        <p class="ve-foot">Its research build passed its pre-registered test on ESA’s data, but a
        re-check found that test’s baseline weakened by how its scores were pooled: against the
        latest risk itself, VIGIL-E ranks quiet events better overall but not at the top of the
        list (see the benchmark above). The version built on the features Kessler computes scores
        below the latest risk itself, so no early-warning number is shown on live events.</p>
      </div>
    </div>`;
  }

  function renderError(message) {
    slot.innerHTML = `<div class="panel">${header('VIGIL-E')}
      <div class="panel__bd"><div class="empty"><strong>Couldn’t load the early-warning card</strong>${esc(message)}</div></div></div>`;
  }

  const GATE_WORDS = {
    validation_has_both_classes: 'Validation set holds both escalating and non-escalating events',
    pr_auc_beats_logistic: 'Ranks escalations better than the latest-risk baseline, by a fixed margin',
    median_lead_time_positive: 'Warns earlier than carry-forward on the events it catches',
    early_recall_not_below_logistic: 'Catches at least as many escalations early as the baseline',
    seed_check_beats_logistic_every_seed: 'Beats the baseline on every re-split, not just one lucky one',
  };

  function renderTrained(ew) {
    const m = ew.metrics || {};
    const base = ew.baselines || {};
    const lg = base.logistic_latest_risk || {};
    const cf = base.carry_forward || {};
    const live = ew.live || {};
    const thr = thresholdWords(ew.threshold_log10_pc);
    const lead = m.median_lead_time_h;
    const budget = finite(ew.alert_budget) ? pct(ew.alert_budget) : dash;
    /* the measured held-out alert rate, not the training budget */
    const alertRate = finite(m.alert_rate) ? pct(m.alert_rate) : budget;
    const caught = (finite(m.frac_flagged_early) && finite(m.n_lead_eligible_test))
      ? `${Math.round(m.frac_flagged_early * m.n_lead_eligible_test)} of ${num(m.n_lead_eligible_test)}` : dash;

    const rows = [
      ['VIGIL-E', m, true],
      ['Latest risk only (logistic)', lg, false],
      ['Carry-forward rule', cf, false],
    ].map(([name, r, isModel]) => `<tr>
        <td class="${isModel ? 'is-model' : ''}">${esc(name)}</td>
        <td class="num">${fix(r.pr_auc)}</td>
        <td class="num">${fix(r.roc_auc)}</td>
        <td class="num">${pct(r.frac_flagged_early)}</td>
        <td class="num">${finite(r.median_lead_time_h) ? `${hours(r.median_lead_time_h)} h` : dash}</td>
      </tr>`).join('');

    const gate = (ew.gate && Array.isArray(ew.gate.checks)) ? ew.gate.checks : [];
    const gateHtml = gate.map((c) => `<li>
        <span class="ve-gate__mark ${c.passed ? 'is-pass' : ''}">${c.passed ? 'Passed' : 'Failed'}</span>
        <span>${esc(GATE_WORDS[c.name] || c.detail || c.name)}</span>
      </li>`).join('');

    const seeds = Array.isArray(ew.seed_check) ? ew.seed_check : [];
    const seedHtml = seeds.map((s) => (s.valid
      ? `<span class="chip chip--type" title="Precision–recall AUC, VIGIL-E vs latest-risk baseline">Seed ${esc(s.seed)} · ${fix(s.pr_auc)} vs ${fix(s.logistic_pr_auc)}</span>`
      : `<span class="chip chip--type">Seed ${esc(s.seed)} · not evaluable</span>`)).join('');

    slot.innerHTML = `<div class="panel k-rise">
      ${header(`${esc(ew.version)} · trained ${esc(day(ew.trained_at))}`)}
      <div class="panel__bd">
        <p class="ve-lede">For an event whose risk is still below <strong>${thr}</strong>,
        VIGIL-E estimates the chance that its final message crosses it. Waiting for the
        number itself — the carry-forward rule — only flags an event once it has already crossed.</p>

        <div class="ve-hero">
          <span class="ve-hero__num">${hours(lead)}</span><span class="ve-hero__unit">hours earlier</span>
          <span class="ve-hero__txt">Median head start over carry-forward on the ${caught} held-out escalations
          it caught early, while alerting on ${alertRate} of held-out events.</span>
        </div>

        <div class="stats ve-stats">
          <div class="stat">
            <div class="stat__label">Escalations alerted</div>
            <div class="stat__value stat__value--sm">${pct(m.recall_at_budget)}</div>
            <div class="stat__sub">of escalating events, at a ${budget} alert budget</div>
          </div>
          <div class="stat">
            <div class="stat__label">Alerts that escalated</div>
            <div class="stat__value stat__value--sm">${pct(m.precision_at_budget)}</div>
            <div class="stat__sub">precision at that budget</div>
          </div>
          <div class="stat">
            <div class="stat__label">Ranking quality</div>
            <div class="stat__value stat__value--sm">${fix(m.pr_auc)}</div>
            <div class="stat__sub">PR-AUC · baseline ${fix(lg.pr_auc)}</div>
          </div>
          <div class="stat">
            <div class="stat__label">Carry-forward flags at</div>
            <div class="stat__value stat__value--sm">${hours(m.median_hours_before_tca_carry_forward)} h</div>
            <div class="stat__sub">before closest approach (median)</div>
          </div>
        </div>

        <div class="ve-block">
          <div class="ve-block__title">Against honest baselines</div>
          <div class="ve-tbl-wrap"><table class="tbl">
            <thead><tr>
              <th>Method</th><th class="num" title="Precision–recall area under curve">PR-AUC</th>
              <th class="num" title="Receiver operating characteristic area under curve">ROC-AUC</th>
              <th class="num">Caught early</th><th class="num">Median lead</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table></div>
        </div>

        <div class="cols cols--2 ve-block">
          <div>
            <div class="ve-block__title">Ship gate</div>
            <ul class="ve-gate">${gateHtml || `<li><span>${dash}</span></li>`}</ul>
          </div>
          <div>
            <div class="ve-block__title">Seed check</div>
            <div class="ve-seeds">${seedHtml || dash}</div>
            <p class="ve-foot">The whole protocol re-run on fresh splits: PR-AUC of VIGIL-E vs the
            latest-risk baseline. The gate requires a win on every one.</p>
          </div>
        </div>

        <p class="ve-foot">Held out: ${num(m.n_events_test)} Kelvins events, ${num(m.n_escalating_test)} of them
        escalating (base rate ${pct(m.base_rate)}). The alert threshold (probability ≥ ${fix(ew.alert_threshold, 3)})
        was chosen on a validation carve inside the training events; the held-out set chose nothing.
        Live: ${num(live.conjunctions_scored)} events scored${live.last_prediction_at ? `, last ${esc(day(live.last_prediction_at))}` : ''}.
        Trained on operator CDMs with real covariances; live screening Pc rests on assumed uncertainty,
        so treat the probability as trend guidance.</p>
      </div>
    </div>`;
  }

  async function load() {
    injectStyle();
    let card;
    try {
      const res = await fetch('/model/card');
      if (!res.ok) throw new Error(`The server answered ${res.status}.`);
      card = await res.json();
    } catch (err) {
      renderError(err && err.message ? ` ${err.message}` : '');
      return;
    }
    if (!card || card.early_warning === undefined) {
      renderUntrained('This server doesn’t support early warning yet.');
    } else if (card.early_warning === null) {
      renderUntrained('');
    } else {
      renderTrained(card.early_warning);
    }
  }

  load();
})();
