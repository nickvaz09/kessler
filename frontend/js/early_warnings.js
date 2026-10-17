/* ===========================================================================
   KESSLER - Early warnings (VIGIL-E) on the Alerts page

   Events that are still nominal (Pc < 1e-6) but that the VIGIL-E classifier
   expects to escalate to high risk by their final message, at or above the
   model's own alert threshold (GET /conjunctions?early_warning=true). The
   panel stays hidden when no escalation model is trained or nothing is
   flagged - it never shows placeholder warnings.
   ======================================================================== */
(function () {
  'use strict';

  const anchor = document.querySelector('.panel--reentry');
  if (!anchor) return;

  const panel = document.createElement('section');
  panel.className = 'panel panel--flush ew-panel';
  panel.hidden = true;
  panel.setAttribute('aria-labelledby', 'ew-title');
  panel.innerHTML =
    '<div class="panel__hd">' +
      '<h2 class="panel__title" id="ew-title">Early warnings</h2>' +
      '<span class="panel__meta" id="ew-meta"></span>' +
    '</div>' +
    '<p class="ew-lede">Still below the alert line today, but VIGIL-E expects these to become high-risk before closest approach.</p>' +
    '<div class="ew-list" id="ew-list"></div>';
  anchor.parentNode.insertBefore(panel, anchor);

  const style = document.createElement('style');
  style.textContent = `
    .ew-panel { margin-top: 16px; }
    .ew-lede { padding: 0 18px 10px; color: var(--k-text-2); font-size: 13.5px; }
    .ew-row { display: grid; grid-template-columns: minmax(0, 2fr) 1fr 1fr auto; gap: 16px;
      align-items: center; padding: 14px 18px; border-top: 1px solid var(--k-grid); }
    .ew-pair { font-weight: 600; font-size: 15px; letter-spacing: -0.01em; min-width: 0;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .ew-k { font-size: 12px; color: var(--k-text-3); }
    .ew-v { font-size: 17px; font-weight: 650; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
    .ew-v--p { color: var(--k-accent); }
    .ew-bar { height: 4px; border-radius: 4px; background: var(--k-surface-3); margin-top: 6px; overflow: hidden; }
    .ew-bar > i { display: block; height: 100%; background: var(--k-accent); border-radius: 4px; }
    @media (max-width: 700px) {
      .ew-row { grid-template-columns: 1fr 1fr; }
      .ew-pair { grid-column: 1 / -1; }
    }`;
  document.head.appendChild(style);

  const esc = (v) => (window.FMT ? FMT.esc(v) : String(v));

  function row(c) {
    const p = Math.max(0, Math.min(1, Number(c.escalation_prob) || 0));
    const pct = Math.round(p * 100);
    const when = window.FMT && FMT.rel ? FMT.rel(c.tca) : c.tca;
    return (
      '<div class="ew-row">' +
        `<div class="ew-pair"><a href="/object.html?id=${esc(c.primary.norad_id)}">${esc(c.primary.name)}</a>` +
        ` <span class="faint">and</span> <a href="/object.html?id=${esc(c.secondary.norad_id)}">${esc(c.secondary.name)}</a></div>` +
        `<div><div class="ew-k">Chance it escalates</div><div class="ew-v ew-v--p">${pct}%</div>` +
        `<div class="ew-bar" aria-hidden="true"><i style="width:${pct}%"></i></div></div>` +
        `<div><div class="ew-k">Closest approach</div><div class="ew-v">${esc(when)}</div></div>` +
        `<a class="btn btn--xs" href="/?conj=${esc(c.id)}">Globe</a>` +
      '</div>'
    );
  }

  async function load() {
    try {
      const res = await fetch('/conjunctions?status=active&early_warning=true&limit=20');
      if (!res.ok) throw new Error(String(res.status));
      const body = await res.json();
      const items = Array.isArray(body.items) ? body.items : [];
      if (!items.length) { panel.hidden = true; return; }
      const version = items[0].escalation_model_version || 'VIGIL-E';
      document.getElementById('ew-meta').textContent =
        `${body.count} flagged · ${version}`;
      document.getElementById('ew-list').innerHTML = items.map(row).join('');
      panel.hidden = false;
    } catch (_) {
      panel.hidden = true; /* no warnings is never shown as an error here */
    }
  }

  load();
  setInterval(load, 60000);
})();
