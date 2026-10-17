# VIGIL-E - early-warning escalation model (contract)

**What it does.** For a conjunction whose latest observed risk is still *below*
the high-risk threshold (log10 Pc < -6, `risk_model.high_risk_threshold_log10_pc()`),
VIGIL-E estimates the probability that the event's **final** CDM lands at or above
it. The carry-forward rule (and VIGIL-0.4, which defers to carry-forward at
Pc >= 1e-6) only reacts once the latest risk has already crossed. The headline
metric is how many hours earlier VIGIL-E warns.

## Files

| Path | Role |
|---|---|
| `backend/app/models/escalation.py` | Features (`escalation_features` = `features.build_features` + 3 extras), `EscalationModel.load/predict/predict_proba/explain`, `MODEL_VERSION = "vigil-e-0.1"` |
| `backend/app/models/train_escalation.py` | Training CLI, ship gate, metrics, seed check |
| `data/models/vigil_e.json`, `data/models/vigil_e.meta.json` | Artifact + meta, written **only** when the ship gate passes. Not in the repo until the owner trains. |
| `frontend/js/vigil_e.js` | Renders `/model/card → early_warning` into `<section id="vigil-e-slot">` |

## Training (owner, locally, with the Kelvins CSV)

```
.venv/bin/python -m backend.app.models.train_escalation --data data/kelvins/train_data.csv
# optional: --seed 42 (default) --report /tmp/vigil_e_report.json (full evaluation, written even if not shipped)
```

Protocol, fixed in code before evaluation:
- Decision points = every prefix of an event's CDM sequence (before its final CDM)
  whose latest risk < -6. Label = final risk >= -6. Each event's points share weight 1.
- Seeded, label-stratified, **event-level** 80/20 split; the 20% held-out set selects nothing.
  A seeded event-level validation carve (20% of train) drives early stopping, the Platt
  calibration (the booster is class-weighted, so its raw output is not a probability) and the
  alert threshold (the probability at which 5% of validation events would be alerted - `ALERT_BUDGET`).
- Baselines on the same held-out events: carry-forward rule; logistic regression on latest log10 risk.
- **Ship gate** (`ship_gate()`): validation holds both classes; PR-AUC >= logistic + 0.02;
  median lead time over early-flagged escalations > 0 h; early catch rate >= logistic's;
  PR-AUC beats logistic on **every** seed in `SEED_CHECK = (7, 1234, 2027)`.
  Failing -> any previous artifact is deleted and nothing is saved.
- Lead time: for held-out escalating events that start below -6, carry-forward flags at the first CDM
  with risk >= -6; VIGIL-E flags at its first decision point with p >= alert threshold.
  `median_lead_time_h` = median over events flagged early; `frac_flagged_early` says how many;
  `median_lead_time_h_all` counts the rest as 0. Quote them together.

## Storage

`risk_predictions` gains two nullable columns (both dialects; `db._migrate_add_columns`
adds them to existing databases at `init_db`):
`escalation_prob REAL/DOUBLE PRECISION`, `escalation_model_version TEXT`.
Written by `risk_model.predict_for_conjunction(..., escalation_model=...)`, which
`pipeline._predict_active` calls with `EscalationModel.load()` (None without artifact).

Semantics of the pair:

| escalation_prob | escalation_model_version | meaning |
|---|---|---|
| null | null | no VIGIL-E artifact (or scoring failed) |
| null | `"vigil-e-0.1"` | model ran and declined: latest risk already >= -6 (or no CDMs) - the event is already high-risk |
| 0..1 | `"vigil-e-0.1"` | calibrated probability the final CDM reaches >= -6 |

## API

- `GET /conjunctions` items, `GET /conjunctions/{id}`, `GET /risk/{id}`: add
  `escalation_prob` (float|null) and `escalation_model_version` (string|null), from the
  latest stored prediction row (updated each refresh).
- `GET /risk/{id}/escalation` (new): live glass box -
  `{conjunction_id, threshold_log10_pc, model_version, escalation_prob, mode, note}` with
  `mode ∈ unavailable | not_applicable | early_warning`; in `early_warning` also
  `alert_threshold, base_value, contributions:[{name, value, contribution}]`
  (SHAP, log-odds units before calibration, sorted by |contribution|).
- `GET /model/card` gains `early_warning`: `null` without a loadable artifact, else
  ```
  {version, target, trained_at, dataset, threshold_log10_pc, alert_budget, alert_threshold,
   metrics: {roc_auc, pr_auc, brier, base_rate, recall_at_budget, precision_at_budget, alert_rate,
             median_lead_time_h, median_lead_time_h_all, frac_flagged_early,
             median_hours_before_tca_model, median_hours_before_tca_carry_forward,
             n_events_test, n_escalating_test, n_lead_eligible_test, n_decision_points_test},
   baselines: {logistic_latest_risk: {...same metric names...}, carry_forward: {...}},
   gate: {checks: [{name, passed, detail}], passed},
   seed_check: [{seed, valid, roc_auc, pr_auc, logistic_pr_auc, recall_at_budget,
                 frac_flagged_early, median_lead_time_h}],
   data: {...split sizes...}, feature_importance: [{name, gain}],
   live: {predictions_total, conjunctions_scored, last_prediction_at}}
  ```
  Metric values are the training meta verbatim (nulls where undefined).

## Frontend integration

Add to `model.html`: the slot where the card should appear, and after `js/model.js`:

```html
<section id="vigil-e-slot"></section>
...
<script src="js/vigil_e.js"></script>
```

The module injects its own small token-based stylesheet (`#vigil-e-style`), fetches
`/model/card` once, and renders either the trained card (lead-time hero, stats, baseline
table, ship gate, seed check) or "Not trained yet - run
`python -m backend.app.models.train_escalation`". Without the slot it does nothing.

## Limitations

Trained on operator CDMs with real covariances; live screening Pc uses assumed TLE-class
uncertainty, so live probabilities are trend guidance. Kelvins features beyond the CDM
fields Kessler stores (covariance terms, object metadata) are deliberately unused so that
training and serving share one feature function.
