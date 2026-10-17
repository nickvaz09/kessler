"""Train VIGIL-E, the early-warning escalation classifier, on ESA Kelvins.

CLI (run from the repo root):
    .venv/bin/python -m backend.app.models.train_escalation \\
        --data data/kelvins/train_data.csv

Question answered: for an event whose latest risk is still BELOW the
high-risk threshold (log10 Pc < -6), will its FINAL CDM be at/above it?

Dataset (one row per DECISION POINT, grouped by event):
- Events: Kelvins rows grouped by event_id, ascending time (descending
  time_to_tca), >= 2 CDMs (same rule as train.py).
- Decision points: every prefix of the sequence that ends before the final
  CDM and whose latest risk is below the threshold - each is a moment an
  operator could have asked "should I worry now?". Features come from
  models.escalation.escalation_features (the serving function) on that
  prefix; the label is the event's final risk >= threshold.
- Weights: each event's decision points share weight 1 (1/n per point), so
  long sequences do not dominate either training or the metrics.

Protocol (fixed before any evaluation - see SHIP GATE below):
1. Seeded, event-level, label-stratified split: 20% of events held out as
   the test set. The held-out set is never used to select anything.
2. Inside the remaining 80%, a seeded event-level validation carve (20%)
   drives early stopping, the Platt calibration of the class-weighted
   booster, and the alert threshold (the probability above which the
   ALERT_BUDGET share of validation events would be alerted).
3. Class imbalance: positive decision points are up-weighted so both
   classes carry equal total weight; calibration on the (unweighted)
   validation carve then maps the booster margin back to a probability.
4. Baselines, scored on the same held-out events:
   - carry-forward rule: flags only once the latest risk crosses the
     threshold - at every decision point it says "no", so its early recall
     and lead time are 0 by construction, ROC-AUC 0.5, PR-AUC = base rate;
   - logistic regression on the latest log10 risk alone (the obvious
     "watch the number" model), with its own validation-chosen threshold.
5. Lead time - the headline. For each held-out escalating event that
   starts below the threshold, carry-forward first flags it at the first
   CDM with risk >= threshold. The model flags it at its first decision
   point with probability >= the alert threshold. Lead = hours between the
   two when the model is earlier (else 0). ``median_lead_time_h`` is the
   median over events the model flagged early; ``frac_flagged_early`` says
   how many that is; ``median_lead_time_h_all`` counts the rest as 0.
6. Seed-sensitivity: the whole protocol is re-run with SEED_CHECK seeds
   (fresh split and fresh model each) and recorded in the meta json.

Artifacts: data/models/vigil_e.json (booster) + vigil_e.meta.json, written
only when the ship gate passes; a failing run deletes any previous artifact
so the API serves null rather than a stale model. Exits 0 with download
instructions when the CSV is absent.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from backend.app import config
from backend.app.models import escalation
from backend.app.models.escalation import ESCALATION_FEATURE_NAMES

_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
_TCA_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)  # as train.py
_TCA_ISO = _TCA_EPOCH.strftime(_ISO_FMT)

REQUIRED_COLUMNS = ["event_id", "time_to_tca", "miss_distance",
                    "relative_speed", "risk"]

TEST_FRACTION = 0.2
VAL_FRACTION = 0.2          # of the non-test events
ALERT_BUDGET = 0.05         # share of (validation) events an operator accepts alerts on
SEED_CHECK = (7, 1234, 2027)

# --- SHIP GATE - fixed in code before the model was ever evaluated ----------
# 1. the validation carve holds both classes (else calibration/threshold are
#    undefined and nothing may ship);
# 2. held-out PR-AUC beats the latest-risk logistic baseline by at least
#    GATE_PR_AUC_MARGIN (absolute);
# 3. held-out median lead time over early-flagged escalations > 0 h;
# 4. at the same alert budget, the model flags at least as many held-out
#    escalations early as the logistic baseline does;
# 5. seed check: the model's PR-AUC beats the logistic baseline on every
#    SEED_CHECK re-run (a lucky split must not ship a model).
GATE_PR_AUC_MARGIN = 0.02

# Fixed a priori (no search on any split): shallow trees, strong
# regularization, early stopping on the validation carve.
XGB_PARAMS = dict(
    n_estimators=2000, max_depth=4, learning_rate=0.03, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, gamma=0.0,
    objective="binary:logistic", eval_metric="aucpr",
    early_stopping_rounds=100, n_jobs=4,
)

MISSING_CSV_MESSAGE = """\
Kelvins training data not found at: {path}

VIGIL-E trains on the ESA Kelvins Collision Avoidance Challenge dataset.
Download it from its public Zenodo archive (no account needed):
    https://zenodo.org/record/4463683
unzip it so train_data.csv sits at data/kelvins/train_data.csv (or pass
--data <path>), then re-run:
    .venv/bin/python -m backend.app.models.train_escalation --data data/kelvins/train_data.csv

Until then every escalation field in the API is null (no artifact).\
"""


# --- dataset -----------------------------------------------------------------


def build_decision_points(df, threshold: float) -> dict:
    """Decision-point dataset from a Kelvins dataframe (see module doc).

    Returns a dict of numpy arrays:
      X (P, F), point_event (P,), point_ttca_days (P,), point_weight (P,),
      y (P,) - and per event (E,): event_label, event_cf_ttca_days (time to
      TCA at carry-forward's first flag; nan if the risk never crossed),
      event_lead_eligible (escalates and starts below the threshold),
      event_ids.
    Events with no decision point (e.g. above the threshold throughout) are
    not part of the task and are dropped.
    """
    X, p_event, p_ttca = [], [], []
    e_label, e_cf, e_elig, e_ids = [], [], [], []
    for event_id, group in df.groupby("event_id", sort=True):
        group = group.sort_values("time_to_tca", ascending=False)
        if len(group) < 2:
            continue
        rows = group.to_dict("records")
        risks = [float(r["risk"]) for r in rows]
        ttcas = [float(r["time_to_tca"]) for r in rows]
        cdms = [{
            "message_time": (_TCA_EPOCH - timedelta(days=t)).strftime(_ISO_FMT),
            "miss_distance_m": float(r["miss_distance"]),
            "relative_velocity_ms": float(r["relative_speed"]),
            "probability_of_collision": 10.0 ** rk,
        } for r, rk, t in zip(rows, risks, ttcas)]
        label = risks[-1] >= threshold
        points = [k for k in range(len(rows) - 1) if risks[k] < threshold]
        if not points:
            continue
        e = len(e_label)
        for k in points:
            feats = escalation.escalation_features(cdms[:k + 1], _TCA_ISO)
            X.append([feats[name] for name in ESCALATION_FEATURE_NAMES])
            p_event.append(e)
            p_ttca.append(ttcas[k])
        crossed = [i for i, rk in enumerate(risks) if rk >= threshold]
        e_label.append(label)
        e_cf.append(ttcas[crossed[0]] if crossed else float("nan"))
        e_elig.append(bool(label and risks[0] < threshold))
        e_ids.append(event_id)

    p_event = np.asarray(p_event, dtype=int)
    e_label = np.asarray(e_label, dtype=bool)
    counts = np.bincount(p_event, minlength=len(e_label)) if len(p_event) else np.zeros(0)
    return {
        "X": np.asarray(X, dtype=float).reshape(-1, len(ESCALATION_FEATURE_NAMES)),
        "point_event": p_event,
        "point_ttca_days": np.asarray(p_ttca, dtype=float),
        "point_weight": (1.0 / counts[p_event]) if len(p_event) else np.zeros(0),
        "y": e_label[p_event].astype(int) if len(p_event) else np.zeros(0, int),
        "event_label": e_label,
        "event_cf_ttca_days": np.asarray(e_cf, dtype=float),
        "event_lead_eligible": np.asarray(e_elig, dtype=bool),
        "event_ids": e_ids,
    }


def split_events(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded, label-stratified, event-level (fit, val, test) index split."""
    rng = np.random.RandomState(seed)
    fit, val, test = [], [], []
    for cls in (True, False):
        idx = np.flatnonzero(labels == cls)
        idx = idx[rng.permutation(len(idx))]
        n_test = int(round(len(idx) * TEST_FRACTION))
        rest = idx[n_test:]
        n_val = int(round(len(rest) * VAL_FRACTION))
        test.extend(idx[:n_test])
        val.extend(rest[:n_val])
        fit.extend(rest[n_val:])
    return (np.sort(np.asarray(fit, int)), np.sort(np.asarray(val, int)),
            np.sort(np.asarray(test, int)))


# --- metrics -----------------------------------------------------------------


def _nan_to_none(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _auc_metrics(y, score, w) -> dict:
    """Weighted ROC-AUC / PR-AUC (average precision); nan with one class."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, score, sample_weight=w)),
            "pr_auc": float(average_precision_score(y, score, sample_weight=w))}


def _event_max(values: np.ndarray, point_event: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Per-event max of a per-point value, for the given event ids."""
    out = np.full(len(events), -np.inf)
    pos = {e: i for i, e in enumerate(events)}
    for v, e in zip(values, point_event):
        i = pos.get(e)
        if i is not None and v > out[i]:
            out[i] = v
    return out


def alert_threshold(prob: np.ndarray, point_event: np.ndarray,
                    events: np.ndarray, budget: float = ALERT_BUDGET) -> float:
    """Probability threshold at which ``budget`` of ``events`` get alerted
    (an event is alerted when any of its decision points reaches it)."""
    peak = _event_max(prob, point_event, events)
    return float(np.quantile(peak, 1.0 - budget, method="higher"))


def event_alert_metrics(prob, point_event, events, labels, tau) -> dict:
    """Event-level recall/precision/alert-rate at threshold tau."""
    peak = _event_max(prob, point_event, events)
    alerted = peak >= tau
    lab = labels[events]
    n_pos = int(lab.sum())
    return {
        "recall_at_budget": float(alerted[lab].mean()) if n_pos else float("nan"),
        "precision_at_budget": (float(lab[alerted].mean())
                                if alerted.any() else float("nan")),
        "alert_rate": float(alerted.mean()) if len(events) else float("nan"),
    }


def lead_time_metrics(prob, point_event, point_ttca, data, events, tau) -> dict:
    """Lead time of the model's first alert vs carry-forward's first flag,
    over the given escalating events that start below the threshold.
    ``prob``/``point_event``/``point_ttca`` are aligned per decision point."""
    elig = [e for e in events if data["event_lead_eligible"][e]]
    by_event: dict[int, list[tuple[float, float]]] = {e: [] for e in elig}
    for p, e, t in zip(prob, point_event, point_ttca):
        if e in by_event:
            by_event[e].append((t, p))
    leads, model_h, cf_h = [], [], []
    for e in elig:
        cf_t = float(data["event_cf_ttca_days"][e])
        first = None
        # descending time_to_tca == ascending wall-clock time
        for t, p in sorted(by_event[e], key=lambda tp: -tp[0]):
            if p >= tau and t > cf_t:
                first = t
                break
        lead = (first - cf_t) * 24.0 if first is not None else 0.0
        leads.append(lead)
        model_h.append(max(first if first is not None else cf_t, cf_t) * 24.0)
        cf_h.append(cf_t * 24.0)
    leads = np.asarray(leads, dtype=float)
    early = leads[leads > 0.0]
    return {
        "median_lead_time_h": float(np.median(early)) if len(early) else float("nan"),
        "median_lead_time_h_all": float(np.median(leads)) if len(leads) else float("nan"),
        "frac_flagged_early": float(len(early) / len(leads)) if len(leads) else float("nan"),
        "median_hours_before_tca_model": float(np.median(model_h)) if model_h else float("nan"),
        "median_hours_before_tca_carry_forward": float(np.median(cf_h)) if cf_h else float("nan"),
        "n_lead_eligible_test": int(len(leads)),
    }


# --- one full protocol run ----------------------------------------------------


def _fit_logistic(x, y, w):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(x.reshape(-1, 1), y, sample_weight=w)
    return lr


def run_protocol(data: dict, seed: int) -> dict:
    """Split, train, calibrate, pick the alert threshold (all inside the
    training events) and score once on the held-out events.

    Returns {"booster", "calibration", "alert_threshold", "best_iteration",
    "metrics", "baselines", "val_has_both_classes", "split"}; "booster" is
    None when the validation carve lacks a class (nothing to calibrate).
    """
    import xgboost as xgb

    X, y, w = data["X"], data["y"], data["point_weight"]
    pe = data["point_event"]
    labels = data["event_label"]
    fit_e, val_e, test_e = split_events(labels, seed)
    in_fit, in_val, in_test = (np.isin(pe, fit_e), np.isin(pe, val_e),
                               np.isin(pe, test_e))
    split = {"n_events_fit": int(len(fit_e)), "n_events_val": int(len(val_e)),
             "n_events_test": int(len(test_e))}
    val_both = (len(np.unique(y[in_val])) == 2 and len(np.unique(y[in_fit])) == 2)
    result = {"booster": None, "calibration": None, "alert_threshold": None,
              "best_iteration": None, "metrics": None, "baselines": None,
              "val_has_both_classes": bool(val_both), "split": split}
    if not val_both:
        return result

    # class balance on the fit set: positives up-weighted to equal total weight
    wf, yf = w[in_fit], y[in_fit]
    pos_w = float(wf[yf == 0].sum() / max(wf[yf == 1].sum(), 1e-12))
    sw = wf * np.where(yf == 1, pos_w, 1.0)
    clf = xgb.XGBClassifier(random_state=seed, **XGB_PARAMS)
    clf.fit(X[in_fit], yf, sample_weight=sw,
            eval_set=[(X[in_val], y[in_val])],
            sample_weight_eval_set=[w[in_val]], verbose=False)
    best = int(getattr(clf, "best_iteration", XGB_PARAMS["n_estimators"] - 1))
    booster = clf.get_booster()[: best + 1]

    def margins(mask):
        return booster.predict(xgb.DMatrix(X[mask]), output_margin=True)

    # Platt calibration on the validation carve (event-weighted points)
    from sklearn.linear_model import LogisticRegression
    m_val = margins(in_val)
    platt = LogisticRegression(C=1e6, max_iter=1000)
    platt.fit(m_val.reshape(-1, 1), y[in_val], sample_weight=w[in_val])
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    p_val = escalation.calibrate(m_val, a, b)
    tau = alert_threshold(p_val, pe[in_val], val_e)

    # held-out scoring - done once, nothing below feeds back into a choice
    p_test = escalation.calibrate(margins(in_test), a, b)
    yt, wt, pet = y[in_test], w[in_test], pe[in_test]
    ttt = data["point_ttca_days"][in_test]
    metrics = _auc_metrics(yt, p_test, wt)
    metrics["brier"] = float(np.average((p_test - yt) ** 2, weights=wt))
    metrics["base_rate"] = float(labels[test_e].mean())
    metrics.update(event_alert_metrics(p_test, pet, test_e, labels, tau))
    metrics.update(lead_time_metrics(p_test, pet, ttt, data, test_e, tau))
    metrics["n_events_test"] = int(len(test_e))
    metrics["n_escalating_test"] = int(labels[test_e].sum())
    metrics["n_decision_points_test"] = int(in_test.sum())

    # baseline: logistic regression on the latest log10 risk only
    li = ESCALATION_FEATURE_NAMES.index("log10_pc_latest")
    lr = _fit_logistic(X[in_fit, li], yf, wf)
    lp_val = lr.predict_proba(X[in_val, li].reshape(-1, 1))[:, 1]
    ltau = alert_threshold(lp_val, pe[in_val], val_e)
    lp_test = lr.predict_proba(X[in_test, li].reshape(-1, 1))[:, 1]
    logistic = _auc_metrics(yt, lp_test, wt)
    logistic["brier"] = float(np.average((lp_test - yt) ** 2, weights=wt))
    logistic.update(event_alert_metrics(lp_test, pet, test_e, labels, ltau))
    logistic.update(lead_time_metrics(lp_test, pet, ttt, data, test_e, ltau))

    # baseline: the carry-forward rule (never alerts below the threshold)
    cf_lead = lead_time_metrics(np.zeros(len(pet)), pet, ttt, data, test_e,
                                np.inf)
    carry_forward = {
        "roc_auc": 0.5, "pr_auc": metrics["base_rate"],
        "recall_at_budget": 0.0, "frac_flagged_early": 0.0,
        "median_lead_time_h": 0.0,
        "median_hours_before_tca": cf_lead["median_hours_before_tca_carry_forward"],
        "rule": "flags an event only once its latest log10 risk >= threshold",
    }

    result.update({
        "booster": booster, "calibration": {"method": "platt", "a": a, "b": b},
        "alert_threshold": tau, "best_iteration": best, "metrics": metrics,
        "baselines": {"logistic_latest_risk": logistic,
                      "carry_forward": carry_forward},
    })
    return result


def ship_gate(run: dict, seed_runs: list[dict]) -> dict:
    """Evaluate the fixed SHIP GATE -> {"checks": [...], "passed": bool}."""
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    check("validation_has_both_classes", run["val_has_both_classes"],
          "calibration and alert threshold need both classes in the carve")
    m = run["metrics"] or {}
    lg = (run["baselines"] or {}).get("logistic_latest_risk", {})
    pr, lpr = m.get("pr_auc", float("nan")), lg.get("pr_auc", float("nan"))
    check("pr_auc_beats_logistic",
          np.isfinite(pr) and np.isfinite(lpr) and pr >= lpr + GATE_PR_AUC_MARGIN,
          f"PR-AUC >= logistic + {GATE_PR_AUC_MARGIN}")
    lead = m.get("median_lead_time_h", float("nan"))
    check("median_lead_time_positive", np.isfinite(lead) and lead > 0.0,
          "median lead time over early-flagged escalations > 0 h")
    fe, lfe = m.get("frac_flagged_early", float("nan")), lg.get("frac_flagged_early", float("nan"))
    check("early_recall_not_below_logistic",
          np.isfinite(fe) and np.isfinite(lfe) and fe >= lfe,
          "flags at least as many escalations early as the logistic baseline")
    seeds_ok = all(r["metrics"] is not None and r["baselines"] is not None
                   and np.isfinite(r["metrics"]["pr_auc"])
                   and r["metrics"]["pr_auc"] > r["baselines"]["logistic_latest_risk"]["pr_auc"]
                   for r in seed_runs)
    check("seed_check_beats_logistic_every_seed", seeds_ok,
          f"PR-AUC > logistic on every seed in {list(SEED_CHECK)}")
    return {"checks": checks, "passed": all(c["passed"] for c in checks)}


def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_clean(v) for v in d]
    return _nan_to_none(d)


def _seed_summary(seed, run) -> dict:
    m, b = run["metrics"], run["baselines"]
    if m is None:
        return {"seed": seed, "valid": False}
    lg = b["logistic_latest_risk"]
    return {"seed": seed, "valid": True, "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"], "logistic_pr_auc": lg["pr_auc"],
            "recall_at_budget": m["recall_at_budget"],
            "frac_flagged_early": m["frac_flagged_early"],
            "median_lead_time_h": m["median_lead_time_h"]}


def _fmt(v, spec=".3f"):
    return "n/a" if v is None or not np.isfinite(v) else format(v, spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train VIGIL-E (early-warning escalation classifier) "
                    "on the ESA Kelvins dataset.")
    parser.add_argument("--data", default=str(config.KELVINS_DIR / "train_data.csv"),
                        help="path to the Kelvins train_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default=None,
                        help="optional path: write the full evaluation json "
                             "here whether or not the model ships")
    args = parser.parse_args(argv)

    csv_path = Path(args.data)
    if not csv_path.exists():
        print(MISSING_CSV_MESSAGE.format(path=csv_path))
        return 0

    import pandas as pd

    import hashlib
    data_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"error: {csv_path} lacks required columns: {missing}")
        return 1
    df = df.dropna(subset=REQUIRED_COLUMNS)
    threshold = escalation.threshold_log10_pc()
    print(f"loading {csv_path} — {len(df)} CDM rows, "
          f"{df['event_id'].nunique()} events; threshold log10 Pc {threshold}")

    data = build_decision_points(df, threshold)
    n_events, n_points = len(data["event_label"]), len(data["y"])
    n_pos = int(data["event_label"].sum())
    print(f"decision points: {n_points} across {n_events} events "
          f"({n_pos} escalate to >= {threshold})")
    if n_events < 20 or n_pos < 5:
        print("error: not enough events/escalations to train and evaluate")
        return 1

    run = run_protocol(data, args.seed)
    seed_runs = [run_protocol(data, s) for s in SEED_CHECK]
    gate = ship_gate(run, seed_runs)

    m = run["metrics"] or {}
    lg = (run["baselines"] or {}).get("logistic_latest_risk", {})
    print()
    print(f"held-out (seed {args.seed}):         VIGIL-E   logistic(latest risk)")
    for key in ("roc_auc", "pr_auc", "brier", "recall_at_budget",
                "precision_at_budget", "frac_flagged_early"):
        print(f"  {key:<26} {_fmt(m.get(key)):>8}   {_fmt(lg.get(key)):>8}")
    print(f"  {'median_lead_time_h':<26} {_fmt(m.get('median_lead_time_h'), '.1f'):>8}"
          f"   {_fmt(lg.get('median_lead_time_h'), '.1f'):>8}")
    print(f"  carry-forward first flags escalations a median "
          f"{_fmt(m.get('median_hours_before_tca_carry_forward'), '.1f')} h before TCA")
    print("seed check:")
    for s, r in zip(SEED_CHECK, seed_runs):
        sm = _seed_summary(s, r)
        print(f"  seed {s}: " + (f"PR-AUC {_fmt(sm['pr_auc'])} vs logistic "
                                 f"{_fmt(sm['logistic_pr_auc'])}, median lead "
                                 f"{_fmt(sm['median_lead_time_h'], '.1f')} h"
                                 if sm["valid"] else "invalid (a class missing)"))
    print("ship gate:")
    for c in gate["checks"]:
        print(f"  [{'pass' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")

    meta = _clean({
        "version": escalation.MODEL_VERSION,
        "target": escalation.TARGET,
        "trained_at": datetime.now(timezone.utc).strftime(_ISO_FMT),
        "dataset": "ESA Kelvins collision avoidance challenge format",
        "data_file": csv_path.name,
        "data_sha256": data_sha256,  # provenance: which CSV produced this
        "threshold_log10_pc": threshold,
        "features": ESCALATION_FEATURE_NAMES,
        "alert_budget": ALERT_BUDGET,
        "alert_threshold": run["alert_threshold"],
        "calibration": run["calibration"],
        "best_iteration": run["best_iteration"],
        "seed": args.seed,
        "params": {k: v for k, v in XGB_PARAMS.items()},
        "data": {"n_events": n_events, "n_decision_points": n_points,
                 "n_escalating_events": n_pos, **run["split"]},
        "metrics": {k: m.get(k) for k in escalation.METRIC_KEYS},
        "baselines": run["baselines"],
        "gate": gate,
        "seed_check": [_seed_summary(s, r) for s, r in zip(SEED_CHECK, seed_runs)],
    })
    if args.report:
        Path(args.report).write_text(json.dumps(meta, indent=2))
        print(f"report -> {args.report}")

    print()
    if not gate["passed"]:
        for stale in (escalation.artifact_path(), escalation.meta_path()):
            Path(stale).unlink(missing_ok=True)
        print("NOT SHIPPED: VIGIL-E fails the ship gate. Any previous artifact "
              "was removed; every escalation field in the API stays null.")
        return 0

    Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    run["booster"].save_model(str(escalation.artifact_path()))
    escalation.meta_path().write_text(json.dumps(meta, indent=2))
    print(f"saved model -> {escalation.artifact_path()}")
    print(f"saved meta  -> {escalation.meta_path()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
