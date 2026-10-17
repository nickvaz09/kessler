"""Export gated serving-track models in the formats the app already loads.

Never touches the production artifacts: files go to ``--export-dir``
(default data/models/lab/). Promoting them (copying over
data/models/risk_model.json / vigil_e.json) is a deliberate, separate step,
because the Model page's copy describes the shipped model.

VIGIL  -> risk_model.json + risk_model.meta.json, target "residual": the
         production RiskModel serves base + clip(yhat, ±4) on FEATURE_NAMES,
         exactly how the lab's ``serving_resid`` candidate was evaluated.
VIGIL-E -> vigil_e.json + vigil_e.meta.json: booster margin on
         ESCALATION_FEATURE_NAMES + Platt calibration + alert threshold, the
         format escalation.EscalationModel loads.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.app.models.lab import candidates as C
from backend.app.models.lab import metric
from backend.app.models.lab.protocol import Bundle

ISO = "%Y-%m-%dT%H:%M:%SZ"


def _now():
    return datetime.now(timezone.utc).strftime(ISO)


def _evaluated_booster(model):
    """The booster the lab actually scored. Early-stopped sklearn wrappers
    predict with trees [0, best_iteration]; the full booster carries up to
    early_stopping_rounds more, and the app's Booster.predict would use them
    all - so the saved file must be sliced like train_escalation.py does."""
    bst = model.get_booster()
    bi = getattr(model, "best_iteration", None)
    return bst[: bi + 1] if bi is not None else bst


# The lab evaluates one decision per event, made from CDMs received >= 2 days
# before TCA. The app scores at every refresh, including inside 48 h, where
# these metrics say nothing; promotion must respect this (see the doc).
DECISION_POINT = {"min_time_to_tca_h": 48.0,
                  "metrics_scope": "event-level, at the last CDM >= 2 days before TCA"}


def _gate_for_card(gate_result: dict) -> dict:
    """The Model card (frontend/js/vigil_e.js) reads checks[].passed; the lab
    writes checks[].ok. Carry both so a promoted model renders truthfully."""
    g = dict(gate_result)
    g["checks"] = [{**c, "passed": bool(c.get("ok"))} for c in gate_result.get("checks", [])]
    return g


def _dataset(test_summary, pooled: bool) -> str:
    if pooled:
        return "ESA Kelvins CAC (train + official test, pooled into CV; no sealed test)"
    return "ESA Kelvins CAC (train" + (" + official test" if test_summary else "") + ")"


def export_vigil(events, gate_result: dict, cv_summary: dict, test_summary, out_dir: Path,
                 version: str = "vigil-1.0", seed: int = 42, pooled: bool = False) -> Path:
    b = Bundle(events)
    cand = C.GBMResid("serving_resid", track="serving", resid_clip=C.SERVING_RESID_CLIP)
    tr = np.flatnonzero(b.trainable)
    cand.fit(C.Context(b.X["serving"][tr], b.y[tr], b.latest[tr], b.strata[tr], seed,
                       b.eligible[tr], {"colindex": b.colindex, "colnames": b.columns["serving"]}))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "risk_model.json"
    _evaluated_booster(cand.model).save_model(str(path))
    s = cv_summary["regression"]["serving_resid"]
    meta = {
        "version": version, "target": "residual", "trained_at": _now(),
        "dataset": _dataset(test_summary, pooled),
        "decision_point": DECISION_POINT,
        "lab": {"protocol": "official Kelvins (2-day cutoff, test filters, L = MSE_HR/F2)",
                "cv": s, "sealed_test": (test_summary or {}).get("regression", {}).get("serving_resid"),
                "gate": gate_result},
        "metrics": {"n_events": int(len(events))},
    }
    (out_dir / "risk_model.meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return path


def export_vigil_e(events, gate_result: dict, cv_summary: dict, test_summary, out_dir: Path,
                   budget: float = 0.05, version: str = "vigil-e-1.0", seed: int = 42,
                   pooled: bool = False) -> Path:
    from backend.app.models import escalation

    b = Bundle(events)
    cand = C.EClassifier("e_serving", track="eserving")
    tr = np.flatnonzero(b.trainable)
    cand.fit(C.Context(b.X["eserving"][tr], b.y[tr], b.latest[tr], b.strata[tr], seed,
                       b.eligible[tr], {"colindex": b.colindex, "colnames": b.columns["eserving"]}))
    m = b.cand & b.trainable
    # alert threshold from the classifier's HELD-OUT carve, not in-sample
    held = np.asarray(cand.val_prob, float)
    tau = float(np.quantile(held, 1.0 - budget)) if len(held) else None
    a, bb = ((float(cand.platt.coef_[0, 0]), float(cand.platt.intercept_[0]))
             if cand.platt is not None else (1.0, 0.0))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "vigil_e.json"
    _evaluated_booster(cand.model).save_model(str(path))
    e = cv_summary["early_warning"]["e_serving"]
    metrics = {k: None for k in escalation.METRIC_KEYS}
    metrics.update({"roc_auc": e["roc_auc"]["mean"], "pr_auc": e["pr_auc"]["mean"],
                    "recall_at_budget": e["recall_at_budget"]["mean"],
                    "brier": e.get("brier", {}).get("mean"),
                    "base_rate": float(b.esc[m].mean()) if m.any() else None,
                    "n_events_test": None, "n_escalating_test": None})
    meta = {
        "version": version, "target": escalation.TARGET, "trained_at": _now(),
        "dataset": _dataset(test_summary, pooled), "threshold_log10_pc": -6.0,
        "alert_budget": budget, "alert_threshold": tau,
        "calibration": {"method": "platt", "a": a, "b": bb},
        "decision_point": DECISION_POINT,
        "metrics": metrics, "gate": _gate_for_card(gate_result),
        "data": {"n_events_train": int(m.sum()), "n_escalating_train": int(b.esc[m].sum())},
        "lab": {"cv": e, "sealed_test": (test_summary or {}).get("early_warning", {}).get("e_serving")},
        "feature_names": list(escalation.ESCALATION_FEATURE_NAMES),
    }
    (out_dir / "vigil_e.meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return path
