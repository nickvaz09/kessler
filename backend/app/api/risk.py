"""GET /risk/{conjunction_id} - latest prediction plus model input features.
GET /risk/{conjunction_id}/explain - glass-box per-feature attribution.
GET /risk/{conjunction_id}/escalation - VIGIL-E early-warning glass box.

`models.features` / `models.risk_model` are imported lazily inside the
handlers so a missing module can never break API import order;
feature/artifact failures degrade to nulls or mode 'unavailable', never 500s.
"""
from fastapi import APIRouter, Depends, HTTPException, Path

from backend.app import db
from backend.app.api import conjunctions as conjunctions_api

router = APIRouter()

# Failure-shape fallback only (module unimportable): the success path
# returns the COMPLETE feature dict, whatever width FEATURE_NAMES has - 
# a hardcoded key filter here once silently truncated vigil-0.3's new
# features and broke attribution.
_FEATURE_KEYS = (
    "miss_distance_m",
    "relative_velocity_ms",
    "pc_latest",
    "time_to_tca_h",
    "cdm_count",
    "md_delta",
    "pc_log_delta",
    "md_slope",
    "pc_log_slope",
    "pc_log_max",
    "msg_gap_last_h",
)


def _scored_cdms(conn, conjunction_id: int) -> list[dict]:
    """The same CDM sequence predict_for_conjunction scores (operator CDMs
    win when present). Falls back to the same rule inline if the model
    module is unimportable, so this endpoint never 500s on it."""
    try:
        from backend.app.models import risk_model as risk_model_mod
        return risk_model_mod.scored_cdms(conn, conjunction_id)
    except ImportError:
        rows = [dict(r) for r in conn.execute(
            "SELECT message_time, miss_distance_m, relative_velocity_ms,"
            " probability_of_collision, source FROM cdm_messages"
            " WHERE conjunction_id = ? ORDER BY message_time ASC, id ASC",
            (conjunction_id,))]
        chosen = [r for r in rows if r["source"] == "spacetrack"] or rows
        for r in chosen:
            r.pop("source", None)
        return chosen


def _build_features(cdms: list[dict], tca_iso: str) -> dict:
    try:
        from backend.app.models import features as features_mod

        return dict(features_mod.build_features(cdms, tca_iso))
    except Exception:
        return {key: None for key in _FEATURE_KEYS}


@router.get("/risk/{conjunction_id}")
def get_risk(
    conjunction_id: int = Path(ge=0, le=conjunctions_api.INT64_MAX),
    conn=Depends(db.get_db),
):
    conj = conn.execute(
        "SELECT time_of_closest_approach FROM conjunctions WHERE id = ?",
        (conjunction_id,),
    ).fetchone()
    if conj is None:
        raise HTTPException(
            status_code=404, detail=f"Conjunction {conjunction_id} not found"
        )
    tca_iso = conj["time_of_closest_approach"]

    cdms = _scored_cdms(conn, conjunction_id)
    pc_latest = cdms[-1]["probability_of_collision"] if cdms else None

    prediction = conn.execute(
        "SELECT predicted_final_risk, prediction_time, model_version,"
        "       escalation_prob, escalation_model_version"
        "  FROM risk_predictions WHERE conjunction_id = ?"
        " ORDER BY prediction_time DESC, id DESC LIMIT 1",
        (conjunction_id,),
    ).fetchone()
    predicted_final_risk = prediction["predicted_final_risk"] if prediction else None

    effective = conjunctions_api.effective_pc(
        {"predicted_final_risk": predicted_final_risk, "pc_latest": pc_latest}
    )
    return {
        "conjunction_id": conjunction_id,
        "risk_level": conjunctions_api.risk_level_for(effective),
        "predicted_final_risk": predicted_final_risk,
        "pc_latest": pc_latest,
        "model_version": prediction["model_version"] if prediction else None,
        "prediction_time": prediction["prediction_time"] if prediction else None,
        # VIGIL-E early warning, from the same stored prediction row
        "escalation_prob": prediction["escalation_prob"] if prediction else None,
        "escalation_model_version": (prediction["escalation_model_version"]
                                     if prediction else None),
        "features": _build_features(cdms, tca_iso),
    }


@router.get("/risk/{conjunction_id}/escalation")
def explain_escalation(
    conjunction_id: int = Path(ge=0, le=conjunctions_api.INT64_MAX),
    conn=Depends(db.get_db),
):
    """VIGIL-E glass box: the escalation probability for the CURRENT CDM
    sequence, with SHAP contributions (log-odds units, before calibration).

    `mode`:
    - unavailable: no gated VIGIL-E artifact (escalation_prob null);
    - not_applicable: latest log10 Pc already >= threshold_log10_pc - the
      event is high-risk now, there is nothing left to warn about early;
    - early_warning: probability + contributions.
    Computed live from the loaded artifact (the stored per-refresh value is
    on /risk/{id}); never a 500 on artifact trouble.
    """
    conj = conn.execute(
        "SELECT time_of_closest_approach FROM conjunctions WHERE id = ?",
        (conjunction_id,),
    ).fetchone()
    if conj is None:
        raise HTTPException(
            status_code=404, detail=f"Conjunction {conjunction_id} not found"
        )
    from backend.app.models import escalation as escalation_mod

    out = {"conjunction_id": conjunction_id,
           "threshold_log10_pc": escalation_mod.threshold_log10_pc(),
           "model_version": None, "escalation_prob": None}
    model = escalation_mod.EscalationModel.load()
    if model is None:
        out["mode"] = "unavailable"
        out["note"] = ("VIGIL-E is not trained yet — run "
                       "python -m backend.app.models.train_escalation")
        return out
    out["model_version"] = model.version
    cdms = _scored_cdms(conn, conjunction_id)
    try:
        feats = escalation_mod.escalation_features(
            cdms, conj["time_of_closest_approach"])
        applicable = bool(cdms) and escalation_mod.in_domain(feats)
    except Exception:
        out["mode"] = "unavailable"
        out["note"] = "features could not be built for this event"
        return out
    if not applicable:
        out["mode"] = "not_applicable"
        out["note"] = ("latest observed risk is already at/above the "
                       "high-risk threshold (or no CDMs yet) — early warning "
                       "applies only below it")
        return out
    explanation = model.explain(feats)
    out["mode"] = "early_warning"
    out["escalation_prob"] = model.predict_proba(feats)
    out["alert_threshold"] = model.alert_threshold
    out["base_value"] = explanation["base_value"] if explanation else None
    out["contributions"] = explanation["contributions"] if explanation else []
    out["note"] = ("SHAP contributions to the escalation log-odds; positive "
                   "values push toward escalation")
    return out


@router.get("/risk/{conjunction_id}/explain")
def explain_risk(
    conjunction_id: int = Path(ge=0, le=conjunctions_api.INT64_MAX),
    conn=Depends(db.get_db),
):
    """Which features push the model's number, and in which direction.

    `mode` mirrors the hybrid target (models/train.py):
    - carry_forward: latest log10 Pc >= threshold_log10_pc - the model
      defers to the observed Pc, so no contributions exist;
    - learned_residual: SHAP rows, in log10-Pc residual units;
    - unavailable: heuristic/missing artifact, or attribution failed.
    Artifact absence must degrade to 'unavailable', never a 500.
    """
    conj = conn.execute(
        "SELECT time_of_closest_approach FROM conjunctions WHERE id = ?",
        (conjunction_id,),
    ).fetchone()
    if conj is None:
        raise HTTPException(
            status_code=404, detail=f"Conjunction {conjunction_id} not found"
        )

    from backend.app.models import risk_model as risk_model_mod

    model = risk_model_mod.RiskModel.load()
    out = {
        "model_version": model.version,
        "threshold_log10_pc": risk_model_mod.high_risk_threshold_log10_pc(),
    }
    if model.booster is None:
        out["mode"] = "unavailable"
        out["note"] = (
            "no trained artifact loaded — the heuristic model is "
            "carry-forward extrapolation with no per-feature attribution"
        )
        return out

    cdms = _scored_cdms(conn, conjunction_id)
    feats = _build_features(cdms, conj["time_of_closest_approach"])

    if model.defers_to_carry_forward(feats.get("pc_latest") or 0.0):
        out["mode"] = "carry_forward"
        out["note"] = (
            "This event is already above the high-risk line, so VIGIL "
            "defers to the latest observed probability rather than "
            "adjusting it."
        )
        return out

    explanation = model.explain(feats)
    if explanation is None:
        out["mode"] = "unavailable"
        out["note"] = "attribution failed on the loaded artifact"
        return out

    out["mode"] = "learned_residual"
    out["base_value"] = explanation["base_value"]
    out["contributions"] = explanation["contributions"]
    out["note"] = (
        "SHAP contributions to the learned log10-Pc residual on top of "
        "carry-forward; positive contributions push predicted risk up"
    )
    return out
