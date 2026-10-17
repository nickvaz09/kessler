"""GET /model/card - provenance, held-out metrics and live usage of the
risk model.

`models.risk_model` / `models.features` are imported lazily inside the
handler (the api/risk.py idiom) so a missing or test-stubbed module can
never break API import order.

Card fields come from three guarded sources, none of which may ever 500:
- data/models/risk_model.meta.json (version/target/trained_at/metrics);
  absent or corrupt -> metrics null, version from the loaded model's
  fallback (the heuristic version when no artifact serves).
- The loaded xgboost booster's gain importances, mapped through
  features.FEATURE_NAMES ('f<i>' booster keys -> feature names); any
  fault -> [].
- risk_predictions counts on the live DB (0s on an empty database).

`early_warning` (VIGIL-E, docs/v2/vigil-e.md) is null unless a ship-gated
artifact loads (data/models/vigil_e.json + vigil_e.meta.json); then it
carries the training meta's metrics/baselines/gate/seed check verbatim,
booster gain importances and live escalation-scoring counts.
"""
import json
import re

from fastapi import APIRouter, Depends

from backend.app import config, db

router = APIRouter()

# Contract shape: exactly these metric keys, always all present (null-filled
# when the meta json carries a subset).
METRIC_KEYS = ("mse_test_all", "mse_test_high_risk", "baseline_mse_all",
               "baseline_mse_high_risk", "n_events", "n_train", "n_test",
               "n_test_high_risk")
IMPORTANCE_TOP_N = 12

# Boosters trained from bare numpy arrays (models/train.py) name features
# positionally: 'f0'..'fN' in FEATURE_NAMES order (11 wide as of vigil-0.3).
_POSITIONAL_KEY = re.compile(r"^f(\d+)$")


def _read_meta() -> dict | None:
    """The meta json as a dict; None when absent/corrupt (never raises)."""
    try:
        meta = json.loads(config.RISK_MODEL_META_FILE.read_text())
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def _feature_importance(booster, feature_names: list[str]) -> list[dict]:
    """Top-N gain importances as [{name, gain}], desc; [] on any fault."""
    if booster is None:
        return []
    try:
        scores = booster.get_score(importance_type="gain")
        items = list(scores.items())
    except Exception:
        return []   # artifact quirks must degrade, never 500
    rows = []
    for key, gain in items:
        name = None
        if key in feature_names:
            name = key
        else:
            match = _POSITIONAL_KEY.match(str(key))
            if match and int(match.group(1)) < len(feature_names):
                name = feature_names[int(match.group(1))]
        if name is None:
            continue   # unknown booster key - never invent a feature name
        try:
            rows.append({"name": name, "gain": float(gain)})
        except (TypeError, ValueError):
            continue
    rows.sort(key=lambda r: r["gain"], reverse=True)
    return rows[:IMPORTANCE_TOP_N]


def _early_warning_section(conn) -> dict | None:
    """VIGIL-E card section; None whenever no gated artifact is loadable.

    Every number is read from the training meta json (what the ship gate
    judged) or counted from the live DB - nothing is defaulted or invented.
    """
    try:
        from backend.app.models import escalation
        model = escalation.EscalationModel.load()
    except Exception:
        return None
    if model is None:
        return None
    meta = model.meta
    raw = meta.get("metrics") if isinstance(meta.get("metrics"), dict) else {}
    try:
        live = conn.execute(
            "SELECT COUNT(escalation_prob),"
            " COUNT(DISTINCT CASE WHEN escalation_prob IS NOT NULL"
            "                     THEN conjunction_id END),"
            " MAX(prediction_time) FROM risk_predictions"
            " WHERE escalation_model_version = ?", (model.version,)).fetchone()
        live = {"predictions_total": live[0], "conjunctions_scored": live[1],
                "last_prediction_at": live[2]}
    except Exception:
        live = {"predictions_total": 0, "conjunctions_scored": 0,
                "last_prediction_at": None}
    return {
        "version": model.version,
        "target": meta.get("target"),
        "trained_at": meta.get("trained_at"),
        "dataset": meta.get("dataset"),
        "threshold_log10_pc": meta.get("threshold_log10_pc"),
        "alert_budget": meta.get("alert_budget"),
        "alert_threshold": meta.get("alert_threshold"),
        "metrics": {k: raw.get(k) for k in escalation.METRIC_KEYS},
        "baselines": meta.get("baselines"),
        "gate": meta.get("gate"),
        "seed_check": meta.get("seed_check"),
        "data": meta.get("data"),
        "feature_importance": model.feature_importance(IMPORTANCE_TOP_N),
        "live": live,
    }


BENCHMARK_CARD = "backend/app/models/lab/benchmark.json"


@router.get("/model/benchmark")
def get_model_benchmark():
    """VIGIL's results on the official ESA benchmark, as generated from the
    committed VIGIL Lab reports by models.lab.benchmark_card (research
    builds, not the model serving live). 404 when the card is absent."""
    from pathlib import Path

    from fastapi import HTTPException

    path = Path(__file__).resolve().parents[3] / BENCHMARK_CARD
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="no benchmark card on this server")


@router.get("/model/card")
def get_model_card(conn=Depends(db.get_db)):
    from backend.app.models import risk_model
    from backend.app.models.features import FEATURE_NAMES

    # load() already absorbs missing/corrupt artifacts (heuristic fallback),
    # so `heuristic` reports what the API is actually serving right now.
    model = risk_model.RiskModel.load()
    meta = _read_meta()

    raw_metrics = meta.get("metrics") if meta else None
    metrics = ({k: raw_metrics.get(k) for k in METRIC_KEYS}
               if isinstance(raw_metrics, dict) else None)

    if meta:
        version = meta.get("version") or model.version
        target = meta.get("target") or model.target
        trained_at = meta.get("trained_at")
    else:
        version = model.version
        target = model.target if model.booster is not None else None
        trained_at = None

    live_row = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT conjunction_id), MAX(prediction_time)"
        " FROM risk_predictions"
    ).fetchone()

    return {
        "early_warning": _early_warning_section(conn),
        "version": version,
        "target": target,
        "trained_at": trained_at,
        "heuristic": model.booster is None,
        "metrics": metrics,
        "feature_importance": _feature_importance(model.booster, FEATURE_NAMES),
        "live": {
            "predictions_total": live_row[0],
            "conjunctions_scored": live_row[1],
            "last_prediction_at": live_row[2],
        },
    }
