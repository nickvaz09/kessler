"""Risk model: predict the final collision probability of a conjunction.

Two backends behind one interface:
- Trained XGBoost regressor (artifact at config.RISK_MODEL_FILE, produced by
  ``backend.app.models.train``). The model is trained on log10(Pc), so its
  raw output is exponentiated back to a probability.
- Heuristic fallback when no artifact exists: latest non-null Pc extrapolated
  by the observed pc_log_delta, capped at 10x in either direction.

Predictions are always clamped to [0, 1]. ``risk_level`` is the single
backend-side definition of the severity buckets (thresholds from config).
``RiskModel.explain`` opens the glass box: per-feature SHAP contributions of
the loaded booster (None for the heuristic - it has nothing to attribute).
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from backend.app import config
from backend.app.models.features import FEATURE_NAMES, build_features

_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

_LOG10_PC_FLOOR = -30.0  # stand-in log10 Pc when no positive Pc is observed
# Residual clamp, in decades of Pc. one constant shared by training
# evaluation (models/train.py) and inference - a mismatch would make the
# shipped artifact behave differently from the numbers that gated it.
# vigil-0.4 widened this from 3 to 4 (selected on validation carves).
RESID_CLIP_DECADES = 4.0


def high_risk_threshold_log10_pc() -> float:
    """log10-Pc threshold of the hybrid model: at/above it predictions defer
    to carry-forward; the learned residual applies only below (train.py)."""
    return math.log10(config.PC_CRITICAL) - 2.0  # -6.0 at the shipped config


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


_LOAD_CACHE: dict = {}


class RiskModel:
    """Loaded risk model - XGBoost booster if an artifact exists, else heuristic."""

    def __init__(self, booster=None, version: str | None = None,
                 target: str = "absolute"):
        self.booster = booster
        self.version = version or config.HEURISTIC_MODEL_VERSION
        self.target = target  # 'absolute': yhat = log10 Pc; 'residual': yhat += last known

    @classmethod
    def load(cls) -> "RiskModel":
        """Load the XGBoost artifact at config.RISK_MODEL_FILE if present
        (version and target mode from the meta json), else the heuristic."""
        path = Path(config.RISK_MODEL_FILE)
        meta_file = Path(config.RISK_MODEL_META_FILE)
        try:  # cache per (path, mtime): the model card and explain endpoints
            key = (str(path), path.stat().st_mtime_ns,  # load per request
                   meta_file.stat().st_mtime_ns if meta_file.exists() else None)
        except OSError:
            key = None
        if key is not None and _LOAD_CACHE.get("key") == key:
            return _LOAD_CACHE["model"]
        model = cls._load_uncached(path)
        if key is not None:
            _LOAD_CACHE.update(key=key, model=model)
        return model

    @classmethod
    def _load_uncached(cls, path: Path) -> "RiskModel":
        if path.exists():
            try:
                import xgboost as xgb
                booster = xgb.Booster()
                booster.load_model(str(path))
                version, target = "xgb-0.1", "absolute"
                meta_path = Path(config.RISK_MODEL_META_FILE)
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    version = meta.get("version", version)
                    target = meta.get("target", target)
                return cls(booster=booster, version=version, target=target)
            except Exception:
                pass  # corrupt/unreadable artifact -> heuristic fallback
        return cls()

    def predict(self, cdms: list[dict], tca_iso: str) -> tuple[float, str]:
        """Predict the final Pc for a CDM sequence -> (risk in [0,1], version).

        Heuristic: latest non-null pc (0.0 when the sequence has none)
        extrapolated by pc_log_delta capped at 10x in either direction.
        XGBoost: model output is log10(Pc) -> 10**yhat, clamped to [0, 1].
        """
        feats = build_features(cdms, tca_iso)
        if self.booster is not None:
            import numpy as np
            import xgboost as xgb
            vec = np.asarray([[feats[name] for name in FEATURE_NAMES]], dtype=float)
            yhat = float(self.booster.predict(xgb.DMatrix(vec))[0])
            if self.target in ("residual", "residual_below_threshold"):
                # Model predicts the log10 correction to last-known Pc
                # (carry-forward is its zero point), clamped to
                # RESID_CLIP_DECADES - the same clamp used at training
                # evaluation. The hybrid variant defers entirely to
                # carry-forward at/above the high-risk threshold.
                base = (math.log10(feats["pc_latest"])
                        if feats["pc_latest"] > 0.0 else _LOG10_PC_FLOOR)
                defer = self.defers_to_carry_forward(feats["pc_latest"])
                clip = RESID_CLIP_DECADES
                log10_pc = base if defer else base + min(clip, max(-clip, yhat))
            else:
                log10_pc = yhat
            return _clamp01(10.0 ** log10_pc), self.version

        pc_latest = feats["pc_latest"]
        if pc_latest <= 0.0:
            return 0.0, self.version
        factor = 10.0 ** min(1.0, max(-1.0, feats["pc_log_delta"]))
        return _clamp01(pc_latest * factor), self.version

    def defers_to_carry_forward(self, pc_latest: float | None) -> bool:
        """True when the hybrid target hands this prediction to carry-forward:
        latest observed log10 Pc at/above high_risk_threshold_log10_pc() -
        the exact deference condition predict() applies."""
        if self.booster is None or self.target != "residual_below_threshold":
            return False
        base = (math.log10(pc_latest) if pc_latest and pc_latest > 0.0
                else _LOG10_PC_FLOOR)
        return base >= high_risk_threshold_log10_pc()

    def explain(self, features: dict) -> dict | None:
        """SHAP attribution of the booster's raw output for one feature dict
        (xgboost pred_contribs on a single-row DMatrix, FEATURE_NAMES order).

        -> {base_value, contributions: [{name, value, contribution}]} sorted
        by |contribution| desc. Contribution units are the model's training
        target (log10-Pc residual for the shipped hybrid). Heuristic model or
        any failure -> None - callers degrade instead of raising.
        """
        if self.booster is None:
            return None
        try:
            import numpy as np
            import xgboost as xgb
            vec = np.asarray(
                [[float(features[name]) for name in FEATURE_NAMES]],
                dtype=float,
            )
            # pred_contribs row = FEATURE_NAMES-order SHAP values + bias last
            contribs = self.booster.predict(
                xgb.DMatrix(vec), pred_contribs=True
            )[0]
            rows = [
                {"name": name, "value": float(vec[0, i]),
                 "contribution": float(contribs[i])}
                for i, name in enumerate(FEATURE_NAMES)
            ]
            rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
            return {"base_value": float(contribs[-1]), "contributions": rows}
        except Exception:
            return None


def scored_cdms(conn, conjunction_id: int) -> list[dict]:
    """The CDM sequence the model scores, ascending by message_time.

    Operator (Space-Track) CDMs win when present: they are what the Kelvins
    training data is made of, and screening rows are always stamped "now",
    so mixing the two would let our assumed-sigma estimate mask the
    operator's Pc. Without operator data, every source is used.
    """
    rows = [dict(r) for r in conn.execute(
        "SELECT message_time, miss_distance_m, relative_velocity_ms,"
        " probability_of_collision, source"
        " FROM cdm_messages WHERE conjunction_id = ?"
        " ORDER BY message_time ASC, id ASC",
        (conjunction_id,),
    ).fetchall()]
    operator = [r for r in rows if r["source"] == "spacetrack"]
    chosen = operator or rows
    for r in chosen:
        del r["source"]
    return chosen


def predict_for_conjunction(conn, conjunction_id: int,
                            model: RiskModel | None = None,
                            escalation_model=None) -> dict:
    """Predict for one conjunction and persist a risk_predictions row.

    Reads the conjunction's CDMs (ascending by message_time) from ``conn``,
    runs the model, INSERTs into risk_predictions, and returns the contract
    dict. Raises ValueError for an unknown conjunction id.

    ``escalation_model``: a loaded models.escalation.EscalationModel (VIGIL-E)
    or None. Its output lands in the same row (escalation_prob,
    escalation_model_version); None -> both null. Scored on the same CDM
    sequence as the risk model; a VIGIL-E fault never blocks the row.
    """
    row = conn.execute(
        "SELECT time_of_closest_approach FROM conjunctions WHERE id = ?",
        (conjunction_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown conjunction id {conjunction_id}")
    tca_iso = row["time_of_closest_approach"]

    cdms = scored_cdms(conn, conjunction_id)

    if model is None:
        model = RiskModel.load()
    predicted_final_risk, model_version = model.predict(cdms, tca_iso)
    prediction_time = datetime.now(timezone.utc).strftime(_ISO_FMT)

    from backend.app.models import escalation as escalation_mod
    escalation_prob, escalation_version = escalation_mod.score_cdms(
        escalation_model, cdms, tca_iso)

    conn.execute(
        "INSERT INTO risk_predictions"
        " (conjunction_id, predicted_final_risk, prediction_time, model_version,"
        "  escalation_prob, escalation_model_version)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (conjunction_id, predicted_final_risk, prediction_time, model_version,
         escalation_prob, escalation_version),
    )
    conn.commit()
    return {
        "conjunction_id": conjunction_id,
        "predicted_final_risk": predicted_final_risk,
        "model_version": model_version,
        "prediction_time": prediction_time,
        "escalation_prob": escalation_prob,
        "escalation_model_version": escalation_version,
    }


def risk_level(effective_pc: float) -> str:
    """Severity bucket for an effective Pc: 'critical' | 'elevated' | 'nominal'."""
    pc = 0.0 if effective_pc is None else float(effective_pc)
    if pc >= config.PC_CRITICAL:
        return "critical"
    if pc >= config.PC_ELEVATED:
        return "elevated"
    return "nominal"
