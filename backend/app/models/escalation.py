"""VIGIL-E - early warning: will this event escalate to high risk?

For a conjunction whose latest observed risk is still BELOW the high-risk
threshold (log10 Pc < risk_model.high_risk_threshold_log10_pc(), -6 at the
shipped config), VIGIL-E estimates the probability that the event's FINAL
CDM lands at or above that threshold. It is the operator's "should I start
worrying now?" decision, answered before the carry-forward rule - which
only reacts once the latest risk itself crosses the threshold - can.

One module, shared by training (models/train_escalation.py) and serving
(pipeline._predict_active -> risk_model.predict_for_conjunction):

- ``escalation_features`` is the feature function: features.build_features
  (the VIGIL feature contract, unchanged) plus the escalation-specific
  extras in EXTRA_FEATURE_NAMES, computed here and nowhere else.
- ``EscalationModel.load()`` returns None when no artifact exists (or it is
  unreadable). There is no heuristic stand-in: without a trained, gated
  artifact every escalation field is null. Never a placeholder number.
- ``predict_proba`` applies the Platt calibration fitted on the training
  validation carve (the booster is trained class-weighted, so its raw
  output is a ranking score, not a probability), and returns None when the
  event is outside the model's domain (already at/above the threshold).
- ``explain`` is xgboost pred_contribs (SHAP) on the booster's raw margin.
"""
import json
import math
from pathlib import Path

from backend.app import config
from backend.app.models.features import FEATURE_NAMES, build_features

MODEL_VERSION = "vigil-e-0.1"
TARGET = "escalation_to_high_risk"

# Stand-in log10 Pc when no positive Pc has been observed - the same floor
# the Kelvins data uses for "effectively zero" (risk = -30) and the same
# constant as risk_model._LOG10_PC_FLOOR.
LOG10_PC_FLOOR = -30.0

# Escalation-specific features, appended to FEATURE_NAMES (order is the
# booster's column order - never reorder without retraining).
EXTRA_FEATURE_NAMES = [
    "log10_pc_latest",        # latest observed risk, log10 (floor -30)
    "pc_log_projected_tca",   # latest log10 Pc + trend slope x days to TCA
    "pc_log_drop_from_max",   # peak log10 Pc seen minus latest (>= 0)
]
ESCALATION_FEATURE_NAMES = FEATURE_NAMES + EXTRA_FEATURE_NAMES

# Keys of meta["metrics"] (the /model/card early_warning.metrics contract;
# always all present, null-filled when absent).
METRIC_KEYS = (
    "roc_auc", "pr_auc", "brier", "base_rate",
    "recall_at_budget", "precision_at_budget", "alert_rate",
    "median_lead_time_h", "median_lead_time_h_all", "frac_flagged_early",
    "median_hours_before_tca_model", "median_hours_before_tca_carry_forward",
    "n_events_test", "n_escalating_test", "n_lead_eligible_test",
    "n_decision_points_test",
)


def artifact_path() -> Path:
    """Booster file. Derived from config.MODEL_DIR at call time so tests that
    point MODEL_DIR at a temp dir are hermetic."""
    return Path(config.MODEL_DIR) / "vigil_e.json"


def meta_path() -> Path:
    return Path(config.MODEL_DIR) / "vigil_e.meta.json"


def threshold_log10_pc() -> float:
    """The high-risk threshold (single definition lives in risk_model)."""
    from backend.app.models import risk_model
    return risk_model.high_risk_threshold_log10_pc()


def escalation_features(cdms: list[dict], tca_iso: str,
                        as_of_iso: str | None = None) -> dict:
    """ESCALATION_FEATURE_NAMES dict for one CDM sequence.

    The base eleven come from features.build_features unchanged; the extras:
    - log10_pc_latest: log10 of the latest non-null Pc, LOG10_PC_FLOOR when
      it is 0/absent.
    - pc_log_projected_tca: log10_pc_latest + pc_log_slope [decades/day] x
      time_to_tca [days] (time clamped at >= 0), clipped to [floor, 0] - a
      naive linear extrapolation of the risk trend to closest approach.
    - pc_log_drop_from_max: pc_log_max - log10_pc_latest when a positive Pc
      was seen, else 0.0 (how far the risk has already fallen from its peak).
    """
    feats = build_features(cdms, tca_iso, as_of_iso)
    pc_latest = feats["pc_latest"]
    log_latest = math.log10(pc_latest) if pc_latest > 0.0 else LOG10_PC_FLOOR
    feats["log10_pc_latest"] = log_latest
    days = max(feats["time_to_tca_h"], 0.0) / 24.0
    projected = log_latest + feats["pc_log_slope"] * days
    feats["pc_log_projected_tca"] = min(0.0, max(LOG10_PC_FLOOR, projected))
    has_positive = any(c.get("probability_of_collision") is not None
                       and float(c["probability_of_collision"]) > 0.0
                       for c in cdms)
    feats["pc_log_drop_from_max"] = (
        max(0.0, feats["pc_log_max"] - log_latest) if has_positive else 0.0)
    return feats


def in_domain(feats: dict) -> bool:
    """True when VIGIL-E applies: latest risk still below the threshold."""
    return feats["log10_pc_latest"] < threshold_log10_pc()


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def calibrate(margin, a: float, b: float):
    """Platt scaling of the booster margin: sigmoid(a * margin + b).
    Works on floats and numpy arrays (training uses the array form)."""
    try:
        import numpy as np
        if isinstance(margin, np.ndarray):
            return 1.0 / (1.0 + np.exp(-(a * margin + b)))
    except Exception:
        pass
    return _sigmoid(a * float(margin) + b)


class EscalationModel:
    """A loaded, ship-gated VIGIL-E artifact."""

    def __init__(self, booster, meta: dict):
        self.booster = booster
        self.meta = meta
        self.version = meta.get("version") or MODEL_VERSION
        cal = meta.get("calibration") or {}
        self.cal_a = float(cal.get("a", 1.0))
        self.cal_b = float(cal.get("b", 0.0))
        self.alert_threshold = meta.get("alert_threshold")

    @classmethod
    def load(cls) -> "EscalationModel | None":
        """The artifact at artifact_path() + meta, or None when either is
        missing or unreadable (callers then serve null - never a guess).
        Cached per (path, mtime) so API requests don't re-read the booster."""
        path, mpath = artifact_path(), meta_path()
        if not (path.exists() and mpath.exists()):
            return None
        try:
            key = (str(path), path.stat().st_mtime_ns, mpath.stat().st_mtime_ns)
        except OSError:
            return None
        cached = _LOAD_CACHE.get("model")
        if cached is not None and cached[0] == key:
            return cached[1]
        model = cls._load_uncached(path, mpath)
        _LOAD_CACHE["model"] = (key, model)
        return model

    @classmethod
    def _load_uncached(cls, path, mpath) -> "EscalationModel | None":
        try:
            meta = json.loads(mpath.read_text())
            if not isinstance(meta, dict):
                return None
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(str(path))
            return cls(booster, meta)
        except Exception:
            return None

    def _vector(self, feats: dict):
        import numpy as np
        return np.asarray(
            [[float(feats[name]) for name in ESCALATION_FEATURE_NAMES]],
            dtype=float)

    def margin(self, feats: dict) -> float:
        import xgboost as xgb
        return float(self.booster.predict(
            xgb.DMatrix(self._vector(feats)), output_margin=True)[0])

    def predict_proba(self, feats: dict) -> float | None:
        """Calibrated escalation probability, or None outside the domain."""
        if not in_domain(feats):
            return None
        p = calibrate(self.margin(feats), self.cal_a, self.cal_b)
        return min(1.0, max(0.0, float(p)))

    def predict(self, cdms: list[dict], tca_iso: str) -> float | None:
        """Escalation probability for a CDM sequence (None: not applicable)."""
        if not cdms:
            return None
        return self.predict_proba(escalation_features(cdms, tca_iso))

    def explain(self, feats: dict) -> dict | None:
        """SHAP attribution of the raw booster margin (log-odds, before
        calibration) -> {base_value, contributions:[{name,value,
        contribution}]} sorted by |contribution|; None on any failure."""
        try:
            import xgboost as xgb
            vec = self._vector(feats)
            contribs = self.booster.predict(
                xgb.DMatrix(vec), pred_contribs=True)[0]
            rows = [{"name": name, "value": float(vec[0, i]),
                     "contribution": float(contribs[i])}
                    for i, name in enumerate(ESCALATION_FEATURE_NAMES)]
            rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
            return {"base_value": float(contribs[-1]), "contributions": rows}
        except Exception:
            return None

    def feature_importance(self, top_n: int = 14) -> list[dict]:
        """Gain importances mapped to ESCALATION_FEATURE_NAMES; [] on fault."""
        try:
            scores = self.booster.get_score(importance_type="gain")
        except Exception:
            return []
        rows = []
        for key, gain in scores.items():
            name = key if key in ESCALATION_FEATURE_NAMES else None
            if name is None and str(key).startswith("f") and str(key)[1:].isdigit():
                i = int(str(key)[1:])
                if i < len(ESCALATION_FEATURE_NAMES):
                    name = ESCALATION_FEATURE_NAMES[i]
            if name is not None:
                rows.append({"name": name, "gain": float(gain)})
        rows.sort(key=lambda r: r["gain"], reverse=True)
        return rows[:top_n]


_LOAD_CACHE: dict = {}


def read_meta() -> dict | None:
    """The meta json (None when absent/corrupt). Never raises."""
    try:
        meta = json.loads(meta_path().read_text())
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def score_cdms(model: "EscalationModel | None", cdms: list[dict],
               tca_iso: str) -> tuple[float | None, str | None]:
    """(escalation_prob, escalation_model_version) for persistence.

    No model -> (None, None). Model present but the event is outside its
    domain (latest risk already >= threshold) -> (None, version): the model
    ran and declined - the event is already high-risk. Any failure ->
    (None, None); an escalation fault must never block the risk prediction.
    """
    if model is None:
        return None, None
    try:
        return model.predict(cdms, tca_iso), model.version
    except Exception:
        return None, None
