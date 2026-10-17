"""Per-event feature tables.

Two tracks, never mixed:

BENCHMARK track - everything the Kelvins CDMs carry (103 columns incl. both
objects' covariance, tracking quality, space weather) plus sequence history
and encounter-plane physics. This is what the leaderboard comparison uses.

SERVING track - only what Kessler's live pipeline can compute from its own
CDMs (risk, miss distance, relative speed, message timing): exactly
``backend.app.models.features.FEATURE_NAMES`` built by the production
``build_features``. A model trained here can be exported to the app; the
app's claims must come from this track, never from the benchmark track.

All features use only ``Event.inputs`` (CDMs at or before the 2-day cutoff).
``tests/test_lab.py`` pins that by perturbing late CDMs and asserting the
tables do not change.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from backend.app.models.lab import physics
from backend.app.models.lab.data import RISK_FLOOR, Event

T_EVAL_DAYS = 0.5   # nominal "final CDM" horizon for extrapolated features
_EXCLUDE = {"event_id", "mission_id"}   # identifiers, not physics


def _num(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


MIN_SLOPE_SPAN_DAYS = 0.1   # CDMs minutes apart make per-day slopes explode


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.ptp(x) < MIN_SLOPE_SPAN_DAYS:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def history_features(ev: Event) -> dict:
    rows = ev.inputs
    tta = np.array([_num(r["time_to_tca"]) for r in rows])
    t = -tta                                   # days, increasing toward TCA
    risk = np.array([_num(r["risk"]) for r in rows])
    miss = np.array([_num(r.get("miss_distance")) for r in rows])
    last = rows[-1]
    f = {
        "h_n_inputs": float(len(rows)),
        "h_last_tta": float(tta[-1]),
        "h_span_days": float(tta[0] - tta[-1]),
        "h_gap_last": float(tta[-2] - tta[-1]) if len(rows) >= 2 else 0.0,
        "h_risk_last": float(risk[-1]),
        "h_risk_first": float(risk[0]),
        "h_risk_max": float(np.nanmax(risk)),
        "h_risk_min": float(np.nanmin(risk)),
        "h_risk_mean": float(np.nanmean(risk)),
        "h_risk_std": float(np.nanstd(risk)) if len(rows) >= 2 else 0.0,
        "h_risk_delta_last": float(risk[-1] - risk[-2]) if len(rows) >= 2 else 0.0,
        "h_risk_slope": _slope(t, risk),
        "h_risk_last_is_floor": float(risk[-1] <= RISK_FLOOR + 1e-9),
        "h_n_floor": float(np.sum(risk <= RISK_FLOOR + 1e-9)),
        "h_risk_max_minus_last": float(np.nanmax(risk) - risk[-1]),
        "h_miss_last": float(miss[-1]),
        "h_miss_log_ratio": (float(math.log(max(miss[-1], 1e-3) / max(miss[0], 1e-3)))
                             if np.isfinite(miss[[0, -1]]).all() else 0.0),
        "h_miss_slope": _slope(t, miss),
    }
    # CARA-style parabolic trend of risk, extrapolated to T_EVAL_DAYS
    ok = np.isfinite(risk) & (risk > RISK_FLOOR + 1e-9)
    if ok.sum() >= 3 and np.ptp(t[ok]) > 0:
        c = np.polyfit(t[ok][-6:], risk[ok][-6:], 2)
        f["h_risk_quad_extrap"] = float(np.clip(np.polyval(c, -T_EVAL_DAYS), RISK_FLOOR, 0.0))
        f["h_risk_quad_curv"] = float(c[0])
    else:
        f["h_risk_quad_extrap"] = float(risk[-1])
        f["h_risk_quad_curv"] = 0.0
    # linear extrapolation to T_EVAL_DAYS (bounded)
    f["h_risk_lin_extrap"] = float(np.clip(risk[-1] + f["h_risk_slope"] * (tta[-1] - T_EVAL_DAYS),
                                           RISK_FLOOR, 0.0))
    # covariance shrinkage: per-day slope of log combined along-track sigma
    def comb_sigma(r):
        a, b = _num(r.get("t_sigma_t")), _num(r.get("c_sigma_t"))
        return math.sqrt(a * a + b * b) if np.isfinite(a) and np.isfinite(b) else float("nan")
    sig = np.array([comb_sigma(r) for r in rows])
    with np.errstate(divide="ignore", invalid="ignore"):
        lsig = np.log(sig)
    rate = _slope(t, lsig)                     # d log sigma / day (negative = shrinking)
    f["h_log_sigma_rate"] = rate
    csig = np.array([_num(r.get("c_sigma_t")) for r in rows])
    with np.errstate(divide="ignore", invalid="ignore"):
        f["h_c_log_sigma_rate"] = _slope(t, np.log(csig))
    # clip in log space: exp() of an extreme rate would overflow
    shrink = (float(np.exp(np.clip(rate * (tta[-1] - T_EVAL_DAYS), math.log(0.05), math.log(1.5))))
              if np.isfinite(rate) else 1.0)
    f["h_sigma_shrink_to_eval"] = shrink
    maha = np.array([_num(r.get("mahalanobis_distance")) for r in rows])
    f["h_maha_slope"] = _slope(t, maha)
    # headroom to the CDM's own maximum-risk estimate
    mre = _num(last.get("max_risk_estimate"))
    f["h_headroom_mre"] = float(mre - risk[-1]) if np.isfinite(mre) else float("nan")
    # physics on the latest CDM + shrink-projected risk
    ph = physics.encounter_plane(last)
    f.update({f"p_{k}": v for k, v in ph.items()})
    f["p_proj_risk_trend"] = physics.projected_risk(float(risk[-1]), ph["m2"], shrink)
    # Pc can only rise with a SHRINKING covariance in the dilution region
    # (m^2 < 2); above it the Pc-max gap needs covariance growth instead
    f["p_escalation_headroom"] = (ph["log10_pcmax_gap"] if np.isfinite(ph["m2"]) and ph["m2"] < 2.0
                                  else 0.0 if np.isfinite(ph["m2"]) else float("nan"))
    ph0 = physics.encounter_plane(rows[0])
    f["p_m2_log_ratio"] = (float(math.log(max(ph["m2"], 1e-9) / max(ph0["m2"], 1e-9)))
                           if np.isfinite(ph["m2"]) and np.isfinite(ph0["m2"]) else float("nan"))
    return f


def _slog10(x: float) -> float:
    """Signed log10: monotone, so tree splits are unchanged, and finite in
    float32 (xgboost's input type) for any float64 value."""
    return math.copysign(math.log10(1.0 + abs(x)), x) if np.isfinite(x) else float("nan")


def last_cdm_features(ev: Event) -> dict:
    """All numeric columns of the latest admissible CDM (the uriot_lgbm
    input) plus a one-hot of the chaser object type.

    The Kelvins ``*_covariance_det`` columns reach 6.7e46 on the real files,
    beyond float32 (3.4e38): xgboost would read them as inf and refuse to
    fit. They enter as a signed log10 of the whole column (every event, every
    table), never only the overflowing values."""
    last = ev.inputs[-1]
    out = {}
    for k, v in last.items():
        if k in _EXCLUDE or k == "c_object_type":
            continue
        x = _num(v)
        if isinstance(v, (int, float, np.integer, np.floating)) or np.isfinite(x):
            if k.endswith("_covariance_det"):
                out[f"l_{k}_slog10"] = _slog10(x)
            else:
                out[f"l_{k}"] = x
    # one-hot of the chaser object type, categories taken from the data
    # (Bundle aligns test tables to the training columns and fills 0)
    ctype = str(last.get("c_object_type", "") or "NONE").strip().upper()
    out[f"l_ctype_{ctype.replace(' ', '_').lower()}"] = 1.0
    return out


# --- serving track -------------------------------------------------------------

_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)
_ISO = "%Y-%m-%dT%H:%M:%SZ"


def serving_cdms(ev: Event) -> list[dict]:
    """Kelvins inputs rendered as Kessler CDM dicts (what the app stores)."""
    out = []
    for r in ev.inputs:
        mt = _EPOCH - timedelta(days=float(r["time_to_tca"]))
        risk = _num(r["risk"])
        out.append({
            "message_time": mt.strftime(_ISO),
            "miss_distance_m": _num(r.get("miss_distance")),
            "relative_velocity_ms": _num(r.get("relative_speed")),
            "probability_of_collision": 10.0 ** risk if np.isfinite(risk) else None,
        })
    return out


def serving_features(ev: Event) -> dict:
    from backend.app.models.features import FEATURE_NAMES, build_features

    feats = build_features(serving_cdms(ev), _EPOCH.strftime(_ISO))
    return {f"s_{k}": float(feats[k]) for k in FEATURE_NAMES}


def escalation_serving_features(ev: Event) -> dict:
    """Exactly the production VIGIL-E feature vector (escalation.py)."""
    from backend.app.models.escalation import ESCALATION_FEATURE_NAMES, escalation_features

    feats = escalation_features(serving_cdms(ev), _EPOCH.strftime(_ISO))
    return {f"e_{k}": float(feats[k]) for k in ESCALATION_FEATURE_NAMES}


# --- tables --------------------------------------------------------------------

def table(events: list[Event], track: str = "benchmark"):
    """pandas DataFrame, one row per event, index = event_id.

    track='benchmark' -> last-CDM columns + history + physics + serving;
    track='serving'   -> serving features only (VIGIL's production vector);
    track='eserving'  -> VIGIL-E's production vector;
    track='last'      -> last-CDM columns only (the sesc recipe).
    """
    import pandas as pd

    rows = []
    for ev in events:
        if track == "serving":
            row = serving_features(ev)
        elif track == "eserving":
            row = escalation_serving_features(ev)
        elif track == "last":
            row = last_cdm_features(ev)
        else:
            row = {**last_cdm_features(ev), **history_features(ev), **serving_features(ev)}
        rows.append(row)
    df = pd.DataFrame(rows, index=[e.event_id for e in events])
    return df.astype(float)
