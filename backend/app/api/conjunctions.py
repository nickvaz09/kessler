"""Conjunction endpoints and the shared summary-item builder.

The summary item here is the single implementation consumed by /conjunctions,
/objects/{norad_id} (embedded conjunction list) and /status (risk-level
counting). GET /conjunctions/{id}/ephemeris serves the Encounter Simulator's
TCA-centered 1 s grid. Shapes are authoritative in docs/CONTRACTS.md.
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field

from backend.app import config, db

router = APIRouter()

# SQLite INTEGER is int64; range-check every int bound into SQL (422, not 500).
INT64_MAX = 9223372036854775807


def now_iso() -> str:
    """Current UTC time, ISO Z. Module-level so tests can freeze it."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# A conjunction is live only while its TCA is still ahead. Rows whose TCA has
# passed between ingest sweeps (pipeline.refresh step 5) must never read as
# active - the sweep only runs per-ingest, so the read side enforces it.
ACTIVE_WHERE = "(c.status = 'active' AND c.time_of_closest_approach >= ?)"
PAST_WHERE = ("(c.status = 'past'"
              " OR (c.status = 'active' AND c.time_of_closest_approach < ?))")

# Latest CDM and latest risk prediction per conjunction via correlated
# subqueries - fine at this catalog scale (see CONTRACTS.md).
_SUMMARY_SQL = """
SELECT
    c.id                         AS id,
    c.time_of_closest_approach   AS tca,
    c.first_seen_at              AS first_seen_at,
    c.status                     AS status,
    p.id                         AS primary_id,
    p.norad_id                   AS primary_norad_id,
    p.name                       AS primary_name,
    p.object_type                AS primary_object_type,
    s.id                         AS secondary_id,
    s.norad_id                   AS secondary_norad_id,
    s.name                       AS secondary_name,
    s.object_type                AS secondary_object_type,
    lc.miss_distance_m           AS miss_distance_m,
    lc.relative_velocity_ms      AS relative_velocity_ms,
    lc.probability_of_collision  AS pc_latest,
    (lc.covariance_json IS NOT NULL) AS has_covariance,
    (SELECT COUNT(*) FROM cdm_messages m
      WHERE m.conjunction_id = c.id)  AS cdm_count,
    rp.predicted_final_risk      AS predicted_final_risk,
    rp.model_version             AS model_version,
    rp.escalation_prob           AS escalation_prob,
    rp.escalation_model_version  AS escalation_model_version
FROM conjunctions c
JOIN satellites p ON p.id = c.primary_object_id
JOIN satellites s ON s.id = c.secondary_object_id
LEFT JOIN cdm_messages lc ON lc.id = (
    SELECT m.id FROM cdm_messages m
     WHERE m.conjunction_id = c.id
     ORDER BY CASE WHEN m.source = 'spacetrack' THEN 0 ELSE 1 END,
              m.message_time DESC, m.id DESC LIMIT 1)
LEFT JOIN risk_predictions rp ON rp.id = (
    SELECT r.id FROM risk_predictions r
     WHERE r.conjunction_id = c.id
     ORDER BY r.prediction_time DESC, r.id DESC LIMIT 1)
"""


def risk_level_for(effective: float) -> str:
    """Single risk-level definition (docs/CONTRACTS.md 'Risk levels')."""
    if effective >= config.PC_CRITICAL:
        return "critical"
    if effective >= config.PC_ELEVATED:
        return "elevated"
    return "nominal"


def effective_pc(item: dict) -> float:
    """predicted_final_risk if present, else pc_latest, else 0."""
    if item["predicted_final_risk"] is not None:
        return item["predicted_final_risk"]
    if item["pc_latest"] is not None:
        return item["pc_latest"]
    return 0.0


def sort_key(item: dict):
    """Effective risk desc, then TCA asc (ISO strings sort chronologically)."""
    return (-effective_pc(item), item["tca"])


def query_summaries(conn, where: str = "", params=()) -> list[dict]:
    """Conjunction summary items (without pc_trend) matching `where`."""
    sql = _SUMMARY_SQL + (f" WHERE {where}" if where else "")
    items = []
    for row in conn.execute(sql, params):
        item = {
            "id": row["id"],
            "tca": row["tca"],
            "primary": {
                "id": row["primary_id"],
                "norad_id": row["primary_norad_id"],
                "name": row["primary_name"],
                "object_type": row["primary_object_type"],
            },
            "secondary": {
                "id": row["secondary_id"],
                "norad_id": row["secondary_norad_id"],
                "name": row["secondary_name"],
                "object_type": row["secondary_object_type"],
            },
            "miss_distance_m": row["miss_distance_m"],
            "relative_velocity_ms": row["relative_velocity_ms"],
            "pc_latest": row["pc_latest"],
            # covariance passthrough: 'covariance' when the latest CDM
            # carries a real RTN covariance (cdm_messages.covariance_json,
            # Space-Track cdm class), else the assumed-sigma path.
            "pc_source": "covariance" if row["has_covariance"] else "assumed",
            "predicted_final_risk": row["predicted_final_risk"],
            "model_version": row["model_version"],
            # VIGIL-E early warning (docs/v2/vigil-e.md); null = no artifact
            # or not applicable (latest risk already >= threshold)
            "escalation_prob": row["escalation_prob"],
            "escalation_model_version": row["escalation_model_version"],
            "cdm_count": row["cdm_count"],
            "first_seen_at": row["first_seen_at"],
            "status": row["status"],
        }
        item["risk_level"] = risk_level_for(effective_pc(item))
        items.append(item)
    return items


def attach_pc_trends(conn, items: list[dict]) -> None:
    """Add pc_trend (Pc ascending by message_time, nulls skipped) to items."""
    trends: dict[int, list[float]] = {item["id"]: [] for item in items}
    if trends:
        placeholders = ",".join("?" * len(trends))
        rows = conn.execute(
            "SELECT conjunction_id, probability_of_collision"
            "  FROM cdm_messages"
            f" WHERE conjunction_id IN ({placeholders})"
            "   AND probability_of_collision IS NOT NULL"
            " ORDER BY message_time ASC, id ASC",
            list(trends),
        )
        for row in rows:
            trends[row["conjunction_id"]].append(row["probability_of_collision"])
    for item in items:
        item["pc_trend"] = trends[item["id"]]


# --- space weather layer (docs/v2/spaceweather.md) ---------------------------

STORM_WIDEN_FACTOR = 1.25   # flag a pair when the storm widens its 1-sigma >= 25 %
_TLE_BATCH = 400


def latest_tles(conn, satellite_ids) -> dict:
    """satellite_id -> newest tle_history row (epoch, line1, line2), batched."""
    ids = sorted({int(i) for i in satellite_ids if i is not None})
    out: dict = {}
    for k in range(0, len(ids), _TLE_BATCH):
        chunk = ids[k:k + _TLE_BATCH]
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
                "SELECT t.satellite_id, t.epoch, t.line1, t.line2 FROM tle_history t"
                f" WHERE t.satellite_id IN ({marks}) AND t.id = ("
                "   SELECT th.id FROM tle_history th WHERE th.satellite_id = t.satellite_id"
                "    ORDER BY th.epoch DESC, th.id DESC LIMIT 1)", chunk):
            out[row["satellite_id"]] = row
    return out


def _isotropic_pc(miss_m: float, sigma_m: float) -> float:
    """The assumed-sigma Pc of models/pc.py for a given combined sigma."""
    if not sigma_m or sigma_m <= 0:
        return 0.0
    hbr = config.HARD_BODY_RADIUS_M
    two_var = 2.0 * sigma_m * sigma_m
    return min(1.0, (hbr * hbr / two_var) * math.exp(-(miss_m * miss_m) / two_var))


def attach_space_weather(conn, items: list[dict], ctx=None) -> None:
    """Add `space_weather` to conjunction items while live space-weather data
    is available (absent otherwise): both objects' storm drift at TCA from
    their own TLE drag terms (app.spaceweather), next to the assumed sigma
    of the encounter view.

    The stored Pc and the risk level are untouched - a storm makes the
    prediction less certain; it does not move it. `flag` marks the pairs an
    operator should look at again: 'watch' (already elevated/critical and the
    storm widens it) or 'may_escalate' (nominal now, but the widened
    uncertainty allows an elevated Pc). A wider sigma that only dilutes Pc
    is never presented as comfort."""
    if not items:
        return
    try:
        from backend.app import spaceweather as sw
        ctx = ctx or sw.storm_context()
    except Exception:
        return
    if ctx is None:
        return
    tles = latest_tles(conn, [it[side]["id"] for it in items
                              for side in ("primary", "secondary") if it.get(side)])
    for it in items:
        tle_p = tles.get(it["primary"]["id"])
        tle_s = tles.get(it["secondary"]["id"])
        try:
            tca = _parse_iso_utc(it["tca"])
        except (TypeError, ValueError):
            continue
        sigma = _combined_sigma_m(tca, tle_p, tle_s)
        if not sigma:
            continue
        drift = sw.pair_drift_m(
            (tle_p["line1"], tle_p["line2"]) if tle_p else None,
            (tle_s["line1"], tle_s["line2"]) if tle_s else None, tca, ctx)
        total = math.hypot(sigma, drift)
        widened = total / sigma >= STORM_WIDEN_FACTOR
        miss = float(it.get("miss_distance_m") or 0.0)
        pc_now, pc_storm = _isotropic_pc(miss, sigma), _isotropic_pc(miss, total)
        flag = None
        if widened:
            if it.get("risk_level") in ("elevated", "critical"):
                flag = "watch"
            elif pc_storm >= config.PC_ELEVATED and pc_storm > pc_now:
                flag = "may_escalate"
        it["space_weather"] = {
            "drift_m": round(drift),
            "sigma_m": round(sigma),
            "sigma_storm_m": round(total),
            "factor": round(total / sigma, 2),
            "widened": widened,
            "pc_storm": pc_storm,
            "flag": flag,
            "kp": ctx.kp_now,
            "updated": ctx.updated,
        }


_active_storm = {"key": None, "items": None}


def active_storm_items(conn, ctx, active_items: list[dict] | None = None) -> list[dict]:
    """Every active conjunction with the storm layer, cached per (conditions
    payload x active set) - shared by /status, /spaceweather/impact and the
    assistant, so the catalog-wide pass runs once per NOAA refresh."""
    items = active_items if active_items is not None else query_summaries(
        conn, ACTIVE_WHERE, (now_iso(),))
    key = (ctx.updated, tuple(i["id"] for i in items))
    if _active_storm["key"] != key:
        layered = [dict(i) for i in items]
        attach_space_weather(conn, layered, ctx)
        _active_storm.update(key=key, items=layered)
    return _active_storm["items"]


FLAG_ORDER = {"watch": 0, "may_escalate": 1}


def storm_flagged(items: list[dict]) -> list[dict]:
    """Flagged pairs, the ones to look at first: 'watch' before
    'may_escalate', then the highest storm-time Pc."""
    out = [i for i in items if (i.get("space_weather") or {}).get("flag")]
    out.sort(key=lambda i: (FLAG_ORDER[i["space_weather"]["flag"]],
                            -i["space_weather"]["pc_storm"]))
    return out


def _early_warning_threshold() -> tuple[float | None, str | None]:
    """(alert_threshold, model version) of the trained VIGIL-E, else (None, None)."""
    try:
        from backend.app.models import escalation
        meta = escalation.read_meta()
    except Exception:
        return None, None
    if not meta:
        return None, None
    value = meta.get("alert_threshold")
    if not isinstance(value, (int, float)):
        return None, None
    version = meta.get("version")
    return float(value), (version if isinstance(version, str) else None)


@router.get("/conjunctions")
def list_conjunctions(
    status: str = "active",
    risk: str | None = None,
    limit: int = Query(100, ge=0, le=1000),
    offset: int = Query(0, ge=0),
    early_warning: bool = False,
    conn=Depends(db.get_db),
):
    if status == "active":
        where, params = ACTIVE_WHERE, (now_iso(),)
    elif status == "past":
        where, params = PAST_WHERE, (now_iso(),)
    elif status == "all":
        where, params = "", ()
    else:
        where, params = "c.status = ?", (status,)
    items = query_summaries(conn, where, params)
    if risk:
        items = [item for item in items if item["risk_level"] == risk]
    if early_warning:
        # VIGIL-E early warnings: events still below the high-risk threshold
        # whose escalation probability meets the model's own alert threshold
        # (fixed at training time from the alert budget). No artifact -> none.
        threshold, version = _early_warning_threshold()
        items = [] if threshold is None else [
            item for item in items
            if item["risk_level"] == "nominal"
            and item.get("escalation_prob") is not None
            and item["escalation_prob"] >= threshold
            # a previous model's probabilities are not on this threshold
            and (version is None or item.get("escalation_model_version") == version)]
        items.sort(key=lambda it: -it["escalation_prob"])
    else:
        items.sort(key=sort_key)
    count = len(items)
    page = items[offset:offset + limit]
    attach_pc_trends(conn, page)
    attach_space_weather(conn, page)
    return {"count": count, "items": page}


@router.get("/conjunctions/{conjunction_id}")
def get_conjunction(
    conjunction_id: int = Path(ge=0, le=INT64_MAX),
    conn=Depends(db.get_db),
):
    items = query_summaries(conn, "c.id = ?", (conjunction_id,))
    if not items:
        raise HTTPException(
            status_code=404, detail=f"Conjunction {conjunction_id} not found"
        )
    item = items[0]
    attach_pc_trends(conn, [item])
    attach_space_weather(conn, [item])
    item["cdms"] = [
        dict(row)
        for row in conn.execute(
            "SELECT message_time, miss_distance_m, relative_velocity_ms,"
            "       probability_of_collision, source"
            "  FROM cdm_messages WHERE conjunction_id = ?"
            " ORDER BY message_time ASC, id ASC",
            (conjunction_id,),
        )
    ]
    item["risk_predictions"] = [
        dict(row)
        for row in conn.execute(
            "SELECT predicted_final_risk, prediction_time, model_version"
            "  FROM risk_predictions WHERE conjunction_id = ?"
            " ORDER BY prediction_time ASC, id ASC",
            (conjunction_id,),
        )
    ]
    return item


# --- Encounter ephemeris (Encounter Simulator) --------------------------------

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

EARTH_ROT_RAD_S = 7.2921159e-5  # mean Earth rotation rate [rad/s]
EPHEM_WINDOW_MIN_S = 60         # window_s clamp (contract: 60..3600)
EPHEM_WINDOW_MAX_S = 3600
EPHEM_STEP_S = 1.0              # grid step is fixed by contract
REFINE_HALF_WINDOW_S = 90.0     # TCA re-refinement half-window around stored TCA
REFINE_STEP_S = 0.05            # TCA re-refinement grid resolution [s]

_EPHEMERIS_PAIR_SQL = """
SELECT c.id                       AS id,
       c.time_of_closest_approach AS tca,
       p.id                       AS primary_id,
       p.norad_id                 AS primary_norad_id,
       p.name                     AS primary_name,
       s.id                       AS secondary_id,
       s.norad_id                 AS secondary_norad_id,
       s.name                     AS secondary_name
FROM conjunctions c
JOIN satellites p ON p.id = c.primary_object_id
JOIN satellites s ON s.id = c.secondary_object_id
WHERE c.id = ?
"""

_LATEST_TLE_SQL = (
    "SELECT epoch, line1, line2 FROM tle_history"
    " WHERE satellite_id = ? ORDER BY epoch DESC, id DESC LIMIT 1"
)


def _null_tca_state() -> dict:
    """Degraded tca_state shape (contract: nulls when the pair cannot be built)."""
    return {
        "sep_m": None,
        "rel_vel_ms": None,
        "ric_m": {"radial": None, "in_track": None, "cross_track": None},
        "approach_angle_deg": None,
        "rel_vel_hat": None,
    }


def _parse_iso_utc(s: str) -> datetime:
    """ISO-8601 'Z' string -> aware UTC datetime (fractional seconds tolerated)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def _satrec_or_none(tle_row):
    """Satrec from a tle_history row; None when absent or unparseable."""
    if tle_row is None:
        return None
    try:
        from backend.app.ingestion import propagate  # lazy, like api/objects.py

        sat = propagate.satrec_from_tle(tle_row["line1"], tle_row["line2"])
    except Exception:
        return None
    return sat if sat.no_kozai > 0 else None


def _pair_states(sat_p, sat_s, times):
    """Batch-propagate both objects to `times` (one sgp4_array call each).

    Returns (rp, vp, rs, vs, ok) - km / km/s TEME arrays plus a per-sample
    mask that is True only where both propagations are clean and finite.
    """
    from backend.app.ingestion import propagate

    jd, fr = propagate.jday_arrays(times)
    ep, rp, vp = sat_p.sgp4_array(jd, fr)
    es, rs, vs = sat_s.sgp4_array(jd, fr)
    ok = ((ep == 0) & (es == 0)
          & np.isfinite(rp).all(axis=1) & np.isfinite(rs).all(axis=1))
    return rp, vp, rs, vs, ok


def _refine_tca(sat_p, sat_s, stored_tca: datetime) -> datetime | None:
    """SGP4 re-refinement of the stored (screening-grade) TCA.

    Both latest TLEs are propagated on a REFINE_STEP_S grid spanning
    +/-REFINE_HALF_WINDOW_S around the stored TCA; the discrete minimum of the
    squared separation is then sharpened by parabolic interpolation of d^2
    through its neighbors (near TCA the relative motion is locally
    rectilinear, so d^2(t) is quadratic - same technique as
    ingestion.screening._refine) for sub-sample truth. Returns None when the
    pair never propagates cleanly anywhere on the grid.
    """
    n = int(round(2.0 * REFINE_HALF_WINDOW_S / REFINE_STEP_S)) + 1
    t0 = stored_tca - timedelta(seconds=REFINE_HALF_WINDOW_S)
    times = [t0 + timedelta(seconds=REFINE_STEP_S * i) for i in range(n)]
    rp, _vp, rs, _vs, ok = _pair_states(sat_p, sat_s, times)
    if not ok.any():
        return None
    d = rs - rp
    d2 = np.einsum("ij,ij->i", d, d)
    d2[~ok] = np.inf
    i = int(np.argmin(d2))
    tca = times[i]
    if 0 < i < n - 1 and np.isfinite(d2[i - 1]) and np.isfinite(d2[i + 1]):
        y0, y1, y2 = float(d2[i - 1]), float(d2[i]), float(d2[i + 1])
        denom = y0 - 2.0 * y1 + y2
        if denom > 0.0:
            x = max(-1.0, min(1.0, 0.5 * (y0 - y2) / denom))
            tca = times[i] + timedelta(seconds=REFINE_STEP_S * x)
    return tca


def _tca_state(sat_p, sat_s, tca: datetime) -> dict | None:
    """Relative geometry at the refined instant; None when it cannot be built.

    ric_m is secondary-minus-primary in the primary's RIC frame
    (R-hat = r/|r|, C-hat = r x v / |r x v|, I-hat = C-hat x R-hat);
    approach_angle_deg is the angle between the two velocity vectors;
    rel_vel_hat is the unit relative velocity in TEME - the client's
    encounter-plane normal.
    """
    rp_, vp_, rs_, vs_, ok = _pair_states(sat_p, sat_s, [tca])
    if not bool(ok[0]):
        return None
    rp, vp, rs, vs = rp_[0], vp_[0], rs_[0], vs_[0]
    dr_m = (rs - rp) * 1000.0
    dv_ms = (vs - vp) * 1000.0
    sep_m = float(np.linalg.norm(dr_m))
    rel_vel_ms = float(np.linalg.norm(dv_ms))

    r_norm = float(np.linalg.norm(rp))
    h = np.cross(rp, vp)
    h_norm = float(np.linalg.norm(h))
    vp_norm = float(np.linalg.norm(vp))
    vs_norm = float(np.linalg.norm(vs))
    if r_norm == 0.0 or h_norm == 0.0 or vp_norm == 0.0 or vs_norm == 0.0:
        return None  # degenerate state: cannot build the RIC frame
    r_hat = rp / r_norm
    c_hat = h / h_norm
    i_hat = np.cross(c_hat, r_hat)

    cos_angle = float(np.dot(vp, vs)) / (vp_norm * vs_norm)
    approach_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))

    rel_vel_hat = None
    if rel_vel_ms > 0.0:
        rel_vel_hat = [round(float(u), 6) for u in dv_ms / rel_vel_ms]

    return {
        "sep_m": round(sep_m, 2),
        "rel_vel_ms": round(rel_vel_ms, 2),
        "ric_m": {
            "radial": round(float(np.dot(dr_m, r_hat)), 2),
            "in_track": round(float(np.dot(dr_m, i_hat)), 2),
            "cross_track": round(float(np.dot(dr_m, c_hat)), 2),
        },
        "approach_angle_deg": round(approach_deg, 2),
        "rel_vel_hat": rel_vel_hat,
    }


def _grid_positions(sat, times) -> list[list[float]]:
    """Positions on the 1 s grid, km TEME rounded to 3 dp; [] on any failure."""
    if sat is None:
        return []
    try:
        from backend.app.ingestion import propagate

        jd, fr = propagate.jday_arrays(times)
        e, r, _v = sat.sgp4_array(jd, fr)
        if np.any(e != 0) or not np.all(np.isfinite(r)):
            return []
        return np.round(r, 3).tolist()
    except Exception:
        return []


def _combined_sigma_m(tca: datetime, tle_p, tle_s) -> float | None:
    """Combined ASSUMED 1-sigma [m]: RSS of models.pc.sigma_at per TLE age.

    TLE states carry no covariance, so this is the documented TLE-class
    assumption (models/pc.py): each object contributes sigma_at(its TLE's
    propagation age at TCA), combined in quadrature. None when either TLE row
    is missing or models.pc is unavailable (lazy import).
    """
    if tle_p is None or tle_s is None:
        return None
    try:
        from backend.app.models import pc as pc_mod

        variance = 0.0
        for tle in (tle_p, tle_s):
            age_days = (tca - _parse_iso_utc(tle["epoch"])).total_seconds() / 86400.0
            variance += pc_mod.sigma_at(age_days) ** 2
        return round(math.sqrt(variance), 1)
    except Exception:
        return None


@router.get("/conjunctions/{conjunction_id}/ephemeris")
def get_ephemeris(
    conjunction_id: int = Path(ge=0, le=INT64_MAX),
    window_s: int = Query(1200),
    conn=Depends(db.get_db),
):
    """Encounter ephemeris: both objects on a 1 s grid centered on the TCA.

    The stored TCA is re-refined server-side (0.05 s grid + parabolic vertex,
    see _refine_tca) and the 2*window_s+1 sample grid is centered on that
    sub-second instant; tca_state comes from the refined instant. window_s is
    clamped to 60..3600, step_s is fixed at 1.0. `tca`/`start` strings are
    floored to the whole second (ISO Z convention) while gmst_start_rad
    corresponds to the exact grid start, so client-derived lat/lon stays
    consistent with the samples.

    Degrade paths (never a 500 on a bad TLE): a missing/corrupt/failing TLE
    yields r:[] for that object and, when the pair cannot be propagated
    together, the stored TCA with an all-null tca_state.
    """
    row = conn.execute(_EPHEMERIS_PAIR_SQL, (conjunction_id,)).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"Conjunction {conjunction_id} not found"
        )
    window = max(EPHEM_WINDOW_MIN_S, min(EPHEM_WINDOW_MAX_S, window_s))

    tle_p = conn.execute(_LATEST_TLE_SQL, (row["primary_id"],)).fetchone()
    tle_s = conn.execute(_LATEST_TLE_SQL, (row["secondary_id"],)).fetchone()
    sat_p = _satrec_or_none(tle_p)
    sat_s = _satrec_or_none(tle_s)

    tca = _parse_iso_utc(row["tca"])
    state = None
    if sat_p is not None and sat_s is not None:
        try:
            refined = _refine_tca(sat_p, sat_s, tca)
            if refined is not None:
                tca = refined
                state = _tca_state(sat_p, sat_s, tca)
        except Exception:
            state = None  # runtime propagation blow-up: degrade, never 500

    start = tca - timedelta(seconds=window)
    times = [start + timedelta(seconds=EPHEM_STEP_S * i)
             for i in range(2 * window + 1)]

    from backend.app.ingestion import propagate  # lazy, like api/objects.py

    return {
        "conjunction_id": row["id"],
        "tca": tca.strftime(ISO_FMT),
        "window_s": window,
        "step_s": EPHEM_STEP_S,
        "start": start.strftime(ISO_FMT),
        "gmst_start_rad": propagate.gmst_rad(start),
        "earth_rot_rad_s": EARTH_ROT_RAD_S,
        "primary": {
            "norad_id": row["primary_norad_id"],
            "name": row["primary_name"],
            "r": _grid_positions(sat_p, times),
        },
        "secondary": {
            "norad_id": row["secondary_norad_id"],
            "name": row["secondary_name"],
            "r": _grid_positions(sat_s, times),
        },
        "tca_state": state if state is not None else _null_tca_state(),
        "sigma_m": _combined_sigma_m(tca, tle_p, tle_s),
        **_geometry_check(conn, row["id"], state, tca, tle_p, tle_s),
    }


def _geometry_check(conn, conjunction_id: int, state: dict | None,
                    tca: datetime, tle_p, tle_s) -> dict:
    """Does the latest-TLE geometry agree with the latest message?

    The same rule the avoidance plan uses for status "geometry_mismatch"
    (models.maneuver.plan_avoidance): the pair is inconsistent when the
    latest-TLE miss already clears the plan's required miss - max(default
    target miss, the miss where the assumed-sigma Pc drops below
    PC_ELEVATED * PLAN_PC_MARGIN) - while the latest message's screened miss
    does not. Here the latest-TLE miss is this endpoint's refined tca_state
    sep_m (what the simulator shows) rather than the plan's +/-300 s grid
    minimum; the two agree whenever the minimum lies within the refinement
    window.

    Returns latest_message_miss_m (None without a message) and
    geometry_consistent: True / False, or None when either miss is unknown.
    """
    from backend.app.models import maneuver as maneuver_mod

    summary = query_summaries(conn, "c.id = ?", (conjunction_id,))
    screened = summary[0]["miss_distance_m"] if summary else None
    out = {
        "latest_message_miss_m": None if screened is None
        else round(float(screened), 1),
        "geometry_consistent": None,
    }
    sep = state.get("sep_m") if state else None
    if screened is None or sep is None:
        return out
    rel_vel = state.get("rel_vel_ms") or 0.0
    try:
        pc_req = maneuver_mod._pc_required_miss_m(
            lambda m: _assumed_pc(m, rel_vel, tca, tle_p, tle_s),
            config.PC_ELEVATED * maneuver_mod.PLAN_PC_MARGIN)
    except Exception:
        pc_req = None
    required = max(maneuver_mod.PLAN_DEFAULT_TARGET_MISS_M, pc_req or 0.0)
    out["geometry_consistent"] = not (sep >= required and screened < required)
    return out


# --- What-if maneuver sandbox ------------------------------------------------

MANEUVER_DV_MAX_MS = 5.0        # contract: dv_along_ms in -5..5 m/s
MANEUVER_MIN_LEAD_S = 300.0     # burn must be >= 5 min before TCA
MANEUVER_MAX_PAST_S = 3600.0    # burn must be >= now - 1 h
MANEUVER_SEARCH_HALF_S = 300.0  # min-separation search half-window around TCA
MANEUVER_STEP_S = 0.1           # search grid resolution [s]

MANEUVER_METHOD = (
    "two-body differential: SGP4 state at the burn, dv applied along the "
    "velocity vector, post-burn minus no-burn two-body offset added to the "
    "SGP4 primary (drag/J2 cancel common-mode); min-separation search "
    "TCA+/-300 s at 0.1 s; assumed-sigma Pc"
)


class ManeuverIn(BaseModel):
    dv_along_ms: float = Field(ge=-MANEUVER_DV_MAX_MS, le=MANEUVER_DV_MAX_MS)
    burn_iso: str


def _argmin_refined(d2: np.ndarray, ok: np.ndarray) -> tuple[float, float]:
    """(offset_samples, min_dist_m) of the masked d^2 [km^2] series.

    Discrete argmin sharpened by parabolic interpolation through its
    neighbors (d^2(t) is locally quadratic near a close approach - same
    technique as _refine_tca), for both the sub-sample instant and the
    sub-sample minimum value.
    """
    d2 = np.where(ok, d2, np.inf)
    i = int(np.argmin(d2))
    off = float(i)
    d2min = float(d2[i])
    if 0 < i < len(d2) - 1 and np.isfinite(d2[i - 1]) and np.isfinite(d2[i + 1]):
        y0, y1, y2 = float(d2[i - 1]), float(d2[i]), float(d2[i + 1])
        a = 0.5 * (y0 - 2.0 * y1 + y2)   # parabola y1 + b*x + a*x^2, x in samples
        b = 0.5 * (y2 - y0)
        if a > 0.0:
            x = max(-1.0, min(1.0, -b / (2.0 * a)))
            off = i + x
            d2min = max(0.0, y1 + b * x + a * x * x)
    return off, math.sqrt(max(d2min, 0.0)) * 1000.0


def _assumed_pc(miss_m: float, rel_vel_ms: float, tca: datetime,
                tle_p, tle_s) -> float | None:
    """Assumed-sigma Pc with both TLEs' propagation ages at tca.

    Same construction as the user-object screen and the ephemeris sigma_m:
    each TLE contributes sigma_at(its own age to TCA), combined by RSS.
    compute_pc integrates sigma = sqrt(2)*sigma_at(age) for one shared age
    and sigma_at is linear in age, so the RSS maps to an exact equivalent
    age. None when models.pc or a TLE epoch is unavailable.
    """
    try:
        from backend.app.models import pc as pc_mod

        variance = 0.0
        for tle in (tle_p, tle_s):
            age = max((tca - _parse_iso_utc(tle["epoch"])).total_seconds(),
                      0.0) / 86400.0
            variance += pc_mod.sigma_at(age) ** 2
        per_object = math.sqrt(variance) / math.sqrt(2.0)
        growth = config.SIGMA_GROWTH_M_PER_DAY
        age_eq = ((per_object - config.SIGMA_BASE_M) / growth) if growth > 0 \
            else 0.0
        return pc_mod.compute_pc(miss_m, rel_vel_ms,
                                 propagation_age_days=max(age_eq, 0.0))
    except Exception:
        return None


def load_maneuver_context(conn, conjunction_id: int, burn: datetime) -> dict:
    """Everything dv-INDEPENDENT about a maneuver recompute, computed once.

    This is the endpoint's former inline setup, factored out so
    models.maneuver.optimize_dv can search many candidate dv values without
    re-running the SGP4 grid or the DB reads per evaluation. Validates the
    burn against the stored TCA and the wall clock, loads both latest TLEs,
    propagates the pair on the shared 0.1 s search grid across TCA +/- 300 s,
    computes the burn-instant state and the coast (no-burn) two-body grid, and
    the unmaneuvered baseline minimum.

    Raises LookupError for an unknown conjunction id and ValueError for every
    validation/propagation problem (message text is the HTTP detail the
    endpoint has always returned). No FastAPI types in or out.
    """
    row = conn.execute(_EPHEMERIS_PAIR_SQL, (conjunction_id,)).fetchone()
    if row is None:
        raise LookupError(f"Conjunction {conjunction_id} not found")
    tca = _parse_iso_utc(row["tca"])
    now = datetime.now(timezone.utc)
    if (tca - burn).total_seconds() < MANEUVER_MIN_LEAD_S:
        raise ValueError("burn must be at least 5 minutes before TCA")
    if (now - burn).total_seconds() > MANEUVER_MAX_PAST_S:
        raise ValueError("burn must not be more than 1 hour in the past")

    tle_p = conn.execute(_LATEST_TLE_SQL, (row["primary_id"],)).fetchone()
    tle_s = conn.execute(_LATEST_TLE_SQL, (row["secondary_id"],)).fetchone()
    sat_p = _satrec_or_none(tle_p)
    sat_s = _satrec_or_none(tle_s)
    if sat_p is None or sat_s is None:
        raise ValueError(
            "maneuver unavailable: a usable TLE is missing for one of"
            " the pair")

    from backend.app.ingestion import propagate  # lazy, like api/objects.py
    from backend.app.models import maneuver as maneuver_mod

    try:
        r_burn, v_burn = propagate.positions_at(
            tle_p["line1"], tle_p["line2"], [burn])
        r0 = np.asarray(r_burn[0], dtype=float)
        v0 = np.asarray(v_burn[0], dtype=float)
        speed = float(np.linalg.norm(v0))
        if speed == 0.0:
            raise ValueError("zero velocity at burn time")
        v_hat = v0 / speed
        el_coast = maneuver_mod.rv_to_elements(r0, v0)
    except ValueError as exc:
        raise ValueError(
            f"maneuver unavailable: burn-state propagation failed ({exc})")

    # 0.1 s search grid across TCA +/- 300 s, shared by all trajectories.
    n = int(round(2.0 * MANEUVER_SEARCH_HALF_S / MANEUVER_STEP_S)) + 1
    t0 = tca - timedelta(seconds=MANEUVER_SEARCH_HALF_S)
    times = [t0 + timedelta(seconds=MANEUVER_STEP_S * i) for i in range(n)]
    rp, vp, rs, vs, ok = _pair_states(sat_p, sat_s, times)
    if not ok.any():
        raise ValueError(
            "maneuver unavailable: the pair does not propagate across"
            " the search window")
    dt0 = (t0 - burn).total_seconds()
    dts = dt0 + MANEUVER_STEP_S * np.arange(n)
    rc, _ = maneuver_mod.propagate_two_body(el_coast, dts)

    # Unmaneuvered baseline (dv-independent): sub-sample minimum of sec - pri.
    d_old = rs - rp
    old_off, old_miss_m = _argmin_refined(
        np.einsum("ij,ij->i", d_old, d_old), ok)
    old_tca = t0 + timedelta(seconds=MANEUVER_STEP_S * old_off)
    i_old = int(round(min(max(old_off, 0.0), n - 1)))
    rel_vel_old = float(np.linalg.norm(vs[i_old] - vp[i_old])) * 1000.0

    return {
        "conjunction_id": row["id"],
        "tca": tca,
        "burn": burn,
        "tle_p": tle_p,
        "tle_s": tle_s,
        "r0": r0,
        "v0": v0,
        "v_hat": v_hat,
        "t0": t0,
        "n": n,
        "dts": dts,
        "vp": vp,
        "vs": vs,
        "ok": ok,
        "d_old": d_old,
        "rc": rc,
        "old": {
            "miss_m": old_miss_m,
            "tca": old_tca,
            "rel_vel_ms": rel_vel_old,
        },
    }


def evaluate_maneuver(ctx: dict, dv_along_ms: float) -> dict:
    """dv-DEPENDENT half of the maneuver recompute, against a loaded context.

    The post-burn primary is two-body (models/maneuver.py); the omitted
    post-burn forces are cancelled common-mode by differencing the burned
    two-body grid against the context's coast grid and applying that offset to
    the SGP4 primary. At dv = 0 the two grids are computed from bitwise-equal
    states, so the offset - and the delta vs the baseline - is exactly zero.
    Returns un-rounded numbers; callers round for presentation.
    """
    from backend.app.models import maneuver as maneuver_mod

    el_burned = maneuver_mod.rv_to_elements(
        ctx["r0"], ctx["v0"] + (float(dv_along_ms) / 1000.0) * ctx["v_hat"])
    rb, _ = maneuver_mod.propagate_two_body(el_burned, ctx["dts"])
    offset = rb - ctx["rc"]  # km: maneuvered- minus unmaneuvered-primary

    d_new = ctx["d_old"] - offset  # sec - (pri + offset)
    new_off, new_miss_m = _argmin_refined(
        np.einsum("ij,ij->i", d_new, d_new), ctx["ok"])
    new_tca = ctx["t0"] + timedelta(seconds=MANEUVER_STEP_S * new_off)
    i_new = int(round(min(max(new_off, 0.0), ctx["n"] - 1)))
    rel_vel_new = float(
        np.linalg.norm(ctx["vs"][i_new] - ctx["vp"][i_new])) * 1000.0

    return {
        "new_miss_m": new_miss_m,
        "new_tca": new_tca,
        "rel_vel_ms": rel_vel_new,
        "offset_km": offset[i_new],
    }


@router.post("/conjunctions/{conjunction_id}/maneuver")
def post_maneuver(
    body: ManeuverIn,
    conjunction_id: int = Path(ge=0, le=INT64_MAX),
    conn=Depends(db.get_db),
):
    """What-if along-track burn: recompute the encounter post-maneuver.

    Thin HTTP shell over load_maneuver_context + evaluate_maneuver (the same
    core the optimizer reuses): LookupError maps to 404, ValueError to 422.
    Pc uses the assumed-sigma path with the same per-TLE-age sigmas as the
    rest of the API (never the covariance path - a burn invalidates any
    delivered covariance).
    """
    try:
        burn = _parse_iso_utc(body.burn_iso)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="burn_iso is not a valid ISO-8601 UTC timestamp")
    try:
        ctx = load_maneuver_context(conn, conjunction_id, burn)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    new = evaluate_maneuver(ctx, body.dv_along_ms)
    old = ctx["old"]
    tle_p, tle_s = ctx["tle_p"], ctx["tle_s"]

    return {
        "conjunction_id": ctx["conjunction_id"],
        "dv_along_ms": body.dv_along_ms,
        "burn_iso": burn.strftime(ISO_FMT),
        "new_miss_m": round(new["new_miss_m"], 1),
        "new_tca": new["new_tca"].strftime(ISO_FMT),
        "delta_miss_m": round(new["new_miss_m"] - old["miss_m"], 1),
        "new_pc": _assumed_pc(new["new_miss_m"], new["rel_vel_ms"],
                              new["new_tca"], tle_p, tle_s),
        "old": {
            "miss_m": round(old["miss_m"], 1),
            "pc": _assumed_pc(old["miss_m"], old["rel_vel_ms"],
                              old["tca"], tle_p, tle_s),
        },
        "primary_rel_offset_km": [round(float(c), 6)
                                  for c in new["offset_km"]],
        "method": MANEUVER_METHOD,
    }


# --- One-click avoidance plan -------------------------------------------------

class AvoidancePlanIn(BaseModel):
    target_miss_m: float | None = Field(default=None, gt=0, le=100_000)
    max_dv_ms: float | None = Field(default=None, gt=0,
                                    le=MANEUVER_DV_MAX_MS)


@router.post("/conjunctions/{conjunction_id}/avoidance-plan")
def post_avoidance_plan(
    body: AvoidancePlanIn | None = None,
    conjunction_id: int = Path(ge=0, le=INT64_MAX),
    conn=Depends(db.get_db),
):
    """Smallest along-track burn per lead time, plus a recommendation.

    Thin HTTP shell over models.maneuver.plan_avoidance (contract in
    docs/v2/simulator-avoidance.md): LookupError -> 404, PlanConflict (TCA
    passed / too soon for any burn) -> 409, other ValueError -> 422.
    """
    from backend.app.models import maneuver as maneuver_mod

    body = body or AvoidancePlanIn()
    try:
        return maneuver_mod.plan_avoidance(
            conn, conjunction_id,
            target_miss_m=body.target_miss_m, max_dv_ms=body.max_dv_ms)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except maneuver_mod.PlanConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
