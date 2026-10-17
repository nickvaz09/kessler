"""GET /reentry - decaying-object watch list.

TLE-DERIVED ESTIMATE - not A REENTRY PREDICTION. Every number here comes
from the latest general-perturbations TLE per satellite: perigee/apogee from
mean motion + eccentricity, and the decay rate from the TLE's mean-motion
derivative (ndot) term. Real reentry forecasting needs drag modeling and
observation fits far beyond a TLE; the est_window buckets are coarse triage
labels, and the frontend carries the honesty label verbatim.

Decay-rate conversion (documented per contract):

    a = (mu / n^2)^(1/3)         =>  a is proportional to n^(-2/3)
    da/dt = -(2/3) * (a / n) * dn/dt

TLE line 1 columns 34-43 store ndot/2 in rev/day^2 (drag makes it positive:
the orbit shrinks, the satellite speeds up). With n in rev/day from the same
TLE the rev units cancel, so

    decay_rate_km_day = (2/3) * a_km * (2 * ndot_field) / n

is the semi-major-axis LOSS in km/day (positive = decaying; a negative
value means the fitted ndot is negative - no decay estimate is possible and
the window falls in the '>90 D' bucket).

est_window buckets divide the perigee margin above REENTRY_FLOOR_KM by that
rate:  days = (perigee_km - 100) / decay_rate_km_day  ->
'<7 D' | '<30 D' | '<90 D' | '>90 D'.

Results are cached in-process for 10 minutes, keyed by the CLAMPED perigee
threshold (120..400 km) - the scan derives elements for every catalog object,
which is too hot to run per request at full-catalog size.
"""
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query

from backend.app import db

router = APIRouter()

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

THRESHOLD_MIN_KM = 120.0   # clamp floor for ?perigee_km
THRESHOLD_MAX_KM = 400.0   # clamp ceiling
REENTRY_FLOOR_KM = 100.0   # perigee margin is measured above this altitude
MAX_ITEMS = 100            # response cap, lowest perigees first
CACHE_TTL_S = 600.0        # 10-minute in-process cache

# threshold_km -> (computed_at_s, payload). In-process only: one entry per
# distinct clamped threshold; entries expire by age, never eagerly evicted.
_cache: dict[float, tuple[float, dict]] = {}

# Latest TLE per satellite - same correlated-subquery pattern as
# api/objects._OBJECTS_SQL (newest epoch wins, id breaks ties).
_LATEST_TLE_SQL = """
SELECT
    sat.norad_id AS norad_id,
    sat.name     AS name,
    t.epoch      AS epoch,
    t.line1      AS line1,
    t.line2      AS line2
FROM satellites sat
JOIN tle_history t ON t.id = (
    SELECT th.id FROM tle_history th
     WHERE th.satellite_id = sat.id
     ORDER BY th.epoch DESC, th.id DESC LIMIT 1)
"""


def _now_s() -> float:
    """Cache clock (wall seconds). Split out so tests can monkeypatch it."""
    return time.time()


def _ndot_rev_day2(line1: str) -> float:
    """Mean-motion first derivative in rev/day^2 from TLE line 1.

    Columns 34-43 (0-indexed [33:43]) hold ndot/2 in rev/day^2 per the TLE
    format spec; multiply by 2 for the true derivative. Parsed from the raw
    line because sgp4's Satrec.ndot is unit-converted (rad/min^2) and keeps
    the /2 convention - the raw field is unambiguous.
    """
    return 2.0 * float(line1[33:43])


def _est_window(perigee_km: float, decay_rate_km_day: float) -> str:
    """Bucketed time-to-floor estimate. Triage label, not a prediction."""
    if decay_rate_km_day <= 0.0:
        return ">90 D"
    days = (perigee_km - REENTRY_FLOOR_KM) / decay_rate_km_day
    if days < 7.0:
        return "<7 D"
    if days < 30.0:
        return "<30 D"
    if days < 90.0:
        return "<90 D"
    return ">90 D"


def _epoch_age_h(epoch_iso: str, now: datetime) -> float | None:
    try:
        epoch = datetime.strptime(epoch_iso, ISO_FMT).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return round((now - epoch).total_seconds() / 3600.0, 1)


def _scan(conn, threshold_km: float) -> list[dict]:
    """All catalog objects whose latest-TLE perigee is under the threshold,
    lowest perigee first, uncapped - the endpoint slices to MAX_ITEMS and
    reports the true total as `count`. Objects whose TLE fails to parse (or
    lacks a stored TLE - excluded by the JOIN) are skipped, never 500s: a
    corrupt line must not take the watch list down.
    """
    from backend.app.ingestion import propagate  # lazy, like api/objects.py

    now = datetime.now(timezone.utc)
    items = []
    for row in conn.execute(_LATEST_TLE_SQL):
        try:
            elements = propagate.orbital_elements(row["line1"], row["line2"])
            perigee = float(elements["perigee_km"])
            if not perigee < threshold_km:
                continue
            # A perigee at/below the reentry floor is a degenerate TLE fit
            # (HEO mean elements can even go negative) - the object has
            # effectively decayed, and '-36.4 km · <7 D' on the decay-watch
            # panel reads as nonsense, not honesty.
            if perigee < REENTRY_FLOOR_KM:
                continue
            ndot = _ndot_rev_day2(row["line1"])
            rate = (2.0 / 3.0) * float(elements["semi_major_axis_km"]) \
                * ndot / float(elements["mean_motion_rev_day"])
        except Exception:
            continue
        items.append({
            "norad_id": row["norad_id"],
            "name": row["name"],
            "perigee_km": round(perigee, 1),
            "apogee_km": round(float(elements["apogee_km"]), 1),
            "epoch": row["epoch"],
            "epoch_age_h": _epoch_age_h(row["epoch"], now),
            "decay_rate_km_day": round(rate, 4),
            "est_window": _est_window(perigee, rate),
        })
    items.sort(key=lambda item: (item["perigee_km"], item["norad_id"]))
    return items


@router.get("/reentry")
def get_reentry(perigee_km: float = Query(250.0), conn=Depends(db.get_db)):
    """Decaying objects with perigee under the (clamped) threshold.

    The threshold clamps to 120..400 km instead of 422ing - any numeric
    input yields a sane watch list. Cached 10 min per clamped threshold.
    `count` is the true number of matches before the MAX_ITEMS cap.
    """
    threshold = max(THRESHOLD_MIN_KM, min(THRESHOLD_MAX_KM, perigee_km))
    now_s = _now_s()
    cached = _cache.get(threshold)
    if cached is not None and now_s - cached[0] < CACHE_TTL_S:
        return cached[1]
    items = _scan(conn, threshold)
    payload = {"count": len(items), "items": items[:MAX_ITEMS]}
    _cache[threshold] = (now_s, payload)
    return payload
