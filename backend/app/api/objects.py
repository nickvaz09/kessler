"""Catalog object endpoints: paginated listing with derived orbital elements,
per-object detail (TLE history, orbit path, conjunction summaries), and the
one-period ground track (/objects/{norad_id}/track).

`ingestion.propagate` is imported lazily inside helpers so a concurrently
built module can never break API import order; derivation failures degrade to
null elements instead of 500s.
"""
import math
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from backend.app import db
from backend.app.api import conjunctions as conjunctions_api

router = APIRouter()

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

# SQLite INTEGER is int64; Python ints are unbounded, so every int that is
# bound into SQL must be range-checked at the FastAPI layer (422, never 500).
INT64_MAX = 9223372036854775807

# Ground-track sample-count clamp (contract: 16..720, default 240).
TRACK_SAMPLES_MIN = 16
TRACK_SAMPLES_MAX = 720

# Per-object conjunction list cap: busy debris objects accumulate 1000+
# events; the detail payload carries the top CONJUNCTIONS_MAX by the existing
# risk sort and reports the true total in `conjunction_count`.
CONJUNCTIONS_MAX = 200

# Propagated samples below this altitude mean the object has effectively
# re-entered (stale TLE past decay): treat the whole track as failed.
TRACK_MIN_ALT_KM = 100.0

_NULL_SUBPOINT = {"lat_deg": None, "lon_deg": None, "alt_km": None, "velocity_kms": None}

# Element keys exposed on the object item (subset of propagate.orbital_elements).
_ELEMENT_KEYS = (
    "inclination_deg",
    "raan_deg",
    "eccentricity",
    "arg_perigee_deg",
    "mean_motion_rev_day",
    "period_min",
    "apogee_km",
    "perigee_km",
)

# Each satellite joined with its latest TLE (by epoch).
_OBJECTS_SQL = """
SELECT
    sat.id              AS id,
    sat.norad_id        AS norad_id,
    sat.name            AS name,
    sat.object_type     AS object_type,
    sat.intl_designator AS intl_designator,
    sat.last_updated    AS last_updated,
    t.epoch             AS tle_epoch,
    t.line1             AS line1,
    t.line2             AS line2
FROM satellites sat
LEFT JOIN tle_history t ON t.id = (
    SELECT th.id FROM tle_history th
     WHERE th.satellite_id = sat.id
     ORDER BY th.epoch DESC, th.id DESC LIMIT 1)
"""


def _derive_elements(line1, line2) -> dict:
    """Orbital elements from a TLE; nulls when absent or derivation fails."""
    if line1 and line2:
        try:
            from backend.app.ingestion import propagate

            elements = propagate.orbital_elements(line1, line2)
            return {key: elements.get(key) for key in _ELEMENT_KEYS}
        except Exception:
            pass
    return {key: None for key in _ELEMENT_KEYS}


def _orbit_path(line1, line2) -> list[list[float]]:
    """Sampled orbit positions [x,y,z] km; empty when unavailable."""
    if line1 and line2:
        try:
            from backend.app.ingestion import propagate

            return [[float(c) for c in point] for point in propagate.sample_orbit(line1, line2)]
        except Exception:
            pass
    return []


def _item_from_row(row) -> dict:
    return {
        "id": row["id"],
        "norad_id": row["norad_id"],
        "name": row["name"],
        "object_type": row["object_type"],
        "intl_designator": row["intl_designator"],
        "epoch": row["tle_epoch"],
        **_derive_elements(row["line1"], row["line2"]),
        "last_updated": row["last_updated"],
    }


def _escape_like(term: str) -> str:
    """Escape LIKE metacharacters so %, _ and \\ in a search match literally."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _search_filter(search) -> tuple[str, tuple]:
    if search and search.strip():
        term = search.strip()
        # Numeric -> exact norad match, but only when the value fits SQLite's
        # int64 (isdecimal, not isdigit: every isdecimal string int()-parses);
        # anything else falls through to the literal name search.
        if term.isdecimal() and int(term) <= INT64_MAX:
            return "sat.norad_id = ?", (int(term),)
        # LOWER(...) LIKE LOWER(...): case-insensitive on both SQLite and
        # Postgres (sqlite LIKE is insensitive by default; pg LIKE is not).
        return "LOWER(sat.name) LIKE LOWER(?) ESCAPE '\\'", (f"%{_escape_like(term)}%",)
    return "", ()


@router.get("/objects")
def list_objects(
    search: str | None = None,
    limit: int = Query(100, ge=0, le=1000),
    offset: int = Query(0, ge=0, le=INT64_MAX),
    conn=Depends(db.get_db),
):
    where, params = _search_filter(search)
    where_sql = f" WHERE {where}" if where else ""
    count = conn.execute(
        f"SELECT COUNT(*) FROM satellites sat{where_sql}", params
    ).fetchone()[0]
    rows = conn.execute(
        _OBJECTS_SQL + where_sql
        + " ORDER BY LOWER(sat.name) ASC, sat.norad_id ASC LIMIT ? OFFSET ?",
        (*params, limit, offset),
    ).fetchall()
    return {"count": count, "items": [_item_from_row(row) for row in rows]}


def _track_payload(line1, line2, epoch: datetime, n: int) -> dict:
    """Ground-track payload: one orbital period of subpoints from `epoch`.

    TEME positions from real SGP4 (propagate.positions_at); Earth rotation
    applied via IAU-82 GMST (propagate.gmst_rad). Degrades to the nulls
    shape ({"samples": [], ...}) when the TLE is absent, unparseable, or
    fails to propagate - mirroring _derive_elements.
    """
    payload = {
        "epoch": epoch.strftime(ISO_FMT),
        "period_min": None,
        "samples": [],
        "subpoint": dict(_NULL_SUBPOINT),
    }
    if not (line1 and line2):
        return payload
    try:
        from backend.app.ingestion import propagate

        period_min = propagate.TWO_PI / propagate.satrec_from_tle(line1, line2).no_kozai
        times = [epoch + timedelta(minutes=period_min * i / (n - 1)) for i in range(n)]
        r, _v = propagate.positions_at(line1, line2, times)
        track = []
        for t, (x, y, z) in zip(times, r):
            x, y, z = float(x), float(y), float(z)
            rmag = math.sqrt(x * x + y * y + z * z)
            lon = math.degrees(math.atan2(y, x) - propagate.gmst_rad(t))
            lon = ((lon + 180.0) % 360.0) - 180.0
            lat = math.degrees(math.asin(z / rmag))
            alt = rmag - propagate.RE_KM
            if not all(map(math.isfinite, (lon, lat, alt))):
                raise ValueError("non-finite ground-track sample")
            if alt < TRACK_MIN_ALT_KM:
                # SGP4 still returns states after effective re-entry; a track
                # dipping below ~100 km contradicts the TLE-derived elements
                # shown beside it. Degrade to the empty shape instead.
                raise ValueError("ground-track sample below re-entry altitude")
            track.append([round(lon, 2), round(lat, 2), round(alt, 1)])
        sub = propagate.subpoint(line1, line2, epoch)
    except Exception:
        return payload
    payload["period_min"] = round(float(period_min), 2)
    payload["samples"] = track
    payload["subpoint"] = {
        "lat_deg": round(sub["lat_deg"], 2),
        "lon_deg": round(sub["lon_deg"], 2),
        "alt_km": round(sub["alt_km"], 1),
        "velocity_kms": round(sub["velocity_kms"], 2),
    }
    return payload


@router.get("/objects/{norad_id}/track")
def get_track(
    norad_id: int = Path(ge=0, le=INT64_MAX),
    samples: int = 240,
    conn=Depends(db.get_db),
):
    """One-orbital-period ground track from now (whole second), latest TLE."""
    row = conn.execute(
        _OBJECTS_SQL + " WHERE sat.norad_id = ?", (norad_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Object {norad_id} not found")
    n = max(TRACK_SAMPLES_MIN, min(TRACK_SAMPLES_MAX, samples))
    epoch = datetime.now(timezone.utc).replace(microsecond=0)
    return _track_payload(row["line1"], row["line2"], epoch, n)


@router.get("/objects/{norad_id}/passes")
def get_passes(
    norad_id: int = Path(ge=0, le=INT64_MAX),
    lat: float = Query(ge=-90.0, le=90.0),
    lon: float = Query(ge=-180.0, le=180.0),
    alt_m: float = Query(0.0, ge=-500.0, le=10000.0),
    hours: int = Query(48, ge=1, le=168),
    min_el: float = Query(10.0, ge=0.0, le=90.0),
    conn=Depends(db.get_db),
):
    """Ground-station passes over the next `hours` from now (whole second).

    Topocentric AOS/LOS/culmination via propagate.passes on the latest TLE.
    Never 500s on a bad TLE: degrades to {"passes": [], "note": <reason>};
    `note` is null on success (an empty list then simply means no passes).
    """
    row = conn.execute(
        _OBJECTS_SQL + " WHERE sat.norad_id = ?", (norad_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Object {norad_id} not found")
    start = datetime.now(timezone.utc).replace(microsecond=0)
    payload = {
        "site": {
            "lat_deg": lat,
            "lon_deg": lon,
            "alt_m": alt_m,
            "min_el_deg": min_el,
            "hours": hours,
            "start": start.strftime(ISO_FMT),
        },
        "passes": [],
        "note": None,
    }
    if not (row["line1"] and row["line2"]):
        payload["note"] = "No TLE available for this object"
        return payload
    try:
        from backend.app.ingestion import propagate

        payload["passes"] = propagate.passes(
            row["line1"], row["line2"], lat, lon, alt_m, start, hours, min_el
        )
    except Exception:
        payload["note"] = "Propagation failed for the latest TLE"
    return payload


@router.get("/objects/{norad_id}")
def get_object(norad_id: int = Path(ge=0, le=INT64_MAX), conn=Depends(db.get_db)):
    row = conn.execute(
        _OBJECTS_SQL + " WHERE sat.norad_id = ?", (norad_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Object {norad_id} not found")

    item = _item_from_row(row)
    item["tle_history"] = [
        dict(tle)
        for tle in conn.execute(
            "SELECT epoch, line1, line2, ingested_at FROM tle_history"
            " WHERE satellite_id = ? ORDER BY epoch DESC, id DESC LIMIT 20",
            (row["id"],),
        )
    ]
    item["orbit_path"] = _orbit_path(row["line1"], row["line2"])

    conj = conjunctions_api.query_summaries(
        conn,
        "(c.primary_object_id = ? OR c.secondary_object_id = ?)",
        (row["id"], row["id"]),
    )
    conj.sort(
        key=lambda i: (
            0 if i["status"] == "active" else 1,
            *conjunctions_api.sort_key(i),
        )
    )
    # True total before the cap - additive field, existing keys unchanged.
    item["conjunction_count"] = len(conj)
    conj = conj[:CONJUNCTIONS_MAX]
    conjunctions_api.attach_pc_trends(conn, conj)
    conjunctions_api.attach_space_weather(conn, conj)
    item["conjunctions"] = conj
    # what the current storm does to this object's own orbit (null without
    # live space-weather data) - app.spaceweather.object_effects
    try:
        from backend.app import spaceweather as sw
        item["space_weather"] = sw.object_effects(row["line1"], row["line2"])
    except Exception:
        item["space_weather"] = None
    return item
