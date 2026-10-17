"""User-fleet endpoints: CRUD over user_objects plus on-demand screen and scan.

User objects are operator-supplied TLEs stored in the ``user_objects`` table -
they are not catalog satellites. Screening a user object runs the real
coarse+refine pipeline (ingestion.screening.screen) with the stored TLE
injected under a synthetic norad (900000 + id) as the sole primary; results
are computed on demand and returned, never written to conjunctions /
cdm_messages, so user objects can never pollute the real event tables.
The launch-window scan (POST /{id}/window-scan) runs the same coarse capture
machinery over a multi-day horizon with no refinement, bucketing coarse
minima by UTC day - see _coarse_window_scan.

TLE validation on POST is strict (the pure-Python sgp4 parser enforces the
fixed-column format the C-accelerated parser silently ignores) plus one test
propagation at the TLE's own epoch; 422 details always name the failing line.

`ingestion.*` / `models.*` are imported lazily inside helpers, matching
api/objects.py, so a missing module can never break import order.
"""
import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from pydantic import BaseModel, Field
from sgp4 import io as sgp4_io
from sgp4.api import SGP4_ERRORS, Satrec, SatrecArray
from sgp4.earth_gravity import wgs72

from backend.app import config, db
from backend.app.api import conjunctions as conjunctions_api

router = APIRouter(prefix="/user-objects")

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

# SQLite INTEGER is int64; range-check every int bound into SQL (422, not 500).
INT64_MAX = 9223372036854775807

# Synthetic norad namespace for injected user objects (contract: 900000 + id).
# Real catalog ids top out well below this (alpha-5 encodes at most 339999).
SYNTHETIC_NORAD_BASE = 900000

# on-demand screen window clamp, hours (default 72 = config screening window)
WINDOW_HOURS_MIN = 1.0
WINDOW_HOURS_MAX = 168.0

# Fleet size cap: POST answers 413 once user_objects already holds this many.
USER_OBJECTS_MAX = 500

# Element keys exposed on the user-object item (subset of orbital_elements).
_ELEMENT_KEYS = ("epoch", "inclination_deg", "apogee_km", "perigee_km",
                 "period_min")


class UserObjectIn(BaseModel):
    name: str | None = Field(default=None, max_length=64)
    line1: str
    line2: str
    note: str | None = Field(default=None, max_length=500)


def _failing_line(line1: str) -> str:
    """Attribute a bare numeric-field parse error to a line.

    The strict parser's column check can pass while a field still fails float
    conversion (raised without naming a line); re-running line 1's numeric
    fields decides whether line 1 or line 2 was the culprit.
    """
    line = line1.rstrip()
    try:
        float(line[20:32])
        float(line[33:43])
        float(line[44] + "." + line[45:50])
        int(line[50:52])
        float(line[53] + "." + line[54:59])
        int(line[59:61])
        int(line[64:68])
        return "line 2"
    except (ValueError, IndexError):
        return "line 1"


def _validate_tle(line1: str, line2: str) -> Satrec:
    """Strict TLE validation -> C-accelerated Satrec; 422 names the failing line.

    Three gates:
    1. leader check ('1 ' / '2 ') - catches swapped or non-TLE lines crisply;
    2. the pure-Python sgp4 parser (fixed-column format, norad agreement,
       element initialization) - the C parser accepts almost anything, so the
       strict one does the rejecting;
    3. one test propagation at the TLE's own epoch - elements that parse but
       cannot produce a finite state (decayed/suborbital) are rejected too.
    """
    for label, leader, line in (("line 1", "1", line1), ("line 2", "2", line2)):
        if not line.startswith(leader + " "):
            raise HTTPException(
                status_code=422,
                detail=f"TLE {label} invalid: must start with '{leader} '")
    try:
        sgp4_io.twoline2rv(line1, line2, wgs72)
    except ValueError as exc:
        msg = str(exc)
        if "do not match" in msg:
            raise HTTPException(
                status_code=422,
                detail="TLE line 2 invalid: NORAD id does not match line 1")
        if "non-ASCII" in msg:
            which = "line 1" if not line1.isascii() else "line 2"
        elif "for line 1" in msg:
            which = "line 1"
        elif "for line 2" in msg:
            which = "line 2"
        else:
            which = _failing_line(line1)
        raise HTTPException(
            status_code=422,
            detail=f"TLE {which} invalid: does not match the fixed-column"
                   " TLE format")
    except Exception:
        # e.g. ZeroDivisionError from sgp4init on zero mean motion
        raise HTTPException(
            status_code=422,
            detail="TLE line 2 invalid: orbital elements fail sgp4"
                   " initialization")

    sat = Satrec.twoline2rv(line1, line2)
    if sat.no_kozai <= 0:
        raise HTTPException(
            status_code=422,
            detail="TLE line 2 invalid: non-positive mean motion")
    error, r, _v = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
    if error != 0 or not all(math.isfinite(c) for c in r):
        raise HTTPException(
            status_code=422,
            detail="TLE line 2 invalid: test propagation at epoch failed"
                   f" (sgp4 error {error}:"
                   f" {SGP4_ERRORS.get(error, 'unknown error')})")
    return sat


def _tle_meta(line1, line2) -> dict:
    """norad + derived elements from a stored TLE; nulls if derivation fails."""
    meta = {"norad_id": None, **{key: None for key in _ELEMENT_KEYS}}
    try:
        from backend.app.ingestion import propagate

        meta["norad_id"] = propagate.satrec_from_tle(line1, line2).satnum
        elements = propagate.orbital_elements(line1, line2)
        for key in _ELEMENT_KEYS:
            meta[key] = elements.get(key)
    except Exception:
        pass
    return meta


def _item(row) -> dict:
    meta = _tle_meta(row["line1"], row["line2"])
    return {
        "id": row["id"],
        "name": row["name"],
        "norad_id": meta["norad_id"],
        "note": row["note"],
        "added_at": row["added_at"],
        "epoch": meta["epoch"],
        "inclination_deg": meta["inclination_deg"],
        "apogee_km": meta["apogee_km"],
        "perigee_km": meta["perigee_km"],
        "period_min": meta["period_min"],
    }


@router.get("")
def list_user_objects(conn=Depends(db.get_db)):
    rows = conn.execute(
        "SELECT id, name, line1, line2, note, added_at FROM user_objects"
        " ORDER BY id ASC").fetchall()
    return {"items": [_item(row) for row in rows]}


def _fleet_count(conn) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS n FROM user_objects").fetchone()["n"]


@router.post("", status_code=201)
def create_user_object(body: UserObjectIn, conn=Depends(db.get_db)):
    if _fleet_count(conn) >= USER_OBJECTS_MAX:
        raise HTTPException(
            status_code=413,
            detail=f"user-object fleet is full ({USER_OBJECTS_MAX} objects)")
    line1 = body.line1.strip()
    line2 = body.line2.strip()
    sat = _validate_tle(line1, line2)
    name = (body.name or "").strip() or f"USER OBJECT {sat.satnum}"
    note = body.note.strip() if body.note and body.note.strip() else None
    added_at = datetime.now(timezone.utc).strftime(ISO_FMT)
    cur = conn.execute(
        "INSERT INTO user_objects (name, line1, line2, note, added_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (name, line1, line2, note, added_at))
    conn.commit()
    return _item({"id": cur.lastrowid, "name": name, "line1": line1,
                  "line2": line2, "note": note, "added_at": added_at})


@router.delete("/{user_object_id}", status_code=204)
def delete_user_object(
    user_object_id: int = Path(ge=0, le=INT64_MAX),
    conn=Depends(db.get_db),
):
    cur = conn.execute(
        "DELETE FROM user_objects WHERE id = ?", (user_object_id,))
    if cur.rowcount == 0:
        raise HTTPException(
            status_code=404, detail=f"User object {user_object_id} not found")
    conn.commit()
    return Response(status_code=204)


def _age_days(tca: datetime, epoch: datetime | None) -> float:
    if epoch is None:
        return 0.0
    return max((tca - epoch).total_seconds(), 0.0) / 86400.0


def _event_pc(miss_distance_m: float, relative_velocity_ms: float,
              age_p_days: float, age_s_days: float) -> float | None:
    """Assumed-sigma Pc via models.pc with both TLEs' propagation ages.

    Each TLE contributes sigma_at(its own age to TCA); the combined 1-sigma is
    their RSS - the same assumption as the ephemeris endpoint's sigma_m.
    models.pc.compute_pc integrates sigma = sqrt(2)*sigma_at(age) (one shared
    age), and sigma_at is linear in age, so the RSS maps back to an exact
    equivalent age: compute_pc then applies precisely the combined covariance.
    None when models.pc is unavailable (lazy import, like screening's bridge).
    """
    try:
        from backend.app.models import pc as pc_mod

        combined = math.sqrt(pc_mod.sigma_at(age_p_days) ** 2
                             + pc_mod.sigma_at(age_s_days) ** 2)
        per_object = combined / math.sqrt(2.0)
        growth = config.SIGMA_GROWTH_M_PER_DAY
        age_eq = ((per_object - config.SIGMA_BASE_M) / growth) if growth > 0 \
            else 0.0
        return pc_mod.compute_pc(miss_distance_m, relative_velocity_ms,
                                 propagation_age_days=max(age_eq, 0.0))
    except Exception:
        return None


def _parse_iso_utc(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")) \
            .astimezone(timezone.utc)
    except Exception:
        return None


def _secondary_meta(conn, norad_ids: list[int]) -> dict[int, dict]:
    """{norad_id: {name, object_type, epoch}} for event secondaries."""
    if not norad_ids:
        return {}
    placeholders = ",".join("?" * len(norad_ids))
    rows = conn.execute(
        f"""
        SELECT s.norad_id AS norad_id, s.name AS name,
               s.object_type AS object_type, t.epoch AS epoch
        FROM satellites s
        LEFT JOIN tle_history t ON t.id = (
            SELECT t2.id FROM tle_history t2
             WHERE t2.satellite_id = s.id
             ORDER BY t2.epoch DESC, t2.id DESC LIMIT 1)
        WHERE s.norad_id IN ({placeholders})
        """,
        norad_ids).fetchall()
    return {row["norad_id"]: dict(row) for row in rows}


@router.post("/{user_object_id}/screen")
def screen_user_object(
    user_object_id: int = Path(ge=0, le=INT64_MAX),
    window_hours: float = Query(72.0),
    conn=Depends(db.get_db),
):
    """On-demand screen of one user object against the live catalog.

    The stored TLE is injected into screening.screen under a synthetic norad
    (900000 + id) as the sole primary - the real coarse+refine pipeline runs,
    but results are returned directly and nothing is written: no satellites
    row for the synthetic norad, no conjunctions, no CDMs. window_hours is
    clamped to 1..168; events are sorted by Pc descending. Pc uses the
    assumed-sigma path with both TLEs' propagation ages at each event's TCA
    (see _event_pc); risk_level applies the standard config thresholds.
    """
    row = conn.execute(
        "SELECT id, line1, line2 FROM user_objects WHERE id = ?",
        (user_object_id,)).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"User object {user_object_id} not found")
    window = max(WINDOW_HOURS_MIN, min(WINDOW_HOURS_MAX, window_hours))
    synthetic_norad = SYNTHETIC_NORAD_BASE + row["id"]

    from backend.app.ingestion import celestrak, screening

    events = screening.screen(
        conn, {synthetic_norad}, window_hours=window,
        extra=[{"norad_id": synthetic_norad,
                "line1": row["line1"], "line2": row["line2"]}])
    screened_at = datetime.now(timezone.utc).strftime(ISO_FMT)

    try:
        user_epoch = celestrak.tle_epoch(row["line1"])
    except Exception:
        user_epoch = None
    meta = _secondary_meta(conn, [ev.secondary_norad for ev in events])

    items = []
    for ev in events:
        sec = meta.get(ev.secondary_norad) or {
            "name": f"OBJECT {ev.secondary_norad}", "object_type": "UNKNOWN",
            "epoch": None}
        sec_epoch = _parse_iso_utc(sec["epoch"]) if sec["epoch"] else None
        pc = _event_pc(ev.miss_distance_m, ev.relative_velocity_ms,
                       _age_days(ev.tca, user_epoch),
                       _age_days(ev.tca, sec_epoch))
        items.append({
            "secondary": {
                "norad_id": ev.secondary_norad,
                "name": sec["name"],
                "object_type": sec["object_type"],
            },
            "tca": ev.tca.strftime(ISO_FMT),
            "miss_distance_m": round(ev.miss_distance_m, 1),
            "relative_velocity_ms": round(ev.relative_velocity_ms, 1),
            "pc": pc,
            "risk_level": conjunctions_api.risk_level_for(
                pc if pc is not None else 0.0),
        })
    items.sort(key=lambda item:
               (-(item["pc"] if item["pc"] is not None else 0.0),
                item["miss_distance_m"]))
    return {"screened_at": screened_at, "window_hours": window,
            "events": items}


# --- Launch-window scan --------------------------------------------------------

WINDOW_SCAN_STEP_S = 120.0      # coarse grid step -> method label 'coarse-120s'
WINDOW_SCAN_DAYS_DEFAULT = 14   # body 'days' validated to 3..14 (422 otherwise)
_SCAN_CHUNK_STEPS = 240         # coarse steps per SatrecArray call (~8 h chunks)


class WindowScanIn(BaseModel):
    days: int = Field(default=WINDOW_SCAN_DAYS_DEFAULT, ge=3, le=14)


def _coarse_window_scan(conn, line1: str, line2: str, days: int) -> list[dict]:
    """Coarse-only screen of one TLE against the LEO catalog, bucketed by UTC day.

    Reuses screening.screen's coarse machinery: the same catalog load (latest
    TLE per satellite, apogee <= config.SCREENING_MAX_LEO_ALT_KM), the same
    shared SatrecArray time grid (chunked along time), and the same two-part
    capture gate described in screening's docstring -

      1. sampled separation under config.SCREENING_THRESHOLD_KM, and
      2. the rectilinear minimum over each segment between consecutive samples
         (min_tau |dr + dv*tau|, closed form from the segment-start relative
         state) under a capture radius of threshold +
         screening.COARSE_CAPTURE_MARGIN_KM

    - but with no per-candidate refinement. The reported metric is the coarse
    minimum itself (method 'coarse-120s'): screening-grade only, and a gate-2
    capture may exceed the reporting threshold by up to the capture margin.

    Candidates are bucketed by the UTC calendar day of their coarse-minimum
    time, counting DISTINCT (user object, secondary) pairs per bucket; a
    pair's deepest coarse approach per day carries its miss/Pc. Coarse-grade
    analog of screening's co-moving exclusion: candidates whose sampled
    relative speed at the coarse minimum is under
    config.SCREENING_MIN_REL_VEL_MS are skipped, and a same-norad catalog
    copy of the user object is excluded from the secondary set up front
    (mirrors screen()'s extra-injection replacement). Documented deviation
    from screen(): objects that fail to propagate are skipped per-sample, not
    dropped for the whole run - over a 14-day horizon an object decaying on
    day 10 still screens on days 1..9.

    Per-event Pc is the assumed-sigma path with both TLEs' propagation ages at
    the coarse-minimum time (same combination as the on-demand screen's
    _event_pc). Returns one bucket per UTC date intersecting [now, now+days]
    - days+1 entries whenever the window straddles a final partial day (i.e.
    always, except a scan started exactly at 00:00:00Z); zero-event days
    carry null min_miss_km/max_pc. Read-only: nothing is written to the DB.
    """
    from backend.app.ingestion import screening
    from backend.app.ingestion.propagate import jday_arrays

    step_s = WINDOW_SCAN_STEP_S
    threshold_km = config.SCREENING_THRESHOLD_KM
    capture_km = threshold_km + screening.COARSE_CAPTURE_MARGIN_KM
    thr2 = threshold_km * threshold_km
    cap2 = capture_km * capture_km
    min_rel_vel_ms = config.SCREENING_MIN_REL_VEL_MS

    start = datetime.now(timezone.utc).replace(microsecond=0)
    n_steps = int(round(days * 86400.0 / step_s)) + 1
    end = start + timedelta(seconds=step_s * (n_steps - 1))
    dates = [start.date() + timedelta(days=k)
             for k in range((end.date() - start.date()).days + 1)]
    buckets = {d: {"date": d.isoformat(), "events": 0,
                   "min_miss_km": None, "max_pc": None} for d in dates}

    try:
        user_sat = Satrec.twoline2rv(line1, line2)
    except Exception:
        user_sat = None
    if user_sat is None or user_sat.no_kozai <= 0:
        return list(buckets.values())    # stored TLE degraded: empty scan

    sats: list[Satrec] = []
    norads: list[int] = []
    for row in screening._latest_tles(conn):
        if row["norad_id"] == user_sat.satnum:
            continue     # the user's own catalog copy is not a partner
        try:
            sat = Satrec.twoline2rv(row["line1"], row["line2"])
        except Exception:
            continue
        if sat.no_kozai <= 0 or \
                screening._apogee_km(sat) > config.SCREENING_MAX_LEO_ALT_KM:
            continue
        sats.append(sat)
        norads.append(row["norad_id"])
    if not sats:
        return list(buckets.values())

    times = [start + timedelta(seconds=step_s * i) for i in range(n_steps)]
    jd, fr = jday_arrays(times)
    jd_abs = jd + fr                       # absolute JD per grid step (ages)
    arr = SatrecArray(sats)
    user_epoch_jd = user_sat.jdsatepoch + user_sat.jdsatepochF
    sat_epoch_jd = [s.jdsatepoch + s.jdsatepochF for s in sats]

    # (secondary_norad, utc_date) -> [metric2_km2, rel_speed_ms, age_u, age_s]
    best: dict[tuple, list] = {}

    def note(si: int, metric2: float, vv_km2_s2: float, t_jd: float,
             t_date) -> None:
        rel_ms = math.sqrt(max(float(vv_km2_s2), 0.0)) * 1000.0
        if rel_ms < min_rel_vel_ms:
            return   # co-moving at the coarse minimum - not a conjunction
        key = (norads[si], t_date)
        metric2 = max(float(metric2), 0.0)
        cur = best.get(key)
        if cur is None or metric2 < cur[0]:
            best[key] = [metric2, rel_ms,
                         max(t_jd - user_epoch_jd, 0.0),
                         max(t_jd - sat_epoch_jd[si], 0.0)]

    prev = None    # (d2, rv, vv) at the previous chunk's last grid step
    for c0 in range(0, n_steps, _SCAN_CHUNK_STEPS):
        jd_c = jd[c0:c0 + _SCAN_CHUNK_STEPS]
        fr_c = fr[c0:c0 + _SCAN_CHUNK_STEPS]
        eu, ru, vu = user_sat.sgp4_array(jd_c, fr_c)
        e, r, v = arr.sgp4(jd_c, fr_c)
        ubad = eu != 0                     # bad samples -> NaN; NaN separations
        ru[ubad] = np.nan                  # compare False and fall out of both
        vu[ubad] = np.nan                  # gates (per-sample skip, see docstring)
        bad = e != 0
        r[bad] = np.nan
        v[bad] = np.nan
        dr = r - ru[None, :, :]
        dv = v - vu[None, :, :]
        d2 = np.einsum("stk,stk->st", dr, dr)      # km^2     (nsat, T)
        rv = np.einsum("stk,stk->st", dr, dv)      # km^2/s
        vv = np.einsum("stk,stk->st", dv, dv)      # km^2/s^2

        # Gate 1: sampled separation already under the reporting threshold.
        for si, t in zip(*np.nonzero(d2 < thr2)):
            gi = c0 + int(t)
            note(int(si), d2[si, t], vv[si, t], jd_abs[gi], times[gi].date())

        # Gate 2: rectilinear segment minimum dips under the capture radius.
        # Segment starts are this chunk's steps 0..T-2, seeded with the
        # previous chunk's last step for the chunk-boundary segment.
        if prev is not None:
            d2s = np.concatenate([prev[0][:, None], d2[:, :-1]], axis=1)
            rvs = np.concatenate([prev[1][:, None], rv[:, :-1]], axis=1)
            vvs = np.concatenate([prev[2][:, None], vv[:, :-1]], axis=1)
            g0 = c0 - 1                # segment j starts at grid index g0 + j
        else:
            d2s, rvs, vvs = d2[:, :-1], rv[:, :-1], vv[:, :-1]
            g0 = 0
        with np.errstate(invalid="ignore", divide="ignore"):
            tau = -rvs / vvs
            dmin2 = d2s - rvs * rvs / vvs
            interior = (tau > 0.0) & (tau < step_s) & (dmin2 < cap2)
        for si, j in zip(*np.nonzero(interior)):
            gi = g0 + int(j)
            t_off = float(tau[si, j])
            t_min = times[gi] + timedelta(seconds=t_off)
            note(int(si), dmin2[si, j], vvs[si, j],
                 jd_abs[gi] + t_off / 86400.0, t_min.date())
        prev = (d2[:, -1].copy(), rv[:, -1].copy(), vv[:, -1].copy())

    for (_sec_norad, day), (metric2, rel_ms, age_u, age_s) in best.items():
        bucket = buckets.get(day)
        if bucket is None:
            continue   # segment tau nudged a boundary minimum past the window
        miss_km = math.sqrt(metric2)
        pc = _event_pc(miss_km * 1000.0, rel_ms, age_u, age_s)
        bucket["events"] += 1
        if bucket["min_miss_km"] is None or miss_km < bucket["min_miss_km"]:
            bucket["min_miss_km"] = miss_km
        if pc is not None and (bucket["max_pc"] is None
                               or pc > bucket["max_pc"]):
            bucket["max_pc"] = pc
    for bucket in buckets.values():
        if bucket["min_miss_km"] is not None:
            bucket["min_miss_km"] = round(bucket["min_miss_km"], 3)
    return list(buckets.values())


@router.post("/{user_object_id}/window-scan")
def window_scan_user_object(
    user_object_id: int = Path(ge=0, le=INT64_MAX),
    body: WindowScanIn | None = None,
    conn=Depends(db.get_db),
):
    """Coarse launch-window scan of one user object across now -> +days.

    Screening-grade only: 120 s coarse grid, capture-gated exactly like
    screening.screen's coarse pass, no per-candidate refinement - per-day
    misses are coarse metrics, not refined TCA misses (method 'coarse-120s').
    ``days`` is validated to 3..14 (422 otherwise; default 14); buckets are
    UTC calendar days with distinct (user object, secondary) pairs counted
    once per day. Nothing is stored - results are computed on demand.
    Runtime scales with catalog size x days: of order a minute against the
    full catalog (duration_s reports the measured wall time).
    """
    row = conn.execute(
        "SELECT id, line1, line2 FROM user_objects WHERE id = ?",
        (user_object_id,)).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"User object {user_object_id} not found")
    days = body.days if body is not None else WINDOW_SCAN_DAYS_DEFAULT
    t0 = time.monotonic()
    day_buckets = _coarse_window_scan(conn, row["line1"], row["line2"], days)
    return {"days": day_buckets, "method": "coarse-120s",
            "duration_s": round(time.monotonic() - t0, 2)}


# --- Add from catalog -----------------------------------------------------------

class FromCatalogIn(BaseModel):
    norad_id: int = Field(ge=0, le=INT64_MAX)
    name: str | None = Field(default=None, max_length=64)
    note: str | None = Field(default=None, max_length=500)


@router.post("/from-catalog", status_code=201)
def create_user_object_from_catalog(
    body: FromCatalogIn,
    response: Response,
    conn=Depends(db.get_db),
):
    """Copy a tracked object's latest catalog TLE into a new user object.

    Read-only on the catalog: satellites / tle_history are only SELECTed; the
    copy lands in user_objects exactly as POST /user-objects would store it
    (same strict validation, same item shape) plus a ``created`` flag.
    Idempotent per element set: when a user object already holds this exact
    TLE pair it is returned with 200 and created=false instead of adding a
    duplicate, so a demo re-run never fills the fleet. 404 when the norad id
    is not in the catalog, 422 when it has no TLE or its latest TLE fails
    validation, 413 when the fleet is full.
    """
    row = conn.execute(
        """
        SELECT s.name AS name, t.line1 AS line1, t.line2 AS line2
        FROM satellites s
        LEFT JOIN tle_history t ON t.id = (
            SELECT t2.id FROM tle_history t2
             WHERE t2.satellite_id = s.id
             ORDER BY t2.epoch DESC, t2.id DESC LIMIT 1)
        WHERE s.norad_id = ?
        """,
        (body.norad_id,)).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Object {body.norad_id} is not in the tracked catalog")
    if not (row["line1"] and row["line2"]):
        raise HTTPException(
            status_code=422,
            detail=f"Object {body.norad_id} has no TLE in the catalog yet")
    line1 = row["line1"].strip()
    line2 = row["line2"].strip()

    existing = conn.execute(
        "SELECT id, name, line1, line2, note, added_at FROM user_objects"
        " WHERE line1 = ? AND line2 = ? ORDER BY id ASC LIMIT 1",
        (line1, line2)).fetchone()
    if existing is not None:
        response.status_code = 200
        return {**_item(existing), "created": False}

    # Same satellite, older element set (the catalog refreshed since it was
    # added): refresh that user object in place instead of adding a twin.
    _validate_tle(line1, line2)
    same_sat = conn.execute(
        "SELECT id, name, line1, line2, note, added_at FROM user_objects"
        " WHERE SUBSTR(line1, 3, 5) = ? ORDER BY id ASC LIMIT 1",
        (line1[2:7],)).fetchone()
    if same_sat is not None:
        conn.execute("UPDATE user_objects SET line1 = ?, line2 = ? WHERE id = ?",
                     (line1, line2, same_sat["id"]))
        conn.commit()
        response.status_code = 200
        return {**_item({**dict(same_sat), "line1": line1, "line2": line2}),
                "created": False, "updated": True}

    if _fleet_count(conn) >= USER_OBJECTS_MAX:
        raise HTTPException(
            status_code=413,
            detail=f"user-object fleet is full ({USER_OBJECTS_MAX} objects)")
    _validate_tle(line1, line2)
    name = ((body.name or "").strip() or (row["name"] or "").strip()
            or f"USER OBJECT {body.norad_id}")[:64]
    note = body.note.strip() if body.note and body.note.strip() else None
    added_at = datetime.now(timezone.utc).strftime(ISO_FMT)
    cur = conn.execute(
        "INSERT INTO user_objects (name, line1, line2, note, added_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (name, line1, line2, note, added_at))
    conn.commit()
    return {**_item({"id": cur.lastrowid, "name": name, "line1": line1,
                     "line2": line2, "note": note, "added_at": added_at}),
            "created": True}
