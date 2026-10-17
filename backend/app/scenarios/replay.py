"""Replay Kessler's live screening pipeline at a historical epoch.

Nothing here re-implements the physics. Each replayed run calls the same
functions the live refresh uses:

  - ingestion.screening.screen    coarse grid + segment gate + 1 s refinement
                                  (with start= pinned to the run time and the
                                  scenario's element sets as the catalog)
  - ingestion.screening._compute_pc  -> models.pc.compute_pc, with the
                                  propagation age record_events uses
                                  (TCA minus run time)
  - api.conjunctions.risk_level_for  the single risk-level definition

Runs are replayed on the scenario's cadence (hourly by default). A run at
time T only sees element sets whose epoch is <= T - the catalog as it stood
then. The first run that reports the pair is the moment Kessler would first
have flagged it; the lead time is TCA minus that run time.

Ephemeris reuses the Encounter Simulator helpers in api.conjunctions
(0.05 s TCA re-refinement, RIC tca_state, 1 s grid) so the payload has the
exact /conjunctions/{id}/ephemeris shape.
"""
import math
import threading
from datetime import datetime, timedelta, timezone

from backend.app import config
from backend.app.ingestion import propagate, screening
from backend.app.ingestion.celestrak import ISO_FMT
from backend.app import scenarios

PC_METHOD = ("models.pc.compute_pc: isotropic 2-D encounter-plane Pc with an"
             " assumed TLE-class sigma growing with propagation age"
             " (the live screening Pc)")
HISTORY_OFFSETS_H = (48.0, 24.0, 12.0, 6.0, 3.0, 1.0)

_cache: dict = {}
_lock = threading.Lock()


def _parse(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime(ISO_FMT)


def _iso_ms(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _catalog_at(scn: dict, run_at: datetime) -> list[dict]:
    rows = []
    for obj in scn["objects"]:
        rec = scenarios.elset_at(obj, _iso(run_at))
        if rec is not None:
            rows.append({"norad_id": obj["norad_id"],
                         "line1": rec["line1"], "line2": rec["line2"]})
    return rows


def run_screen(scn: dict, run_at: datetime) -> dict:
    """One live-code screening run at `run_at` against the scenario pair."""
    p = scenarios.primary(scn)
    s = scenarios.secondary(scn)
    catalog = _catalog_at(scn, run_at)
    window_h = float(scn.get("screening_window", {}).get(
        "hours", config.SCREENING_WINDOW_HOURS))
    event = None
    if len(catalog) == 2:
        events = screening.screen(None, {p["norad_id"]}, window_hours=window_h,
                                  start=run_at, catalog=catalog)
        event = next((ev for ev in events
                      if {ev.primary_norad, ev.secondary_norad}
                      == {p["norad_id"], s["norad_id"]}), None)
    out = {
        "run_at": _iso(run_at),
        "elsets_available": len(catalog),
        "detected": event is not None,
    }
    if event is not None:
        tca = event.tca
        age_days = max((tca - run_at).total_seconds() / 3600.0, 0.0) / 24.0
        pc = screening._compute_pc(event.miss_distance_m,
                                   event.relative_velocity_ms, age_days)
        from backend.app.api.conjunctions import risk_level_for

        out.update({
            "tca": _iso_ms(tca),
            "hours_before_tca": round((tca - run_at).total_seconds() / 3600.0, 2),
            "miss_distance_m": round(event.miss_distance_m, 1),
            "relative_velocity_ms": round(event.relative_velocity_ms, 1),
            "pc": pc,
            "risk_level": risk_level_for(pc or 0.0),
        })
    return out


def _subpoint(line1: str, line2: str, when: datetime) -> dict | None:
    """The event location for the page: longitude and speed from
    propagate.subpoint, latitude and altitude on the WGS-84 ellipsoid (how
    published accounts quote an event's altitude - see geodetic_lat_alt).
    Object pages keep the spherical convention of their perigee/apogee."""
    try:
        sp = propagate.subpoint(line1, line2, when)
        r, _ = propagate.positions_at(line1, line2, [when])
    except ValueError:
        return None
    sp["lat_deg"], sp["alt_km"] = geodetic_lat_alt([float(c) for c in r[0]])
    return {k: round(v, 3) for k, v in sp.items()}


def _cache_key(scn: dict) -> tuple:
    return (scn["id"], scn["_file"],
            tuple(r["line1"] + r["line2"] for o in scn["objects"] for r in o["elsets"]))


def screening_replay(scn: dict) -> dict:
    """Detection timeline for the scenario (cached per scenario content).

    first_detection: the earliest cadence run (hourly, from the first hour at
    which both objects have a published element set) that reports the pair.
    history: that run plus runs at fixed lead times (48, 24, 12, 6, 3, 1 h)
    - how miss distance and Pc evolve as the encounter approaches.
    """
    key = _cache_key(scn)
    # Held across the (once per process, ~2 s) computation so concurrent
    # page requests wait for one result instead of each recomputing it.
    with _lock:
        if key not in _cache:
            _cache[key] = _compute_replay(scn)
        return _cache[key]


def _compute_replay(scn: dict) -> dict:
    win = scn.get("screening_window", {})
    window_h = float(win.get("hours", config.SCREENING_WINDOW_HOURS))
    cadence_h = float(win.get("cadence_hours", 1.0))
    event_time = _parse(scn["event"]["time_utc"])

    first_epochs = [_parse(o["elsets"][0]["epoch"]) for o in scn["objects"]]
    t0 = max(first_epochs)
    # First whole-hour run once both objects are in the catalog, but no
    # earlier than one window before the event (earlier runs cannot see it).
    t0 = max(t0, event_time - timedelta(hours=window_h + cadence_h))
    t0 = t0.replace(minute=0, second=0, microsecond=0)
    if t0 < max(first_epochs):
        t0 += timedelta(hours=1)

    first = None
    misses = 0
    t = t0
    step = timedelta(hours=cadence_h)
    while t < event_time:
        run = run_screen(scn, t)
        if run["detected"]:
            first = run
            break
        misses += 1
        t += step

    history = []
    if first is not None:
        history.append(first)
        tca = _parse(first["tca"])
        for off in HISTORY_OFFSETS_H:
            run_at = (tca - timedelta(hours=off)).replace(second=0, microsecond=0)
            if run_at <= _parse(first["run_at"]):
                continue
            history.append(run_screen(scn, run_at))

    p = scenarios.primary(scn)
    s = scenarios.secondary(scn)
    location = None
    if first is not None:
        rec = scenarios.elset_at(p, first["run_at"])
        location = _subpoint(rec["line1"], rec["line2"], _parse(first["tca"]))

    result = {
        "method": {
            "screening": ("ingestion.screening.screen — the live coarse grid,"
                          " segment gate and 1 s refinement"),
            "pc": PC_METHOD,
            "window_hours": window_h,
            "threshold_km": config.SCREENING_THRESHOLD_KM,
            "coarse_step_s": config.SCREENING_COARSE_STEP_S,
            "cadence_hours": cadence_h,
            "risk_thresholds": {"elevated": config.PC_ELEVATED,
                                "critical": config.PC_CRITICAL},
        },
        "primary_norad": p["norad_id"],
        "secondary_norad": s["norad_id"],
        "detected": first is not None,
        "runs_before_detection": misses,
        "first_detection": first,
        "detection": None,
        "history": history,
    }
    if first is not None:
        result["detection"] = {
            "run_at": first["run_at"],
            "tca": first["tca"],
            "miss_distance_m": first["miss_distance_m"],
            "relative_velocity_ms": first["relative_velocity_ms"],
            "pc": first["pc"],
            "risk_level": first["risk_level"],
            "hours_before_tca": first["hours_before_tca"],
            "location": location,
        }
    return result


# --- ephemeris (same shape as /conjunctions/{id}/ephemeris) -------------------

def ephemeris(scn: dict, window_s: int) -> dict:
    from backend.app.api import conjunctions as conj

    window = max(conj.EPHEM_WINDOW_MIN_S, min(conj.EPHEM_WINDOW_MAX_S, window_s))
    p = scenarios.primary(scn)
    s = scenarios.secondary(scn)
    event_iso = scn["event"]["time_utc"]
    tle_p = scenarios.elset_at(p, event_iso) or scenarios.latest_elset(p)
    tle_s = scenarios.elset_at(s, event_iso) or scenarios.latest_elset(s)
    sat_p = conj._satrec_or_none(tle_p)
    sat_s = conj._satrec_or_none(tle_s)

    rep = screening_replay(scn)
    tca = _parse(rep["first_detection"]["tca"]) if rep["detected"] \
        else _parse(event_iso)
    state = None
    if sat_p is not None and sat_s is not None:
        try:
            refined = conj._refine_tca(sat_p, sat_s, tca)
            if refined is not None:
                tca = refined
                state = conj._tca_state(sat_p, sat_s, tca)
        except Exception:
            state = None

    start = tca - timedelta(seconds=window)
    times = [start + timedelta(seconds=conj.EPHEM_STEP_S * i)
             for i in range(2 * window + 1)]
    return {
        "conjunction_id": None,
        "scenario_id": scn["id"],
        "provenance": scn["provenance"],
        "tca": tca.strftime(ISO_FMT),
        "tca_precise": _iso_ms(tca),
        "window_s": window,
        "step_s": conj.EPHEM_STEP_S,
        "start": start.strftime(ISO_FMT),
        "start_precise": _iso_ms(start),
        "gmst_start_rad": propagate.gmst_rad(start),
        "earth_rot_rad_s": conj.EARTH_ROT_RAD_S,
        "primary": {"norad_id": p["norad_id"], "name": p["name"],
                    "r": conj._grid_positions(sat_p, times)},
        "secondary": {"norad_id": s["norad_id"], "name": s["name"],
                      "r": conj._grid_positions(sat_s, times)},
        "tca_state": state if state is not None else conj._null_tca_state(),
        "sigma_m": conj._combined_sigma_m(tca, tle_p, tle_s),
    }


TRACK_MINUTES = 120
TRACK_STEP_S = 30


def run_tracks(scn: dict, minutes: int = TRACK_MINUTES,
               step_s: int = TRACK_STEP_S) -> dict | None:
    """SGP4 positions of both objects from the first detecting run onward.

    What the catalog showed at the moment Kessler flagged the pair: each
    object's element set as of the run, propagated on a step_s grid for
    `minutes` (a little over one orbit, so the page can draw the orbit
    rings and the objects' motion from real SGP4 states).
    """
    from backend.app.api import conjunctions as conj

    rep = screening_replay(scn)
    if not rep["detected"]:
        return None
    run_at = _parse(rep["first_detection"]["run_at"])
    times = [run_at + timedelta(seconds=step_s * i)
             for i in range(int(minutes * 60 / step_s) + 1)]
    out = {"start": _iso(run_at), "step_s": step_s}
    for key, obj in (("primary", scenarios.primary(scn)),
                     ("secondary", scenarios.secondary(scn))):
        rec = scenarios.elset_at(obj, _iso(run_at))
        out[key] = conj._grid_positions(conj._satrec_or_none(rec), times)
    return out


def tca_states(scn: dict) -> tuple[datetime, list[dict]] | None:
    """Refined TCA and each object's TEME state {r, v} at it (km, km/s)."""
    from backend.app.api import conjunctions as conj

    rep = screening_replay(scn)
    event_iso = scn["event"]["time_utc"]
    objs = [scenarios.primary(scn), scenarios.secondary(scn)]
    tles = [scenarios.elset_at(o, event_iso) or scenarios.latest_elset(o) for o in objs]
    sats = [conj._satrec_or_none(t) for t in tles]
    if any(sat is None for sat in sats):
        return None
    tca = _parse(rep["first_detection"]["tca"]) if rep["detected"] else _parse(event_iso)
    refined = conj._refine_tca(sats[0], sats[1], tca)
    if refined is not None:
        tca = refined
    rp, vp, rs, vs, ok = conj._pair_states(sats[0], sats[1], [tca])
    if not bool(ok[0]):
        return None
    return tca, [
        {"r": [float(x) for x in rp[0]], "v": [float(x) for x in vp[0]]},
        {"r": [float(x) for x in rs[0]], "v": [float(x) for x in vs[0]]},
    ]


# WGS-84 ellipsoid - the encounter's altitude is quoted the way published
# accounts quote it: height above the ellipsoid, not above a sphere of the
# equatorial radius (which reads ~20 km low at the 2009 event's 72.5 deg N).
_WGS84_A_KM = 6378.137
_WGS84_E2 = (1.0 / 298.257223563) * (2.0 - 1.0 / 298.257223563)


def geodetic_lat_alt(r: list[float]) -> tuple[float, float]:
    """(geodetic latitude in degrees, height above the WGS-84 ellipsoid in km)
    for a position in an Earth-centred frame. Latitude and height do not
    depend on the frame's rotation about z, so an inertial (TEME) vector
    works as-is. Iterative, converges to sub-millimetre in a few rounds."""
    x, y, z = r
    p = math.hypot(x, y)
    if p < 1e-9:                                   # on the polar axis
        b = _WGS84_A_KM * math.sqrt(1.0 - _WGS84_E2)
        return (90.0 if z >= 0 else -90.0), abs(z) - b
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))
    h = 0.0
    for _ in range(6):
        n = _WGS84_A_KM / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - _WGS84_E2 * n / (n + h)))
    return math.degrees(lat), h


def encounter_geometry(scn: dict) -> dict | None:
    """Replay-derived encounter facts for the page (not published values)."""
    st = tca_states(scn)
    if st is None:
        return None
    tca, states = st
    r = states[0]["r"]
    lat_deg, alt_km = geodetic_lat_alt(r)
    lon = math.degrees(math.atan2(r[1], r[0]) - propagate.gmst_rad(tca))
    return {
        "tca": _iso_ms(tca),
        "lat_deg": round(lat_deg, 2),
        "lon_deg": round(((lon + 180.0) % 360.0) - 180.0, 2),
        "alt_km": round(alt_km, 1),
        "gmst_rad": propagate.gmst_rad(tca),
    }
