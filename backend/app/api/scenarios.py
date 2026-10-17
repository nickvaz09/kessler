"""Time Machine API: historical conjunctions replayed through Kessler.

GET /scenarios                      list (id, title, provenance, event)
GET /scenarios/{id}                 metadata, provenance, the two objects and
                                    the replayed screening detection
GET /scenarios/{id}/ephemeris       both objects on a 1 s grid around TCA -
                                    the /conjunctions/{id}/ephemeris shape
GET /scenarios/{id}/debris?hours=   illustrative breakup cloud at TCA + hours,
                                    plus the live-catalog band-overlap count

Contract notes live in docs/v2/time-machine.md. Numbers the page shows come
from here: published facts are passed through from the scenario file with
their source; screening numbers come from the live screening code; the
debris cloud is labeled as a model everywhere.
"""
import math
import threading
import time
from datetime import timedelta

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app import config, db, scenarios
try:
    from backend.app.ingestion.propagate import MU_KM3_S2, RE_KM
except ImportError:  # propagate swapped for a stub (API tests): same constants
    MU_KM3_S2, RE_KM = 398600.4418, 6378.137


class _LazyModule:
    """Import the scenario engines on first use, not at app import: they pull
    in the screening stack, which API tests replace with stubs."""

    def __init__(self, name: str):
        self._name, self._mod = name, None

    def __getattr__(self, attr):
        if self._mod is None:
            import importlib
            self._mod = importlib.import_module(self._name)
        return getattr(self._mod, attr)


breakup = _LazyModule("backend.app.scenarios.breakup")
replay = _LazyModule("backend.app.scenarios.replay")

router = APIRouter()

DEBRIS_MAX_HOURS = 24.0 * 365.0 * 10.0
DEBRIS_DRAW_DEFAULT = 1500
DEBRIS_DRAW_MAX = 5000
BAND_CACHE_S = 120.0

_band_cache: dict = {}
_band_lock = threading.Lock()
_cloud_cache: dict = {}
_cloud_lock = threading.Lock()


def _get_or_404(scenario_id: str) -> dict:
    scn = scenarios.get_scenario(scenario_id)
    if scn is None:
        raise HTTPException(status_code=404,
                            detail=f"Scenario {scenario_id!r} not found")
    return scn


def _object_summary(obj: dict, event_iso: str) -> dict:
    from backend.app.ingestion import propagate

    rec = scenarios.elset_at(obj, event_iso) or scenarios.latest_elset(obj)
    try:
        el = propagate.orbital_elements(rec["line1"], rec["line2"])
    except Exception:
        el = None
    out = {k: v for k, v in obj.items() if k != "elsets"}
    out["elset_count"] = len(obj["elsets"])
    out["elset"] = rec
    if el is not None:
        out["elements"] = {
            "inclination_deg": round(el["inclination_deg"], 4),
            "raan_deg": round(el["raan_deg"], 4),
            "eccentricity": el["eccentricity"],
            "period_min": round(el["period_min"], 3),
            "apogee_km": round(el["apogee_km"], 1),
            "perigee_km": round(el["perigee_km"], 1),
        }
    return out


def _public(scn: dict) -> dict:
    return {k: v for k, v in scn.items() if not k.startswith("_") and k != "objects"}


@router.get("/scenarios")
def list_scenarios():
    items = []
    for scn in scenarios.list_scenarios():
        items.append({
            "id": scn["id"],
            "title": scn["title"],
            "summary": scn.get("summary"),
            "provenance": scn["provenance"],
            "provenance_label": scn.get("provenance_label"),
            "event": scn["event"],
        })
    return {"count": len(items), "scenarios": items}


@router.get("/scenarios/{scenario_id}")
def get_scenario(scenario_id: str):
    scn = _get_or_404(scenario_id)
    body = _public(scn)
    event_iso = scn["event"]["time_utc"]
    body["objects"] = [_object_summary(o, event_iso) for o in scn["objects"]]
    body["screening"] = replay.screening_replay(scn)
    body["encounter"] = replay.encounter_geometry(scn)
    body["tracks"] = replay.run_tracks(scn)
    return body


@router.get("/scenarios/{scenario_id}/ephemeris")
def get_scenario_ephemeris(scenario_id: str, window_s: int = Query(1200)):
    scn = _get_or_404(scenario_id)
    return replay.ephemeris(scn, window_s)


# --- debris ------------------------------------------------------------------

def _cloud(scn: dict) -> dict | None:
    key = replay._cache_key(scn)
    with _cloud_lock:
        if key in _cloud_cache:
            return _cloud_cache[key]
    st = replay.tca_states(scn)
    if st is None:
        return None
    tca, states = st
    rep = replay.screening_replay(scn)
    v_rel = (rep["detection"]["relative_velocity_ms"] if rep["detected"]
             else float(np.linalg.norm(np.subtract(states[0]["v"], states[1]["v"])) * 1000))
    objs = [scenarios.primary(scn), scenarios.secondary(scn)]
    parents = [{"norad_id": o["norad_id"], "name": o["name"],
                "mass_kg": o["mass_kg"], **s} for o, s in zip(objs, states)]
    cloud = breakup.generate(parents, v_rel, seed=int(scn.get("model_seed",
                                                                breakup.DEFAULT_SEED)))
    cloud["tca"] = tca
    cloud["parent_states"] = states
    cloud["v_rel_ms"] = v_rel
    with _cloud_lock:
        _cloud_cache[key] = cloud
    return cloud


def _line2_orbit(line2: str) -> tuple[float, float] | None:
    """(perigee_km, apogee_km) from TLE line 2 fixed columns; None if bad."""
    try:
        ecc = float("0." + line2[26:33].strip())
        n_rev_day = float(line2[52:63])
    except (ValueError, IndexError):
        return None
    if n_rev_day <= 0.0:
        return None
    n = n_rev_day * 2.0 * math.pi / 86400.0
    a = (MU_KM3_S2 / (n * n)) ** (1.0 / 3.0)
    return a * (1.0 - ecc) - RE_KM, a * (1.0 + ecc) - RE_KM


_BAND_SQL = """
SELECT s.name AS name, s.object_type AS object_type, t.line2 AS line2,
       t.ingested_at AS ingested_at
FROM satellites s
JOIN tle_history t ON t.id = (
    SELECT t2.id FROM tle_history t2
     WHERE t2.satellite_id = s.id
     ORDER BY t2.epoch DESC, t2.id DESC LIMIT 1)
"""


def live_band_overlap(conn, band: tuple[float, float],
                      fragment_prefixes: list[str]) -> dict:
    """Today's catalog objects whose perigee..apogee overlaps the band.

    A geometric band-overlap count from the live catalog - not a risk
    estimate. Also counts catalog objects named as fragments of the
    scenario's collision (name prefixes from the scenario file).
    """
    lo, hi = band
    prefixes = tuple(p.upper() for p in fragment_prefixes)
    key = (str(config.DB_PATH), lo, hi, prefixes, _db_fingerprint(conn))
    now = time.monotonic()
    with _band_lock:
        hit = _band_cache.get(key)
        if hit is not None and now - hit[0] < BAND_CACHE_S:
            return hit[1]
    total = in_band = payloads = fragments = fragments_in_band = 0
    latest_ingest = None
    for row in conn.execute(_BAND_SQL):
        orbit = _line2_orbit(row["line2"] or "")
        if orbit is None:
            continue
        total += 1
        if row["ingested_at"] and (latest_ingest is None or row["ingested_at"] > latest_ingest):
            latest_ingest = row["ingested_at"]
        hp, ha = orbit
        crosses = hp <= hi and ha >= lo
        is_frag = bool(prefixes) and (row["name"] or "").upper().startswith(prefixes)
        if is_frag:
            fragments += 1
        if crosses:
            in_band += 1
            if row["object_type"] == "PAYLOAD":
                payloads += 1
            if is_frag:
                fragments_in_band += 1
    out = {
        "band_km": [lo, hi],
        "catalog_objects": total,
        "objects_in_band": in_band,
        "payloads_in_band": payloads,
        "collision_fragments_cataloged": fragments,
        "collision_fragments_in_band": fragments_in_band,
        "fragment_name_prefixes": list(fragment_prefixes),
        "as_of": latest_ingest,
        "label": ("Band overlap: tracked objects in Kessler's current catalog"
                  " whose orbits (perigee to apogee) cross this altitude"
                  " shell. A geometric count, not a risk estimate."),
    }
    with _band_lock:
        _band_cache[key] = (now, out)
    return out


def _db_fingerprint(conn) -> tuple:
    # MAX(id) is an index lookup; COUNT(*) over tle_history was a full scan
    # on every debris request. New ingests always raise MAX(id).
    row = conn.execute("SELECT MAX(id) AS m FROM tle_history").fetchone()
    return (row["m"],)


def _pack_rows(frags: np.ndarray) -> list[list]:
    rows = []
    for f in frags:
        tau = float(f[13])
        rows.append([
            int(f[0]), round(float(f[1]), 3), round(float(f[2]), 5),
            round(float(f[3]), 1), float(f[4]), float(f[5]), float(f[6]),
            float(f[7]), float(f[8]), float(f[9]), float(f[10]), float(f[11]),
            float(f[12]), (tau if math.isfinite(tau) else None),
            round(float(f[14]), 3),
        ])
    return rows


@router.get("/scenarios/{scenario_id}/debris")
def get_scenario_debris(
    scenario_id: str,
    hours: float = Query(0.0, ge=0.0, le=DEBRIS_MAX_HOURS),
    limit: int = Query(DEBRIS_DRAW_DEFAULT, ge=1, le=DEBRIS_DRAW_MAX),
    conn=Depends(db.get_db),
):
    scn = _get_or_404(scenario_id)
    cloud = _cloud(scn)
    if cloud is None:
        raise HTTPException(status_code=422,
                            detail="Scenario objects cannot be propagated to TCA")
    frags = cloud["fragments"]
    t_s = hours * 3600.0
    n_all = int(frags.shape[0])
    step = max(1, math.ceil(n_all / limit))
    drawn = frags[::step]
    pos = breakup.positions(drawn, t_s)
    in_orbit_all = int(breakup.in_orbit_mask(frags, t_s).sum())
    band = breakup.altitude_band_km(frags)
    tca = cloud["tca"]
    return {
        "scenario_id": scn["id"],
        "provenance": scn["provenance"],
        "model": breakup.MODEL_LABEL,
        "model_version": breakup.MODEL_VERSION,
        "model_notes": [
            "Fragment count, sizes, area-to-mass and ejection speeds follow the"
            " NASA Standard Breakup Model collision branch, simplified.",
            "Fragments are split between the two parents by mass.",
            "Orbits advance with two-body motion plus J2 secular precession"
            " and a simple exponential-atmosphere drag decay — not SGP4.",
            "Masses are approximate public figures.",
        ],
        "epoch": replay._iso_ms(tca),
        "hours": hours,
        "time_utc": replay._iso(tca + timedelta(seconds=t_s)),
        "energy": cloud["energy"],
        "counts": {
            **cloud["counts"],
            "generated": n_all,
            "in_orbit_modeled": in_orbit_all,
            "reentered_modeled": n_all - in_orbit_all,
            "drawn": int(drawn.shape[0]),
        },
        "altitude_band_km": list(band),
        "live_catalog": live_band_overlap(
            conn, band, scn.get("catalog_fragment_prefixes", [])),
        "parents": cloud["parents"],
        "fragment_fields": breakup.FRAGMENT_FIELDS,
        "fragments": _pack_rows(drawn),
        "positions": [None if not np.all(np.isfinite(p)) else
                      [round(float(c), 2) for c in p] for p in pos],
    }
