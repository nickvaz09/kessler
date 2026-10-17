#!/usr/bin/env python3
"""Rebuild backend/app/scenarios/iridium_cosmos_2009.json (RECONSTRUCTED orbits).

Why this exists
---------------
The archived element sets for Iridium 33 (NORAD 24946) and Cosmos 2251
(NORAD 22675) from February 2009 live behind Space-Track's gp_history class,
which this build environment cannot reach. So the Time Machine ships a
*reconstruction*: TLE-format element sets generated here so that the two
objects' SGP4 tracks meet where and when the published record says they did.
scripts/fetch_scenario_tles.py replaces them with the archived element sets
whenever Space-Track credentials are available; the backend prefers that
archived file when it exists.

Published figures used (NASA Orbital Debris Quarterly News, vol. 13 issue 2,
April 2009 - attributed in the scenario file):
  - time:      10 February 2009, about 16:56 UTC
  - altitude:  about 789 km, above northern Siberia
  - speed:     roughly 11.7 km/s relative
  - Iridium 33 inclination about 86.4 deg; Cosmos 2251 about 74.0 deg

Reconstruction choices (not published values - documented in the output):
  - encounter point: longitude 97.9 E, latitude solved so the two orbit
    planes cross at the published relative speed (lands near 72 N)
  - near-circular orbits (eccentricity 0.0002 / 0.0016), drag terms zeroed
    (BSTAR = 0, n-dot = 0) so the 72 h propagation stays purely geometric
  - element-set epochs on 7 February 2009, before the 72 h screening window
  - Iridium 33 northbound, Cosmos 2251 southbound-east through the crossing

Method: for each object, a Newton iteration on (mean anomaly, RAAN, mean
motion) drives the SGP4 position at the encounter time onto the target
point; the TLE is then formatted (4-decimal angles, 8-decimal mean motion)
and the rounding residual is removed by a final 1e-4 deg mean-anomaly /
RAAN grid search on the formatted strings. Deterministic; no network.

Usage: python scripts/reconstruct_scenario.py   (from the repo root)
"""
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sgp4.api import WGS72, Satrec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.ingestion.propagate import (MU_KM3_S2, RE_KM,  # noqa: E402
                                             gmst_rad, jday_from_datetime)

J2 = 1.08262998905e-3
OUT = ROOT / "backend" / "app" / "scenarios" / "iridium_cosmos_2009.json"

TCA = datetime(2009, 2, 10, 16, 56, 0, tzinfo=timezone.utc)
ALT_KM = 789.0
LON_DEG = 97.9
REL_SPEED_TARGET_KMS = 11.7

OBJECTS = [
    {
        "norad_id": 24946, "name": "Iridium 33", "intl": "97051C",
        "inc": 86.4, "ecc": 0.0002, "argp": 90.0, "north": True,
        "epoch": datetime(2009, 2, 7, 11, 43, 12, tzinfo=timezone.utc),
    },
    {
        "norad_id": 22675, "name": "Cosmos 2251", "intl": "93036A",
        "inc": 74.0, "ecc": 0.0016, "argp": 90.0, "north": False,
        "epoch": datetime(2009, 2, 7, 9, 12, 36, tzinfo=timezone.utc),
    },
]


# --- TLE formatting -----------------------------------------------------------

def _checksum(line: str) -> int:
    s = 0
    for ch in line[:68]:
        if ch.isdigit():
            s += int(ch)
        elif ch == "-":
            s += 1
    return s % 10


def _epoch_field(dt: datetime) -> str:
    start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
    doy = (dt - start).total_seconds() / 86400.0 + 1.0
    return f"{dt.year % 100:02d}{doy:012.8f}"


def format_tle(norad: int, intl: str, epoch: datetime, inc: float, raan: float,
               ecc: float, argp: float, mean_anom: float, n_rev_day: float,
               elset: int = 999, revnum: int = 0) -> tuple[str, str]:
    l1 = (f"1 {norad:05d}U {intl:<8s} {_epoch_field(epoch)} "
          f" .00000000  00000-0  00000-0 0 {elset:4d}")
    l1 = l1 + str(_checksum(l1))
    l2 = (f"2 {norad:05d} {inc % 360:8.4f} {raan % 360:8.4f} "
          f"{int(round(ecc * 1e7)):07d} {argp % 360:8.4f} {mean_anom % 360:8.4f} "
          f"{n_rev_day:11.8f}{revnum:5d}")
    l2 = l2 + str(_checksum(l2))
    assert len(l1) == 69 and len(l2) == 69, (l1, l2)
    return l1, l2


# --- geometry -------------------------------------------------------------------

def target_teme(lat_deg: float) -> np.ndarray:
    """Encounter point (spherical Earth, RE + ALT) in TEME at TCA."""
    ra = math.radians(LON_DEG) + gmst_rad(TCA)
    lat = math.radians(lat_deg)
    r = RE_KM + ALT_KM
    return np.array([r * math.cos(lat) * math.cos(ra),
                     r * math.cos(lat) * math.sin(ra),
                     r * math.sin(lat)])


def circular_velocity_dir(lat_deg: float, inc_deg: float, north: bool) -> float:
    """Heading azimuth (deg from north) of an orbit of inclination inc at lat."""
    s = math.cos(math.radians(inc_deg)) / math.cos(math.radians(lat_deg))
    az = math.degrees(math.asin(max(-1.0, min(1.0, s))))
    return az if north else 180.0 - az


def solve_latitude() -> float:
    """Latitude where the two planes cross at the published relative speed."""
    v = math.sqrt(MU_KM3_S2 / (RE_KM + ALT_KM))
    best = None
    for k in range(0, 2001):
        lat = 60.0 + k * 0.007
        if lat >= 74.0:
            break
        a1 = circular_velocity_dir(lat, OBJECTS[0]["inc"], OBJECTS[0]["north"])
        a2 = circular_velocity_dir(lat, OBJECTS[1]["inc"], OBJECTS[1]["north"])
        rel = 2.0 * v * math.sin(math.radians(abs(a1 - a2)) / 2.0)
        err = abs(rel - REL_SPEED_TARGET_KMS)
        if best is None or err < best[0]:
            best = (err, lat)
    return round(best[1], 2)


def initial_guess(obj: dict, lat_deg: float) -> tuple[float, float, float]:
    """Two-body guess: (M0 deg at epoch, RAAN deg, n rev/day)."""
    inc = math.radians(obj["inc"])
    lat = math.radians(lat_deg)
    u = math.asin(math.sin(lat) / math.sin(inc))       # argument of latitude
    if not obj["north"]:
        u = math.pi - u                                # past the northern apex
    p = target_teme(lat_deg)
    ra = math.atan2(p[1], p[0])
    d_ra = math.atan2(math.cos(inc) * math.sin(u), math.cos(u))
    raan = math.degrees(ra - d_ra) % 360.0
    a = RE_KM + ALT_KM
    n_rad_s = math.sqrt(MU_KM3_S2 / a ** 3)
    n_rev_day = n_rad_s * 86400.0 / (2.0 * math.pi)
    dt = (TCA - obj["epoch"]).total_seconds()
    # J2 secular rates carry the guess back to the element epoch closely
    # enough that the Newton fit stays on the intended (north/southbound)
    # branch through the encounter point.
    k = 1.5 * J2 * (RE_KM / a) ** 2 * n_rad_s
    ci = math.cos(inc)
    raan_dot = -k * ci
    argp_dot = 0.5 * k * (5.0 * ci * ci - 1.0)
    m_dot = n_rad_s + 0.5 * k * (3.0 * ci * ci - 1.0)
    raan0 = (raan - math.degrees(raan_dot * dt)) % 360.0
    argp_tca = obj["argp"] + math.degrees(argp_dot * dt)
    m_at_tca = math.degrees(u) - argp_tca
    m0 = (m_at_tca - math.degrees(m_dot * dt)) % 360.0
    return m0, raan0, n_rev_day


def _satrec(obj, m0, raan, n_rev_day) -> Satrec:
    sat = Satrec()
    ep = obj["epoch"]
    jd, fr = jday_from_datetime(ep)
    sat.sgp4init(
        WGS72, "i", obj["norad_id"], (jd + fr) - 2433281.5,
        0.0, 0.0, 0.0, obj["ecc"], math.radians(obj["argp"]),
        math.radians(obj["inc"]), math.radians(m0),
        n_rev_day * 2.0 * math.pi / 1440.0, math.radians(raan))
    return sat


def _pos_at_tca(sat: Satrec) -> np.ndarray:
    jd, fr = jday_from_datetime(TCA)
    e, r, _v = sat.sgp4(jd, fr)
    if e != 0:
        raise RuntimeError(f"sgp4 error {e}")
    return np.array(r)


def fit(obj: dict, lat_deg: float) -> tuple[float, float, float]:
    target = target_teme(lat_deg)
    x = np.array(initial_guess(obj, lat_deg), dtype=float)
    steps = np.array([1e-4, 1e-4, 1e-7])
    for _ in range(30):
        f0 = _pos_at_tca(_satrec(obj, *x)) - target
        if np.linalg.norm(f0) < 1e-4:
            break
        jac = np.empty((3, 3))
        for k in range(3):
            xk = x.copy()
            xk[k] += steps[k]
            jac[:, k] = (_pos_at_tca(_satrec(obj, *xk)) - target - f0) / steps[k]
        dx, *_ = np.linalg.lstsq(jac, -f0, rcond=None)
        x = x + dx
    return float(x[0]) % 360.0, float(x[1]) % 360.0, float(x[2])


def tle_position_at_tca(l1: str, l2: str) -> np.ndarray:
    return _pos_at_tca(Satrec.twoline2rv(l1, l2))


def heading_deg(l1: str, l2: str) -> float:
    """Velocity azimuth (deg from north) at TCA for a formatted TLE."""
    sat = Satrec.twoline2rv(l1, l2)
    jd, fr = jday_from_datetime(TCA)
    _e, r, v = sat.sgp4(jd, fr)
    r, v = np.array(r), np.array(v)
    up = r / np.linalg.norm(r)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return math.degrees(math.atan2(v @ east, v @ north)) % 360.0


def build() -> dict:
    lat = solve_latitude()
    target = target_teme(lat)
    objects_out = []
    for obj in OBJECTS:
        m0, raan, n = fit(obj, lat)
        # Formatting residual: search the 4-decimal grid around the fit.
        best = None
        for dm in range(-3, 4):
            for dr in range(-2, 3):
                mm = round(m0, 4) + dm * 1e-4
                rr = round(raan, 4) + dr * 1e-4
                l1, l2 = format_tle(obj["norad_id"], obj["intl"], obj["epoch"],
                                    obj["inc"], rr, obj["ecc"], obj["argp"], mm, n)
                err = float(np.linalg.norm(tle_position_at_tca(l1, l2) - target))
                if best is None or err < best[0]:
                    best = (err, l1, l2)
        err_km, l1, l2 = best
        heading = heading_deg(l1, l2)
        if (heading < 90.0) != obj["north"]:
            raise SystemExit(f"{obj['name']}: fit landed on the wrong branch"
                             f" (heading {heading:.1f} deg)")
        print(f"{obj['name']}: residual to target {err_km * 1000:.1f} m",
              file=sys.stderr)
        objects_out.append((obj, l1, l2))

    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    rec = {o["norad_id"]: (l1, l2) for o, l1, l2 in objects_out}
    for entry in data.get("objects", []):
        if entry["norad_id"] in rec:
            entry["line1"], entry["line2"] = rec[entry["norad_id"]]
    data.setdefault("reconstruction", {})
    data["reconstruction"].update({
        "method": ("TLE-format element sets fitted so the SGP4 tracks of both"
                   " objects pass through one encounter point at the published"
                   " time and altitude, with the published inclinations."
                   " Generated by scripts/reconstruct_scenario.py."),
        "encounter_point": {"lat_deg": lat, "lon_deg": LON_DEG, "alt_km": ALT_KM,
                            "time_utc": TCA.strftime("%Y-%m-%dT%H:%M:%SZ")},
        "choices": [
            "Encounter longitude 97.9 E is a point inside the published region"
            " (northern Siberia); the latitude is solved so the orbit planes"
            " cross at the published ~11.7 km/s relative speed.",
            "Near-circular orbits (eccentricity 0.0002 and 0.0016) and zero"
            " drag terms are reconstruction choices, not archived values.",
            "Element-set epochs on 7 February 2009 sit before the 72-hour"
            " screening window, as a real pre-event element set would.",
        ],
    })
    return data


if __name__ == "__main__":
    out = build()
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}", file=sys.stderr)
