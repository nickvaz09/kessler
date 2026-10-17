"""SGP4 propagation helpers.

All positions/velocities are TEME frame, km and km/s (sgp4 native output).
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import SGP4_ERRORS, Satrec, jday

from backend.app.ingestion.celestrak import ISO_FMT, tle_epoch

MU_KM3_S2 = 398600.4418   # Earth gravitational parameter
RE_KM = 6378.137          # Earth equatorial radius
TWO_PI = 2.0 * math.pi


def satrec_from_tle(line1: str, line2: str) -> Satrec:
    return Satrec.twoline2rv(line1, line2)


def jday_from_datetime(dt: datetime) -> tuple[float, float]:
    """(jd, fr) for a datetime (naive datetimes are treated as UTC)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                dt.second + dt.microsecond * 1e-6)


def jday_arrays(times: list[datetime]) -> tuple[np.ndarray, np.ndarray]:
    """Vector (jd, fr) arrays for a list of datetimes."""
    jd = np.empty(len(times))
    fr = np.empty(len(times))
    for i, t in enumerate(times):
        jd[i], fr[i] = jday_from_datetime(t)
    return jd, fr


def positions_at(line1: str, line2: str, times: list[datetime]) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a TLE to each time. Returns (r[N,3] km TEME, v[N,3] km/s).

    Raises ValueError with the sgp4 error code if any propagation fails.
    """
    sat = satrec_from_tle(line1, line2)
    jd, fr = jday_arrays(times)
    e, r, v = sat.sgp4_array(jd, fr)
    if np.any(e != 0):
        code = int(e[e != 0][0])
        raise ValueError(
            f"sgp4 error code {code}: {SGP4_ERRORS.get(code, 'unknown error')}"
        )
    return r, v


def _semi_major_axis_km(no_kozai_rad_min: float) -> float:
    n_rad_s = no_kozai_rad_min / 60.0
    return (MU_KM3_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)


def orbital_elements(line1: str, line2: str) -> dict:
    """Mean orbital elements decoded from a TLE.

    apogee/perigee derived from mean motion + eccentricity
    (a from n via mu=398600.4418, Re=6378.137).
    """
    sat = satrec_from_tle(line1, line2)
    a = _semi_major_axis_km(sat.no_kozai)
    ecc = sat.ecco
    period_min = TWO_PI / sat.no_kozai
    return {
        "epoch": tle_epoch(line1).strftime(ISO_FMT),
        "inclination_deg": math.degrees(sat.inclo),
        "raan_deg": math.degrees(sat.nodeo),
        "eccentricity": ecc,
        "arg_perigee_deg": math.degrees(sat.argpo),
        "mean_anomaly_deg": math.degrees(sat.mo),
        "mean_motion_rev_day": sat.no_kozai * 1440.0 / TWO_PI,
        "period_min": period_min,
        "apogee_km": a * (1.0 + ecc) - RE_KM,
        "perigee_km": a * (1.0 - ecc) - RE_KM,
        "semi_major_axis_km": a,
        "bstar": sat.bstar,
    }


def sample_orbit(line1: str, line2: str, n: int = 181) -> list[list[float]]:
    """n positions [x,y,z] km TEME over exactly one orbital period from epoch."""
    sat = satrec_from_tle(line1, line2)
    period_min = TWO_PI / sat.no_kozai
    epoch = tle_epoch(line1)
    times = [epoch + timedelta(minutes=period_min * i / (n - 1)) for i in range(n)]
    r, _v = positions_at(line1, line2, times)
    return [[float(x), float(y), float(z)] for x, y, z in r]


def gmst_rad(dt: datetime) -> float:
    """Greenwich Mean Sidereal Time in radians, wrapped to [0, 2*pi).

    IAU-82 (Vallado) polynomial in julian centuries T from J2000:

        gmst_s = 67310.54841 + (876600*3600 + 8640184.812866)*T
                 + 0.093104*T**2 - 6.2e-6*T**3      [seconds of sidereal time]

    T is computed from the UTC julian date, i.e. UTC stands in for UT1.
    |UT1 - UTC| < 0.9 s by definition, so this is an ~seconds-level
    approximation (< 0.004 deg of ground-track longitude) - ample for
    display-grade ground tracks.
    """
    jd, fr = jday_from_datetime(dt)
    t = (jd - 2451545.0 + fr) / 36525.0
    gmst_s = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t
    )
    return (gmst_s % 86400.0) * TWO_PI / 86400.0


# --- Topocentric ground-station geometry -------------------------------------
#
# Frames: TEME positions are rotated into an Earth-fixed frame by a single
# z-rotation through GMST. That is the standard TEME -> PEF (pseudo-Earth-
# fixed) transform (Vallado, AIAA 2006-6753); true ITRF additionally applies
# polar motion (~arcseconds, ~15 m at the surface) and UT1-UTC (< 0.9 s),
# both of which shift look angles by arcseconds - negligible for pass
# prediction, so PEF stands in for ECEF throughout ("ECEF-ish").

WGS84_F = 1.0 / 298.257223563          # WGS84 flattening (semi-major axis == RE_KM)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)   # first eccentricity squared


def _gmst_rad_many(times: list[datetime]) -> np.ndarray:
    """Vectorized twin of gmst_rad: GMST radians in [0, 2*pi) per datetime."""
    jd, fr = jday_arrays(times)
    t = (jd - 2451545.0 + fr) / 36525.0
    gmst_s = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t
    )
    return (gmst_s % 86400.0) * TWO_PI / 86400.0


def observer_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """WGS84 geodetic site -> ECEF position [x, y, z] km."""
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    h_km = alt_m / 1000.0
    sin_phi = math.sin(phi)
    n = RE_KM / math.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)
    return np.array([
        (n + h_km) * math.cos(phi) * math.cos(lam),
        (n + h_km) * math.cos(phi) * math.sin(lam),
        (n * (1.0 - WGS84_E2) + h_km) * sin_phi,
    ])


def look_angles(r_teme: np.ndarray, times: list[datetime], lat_deg: float,
                lon_deg: float, alt_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Topocentric (elevation_deg[N], azimuth_deg[N]) for TEME positions.

    Each position is rotated TEME -> PEF via GMST (see module note above),
    the site vector is subtracted, and the range vector is resolved in the
    site's SEZ (south-east-zenith) horizon frame built from the geodetic
    latitude. Azimuth is degrees clockwise from true north in [0, 360).
    """
    gmst = _gmst_rad_many(times)
    cos_g, sin_g = np.cos(gmst), np.sin(gmst)
    ecef = np.empty_like(r_teme)
    ecef[:, 0] = cos_g * r_teme[:, 0] + sin_g * r_teme[:, 1]
    ecef[:, 1] = -sin_g * r_teme[:, 0] + cos_g * r_teme[:, 1]
    ecef[:, 2] = r_teme[:, 2]
    rho = ecef - observer_ecef(lat_deg, lon_deg, alt_m)

    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)
    south = rho @ np.array([sin_phi * cos_lam, sin_phi * sin_lam, -cos_phi])
    east = rho @ np.array([-sin_lam, cos_lam, 0.0])
    zenith = rho @ np.array([cos_phi * cos_lam, cos_phi * sin_lam, sin_phi])

    rng = np.linalg.norm(rho, axis=1)
    el = np.degrees(np.arcsin(np.clip(zenith / rng, -1.0, 1.0)))
    az = np.degrees(np.arctan2(east, -south)) % 360.0
    return el, az


TRACK_STEP_S = 10  # sky-view track sampling interval (seconds from AOS)


def _refine_pass(line1: str, line2: str, lat_deg: float, lon_deg: float,
                 alt_m: float, t_lo: datetime, t_hi: datetime,
                 min_el_deg: float) -> dict | None:
    """1 s refinement of one coarse above-threshold run: AOS/LOS crossings by
    scan, max-elevation vertex by parabolic interpolation of the 1 s samples.
    Returns None if the run evaporates at 1 s resolution (grid artifact).

    Also emits the sky-view "track": [[t_offset_s, az_deg, el_deg]] sampled
    every TRACK_STEP_S seconds from AOS to LOS (offsets measured from AOS;
    the LOS sample is always included, so the last offset == duration_s).
    Track samples are read straight off the refined 1 s grid between AOS and
    LOS - elevation is not re-filtered against min_el_deg, so a multi-hump
    pass keeps its below-threshold dip. Azimuth stays in [0, 360).
    """
    total_s = int(round((t_hi - t_lo).total_seconds()))
    times = [t_lo + timedelta(seconds=i) for i in range(total_s + 1)]
    r, _v = positions_at(line1, line2, times)
    el, az = look_angles(r, times, lat_deg, lon_deg, alt_m)

    idx = np.nonzero(el >= min_el_deg)[0]
    if idx.size == 0:
        return None
    a, b = int(idx[0]), int(idx[-1])

    k = a + int(np.argmax(el[a:b + 1]))
    max_el = float(el[k])
    tca = times[k]
    if a < k < b:
        # Parabola through (t[k-1], t[k], t[k+1]) at 1 s spacing: vertex offset
        # x* = (e[k-1]-e[k+1]) / (2*(e[k-1]-2e[k]+e[k+1])), value e[k]-(e[k-1]-e[k+1])*x*/4.
        denom = float(el[k - 1] - 2.0 * el[k] + el[k + 1])
        if denom < 0.0:
            off = max(-1.0, min(1.0, 0.5 * float(el[k - 1] - el[k + 1]) / denom))
            max_el = float(el[k]) - 0.25 * float(el[k - 1] - el[k + 1]) * off
            tca = times[k] + timedelta(seconds=off)

    def iso(dt: datetime) -> str:
        return dt.replace(microsecond=0).strftime(ISO_FMT)

    duration_s = b - a  # samples are exactly 1 s apart
    offsets = list(range(0, duration_s + 1, TRACK_STEP_S))
    if offsets[-1] != duration_s:
        offsets.append(duration_s)  # always close the track at LOS
    track = [
        # round() can push 359.96 to 360.0 - wrap after rounding.
        [off, round(float(az[a + off]), 1) % 360.0, round(float(el[a + off]), 1)]
        for off in offsets
    ]

    return {
        "aos": iso(times[a]),
        "los": iso(times[b]),
        "tca": iso(tca),
        "max_el_deg": round(max_el, 1),
        "aos_az_deg": round(float(az[a]), 1),
        "los_az_deg": round(float(az[b]), 1),
        "duration_s": duration_s,
        "track": track,
    }


def passes(line1: str, line2: str, lat_deg: float, lon_deg: float, alt_m: float,
           start: datetime, hours: float, min_el_deg: float = 10.0,
           step_s: int = 30) -> list[dict]:
    """Ground-station passes over [start, start+hours], chronological.

    Coarse scan: SGP4 states every step_s (30 s default) reduced to
    topocentric elevation via look_angles; contiguous runs at/above
    min_el_deg become candidate passes. Each run is then refined on a 1 s
    grid extended one coarse step either side (clamped to the window):
    AOS = first 1 s sample at/above threshold, LOS = last, culmination by
    parabolic vertex through the peak sample and its neighbors.

    Grazing passes shorter than step_s can slip between coarse samples;
    a pass in progress at a window edge is reported from/to that edge.
    Raises ValueError when SGP4 propagation fails (via positions_at).
    -> [{aos, los, tca, max_el_deg, aos_az_deg, los_az_deg, duration_s,
         track: [[t_offset_s, az_deg, el_deg]]}]   (track: see _refine_pass)
    """
    n_steps = int(round(hours * 3600.0 / step_s))
    times = [start + timedelta(seconds=step_s * i) for i in range(n_steps + 1)]
    r, _v = positions_at(line1, line2, times)
    el, _az = look_angles(r, times, lat_deg, lon_deg, alt_m)
    above = el >= min_el_deg

    step = timedelta(seconds=step_s)
    window_end = times[-1]
    found = []
    i = 0
    while i <= n_steps:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 <= n_steps and above[j + 1]:
            j += 1
        refined = _refine_pass(
            line1, line2, lat_deg, lon_deg, alt_m,
            max(start, times[i] - step), min(window_end, times[j] + step),
            min_el_deg,
        )
        if refined is not None:
            found.append(refined)
        i = j + 1
    return found


def subpoint(line1: str, line2: str, dt: datetime) -> dict:
    """Sub-satellite point at `dt` from real SGP4 propagation.

    Longitude = TEME right ascension (atan2(y, x)) minus GMST (see
    gmst_rad), wrapped to [-180, 180); latitude is geocentric
    (asin(z/|r|)); altitude is height above the RE_KM = 6378.137 km
    sphere; velocity_kms is the inertial speed |v|.
    Raises ValueError when sgp4 propagation fails (via positions_at).
    """
    r, v = positions_at(line1, line2, [dt])
    x, y, z = (float(c) for c in r[0])
    rmag = math.sqrt(x * x + y * y + z * z)
    lon = math.degrees(math.atan2(y, x) - gmst_rad(dt))
    return {
        "lat_deg": math.degrees(math.asin(z / rmag)),
        "lon_deg": ((lon + 180.0) % 360.0) - 180.0,
        "alt_km": rmag - RE_KM,
        "velocity_kms": float(np.linalg.norm(v[0])),
    }
