"""TLE element-delta anomaly scan - maneuver-candidate detection.

For every satellite with two or more TLEs inside the scan window, successive
TLE pairs are differenced on two mean elements that maneuvers move and that
TLEs measure well:

- semi-major axis, derived from the line-2 mean motion via
  a = (mu / n^2)^(1/3) with mu = 398600.4418 km^3/s^2. TLE mean motion is a
  Kozai mean element, so the absolute SMA is offset from the osculating value
  by a few km - but the offset is common to both TLEs of a pair, so the
  DIFFERENCE is a clean signal;
- inclination, read directly from line 2 (degrees).

Drag-consistency heuristic (documented per contract)
----------------------------------------------------
Atmospheric drag lowers the SMA continuously, so an SMA drop is only
anomalous when it exceeds what the TLE's own drag term already explains.
Line 1 cols 34-43 carry n-dot/2 [rev/day^2]; with a proportional to n^(-2/3),
da/dn = -(2/3) a/n, so the drag-expected SMA change across a pair's interval
is

    expected_dsma_km = -(2/3) * (a1 / n1) * ndot1 * dt_days

with a1 [km] and n1 [rev/day] from the EARLIER TLE of the pair (the drag
state in effect across the interval) and ndot1 = 2 * (line-1 field). This is
the same n-dot -> dSMA/day conversion the re-entry watch uses for its decay
rate.

Flag rules
----------
- sma_raise:   dSMA >= +0.5 km. Drag cannot raise an orbit, so any raise at
               or above threshold is a maneuver signature regardless of n-dot.
- sma_drop:    dSMA <= -0.5 km and dSMA < 2 x expected_dsma_km - only drops
               beyond TWICE the drag-explained amount are flagged; anything
               within 2x drag is ordinary decay (plus TLE fit noise).
- inclination: |dinc| >= 0.01 deg. Drag leaves inclination essentially
               untouched; plane changes are expensive and deliberate. A pair
               that trips both an SMA rule and the inclination rule yields
               two items (one per kind).

Results are sorted by |delta| descending (km for the SMA kinds, degrees for
the inclination kind - the contract's single ordering) and capped at 50.
Element extraction is direct fixed-column TLE parsing (no SGP4 needed);
malformed rows are skipped, never fatal.
"""
import math
from datetime import datetime, timedelta, timezone

MU_KM3_S2 = 398600.4418        # matches propagate.py / maneuver.py

SMA_DELTA_KM = 0.5             # |dSMA| flag threshold [km]
INC_DELTA_DEG = 0.01           # |dinc| flag threshold [deg]
DRAG_TOLERANCE_FACTOR = 2.0    # drops within this multiple of drag pass
MAX_RESULTS = 50               # contract cap

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

_TLES_IN_WINDOW_SQL = """
SELECT s.norad_id   AS norad_id,
       s.name       AS name,
       t.satellite_id AS satellite_id,
       t.epoch      AS epoch,
       t.line1      AS line1,
       t.line2      AS line2
  FROM tle_history t
  JOIN satellites s ON s.id = t.satellite_id
 WHERE t.epoch >= ?
 ORDER BY t.satellite_id ASC, t.epoch ASC, t.id ASC
"""


def _parse_iso_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def tle_mean_elements(line1: str, line2: str) -> tuple[float, float, float, float]:
    """(a_km, inc_deg, mm_rev_day, ndot_rev_day2) from fixed TLE columns.

    a from the Kozai mean motion via a = (mu/n^2)^(1/3); ndot is the full
    first derivative of mean motion (2x the line-1 n-dot/2 field), rev/day^2.
    Raises ValueError on malformed lines or a non-positive mean motion.
    """
    mm = float(line2[52:63])                     # rev/day
    if mm <= 0.0:
        raise ValueError(f"non-positive mean motion ({mm})")
    inc = float(line2[8:16])                     # deg
    ndot = 2.0 * float(line1[33:43])             # field stores n-dot/2
    n_rad_s = mm * 2.0 * math.pi / 86400.0
    a_km = (MU_KM3_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)
    return a_km, inc, mm, ndot


def _sort_magnitude(item: dict) -> float:
    """|delta| for the contract's ordering: km for SMA kinds, deg for inc."""
    if item["kind"] == "inclination":
        return abs(item["delta_inc_deg"])
    return abs(item["delta_sma_km"])


def scan_tle_deltas(conn, since_hours: float = 48) -> list[dict]:
    """Scan successive TLE pairs in the window for maneuver signatures.

    Window = TLE epochs within the last `since_hours` (bounds are clamped by
    the API layer, not here). Returns items sorted by |delta| descending,
    capped at MAX_RESULTS. See the module docstring for the heuristic.
    """
    cutoff = (datetime.now(timezone.utc)
              - timedelta(hours=float(since_hours))).strftime(ISO_FMT)
    items: list[dict] = []
    prev = None  # (satellite_id, epoch_dt, a_km, inc_deg, mm, ndot, epoch_iso)

    for row in conn.execute(_TLES_IN_WINDOW_SQL, (cutoff,)):
        try:
            a_km, inc_deg, mm, ndot = tle_mean_elements(
                row["line1"], row["line2"])
            epoch_dt = _parse_iso_utc(row["epoch"])
        except (ValueError, IndexError):
            continue  # malformed TLE/epoch: skip the row, never fatal
        cur = (row["satellite_id"], epoch_dt, a_km, inc_deg, mm, ndot,
               row["epoch"])
        if prev is not None and prev[0] == cur[0]:
            dt_days = (epoch_dt - prev[1]).total_seconds() / 86400.0
            if dt_days > 0.0:
                items.extend(_pair_items(row, prev, cur, dt_days))
        prev = cur

    items.sort(key=_sort_magnitude, reverse=True)
    return items[:MAX_RESULTS]


def _pair_items(row, prev, cur, dt_days: float) -> list[dict]:
    """Flagged items (0..2) for one successive TLE pair of one satellite."""
    _sid, _e1, a1, inc1, mm1, ndot1, epoch1_iso = prev
    _sid2, _e2, a2, inc2, _mm2, _ndot2, epoch2_iso = cur
    d_a = a2 - a1
    d_inc = inc2 - inc1
    # Drag-expected SMA change across the interval, from the EARLIER TLE's
    # own drag term: da/dn = -(2/3) a/n (module docstring).
    expected = -(2.0 / 3.0) * (a1 / mm1) * ndot1 * dt_days

    flagged: list[dict] = []

    def item(kind: str) -> dict:
        return {
            "norad_id": row["norad_id"],
            "name": row["name"],
            "kind": kind,
            "delta_sma_km": round(d_a, 3),
            "delta_inc_deg": round(d_inc, 4),
            "expected_drag_dsma_km": round(expected, 3),
            "sma_km": round(a2, 2),
            "from_epoch": epoch1_iso,
            "to_epoch": epoch2_iso,
            "dt_hours": round(dt_days * 24.0, 2),
        }

    if d_a >= SMA_DELTA_KM:
        flagged.append(item("sma_raise"))
    elif d_a <= -SMA_DELTA_KM and d_a < DRAG_TOLERANCE_FACTOR * expected:
        flagged.append(item("sma_drop"))
    if abs(d_inc) >= INC_DELTA_DEG:
        flagged.append(item("inclination"))
    return flagged
