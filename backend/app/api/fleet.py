"""Fleet positions for the dashboard orbital situational display.

GET /fleet/positions batch-propagates every satellite's latest TLE to one
shared epoch (now, floored to the second) with a single SatrecArray call and
returns packed arrays - shape authoritative in docs/CONTRACTS.md ('Fleet
situational display'). Objects whose TLE fails to parse or whose sgp4
propagation errors are skipped rather than failing the response.
"""
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends
from sgp4.api import Satrec, SatrecArray, jday

from backend.app import db
from backend.app.api import conjunctions as conjunctions_api

router = APIRouter()

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

TYPE_CODES = {"PAYLOAD": 0, "DEBRIS": 1, "ROCKET BODY": 2}
UNKNOWN_TYPE_CODE = 3
RISK_CODES = {"nominal": 1, "elevated": 2, "critical": 3}

# States propagated below ~100 km altitude mean the object has effectively
# re-entered (stale TLE past decay) - skip it like an sgp4 error.
_MIN_RADIUS_KM2 = (6378.137 + 100.0) ** 2

# Each satellite joined with its latest TLE by epoch (same pattern as
# api/objects.py / ingestion/screening.py); satellites without a TLE drop out.
_LATEST_TLE_SQL = """
SELECT s.norad_id    AS norad_id,
       s.name        AS name,
       s.object_type AS object_type,
       t.line1       AS line1,
       t.line2       AS line2
FROM satellites s
JOIN tle_history t ON t.id = (
    SELECT t2.id FROM tle_history t2
     WHERE t2.satellite_id = s.id
     ORDER BY t2.epoch DESC, t2.id DESC LIMIT 1)
"""


def _risk_codes(conn, now_iso: str) -> dict[int, int]:
    """norad_id -> max risk code over the object's ACTIVE conjunctions.

    Reuses the conjunction summary builder (and with it the single
    risk-level definition and honest-active predicate - events whose TCA
    already passed never contribute) from api/conjunctions.py; both the
    primary and the secondary side of every active event carry its risk.
    """
    codes: dict[int, int] = {}
    for item in conjunctions_api.query_summaries(
            conn, conjunctions_api.ACTIVE_WHERE, (now_iso,)):
        code = RISK_CODES[item["risk_level"]]
        for side in (item["primary"], item["secondary"]):
            norad_id = side["norad_id"]
            if code > codes.get(norad_id, 0):
                codes[norad_id] = code
    return codes


@router.get("/fleet/positions")
def fleet_positions(conn=Depends(db.get_db)):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    epoch_iso = now.strftime(ISO_FMT)

    sats: list[Satrec] = []
    meta: list[tuple[int, str, int]] = []
    for row in conn.execute(_LATEST_TLE_SQL):
        try:
            sat = Satrec.twoline2rv(row["line1"], row["line2"])
        except Exception:
            continue  # unparseable TLE: skip the object, not the response
        sats.append(sat)
        meta.append((
            row["norad_id"],
            row["name"],
            TYPE_CODES.get(row["object_type"], UNKNOWN_TYPE_CODE),
        ))

    objects: list[list] = []
    if sats:
        jd, fr = jday(now.year, now.month, now.day,
                      now.hour, now.minute, now.second)
        # one batched propagation for the whole catalog at the shared epoch.
        e, r, v = SatrecArray(sats).sgp4(np.array([jd]), np.array([fr]))
        risk = _risk_codes(conn, epoch_iso)
        for i, (norad_id, name, type_code) in enumerate(meta):
            if e[i, 0] != 0 or not np.all(np.isfinite(r[i, 0])):
                continue  # sgp4 propagation error: skip the object
            x, y, z = r[i, 0]
            if x * x + y * y + z * z < _MIN_RADIUS_KM2:
                continue  # below re-entry altitude: decayed object, skip
            vx, vy, vz = v[i, 0]
            objects.append([
                norad_id, name, type_code,
                round(float(x), 3), round(float(y), 3), round(float(z), 3),
                round(float(vx), 4), round(float(vy), 4), round(float(vz), 4),
                risk.get(norad_id, 0),
            ])

    return {"epoch": epoch_iso, "count": len(objects), "objects": objects}
