"""GET /anomalies - TLE element-delta maneuver candidates.

Thin route over models.anomaly.scan_tle_deltas. The router include lives in
main.py; this module only defines the route. `hours` is clamped to 6..168 - clamped, not
rejected, per contract - and the scan itself caps results at 50.
"""
from fastapi import APIRouter, Depends, Query

from backend.app import db
from backend.app.models import anomaly

router = APIRouter()

HOURS_MIN = 6
HOURS_MAX = 168

METHOD_LABEL = "element-delta heuristic — TLE-derived, not confirmed maneuvers"


@router.get("/anomalies")
def get_anomalies(hours: int = Query(48), conn=Depends(db.get_db)):
    clamped = max(HOURS_MIN, min(HOURS_MAX, hours))
    items = anomaly.scan_tle_deltas(conn, since_hours=clamped)
    return {
        "hours": clamped,
        "count": len(items),
        "method": METHOD_LABEL,
        "items": items,
    }
