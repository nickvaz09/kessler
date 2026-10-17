"""GET /spaceweather - live solar and geomagnetic conditions (NOAA SWPC) and
what they do to the upper atmosphere; GET /spaceweather/orbit - what they do
right now to one circular orbit; GET /spaceweather/impact - the active close
approaches the current storm widens. Live data only: no scenarios.

Plain `def` handlers: FastAPI runs them in its threadpool, so the NOAA
fetches (cached in app.spaceweather) never block the event loop.
"""
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app import db
from backend.app import spaceweather as sw
from backend.app.api import conjunctions as conjunctions_api

router = APIRouter()


@router.get("/spaceweather")
def get_spaceweather():
    cached = sw.peek()                       # stale data now, refresh behind
    data = dict(cached if cached is not None else sw.conditions())
    data["summary"] = sw.summary(data)
    data.pop("storm", None)                  # the risk layer's table, not for drawing
    return data


@router.get("/spaceweather/orbit")
def get_orbit_effects(
    alt_km: float = Query(550, ge=200, le=1000, description="circular orbit altitude"),
    mass_kg: float = Query(4.0, gt=0, le=500000),
    area_m2: float = Query(0.03, gt=0, le=5000, description="mean cross-section"),
    cd: float = Query(2.2, gt=0, le=5),
):
    """What the current storm does to a circular orbit of this ballistic
    coefficient: density vs a quiet day, decay rate, along-track error."""
    c = sw.peek() or sw.conditions()
    kp = (c.get("kp") or {}).get("now")
    f107 = (c.get("f107") or {}).get("now")
    f107a = (c.get("f107") or {}).get("mean") or f107
    if kp is None or f107 is None:
        raise HTTPException(503, "live space-weather conditions are unavailable")
    out = sw.orbit_effects(alt_km, cd * area_m2 / mass_kg, kp, f107, f107a)
    if out.get("model") is None:
        raise HTTPException(503, "the atmosphere model is unavailable on this server")
    return out


@router.get("/spaceweather/impact")
def get_impact(limit: int = Query(5, ge=0, le=50), conn=Depends(db.get_db)):
    """What the current storm does to the active close approaches:
    {"available", "updated", "active", "widened", "flagged", "items"} -
    items are the flagged pairs ('watch' first, then the highest storm-time
    Pc). available=false (zeros, no items) without live data."""
    ctx = sw.storm_context()
    if ctx is None:
        return {"available": False, "updated": None, "active": 0, "widened": 0,
                "flagged": 0, "items": []}
    items = conjunctions_api.active_storm_items(conn, ctx)
    flagged = conjunctions_api.storm_flagged(items)
    return {"available": True, "updated": ctx.updated, "active": len(items),
            "widened": sum(1 for i in items if (i.get("space_weather") or {}).get("widened")),
            "flagged": len(flagged), "items": flagged[:limit]}
