"""GET /status - headline metrics for the dashboard stat strip."""
import json

from fastapi import APIRouter, Depends

from backend.app import config, db, snapshot
from backend.app.api import conjunctions as conjunctions_api

router = APIRouter()


def _model_version(conn) -> str:
    """Version of the latest prediction; else model artifact meta; else heuristic."""
    row = conn.execute(
        "SELECT model_version FROM risk_predictions"
        " ORDER BY prediction_time DESC, id DESC LIMIT 1"
    ).fetchone()
    if row is not None:
        return row[0]
    try:
        if config.RISK_MODEL_META_FILE.exists():
            meta = json.loads(config.RISK_MODEL_META_FILE.read_text())
            return meta.get("version", config.HEURISTIC_MODEL_VERSION)
    except Exception:
        pass
    return config.HEURISTIC_MODEL_VERSION


def _space_weather(conn, active_items) -> dict | None:
    """app.spaceweather.summary() plus what it does to the active close
    approaches (widened / flagged counts, conjunctions_api.active_storm_items
    - cached per NOAA refresh); null when there is no live data."""
    try:
        from backend.app import spaceweather as sw
        out = sw.summary()
        ctx = sw.storm_context() if out else None
    except Exception:
        return None
    if out is None:
        return None
    if ctx is not None:
        items = conjunctions_api.active_storm_items(conn, ctx, active_items)
        layer = [i.get("space_weather") or {} for i in items]
        out["widened_conjunctions"] = sum(1 for x in layer if x.get("widened"))
        out["flagged_conjunctions"] = sum(1 for x in layer if x.get("flag"))
    return out


@router.get("/status")
def get_status(conn=Depends(db.get_db)):
    catalog_size = conn.execute("SELECT COUNT(*) FROM satellites").fetchone()[0]

    active_items = conjunctions_api.query_summaries(
        conn, conjunctions_api.ACTIVE_WHERE, (conjunctions_api.now_iso(),))
    levels = [item["risk_level"] for item in active_items]

    last_ingest_at = conn.execute(
        "SELECT MAX(ingested_at) FROM tle_history"
    ).fetchone()[0]

    has_demo = conn.execute(
        "SELECT EXISTS(SELECT 1 FROM cdm_messages WHERE source = 'demo')"
    ).fetchone()[0]

    # Most recent refresh_runs row - running or finished; null before any run.
    last_run_id = conn.execute(
        "SELECT MAX(id) FROM refresh_runs").fetchone()[0]

    # surface a failed refresh so an empty board never reads as all-clear
    last_run_error = None
    if last_run_id is not None:
        row = conn.execute(
            "SELECT stats_json FROM refresh_runs WHERE id = ?",
            (last_run_id,)).fetchone()
        try:
            stats = json.loads(row[0]) if row and row[0] else {}
            last_run_error = stats.get("error") if isinstance(stats, dict) else None
        except (TypeError, ValueError):
            pass

    return {
        "catalog_size": catalog_size,
        "active_conjunctions": len(active_items),
        "elevated_count": levels.count("elevated"),
        "critical_count": levels.count("critical"),
        "last_ingest_at": last_ingest_at,
        "model_version": _model_version(conn),
        "window_hours": config.SCREENING_WINDOW_HOURS,
        "has_demo_data": bool(has_demo),
        "auto_refresh_interval_h": config.refresh_interval_h(),
        "last_run_id": last_run_id,
        "last_run_error": last_run_error,
        # offline-snapshot provenance (backend/app/snapshot.py); null unless
        # the data was loaded from a snapshot and no live refresh has since
        # completed. last_ingest_at above keeps the snapshot's true age.
        "snapshot": snapshot.snapshot_status(conn),
        # live space weather (app.spaceweather): storm level, flare, what it
        # does to LEO air density, and the close approaches it widens
        "space_weather": _space_weather(conn, active_items),
    }
