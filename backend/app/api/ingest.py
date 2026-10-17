"""POST /ingest/refresh - run the ingestion pipeline synchronously (dev-only).

The route is gated on KESSLER_DEV_ENDPOINTS (truthy '1'/'true'/'yes'/'on'):
without it the endpoint answers 404 as if it did not exist. The env is read
per request, never cached at import.

`ingestion.pipeline` is imported lazily inside the handler so a concurrently
built module can never break API import order. A refresh already holding the
pipeline lock surfaces as a 409; any other pipeline failure surfaces as a
502 with the error in `detail`.
"""
import os

from fastapi import APIRouter, Depends, HTTPException

from backend.app import db

router = APIRouter()


def _dev_endpoints_enabled() -> bool:
    """KESSLER_DEV_ENDPOINTS truthiness ('1'/'true'/'yes'/'on', case-insens).

    Read per request so tests (and ops) can flip the gate without a reload.
    """
    raw = os.environ.get("KESSLER_DEV_ENDPOINTS", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


@router.post("/ingest/refresh")
def refresh(demo: bool = False, conn=Depends(db.get_db)):
    """Dev-only (see module docstring): 404 unless KESSLER_DEV_ENDPOINTS."""
    if not _dev_endpoints_enabled():
        raise HTTPException(status_code=404, detail="Not found")
    try:
        from backend.app.ingestion import pipeline
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ingest failed: {exc}")
    try:
        return pipeline.refresh(conn, demo=demo)
    except pipeline.RefreshBusyError:
        raise HTTPException(status_code=409,
                            detail="a refresh is already running")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ingest failed: {exc}")
