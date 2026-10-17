"""Watch Mode endpoints.

GET  /watchlog      - newest-first AI change-of-watch entries.
POST /watchlog/run  - run one watch pass NOW (the START SHIFT demo path).
                      COSTS REAL API TOKENS; 503 with detail when no key.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app import db
from backend.app.api import assistant as assistant_api
from backend.app.assistant import service

router = APIRouter()

RUN_FAIL_DETAIL = (
    "Watch pass unavailable: no Anthropic API key (set ANTHROPIC_API_KEY or "
    "`ant auth login` and restart) or the Claude call failed — see server "
    "logs."
)

# run-now throttle: the SHIFT BRIEF flow only POSTs when the log is EMPTY,
# so a recent entry means this is an anonymous re-run burning real tokens.
WATCH_RUN_MIN_AGE_S = 1800


def _entry_out(row_or_entry) -> dict:
    """Entry shape + spoken_sig so the drawer can play it via /assistant/tts
    (synthesis is restricted to server-signed text)."""
    e = dict(row_or_entry)
    e["spoken_sig"] = assistant_api.tts_sign(e.get("spoken") or "")
    return e


@router.get("/watchlog")
def get_watchlog(limit: int = Query(10), conn=Depends(db.get_db)):
    limit = max(1, min(50, limit))  # contract: default 10, cap 50
    rows = conn.execute(
        "SELECT id, created_at, model, summary_md, spoken FROM watchlog"
        " ORDER BY created_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return {"entries": [_entry_out(row) for row in rows]}


@router.post("/watchlog/run")
def run_watchlog(conn=Depends(db.get_db)):
    """Run one watch pass immediately and return the new entry.

    COSTS REAL API TOKENS: one Claude agentic tool loop (up to 6 rounds,
    effort 'low' unless KESSLER_AI_EFFORT overrides) per call. Dev/demo
    trigger only - scheduled entries come from the KESSLER_WATCH_MODE=1
    hook after each auto-refresh.
    """
    row = conn.execute(
        "SELECT created_at FROM watchlog ORDER BY created_at DESC, id DESC"
        " LIMIT 1").fetchone()
    if row is not None:
        try:
            age = (datetime.now(timezone.utc) - datetime.strptime(
                row["created_at"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)).total_seconds()
        except ValueError:
            age = None
        if age is not None and age < WATCH_RUN_MIN_AGE_S:
            raise HTTPException(
                status_code=429,
                detail=f"Latest watch entry is {int(age // 60)} minutes old — "
                       "reuse it (a pass costs real model tokens).",
            )
    entry = service.watch_pass(conn)
    if entry is None:
        # watch_pass never raises - no key and API failures both surface
        # here as the keyless 503 degrade (detail rendered by the drawer).
        raise HTTPException(status_code=503, detail=RUN_FAIL_DETAIL)
    return _entry_out(entry)
