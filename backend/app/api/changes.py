"""GET /changes - what moved between two ingestion runs (history diff).

Boundaries are refresh_runs.finished_at timestamps of two COMPLETED runs
(finished_at NOT NULL; failed runs still count as boundaries - their
timestamp is real). The default compares the last two completed runs;
`since_run` accepts a run id or the literal 'latest-1'.

Level at a boundary t reuses the live API's effective-Pc definition
(api/conjunctions.effective_pc + risk_level_for, same thresholds), restricted
to data at-or-before t: the latest risk_predictions row <= t, else the latest
CDM's probability_of_collision <= t, else 0.

Activity at a boundary t uses the honest-active rule from api/conjunctions
(first_seen_at <= t AND TCA >= t) - the status column is mutated in place by
the ingest sweep, so it cannot answer historical questions.
"""
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app import db
from backend.app.api import conjunctions as conjunctions_api

router = APIRouter()

_LEVEL_RANK = {"nominal": 0, "elevated": 1, "critical": 2}

_COMPLETED_RUNS_SQL = (
    "SELECT id, finished_at FROM refresh_runs"
    " WHERE finished_at IS NOT NULL ORDER BY id DESC"
)

_PAIR_SQL = """
SELECT c.id                       AS id,
       c.time_of_closest_approach AS tca,
       p.name                     AS primary_name,
       s.name                     AS secondary_name
FROM conjunctions c
JOIN satellites p ON p.id = c.primary_object_id
JOIN satellites s ON s.id = c.secondary_object_id
"""


def _run_info(row) -> dict | None:
    return None if row is None else {
        "run_id": row["id"], "finished_at": row["finished_at"]}


def _pair(row) -> str:
    return f"{row['primary_name']} ↔ {row['secondary_name']}"


def _effective_pc_at(conn, conjunction_id: int, t_iso: str) -> float:
    """effective_pc as of boundary t: predicted, else latest CDM Pc, else 0.

    Mirrors conjunctions.effective_pc exactly, restricted to rows at-or-before
    t - including its quirk that a latest CDM with NULL Pc reads as 0.
    """
    row = conn.execute(
        "SELECT predicted_final_risk FROM risk_predictions"
        " WHERE conjunction_id = ? AND prediction_time <= ?"
        " ORDER BY prediction_time DESC, id DESC LIMIT 1",
        (conjunction_id, t_iso)).fetchone()
    if row is not None and row["predicted_final_risk"] is not None:
        return row["predicted_final_risk"]
    row = conn.execute(
        "SELECT probability_of_collision FROM cdm_messages"
        " WHERE conjunction_id = ? AND message_time <= ?"
        " ORDER BY CASE WHEN source = 'spacetrack' THEN 0 ELSE 1 END,"
        "          message_time DESC, id DESC LIMIT 1",
        (conjunction_id, t_iso)).fetchone()
    if row is not None and row["probability_of_collision"] is not None:
        return row["probability_of_collision"]
    return 0.0


def _empty_diff(since_row, now_row) -> dict:
    return {
        "since": _run_info(since_row),
        "now": _run_info(now_row),
        "new_conjunctions": [],
        "level_changes": [],
        "resolved": [],
        "counts": {"new": 0, "level_changes": 0, "resolved": 0},
    }


_LIST_CAP = 60   # per-list response cap; counts keep true totals


@router.get("/changes")
def get_changes(since_run: str | None = Query(None), conn=Depends(db.get_db)):
    completed = conn.execute(_COMPLETED_RUNS_SQL).fetchall()
    now_row = completed[0] if completed else None

    if since_run is None or since_run == "latest-1":
        since_row = completed[1] if len(completed) > 1 else None
        if since_row is None:  # <2 completed runs: empty-safe, never an error
            return _empty_diff(None, now_row)
    else:
        try:
            since_id = int(since_run)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="since_run must be a run id or 'latest-1'")
        since_row = next((r for r in completed if r["id"] == since_id), None)
        if since_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"No completed refresh run {since_id}")
        if now_row is None:
            return _empty_diff(since_row, None)

    t_since = since_row["finished_at"]
    t_now = now_row["finished_at"]

    # New: first seen inside (since, now]. Full summary items, risk order.
    new_items = conjunctions_api.query_summaries(
        conn, "c.first_seen_at > ? AND c.first_seen_at <= ?", (t_since, t_now))
    new_items.sort(key=conjunctions_api.sort_key)
    conjunctions_api.attach_pc_trends(conn, new_items)

    # Level changes: active at both boundaries, effective level moved.
    level_changes = []
    for row in conn.execute(
            _PAIR_SQL + " WHERE c.first_seen_at <= ?"
            " AND c.time_of_closest_approach >= ?", (t_since, t_now)):
        pc_from = _effective_pc_at(conn, row["id"], t_since)
        pc_to = _effective_pc_at(conn, row["id"], t_now)
        from_level = conjunctions_api.risk_level_for(pc_from)
        to_level = conjunctions_api.risk_level_for(pc_to)
        if from_level == to_level:
            continue
        level_changes.append({
            "conjunction_id": row["id"],
            "pair": _pair(row),
            "from_level": from_level,
            "to_level": to_level,
            "pc_from": pc_from,
            "pc_to": pc_to,
        })
    level_changes.sort(
        key=lambda ch: (-_LEVEL_RANK[ch["to_level"]], -ch["pc_to"]))

    # resolved: active at since, TCA passed before now
    resolved = []
    for row in conn.execute(
            _PAIR_SQL + " WHERE c.first_seen_at <= ?"
            " AND c.time_of_closest_approach >= ?"
            " AND c.time_of_closest_approach < ?", (t_since, t_since, t_now)):
        resolved.append({
            "conjunction_id": row["id"],
            "pair": _pair(row),
            "last_level": conjunctions_api.risk_level_for(
                _effective_pc_at(conn, row["id"], t_now)),
        })
    resolved.sort(key=lambda r: -_LEVEL_RANK[r["last_level"]])

    return {
        "since": _run_info(since_row),
        "now": _run_info(now_row),
        # Lists are CAPPED; counts carry the true totals. A first full-catalog
        # screening once diffed 8,420 new conjunctions - shipping them all
        # made the payload huge and the dashboard panel unscrollable.
        "new_conjunctions": new_items[:_LIST_CAP],
        "level_changes": level_changes[:_LIST_CAP],
        "resolved": resolved[:_LIST_CAP],
        "counts": {
            "new": len(new_items),
            "level_changes": len(level_changes),
            "resolved": len(resolved),
        },
    }
