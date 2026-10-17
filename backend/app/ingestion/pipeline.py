"""Ingestion pipeline: fetch catalog -> screen -> record -> predict.

CLI: python -m backend.app.ingestion.pipeline [--demo] [--full-catalog]
Prints the refresh stats dict as JSON.

Every refresh() call writes a refresh_runs history row: the start row is
committed immediately (so an in-flight run is visible to /status and
/changes), and finished_at + stats_json land on completion - including on
failure, where stats_json is {"error": str} and the exception still
propagates to the caller.

models.pc / models.risk_model are imported lazily inside functions, so module
import never fails on their absence.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import httpx

from backend.app import config, db
from backend.app.ingestion import celestrak, screening
from backend.app.ingestion.celestrak import ISO_FMT

DEMO_LOOKBACK_H = 48.0
DEMO_STEP_H = 12.0
DEMO_MIN_CDMS = 3


class RefreshBusyError(RuntimeError):
    """refresh() found another refresh in flight - no run row was written."""


# One refresh at a time, process-wide: the API endpoint, the scheduler and
# the assistant tool all contend on this module lock.
_refresh_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FMT)


def _parse_iso(s: str) -> datetime:
    return datetime.strptime(s, ISO_FMT).replace(tzinfo=timezone.utc)


def _start_run(conn) -> int:
    """INSERT and commit the refresh_runs start row; returns its id.

    Committed immediately so a concurrent reader (/status, /changes) sees the
    in-flight run (finished_at NULL) while the refresh is still working.
    """
    cur = conn.execute(
        "INSERT INTO refresh_runs (started_at) VALUES (?)", (_now_iso(),))
    run_id = cur.lastrowid
    conn.commit()
    return run_id


def _finish_run(conn, run_id: int, stats: dict) -> None:
    """Stamp finished_at + stats_json on the run row (success and failure)."""
    conn.execute(
        "UPDATE refresh_runs SET finished_at = ?, stats_json = ? WHERE id = ?",
        (_now_iso(), json.dumps(stats), run_id))
    conn.commit()


def refresh(conn, demo: bool = False, full_catalog: bool | None = None) -> dict:
    """Run one full ingestion cycle. Returns the stats dict (see CONTRACTS.md).

    full_catalog: True/False forces the Space-Track full-catalog upsert on/off
    (CLI --full-catalog); None resolves the KESSLER_FULL_CATALOG env switch
    via config.full_catalog_enabled().

    History: wraps the cycle in a refresh_runs row - start row committed
    before any work, finished_at + stats_json on completion, and on exception
    the row is finished with {"error": str} before the exception propagates.

    Concurrency: the module refresh lock is taken non-blocking; a call that
    finds another refresh in flight raises RefreshBusyError before any
    refresh_runs row is written.
    """
    if not _refresh_lock.acquire(blocking=False):
        raise RefreshBusyError("a refresh is already running")
    try:
        run_id = _start_run(conn)
        try:
            stats = _run_cycle(conn, demo=demo, full_catalog=full_catalog)
        except Exception as exc:
            try:
                conn.rollback()  # drop any half-written step before the run row
            except Exception:
                pass
            _finish_run(conn, run_id, {"error": str(exc)})
            raise
        _finish_run(conn, run_id, stats)
        return stats
    finally:
        _refresh_lock.release()


def _run_cycle(conn, demo: bool, full_catalog: bool | None) -> dict:
    """The ingestion cycle proper (refresh() owns the refresh_runs bookkeeping)."""
    t0 = time.monotonic()

    # 1. Fetch every catalog group once; dedupe by norad keeping newest epoch.
    # Per-group failures are skipped (CelesTrak renames groups and 403s the
    # big ones for scripted clients) - one dead group must never kill the
    # refresh. A total CelesTrak outage only aborts when the Space-Track
    # full catalog isn't there to carry the cycle.
    per_group: dict[str, list[dict]] = {}
    group_failures: list[str] = []
    with httpx.Client(timeout=config.HTTP_TIMEOUT_S, follow_redirects=True) as client:
        for group in config.CATALOG_GROUPS:
            try:
                per_group[group] = celestrak.fetch_group(group, client=client)
            except httpx.HTTPError as exc:
                group_failures.append(group)
                print(f"pipeline: celestrak group {group!r} failed, "
                      f"skipping: {exc}", file=sys.stderr)
    if full_catalog is None:
        full_catalog = config.full_catalog_enabled()
    if group_failures and not per_group and not full_catalog:
        raise RuntimeError(
            "celestrak: every catalog group failed and the Space-Track "
            f"full catalog is disabled (failed: {', '.join(group_failures)})")
    records = celestrak.merge_records(
        rec for recs in per_group.values() for rec in recs)
    stats = celestrak.upsert_catalog(conn, records)
    stats["group_failures"] = group_failures

    # 1b. Optional Space-Track full-catalog upsert (KESSLER_FULL_CATALOG env or
    # CLI --full-catalog; resolved above so a total CelesTrak outage knows
    # whether Space-Track carries the cycle). Runs right after the CelesTrak
    # upsert so the full object set participates in screening as secondaries.
    full_catalog_count = _upsert_full_catalog(conn) if full_catalog else 0

    # 2. Primaries = norad ids fetched from config.PRIMARY_GROUPS this run.
    primaries = {rec["norad_id"]
                 for group in config.PRIMARY_GROUPS
                 for rec in per_group.get(group, [])}

    # 3. Screen and record conjunction events.
    now_iso = _now_iso()
    events = screening.screen(conn, primaries)
    rec_stats = screening.record_events(conn, events, now_iso)
    cdms_added = rec_stats["cdms_added"]

    # 3b. Space-Track CDM enrichment - only when both credential env vars are
    # set; lazy import, never fatal (failures go to stderr, the stat stays 0).
    spacetrack_cdms = 0
    if os.environ.get("SPACETRACK_IDENTITY") and os.environ.get("SPACETRACK_PASSWORD"):
        try:
            from backend.app.ingestion import spacetrack
            st_stats = spacetrack.refresh_cdms(conn)
            spacetrack_cdms = st_stats["cdms_added"]
        except Exception as exc:
            # Postgres aborts the transaction on any failed statement; without
            # a rollback every later statement in this run would fail too.
            conn.rollback()
            print(f"[pipeline] Space-Track CDM ingestion failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)

    # 4. Optional demo CDM backfill.
    if demo:
        cdms_added += backfill_demo_cdms(conn)

    # 5. Mark conjunctions whose TCA has passed.
    conn.execute(
        "UPDATE conjunctions SET status = 'past'"
        " WHERE status = 'active' AND time_of_closest_approach < ?",
        (now_iso,))
    conn.commit()

    # 6. Risk predictions for every active conjunction.
    predictions_made = _predict_active(conn)

    return {
        "fetched": stats["fetched"],
        "satellites_new": stats["satellites_new"],
        "tles_added": stats["tles_added"],
        "full_catalog": full_catalog_count,
        "conjunctions_new": rec_stats["conjunctions_new"],
        "cdms_added": cdms_added,
        "spacetrack_cdms": spacetrack_cdms,
        "predictions_made": predictions_made,
        "duration_s": round(time.monotonic() - t0, 2),
    }


def _upsert_full_catalog(conn) -> int:
    """Space-Track full-catalog upsert; returns records upserted (0 = skipped).

    Behind the same credentials gate as the CDM path (SPACETRACK_IDENTITY +
    SPACETRACK_PASSWORD); spacetrack is imported lazily.
    Skips with a stderr note when creds are absent; fetch/upsert failures go
    to stderr and are never fatal to the refresh.
    """
    if not (os.environ.get("SPACETRACK_IDENTITY")
            and os.environ.get("SPACETRACK_PASSWORD")):
        print("[pipeline] full catalog requested but SPACETRACK_IDENTITY /"
              " SPACETRACK_PASSWORD are not set — skipping", file=sys.stderr)
        return 0
    try:
        from backend.app.ingestion import spacetrack
        records = spacetrack.fetch_full_catalog()
        return celestrak.upsert_catalog(conn, records)["fetched"]
    except Exception as exc:
        conn.rollback()  # Postgres: keep the rest of the run usable
        print(f"[pipeline] Space-Track full-catalog upsert failed: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 0


def _predict_active(conn) -> int:
    """Predict final risk (and VIGIL-E escalation, when trained) for every
    active conjunction (lazy model imports)."""
    try:
        from backend.app.models import risk_model
    except Exception:
        return 0
    try:
        model = risk_model.RiskModel.load()
    except Exception:
        model = None
    # VIGIL-E early warning: scored alongside when its gated artifact exists;
    # None (no artifact) leaves both escalation columns null.
    try:
        from backend.app.models import escalation
        escalation_model = escalation.EscalationModel.load()
    except Exception:
        escalation_model = None
    made = 0
    ids = [row["id"] for row in conn.execute(
        "SELECT id FROM conjunctions WHERE status = 'active'").fetchall()]
    for conj_id in ids:
        try:
            risk_model.predict_for_conjunction(conn, conj_id, model,
                                               escalation_model)
            made += 1
        except Exception:
            conn.rollback()  # Postgres: one bad event must not abort the rest
            continue
    return made


def _compute_pc(miss_distance_m: float, relative_velocity_ms: float,
                propagation_age_days: float):
    """Lazy bridge to models.pc.compute_pc; None if it is missing."""
    try:
        from backend.app.models import pc as pc_mod
    except Exception:
        return None
    try:
        return pc_mod.compute_pc(miss_distance_m, relative_velocity_ms,
                                 propagation_age_days=propagation_age_days)
    except Exception:
        return None


def backfill_demo_cdms(conn) -> int:
    """Synthesize backdated CDM sequences for sparse active conjunctions.

    For active conjunctions with fewer than 3 CDMs: append a sequence every
    ~12 h from first_seen-48h to first_seen, derived from the REAL screened
    geometry (miss distance = final + decaying offset, deterministic per-
    conjunction seed). Pc recomputed per message via models.pc.compute_pc with
    the sigma implied by each message's propagation age. source='demo';
    'screening' rows are never overwritten (INSERT only). Returns rows added.
    """
    added = 0
    rows = conn.execute(
        "SELECT id, first_seen_at, time_of_closest_approach"
        " FROM conjunctions WHERE status = 'active'").fetchall()
    for row in rows:
        conj_id = row["id"]
        n_cdms = conn.execute(
            "SELECT COUNT(*) AS n FROM cdm_messages WHERE conjunction_id = ?",
            (conj_id,)).fetchone()["n"]
        if n_cdms >= DEMO_MIN_CDMS:
            continue
        latest = conn.execute(
            "SELECT miss_distance_m, relative_velocity_ms FROM cdm_messages"
            " WHERE conjunction_id = ? AND source = 'screening'"
            " ORDER BY message_time DESC LIMIT 1",
            (conj_id,)).fetchone()
        if latest is None:
            continue
        final_md = latest["miss_distance_m"]
        final_rv = latest["relative_velocity_ms"]
        first_seen = _parse_iso(row["first_seen_at"])
        tca = _parse_iso(row["time_of_closest_approach"])

        rng = random.Random(conj_id)  # deterministic per-conjunction sequence
        amplitude = rng.uniform(0.8, 3.0) * max(final_md, 250.0)
        n_msgs = int(DEMO_LOOKBACK_H / DEMO_STEP_H)
        for k in range(n_msgs):
            age_h = DEMO_LOOKBACK_H - DEMO_STEP_H * k + rng.uniform(-1.0, 1.0)
            msg_time = first_seen - timedelta(hours=age_h)
            frac = max(age_h, 0.0) / DEMO_LOOKBACK_H
            miss = final_md + amplitude * (frac ** 1.3) * (1.0 + 0.08 * rng.uniform(-1.0, 1.0))
            miss = max(miss, 1.0)
            rel_v = max(final_rv * (1.0 + 0.01 * rng.uniform(-1.0, 1.0)), 0.0)
            age_days = max((tca - msg_time).total_seconds(), 0.0) / 86400.0
            pc = _compute_pc(miss, rel_v, age_days)
            conn.execute(
                "INSERT INTO cdm_messages (conjunction_id, message_time, miss_distance_m,"
                " relative_velocity_ms, probability_of_collision, source)"
                " VALUES (?, ?, ?, ?, ?, 'demo')",
                (conj_id, msg_time.strftime(ISO_FMT), miss, rel_v, pc))
            added += 1
    conn.commit()
    return added


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the Kessler ingestion pipeline.")
    parser.add_argument("--demo", action="store_true",
                        help="backfill synthetic demo CDM sequences")
    parser.add_argument("--full-catalog", action="store_true",
                        help="also upsert the Space-Track full catalog"
                             " (Space-Track credentials required)")
    args = parser.parse_args(argv)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = db.get_conn()
    try:
        db.init_db(conn)
        # Flag absent -> None: refresh() falls back to KESSLER_FULL_CATALOG.
        stats = refresh(conn, demo=args.demo,
                        full_catalog=True if args.full_catalog else None)
    finally:
        conn.close()
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
