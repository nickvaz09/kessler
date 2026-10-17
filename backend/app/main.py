"""Kessler API application.

Startup initializes the SQLite schema; API routers are registered first and
the static frontend is mounted at '/' last so API routes always win.
Run from the repo root: `.venv/bin/uvicorn backend.app.main:app`.

Auto-refresh scheduler: when KESSLER_REFRESH_INTERVAL_H is set (float
hours > 0), lifespan starts one daemon thread running scheduler_loop - the
first refresh happens one FULL interval after boot (never at boot: dev
reloads must not double-ingest), each run takes the module refresh lock (a
tick is skipped, not queued, while a run is in flight), pipeline.refresh
writes the normal refresh_runs history rows, and errors go to stderr without
ever crashing the app. Lifespan shutdown sets the stop event; the loop exits
at its next wake-up (the thread is daemonized, so exit never blocks on it).

Watch Mode: with KESSLER_WATCH_MODE=1, each successful scheduled
refresh is followed - same thread, same connection - by one AI change-of-
watch pass (assistant.service.watch_pass) that logs a shift entry to the
watchlog table. Default OFF; never invoked at import or server start.
"""
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app import config, db, snapshot
from backend.app.api import (anomalies, assistant, changes, conjunctions,
                             fleet, ingest, model, objects, reentry, risk,
                             status, user_objects, watchlog)

# One auto-refresh at a time, process-wide. Module-level so every scheduled
# tick in this process contends on the same lock.
_refresh_lock = threading.Lock()


def _watch_pass_after_refresh(conn) -> None:
    """Watch Mode: one AI change-of-watch log entry, gated on
    KESSLER_WATCH_MODE=1 (default OFF - a pass burns real API tokens).
    Called only after a successful scheduled refresh, in the same thread -
    never at import, never at server start. watch_pass itself swallows
    every failure."""
    if os.environ.get("KESSLER_WATCH_MODE") != "1":
        return
    from backend.app.assistant import service  # lazy, like pipeline below

    service.watch_pass(conn)


def _run_scheduled_refresh() -> None:
    """One locked pipeline run on its own connection; errors never propagate.

    Skips (with a stderr note) when a previous run is still in flight -
    a scheduled one holding this wrapper's lock, or an API/tool-triggered
    one holding pipeline.refresh's module lock (RefreshBusyError).
    pipeline.refresh owns the refresh_runs bookkeeping, including the failure
    row - this wrapper only keeps the scheduler thread alive.
    """
    if not _refresh_lock.acquire(blocking=False):
        print("[scheduler] previous refresh still in flight — tick skipped",
              file=sys.stderr)
        return
    try:
        from backend.app.ingestion import pipeline  # lazy, like api/ingest.py

        conn = db.get_conn()
        try:
            try:
                pipeline.refresh(conn)
            except pipeline.RefreshBusyError:
                print("[scheduler] a refresh is already running elsewhere —"
                      " tick skipped", file=sys.stderr)
                return
            # Reached only when the refresh succeeded - a raise above lands
            # in the scheduler's error handler and skips the watch pass.
            _watch_pass_after_refresh(conn)
        finally:
            conn.close()
    except Exception as exc:
        print(f"[scheduler] auto-refresh failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
    finally:
        _refresh_lock.release()


def scheduler_loop(interval_h: float, stop_event: threading.Event,
                   refresh_fn=_run_scheduled_refresh) -> None:
    """Auto-refresh loop: wait one FULL interval, refresh, repeat.

    stop_event both paces the loop and ends it - Event.wait(interval) returns
    True the moment shutdown sets the event (immediate exit, even mid-wait)
    and False on timeout (time to refresh). The first refresh therefore comes
    one full interval after start, never at boot. refresh_fn is injectable so
    tests drive the loop with a stub and a tiny interval.
    """
    interval_s = interval_h * 3600.0
    while not stop_event.wait(interval_s):
        refresh_fn()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = db.get_conn()
    try:
        db.init_db(conn)
        snapshot.load_on_startup(conn)  # KESSLER_SNAPSHOT, empty catalog only
    finally:
        conn.close()
    from backend.app import spaceweather
    spaceweather.warm()  # first NOAA fetch in the background (never blocks startup)
    stop_event = threading.Event()
    interval_h = config.refresh_interval_h()
    if interval_h:
        threading.Thread(
            target=scheduler_loop, args=(interval_h, stop_event),
            name="kessler-auto-refresh", daemon=True).start()
    try:
        yield
    finally:
        stop_event.set()  # clean shutdown: the loop exits at its next wake-up


app = FastAPI(
    title="Kessler API", version="13.0",
    description="Conjunction screening, ML risk scoring and the Kessler AI"
                " watch officer. Model provenance: /model/card.",
    lifespan=lifespan)

# Large payloads (fleet positions, ground tracks) compress ~5x; small
# responses pass through untouched.
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.include_router(status.router)
app.include_router(objects.router)
app.include_router(conjunctions.router)
app.include_router(risk.router)
app.include_router(ingest.router)
app.include_router(fleet.router)
app.include_router(assistant.router)
app.include_router(changes.router)
app.include_router(user_objects.router)
app.include_router(reentry.router)
app.include_router(anomalies.router)
app.include_router(model.router)
app.include_router(watchlog.router)
app.include_router(__import__("backend.app.api.scenarios", fromlist=["router"]).router)
app.include_router(__import__("backend.app.api.spaceweather", fromlist=["router"]).router)

# mounted last: everything not matched by an API route serves the frontend
class _RevalidatedStaticFiles(StaticFiles):
    """no-cache on code assets (html/js/css): the browser revalidates each
    load (a 304 when unchanged), so a deploy is never demoed against a stale
    cached build. Heavy immutable media (textures, fonts) stays cacheable."""

    _REVALIDATE = (".html", ".js", ".css")

    def file_response(self, full_path, *args, **kwargs):
        resp = super().file_response(full_path, *args, **kwargs)
        if str(full_path).endswith(self._REVALIDATE):
            resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/", _RevalidatedStaticFiles(
    directory=str(config.FRONTEND_DIR), html=True))
