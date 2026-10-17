"""history + scheduler tests: refresh_runs lifecycle, /changes diff,
/status scheduler keys, and the main.py scheduler loop.

Hermetic - temp DB, offline CelesTrak, no network, and no real sleep loops:
the scheduler tests drive the loop function directly with an injected stop
event and a millisecond interval.
"""
import json
import sys
import threading
import time
import types

import pytest
from fastapi.testclient import TestClient

from backend.app import config, db
from backend.app.ingestion import celestrak, pipeline, screening

# Real ISS TLE (fixture string only - nothing here propagates it).
ISS_TLE1 = "1 25544U 98067A   24007.58333333  .00016717  00000-0  30777-3 0  9994"
ISS_TLE2 = "2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.49560532430089"

# Mirror of the refresh_runs schema in db.py. CREATE TABLE IF NOT EXISTS
# makes this a no-op once init_db has created the table.
REFRESH_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS refresh_runs (
    id          INTEGER PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    stats_json  TEXT
);
"""

# conjunction summary item shape; asserted as a subset because later fields
# (pc_source) are added to the same items.
SUMMARY_KEYS = {
    "id", "tca", "primary", "secondary", "miss_distance_m",
    "relative_velocity_ms", "pc_latest", "predicted_final_risk",
    "model_version", "risk_level", "cdm_count", "first_seen_at", "status",
    "pc_trend",
}

T1 = "2026-07-21T00:00:00Z"  # run 1 finished_at (the 'since' boundary)
T2 = "2026-07-22T00:00:00Z"  # run 2 finished_at (the 'now' boundary)

_ENV_VARS = (
    "SPACETRACK_IDENTITY", "SPACETRACK_PASSWORD", "KESSLER_FULL_CATALOG",
    "KESSLER_REFRESH_INTERVAL_H", "KESSLER_DATABASE_URL",
)


def _clear_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _make_conn(path):
    conn = db.get_conn(path)
    db.init_db(conn)
    conn.executescript(REFRESH_RUNS_DDL)
    conn.commit()
    return conn


@pytest.fixture()
def offline(monkeypatch):
    """No network, no credentialed side paths, sqlite backend."""
    _clear_env(monkeypatch)
    monkeypatch.setattr(celestrak, "fetch_group", lambda group, client=None: [])


# --- refresh_runs lifecycle ---------------------------------------------------

def test_refresh_writes_start_row_immediately_then_finishes(
        tmp_path, offline, monkeypatch):
    conn = _make_conn(tmp_path / "runs.db")
    seen = {}

    def probe_screen(c, primaries, **kwargs):
        # Mid-run: the start row must already be COMMITTED - a second
        # connection sees it with finished_at still NULL.
        other = db.get_conn(tmp_path / "runs.db")
        try:
            row = other.execute(
                "SELECT started_at, finished_at FROM refresh_runs").fetchone()
            seen["started"] = row is not None and row["started_at"] is not None
            seen["unfinished"] = row is not None and row["finished_at"] is None
        finally:
            other.close()
        return []

    monkeypatch.setattr(screening, "screen", probe_screen)
    stats = pipeline.refresh(conn)

    assert seen == {"started": True, "unfinished": True}
    rows = conn.execute(
        "SELECT started_at, finished_at, stats_json FROM refresh_runs"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["finished_at"] is not None
    assert rows[0]["finished_at"] >= rows[0]["started_at"]
    assert json.loads(rows[0]["stats_json"]) == stats
    assert stats["full_catalog"] == 0       # switch off by default
    assert stats["spacetrack_cdms"] == 0    # no creds
    conn.close()


def test_refresh_failure_finishes_run_row_with_error(
        tmp_path, offline, monkeypatch):
    conn = _make_conn(tmp_path / "fail.db")

    def boom(group, client=None):
        raise RuntimeError("celestrak unreachable")

    monkeypatch.setattr(celestrak, "fetch_group", boom)
    with pytest.raises(RuntimeError, match="celestrak unreachable"):
        pipeline.refresh(conn)

    row = conn.execute(
        "SELECT started_at, finished_at, stats_json FROM refresh_runs"
    ).fetchone()
    assert row is not None and row["finished_at"] is not None
    assert json.loads(row["stats_json"]) == {"error": "celestrak unreachable"}
    conn.close()


# --- full-catalog switch ------------------------------------------------------

def test_full_catalog_skipped_without_creds(tmp_path, offline, capsys):
    conn = _make_conn(tmp_path / "fc.db")
    stats = pipeline.refresh(conn, full_catalog=True)
    assert stats["full_catalog"] == 0
    assert "full catalog" in capsys.readouterr().err
    conn.close()


def test_full_catalog_env_switch_upserts_spacetrack_records(
        tmp_path, offline, monkeypatch):
    monkeypatch.setenv("SPACETRACK_IDENTITY", "ops_user")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "hunter2-secret")
    monkeypatch.setenv("KESSLER_FULL_CATALOG", "1")

    from backend.app.ingestion import spacetrack
    records = {90001: {
        "name": "FULLCAT OBJ", "object_type": "PAYLOAD",
        "intl_designator": None, "epoch": "2026-07-20T00:00:00Z",
        "line1": ISS_TLE1, "line2": ISS_TLE2,
    }}
    # raising=False: tolerate a spacetrack module without fetch_full_catalog
    monkeypatch.setattr(spacetrack, "fetch_full_catalog",
                        lambda client=None: records, raising=False)
    monkeypatch.setattr(spacetrack, "refresh_cdms",
                        lambda conn: {"cdms_added": 0})

    conn = _make_conn(tmp_path / "fcenv.db")
    stats = pipeline.refresh(conn)  # full_catalog=None -> env switch
    assert stats["full_catalog"] == 1
    row = conn.execute(
        "SELECT name FROM satellites WHERE norad_id = 90001").fetchone()
    assert row is not None and row["name"] == "FULLCAT OBJ"
    conn.close()


def test_cli_full_catalog_flag(tmp_path, offline, monkeypatch, capsys):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "cli.db")
    _make_conn(tmp_path / "cli.db").close()  # pre-create v6 tables

    pipeline.main(["--full-catalog"])
    captured = capsys.readouterr()
    stats = json.loads(captured.out)
    assert stats["full_catalog"] == 0        # creds absent -> skipped...
    assert "full catalog" in captured.err    # ...with the stderr note


# --- /changes -----------------------------------------------------------------

def _seed_history(conn):
    conn.executescript(f"""
    INSERT INTO refresh_runs (id, started_at, finished_at, stats_json) VALUES
        (1, '2026-07-20T23:50:00Z', '{T1}', '{{}}'),
        (2, '2026-07-21T23:50:00Z', '{T2}', '{{}}'),
        (3, '2026-07-22T10:00:00Z', NULL, NULL);

    INSERT INTO satellites (id, norad_id, name, object_type, last_updated) VALUES
        (1, 25544, 'ISS (ZARYA)', 'PAYLOAD', '{T2}'),
        (2, 34454, 'COSMOS 2251 DEB', 'DEBRIS', '{T2}'),
        (3, 43556, 'AEROCUBE 12A', 'PAYLOAD', '{T2}');

    -- 10 NEW (first seen between the runs)          tca far future
    -- 11 LEVEL CHANGE via predictions               elevated -> critical
    -- 12 RESOLVED (TCA passed between the runs)
    -- 13 STABLE (nominal at both boundaries)        must appear nowhere
    -- 14 LEVEL CHANGE, 'from' via CDM-Pc fallback   elevated -> critical
    -- 15 first seen AFTER the 'now' boundary        must appear nowhere
    INSERT INTO conjunctions (id, primary_object_id, secondary_object_id,
                              time_of_closest_approach, first_seen_at, status) VALUES
        (10, 1, 2, '2026-07-30T00:00:00Z', '2026-07-21T12:00:00Z', 'active'),
        (11, 1, 3, '2026-07-29T00:00:00Z', '2026-07-20T00:00:00Z', 'active'),
        (12, 3, 2, '2026-07-21T06:00:00Z', '2026-07-20T00:00:00Z', 'past'),
        (13, 1, 2, '2026-07-28T00:00:00Z', '2026-07-20T00:00:00Z', 'active'),
        (14, 2, 3, '2026-07-28T12:00:00Z', '2026-07-20T00:00:00Z', 'active'),
        (15, 1, 2, '2026-07-27T00:00:00Z', '2026-07-22T05:00:00Z', 'active');

    INSERT INTO cdm_messages (conjunction_id, message_time, miss_distance_m,
                              relative_velocity_ms, probability_of_collision, source) VALUES
        (10, '2026-07-21T12:00:00Z', 812.4, 14210.3, 3.1e-05, 'screening'),
        (14, '2026-07-20T18:00:00Z', 950.0, 13000.0, 2.0e-06, 'screening');

    INSERT INTO risk_predictions (conjunction_id, predicted_final_risk,
                                  prediction_time, model_version) VALUES
        (10, 4.4e-05, '2026-07-21T13:00:00Z', 'heuristic-0.1'),
        (11, 5.0e-06, '2026-07-20T20:00:00Z', 'heuristic-0.1'),
        (11, 2.0e-04, '2026-07-21T20:00:00Z', 'heuristic-0.1'),
        (12, 3.0e-05, '2026-07-20T22:00:00Z', 'heuristic-0.1'),
        (13, 1.0e-07, '2026-07-20T21:00:00Z', 'heuristic-0.1'),
        (14, 5.0e-04, '2026-07-21T18:00:00Z', 'heuristic-0.1');
    """)
    conn.commit()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "api.db")
    conn = _make_conn(tmp_path / "api.db")
    _seed_history(conn)
    conn.close()
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_changes_default_compares_last_two_completed_runs(client):
    body = client.get("/changes").json()
    assert set(body) == {"since", "now", "new_conjunctions", "level_changes",
                         "resolved", "counts"}
    assert body["since"] == {"run_id": 1, "finished_at": T1}
    # run 3 has no finished_at: an in-flight run is never a boundary.
    assert body["now"] == {"run_id": 2, "finished_at": T2}

    new = body["new_conjunctions"]
    assert [item["id"] for item in new] == [10]
    assert SUMMARY_KEYS <= set(new[0])  # full summary item (may add fields)
    assert new[0]["risk_level"] == "elevated"
    assert new[0]["pc_trend"] == [3.1e-05]

    # critical-first, then pc_to desc: 14 (5e-4) before 11 (2e-4)
    assert [ch["conjunction_id"] for ch in body["level_changes"]] == [14, 11]
    ch14, ch11 = body["level_changes"]
    assert ch11 == {
        "conjunction_id": 11, "pair": "ISS (ZARYA) ↔ AEROCUBE 12A",
        "from_level": "elevated", "to_level": "critical",
        "pc_from": 5.0e-06, "pc_to": 2.0e-04,
    }
    # 'from' level fell back to the latest CDM Pc (no prediction at T1 yet).
    assert ch14["from_level"] == "elevated" and ch14["pc_from"] == 2.0e-06
    assert ch14["to_level"] == "critical" and ch14["pc_to"] == 5.0e-04

    assert body["resolved"] == [{
        "conjunction_id": 12, "pair": "AEROCUBE 12A ↔ COSMOS 2251 DEB",
        "last_level": "elevated",
    }]
    assert body["counts"] == {"new": 1, "level_changes": 2, "resolved": 1}


def test_changes_since_run_param(client):
    default = client.get("/changes").json()
    assert client.get("/changes?since_run=latest-1").json() == default
    assert client.get("/changes?since_run=1").json() == default

    # since == now: a well-formed empty diff, not an error.
    same = client.get("/changes?since_run=2").json()
    assert same["since"] == same["now"] == {"run_id": 2, "finished_at": T2}
    assert same["new_conjunctions"] == []
    assert same["level_changes"] == []
    assert same["resolved"] == []
    assert same["counts"] == {"new": 0, "level_changes": 0, "resolved": 0}

    # run 3 exists but never finished -> not addressable as a boundary.
    assert client.get("/changes?since_run=3").status_code == 404
    assert client.get("/changes?since_run=999").status_code == 404
    assert client.get("/changes?since_run=abc").status_code == 422


def test_changes_empty_safe_below_two_runs(client):
    conn = db.get_conn()
    conn.execute("DELETE FROM refresh_runs")
    conn.commit()

    body = client.get("/changes").json()  # zero runs
    assert body["since"] is None and body["now"] is None
    assert body["new_conjunctions"] == []
    assert body["level_changes"] == []
    assert body["resolved"] == []
    assert body["counts"] == {"new": 0, "level_changes": 0, "resolved": 0}

    conn.execute(
        "INSERT INTO refresh_runs (id, started_at, finished_at, stats_json)"
        " VALUES (7, ?, ?, '{}')", (T1, T1))
    conn.commit()
    conn.close()

    body = client.get("/changes").json()  # one run: 'now' known, no diff yet
    assert body["since"] is None
    assert body["now"] == {"run_id": 7, "finished_at": T1}
    assert body["counts"] == {"new": 0, "level_changes": 0, "resolved": 0}


# --- /status scheduler keys ---------------------------------------------------

def test_status_gains_scheduler_keys(client):
    body = client.get("/status").json()
    assert body["auto_refresh_interval_h"] is None  # env unset -> off
    assert body["last_run_id"] == 3                 # newest row, even in flight


def test_status_surfaces_failed_refresh(tmp_path, monkeypatch):
    """A failed refresh must reach /status so an empty board is never shown
    as all-clear; a later successful run clears it."""
    _clear_env(monkeypatch)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "err.db")
    conn = _make_conn(tmp_path / "err.db")
    conn.execute(
        "INSERT INTO refresh_runs (started_at, finished_at, stats_json)"
        " VALUES (?, ?, ?)",
        (T1, T1, json.dumps({"error": "celestrak: every catalog group failed"})))
    conn.commit()
    from backend.app.main import app

    with TestClient(app) as test_client:
        body = test_client.get("/status").json()
        assert body["last_run_error"] == "celestrak: every catalog group failed"
        conn.execute(
            "INSERT INTO refresh_runs (started_at, finished_at, stats_json)"
            " VALUES (?, ?, ?)", (T1, T1, json.dumps({"fetched": 10})))
        conn.commit()
        assert test_client.get("/status").json()["last_run_error"] is None
    conn.close()


def test_status_reports_interval_and_lifespan_starts_thread(
        tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("KESSLER_REFRESH_INTERVAL_H", "6")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "sched.db")
    _make_conn(tmp_path / "sched.db").close()
    from backend.app.main import app

    with TestClient(app) as test_client:
        assert any(t.name == "kessler-auto-refresh"
                   for t in threading.enumerate())
        body = test_client.get("/status").json()
        assert body["auto_refresh_interval_h"] == 6.0
        assert body["last_run_id"] is None  # no runs yet (first is 6h away)


def test_refresh_interval_env_parsing(monkeypatch):
    monkeypatch.delenv("KESSLER_REFRESH_INTERVAL_H", raising=False)
    assert config.refresh_interval_h() is None
    monkeypatch.setenv("KESSLER_REFRESH_INTERVAL_H", "1.5")
    assert config.refresh_interval_h() == 1.5
    monkeypatch.setenv("KESSLER_REFRESH_INTERVAL_H", "0")
    assert config.refresh_interval_h() is None      # 0 = off
    monkeypatch.setenv("KESSLER_REFRESH_INTERVAL_H", "nope")
    assert config.refresh_interval_h() is None      # malformed never crashes


# --- scheduler loop (main.py) -------------------------------------------------

def test_scheduler_loop_first_run_only_after_full_interval():
    from backend.app import main

    calls = []
    stop = threading.Event()
    interval_s = 0.02
    t0 = time.monotonic()

    def stub():
        calls.append(time.monotonic() - t0)
        if len(calls) == 2:
            stop.set()

    main.scheduler_loop(interval_s / 3600.0, stop, refresh_fn=stub)
    assert len(calls) == 2                  # stop event ends the loop
    assert calls[0] >= interval_s * 0.5     # first run after one interval


def test_scheduler_loop_stopped_event_never_runs():
    from backend.app import main

    stop = threading.Event()
    stop.set()  # shutdown before the first interval elapses
    main.scheduler_loop(
        10.0, stop,
        refresh_fn=lambda: pytest.fail("refresh must not run after stop"))


def test_scheduled_refresh_lock_skip_and_error_survival(
        tmp_path, monkeypatch, capsys):
    from backend.app import main
    import backend.app.ingestion as ingestion_pkg

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "lock.db")
    calls = []
    stub = types.ModuleType("backend.app.ingestion.pipeline")
    stub.RefreshBusyError = type("RefreshBusyError", (RuntimeError,), {})
    stub.refresh = lambda conn, **kwargs: calls.append(1)
    monkeypatch.setitem(sys.modules, "backend.app.ingestion.pipeline", stub)
    monkeypatch.setattr(ingestion_pkg, "pipeline", stub, raising=False)

    # A run in flight holds the module lock: the tick is skipped, not queued.
    assert main._refresh_lock.acquire(blocking=False)
    try:
        main._run_scheduled_refresh()
    finally:
        main._refresh_lock.release()
    assert calls == []
    assert "skipped" in capsys.readouterr().err

    # lock free: the refresh runs
    main._run_scheduled_refresh()
    assert calls == [1]

    # A failing refresh is logged to stderr, never raised, lock released.
    def boom(conn, **kwargs):
        raise RuntimeError("ingest blew up")

    stub.refresh = boom
    main._run_scheduled_refresh()
    err = capsys.readouterr().err
    assert "auto-refresh failed" in err and "ingest blew up" in err
    assert not main._refresh_lock.locked()

    # A pipeline-level busy signal (API/tool refresh holding pipeline's own
    # lock) is a skip, not an error.
    def busy(conn, **kwargs):
        raise stub.RefreshBusyError("a refresh is already running")

    stub.refresh = busy
    main._run_scheduled_refresh()
    err = capsys.readouterr().err
    assert "skipped" in err and "auto-refresh failed" not in err
    assert not main._refresh_lock.locked()
