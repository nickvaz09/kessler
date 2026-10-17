"""Postgres adapter tests.

Two layers:

1. Unit tests for the dialect adapter in backend/app/db.py - placeholder
   translation, INSERT OR IGNORE rewriting, RETURNING-id capture rules,
   sqlite3.Row-style access, and the get_conn()/init_db() dialect switch.
   These never touch a server.

2. An end-to-end smoke test against a real postgres:16 container on a random
   high port. It runs only when the Docker daemon answers `docker info`;
   otherwise the tests skip cleanly with an explicit reason.

The sqlite default is exercised by every other test module; nothing here may
alter sqlite behavior.
"""
import os
import sqlite3
import socket
import subprocess
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from backend.app import db


# --- 1. SQL translation -------------------------------------------------------


def test_placeholders_become_psycopg_style():
    sql = "SELECT id FROM satellites WHERE norad_id = ? AND name = ?"
    assert db.translate_sql(sql) == \
        "SELECT id FROM satellites WHERE norad_id = %s AND name = %s"


def test_literal_percent_is_escaped():
    # psycopg parses placeholders client-side; a literal % must survive.
    assert db.translate_sql("SELECT '100%' AS label") == \
        "SELECT '100%%' AS label"


def test_select_without_placeholders_passes_through():
    sql = "SELECT COUNT(*) FROM satellites"
    assert db.translate_sql(sql) == sql


def test_insert_or_ignore_becomes_on_conflict_do_nothing():
    sql = ("INSERT OR IGNORE INTO tle_history (satellite_id, epoch)"
           " VALUES (?, ?)")
    out = db.translate_sql(sql)
    assert out == ("INSERT INTO tle_history (satellite_id, epoch)"
                   " VALUES (%s, %s) ON CONFLICT DO NOTHING")


def test_insert_or_ignore_is_case_insensitive():
    out = db.translate_sql("insert or ignore into satellites (norad_id) values (?)")
    assert out.lower().startswith("insert into satellites")
    assert out.endswith(" ON CONFLICT DO NOTHING")


def test_update_and_select_are_never_rewritten():
    for sql in ("UPDATE conjunctions SET status = 'past' WHERE id = ?",
                "DELETE FROM cdm_messages WHERE id = ?"):
        out = db.translate_sql(sql)
        assert "ON CONFLICT" not in out
        assert "?" not in out


# --- 1b. RETURNING id capture -------------------------------------------------


def test_returning_id_appended_for_known_tables():
    sql, capture = db._with_returning_id(
        "INSERT INTO satellites (norad_id, name) VALUES (%s, %s)")
    assert capture is True
    assert sql.endswith(" RETURNING id")


def test_returning_id_after_on_conflict_clause():
    translated = db.translate_sql(
        "INSERT OR IGNORE INTO conjunctions (primary_object_id) VALUES (?)")
    sql, capture = db._with_returning_id(translated)
    assert capture is True
    assert sql.endswith(" ON CONFLICT DO NOTHING RETURNING id")


def test_returning_id_not_appended_for_unknown_table_or_existing_returning():
    sql, capture = db._with_returning_id("INSERT INTO other_table VALUES (1)")
    assert capture is False and sql == "INSERT INTO other_table VALUES (1)"

    already = "INSERT INTO satellites (norad_id) VALUES (%s) RETURNING id"
    sql, capture = db._with_returning_id(already)
    assert capture is False and sql == already


def test_returning_id_not_appended_for_non_insert():
    sql, capture = db._with_returning_id("UPDATE satellites SET name = %s")
    assert capture is False and "RETURNING" not in sql


# --- 1c. Row mapping access ---------------------------------------------------


def _make_row():
    names = ("id", "norad_id", "name")
    index = {n: i for i, n in enumerate(names)}
    return db.Row(names, index, (7, 25544, "ISS (ZARYA)"))


def test_row_key_and_index_access():
    row = _make_row()
    assert row["id"] == 7
    assert row["name"] == "ISS (ZARYA)"
    assert row[0] == 7
    assert row[2] == "ISS (ZARYA)"


def test_row_dict_keys_len_iter():
    row = _make_row()
    assert dict(row) == {"id": 7, "norad_id": 25544, "name": "ISS (ZARYA)"}
    assert row.keys() == ["id", "norad_id", "name"]
    assert len(row) == 3
    assert list(row) == [7, 25544, "ISS (ZARYA)"]  # iterates values, like sqlite3.Row


def test_row_missing_column_raises_index_error():
    # sqlite3.Row raises IndexError for unknown column names; match it.
    with pytest.raises(IndexError):
        _make_row()["nope"]


# --- 1d. Dialect switch -------------------------------------------------------


def test_get_conn_default_is_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("KESSLER_DATABASE_URL", raising=False)
    conn = db.get_conn(tmp_path / "t.db")
    try:
        assert isinstance(conn, sqlite3.Connection)
        assert getattr(conn, "dialect", "sqlite") == "sqlite"
    finally:
        conn.close()


def test_explicit_path_wins_over_database_url(tmp_path, monkeypatch):
    # Tests always get their temp sqlite file, even in a pg-configured shell.
    monkeypatch.setenv("KESSLER_DATABASE_URL", "postgres://x:x@localhost:1/x")
    conn = db.get_conn(tmp_path / "t.db")
    try:
        assert isinstance(conn, sqlite3.Connection)
    finally:
        conn.close()


def test_database_url_selects_postgres_adapter(monkeypatch):
    seen = {}

    class StubPg:
        def __init__(self, conninfo):
            seen["conninfo"] = conninfo

    monkeypatch.setattr(db, "PgConnection", StubPg)
    monkeypatch.setenv("KESSLER_DATABASE_URL", "postgres://u:p@h:5/d")
    conn = db.get_conn()
    assert isinstance(conn, StubPg)
    assert seen["conninfo"] == "postgres://u:p@h:5/d"


def test_init_db_picks_schema_per_dialect():
    class FakeConn:
        def __init__(self, dialect=None):
            if dialect:
                self.dialect = dialect
            self.script = None
            self.committed = False

        def executescript(self, script):
            self.script = script

        def commit(self):
            self.committed = True

    pg = FakeConn("postgres")
    db.init_db(pg)
    assert pg.script == db.SCHEMA_PG and pg.committed
    assert "GENERATED BY DEFAULT AS IDENTITY" in pg.script

    lite = FakeConn()
    db.init_db(lite)
    assert lite.script == db.SCHEMA and lite.committed


def test_pg_schema_mirrors_sqlite_schema():
    # same tables, same unique constraints, same indexes
    for name in ("satellites", "tle_history", "conjunctions", "cdm_messages",
                 "risk_predictions"):
        assert f"CREATE TABLE IF NOT EXISTS {name}" in db.SCHEMA_PG
    for idx in ("idx_tle_sat_epoch", "idx_cdm_conj_time", "idx_pred_conj_time"):
        assert idx in db.SCHEMA_PG
    assert db.SCHEMA_PG.count("UNIQUE") == db.SCHEMA.count("UNIQUE")
    # sqlite real is 8-byte; the pg variant must not narrow it.
    assert "DOUBLE PRECISION" in db.SCHEMA_PG
    assert " REAL" not in db.SCHEMA_PG


# --- 2. End-to-end against postgres:16 (requires a running Docker daemon) -----


def _docker_ready() -> bool:
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, timeout=20,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def pg_conn_url():
    """Yield a live Postgres URL: KESSLER_TEST_PG_URL when provided (CI
    service container), else a Docker-spawned postgres:16 on a random port."""
    override = os.environ.get("KESSLER_TEST_PG_URL")
    if override:
        yield override
        return
    if not _docker_ready():
        pytest.skip("Docker daemon not available — pg end-to-end test skipped")
    port = _free_port()
    name = f"kessler-pg-test-{uuid.uuid4().hex[:8]}"
    run = subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name,
         "-e", "POSTGRES_USER=kessler", "-e", "POSTGRES_PASSWORD=kessler",
         "-e", "POSTGRES_DB=kessler", "-p", f"127.0.0.1:{port}:5432",
         "postgres:16"],
        capture_output=True, text=True, timeout=600,
    )
    if run.returncode != 0:
        pytest.skip(f"could not start postgres:16 container: {run.stderr.strip()}")
    url = f"postgresql://kessler:kessler@127.0.0.1:{port}/kessler"
    try:
        import psycopg
        deadline = time.monotonic() + 90
        while True:
            try:
                psycopg.connect(url, connect_timeout=3).close()
                break
            except Exception:
                if time.monotonic() > deadline:
                    pytest.fail("postgres container did not become ready in 90s")
                time.sleep(1.0)
        yield url
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture()
def pg_conn(pg_conn_url, monkeypatch):
    monkeypatch.setenv("KESSLER_DATABASE_URL", pg_conn_url)
    conn = db.get_conn()
    assert isinstance(conn, db.PgConnection)
    db.init_db(conn)
    yield conn
    # wipe between tests so each starts clean
    for table in ("risk_predictions", "cdm_messages", "conjunctions",
                  "tle_history", "satellites"):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()


def _fake_records():
    """Records shaped like celestrak.fetch_catalog output (no network)."""
    return {
        25544: {"norad_id": 25544, "name": "ISS (ZARYA)",
                "intl_designator": "1998-067A", "object_type": "PAYLOAD",
                "epoch": "2026-07-20T06:11:52Z",
                "line1": "1 25544U 98067A   ...", "line2": "2 25544  51.6 ..."},
        34454: {"norad_id": 34454, "name": "COSMOS 2251 DEB",
                "intl_designator": None, "object_type": "DEBRIS",
                "epoch": "2026-07-20T01:00:00Z",
                "line1": "1 34454U 93036SX  ...", "line2": "2 34454  74.0 ..."},
    }


def test_pg_upsert_catalog_and_dedupe(pg_conn):
    from backend.app.ingestion import celestrak

    stats = celestrak.upsert_catalog(pg_conn, _fake_records())
    assert stats == {"fetched": 2, "satellites_new": 2, "tles_added": 2}

    # second run: same epochs -> INSERT OR IGNORE dedupes, updates are no-ops
    stats = celestrak.upsert_catalog(pg_conn, _fake_records())
    assert stats == {"fetched": 2, "satellites_new": 0, "tles_added": 0}

    row = pg_conn.execute(
        "SELECT id, norad_id, name FROM satellites WHERE norad_id = ?",
        (25544,)).fetchone()
    assert row["name"] == "ISS (ZARYA)"
    assert row[1] == 25544          # index access
    assert dict(row)["norad_id"] == 25544


def test_pg_lastrowid_and_insert_ignore_rowcount(pg_conn):
    now = "2026-07-21T00:00:00Z"
    cur = pg_conn.execute(
        "INSERT INTO satellites (norad_id, name, object_type, last_updated)"
        " VALUES (?, ?, ?, ?)", (99001, "TESTSAT", "PAYLOAD", now))
    assert cur.rowcount == 1
    sat_id = cur.lastrowid
    assert isinstance(sat_id, int) and sat_id > 0

    cur = pg_conn.execute(
        "INSERT OR IGNORE INTO tle_history (satellite_id, epoch, line1, line2,"
        " ingested_at) VALUES (?, ?, ?, ?, ?)",
        (sat_id, now, "l1", "l2", now))
    assert cur.rowcount == 1 and cur.lastrowid is not None
    cur = pg_conn.execute(
        "INSERT OR IGNORE INTO tle_history (satellite_id, epoch, line1, line2,"
        " ingested_at) VALUES (?, ?, ?, ?, ?)",
        (sat_id, now, "l1", "l2", now))
    assert cur.rowcount == 0        # ignored duplicate, sqlite-equivalent
    assert cur.lastrowid is None
    pg_conn.commit()


def test_pg_record_events_and_summaries(pg_conn):
    from backend.app.api import conjunctions as conjunctions_api
    from backend.app.ingestion import celestrak, screening

    celestrak.upsert_catalog(pg_conn, _fake_records())
    tca = datetime(2026, 7, 23, 4, 12, 9, tzinfo=timezone.utc)
    events = [screening.ConjunctionEvent(
        primary_norad=25544, secondary_norad=34454, tca=tca,
        miss_distance_m=812.4, relative_velocity_ms=14210.3)]
    now_iso = "2026-07-21T00:00:00Z"

    stats = screening.record_events(pg_conn, events, now_iso)
    assert stats == {"conjunctions_new": 1, "cdms_added": 1}

    # Same pair within the 30-min TCA tolerance -> same event, no new row.
    nudged = [screening.ConjunctionEvent(
        primary_norad=25544, secondary_norad=34454,
        tca=tca + timedelta(minutes=5),
        miss_distance_m=700.0, relative_velocity_ms=14200.0)]
    stats = screening.record_events(pg_conn, nudged, now_iso)
    assert stats["conjunctions_new"] == 0 and stats["cdms_added"] == 1

    items = conjunctions_api.query_summaries(pg_conn, "c.status = 'active'")
    assert len(items) == 1
    item = items[0]
    assert item["primary"]["norad_id"] == 25544
    assert item["secondary"]["norad_id"] == 34454
    assert item["cdm_count"] == 2
    assert item["risk_level"] in ("nominal", "elevated", "critical")

    conjunctions_api.attach_pc_trends(pg_conn, items)   # in (?,?) expansion
    assert isinstance(items[0]["pc_trend"], list)


def test_pg_risk_prediction_and_status_queries(pg_conn):
    from backend.app.ingestion import celestrak, screening
    from backend.app.models import risk_model

    celestrak.upsert_catalog(pg_conn, _fake_records())
    tca = datetime(2026, 7, 23, 0, 0, 0, tzinfo=timezone.utc)
    screening.record_events(pg_conn, [screening.ConjunctionEvent(
        primary_norad=25544, secondary_norad=34454, tca=tca,
        miss_distance_m=500.0, relative_velocity_ms=12000.0)],
        "2026-07-21T00:00:00Z")
    conj_id = pg_conn.execute(
        "SELECT id FROM conjunctions").fetchone()["id"]

    result = risk_model.predict_for_conjunction(pg_conn, conj_id)
    assert result["conjunction_id"] == conj_id
    assert 0.0 <= result["predicted_final_risk"] <= 1.0

    # status.py access patterns: positional index on aggregates.
    assert pg_conn.execute(
        "SELECT COUNT(*) FROM satellites").fetchone()[0] == 2
    assert pg_conn.execute(
        "SELECT MAX(ingested_at) FROM tle_history").fetchone()[0] is not None
    has_demo = pg_conn.execute(
        "SELECT EXISTS(SELECT 1 FROM cdm_messages WHERE source = 'demo')"
    ).fetchone()[0]
    assert bool(has_demo) is False

    # objects.py search patterns (case-insensitive LIKE, portable ordering).
    rows = pg_conn.execute(
        "SELECT norad_id FROM satellites sat"
        " WHERE LOWER(sat.name) LIKE LOWER(?)"
        " ORDER BY LOWER(sat.name) ASC, sat.norad_id ASC", ("%iss%",)
    ).fetchall()
    assert [r["norad_id"] for r in rows] == [25544]
