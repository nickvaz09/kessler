"""Offline snapshot (backend/app/snapshot.py) tests - hermetic temp SQLite
files; the Postgres round trip runs only when KESSLER_TEST_PG_URL points at
a server where the user may CREATE DATABASE.
"""
import gzip
import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.app import config, db, snapshot

TABLES = ("satellites", "tle_history", "conjunctions", "cdm_messages",
          "risk_predictions", "user_objects", "refresh_runs", "watchlog")

SEED = """
INSERT INTO satellites (id, norad_id, name, object_type, intl_designator, last_updated) VALUES
    (1, 25544, 'ISS (ZARYA)', 'PAYLOAD', '1998-067A', '2026-07-21T13:02:41Z'),
    (7, 34454, 'COSMOS 2251 DEB', 'DEBRIS', NULL, '2026-07-21T13:02:41Z');
INSERT INTO tle_history (id, satellite_id, epoch, line1, line2, ingested_at) VALUES
    (1, 1, '2026-07-21T06:11:52Z', '1 25544U 98067A   26202.25824074', '2 25544  51.6', '2026-07-21T13:02:41Z'),
    (2, 7, '2026-07-21T02:00:00Z', '1 34454U', '2 34454', '2026-07-21T13:02:41Z');
INSERT INTO conjunctions (id, primary_object_id, secondary_object_id,
                          time_of_closest_approach, first_seen_at, status) VALUES
    (3, 1, 7, '2026-07-23T04:12:09Z', '2026-07-20T10:00:00Z', 'active');
INSERT INTO cdm_messages (id, conjunction_id, message_time, miss_distance_m,
    relative_velocity_ms, probability_of_collision, covariance_json, source) VALUES
    (1, 3, '2026-07-20T10:00:00Z', 1500.0, 14210.3, 1.1e-06, NULL, 'screening'),
    (2, 3, '2026-07-21T10:00:00Z', 812.4, 14210.3, 3.123456789012345e-05, '{"a": [1, 2]}', 'spacetrack'),
    (3, 3, '2026-07-21T11:00:00Z', 800.0, 14210.3, NULL, NULL, 'demo');
INSERT INTO risk_predictions (id, conjunction_id, predicted_final_risk, prediction_time,
    model_version, escalation_prob, escalation_model_version) VALUES
    (1, 3, 4.4e-05, '2026-07-21T11:00:00Z', 'vigil-0.4', NULL, NULL),
    (2, 3, 4.5e-05, '2026-07-21T12:00:00Z', 'vigil-0.4', 0.25, 'vigil-e-0.1');
INSERT INTO user_objects (id, name, line1, line2, note, added_at) VALUES
    (1, 'MYSAT', 'l1', 'l2', 'semi;colon and ''quote'' and 100% and ?', '2026-07-21T00:00:00Z');
INSERT INTO refresh_runs (id, started_at, finished_at, stats_json) VALUES
    (1, '2026-07-21T13:00:00Z', '2026-07-21T13:03:00Z', '{"fetched": 2}');
INSERT INTO watchlog (id, created_at, model, summary_md, spoken, meta_json) VALUES
    (1, '2026-07-21T13:05:00Z', 'claude', '# Shift\\nline two', 'spoken text', NULL);
"""


def _dump(conn):
    out = {}
    for t in TABLES:
        cols = db.table_columns(conn, t)
        out[t] = [tuple(r) for r in conn.execute(
            f"SELECT {', '.join(cols)} FROM {t} ORDER BY id").fetchall()]
    return out


@pytest.fixture()
def seeded(tmp_path):
    conn = db.get_conn(tmp_path / "src.db")
    db.init_db(conn)
    conn.executescript(SEED)
    conn.commit()
    yield conn
    conn.close()


def test_roundtrip_sqlite_is_exact_and_valid_json(seeded, tmp_path):
    path = tmp_path / "snap.json.gz"
    out = snapshot.export_snapshot(seeded, path)
    assert out["counts"]["cdm_messages"] == 3 and out["counts"]["watchlog"] == 1
    assert "app_meta" not in out["counts"]
    doc = json.load(gzip.open(path, "rt"))           # the file is plain JSON
    assert doc["format"] == "kessler-snapshot"
    assert doc["last_ingest_at"] == "2026-07-21T13:02:41Z"
    assert [t["name"] for t in doc["tables"]][:5] == list(TABLES[:5])

    dst = db.get_conn(tmp_path / "dst.db")
    summary = snapshot.import_snapshot(dst, path)
    assert summary["exported_at"] == out["exported_at"]
    assert summary["counts"] == out["counts"]
    assert _dump(dst) == _dump(seeded)                # verbatim, ids included
    # the app keeps working: new rows get fresh ids past the imported ones
    cur = dst.execute("INSERT INTO satellites (norad_id, name, last_updated)"
                      " VALUES (1, 'X', '2026-07-22T00:00:00Z')")
    assert cur.lastrowid == 8
    dst.close()


def test_import_refuses_non_empty_database(seeded, tmp_path):
    path = tmp_path / "snap.json.gz"
    snapshot.export_snapshot(seeded, path)
    with pytest.raises(ValueError, match="not empty"):
        snapshot.import_snapshot(seeded, path)


def test_truncated_snapshot_rolls_back(seeded, tmp_path):
    path = tmp_path / "snap.json.gz"
    snapshot.export_snapshot(seeded, path)
    lines = gzip.open(path, "rt").read().splitlines(True)
    bad = tmp_path / "bad.json.gz"
    with gzip.open(bad, "wt") as fh:
        fh.writelines(lines[:-3])                     # lose the trailer
    dst = db.get_conn(tmp_path / "dst.db")
    with pytest.raises(ValueError, match="truncated"):
        snapshot.import_snapshot(dst, bad)
    assert snapshot.is_empty(dst)                     # nothing half-loaded
    dst.close()


def test_import_tolerates_schema_drift(seeded, tmp_path):
    """Unknown tables are skipped, unknown columns dropped, missing ones NULL."""
    path = tmp_path / "snap.json.gz"
    snapshot.export_snapshot(seeded, path)
    doc = json.load(gzip.open(path, "rt"))
    for t in doc["tables"]:
        if t["name"] == "risk_predictions":           # an older build's shape
            keep = [i for i, c in enumerate(t["columns"]) if not c.startswith("escalation")]
            t["columns"] = [t["columns"][i] for i in keep] + ["future_col"]
            t["rows"] = [[r[i] for i in keep] + ["x"] for r in t["rows"]]
    doc["tables"].append({"name": "some_future_table", "columns": ["id"], "rows": [[1]]})
    drift = tmp_path / "drift.json.gz"
    _write_protocol(drift, doc)
    dst = db.get_conn(tmp_path / "dst.db")
    summary = snapshot.import_snapshot(dst, drift)
    assert summary["skipped_tables"] == ["some_future_table"]
    assert summary["dropped_columns"] == {"risk_predictions": ["future_col"]}
    rows = dst.execute("SELECT escalation_prob FROM risk_predictions").fetchall()
    assert [r[0] for r in rows] == [None, None]
    dst.close()


def _write_protocol(path, doc):
    """Re-emit a parsed snapshot in the exporter's line protocol."""
    header = {k: v for k, v in doc.items() if k not in ("tables", "counts")}
    with gzip.open(path, "wt") as fh:
        fh.write(json.dumps(header)[:-1] + ', "tables": [\n')
        for ti, t in enumerate(doc["tables"]):
            fh.write(("," if ti else "") + json.dumps(
                {"name": t["name"], "columns": t["columns"]})[:-1] + ', "rows": [\n')
            for ri, r in enumerate(t["rows"]):
                fh.write(("," if ri else "") + json.dumps(r) + "\n")
            fh.write("]}\n")
        fh.write('], "counts": {}}\n')


def test_rejects_non_snapshot_file(tmp_path):
    bad = tmp_path / "x.json.gz"
    with gzip.open(bad, "wt") as fh:
        fh.write('{"hello": 1}\n')
    dst = db.get_conn(tmp_path / "dst.db")
    with pytest.raises(ValueError, match="not a Kessler snapshot"):
        snapshot.import_snapshot(dst, bad)
    dst.close()


def test_cli_export_and_import(seeded, tmp_path, capsys):
    src = tmp_path / "src.db"
    path = tmp_path / "cli.json.gz"
    assert snapshot.main(["export", str(path), "--db", str(src)]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["satellites"] == 2
    dst = tmp_path / "cli_dst.db"
    assert snapshot.main(["import", str(path), "--db", str(dst)]) == 0
    assert snapshot.main(["import", str(path), "--db", str(dst)]) == 1  # not empty
    conn = db.get_conn(dst)
    assert _dump(conn) == _dump(seeded)
    assert snapshot.snapshot_status(conn)["path"] == str(path)
    conn.close()


# --- startup + /status ---------------------------------------------------------------


def test_load_on_startup_rules(seeded, tmp_path, capsys):
    path = tmp_path / "snap.json.gz"
    snapshot.export_snapshot(seeded, path)
    assert snapshot.load_on_startup(seeded, {}) is None                 # env unset
    assert snapshot.load_on_startup(seeded, {"KESSLER_SNAPSHOT": str(path)}) is None
    assert "already has 2 objects" in capsys.readouterr().err           # non-empty

    dst = db.get_conn(tmp_path / "dst.db")
    db.init_db(dst)
    assert snapshot.load_on_startup(dst, {"KESSLER_SNAPSHOT": str(tmp_path / "nope.gz")}) is None
    out = snapshot.load_on_startup(dst, {"KESSLER_SNAPSHOT": str(path)})
    assert out["counts"]["satellites"] == 2
    assert "[snapshot] imported" in capsys.readouterr().err
    status = snapshot.snapshot_status(dst)
    assert status["path"] == str(path)
    assert status["exported_at"] == out["exported_at"]
    assert status["last_ingest_at"] == "2026-07-21T13:02:41Z"
    dst.close()


def test_corrupt_snapshot_never_blocks_startup(tmp_path, capsys):
    bad = tmp_path / "bad.json.gz"
    bad.write_bytes(b"not gzip at all")
    dst = db.get_conn(tmp_path / "dst.db")
    db.init_db(dst)
    assert snapshot.load_on_startup(dst, {"KESSLER_SNAPSHOT": str(bad)}) is None
    assert "failed" in capsys.readouterr().err
    assert snapshot.is_empty(dst)
    dst.close()


def test_status_reports_snapshot_until_a_live_refresh(seeded, tmp_path, monkeypatch):
    path = tmp_path / "snap.json.gz"
    snapshot.export_snapshot(seeded, path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")
    monkeypatch.delenv("KESSLER_DATABASE_URL", raising=False)
    monkeypatch.setenv("KESSLER_SNAPSHOT", str(path))
    from backend.app.main import app
    with TestClient(app) as client:
        body = client.get("/status").json()
        assert body["catalog_size"] == 2
        assert body["snapshot"]["path"] == str(path)
        assert body["snapshot"]["exported_at"]
        # the live pill's timestamp is the snapshot's real ingest time
        assert body["last_ingest_at"] == "2026-07-21T13:02:41Z"

        conn = db.get_conn(config.DB_PATH)
        conn.execute("INSERT INTO refresh_runs (started_at, finished_at, stats_json)"
                     " VALUES ('2999-01-01T00:00:00Z', '2999-01-01T00:01:00Z',"
                     " '{\"error\": \"offline\"}')")
        conn.commit()
        assert client.get("/status").json()["snapshot"] is not None   # failed run
        conn.execute("INSERT INTO refresh_runs (started_at, finished_at, stats_json)"
                     " VALUES ('2999-01-02T00:00:00Z', '2999-01-02T00:01:00Z',"
                     " '{\"fetched\": 10}')")
        conn.commit()
        conn.close()
        assert client.get("/status").json()["snapshot"] is None       # live again

    # restart with the env still set: catalog non-empty -> no re-import
    with TestClient(app) as client:
        assert client.get("/status").json()["catalog_size"] == 2


def test_status_snapshot_null_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")
    monkeypatch.delenv("KESSLER_DATABASE_URL", raising=False)
    monkeypatch.delenv("KESSLER_SNAPSHOT", raising=False)
    from backend.app.main import app
    with TestClient(app) as client:
        assert client.get("/status").json()["snapshot"] is None


# --- Postgres --------------------------------------------------------------------------

PG_URL = os.environ.get("KESSLER_TEST_PG_URL")


@pytest.fixture()
def pg_conn():
    if not PG_URL:
        pytest.skip("KESSLER_TEST_PG_URL not set (no Postgres server)")
    import psycopg
    name = f"kessler_s_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(PG_URL, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    conn = db.PgConnection(PG_URL.rsplit("/", 1)[0] + "/" + name)
    try:
        yield conn
    finally:
        conn.close()
        with psycopg.connect(PG_URL, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE {name} WITH (FORCE)")


def test_pg_roundtrip_both_directions(seeded, tmp_path, pg_conn, monkeypatch):
    monkeypatch.setattr(snapshot, "_PAGE", 2)      # exercise keyset paging
    a = tmp_path / "a.json.gz"
    snapshot.export_snapshot(seeded, a)
    snapshot.import_snapshot(pg_conn, a)          # sqlite -> postgres
    assert _dump(pg_conn) == _dump(seeded)
    # identity sequences moved past imported ids
    cur = pg_conn.execute("INSERT INTO satellites (norad_id, name, last_updated)"
                          " VALUES (1, 'X', '2026-07-22T00:00:00Z')")
    assert cur.lastrowid == 8
    pg_conn.execute("DELETE FROM satellites WHERE id = 8")
    pg_conn.commit()

    b = tmp_path / "b.json.gz"
    out = snapshot.export_snapshot(pg_conn, b)    # postgres -> sqlite
    assert json.load(gzip.open(b, "rt"))["source_dialect"] == "postgres"
    assert out["counts"]["cdm_messages"] == 3
    dst = db.get_conn(tmp_path / "back.db")
    snapshot.import_snapshot(dst, b)
    assert _dump(dst) == _dump(seeded)
    dst.close()
