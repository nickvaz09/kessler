"""Offline snapshot: export the whole database to one gzipped JSON file and
load it back into an empty database - demo-day insurance for when the
network (CelesTrak, Space-Track) is not there.

CLI (repo root; the database is the app's: KESSLER_DATABASE_URL -> Postgres,
else the default SQLite file; --db PATH forces a SQLite file):
    python -m backend.app.snapshot export data/snapshot.json.gz
    python -m backend.app.snapshot import data/snapshot.json.gz

Startup: with KESSLER_SNAPSHOT=<path>, app startup imports the file when the
catalog (satellites) is empty, logs it, and records provenance in the
app_meta table; GET /status then reports
``"snapshot": {"path", "exported_at", "imported_at", "last_ingest_at"}``
until a live refresh started after the import finishes cleanly.

Honesty: rows are copied VERBATIM - ingested_at, message_time,
prediction_time, refresh_runs all keep their original values, so
/status.last_ingest_at (the header's live pill) shows the snapshot's real
age, and conjunctions whose TCA has passed read as past on the read side.
Nothing is re-stamped to "now".

File format (valid JSON, written and read line by line so neither side
holds a table in memory):
    {"format": "kessler-snapshot", "format_version": 1, "exported_at": ...,
     "source_dialect": ..., "last_ingest_at": ..., "tables": [
    {"name": "satellites", "columns": [...], "rows": [
    [row values...]
    ,[row values...]
    ]}
    ,{"name": ...
    ], "counts": {"satellites": N, ...}}
"""
import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from backend.app import db

FORMAT = "kessler-snapshot"
FORMAT_VERSION = 1
ENV_VAR = "KESSLER_SNAPSHOT"
META_KEY = "snapshot"
_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
_PAGE = 2000

# Foreign-key-safe order for the known schema; any other app table found in
# the database is exported after these (future tables need no code change).
TABLE_ORDER = ("satellites", "tle_history", "conjunctions", "cdm_messages",
               "risk_predictions", "user_objects", "refresh_runs", "watchlog")
EXCLUDED = frozenset({"app_meta"})  # facts about this database, not data


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(_ISO_FMT)


def _is_pg(conn) -> bool:
    return getattr(conn, "dialect", "sqlite") == "postgres"


def app_tables(conn) -> list[str]:
    """Every application table in FK-safe order (known first, then others
    alphabetically), excluding EXCLUDED and database internals."""
    if _is_pg(conn):
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables"
            " WHERE table_schema = current_schema()"
            "   AND table_type = 'BASE TABLE'").fetchall()
    else:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
            " AND name NOT LIKE 'sqlite_%'").fetchall()
    present = {r[0] for r in rows} - EXCLUDED
    known = [t for t in TABLE_ORDER if t in present]
    return known + sorted(present - set(known))


def _iter_rows(conn, table: str, columns: list[str]):
    """Stream a table's rows: keyset pages on id when the table has one
    (bounded memory on both dialects), else one plain scan."""
    col_sql = ", ".join(columns)
    if "id" in columns:
        last = None
        while True:
            if last is None:
                page = conn.execute(
                    f"SELECT {col_sql} FROM {table} ORDER BY id LIMIT {_PAGE}"
                ).fetchall()
            else:
                page = conn.execute(
                    f"SELECT {col_sql} FROM {table} WHERE id > ?"
                    f" ORDER BY id LIMIT {_PAGE}", (last,)).fetchall()
            if not page:
                return
            idx = columns.index("id")
            for row in page:
                yield list(row)
            last = page[-1][idx]
    else:
        for row in conn.execute(f"SELECT {col_sql} FROM {table}").fetchall():
            yield list(row)


def export_snapshot(conn, path) -> dict:
    """Write every app table to ``path`` (gzip JSON). Returns the summary
    {"path", "exported_at", "counts"}."""
    exported_at = _now_iso()
    try:
        last_ingest = conn.execute(
            "SELECT MAX(ingested_at) FROM tle_history").fetchone()[0]
    except Exception:
        last_ingest = None
    header = {"format": FORMAT, "format_version": FORMAT_VERSION,
              "exported_at": exported_at,
              "source_dialect": "postgres" if _is_pg(conn) else "sqlite",
              "last_ingest_at": last_ingest}
    counts = {}
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(header)[:-1] + ', "tables": [\n')
        for ti, table in enumerate(app_tables(conn)):
            columns = db.table_columns(conn, table)
            fh.write(("," if ti else "")
                     + json.dumps({"name": table, "columns": columns})[:-1]
                     + ', "rows": [\n')
            n = 0
            for row in _iter_rows(conn, table, columns):
                fh.write(("," if n else "") + json.dumps(row) + "\n")
                n += 1
            fh.write("]}\n")
            counts[table] = n
        fh.write('], "counts": ' + json.dumps(counts) + "}\n")
    tmp.replace(path)  # never leave a truncated file under the real name
    return {"path": str(path), "exported_at": exported_at, "counts": counts}


def _read_snapshot(path):
    """Yield ("header", dict) then per table ("table", name, columns) and
    ("row", list) items, then ("end", counts). Line protocol of
    export_snapshot; raises ValueError on anything else."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        first = fh.readline()
        if not first.rstrip().endswith('"tables": ['):
            raise ValueError("not a Kessler snapshot (bad header)")
        header = json.loads(first.rstrip()[: -len(', "tables": [')] + "}")
        if header.get("format") != FORMAT:
            raise ValueError("not a Kessler snapshot (format tag)")
        if header.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported snapshot format_version {header.get('format_version')}")
        yield ("header", header)
        in_rows = False
        for line in fh:
            line = line.rstrip("\n")
            if in_rows:
                if line == "]}":
                    in_rows = False
                    continue
                yield ("row", json.loads(line[1:] if line.startswith(",") else line))
                continue
            if line.startswith("], \"counts\": "):
                yield ("end", json.loads(line[len('], "counts": '):-1]))
                return
            body = line[1:] if line.startswith(",") else line
            if not body.endswith(', "rows": ['):
                raise ValueError(f"unexpected snapshot line: {line[:80]!r}")
            spec = json.loads(body[: -len(', "rows": [')] + "}")
            in_rows = True
            yield ("table", spec["name"], spec["columns"])
        raise ValueError("snapshot is truncated (no trailer)")


def is_empty(conn) -> bool:
    """True when no app table holds a row (the only state import accepts)."""
    for table in app_tables(conn):
        if conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None:
            return False
    return True


def _flush(conn, table, columns, batch):
    placeholders = ", ".join("?" * len(columns))
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        batch)


def _reset_identity(conn, table: str) -> None:
    """Postgres: move the identity sequence past the imported ids (explicit
    ids do not advance it; the next app INSERT would collide)."""
    if not _is_pg(conn):
        return
    conn.execute(
        f"SELECT setval(pg_get_serial_sequence('{table}', 'id'),"
        f" COALESCE((SELECT MAX(id) FROM {table}), 1),"
        f" (SELECT MAX(id) FROM {table}) IS NOT NULL)")


def import_snapshot(conn, path) -> dict:
    """Load ``path`` into ``conn`` (schema initialized here). The database
    must be empty - refuses otherwise (ValueError), never merges.

    Columns present in the snapshot but not the target schema are dropped
    (reported); target columns absent from the snapshot stay NULL/default;
    unknown tables are skipped (reported). One transaction: any failure
    rolls everything back. Returns {"path", "exported_at", "counts",
    "skipped_tables", "dropped_columns"}.
    """
    db.init_db(conn)
    if not is_empty(conn):
        raise ValueError("target database is not empty — snapshot import "
                         "only loads into an empty database")
    target = set(app_tables(conn))
    counts, skipped, dropped = {}, [], {}
    header = None
    table = keep = cols = None
    batch: list = []
    try:
        for item in _read_snapshot(path):
            kind = item[0]
            if kind == "header":
                header = item[1]
            elif kind == "table":
                if table is not None and batch:
                    _flush(conn, table, cols, batch)
                if table is not None:
                    _reset_identity(conn, table)
                batch = []
                name, columns = item[1], item[2]
                if name not in target:
                    skipped.append(name)
                    table = None
                    continue
                have = set(db.table_columns(conn, name))
                keep = [i for i, c in enumerate(columns) if c in have]
                cols = [columns[i] for i in keep]
                missing = [c for c in columns if c not in have]
                if missing:
                    dropped[name] = missing
                table = name
                counts[name] = 0
            elif kind == "row":
                if table is None:
                    continue
                batch.append(tuple(item[1][i] for i in keep))
                counts[table] += 1
                if len(batch) >= _PAGE:
                    _flush(conn, table, cols, batch)
                    batch = []
            elif kind == "end":
                if table is not None:
                    if batch:
                        _flush(conn, table, cols, batch)
                    _reset_identity(conn, table)
                table, batch = None, []
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"path": str(path), "exported_at": header.get("exported_at"),
            "last_ingest_at": header.get("last_ingest_at"),
            "counts": counts, "skipped_tables": skipped,
            "dropped_columns": dropped}


def _set_meta(conn, key: str, value: dict) -> None:
    conn.execute("DELETE FROM app_meta WHERE key = ?", (key,))
    conn.execute("INSERT INTO app_meta (key, value) VALUES (?, ?)",
                 (key, json.dumps(value)))
    conn.commit()


def snapshot_status(conn) -> dict | None:
    """/status "snapshot" field: provenance of an imported snapshot, or None
    when the data did not come from one - or when a refresh STARTED after
    the import has since finished without error (the data is live again)."""
    try:
        row = conn.execute("SELECT value FROM app_meta WHERE key = ?",
                           (META_KEY,)).fetchone()
        if row is None:
            return None
        info = json.loads(row[0])
        imported_at = info.get("imported_at")
        runs = conn.execute(
            "SELECT stats_json FROM refresh_runs"
            " WHERE started_at >= ? AND finished_at IS NOT NULL",
            (imported_at,)).fetchall()
        for r in runs:
            try:
                stats = json.loads(r[0]) if r[0] else {}
            except (TypeError, ValueError):
                stats = {}
            if isinstance(stats, dict) and "error" not in stats:
                return None
        return {"path": info.get("path"), "exported_at": info.get("exported_at"),
                "imported_at": imported_at,
                "last_ingest_at": info.get("last_ingest_at")}
    except Exception:
        return None


def load_on_startup(conn, environ=None) -> dict | None:
    """App-startup hook: KESSLER_SNAPSHOT=<path> and an empty catalog ->
    import it, log it, record provenance in app_meta. Never raises (a bad
    snapshot must not keep the API from booting; the reason goes to stderr).
    Returns the import summary or None when nothing was imported."""
    import os
    env = os.environ if environ is None else environ
    path = (env.get(ENV_VAR) or "").strip()
    if not path:
        return None
    try:
        n = conn.execute("SELECT COUNT(*) FROM satellites").fetchone()[0]
        if n:
            print(f"[snapshot] {ENV_VAR} set but the catalog already has {n} "
                  "objects — not importing", file=sys.stderr)
            return None
        if not Path(path).exists():
            print(f"[snapshot] {ENV_VAR}={path} does not exist — not importing",
                  file=sys.stderr)
            return None
        summary = import_snapshot(conn, path)
        _set_meta(conn, META_KEY, {
            "path": str(path), "exported_at": summary["exported_at"],
            "imported_at": _now_iso(),
            "last_ingest_at": summary.get("last_ingest_at")})
        total = sum(summary["counts"].values())
        print(f"[snapshot] imported {path} (exported {summary['exported_at']},"
              f" last ingest {summary.get('last_ingest_at')}): {total} rows "
              f"{summary['counts']}", file=sys.stderr)
        return summary
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"[snapshot] import of {path} failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export/import a Kessler offline snapshot (.json.gz).")
    parser.add_argument("action", choices=["export", "import"])
    parser.add_argument("path")
    parser.add_argument("--db", default=None,
                        help="SQLite file to use instead of the app database")
    args = parser.parse_args(argv)
    if args.db is None:
        from backend.app import config
        Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = db.get_conn(args.db)
    try:
        if args.action == "export":
            db.init_db(conn)
            out = export_snapshot(conn, args.path)
        else:
            out = import_snapshot(conn, args.path)
            _set_meta(conn, META_KEY, {
                "path": str(args.path), "exported_at": out["exported_at"],
                "imported_at": _now_iso(),
                "last_ingest_at": out.get("last_ingest_at")})
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
