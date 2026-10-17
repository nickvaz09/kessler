"""Persistence layer: SQLite by default, PostgreSQL via KESSLER_DATABASE_URL.

Schema is authoritative here (mirrored in docs/CONTRACTS.md). Plain sqlite3 -
no ORM. SQL is kept ANSI-portable where possible. All timestamps are stored as
ISO-8601 UTC strings with a trailing 'Z'.

Dialect switch
--------------
``get_conn()`` returns a plain :class:`sqlite3.Connection` unless the
``KESSLER_DATABASE_URL`` environment variable is set (``postgres://...`` /
``postgresql://...``), in which case it returns a :class:`PgConnection` - a thin
psycopg adapter that keeps every existing sqlite3 call-site pattern working
unchanged. An explicit ``db_path`` argument always opens SQLite (tests use
temp files regardless of the environment).

Differences the adapter absorbs (application SQL stays sqlite-flavoured):

- ``?`` placeholders are translated to ``%s`` at execute time. Application SQL
  contains no literal ``?`` (verified by grep); literal ``%`` is escaped first
  so psycopg's client-side parsing cannot misread it.
- ``INSERT OR IGNORE`` is rewritten to ``INSERT ... ON CONFLICT DO NOTHING``.
  ``cursor.rowcount`` keeps the sqlite semantics (1 when inserted, 0 when the
  conflict was ignored) on both backends.
- ``cursor.lastrowid``: psycopg has no useful lastrowid, so INSERTs into the
  known schema tables (all of which have an ``id`` identity column) get
  `` RETURNING id`` appended and the value is captured; after an ignored
  conflict it is None - always gate on ``rowcount`` first, as the call sites do.
- Rows support sqlite3.Row-style access on both backends: ``row["col"]``,
  ``row[0]``, ``dict(row)``, ``len(row)``, iteration over values.
- ``executescript`` splits the script on ``;`` (our schemas contain no literal
  semicolons inside strings) and executes each statement.
- The sqlite PRAGMAs (foreign_keys, WAL) have no Postgres counterpart and are
  simply not issued; Postgres enforces foreign keys natively.
- ``init_db`` picks the Postgres schema variant (``SCHEMA_PG``: identity
  primary keys, DOUBLE PRECISION for REAL - sqlite REAL is 8-byte - and the
  same tables/uniques/indexes) via the connection's ``dialect`` attribute.
"""
import os
import re
import sqlite3
import sys

from backend.app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS satellites (
    id              INTEGER PRIMARY KEY,
    norad_id        INTEGER NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    object_type     TEXT    NOT NULL DEFAULT 'UNKNOWN',
    intl_designator TEXT,
    last_updated    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS tle_history (
    id           INTEGER PRIMARY KEY,
    satellite_id INTEGER NOT NULL REFERENCES satellites(id),
    epoch        TEXT    NOT NULL,
    line1        TEXT    NOT NULL,
    line2        TEXT    NOT NULL,
    ingested_at  TEXT    NOT NULL,
    UNIQUE (satellite_id, epoch)
);
CREATE INDEX IF NOT EXISTS idx_tle_sat_epoch
    ON tle_history (satellite_id, epoch DESC);

CREATE TABLE IF NOT EXISTS conjunctions (
    id                       INTEGER PRIMARY KEY,
    primary_object_id        INTEGER NOT NULL REFERENCES satellites(id),
    secondary_object_id      INTEGER NOT NULL REFERENCES satellites(id),
    time_of_closest_approach TEXT    NOT NULL,
    first_seen_at            TEXT    NOT NULL,
    status                   TEXT    NOT NULL DEFAULT 'active',
    UNIQUE (primary_object_id, secondary_object_id, time_of_closest_approach)
);

CREATE TABLE IF NOT EXISTS cdm_messages (
    id                       INTEGER PRIMARY KEY,
    conjunction_id           INTEGER NOT NULL REFERENCES conjunctions(id),
    message_time             TEXT    NOT NULL,
    miss_distance_m          REAL    NOT NULL,
    relative_velocity_ms     REAL    NOT NULL,
    probability_of_collision REAL,
    covariance_json          TEXT,
    source                   TEXT    NOT NULL DEFAULT 'screening'
);
CREATE INDEX IF NOT EXISTS idx_cdm_conj_time
    ON cdm_messages (conjunction_id, message_time);

CREATE TABLE IF NOT EXISTS risk_predictions (
    id                   INTEGER PRIMARY KEY,
    conjunction_id       INTEGER NOT NULL REFERENCES conjunctions(id),
    predicted_final_risk REAL    NOT NULL,
    prediction_time      TEXT    NOT NULL,
    model_version        TEXT    NOT NULL,
    escalation_prob          REAL,
    escalation_model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_conj_time
    ON risk_predictions (conjunction_id, prediction_time);

CREATE TABLE IF NOT EXISTS user_objects (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    line1    TEXT NOT NULL,
    line2    TEXT NOT NULL,
    note     TEXT,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS refresh_runs (
    id          INTEGER PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    stats_json  TEXT
);

CREATE TABLE IF NOT EXISTS watchlog (
    id         INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    model      TEXT NOT NULL,
    summary_md TEXT NOT NULL,
    spoken     TEXT NOT NULL,
    meta_json  TEXT
);

-- key/value facts about this database itself (e.g. which offline snapshot
-- it was loaded from, see backend/app/snapshot.py), never exported
CREATE TABLE IF NOT EXISTS app_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Postgres variant: identity primary keys, DOUBLE PRECISION for real columns
# (sqlite real is 8-byte; Postgres real is 4-byte and would lose Pc precision),
# otherwise the same tables, unique constraints and indexes as SCHEMA.
SCHEMA_PG = """
CREATE TABLE IF NOT EXISTS satellites (
    id              INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    norad_id        INTEGER NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    object_type     TEXT    NOT NULL DEFAULT 'UNKNOWN',
    intl_designator TEXT,
    last_updated    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS tle_history (
    id           INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    satellite_id INTEGER NOT NULL REFERENCES satellites(id),
    epoch        TEXT    NOT NULL,
    line1        TEXT    NOT NULL,
    line2        TEXT    NOT NULL,
    ingested_at  TEXT    NOT NULL,
    UNIQUE (satellite_id, epoch)
);
CREATE INDEX IF NOT EXISTS idx_tle_sat_epoch
    ON tle_history (satellite_id, epoch DESC);

CREATE TABLE IF NOT EXISTS conjunctions (
    id                       INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    primary_object_id        INTEGER NOT NULL REFERENCES satellites(id),
    secondary_object_id      INTEGER NOT NULL REFERENCES satellites(id),
    time_of_closest_approach TEXT    NOT NULL,
    first_seen_at            TEXT    NOT NULL,
    status                   TEXT    NOT NULL DEFAULT 'active',
    UNIQUE (primary_object_id, secondary_object_id, time_of_closest_approach)
);

CREATE TABLE IF NOT EXISTS cdm_messages (
    id                       INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    conjunction_id           INTEGER NOT NULL REFERENCES conjunctions(id),
    message_time             TEXT    NOT NULL,
    miss_distance_m          DOUBLE PRECISION NOT NULL,
    relative_velocity_ms     DOUBLE PRECISION NOT NULL,
    probability_of_collision DOUBLE PRECISION,
    covariance_json          TEXT,
    source                   TEXT    NOT NULL DEFAULT 'screening'
);
CREATE INDEX IF NOT EXISTS idx_cdm_conj_time
    ON cdm_messages (conjunction_id, message_time);

CREATE TABLE IF NOT EXISTS risk_predictions (
    id                   INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    conjunction_id       INTEGER NOT NULL REFERENCES conjunctions(id),
    predicted_final_risk DOUBLE PRECISION NOT NULL,
    prediction_time      TEXT    NOT NULL,
    model_version        TEXT    NOT NULL,
    escalation_prob          DOUBLE PRECISION,
    escalation_model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_conj_time
    ON risk_predictions (conjunction_id, prediction_time);

CREATE TABLE IF NOT EXISTS user_objects (
    id       INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    name     TEXT NOT NULL,
    line1    TEXT NOT NULL,
    line2    TEXT NOT NULL,
    note     TEXT,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS refresh_runs (
    id          INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    stats_json  TEXT
);

CREATE TABLE IF NOT EXISTS watchlog (
    id         INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    created_at TEXT NOT NULL,
    model      TEXT NOT NULL,
    summary_md TEXT NOT NULL,
    spoken     TEXT NOT NULL,
    meta_json  TEXT
);

-- key/value facts about this database itself (e.g. which offline snapshot
-- it was loaded from, see backend/app/snapshot.py), never exported
CREATE TABLE IF NOT EXISTS app_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# --- Postgres adapter ---------------------------------------------------------

# tables whose INSERTs get " RETURNING id" appended so lastrowid works on pg
_ID_TABLES = frozenset(
    {"satellites", "tle_history", "conjunctions", "cdm_messages",
     "risk_predictions", "user_objects", "refresh_runs", "watchlog"})

_INSERT_IGNORE_RE = re.compile(r"^(\s*)INSERT\s+OR\s+IGNORE\s+", re.IGNORECASE)
_INSERT_TABLE_RE = re.compile(
    r"^\s*INSERT\s+INTO\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def translate_sql(sql: str) -> str:
    """Translate one sqlite-flavoured statement to Postgres.

    - escape literal '%' (psycopg parses placeholders client-side), then
      map '?' placeholders to '%s' - application SQL has no literal '?';
    - rewrite a leading 'INSERT OR IGNORE' to 'INSERT ... ON CONFLICT DO NOTHING'.
    """
    out = sql.replace("%", "%%").replace("?", "%s")
    if _INSERT_IGNORE_RE.match(out):
        out = _INSERT_IGNORE_RE.sub(r"\1INSERT ", out, count=1)
        out = out.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
    return out


def _with_returning_id(sql: str) -> tuple[str, bool]:
    """Append ' RETURNING id' to INSERTs into known id-keyed tables.

    Returns (sql, capture_id). Statements that already carry RETURNING, or
    that target unknown tables, pass through unchanged.
    """
    match = _INSERT_TABLE_RE.match(sql)
    if match and match.group(1).lower() in _ID_TABLES \
            and "returning" not in sql.lower():
        return sql.rstrip().rstrip(";") + " RETURNING id", True
    return sql, False


class Row:
    """sqlite3.Row-compatible row: index and column-name access, dict()-able."""

    __slots__ = ("_names", "_index", "_values")

    def __init__(self, names, index, values):
        self._names = names
        self._index = index
        self._values = values

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self._values[self._index[key]]
            except KeyError:
                raise IndexError(f"no such column: {key}") from None
        return self._values[key]

    def keys(self):
        return list(self._names)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return f"Row({dict(zip(self._names, self._values))!r})"


def _pg_row_factory(cursor):
    """psycopg row factory producing :class:`Row` instances."""
    description = cursor.description
    if description is None:
        return lambda values: Row((), {}, tuple(values))
    names = tuple(column.name for column in description)
    index = {name: i for i, name in enumerate(names)}

    def make_row(values):
        return Row(names, index, tuple(values))

    return make_row


class PgCursor:
    """Cursor facade exposing the sqlite3 patterns the app relies on."""

    __slots__ = ("_cursor", "_lastrowid")

    def __init__(self, cursor):
        self._cursor = cursor
        self._lastrowid = None

    def execute(self, sql, params=()):
        pg_sql, capture_id = _with_returning_id(translate_sql(sql))
        # Always pass a sequence so psycopg parses placeholders consistently
        # (and '%%' folds back to '%') even for parameterless statements.
        self._cursor.execute(pg_sql, tuple(params))
        self._lastrowid = None
        if capture_id:
            row = self._cursor.fetchone()
            if row is not None:
                self._lastrowid = row[0]
        return self

    def executemany(self, sql, seq_of_params):
        pg_sql = translate_sql(sql)
        self._cursor.executemany(pg_sql, [tuple(p) for p in seq_of_params])
        self._lastrowid = None
        return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def fetchmany(self, size=None):
        if size is None:
            return self._cursor.fetchmany()
        return self._cursor.fetchmany(size)

    def __iter__(self):
        return iter(self._cursor)

    @property
    def lastrowid(self):
        return self._lastrowid

    @property
    def rowcount(self):
        return self._cursor.rowcount

    @property
    def description(self):
        return self._cursor.description

    def close(self):
        self._cursor.close()


class PgConnection:
    """Thin psycopg connection adapter matching the sqlite3 API surface."""

    dialect = "postgres"

    def __init__(self, conninfo: str):
        import psycopg  # local import: sqlite deployments never need it
        self._conn = psycopg.connect(conninfo, row_factory=_pg_row_factory)

    def execute(self, sql, params=()):
        return PgCursor(self._conn.cursor()).execute(sql, params)

    def executemany(self, sql, seq_of_params):
        return PgCursor(self._conn.cursor()).executemany(sql, seq_of_params)

    def executescript(self, script: str):
        """Execute a multi-statement script (dialect-native, no translation)."""
        for statement in script.split(";"):
            statement = statement.strip()
            if statement:
                self._conn.execute(statement)

    def cursor(self):
        return PgCursor(self._conn.cursor())

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    @property
    def closed(self):
        return self._conn.closed


# --- Connections --------------------------------------------------------------


def get_conn(db_path=None):
    """Open a database connection.

    An explicit ``db_path`` always opens that SQLite file (tests rely on temp
    paths). With no argument, ``KESSLER_DATABASE_URL`` selects Postgres via
    the psycopg adapter; otherwise the default SQLite file is opened with row
    access by column name and FK enforcement.
    """
    if db_path is None:
        url = os.environ.get("KESSLER_DATABASE_URL")
        if url:
            return PgConnection(url)
    path = db_path or config.DB_PATH
    # check_same_thread=False: FastAPI may run a sync dependency and its
    # endpoint in different threadpool threads; each request still owns its
    # connection exclusively, so cross-thread reuse is sequential and safe.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn) -> None:
    schema = SCHEMA_PG if getattr(conn, "dialect", "sqlite") == "postgres" \
        else SCHEMA
    conn.executescript(schema)
    conn.commit()
    _migrate_add_columns(conn)
    _migrate_dedupe_republished_cdms(conn)


# Columns added to existing tables after their first release. CREATE TABLE IF
# NOT EXISTS never alters a table that already exists, so a database created
# by an older build gets them here: (table, column, sqlite type, pg type).
# Nullable only - old rows read as NULL, which every reader treats as
# "not computed".
_ADDED_COLUMNS = (
    ("risk_predictions", "escalation_prob", "REAL", "DOUBLE PRECISION"),
    ("risk_predictions", "escalation_model_version", "TEXT", "TEXT"),
)


def table_columns(conn, table: str) -> list[str]:
    """Column names of ``table`` in declaration order ([] when absent),
    on either dialect."""
    if getattr(conn, "dialect", "sqlite") == "postgres":
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns"
            " WHERE table_schema = current_schema() AND table_name = ?"
            " ORDER BY ordinal_position", (table,)).fetchall()
        return [r[0] for r in rows]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError(f"bad table name: {table!r}")
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _migrate_add_columns(conn) -> None:
    """Idempotently add _ADDED_COLUMNS missing from pre-existing tables."""
    pg = getattr(conn, "dialect", "sqlite") == "postgres"
    try:
        for table, column, sqlite_type, pg_type in _ADDED_COLUMNS:
            if column in table_columns(conn, table):
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} "
                         f"{pg_type if pg else sqlite_type}")
            conn.commit()
            print(f"[db] added column {table}.{column}", file=sys.stderr)
    except Exception as exc:  # a migration must never block startup
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"[db] add-columns migration skipped: {exc}", file=sys.stderr)


def _migrate_dedupe_republished_cdms(conn) -> None:
    """One-time cleanup for CDMs Space-Track republished from the other
    object's perspective minutes apart with identical miss/Pc - the ingest
    dedupe once keyed on exact message_time only, so every update stored
    twice (doubled CDM counts, stair-stepped Pc trends, corrupted
    msg_gap_last_h features). The earliest copy is kept. Idempotent: after
    the first pass (and with the widened ingest dedupe) it deletes nothing."""
    if getattr(conn, "dialect", "sqlite") == "postgres":
        return  # strftime is sqlite-only; pg deploys never ingested pre-fix
    try:
        cur = conn.execute(
            "DELETE FROM cdm_messages WHERE id IN ("
            " SELECT b.id FROM cdm_messages a JOIN cdm_messages b"
            "   ON a.conjunction_id = b.conjunction_id"
            "  AND a.source = 'spacetrack' AND b.source = 'spacetrack'"
            "  AND a.id < b.id"
            "  AND a.miss_distance_m IS b.miss_distance_m"
            "  AND a.probability_of_collision IS b.probability_of_collision"
            "  AND ABS(strftime('%s', a.message_time)"
            "          - strftime('%s', b.message_time)) <= 900)"
        )
        conn.commit()
    except Exception as exc:  # a cleanup must never block startup
        print(f"[db] cdm dedupe migration skipped: {exc}", file=sys.stderr)
        return
    removed = cur.rowcount if cur.rowcount is not None else 0
    if removed > 0:
        print(f"[db] removed {removed} republished spacetrack CDM duplicates")


def get_db():
    """FastAPI dependency: one connection per request."""
    conn = get_conn()
    try:
        yield conn
    finally:
        conn.close()
