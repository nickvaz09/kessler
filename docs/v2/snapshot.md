# Offline snapshot (demo-day insurance) - contract

`backend/app/snapshot.py`. Works on SQLite and Postgres (`KESSLER_DATABASE_URL`).

## CLI

```
python -m backend.app.snapshot export data/snapshot.json.gz      # whole DB -> file
python -m backend.app.snapshot import data/snapshot.json.gz      # file -> EMPTY DB
# --db PATH forces a SQLite file instead of the app database
```

- Export streams every app table (keyset pages of 2000 rows on `id`; tables are discovered,
  so a table added later is included automatically) in FK-safe order. `app_meta` is excluded.
  Written to `<path>.part` and renamed, so a crash never leaves a truncated file under the real name.
- Import refuses a non-empty database (exit 1 / `ValueError`), runs in one transaction
  (failure -> rollback, DB stays empty), keeps ids verbatim, and on Postgres moves each identity
  sequence past the imported ids. Schema drift: unknown tables skipped, unknown columns dropped,
  missing columns left NULL (all reported in the summary JSON).

File: gzip JSON, valid as a whole document, one row per line
(`{"format":"kessler-snapshot","format_version":1,"exported_at","source_dialect","last_ingest_at","tables":[{"name","columns","rows":[...]}],"counts":{...}}`).

## Startup

`KESSLER_SNAPSHOT=<path>`: at app startup (`main.py` lifespan, right after `init_db`), if the
`satellites` table is empty the file is imported and logged to stderr
(`[snapshot] imported … (exported …, last ingest …): N rows`). A non-empty catalog, missing
file or corrupt file is logged and skipped - it never blocks startup. Provenance is stored in
the `app_meta` table (key `snapshot`).

## /status

New key `snapshot`: `null`, or

```json
{"path": "...", "exported_at": "...Z", "imported_at": "...Z", "last_ingest_at": "...Z"}
```

It is non-null while the data came from a snapshot and becomes null once a refresh that
**started after the import** finishes without error (the data is live again).

Honesty: rows are copied verbatim - `tle_history.ingested_at` is untouched, so
`/status.last_ingest_at` (the header's live pill) shows the snapshot's real age, and
conjunctions whose TCA has passed read as past. UI suggestion: when `snapshot` is non-null,
the pill can read "Offline snapshot · updated <age of last_ingest_at>".

## Demo-day recipe

```
python -m backend.app.snapshot export data/snapshot.json.gz     # the night before, on a fresh DB
KESSLER_SNAPSHOT=data/snapshot.json.gz KESSLER_DATABASE_URL= \
  .venv/bin/uvicorn backend.app.main:app          # with data/kessler.db absent or empty
```
