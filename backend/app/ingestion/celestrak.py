"""CelesTrak GP data client and 3-line TLE parser.

Fetches TLE groups from config.CELESTRAK_GP_URL, parses the 3-line element
format (name line + line1 + line2), decodes epochs, and upserts the catalog
into SQLite (satellites + tle_history).
"""
from datetime import datetime, timedelta, timezone
import sys

import httpx

from backend.app import config

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

# alpha-5 NORAD scheme: first character may be a letter (I and O excluded)
_ALPHA5 = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FMT)


def tle_epoch(line1: str) -> datetime:
    """Decode the epoch from TLE line 1 (cols 19-32: YYDDD.DDDDDDDD).

    Two-digit years < 57 map to 20xx, otherwise 19xx.
    """
    yy = int(line1[18:20])
    day_of_year = float(line1[20:32])
    year = 2000 + yy if yy < 57 else 1900 + yy
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1.0)


def _parse_norad(field: str) -> int:
    field = field.strip()
    if field.isdigit():
        return int(field)
    # Alpha-5 designator (e.g. 'A1234' -> 101234)
    return _ALPHA5.index(field[0].upper()) * 10000 + int(field[1:])


def _intl_designator(line1: str) -> str | None:
    """Decode cols 10-17 of line 1 ('98067A  ') to '1998-067A'."""
    field = line1[9:17].strip()
    if len(field) < 5 or not field[:2].isdigit():
        return None
    yy = int(field[:2])
    year = 2000 + yy if yy < 57 else 1900 + yy
    return f"{year}-{field[2:5]}{field[5:].strip()}"


def _object_type(name: str) -> str:
    if " DEB" in name:
        return "DEBRIS"
    if " R/B" in name:
        return "ROCKET BODY"
    return "PAYLOAD"


def _is_tle_line(s: str, leader: str) -> bool:
    return s.startswith(leader + " ") and len(s) >= 68 and s[2:7].strip() != ""


def parse_tle_text(text: str) -> list[dict]:
    """Parse 3-line TLE text into catalog records.

    Robust to blank lines and trailing whitespace. Records whose two lines
    disagree on the NORAD id are skipped.
    """
    records: list[dict] = []
    pending_name: str | None = None
    pending_l1: str | None = None
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if _is_tle_line(s, "1"):
            pending_l1 = s
            continue
        if _is_tle_line(s, "2") and pending_l1 is not None:
            line1, line2 = pending_l1, s
            pending_l1 = None
            try:
                norad1 = _parse_norad(line1[2:7])
                norad2 = _parse_norad(line2[2:7])
                if norad1 != norad2:
                    continue
                epoch = tle_epoch(line1)
            except (ValueError, IndexError):
                continue
            name = (pending_name or f"OBJECT {norad1}").strip()
            pending_name = None
            records.append({
                "norad_id": norad1,
                "name": name,
                "intl_designator": _intl_designator(line1),
                "object_type": _object_type(name),
                "epoch": epoch.strftime(ISO_FMT),
                "line1": line1,
                "line2": line2,
            })
            continue
        # Anything else is a (candidate) name line.
        pending_name = s
        pending_l1 = None
    return records


def fetch_group(group: str, client: httpx.Client | None = None) -> list[dict]:
    """GET config.CELESTRAK_GP_URL?GROUP=<group>&FORMAT=tle and parse."""
    params = {"GROUP": group, "FORMAT": "tle"}
    if client is None:
        with httpx.Client(timeout=config.HTTP_TIMEOUT_S, follow_redirects=True) as c:
            resp = c.get(config.CELESTRAK_GP_URL, params=params)
    else:
        resp = client.get(config.CELESTRAK_GP_URL, params=params)
    resp.raise_for_status()
    return parse_tle_text(resp.text)


def merge_records(records) -> dict[int, dict]:
    """Dedupe an iterable of parsed records by norad_id, keeping newest epoch."""
    out: dict[int, dict] = {}
    for rec in records:
        cur = out.get(rec["norad_id"])
        if cur is None or rec["epoch"] > cur["epoch"]:
            out[rec["norad_id"]] = rec
    return out


def fetch_catalog(groups: list[str] | None = None) -> dict[int, dict]:
    """Fetch all groups (default config.CATALOG_GROUPS), dedupe by norad_id.

    Per-group failures are logged and SKIPPED - CelesTrak renames groups and
    rate-limits the big ones, and one dead group must never take down the
    whole ingest (it did once: 'last-30-days' 404ing failed every refresh).
    Raises only when every group fails, so a total outage still surfaces.
    """
    groups = groups if groups is not None else config.CATALOG_GROUPS
    all_records: list[dict] = []
    failures: list[tuple[str, Exception]] = []
    with httpx.Client(timeout=config.HTTP_TIMEOUT_S, follow_redirects=True) as client:
        for group in groups:
            try:
                all_records.extend(fetch_group(group, client=client))
            except httpx.HTTPError as exc:
                failures.append((group, exc))
                print(f"celestrak: group {group!r} failed, skipping: {exc}",
                      file=sys.stderr)
    if failures and not all_records:
        raise failures[0][1]
    return merge_records(all_records)


def upsert_catalog(conn, records: dict[int, dict]) -> dict:
    """Insert/update satellites and append tle_history rows.

    tle_history dedupe relies on the UNIQUE (satellite_id, epoch) constraint
    via INSERT OR IGNORE.
    """
    now = _now_iso()
    satellites_new = 0
    tles_added = 0
    for norad_id, rec in records.items():
        row = conn.execute(
            "SELECT id FROM satellites WHERE norad_id = ?", (norad_id,)
        ).fetchone()
        if row is None:
            cur = conn.execute(
                "INSERT INTO satellites (norad_id, name, object_type, intl_designator, last_updated)"
                " VALUES (?, ?, ?, ?, ?)",
                (norad_id, rec["name"], rec["object_type"], rec["intl_designator"], now),
            )
            sat_id = cur.lastrowid
            satellites_new += 1
        else:
            sat_id = row["id"]
            conn.execute(
                "UPDATE satellites SET name = ?, object_type = ?, intl_designator = ?,"
                " last_updated = ? WHERE id = ?",
                (rec["name"], rec["object_type"], rec["intl_designator"], now, sat_id),
            )
        cur = conn.execute(
            "INSERT OR IGNORE INTO tle_history (satellite_id, epoch, line1, line2, ingested_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (sat_id, rec["epoch"], rec["line1"], rec["line2"], now),
        )
        if cur.rowcount > 0:
            tles_added += 1
    conn.commit()
    return {"fetched": len(records), "satellites_new": satellites_new, "tles_added": tles_added}
