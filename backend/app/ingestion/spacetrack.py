"""Space-Track.org public CDM (cdm_public) ingestion.

Auth
----
Credentials come STRICTLY from the environment: SPACETRACK_IDENTITY and
SPACETRACK_PASSWORD. They are read at call time, sent only as the form body of
the login POST, and never stored, echoed, or logged (error messages name the
variables, never their values). Login is a POST to
https://www.space-track.org/ajaxauth/login on an httpx.Client created with
follow_redirects=True; the session cookie persists on the client for the
subsequent query. API etiquette: a single query per refresh, limit 200,
ordered by CREATED desc, window restricted to future encounters (TCA > now).

cdm_public column mapping (normalize_row; defensive - rows missing any
required field, or with unparseable dates/numbers, are dropped)
---------------------------------------------------------------
    CDM_ID                  -> cdm_id                   int | None
    CREATED                 -> message_time             ISO-8601 Z (required)
    TCA                     -> tca                      ISO-8601 Z (required)
    MIN_RNG (m)             -> miss_distance_m          meters (required).
                               MIN_RNG is published in meters: live rows show
                               values like 642 alongside PC ~1.7e-4, which is
                               physically coherent only as 642 m (a 642 km
                               miss cannot carry that Pc), matching the CCSDS
                               CDM MISS_DISTANCE unit (m).
    PC                      -> probability_of_collision float | None (PC is
                               frequently absent/empty in cdm_public; None is
                               stored as a NULL Pc, which the schema allows)
    SAT_1_ID / SAT_2_ID     -> sat1_norad / sat2_norad  int (required)
    SAT_1_NAME / SAT_2_NAME -> sat1_name / sat2_name    str | None (informational)
    EMERGENCY_REPORTABLE    -> emergency_reportable     bool ('Y' -> True)
    (no such column)        -> relative_speed_ms        None - cdm_public does
                               not publish relative speed

Relative-velocity policy (documented decision)
----------------------------------------------
cdm_public carries no relative-speed column, but cdm_messages.
relative_velocity_ms is NOT NULL and fabricating a value (0.0 or any
placeholder) would be dishonest data. Rule applied by ingest_cdms, in order:

1. If the normalized CDM itself supplies relative_speed_ms (a fuller CDM
   class; never cdm_public), use it. This path can mint new conjunctions and
   minimal satellite rows (name 'OBJECT <norad>', type UNKNOWN) for objects
   unknown to the catalog.
2. Else, if the CDM's pair matches an existing ACTIVE conjunction (either
   object order, TCA within the record_events 30-minute tolerance), reuse
   that conjunction's latest CDM velocity - the pair's encounter velocity is
   a property of the event geometry and is stable across messages.
3. Else, if BOTH objects have TLEs in the catalog, derive the encounter
   velocity by SGP4-propagating the two latest TLEs at the message's TCA and
   taking |v1 - v2|. Relative velocity is robust to TLE positional staleness
   (it is set by the orbit geometry, not the along-track phase), so this is a
   real measurement, not a placeholder; the CDM's own operator miss distance
   and Pc are stored untouched. This path can mint new conjunctions for pairs
   the screener has not seen.
4. Else no velocity is derivable: the row is SKIPPED and counted in the
   stats key skipped_no_velocity. Nothing is written for skipped rows.

Other ingest choices: a matched conjunction's TCA is NOT rewritten from the
message (the screener's refined TCA stays authoritative; the message only
appends history). Duplicate messages are deduped on (conjunction_id,
message_time) within source='spacetrack' - scoping to our source avoids ever
colliding with a same-second screening row. Messages are processed in
message_time order regardless of fetch order (fetch is CREATED desc).

Full catalog (fetch_full_catalog)
---------------------------------
One authenticated gp-class query - EPOCH > now-30, the same predicates as
fetch_gp_tles - returning upsert_catalog-shaped records for EVERY object with
a fresh elset (~25-28k rows). Rows are normalized destructively (_gp_records
releases each raw row as its record is built) so peak memory stays roughly
the parsed JSON list alone.

Covariance enrichment - RESTRICTED cdm class (READ THIS FIRST)
--------------------------------------------------------------
Space-Track's full ``cdm`` class carries the per-object RTN position
covariance (SAT1_CR_R .. SAT2_CN_N) that ``cdm_public`` omits - but it sits
behind the USSPACECOM SSA data-sharing agreement and FREE ACCOUNTS DO NOT
HAVE ACCESS. Denial is signalled by HTTP 401/403, or - as observed live on
2026-07-23 with a free account - by HTTP 400 with body
``{"error":"Your Class Does Not Exist "}`` (Space-Track masks restricted
classes as nonexistent rather than admitting they are forbidden).

refresh_cdms ATTEMPTS the class once per call. On any denial signature,
fetch_covariances sets the module-level _COVARIANCE_ACCESS_DENIED flag, logs
exactly ONE stderr line, and never queries the class again within the
process. Other errors (timeouts, 5xx) are transient: logged, not flagged.

When rows do arrive (agreement-holding accounts), each row's two
lower-triangle RTN covariances are attached to the matching ingested CDM -
matched by CDM_ID against the just-ingested cdm_public batch when possible,
else by pair + TCA within the record_events tolerance - as
cdm_messages.covariance_json:

    {"sat1_rtn": [[3x3]], "sat2_rtn": [[3x3]], "source": "spacetrack-cdm"}

Matrices are in m^2, axis order R/T/N (radial, transverse/in-track,
normal/cross-track) - exactly the frame models.pc.compute_pc_from_covariance
consumes. Within the matched conjunction the update targets the
source='spacetrack' message with the same message_time, falling back to the
event's latest source='spacetrack' message (rows are applied in
message_time order, so the newest covariance wins the fallback).
"""
import json
import os
import sys
from datetime import datetime, timezone

import httpx

from backend.app import config
from backend.app.ingestion import celestrak
from backend.app.ingestion.celestrak import ISO_FMT
from backend.app.ingestion.screening import TCA_MATCH_TOLERANCE_S

BASE_URL = "https://www.space-track.org"
LOGIN_URL = f"{BASE_URL}/ajaxauth/login"
QUERY_LIMIT = 200


def _query_url(limit: int) -> str:
    """Single-query URL per Space-Track etiquette (httpx percent-encodes)."""
    return (f"{BASE_URL}/basicspacedata/query/class/cdm_public"
            f"/TCA/>now/orderby/CREATED desc/limit/{int(limit)}/format/json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FMT)


def _parse_iso(s: str) -> datetime:
    return datetime.strptime(s, ISO_FMT).replace(tzinfo=timezone.utc)


def _parse_dt(value) -> datetime | None:
    """Parse Space-Track date strings defensively.

    Accepts 'YYYY-MM-DD HH:MM:SS' and 'YYYY-MM-DDTHH:MM:SS', optional
    fractional seconds (truncated - the DB stores whole-second Z strings) and
    an optional trailing 'Z'. Returns None on anything unparseable.
    """
    if value is None:
        return None
    s = str(value).strip().replace("T", " ")
    if s.endswith("Z"):
        s = s[:-1]
    if "." in s:
        s = s[:s.index(".")]
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _to_int(value) -> int | None:
    f = _to_float(value)
    if f is None:
        return None
    try:
        return int(f)
    except (ValueError, OverflowError):
        return None


def _clean_name(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def normalize_row(row: dict) -> dict | None:
    """Map one raw cdm_public row to the normalized CDM dict (see module doc).

    Returns None when any required field (CREATED, TCA, MIN_RNG, SAT_1_ID,
    SAT_2_ID) is missing or unparseable, or when MIN_RNG is negative.
    Out-of-range PC values are demoted to None rather than dropping the row.
    """
    created = _parse_dt(row.get("CREATED"))
    tca = _parse_dt(row.get("TCA"))
    min_rng_m = _to_float(row.get("MIN_RNG"))
    sat1 = _to_int(row.get("SAT_1_ID"))
    sat2 = _to_int(row.get("SAT_2_ID"))
    if created is None or tca is None or min_rng_m is None or min_rng_m < 0.0 \
            or sat1 is None or sat2 is None:
        return None
    pc = _to_float(row.get("PC"))
    if pc is not None and not 0.0 <= pc <= 1.0:
        pc = None
    emergency = str(row.get("EMERGENCY_REPORTABLE") or "").strip().upper() == "Y"
    return {
        "cdm_id": _to_int(row.get("CDM_ID")),
        "message_time": created.strftime(ISO_FMT),
        "tca": tca.strftime(ISO_FMT),
        "miss_distance_m": min_rng_m,  # MIN_RNG is already meters (CCSDS)
        "relative_speed_ms": None,  # not published in cdm_public
        "probability_of_collision": pc,
        "sat1_norad": sat1,
        "sat1_name": _clean_name(row.get("SAT_1_NAME")),
        "sat2_norad": sat2,
        "sat2_name": _clean_name(row.get("SAT_2_NAME")),
        "emergency_reportable": emergency,
    }


def login(client: httpx.Client) -> None:
    """POST env credentials to ajaxauth/login; session cookie stays on client.

    Raises RuntimeError when the env vars are unset or Space-Track rejects the
    credentials. Error messages never contain the credential values.
    """
    identity = os.environ.get("SPACETRACK_IDENTITY")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not identity or not password:
        raise RuntimeError(
            "Space-Track credentials not configured: set SPACETRACK_IDENTITY"
            " and SPACETRACK_PASSWORD")
    resp = client.post(LOGIN_URL, data={"identity": identity, "password": password})
    resp.raise_for_status()
    if "Failed" in resp.text:
        raise RuntimeError("Space-Track login failed (credentials rejected)")


def fetch_cdms(client: httpx.Client | None = None, limit: int = QUERY_LIMIT) -> list[dict]:
    """Fetch + normalize the newest future-TCA cdm_public rows.

    When client is None, a client is created (follow_redirects=True - required
    for the Space-Track session flow) and login() runs with env credentials.
    A caller-supplied client is used as-is and assumed authenticated (tests
    inject an httpx.MockTransport client). Malformed rows are dropped by
    normalize_row.
    """
    if client is None:
        with httpx.Client(timeout=config.HTTP_TIMEOUT_S, follow_redirects=True) as c:
            login(c)
            return _fetch(c, limit)
    return _fetch(client, limit)


def _fetch(client: httpx.Client, limit: int) -> list[dict]:
    resp = client.get(_query_url(limit))
    resp.raise_for_status()
    rows = resp.json()
    if not isinstance(rows, list):
        raise RuntimeError("Space-Track returned an unexpected response shape")
    out = []
    for row in rows:
        rec = normalize_row(row) if isinstance(row, dict) else None
        if rec is not None:
            out.append(rec)
    return out


GP_PREDICATES = ("NORAD_CAT_ID,OBJECT_NAME,OBJECT_ID,OBJECT_TYPE,EPOCH,"
                 "TLE_LINE1,TLE_LINE2")


def _gp_records(rows: list) -> dict[int, dict]:
    """Normalize raw gp rows into celestrak.upsert_catalog-shaped records.

    Consumes the list destructively: each slot is released (set to None) as
    its record is built, so a full-catalog pull (~28k rows) peaks at roughly
    the parsed JSON list alone - the records dict mostly re-references the
    same strings. Malformed rows (missing norad/epoch/TLE lines) are dropped;
    unknown object types normalize to UNKNOWN.
    """
    records: dict[int, dict] = {}
    for idx in range(len(rows)):
        row = rows[idx]
        rows[idx] = None  # release the raw dict as soon as it is consumed
        if not isinstance(row, dict):
            continue
        norad = _to_int(row.get("NORAD_CAT_ID"))
        epoch = _parse_dt(row.get("EPOCH"))
        line1, line2 = row.get("TLE_LINE1"), row.get("TLE_LINE2")
        if norad is None or epoch is None or not line1 or not line2:
            continue
        otype = str(row.get("OBJECT_TYPE") or "UNKNOWN").strip().upper()
        if otype not in ("PAYLOAD", "ROCKET BODY", "DEBRIS"):
            otype = "UNKNOWN"
        records[norad] = {
            "name": _clean_name(row.get("OBJECT_NAME")) or f"OBJECT {norad}",
            "object_type": otype,
            "intl_designator": _clean_name(row.get("OBJECT_ID")),
            "epoch": epoch.strftime(ISO_FMT),
            "line1": line1.strip(),
            "line2": line2.strip(),
        }
    return records


def fetch_gp_tles(norads, client: httpx.Client) -> dict[int, dict]:
    """One batched, authenticated gp query for the given NORAD ids.

    Returns celestrak.upsert_catalog-shaped records {norad: {name,
    object_type, intl_designator, epoch, line1, line2}}. CDM participants are
    frequently outside the curated CelesTrak groups (dead payloads, debris),
    so this is how their TLEs enter the catalog. Ids absent from gp (decayed
    or unpublished) are simply missing from the result.
    """
    ids = ",".join(str(n) for n in sorted(set(norads)))
    if not ids:
        return {}
    resp = client.get(
        f"{BASE_URL}/basicspacedata/query/class/gp/NORAD_CAT_ID/{ids}"
        f"/predicates/{GP_PREDICATES}/format/json")
    resp.raise_for_status()
    rows = resp.json()
    if not isinstance(rows, list):
        raise RuntimeError("Space-Track returned an unexpected gp shape")
    return _gp_records(rows)


FULL_CATALOG_URL = (f"{BASE_URL}/basicspacedata/query/class/gp/EPOCH/>now-30"
                    f"/predicates/{GP_PREDICATES}/format/json")


def fetch_full_catalog(client: httpx.Client | None = None) -> dict[int, dict]:
    """Fetch the full Space-Track gp catalog (EPOCH newer than 30 days).

    one query per Space-Track etiquette, the same predicates as
    fetch_gp_tles, returning celestrak.upsert_catalog-shaped records for
    every object with a fresh elset - expect ~25-28k rows. Memory: the raw
    row list is consumed destructively by _gp_records, so the transient peak
    is roughly the parsed JSON alone (tens of MB), never rows + a full deep
    copy. When client is None, an authenticated client is created via login()
    with env credentials; a caller-supplied client is assumed authenticated.
    """
    if client is None:
        with httpx.Client(timeout=config.HTTP_TIMEOUT_S, follow_redirects=True) as c:
            login(c)
            return _fetch_full_catalog(c)
    return _fetch_full_catalog(client)


def _fetch_full_catalog(client: httpx.Client) -> dict[int, dict]:
    resp = client.get(FULL_CATALOG_URL)
    resp.raise_for_status()
    rows = resp.json()
    if not isinstance(rows, list):
        raise RuntimeError("Space-Track returned an unexpected gp shape")
    return _gp_records(rows)


def _has_tle(conn, norad_id: int) -> bool:
    return conn.execute(
        "SELECT 1 FROM satellites s JOIN tle_history t ON t.satellite_id = s.id"
        " WHERE s.norad_id = ? LIMIT 1", (norad_id,)).fetchone() is not None


# --- Restricted-cdm-class covariance enrichment -------------------------------
# FREE ACCOUNTS LACK ACCESS (no SSA data-sharing agreement) - see the module
# docstring. Once a denial is seen the flag stays set for the process lifetime:
# one stderr line total, never a retry.
_COVARIANCE_ACCESS_DENIED = False

# Lower-triangle RTN covariance components published per object (m^2), CCSDS
# CDM key names prefixed SAT1_/SAT2_ in the Space-Track cdm class.
_COV_COMPONENTS = ("CR_R", "CT_R", "CT_T", "CN_R", "CN_T", "CN_N")
CDM_COV_PREDICATES = ("CDM_ID,CREATION_DATE,TCA,SAT1_OBJECT_DESIGNATOR,"
                      "SAT2_OBJECT_DESIGNATOR,"
                      + ",".join(f"SAT{i}_{c}" for i in (1, 2)
                                 for c in _COV_COMPONENTS))


def _covariance_query_url(limit: int) -> str:
    return (f"{BASE_URL}/basicspacedata/query/class/cdm"
            f"/TCA/>now/orderby/CREATION_DATE desc/limit/{int(limit)}"
            f"/predicates/{CDM_COV_PREDICATES}/format/json")


def _rtn_matrix(row: dict, sat: str) -> list[list[float]] | None:
    """Build one object's symmetric 3x3 RTN covariance [m^2] from the
    SAT<i>_CR_R.. lower-triangle columns; None when any component is absent.

    Axis order (R, T, N) = (radial, transverse/in-track, normal/cross-track).
    """
    v = {c: _to_float(row.get(f"{sat}_{c}")) for c in _COV_COMPONENTS}
    if any(value is None for value in v.values()):
        return None
    return [[v["CR_R"], v["CT_R"], v["CN_R"]],
            [v["CT_R"], v["CT_T"], v["CN_T"]],
            [v["CN_R"], v["CN_T"], v["CN_N"]]]


def normalize_covariance_row(row: dict) -> dict | None:
    """Map one raw cdm-class row to a normalized covariance record.

    Returns None when CREATION_DATE, TCA, either object designator, or any of
    the twelve covariance components is missing/unparseable - a partial
    covariance is useless for Pc work. CDM_ID stays optional (used only to
    strengthen matching against the just-ingested cdm_public batch).
    """
    created = _parse_dt(row.get("CREATION_DATE"))
    tca = _parse_dt(row.get("TCA"))
    sat1 = _to_int(row.get("SAT1_OBJECT_DESIGNATOR"))
    sat2 = _to_int(row.get("SAT2_OBJECT_DESIGNATOR"))
    sat1_rtn = _rtn_matrix(row, "SAT1")
    sat2_rtn = _rtn_matrix(row, "SAT2")
    if created is None or tca is None or sat1 is None or sat2 is None \
            or sat1_rtn is None or sat2_rtn is None:
        return None
    return {
        "cdm_id": _to_int(row.get("CDM_ID")),
        "message_time": created.strftime(ISO_FMT),
        "tca": tca.strftime(ISO_FMT),
        "sat1_norad": sat1,
        "sat2_norad": sat2,
        "sat1_rtn": sat1_rtn,
        "sat2_rtn": sat2_rtn,
    }


def _is_access_denial(resp: httpx.Response) -> bool:
    """True for the restricted-class denial signatures.

    401/403 are the documented restricted-access codes; a free account was
    additionally observed live (2026-07-23) to get HTTP 400 with
    '"error":"Your Class Does Not Exist "' - Space-Track masks restricted
    classes as nonexistent for accounts without the SSA agreement.
    """
    if resp.status_code in (401, 403):
        return True
    return resp.status_code == 400 and "does not exist" in resp.text.lower()


def fetch_covariances(client: httpx.Client,
                      limit: int = QUERY_LIMIT) -> list[dict] | None:
    """Attempt one restricted cdm-class query; None when access is denied.

    First denial sets _COVARIANCE_ACCESS_DENIED and logs a single stderr
    line; every later call returns None without issuing a request (never
    retry within the process - free accounts lack the SSA agreement, see the
    module docstring). Non-denial HTTP errors raise for the caller to treat
    as transient. Rows failing normalize_covariance_row are dropped.
    """
    global _COVARIANCE_ACCESS_DENIED
    if _COVARIANCE_ACCESS_DENIED:
        return None
    resp = client.get(_covariance_query_url(limit))
    if _is_access_denial(resp):
        _COVARIANCE_ACCESS_DENIED = True
        print(f"[spacetrack] restricted cdm-class query denied"
              f" (HTTP {resp.status_code}): free accounts lack the SSA"
              " data-sharing agreement — covariance enrichment disabled for"
              " this process", file=sys.stderr)
        return None
    resp.raise_for_status()
    rows = resp.json()
    if not isinstance(rows, list):
        raise RuntimeError("Space-Track returned an unexpected cdm shape")
    out = []
    for row in rows:
        rec = normalize_covariance_row(row) if isinstance(row, dict) else None
        if rec is not None:
            out.append(rec)
    return out


def apply_covariances(conn, cov_rows: list[dict],
                      cdms: list[dict] | None = None) -> int:
    """Attach normalized covariance rows to matching ingested CDM messages.

    Matching (module docstring): a row whose CDM_ID appears in the
    just-ingested cdm_public batch adopts that message's pair/TCA/
    message_time (guaranteed alignment); otherwise the row's own pair + TCA
    must match an active conjunction within the record_events tolerance.
    Within the conjunction, the source='spacetrack' message with the same
    message_time is updated, else the latest source='spacetrack' message
    (rows are applied in message_time order so the newest covariance wins).
    Returns the number of messages updated.
    """
    by_cdm_id = {c["cdm_id"]: c for c in (cdms or [])
                 if c.get("cdm_id") is not None}
    updated = 0
    for cov in sorted(cov_rows, key=lambda c: c["message_time"]):
        ingested = by_cdm_id.get(cov["cdm_id"]) if cov.get("cdm_id") else None
        source = ingested if ingested is not None else cov
        a_row = conn.execute("SELECT id FROM satellites WHERE norad_id = ?",
                             (source["sat1_norad"],)).fetchone()
        b_row = conn.execute("SELECT id FROM satellites WHERE norad_id = ?",
                             (source["sat2_norad"],)).fetchone()
        if a_row is None or b_row is None:
            continue
        conj = _match_conjunction(conn, a_row["id"], b_row["id"],
                                  _parse_iso(source["tca"]))
        if conj is None:
            continue
        cov_json = json.dumps({"sat1_rtn": cov["sat1_rtn"],
                               "sat2_rtn": cov["sat2_rtn"],
                               "source": "spacetrack-cdm"})
        cur = conn.execute(
            "UPDATE cdm_messages SET covariance_json = ?"
            " WHERE conjunction_id = ? AND source = 'spacetrack'"
            " AND message_time = ?",
            (cov_json, conj["id"], source["message_time"]))
        if cur.rowcount == 0:
            row = conn.execute(
                "SELECT id FROM cdm_messages WHERE conjunction_id = ?"
                " AND source = 'spacetrack'"
                " ORDER BY message_time DESC, id DESC LIMIT 1",
                (conj["id"],)).fetchone()
            if row is None:
                continue
            conn.execute("UPDATE cdm_messages SET covariance_json = ?"
                         " WHERE id = ?", (cov_json, row["id"]))
        updated += 1
    conn.commit()
    return updated


def _attempt_covariance_refresh(conn, client: httpx.Client,
                                cdms: list[dict]) -> int:
    """Non-fatal covariance attempt for refresh_cdms; returns rows updated.

    Denial is absorbed inside fetch_covariances (flag + one stderr line);
    anything else (timeout, 5xx, bad payload) is transient - logged to
    stderr, the CDM ingest that already succeeded is never jeopardized.
    """
    try:
        rows = fetch_covariances(client)
        if not rows:
            return 0
        return apply_covariances(conn, rows, cdms)
    except Exception as exc:
        print(f"[spacetrack] covariance enrichment failed:"
              f" {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0


def refresh_cdms(conn) -> dict:
    """login -> fetch CDMs -> fetch missing participants' TLEs -> ingest ->
    attempt restricted-class covariance enrichment (once; see module doc).

    The pipeline entry point: one authenticated client for the whole flow.
    Stats gain "covariance_updated" (0 whenever the restricted class is
    inaccessible - the free-account norm).
    """
    with httpx.Client(timeout=config.HTTP_TIMEOUT_S, follow_redirects=True) as c:
        login(c)
        cdms = _fetch(c, QUERY_LIMIT)
        stats = ingest_cdms(conn, cdms, client=c)
        stats["covariance_updated"] = _attempt_covariance_refresh(conn, c, cdms)
        return stats


def ensure_satellite(conn, norad_id: int) -> int:
    """Return the satellite id for norad_id, creating a minimal catalog row
    (name 'OBJECT <norad>', type UNKNOWN) when the object is unknown."""
    row = conn.execute(
        "SELECT id FROM satellites WHERE norad_id = ?", (norad_id,)).fetchone()
    if row is not None:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO satellites (norad_id, name, object_type, last_updated)"
        " VALUES (?, ?, 'UNKNOWN', ?)",
        (norad_id, f"OBJECT {norad_id}", _now_iso()))
    return cur.lastrowid


def _match_conjunction(conn, sat_a_id: int, sat_b_id: int, tca: datetime):
    """Nearest ACTIVE conjunction for the pair (either object order) whose TCA
    is within the record_events tolerance; None when nothing matches."""
    best_row, best_diff = None, None
    for p_id, s_id in ((sat_a_id, sat_b_id), (sat_b_id, sat_a_id)):
        rows = conn.execute(
            "SELECT id, time_of_closest_approach FROM conjunctions"
            " WHERE primary_object_id = ? AND secondary_object_id = ?"
            " AND status = 'active'", (p_id, s_id)).fetchall()
        for row in rows:
            diff = abs((_parse_iso(row["time_of_closest_approach"]) - tca)
                       .total_seconds())
            if diff < TCA_MATCH_TOLERANCE_S and (best_diff is None or diff < best_diff):
                best_row, best_diff = row, diff
    return best_row


def _is_republished(conn, conj_id: int, cdm: dict) -> bool:
    """True when this conjunction already holds the same Space-Track message:
    the exact message_time, or identical miss/Pc (null-safe) within 900 s."""
    t_new = _parse_iso(cdm["message_time"])
    for row in conn.execute(
            "SELECT message_time, miss_distance_m, probability_of_collision"
            " FROM cdm_messages WHERE conjunction_id = ? AND source = 'spacetrack'",
            (conj_id,)).fetchall():
        if row["message_time"] == cdm["message_time"]:
            return True
        if (row["miss_distance_m"] == cdm["miss_distance_m"]
                and row["probability_of_collision"] == cdm["probability_of_collision"]
                and abs((_parse_iso(row["message_time"]) - t_new)
                        .total_seconds()) <= 900):
            return True
    return False


def _latest_velocity(conn, conjunction_id: int) -> float | None:
    row = conn.execute(
        "SELECT relative_velocity_ms FROM cdm_messages WHERE conjunction_id = ?"
        " ORDER BY message_time DESC, id DESC LIMIT 1", (conjunction_id,)).fetchone()
    return None if row is None else row["relative_velocity_ms"]


def _sgp4_rel_velocity(conn, norad_a: int, norad_b: int, tca) -> float | None:
    """Encounter velocity |v1 - v2| (m/s) from the two latest TLEs at TCA.

    Relative velocity is set by the orbit geometry, not the along-track
    phase, so it is robust to TLE staleness - a real measurement, unlike a
    placeholder. None when either object lacks a propagatable TLE.
    """
    from backend.app.ingestion import propagate

    import numpy as np

    vels = []
    for norad in (norad_a, norad_b):
        row = conn.execute(
            """
            SELECT t.line1 AS line1, t.line2 AS line2
            FROM satellites s JOIN tle_history t ON t.satellite_id = s.id
            WHERE s.norad_id = ?
            ORDER BY t.epoch DESC, t.id DESC LIMIT 1
            """,
            (norad,)).fetchone()
        if row is None:
            return None
        try:
            _r, v = propagate.positions_at(row["line1"], row["line2"], [tca])
        except Exception:
            return None
        vels.append(np.asarray(v[0]))
    return float(np.linalg.norm(vels[0] - vels[1])) * 1000.0  # km/s -> m/s


def ingest_cdms(conn, cdms: list[dict], client: httpx.Client | None = None) -> dict:
    """Ingest normalized CDMs (see fetch_cdms) under the velocity policy
    documented in the module docstring. Returns stats.

    With an authenticated client, participants missing from the catalog get
    their TLEs fetched first via one batched gp query (fetch_gp_tles) so the
    SGP4 velocity derivation can run; client=None (tests, offline) skips the
    fetch and rows without derivable velocity are counted as skipped.
    """
    stats = {
        "fetched": len(cdms),
        "cdms_added": 0,
        "conjunctions_new": 0,
        "conjunctions_matched": 0,
        "satellites_created": 0,
        "tles_fetched": 0,
        "skipped_no_velocity": 0,
        "skipped_duplicate": 0,
    }
    if client is not None and cdms:
        missing = {n for c in cdms for n in (c["sat1_norad"], c["sat2_norad"])
                   if not _has_tle(conn, n)}
        if missing:
            records = fetch_gp_tles(missing, client)
            up = celestrak.upsert_catalog(conn, records)
            stats["tles_fetched"] = up.get("tles_added", 0)
    now_iso = _now_iso()
    for cdm in sorted(cdms, key=lambda c: c["message_time"]):
        tca = _parse_iso(cdm["tca"])
        a_row = conn.execute("SELECT id FROM satellites WHERE norad_id = ?",
                             (cdm["sat1_norad"],)).fetchone()
        b_row = conn.execute("SELECT id FROM satellites WHERE norad_id = ?",
                             (cdm["sat2_norad"],)).fetchone()
        conj = None
        if a_row is not None and b_row is not None:
            conj = _match_conjunction(conn, a_row["id"], b_row["id"], tca)

        # Velocity policy (module docstring): message-supplied, else reused
        # from the matched event's latest CDM, else derived by SGP4 at TCA,
        # else the row is not storable - skip and count.
        velocity = cdm.get("relative_speed_ms")
        if velocity is None and conj is not None:
            velocity = _latest_velocity(conn, conj["id"])
        if velocity is None:
            velocity = _sgp4_rel_velocity(
                conn, cdm["sat1_norad"], cdm["sat2_norad"], tca)
        if velocity is None:
            stats["skipped_no_velocity"] += 1
            continue

        # velocity is real - safe to materialize rows
        if conj is not None:
            conj_id = conj["id"]
            stats["conjunctions_matched"] += 1
        else:
            if a_row is None:
                a_row = {"id": ensure_satellite(conn, cdm["sat1_norad"])}
                stats["satellites_created"] += 1
            if b_row is None:
                b_row = {"id": ensure_satellite(conn, cdm["sat2_norad"])}
                stats["satellites_created"] += 1
            tca_iso = tca.strftime(ISO_FMT)
            try:
                cur = conn.execute(
                    "INSERT INTO conjunctions (primary_object_id, secondary_object_id,"
                    " time_of_closest_approach, first_seen_at, status)"
                    " VALUES (?, ?, ?, ?, 'active')",
                    (a_row["id"], b_row["id"], tca_iso, now_iso))
                conj_id = cur.lastrowid
                stats["conjunctions_new"] += 1
            except Exception:  # UNIQUE (pair, tca) already present
                conj_id = conn.execute(
                    "SELECT id FROM conjunctions WHERE primary_object_id = ?"
                    " AND secondary_object_id = ? AND time_of_closest_approach = ?",
                    (a_row["id"], b_row["id"], tca_iso)).fetchone()["id"]
                stats["conjunctions_matched"] += 1

        # Space-Track republishes each CDM from the other object's
        # perspective minutes later with identical miss/Pc - the exact
        # message_time check alone stored every update twice (doubled CDM
        # counts, stair-stepped Pc trends, corrupted msg_gap features).
        # Compared in Python: SQLite's `is ?` and strftime() do not exist on
        # Postgres, and a failed statement there aborts the whole refresh.
        dup = _is_republished(conn, conj_id, cdm)
        if dup:
            stats["skipped_duplicate"] += 1
            continue
        conn.execute(
            "INSERT INTO cdm_messages (conjunction_id, message_time, miss_distance_m,"
            " relative_velocity_ms, probability_of_collision, source)"
            " VALUES (?, ?, ?, ?, ?, 'spacetrack')",
            (conj_id, cdm["message_time"], cdm["miss_distance_m"], velocity,
             cdm["probability_of_collision"]))
        stats["cdms_added"] += 1
    conn.commit()
    return stats
