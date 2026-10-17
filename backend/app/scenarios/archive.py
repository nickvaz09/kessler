"""Archived element sets for a scenario, from Space-Track's gp_history class.

scripts/fetch_scenario_tles.py is the CLI; the logic lives here so it can be
unit-tested with a fixture (no network):

  history_url(norads, start, end)      the single gp_history query
  parse_history_rows(rows, before)     raw rows -> {norad: [elset, ...]}
  build_archived(base, elsets, ...)    base scenario -> archived scenario dict

Only element sets with an epoch strictly before the event are kept - the
replay must never see post-collision tracking. Rows are validated with the
same defensive parsers the live Space-Track ingest uses; malformed rows are
dropped. Duplicate epochs keep the first row seen.
"""
import copy
from datetime import datetime, timezone

from backend.app.ingestion.celestrak import ISO_FMT, tle_epoch
from backend.app.ingestion.spacetrack import BASE_URL, _parse_dt, _to_int
from backend.app.scenarios import epoch_iso

HISTORY_PREDICATES = "NORAD_CAT_ID,OBJECT_NAME,EPOCH,CREATION_DATE,TLE_LINE1,TLE_LINE2"


def history_url(norads, start: datetime, end: datetime) -> str:
    ids = ",".join(str(n) for n in sorted(set(int(n) for n in norads)))
    rng = f"{start.strftime('%Y-%m-%d')}--{end.strftime('%Y-%m-%d')}"
    return (f"{BASE_URL}/basicspacedata/query/class/gp_history"
            f"/NORAD_CAT_ID/{ids}/EPOCH/{rng}/orderby/EPOCH asc"
            f"/predicates/{HISTORY_PREDICATES}/format/json")


def _valid_tle(line1: str, line2: str, norad: int) -> bool:
    if len(line1) < 69 or len(line2) < 69:
        return False
    if not (line1.startswith("1 ") and line2.startswith("2 ")):
        return False
    try:
        if int(line1[2:7]) != norad or int(line2[2:7]) != norad:
            return False
        tle_epoch(line1)
    except ValueError:
        return False
    return True


def parse_history_rows(rows, before: datetime, norads=None) -> dict[int, list[dict]]:
    """gp_history rows -> {norad: [{epoch, line1, line2}]} sorted by epoch.

    Keeps rows whose EPOCH < `before` (the event time) and, when `norads` is
    given, only those objects.
    """
    if not isinstance(rows, list):
        raise ValueError("gp_history returned an unexpected shape")
    wanted = None if norads is None else {int(n) for n in norads}
    before = before.astimezone(timezone.utc)
    out: dict[int, dict[str, dict]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        norad = _to_int(row.get("NORAD_CAT_ID"))
        epoch = _parse_dt(row.get("EPOCH"))
        line1 = str(row.get("TLE_LINE1") or "").strip()
        line2 = str(row.get("TLE_LINE2") or "").strip()
        if norad is None or epoch is None or (wanted and norad not in wanted):
            continue
        # A TLE is published hours after its epoch: a replay run must only see
        # sets that existed at the run time, so keep the publication time
        # (CREATION_DATE) when Space-Track supplies it and filter on it.
        published = _parse_dt(row.get("CREATION_DATE"))
        if epoch >= before or (published is not None and published >= before):
            continue
        if not _valid_tle(line1, line2, norad):
            continue
        key = epoch_iso(line1)
        rec = {"epoch": key, "line1": line1, "line2": line2}
        if published is not None:
            rec["published"] = published.astimezone(timezone.utc).strftime(ISO_FMT)
        out.setdefault(norad, {}).setdefault(key, rec)
    return {n: [recs[k] for k in sorted(recs)] for n, recs in out.items()}


def build_archived(base: dict, elsets: dict[int, list[dict]],
                   fetched_at: datetime | None = None) -> dict:
    """The archived twin of a base scenario: same metadata, archived elsets.

    Raises ValueError when either object has no archived element set before
    the event (the reconstructed file then stays in use).
    """
    out = copy.deepcopy(base)
    for obj in out["objects"]:
        recs = elsets.get(int(obj["norad_id"])) or []
        if not recs:
            raise ValueError(f"no archived element set for NORAD {obj['norad_id']}"
                             " before the event")
        obj.pop("line1", None)
        obj.pop("line2", None)
        obj["elsets"] = recs
    fetched = (fetched_at or datetime.now(timezone.utc)).strftime(ISO_FMT)
    out["provenance"] = "archived"
    out["provenance_label"] = "Archived element sets from Space-Track"
    out["provenance_note"] = (
        "Element sets are the archived two-line element sets published by"
        " Space-Track (gp_history) before the event, fetched " + fetched + "."
        " Each replayed screening run uses only the element sets published"
        " before it. Screening, closest-approach refinement and collision"
        " probability are computed by the same code Kessler runs on the live"
        " catalog.")
    out["replay_hint"] = None
    out.pop("reconstruction", None)
    out["archive"] = {"source": "Space-Track gp_history", "fetched_at": fetched,
                      "elset_counts": {str(k): len(v) for k, v in elsets.items()}}
    return out
