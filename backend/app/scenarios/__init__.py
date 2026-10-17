"""Time Machine scenarios: historical conjunctions replayed through Kessler.

A scenario is a JSON file in this directory, ``<stem>.json``, describing a
historical event (published facts + sources) and the two objects involved
with their element sets. When ``<stem>.archived.json`` exists next to it -
written by scripts/fetch_scenario_tles.py from Space-Track's gp_history
archive - that file is PREFERRED: it carries the same metadata with the
archived element sets and ``"provenance": "archived"``.

Element sets per object are normalized to ``elsets``: a list of
{epoch, line1, line2} sorted by epoch. A reconstructed scenario carries one
elset per object (top-level line1/line2); an archived one carries every
element set published in the days before the event, so each replayed
screening run can use only what had been published by then.

Scenario ids are the JSON "id" field (e.g. "iridium-cosmos-2009").
"""
import json
from datetime import timedelta
from pathlib import Path

from backend.app.ingestion.celestrak import ISO_FMT, tle_epoch

SCENARIO_DIR = Path(__file__).resolve().parent
ARCHIVED_SUFFIX = ".archived.json"


def epoch_iso(line1: str) -> str:
    """TLE epoch as ISO Z, rounded to the nearest second (the day-of-year
    float otherwise truncates 11:43:12 to 11:43:11)."""
    ep = tle_epoch(line1) + timedelta(microseconds=500_000)
    return ep.replace(microsecond=0).strftime(ISO_FMT)


class ScenarioError(ValueError):
    """A scenario file is present but malformed."""


def _normalize_object(obj: dict) -> dict:
    elsets = []
    for rec in obj.get("elsets") or []:
        if rec.get("line1") and rec.get("line2"):
            elsets.append({"line1": rec["line1"], "line2": rec["line2"]})
    if not elsets and obj.get("line1") and obj.get("line2"):
        elsets.append({"line1": obj["line1"], "line2": obj["line2"]})
    if not elsets:
        raise ScenarioError(f"object {obj.get('norad_id')} has no element set")
    for rec in elsets:
        rec["epoch"] = epoch_iso(rec["line1"])
    elsets.sort(key=lambda r: r["epoch"])
    out = dict(obj)
    out.pop("line1", None)
    out.pop("line2", None)
    out["elsets"] = elsets
    return out


def _load_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("id", "title", "provenance", "event", "objects"):
        if key not in data:
            raise ScenarioError(f"{path.name}: missing '{key}'")
    if len(data["objects"]) != 2:
        raise ScenarioError(f"{path.name}: a scenario pairs exactly two objects")
    roles = sorted(o.get("role") for o in data["objects"])
    if roles != ["primary", "secondary"]:
        raise ScenarioError(f"{path.name}: objects need roles primary + secondary")
    data["objects"] = [_normalize_object(o) for o in data["objects"]]
    data["_file"] = path.name
    return data


def _base_files() -> list[Path]:
    return sorted(p for p in SCENARIO_DIR.glob("*.json")
                  if not p.name.endswith(ARCHIVED_SUFFIX))


def resolve_file(base: Path) -> Path:
    """The archived twin when present, else the base (reconstructed) file."""
    archived = base.with_name(base.stem + ARCHIVED_SUFFIX)
    return archived if archived.exists() else base


def list_scenarios() -> list[dict]:
    """Every scenario (archived twin preferred), sorted by base file name."""
    out = []
    for base in _base_files():
        try:
            out.append(_load_file(resolve_file(base)))
        except (OSError, ValueError, KeyError):
            continue  # a malformed file never takes the listing down
    return out


def get_scenario(scenario_id: str) -> dict | None:
    for scn in list_scenarios():
        if scn["id"] == scenario_id:
            return scn
    return None


def primary(scn: dict) -> dict:
    return next(o for o in scn["objects"] if o["role"] == "primary")


def secondary(scn: dict) -> dict:
    return next(o for o in scn["objects"] if o["role"] == "secondary")


def elset_at(obj: dict, when_iso: str) -> dict | None:
    """Latest element set published at or before `when_iso` (ISO Z). Archived
    sets carry their Space-Track publication time; reconstructed ones only
    an epoch, which then stands in for publication."""
    best = None
    for rec in obj["elsets"]:
        if rec.get("published", rec["epoch"]) <= when_iso:
            best = rec
    return best


def latest_elset(obj: dict) -> dict:
    return obj["elsets"][-1]
