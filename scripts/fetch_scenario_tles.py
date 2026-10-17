#!/usr/bin/env python3
"""Fetch the ARCHIVED element sets for a Time Machine scenario.

Pulls every element set Space-Track published for the scenario's two
objects in the days before the event (gp_history class), keeps only those
with an epoch before the event, and writes
backend/app/scenarios/<stem>.archived.json with "provenance": "archived".
The backend prefers that file over the reconstructed one automatically.

Credentials come strictly from the environment (SPACETRACK_IDENTITY,
SPACETRACK_PASSWORD) through backend.app.ingestion.spacetrack.login - the
same session pattern the live ingest uses; they are never printed.

Usage (from the repo root):
    SPACETRACK_IDENTITY=... SPACETRACK_PASSWORD=... \\
        python scripts/fetch_scenario_tles.py [--scenario iridium_cosmos_2009]
                                              [--days 7]
One gp_history query per run (Space-Track etiquette).
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from backend.app import config  # noqa: E402
from backend.app.ingestion import spacetrack  # noqa: E402
from backend.app.scenarios import SCENARIO_DIR, ARCHIVED_SUFFIX  # noqa: E402
from backend.app.scenarios import archive  # noqa: E402


def main(argv=None, transport=None) -> int:
    """CLI entry. `transport` is an httpx transport hook for tests only."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenario", default="iridium_cosmos_2009",
                    help="scenario file stem in backend/app/scenarios/")
    ap.add_argument("--days", type=float, default=7.0,
                    help="how many days of history before the event to pull")
    ap.add_argument("--out", default=None,
                    help="output path (default: <stem>.archived.json beside the base)")
    args = ap.parse_args(argv)

    base_path = SCENARIO_DIR / f"{args.scenario}.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    event = datetime.fromisoformat(
        base["event"]["time_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    norads = [o["norad_id"] for o in base["objects"]]
    start = event - timedelta(days=args.days)
    end = event + timedelta(days=1)   # day-granular range; filtered below

    with httpx.Client(timeout=config.HTTP_TIMEOUT_S, follow_redirects=True,
                      transport=transport) as client:
        spacetrack.login(client)
        resp = client.get(archive.history_url(norads, start, end))
        resp.raise_for_status()
        rows = resp.json()

    elsets = archive.parse_history_rows(rows, before=event, norads=norads)
    for n in norads:
        print(f"NORAD {n}: {len(elsets.get(n, []))} element sets before the event",
              file=sys.stderr)
    out = archive.build_archived(base, elsets)
    out_path = (Path(args.out) if args.out
                else base_path.with_name(args.scenario + ARCHIVED_SUFFIX))
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"wrote {out_path} (provenance: archived)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, httpx.HTTPError) as exc:
        print(f"fetch_scenario_tles: {exc}", file=sys.stderr)
        sys.exit(1)
