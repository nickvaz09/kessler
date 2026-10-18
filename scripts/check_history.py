#!/usr/bin/env python3
"""Check the original version history against the paper's pre-registration table.

This repository starts from a clean snapshot. The original history, which holds
the commits that date each VIGIL Lab round, is kept as a git bundle that editors
and reviewers can ask for. Given the bundle, this checks its SHA-256, prints when
each round's pre-registration, frozen code and first results were committed
(UTC), and checks that every round ran in that order.

Usage (from the repo root):
    python scripts/check_history.py kessler-history-2026-10-17.bundle
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SHA256 = "518390d424eb1a793daab90e9a5a3a2798fe8b3b3ddddf8f790316233937875d"

# (round, pre-registration, frozen code, results); one commit can hold two roles
ROUNDS = [
    ("1", "c98a79b", "034c879", "02428e7"),
    ("2", "280ce55", "280ce55", "ad70749"),
    ("3", "c568d6a", "948ad3a", "7b603d2"),
    ("4", "2ebc3e3", "6e8b6e2", "c1c5290"),
]
POSTHOC = ("6a3bd02", "7c54e66")    # first and last post-hoc commit after round 4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def commit_times(bundle: Path, hashes) -> dict[str, datetime]:
    """Committer time (UTC) of each commit, read from a bare clone of the bundle."""
    with tempfile.TemporaryDirectory() as tmp:
        git_dir = Path(tmp) / "history.git"
        subprocess.run(["git", "clone", "--quiet", "--bare", str(bundle), str(git_dir)], check=True,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        out = {}
        for h in sorted(set(hashes)):
            ts = subprocess.run(["git", f"--git-dir={git_dir}", "log", "-1", "--format=%ct", h],
                                capture_output=True, text=True, check=True).stdout.strip()
            out[h] = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    return out


def check(times: dict, rounds=ROUNDS, posthoc=POSTHOC) -> list[str]:
    """Problems found; empty when every round ran in order."""
    bad = [f"round {r}: commits out of order" for r, pre, code, res in rounds
           if not times[pre] <= times[code] < times[res]]
    if posthoc and not times[rounds[-1][3]] < times[posthoc[0]] <= times[posthoc[1]]:
        bad.append("post-hoc commits do not follow the last round")
    return bad


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Check the original version history against the paper")
    p.add_argument("bundle", type=Path)
    a = p.parse_args(argv)
    digest = sha256(a.bundle)
    print(f"sha256 {digest} ({'matches' if digest == SHA256 else 'does NOT match'} the paper)")
    t = commit_times(a.bundle, [h for row in ROUNDS for h in row[1:]] + list(POSTHOC))

    def cell(h):
        return f"{h} {t[h]:%d %b %H:%M}"

    print(f"{'round':<6} {'pre-registration':<22} {'frozen code':<22} results")
    for r, pre, code, res in ROUNDS:
        print(f"{r:<6} {cell(pre):<22} {cell(code):<22} {cell(res)}")
    print(f"post hoc: {cell(POSTHOC[0])} to {cell(POSTHOC[1])} (UTC)")
    bad = check(t)
    for b in bad:
        print(b)
    ok = digest == SHA256 and not bad
    print("ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
