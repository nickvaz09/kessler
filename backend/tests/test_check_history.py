"""scripts/check_history.py, on a small history bundled here."""
import importlib.util
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("check_history", ROOT / "scripts" / "check_history.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bundle(tmp_path, dates):
    repo = tmp_path / "repo"
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env)
    shas = []
    for i, d in enumerate(dates):
        (repo / "f.txt").write_text(str(i))
        subprocess.run(["git", "-C", str(repo), "add", "f.txt"], check=True, env=env)
        subprocess.run(["git", "-c", "commit.gpgsign=false", "-C", str(repo), "commit", "-q", "-m", f"c{i}"],
                       check=True, env={**env, "GIT_AUTHOR_DATE": d, "GIT_COMMITTER_DATE": d})
        shas.append(subprocess.run(["git", "-C", str(repo), "rev-parse", "--short=7", "HEAD"],
                                   capture_output=True, text=True, check=True).stdout.strip())
    bundle = tmp_path / "history.bundle"
    subprocess.run(["git", "-C", str(repo), "bundle", "create", str(bundle), "--all"],
                   check=True, env=env, capture_output=True)
    return bundle, shas


def test_order_check_passes_and_catches_a_swap(tmp_path):
    mod = _load()
    bundle, (a, b, c) = _bundle(tmp_path, ["2026-01-01 10:00:00 +0000", "2026-01-01 11:00:00 +0000",
                                           "2026-01-01 12:00:00 +0000"])
    t = mod.commit_times(bundle, [a, b, c])
    assert t[a] < t[b] < t[c]
    assert mod.check(t, rounds=[("1", a, b, c)], posthoc=None) == []
    assert mod.check(t, rounds=[("1", c, b, a)], posthoc=None) == ["round 1: commits out of order"]
    # the same commit can be both the pre-registration and the frozen code
    assert mod.check(t, rounds=[("1", a, a, b)], posthoc=(c, c)) == []


def test_the_paper_table_is_listed():
    mod = _load()
    assert [r[0] for r in mod.ROUNDS] == ["1", "2", "3", "4"]
    assert len(mod.SHA256) == 64 and len(mod.POSTHOC) == 2
