"""VIGIL Lab CLI.

    .venv/bin/python -m backend.app.models.lab \\
        --train data/kelvins/train_data.csv \\
        --test data/kelvins/test_data.csv --test-targets <final-CDM file> --final \\
        --repeats 5 --folds 5 --out reports/vigil-lab

Writes <out>/lab_report.json and <out>/lab_report.md. With --export, gated
serving-track models are written to --export-dir (never over production).
Add --pool-test-into-cv to put the official test events into cross-
validation too (more positives for VIGIL-E; there is then no sealed test,
and the report says so).

--test-targets scores the sealed official test, whose numbers are final the
first time they are seen (sealed_test ignores --repeats/--folds/--bootstrap),
so it is refused without --final: smoke and debug runs leave it out.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from backend.app.models.lab import candidates as C
from backend.app.models.lab import data, gate, protocol, report


def _code_version() -> dict:
    """The commit the run used, so a sealed result can be tied to frozen code."""
    import subprocess

    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                                  cwd=Path(__file__).resolve().parent).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain", "--", ":/backend/app/models"))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="VIGIL Lab — benchmark-grade risk-model evaluation")
    p.add_argument("--train", required=True)
    p.add_argument("--test", default=None, help="official test inputs (test_data.csv)")
    p.add_argument("--test-targets", default=None, help="official final CDMs for the test events")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--budget", type=float, default=0.05, help="VIGIL-E alert budget (fraction)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-ceiling", action="store_true")
    p.add_argument("--pool-test-into-cv", action="store_true")
    p.add_argument("--out", default="reports/vigil-lab")
    p.add_argument("--export", action="store_true")
    p.add_argument("--export-dir", default="data/models/lab")
    p.add_argument("--limit-events", type=int, default=0, help="debug: subsample events")
    p.add_argument("--final", action="store_true",
                   help="required with --test-targets: this is the one scoring of the sealed test")
    p.add_argument("--require-cv-go", action="store_true",
                   help="score the sealed test only if the VIGIL serving primary beats LRP in CV "
                        "at the official test's class mix (Round 2 pre-registration)")
    p.add_argument("--force-overwrite", action="store_true",
                   help="allow replacing a report that holds sealed-test results")
    a = p.parse_args(argv)
    if a.test_targets and not a.final:
        p.error("--test-targets scores the sealed official test, which is looked at once. "
                "Smoke/debug runs go without it; pass --final for the real run.")
    prior = Path(a.out) / "lab_report.json"
    if prior.exists() and not a.force_overwrite:
        try:
            sealed_before = json.loads(prior.read_text()).get("test") is not None
        except (OSError, ValueError):
            sealed_before = False
        if sealed_before:
            p.error(f"{prior} holds sealed-test results, which cannot be regenerated without "
                    "scoring the sealed test again; choose another --out (or --force-overwrite)")

    t0 = time.time()
    train, test = data.load(a.train, a.test, a.test_targets)
    if a.limit_events:
        import random
        random.Random(a.seed).shuffle(train)
        train = train[:a.limit_events]
    print(f"loaded {len(train)} train events, {len(test)} test events with targets")
    mode = {"pooled": bool(a.pool_test_into_cv and test), "n_pooled_test_events": 0,
            "targets_supplied": bool(a.test_targets), "code": _code_version()}
    if a.test and not a.test_targets:
        print("test inputs given without --test-targets — they cannot be scored; no sealed test")
    if a.pool_test_into_cv and test:
        mode["n_pooled_test_events"] = len(test)
        train, test = train + test, []
        print("pooling the official test events into CV — no sealed test this run")

    augment = not a.no_augment
    cv = protocol.cross_validate(train, a.repeats, a.folds, a.seed, a.bootstrap, a.budget, augment)
    if a.require_cv_go and test:
        # Round 2 pre-registration: the sealed test's last look is spent only
        # if the VIGIL primary beats LRP in training-file CV at the official
        # test's published class mix
        prim = C.PRIMARY.get("vigil_serving")
        dlb = cv["regression"][prim]["bench_mix"]["delta_L_pct"]["mean"] if prim else float("nan")
        go = bool(dlb == dlb and dlb < 0)
        mode["cv_go"] = {"candidate": prim, "cv_bench_mix_delta_L_pct": dlb, "go": go}
        print(f"go/no-go: {prim} CV ΔL at the benchmark mix {dlb:+.2f}% → "
              f"{'GO: scoring the sealed test' if go else 'NO-GO: the sealed test stays unscored'}")
        if not go:
            test = []
    sealed = protocol.sealed_test(train, test, a.seed, a.budget, augment) if test else None
    ceil = None if a.no_ceiling else protocol.ceiling(train, 1, a.folds, a.seed)
    run = {"cv": cv, "test": sealed, "ceiling": ceil, "mode": mode,
           "gate_vigil": gate.vigil(cv, sealed), "gate_vigil_e": gate.vigil_e(cv, sealed),
           "config": vars(a), "runtime_s": round(time.time() - t0, 1)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lab_report.json").write_text(json.dumps(run, indent=2, default=float))
    (out / "lab_report.md").write_text(report.markdown(run))
    print(f"\nreport → {out / 'lab_report.md'}")

    if a.export:
        from backend.app.models.lab import export
        all_events = train + test
        d = Path(a.export_dir)
        gv = run["gate_vigil"].get("serving_exportable")
        ge = run["gate_vigil_e"].get("serving")
        if ge and ge["candidate"] != "e_serving":
            # export_vigil_e writes the e_serving classifier; any other primary
            # (Round 2's hurdle classifier) needs its own exporter first
            print(f"VIGIL-E primary {ge['candidate']} has no exporter in the app's artifact "
                  f"format yet (status: {ge['status']}) — nothing exported")
            ge = {"passed": False, "skip_message": True}
        # VIGIL's verdict comes only from a sealed test (Pre-run amendment);
        # a pooled or CV-only pass never exports it
        if gv and gv["passed"] and gv.get("basis") == "sealed_test":
            print("VIGIL serving model passed → export",
                  export.export_vigil(all_events, gv, cv, sealed, d, pooled=mode["pooled"]))
        elif gv and gv["passed"]:
            print("VIGIL serving model passed on CV only — VIGIL is decided by the sealed run; "
                  "not exported")
        else:
            print("VIGIL serving model did not pass its gate — nothing exported")
        if ge and ge["passed"]:
            print("VIGIL-E serving model passed → export",
                  export.export_vigil_e(all_events, ge, cv, sealed, d, a.budget,
                                        pooled=mode["pooled"]))
        elif not (ge and ge.get("skip_message")):
            print("VIGIL-E serving model did not pass its gate — nothing exported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
