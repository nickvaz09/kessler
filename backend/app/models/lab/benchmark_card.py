"""The benchmark card: VIGIL's results on the official ESA benchmark, as the
Model page shows them (GET /model/benchmark).

Every number is read from the committed VIGIL Lab reports (each file's
SHA-256 is recorded), except the cited constants from the organisers'
paper - no figure is typed by hand. Regenerate after any new lab run:

    .venv/bin/python -m backend.app.models.lab.benchmark_card
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from backend.app.models.lab import compete

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("benchmark.json")
SEALED_R2 = "reports/vigil-lab-r2-sealed/lab_report.json"
POOLED_R1 = "reports/vigil-lab-pooled/lab_report.json"
POOLED_R2 = "reports/vigil-lab-r2-pooled/lab_report.json"
PUBLIC = "reports/paper/public.json"          # round 4 P8: VIGIL's classifier as a false-alarm filter
EW_POSTHOC = "reports/paper/ew_posthoc.json"  # post hoc: the latest risk itself as the baseline
# Uriot et al. 2022, §6.1: the organisers trained 5,000 gradient-boosting
# models; the single one that beat the baseline on both train and test
# scored 0.684 on the official test, "would have ranked 11th". ESA does not
# say it was the best of the 5,000, only that it was the one to do both.
ESA_MODEL = {"L": 0.684, "rank": 11, "models": 5000,
                    "source": "Uriot et al. 2022, Astrodynamics 6:121–140, §6.1"}


def _load(rel: str) -> tuple[dict, dict]:
    p = ROOT / rel
    raw = p.read_bytes()
    return json.loads(raw), {"path": rel, "sha256": hashlib.sha256(raw).hexdigest()}


def build() -> dict:
    sealed, f1 = _load(SEALED_R2)
    pooled1, f2 = _load(POOLED_R1)
    pooled2, f3 = _load(POOLED_R2)
    public, f4 = _load(PUBLIC)
    ewp, f5 = _load(EW_POSTHOC)
    lat = ewp["early_warning"]["latest_risk_ranking"]
    a0 = ewp["early_warning"]["contrasts_vs_latest_risk"]["A0"]
    a0_roc = ewp["early_warning"]["roc_contrasts_vs_latest_risk"]["A0"]
    fa = public["p8_false_alarms"]
    fa_lat = ewp["false_alarms"]["latest_risk_ranking"]["false_alarms_removed_at_95pct"]
    t = sealed["test"]
    name = sealed["gate_vigil"]["serving"]["candidate"]
    r = t["regression"][name]
    d = r["delta_L_pct_vs_lrp"]
    fid = t["lrp_fidelity"]
    ew1 = pooled1["cv"]["early_warning"]
    ew2 = pooled2["cv"]["early_warning"]
    ge1 = pooled1["gate_vigil_e"]["selected"]
    ge2 = pooled2["gate_vigil_e"]["serving"]
    return {
        "benchmark": {
            "name": "ESA Kelvins Collision Avoidance Challenge (2019)",
            "metric": "L = MSE_HR / F2 (lower is better)",
            "test_events": t["n_eligible"], "test_high_risk": t["n_hr"],
            "teams": compete.N_TEAMS,
        },
        "fidelity": {k: {"lab": round(fid[k]["lab"], 4), "published": fid[k]["published"]}
                     for k in ("L", "mse_hr", "f2")}
                    | {"reproduces_published": fid["reproduces_published"]},
        "vigil": {
            "model": name, "L": round(r["L"]["mean"], 4),
            "delta_pct_vs_baseline": round(d["mean"], 2),
            "ci975_pct": [round(v, 2) for v in d["ci975"]],
            "gate_status": sealed["gate_vigil"]["serving"]["status"],
            "placement": compete.placement(r["L"]["mean"]),
            "locked_before_testing": True,
            "decision": "set for the official test's published class mix (150 of 2,167 high-risk)",
            "code": (sealed.get("mode") or {}).get("code", {}).get("commit"),
        },
        "baseline": {"name": "Latest-risk baseline", "L": round(t["regression"]["lrp"]["L"]["mean"], 4)},
        "esa_model": ESA_MODEL,
        "leaderboard": [{"rank": i + 1, "team": n, "L": s}
                        for i, (n, s) in enumerate(compete.LEADERBOARD)],
        "early_warning": {
            "research": {
                "model": ge1["candidate"], "pr_auc": round(ew1[ge1["candidate"]]["pr_auc"]["mean"], 4),
                "baseline_pr_auc": round(ew1["e_logit"]["pr_auc"]["mean"], 4),
                "ci95": [round(v, 4) for v in
                         ew1[ge1["candidate"]]["delta_pr_auc_vs_logit"]["ci95"]],
                "escalations": pooled1["cv"]["n_escalations"], "gate_status": ge1["status"],
                "needs": "the full CDM data (including covariance) that Kessler's live feed lacks",
                # post hoc: the registered baseline is weakened by pooling its per-fold fits;
                # ranking by the latest risk itself needs no fit
                "latest_risk_pr_auc": round(lat["pr_auc"], 4),
                "delta_vs_latest_risk": round(a0["mean"], 4),
                "ci95_vs_latest_risk": [round(v, 4) for v in a0["ci95"]],
                "roc_auc": round(ewp["early_warning"]["pooled_roc_auc"]["A0"], 4),
                "latest_risk_roc_auc": round(lat["roc_auc"], 4),
                "roc_ci95_vs_latest_risk": [round(v, 4) for v in a0_roc["ci95"]],
                "reading": "not shown: against the latest risk itself it ranks quiet events better "
                           "overall, but not at the top of the list",
            },
            "app_features": {
                "model": ge2["candidate"], "pr_auc": round(ew2[ge2["candidate"]]["pr_auc"]["mean"], 4),
                "baseline_pr_auc": round(ew2["e_logit"]["pr_auc"]["mean"], 4),
                "ci975": [round(v, 4) for v in
                          ew2[ge2["candidate"]]["delta_pr_auc_vs_logit"]["ci975"]],
                "gate_status": ge2["status"],
                "latest_risk_pr_auc": round(lat["pr_auc"], 4),
            },
        },
        "false_alarms": {
            "model": "VIGIL's classifier, all CDM fields (research build)",
            "calls": fa["n_calls"], "stay_high_risk": fa["n_stay_high_risk"], "keep_true_calls_pct": 95,
            "removed_pct": round(100 * fa["false_alarms_removed_at_95pct"]["A0"]["mean"], 1),
            "ci95_pct": [round(100 * v, 1) for v in fa["false_alarms_removed_at_95pct"]["A0"]["ci95"]],
            "public_counterparts_removed_pct": round(100 * fa["false_alarms_removed_at_95pct"]["PUB"]["mean"], 1),
            "latest_risk_removed_pct": round(100 * fa_lat["mean"], 1),
            "latest_risk_ci95_pct": [round(100 * v, 1) for v in fa_lat["ci95"]],
            "evaluation": f"pooled cross-validation over {public['n_eligible']:,} ESA events "
                          "(the latest-risk filter is post hoc)",
        },
        "live_model_note": "The console scores live events with VIGIL-0.4; these are research "
                           "builds evaluated in the VIGIL Lab.",
        "sources": [f1, f2, f3, f4, f5],
    }


def main() -> int:
    card = build()
    OUT.write_text(json.dumps(card, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: VIGIL L {card['vigil']['L']} "
          f"(rank {card['vigil']['placement']}), ESA's model {ESA_MODEL['L']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
