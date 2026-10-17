"""Human-readable Markdown report of a lab run."""
from __future__ import annotations

from backend.app.models.lab import gate
from backend.app.models.lab.candidates import DERIVED, PRIMARY


def _f(v, spec=".3f"):
    import math

    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    return format(v, spec)


def _ci(ci, spec=".1f"):
    if not ci or ci[0] is None:
        return "—"
    return f"[{format(ci[0], spec)}, {format(ci[1], spec)}]"


def _pct(k, n):
    return f"{100 * k / n:.2f}%" if n else "—"


def _verdict(L, label, s, mde_key, unit, sealed_exists):
    basis = s.get("basis")
    if s["status"] == "underpowered":
        status = "is underpowered — too little data for a claim either way"
    elif basis == "sealed_test":
        status = ("passes the ship gate on the sealed test" if s["passed"]
                  else "does not pass the ship gate on the sealed test")
    elif not sealed_exists:
        status = ("passes the CV gate (no sealed test, so no veto was possible)" if s["passed"]
                  else "does not pass the CV gate (no sealed test)")
    else:
        status = "passes the ship gate" if s["passed"] else "does not pass the ship gate"
    extra = ""
    if s["passed"] and basis != "sealed_test" and sealed_exists:
        extra = (" Confirmed on the sealed test." if s.get("confirmed_on_sealed_test")
                 else " Not confirmed on the sealed test (its interval includes 0).")
    L.append(f"**{label}: `{s['candidate']}` {status}.**{extra} "
             f"Minimum detectable effect with this data ≈ {_f(s.get(mde_key), '.3f')}{unit}.")
    for c in s["checks"]:
        L.append(f"- {'✓' if c['ok'] else '✗'} {c['detail']}")
    for c in s.get("supporting", []):
        L.append(f"- ({'✓' if c['ok'] else '✗'} supporting, not blocking) {c['detail']}")
    L.append("")


def _banner(run, test):
    mode = run.get("mode") or {}
    code = mode.get("code") or {}
    out = []
    if not test:
        if mode.get("pooled"):
            why = (f"the {mode.get('n_pooled_test_events', 0)} official test events were pooled "
                   "into cross-validation")
        elif (mode.get("cv_go") or {}).get("go") is False:
            cg = mode["cv_go"]
            why = (f"the pre-registered go/no-go said no (`{cg['candidate']}` CV ΔL at the official "
                   f"test's class mix {_f(cg['cv_bench_mix_delta_L_pct'], '+.2f')}%, needed < 0), so "
                   "the sealed test's last look was not spent")
        elif not mode.get("targets_supplied"):
            why = "the sealed test's final CDMs were not supplied"
        else:
            why = "no sealed test was scored"
        out.append(f"> **NO SEALED TEST** — {why}. Verdicts are CV-only, not comparable to the "
                   "2019 leaderboard, and the data-fidelity check did not run.\n")
        if mode.get("pooled"):
            out.append("> Pre-declared role (docs/v2/vigil-lab.md, \"Pre-run amendment\"): this run "
                       "decides **VIGIL-E only**. Its VIGIL rows and verdicts are exploratory; "
                       "VIGIL's verdict comes from the sealed run.\n")
    if code.get("commit"):
        out.append(f"Code: `{code['commit'][:12]}`"
                   f"{' (uncommitted changes in backend/app/models)' if code.get('dirty') else ''}.\n")
    return out


def _sealed_dl(t, primary):
    """Sealed ΔL cell: primaries are single pre-registered tests; exploratory
    rows get the max-T adjusted interval (the best of many sealed rows would
    otherwise repeat the winner's curse on the test set)."""
    td = t["delta_L_pct_vs_lrp"]
    if primary and td.get(gate.CI_KEY):
        return f"{_f(td['mean'], '+.1f')}% {gate.CI_LABEL} {_ci(td[gate.CI_KEY])}"
    if primary or td.get("ci95_adjusted") in (None, [None, None]):
        return f"{_f(td['mean'], '+.1f')}% {_ci(td['ci95'])}"
    return f"{_f(td['mean'], '+.1f')}% adj {_ci(td['ci95_adjusted'])}"


def _sealed_dap(t, primary):
    if primary and t.get(gate.CI_KEY):
        return f"{_f(t['mean'], '+.4f')} {gate.CI_LABEL} {_ci(t[gate.CI_KEY], '+.4f')}"
    if primary or t.get("ci95_adjusted") in (None, [None, None]):
        return f"{_f(t['mean'], '+.4f')} {_ci(t['ci95'], '+.4f')}"
    return f"{_f(t['mean'], '+.4f')} adj {_ci(t['ci95_adjusted'], '+.4f')}"


def markdown(run: dict) -> str:
    cv, test, g, ge = run["cv"], run.get("test"), run["gate_vigil"], run["gate_vigil_e"]
    mode = run.get("mode") or {}
    L = ["# VIGIL Lab report\n"]
    L += _banner(run, test)
    source = (f"CV pool (training file + {mode.get('n_pooled_test_events', 0)} official test events)"
              if mode.get("pooled") else "Training file")
    L.append(f"{source}: {cv['n_events']} events, {cv['n_trainable']} usable for training, "
             f"{cv['n_eligible']} meeting the official test filters ({cv['n_hr_eligible']} of them "
             f"high-risk, {_pct(cv['n_hr_eligible'], cv['n_eligible'])}). Early warning is scored on "
             f"{cv['n_candidates']} eligible escalation candidates with {cv['n_escalations']} "
             f"escalations ({_pct(cv['n_escalations'], cv['n_candidates'])}; "
             f"{cv['n_escalations_all']} escalations in the whole file). Protocol: "
             f"{cv['repeats']}×{cv['folds']} stratified CV, {cv['bootstrap']} shared-index paired "
             f"bootstrap replicates, {cv['augmented_rows']} cutoff-augmentation rows.\n")
    if test:
        L.append(f"Sealed official test: {test['n_eligible']} events, {test['n_hr']} high-risk "
                 f"({_pct(test['n_hr'], test['n_eligible'])}); {test['n_candidates']} escalation "
                 f"candidates with {test['n_escalations']} escalations "
                 f"({_pct(test['n_escalations'], test['n_candidates'])}). Same eligibility filters as "
                 "CV, different class mix: the organisers enriched the test set with high-risk events "
                 "(Uriot et al. 2022, §4.2), so CV and sealed magnitudes are not replicates of each "
                 "other.\n")
    L.append(f"Pre-registered primary candidates: {', '.join(f'`{v}`' for v in PRIMARY.values())}. "
             "Each is tested by the ship gate as a single hypothesis. `serving_resid`, the only "
             "VIGIL the app's current artifact format can load, is also gated (it decides "
             "`--export`), on its max-T adjusted interval. Every other row is exploratory, and its "
             "*adjusted* interval accounts for comparing many candidates at once.\n")
    if gate.CI_KEY == "ci975":
        L.append("**Round 2** (pre-registered 2026-10-08, docs/v2/vigil-lab.md): a second look at "
                 "data Round 1 already judged, so the primaries must clear a **97.5%** interval "
                 "(half of Round 1's error rate). Round 1's primaries are not re-tested.\n")
    if test:
        fid = test["lrp_fidelity"]
        ok = "reproduces" if fid["reproduces_published"] else "DOES NOT reproduce"
        L.append(f"**Data fidelity:** on the sealed test the latest-risk baseline {ok} the published "
                 f"score (lab L {fid['L']['lab']:.3f} vs {fid['L']['published']}, MSE_HR "
                 f"{fid['mse_hr']['lab']:.3f} vs {fid['mse_hr']['published']}, F2 {fid['f2']['lab']:.3f} "
                 f"vs {fid['f2']['published']}). If it does not, fix the data/target handling before "
                 "trusting anything below (on synthetic test data this check is expected to fail).\n")
    L.append("## VIGIL — official score L = MSE_HR / F2 (lower is better)\n")
    hdr = ("| Candidate | Track | L (CV mean ± sd) | ΔL vs LRP | 95% CI (%) | adjusted CI (%) | MDE (pp) "
           "| MSE_HR | F2 | CV ΔL at test mix |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    if test:
        hdr += " Sealed L | Sealed ΔL [95% CI] |"
        sep += "---|---|"
    L += [hdr, sep]
    undefined = []
    for k in sorted(cv["regression"], key=lambda k: cv["regression"][k]["L"]["mean"]):
        r = cv["regression"][k]
        d = r["delta_L_pct_vs_lrp"]
        name = f"**{k}**" if r.get("primary") else k
        if r["f2"]["mean"] == 0:
            name += " (never predicts high risk)"
        if (d.get("undefined_frac") or 0) > 0:
            undefined.append(f"`{k}` {100 * d['undefined_frac']:.0f}%")
        row = (f"| {name} | {r['track']} | {_f(r['L']['mean'])} ± {_f(r['L']['sd'])} | "
               f"{_f(d['mean'], '+.1f')}% | {_ci(d['ci95'])} | {_ci(d.get('ci95_adjusted'))} | "
               f"{_f(d.get('mde'), '.1f')} | {_f(r['mse_hr']['mean'])} | {_f(r['f2']['mean'])} | "
               f"{_f((r.get('bench_mix') or {}).get('delta_L_pct', {}).get('mean'), '+.1f')}% |")
        if test:
            t = test["regression"].get(k)
            row += (f" {_f(t['L']['mean'])} | {_sealed_dl(t, r.get('primary'))} |" if t
                    else " — | — |")
        L.append(row)
    L.append("")
    if test:
        L.append(f"Sealed ΔL: primaries show the {gate.CI_LABEL} CI their gate judges; exploratory "
                 "rows show the max-T multiplicity-adjusted 95% CI (\"adj\").\n")
    L.append("CV ΔL at test mix: the same out-of-fold predictions scored with label-shift weights "
             "that give the official test's published class mix (150 of 2,167 high-risk; Uriot et "
             "al. 2022, §4.2) — what CV says about a sealed look. Point estimate, mean over "
             "repeats.\n")
    if undefined:
        L.append("CV bootstrap replicates where ΔL is undefined (F2 = 0) are dropped, which flatters "
                 f"those rows: {', '.join(undefined)} of replicates.\n")
    if g.get("selected"):
        _verdict(L, "VIGIL (benchmark)", g["selected"], "min_detectable_delta_pct",
                 " percentage points", bool(test))
    if g.get("serving"):
        lab = "VIGIL (serving-track features; needs a RiskModel hurdle extension to deploy)"
        if g["serving"]["candidate"] in DERIVED:
            lab = ("VIGIL leaderboard entry (serving features; decision set for the official "
                   "test's published class mix — a scoring adjustment for the benchmark, not a "
                   "live-app improvement)")
        _verdict(L, lab, g["serving"], "min_detectable_delta_pct", " percentage points", bool(test))
    if g.get("serving_exportable"):
        _verdict(L, "The only app-loadable VIGIL result (`serving_resid`, current artifact format)",
                 g["serving_exportable"], "min_detectable_delta_pct", " percentage points", bool(test))
    L.append("## VIGIL-E — early warning (eligible events still below −6 at the cutoff; one "
             "decision per event, at the last CDM ≥ 2 days before TCA)\n")
    hdr = ("| Candidate | PR-AUC | ROC-AUC | pAUC@FPR≤0.1 | Recall@budget | ΔPR-AUC vs logit | 95% CI "
           "| adjusted CI | MDE |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    if test:
        hdr += " Sealed ΔPR-AUC [95% CI] |"
        sep += "---|"
    L += [hdr, sep]
    for k in sorted(cv["early_warning"], key=lambda k: -cv["early_warning"][k]["pr_auc"]["mean"]):
        e = cv["early_warning"][k]
        d = e.get("delta_pr_auc_vs_logit", {})
        name = f"**{k}**" if e.get("primary") else k
        row = (f"| {name} | {_f(e['pr_auc']['mean'], '.4f')} ± {_f(e['pr_auc']['sd'], '.4f')} | "
               f"{_f(e['roc_auc']['mean'])} | {_f(e['pauc_fpr10']['mean'])} | "
               f"{_f(e['recall_at_budget']['mean'])} | {_f(d.get('mean'), '+.4f')} | "
               f"{_ci(d.get('ci95'), '+.4f')} | {_ci(d.get('ci95_adjusted'), '+.4f')} | "
               f"{_f(d.get('mde'), '.4f')} |")
        if test:
            t = test["early_warning"].get(k, {}).get("delta_pr_auc_vs_logit")
            row += f" {_sealed_dap(t, e.get('primary'))} |" if t else " — |"
        L.append(row)
    L.append("")
    if ge.get("selected"):
        _verdict(L, "VIGIL-E (benchmark)", ge["selected"], "min_detectable_delta_pr_auc", "",
                 bool(test))
    if ge.get("serving"):
        _verdict(L, "VIGIL-E (serving track: features the app computes)", ge["serving"],
                 "min_detectable_delta_pr_auc", "", bool(test))
    if run.get("ceiling"):
        L.append("## Ceiling diagnostic (not admissible)\n")
        L.append("The primary model, allowed to also see CDMs between 2 and 1 day before TCA:")
        for k, v in run["ceiling"].items():
            L.append(f"- {k}: model L {_f(v['model']['L'])} vs latest-risk {_f(v['lrp']['L'])}")
        L.append("The gap between the two rows is the part of the outcome decided by late tracking "
                 "updates that no admissible model can see."
                 + (" Pooled test events carry no late CDMs, so this run's gap is diluted."
                    if mode.get("pooled") else "") + "\n")
    L.append("## How to read this\n")
    L.append("- Only the **sealed test** columns are comparable to the 2019 leaderboard (published: "
             "latest-risk baseline L ≈ 0.694; the top entry, ≈ 0.556, was a hand-tuned rule cascade "
             "partly fitted by trial against the test leaderboard — Uriot et al. 2022, §5.3.1).")
    L.append("- `sesc` here is the organisers' post-competition LightGBM control (Uriot et al. 2022, "
             "§6.1, without its quantile-transformed target), not team sesc's winning entry.")
    L.append("- CV rows use test-like events (same filters, different class mix); the sealed test "
             "can judge a pre-registered primary but never select one.")
    L.append("- `serving` rows use only what the live app can compute, for predictions made ≥ 2 days "
             "before TCA; those are the only numbers the app may claim, and only for that decision "
             "point. `hurdle_lrp_serving` cannot be loaded by the current RiskModel.")
    L.append("- MDE = minimum detectable effect (80% power). A difference smaller than the MDE can "
             "be real but is invisible to this much data.")
    L.append("- CV intervals hold the fitted out-of-fold predictions fixed, so they leave out refit "
             "variance (Bates et al. 2023): read them as lower bounds on the uncertainty.")
    return "\n".join(L) + "\n"
