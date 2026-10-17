"""Pre-registered ship rules, fixed in code before any real-data run.

The PRE-REGISTERED PRIMARY candidates (candidates.PRIMARY) are each tested
as a single hypothesis - there is no selection among the pool, so no
winner's curse and no multiplicity correction is needed for them. The one
non-primary that is gated, ``serving_resid`` (the only VIGIL the app's
current artifact format loads; it decides --export), is judged on its max-T
multiplicity-adjusted interval instead. Every other row is exploratory.

CONFIRMATORY POPULATION - pre-run amendment, 2026-09-24 (docs/v2/vigil-lab.md,
"Pre-run amendment"). When a sealed official test exists, every check except
VIGIL-E's calibration slope is judged ON THE SEALED TEST, and the other CV
checks are reported as non-blocking supporting evidence (the calibration
slope always comes from CV and stays blocking: only cross_validate computes
it). Without a sealed test
(e.g. --pool-test-into-cv) the gate is judged on CV. Why: the organisers put
150 of the 216 eligible high-risk events in the test set (Uriot et al. 2022,
§4.2), so the training file alone (66 high-risk, 19 escalations) could
never reach MIN_HR_EVENTS / MIN_ESCALATIONS, and under the original CV-only
power rule every verdict was "underpowered" by construction. The amendment
rests only on those label counts: no trained candidate had been scored on
real data when it was made (the model-free baselines computed on the sealed
labels before then are disclosed in the doc). PRIMARY and every threshold
below are unchanged.

VIGIL (official L = MSE_HR / F2, lower is better; ΔL in % vs LRP)
  powered     at least MIN_HR_EVENTS high-risk eligible events in the judged
              population; else "underpowered" (no claim either way)
  gain        mean ΔL <= -MIN_GAIN_PCT %
  ci          bootstrap CI of ΔL entirely below 0 (95% in Round 1, CI_LABEL now)

VIGIL-E (early warning; ΔPR-AUC vs e_logit on eligible candidates)
  powered     at least MIN_ESCALATIONS escalations in the judged population
  ci          CI of ΔPR-AUC entirely above 0 (95% in Round 1, CI_LABEL now)
  recall      Δrecall@budget 95% CI does not lie entirely below 0
  calibration CV mean calibration slope in [CAL_LO, CAL_HI] (probabilistic)

Every verdict carries the minimum detectable effect of the judged population
(2.8 x bootstrap SD: 80% power, one-sided 2.5%; for the max-T adjusted
serving_resid verdict, (c* + 0.84) x SD), so "not passed" reads as "not
detectable with this much data" rather than "worse".
"""
from __future__ import annotations

from backend.app.models.lab.candidates import PRIMARY

MIN_GAIN_PCT = 2.0
MIN_HR_EVENTS = 100
MIN_ESCALATIONS = 50
CAL_LO, CAL_HI = 0.6, 1.6

# ROUND 2 (pre-registered 2026-10-08): each analysis is a SECOND look at data
# Round 1 already judged (the sealed test's second and last look; a second
# pooled-CV analysis), so the primaries' interval is 97.5% (one-sided 1.25%,
# half of Round 1's 2.5%) instead of 95%. Thresholds above are unchanged;
# the "not worse" recall check stays at 95% (a wider interval would be laxer).
CI_KEY, CI_LABEL, CI_Z = "ci975", "97.5%", 2.2414


def _fmt_ci(ci, spec):
    if not ci or ci[0] is None:
        return "no CI"
    return f"[{format(ci[0], spec)}, {format(ci[1], spec)}]"


def _fmt(v, spec):
    import math

    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "undefined"                      # e.g. F2 = 0: the ratio does not exist
    return format(v, spec)


def _pct_ci(ci):
    txt = _fmt_ci(ci, "+.2f")
    return txt if txt == "no CI" else f"{txt}%"


def _mde(summary: dict, pop: dict, adjusted: bool):
    """80%-power MDE of the test actually applied: (z + 0.84)·SD for the
    primaries' CI_LABEL interval, or (c* + 0.84)·SD for a max-T adjusted
    interval of half-width c*·SD."""
    sd = summary.get("sd_boot")
    c = ((pop or {}).get("multiplicity") or {}).get("maxT_c_regression")
    if adjusted and c is not None and sd:
        return (c + 0.8416) * sd
    return (CI_Z + 0.8416) * sd if sd else summary.get("mde")


def _ci_of(summary: dict, adjusted: bool):
    """(interval, label) the gate judges: max-T adjusted 95% for the gated
    non-primary, else the primaries' CI_LABEL interval."""
    if adjusted:
        return summary.get("ci95_adjusted") or [None, None], "max-T adjusted 95%"
    return summary.get(CI_KEY) or [None, None], CI_LABEL


def _result(name, basis, powered, checks, supporting, mde, mde_key):
    passed = all(ok for _, ok, _ in checks)
    status = "underpowered" if not powered else ("passed" if passed else "not_passed")
    return {"candidate": name, "basis": basis, "passed": passed, "status": status,
            # judged on the sealed test, a pass is the sealed confirmation
            "confirmed_on_sealed_test": bool(passed) if basis == "sealed_test" else None,
            mde_key: mde,
            "checks": [{"name": n, "ok": bool(ok), "detail": det} for n, ok, det in checks],
            "supporting": [{"name": n, "ok": bool(ok), "detail": det} for n, ok, det in supporting]}


def _vigil_check(name, cv, test, adjusted=False):
    d = cv["regression"][name]["delta_L_pct_vs_lrp"]
    ci, lvl = _ci_of(d, adjusted)
    cv_checks = [
        ("cv_mean_gain", d["mean"] is not None and d["mean"] <= -MIN_GAIN_PCT,
         f"CV mean ΔL {_fmt(d['mean'], '+.2f')}% (need <= -{MIN_GAIN_PCT}%)"),
        ("cv_ci_below_zero", ci[1] is not None and ci[1] < 0,
         f"CV {lvl} CI {_pct_ci(ci)} (need upper < 0)"),
    ]
    sealed = (test or {}).get("regression", {}).get(name)
    if sealed is None:
        powered = cv["n_hr_eligible"] >= MIN_HR_EVENTS
        checks = [("powered", powered, f"{cv['n_hr_eligible']} high-risk eligible events in CV "
                                       f"(need >= {MIN_HR_EVENTS})")] + cv_checks
        return _result(name, "cv", powered, checks, [], _mde(d, cv, adjusted),
                       "min_detectable_delta_pct")
    t = sealed["delta_L_pct_vs_lrp"]
    tci, lvl = _ci_of(t, adjusted)
    powered = test["n_hr"] >= MIN_HR_EVENTS
    checks = [
        ("powered", powered, f"{test['n_hr']} high-risk events in the sealed test "
                             f"(need >= {MIN_HR_EVENTS})"),
        ("sealed_gain", t["mean"] is not None and t["mean"] <= -MIN_GAIN_PCT,
         f"sealed-test ΔL {_fmt(t['mean'], '+.2f')}% (need <= -{MIN_GAIN_PCT}%)"),
        ("sealed_ci_below_zero", tci[1] is not None and tci[1] < 0,
         f"sealed-test {lvl} CI {_pct_ci(tci)} (need upper < 0)"),
    ]
    return _result(name, "sealed_test", powered, checks, cv_checks, _mde(t, test, adjusted),
                   "min_detectable_delta_pct")


def vigil(cv: dict, test: dict | None) -> dict:
    out = {slot: _vigil_check(PRIMARY[key], cv, test)
           for key, slot in (("vigil", "selected"), ("vigil_serving", "serving")) if key in PRIMARY}
    # the model the app's CURRENT artifact format can load; not pre-registered,
    # so judged on its multiplicity-adjusted interval
    out["serving_exportable"] = _vigil_check("serving_resid", cv, test, adjusted=True)
    return out


def _ew_checks(e, where, adjusted):
    da, dr = e["delta_pr_auc_vs_logit"], e["delta_recall_budget_vs_logit"]
    ci, lvl = _ci_of(da, adjusted)
    return [
        (f"{where}_pr_auc_ci_above_zero", ci[0] is not None and ci[0] > 0,
         f"{where} ΔPR-AUC {lvl} CI {_fmt_ci(ci, '+.4f')} (need lower > 0)"),
        (f"{where}_recall_budget_not_worse", dr["ci95"][1] is not None and dr["ci95"][1] >= 0,
         f"{where} Δrecall@budget 95% CI {_fmt_ci(dr['ci95'], '+.3f')} (need upper >= 0)"),
    ]


def _ew_check(name, cv, test, adjusted=False):
    e = cv["early_warning"][name]
    cal = []
    if "cal_slope" in e:
        s = e["cal_slope"]["mean"]
        cal = [("calibration", CAL_LO <= s <= CAL_HI,
                f"CV calibration slope {s:.2f} (need {CAL_LO}–{CAL_HI})")]
    cv_checks = _ew_checks(e, "CV", adjusted)
    sealed = (test or {}).get("early_warning", {}).get(name)
    if not sealed or "delta_pr_auc_vs_logit" not in sealed:
        powered = cv["n_escalations"] >= MIN_ESCALATIONS
        checks = ([("powered", powered, f"{cv['n_escalations']} escalations among eligible "
                                        f"candidates in CV (need >= {MIN_ESCALATIONS})")]
                  + cv_checks + cal)
        return _result(name, "cv", powered, checks, [], _mde(e["delta_pr_auc_vs_logit"], cv, False),
                       "min_detectable_delta_pr_auc")
    powered = test["n_escalations"] >= MIN_ESCALATIONS
    checks = ([("powered", powered, f"{test['n_escalations']} escalations among eligible "
                                    f"candidates in the sealed test (need >= {MIN_ESCALATIONS})")]
              + _ew_checks(sealed, "sealed-test", adjusted) + cal)
    return _result(name, "sealed_test", powered, checks, cv_checks,
                   _mde(sealed["delta_pr_auc_vs_logit"], test, False), "min_detectable_delta_pr_auc")


def vigil_e(cv: dict, test: dict | None) -> dict:
    return {slot: _ew_check(PRIMARY[key], cv, test)
            for key, slot in (("vigil_e", "selected"), ("vigil_e_serving", "serving")) if key in PRIMARY}
