"""Build every figure, table and in-text number of the paper from the
committed lab reports (nothing is typed in by hand).

    pip install matplotlib                  # figures only; the app doesn't need it
    python scripts/paper_assets.py          # from the repo root
    python scripts/paper_assets.py --out /tmp/paper

Reads reports/vigil-lab-*/lab_report.json, reports/compete/final.json and
reports/paper/*.json, and writes <out>/generated/{numbers,tab_*}.tex (the LaTeX
macros behind every number in the text, and the tables) and <out>/figures/*.pdf.
Every claim that rests on a number is checked here first (the asserts), so a
report that no longer supports the text stops the build. The paper's Figs. 1
and 5 also read ESA's Kelvins files (data/kelvins/, not redistributed); without
them those two figures are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_cli = argparse.ArgumentParser(description="Build the paper's figures, tables and numbers from reports/")
_cli.add_argument("--out", type=Path, default=ROOT / "paper",
                  help="folder for generated/ and figures/ (default: paper/)")
OUT_DIR = _cli.parse_args().out
OUT = OUT_DIR / "generated"
FIG = OUT_DIR / "figures"


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


R1 = load("reports/vigil-lab-sealed/lab_report.json")
R2 = load("reports/vigil-lab-r2-sealed/lab_report.json")
P1 = load("reports/vigil-lab-pooled/lab_report.json")
P2 = load("reports/vigil-lab-r2-pooled/lab_report.json")
COMP = load("reports/compete/final.json")
NB = load("reports/paper/nbcorr.json")
RES = load("reports/paper/resolution.json")
ATT = load("reports/paper/attack.json")
HOR = load("reports/paper/horizon.json")
ABL = load("reports/paper/ablation.json")
COH = load("reports/paper/coherence.json")
CPH = load("reports/paper/coherence_posthoc.json")    # post hoc, not pre-registered
PUB = load("reports/paper/public.json")
EWP = load("reports/paper/ew_posthoc.json")    # post hoc, not pre-registered
REVC = load("reports/paper/review_coherence.json")    # post hoc, after the review
REVV = load("reports/paper/review_cv.json")    # post hoc, after the review
REVM = load("reports/paper/review_mix.json")    # post hoc, after the review: the operational class mix

from backend.app.models.lab import compete  # noqa: E402  (leaderboard constants)


# --- number formatting --------------------------------------------------------

MINUS = r"\ensuremath{-}"     # a true minus sign in text and in math


def f(x, d=1):
    """Fixed decimals with a real minus sign, for LaTeX text mode."""
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) != 0:
        return MINUS + s[1:]
    return s.lstrip("-")


def sg(x, d=1):
    """Signed: +1.2 / -1.2."""
    return ("+" if round(x, d) > 0 else "") + f(x, d)


def ci(c, d=1, signed=True):
    g = sg if signed else f
    return f"[{g(c[0], d)}, {g(c[1], d)}]"


def n(x):
    return f"{int(x):,}".replace(",", "{,}")


def sci(x, d=1):
    """x as $a\\times10^{b}$ (math mode), for very small numbers."""
    e = int(math.floor(math.log10(abs(x))))
    a = x / 10 ** e
    if round(a, d) >= 10:
        a, e = a / 10, e + 1
    return f"${a:.{d}f}\\times10^{{{e}}}$"


MACROS: list[tuple[str, str]] = []


def m(name: str, value: str):
    if not name.isalpha():
        raise ValueError(f"macro names must be letters only: {name}")
    if any(k == name for k, _ in MACROS):
        raise ValueError(f"macro defined twice: {name}")
    MACROS.append((name, value))


# --- data counts ---------------------------------------------------------------

cv1, t1 = R1["cv"], R1["test"]
m("nTrainEvents", n(cv1["n_events"]))
m("nTrainTrainable", n(cv1["n_trainable"]))
m("nTrainEligible", n(cv1["n_eligible"]))
m("nTrainHR", n(cv1["n_hr_eligible"]))
m("nTrainCand", n(cv1["n_candidates"]))
m("nTrainEsc", n(cv1["n_escalations"]))
m("nTestEvents", n(t1["n_events"]))
m("nTestHR", n(t1["n_hr"]))
m("nTestCand", n(t1["n_candidates"]))
m("nTestEsc", n(t1["n_escalations"]))
pcv = P1["cv"]
m("nPoolEvents", n(pcv["n_events"]))
m("nPoolEligible", n(pcv["n_eligible"]))
m("nPoolHR", n(pcv["n_hr_eligible"]))
m("nPoolCand", n(pcv["n_candidates"]))
m("nPoolEsc", n(pcv["n_escalations"]))
m("pctTrainHR", f(100 * cv1["n_hr_eligible"] / cv1["n_eligible"], 1))
m("pctTestHR", f(100 * t1["n_hr"] / t1["n_events"], 1))

# --- fidelity and sealed results -------------------------------------------------

fid = t1["lrp_fidelity"]
m("lrpL", f(fid["L"]["lab"], 3))
m("lrpMSE", f(fid["mse_hr"]["lab"], 3))
m("lrpFtwo", f(fid["f2"]["lab"], 3))


def sealed(rep, name):
    return rep["test"]["regression"][name]


for key, rep, name in (("One", R1, "hurdle_lrp"), ("Two", R1, "hurdle_lrp_serving"),
                       ("Three", R2, "hurdle_lrp_serving_bench")):
    row = sealed(rep, name)
    d = row["delta_L_pct_vs_lrp"]
    m(f"prim{key}L", f(row["L"]["mean"], 3))
    m(f"prim{key}dL", sg(d["mean"], 1))
    m(f"prim{key}CI", ci(d["ci95"]))
    if "ci975" in d and rep is R2:
        m(f"prim{key}CIb", ci(d["ci975"]))      # round 2's pre-registered level
    m(f"prim{key}MDE", f(d["mde"], 1))

ex = sealed(R2, "hurdle_eserving")["delta_L_pct_vs_lrp"]
m("expEsdL", sg(ex["mean"], 1))
m("expEsCI", ci(ex["ci95"]))

cvb = R2["cv"]["regression"]
m("cvBenchMixdL", sg(cvb["hurdle_lrp_serving_bench"]["bench_mix"]["delta_L_pct"]["mean"], 1))
m("cvNaturalMixdL", sg(cvb["hurdle_lrp_serving_bench"]["delta_L_pct_vs_lrp"]["mean"], 1))
m("cvOneHurdledL", sg(cvb["hurdle_lrp"]["delta_L_pct_vs_lrp"]["mean"], 1))
m("cvNaturalMixCI", ci(cvb["hurdle_lrp_serving_bench"]["delta_L_pct_vs_lrp"]["ci95"]))
m("compPublicL", f(COMP["public_L"], 3))
m("compPrivateL", f(COMP["private"]["L"], 3))
m("compPrivateLRP", f(COMP["private"]["lrp_L"], 3))
m("compPrivatedL", sg(COMP["private"]["delta_L_pct_vs_lrp"], 1))
m("compFullL", f(COMP["full"]["L"], 3))
m("compTrainOOF", f(COMP["train_oof_test_mix_L"], 3))
m("compSubs", str(COMP["public_submissions"]))
m("leaderSesc", f(compete.LEADERBOARD[0][1], 3))
m("nLeaderBeatLRP", str(len(compete.LEADERBOARD)))     # matched or beat it: the organizers' count
m("nTeams", str(compete.N_TEAMS))
# the last of those teams ties LRP: equal to five decimals, a hair above it at the sixth
lrp_exact = fid["L"]["lab"]
tie_ = [s_ for _, s_ in compete.LEADERBOARD if s_ >= lrp_exact]
assert len(tie_) == 1 and round(tie_[0], 5) == round(lrp_exact, 5)
m("nLeaderBeatOnly", str(len(compete.LEADERBOARD) - 1))
m("leaderTie", f(tie_[0], 5))

# --- P5 --------------------------------------------------------------------------

m("nbT", f(NB["t_naive"], 2))
m("nbTc", f(NB["t_corrected"], 2))
m("nbP", f(NB["p_corrected"], 2))
m("nbFactor", f(NB["factor"], 3))
m("nbGain", sg(NB["f2_gain_pct"], 2))
m("nbCI", ci(NB["f2_gain_pct_ci95_corrected"], 1))
m("nbJ", str(NB["J"]))

# --- P2 ----------------------------------------------------------------------------

a = RES["a_lrp_bootstrap"]["plain"]["L"]
m("bootLo", f(a["p025"], 2))
m("bootHi", f(a["p975"], 2))
m("bootSD", f(a["sd"], 3))
sa = RES["a_lrp_bootstrap"]["stratified"]["L"]
m("bootStratLo", f(sa["p025"], 2))
m("bootStratHi", f(sa["p975"], 2))
c = RES["c_visible_subsets"]
m("visAbs", f(c["lrp_visible_vs_full_abs_pct"]["mean"], 1))
m("visLo", sg(c["lrp_visible_vs_full_pct"]["p025"], 1))
m("visHi", sg(c["lrp_visible_vs_full_pct"]["p975"], 1))
m("holdLRPLo", f(c["lrp_holdout_L"]["p025"], 2))
m("holdLRPHi", f(c["lrp_holdout_L"]["p975"], 2))
m("holdNHR", str(c["holdout_n_hr"]))
m("sescFulldL", sg(c["sesc_delta_L_pct_full"], 1))
m("sescFullL", f(RES["full_test"]["sesc_cascade"]["L"], 3))
m("sescVisMed", sg(c["sesc_delta_L_pct_visible"]["median"], 1))
m("sescHoldMed", sg(c["sesc_delta_L_pct_holdout"]["median"], 1))
m("sescHoldLo", sg(c["sesc_delta_L_pct_holdout"]["p025"], 0))
m("sescHoldHi", sg(c["sesc_delta_L_pct_holdout"]["p975"], 0))
d = RES["d_sesc_out_of_sample"]
m("oosN", n(d["n"]))
m("oosHR", n(d["n_hr"]))
m("oosNatdL", sg(d["natural"]["delta_L_pct"], 1))
m("oosNatCI", ci(d["natural"]["delta_L_pct_ci95"]))
m("oosMixdL", sg(d["bench_mix"]["delta_L_pct"], 1))
m("oosMixCI", ci(d["bench_mix"]["delta_L_pct_ci95"]))
m("oosNatLRP", f(d["natural"]["lrp"]["L"], 3))
m("oosNatSesc", f(d["natural"]["sesc_cascade"]["L"], 3))
m("oosNatPrecLRP", f(d["natural"]["lrp"]["precision"], 2))
m("oosNatPrecSesc", f(d["natural"]["sesc_cascade"]["precision"], 2))
mde = {(r["report"], r["candidate"]): r for r in RES["b_sealed_mde"]}
r_one = mde[("reports/vigil-lab-sealed/lab_report.json", "hurdle_lrp_serving")]
r_two = mde[("reports/vigil-lab-r2-sealed/lab_report.json", "hurdle_lrp_serving_bench")]
m("mdeServ", f(r_one["mde_pct"], 1))
m("mdeServTwo", n(r_one["n_hr_for_2pct"]))
m("mdeServFive", n(r_one["n_hr_for_5pct"]))
m("mdeBench", f(r_two["mde_pct"], 1))
m("mdeBenchTwo", n(r_two["n_hr_for_2pct"]))
m("mdeBenchFive", n(r_two["n_hr_for_5pct"]))
r_b = mde[("reports/vigil-lab-sealed/lab_report.json", "hurdle_lrp")]
m("mdeHurdle", f(r_b["mde_pct"], 1))
m("mdeHurdleTwo", n(r_b["n_hr_for_2pct"]))
m("mdeServTwoR", n(round(r_one["n_hr_for_2pct"], -2)))       # to the nearest hundred, for "roughly"
m("mdeHurdleTwoR", n(round(r_b["n_hr_for_2pct"], -2)))
m("mdeTimesLo", f(r_one["n_hr_for_2pct"] / r_one["n_hr"], 0))
m("mdeTimesHi", f(r_b["n_hr_for_2pct"] / r_b["n_hr"], 0))

# --- P3 ------------------------------------------------------------------------------

k0 = ATT["by_k"][str(ATT["primary_k"])]
m("atkBudget", str(ATT["budget"]))
m("atkRuns", str(ATT["runs"]))
m("atkK", str(ATT["primary_k"]))
m("atkVisMed", sg(k0["visible_dl_pct"]["median"], 1))
m("atkFullMed", sg(k0["full_dl_pct"]["median"], 1))
m("atkFullLo", sg(k0["full_dl_pct"]["p05"], 1))
m("atkFullHi", sg(k0["full_dl_pct"]["p95"], 1))
m("atkHoldMed", sg(k0["holdout_dl_pct"]["median"], 1))
m("atkHoldMean", sg(k0["holdout_dl_pct"]["mean"], 1))
m("atkFullL", f(k0["full_L"]["median"], 3))
m("atkFullLbest", f(k0["full_L"]["p05"], 3))
m("atkRank", str(k0["median_full_L_rank"]))
m("atkRankBest", str(compete.placement(k0["full_L"]["p05"])))
m("atkShare", f(100 * k0["share_beating_lrp_full"], 0))
m("atkAccepted", f(k0["accepted"]["median"], 0))
for k in ("5", "50"):
    kk = ATT["by_k"][k]
    word = {"5": "Five", "50": "Fifty"}[k]
    m(f"atkFullMed{word}", sg(kk["full_dl_pct"]["median"], 1))
    m(f"atkHoldMed{word}", sg(kk["holdout_dl_pct"]["median"], 1))

# --- P1 -------------------------------------------------------------------------------

rows = HOR["rows"]
cut = [str(c) for c in HOR["cutoffs"]]
r2, r1 = rows[cut[0]], rows[cut[-1]]
m("horN", n(HOR["n_scored"]))
m("horHR", n(HOR["n_scored_high_risk"]))
m("horLRPtwo", f(r2["lrp"]["L"], 3))
m("horLRPone", f(r1["lrp"]["L"], 3))
m("horModtwo", f(r2["model"]["L"], 3))
m("horModone", f(r1["model"]["L"], 3))
m("horRatio", f(100 * (1 - r1["lrp_L_ratio_to_2d"]), 0))
m("horRatioLo", f(100 * (1 - r1["lrp_L_ratio_to_2d_ci95"][1]), 0))
m("horRatioHi", f(100 * (1 - r1["lrp_L_ratio_to_2d_ci95"][0]), 0))
m("horRatioMix", f(100 * (1 - r1["lrp_L_ratio_to_2d_bench_mix"]), 0))
m("hordLtwo", sg(r2["delta_L_pct"], 1))
m("hordLtwoCI", ci(r2["delta_L_pct_ci95"]))
m("hordLone", sg(r1["delta_L_pct"], 1))
m("hordLoneCI", ci(r1["delta_L_pct_ci95"]))
m("horModGain", f(100 * (1 - r2["model"]["L"] / r2["lrp"]["L"]), 1))
m("horRepeats", str(HOR["repeats"]))
ceil = R1["ceiling"]
m("ceilAdm", f(ceil["admissible"]["model"]["L"], 3))
m("ceilLate", f(ceil["with_late_cdms"]["model"]["L"], 3))
m("horRepOne", f(r2["model_L_per_repeat"][0], 3))
m("horMixTwo", sg(r2["delta_L_pct_bench_mix"], 1))
m("horGainNat", f(-r2["delta_L_pct"], 0))
m("horGainMix", f(-r2["delta_L_pct_bench_mix"], 0))
m("horMixTwoCI", ci(r2["delta_L_pct_bench_mix_ci95"]))
m("horMixOne", sg(r1["delta_L_pct_bench_mix"], 1))
m("horMixOneCI", ci(r1["delta_L_pct_bench_mix_ci95"]))
m("horLRPmixtwo", f(r2["lrp_bench_mix"]["L"], 3))
m("horLRPmixone", f(r1["lrp_bench_mix"]["L"], 3))

# post hoc, after the review: the training-file analyses at the operational class mix. The
# organizers put most high-risk events in the test, so the training file's share is depleted
# and the pooled events (ESA's full population) carry the operational one
mx_ = REVM["class_mixes"]
assert mx_["pooled"]["n"] == PUB["n_eligible"] and mx_["pooled"]["n_high_risk"] == PUB["n_high_risk_eligible"]
assert mx_["training_file"]["prevalence"] < mx_["pooled"]["prevalence"] < mx_["test"]["prevalence"]
assert abs(mx_["training_file"]["prevalence"] - cv1["n_hr_eligible"] / cv1["n_eligible"]) < 1e-12
m("opPct", f(100 * mx_["pooled"]["prevalence"], 1))
rc_, wr_, hm_ = REVM["round2_cv"], REVM["winner_rules"], REVM["horizon"]
for part in (rc_, wr_, hm_):
    assert part["reproduces_registered"] and all(part["reproduces_registered"].values())
op_r2 = rc_["mixes"]["operational"]["round2"]
m("cvOpMixdL", sg(op_r2["delta_L_pct"], 1))
m("cvOpMixCI", ci(op_r2["delta_L_pct_ci95"]))
m("cvOpPrecLRP", f(rc_["mixes"]["operational"]["lrp"]["precision"], 2))
m("cvOpPrecRtwo", f(op_r2["precision"], 2))
m("cvOpRecLRP", f(rc_["mixes"]["operational"]["lrp"]["recall"], 2))
m("cvOpRecRtwo", f(op_r2["recall"], 2))
op_own = rc_["mixes"]["operational"]["operational"]
m("cvOpTunedDL", sg(op_own["delta_L_pct"], 1))
m("cvOpTunedCI", ci(op_own["delta_L_pct_ci95"]))
wo_ = wr_["mixes"]["operational"]
m("oosOpdL", sg(wo_["delta_L_pct"], 1))
m("oosOpCI", ci(wo_["delta_L_pct_ci95"]))
m("oosOpPrecLRP", f(wo_["lrp"]["precision"], 2))
m("oosOpPrecSesc", f(wo_["sesc_cascade"]["precision"], 2))
hr2_, hr1_ = hm_["rows"][cut[0]]["operational"], hm_["rows"][cut[-1]]["operational"]
assert hm_["n_scored"] == HOR["n_scored"] and hm_["cutoffs"] == HOR["cutoffs"]
m("horRatioOp", f(100 * (1 - hr1_["lrp_L_ratio_to_2d"]), 0))
m("horRatioOpLo", f(100 * (1 - hr1_["lrp_L_ratio_to_2d_ci95"][1]), 0))
m("horRatioOpHi", f(100 * (1 - hr1_["lrp_L_ratio_to_2d_ci95"][0]), 0))
m("horOpTwo", sg(hr2_["delta_L_pct"], 1))
m("horOpTwoCI", ci(hr2_["delta_L_pct_ci95"]))
m("horOpOne", sg(hr1_["delta_L_pct"], 1))
m("horOpOneCI", ci(hr1_["delta_L_pct_ci95"]))

# --- P4 and the earlier VIGIL-E runs -----------------------------------------------------

ew1 = P1["cv"]["early_warning"]
m("eHurdlePR", f(ew1["e_hurdle"]["pr_auc"]["mean"], 3))
m("eLogitPR", f(ew1["e_logit"]["pr_auc"]["mean"], 3))
m("eHurdleD", sg(ew1["e_hurdle"]["delta_pr_auc_vs_logit"]["mean"], 3))
m("eHurdleCI", ci(ew1["e_hurdle"]["delta_pr_auc_vs_logit"]["ci95"], 3))
m("eHurdleCal", f(ew1["e_hurdle"]["cal_slope"]["mean"], 2))
m("eServingD", sg(ew1["e_serving"]["delta_pr_auc_vs_logit"]["mean"], 3))
m("eServingCI", ci(ew1["e_serving"]["delta_pr_auc_vs_logit"]["ci95"], 3))
ew2 = P2["cv"]["early_warning"]
m("eHurdleEsD", sg(ew2["e_hurdle_eserving"]["delta_pr_auc_vs_logit"]["mean"], 3))
m("eHurdleEsCI", ci(ew2["e_hurdle_eserving"]["delta_pr_auc_vs_logit"]["ci975"], 3))
m("eRatio", f(ew1["e_hurdle"]["pr_auc"]["mean"] / ew1["e_logit"]["pr_auc"]["mean"], 1))
S, CT = ABL["summary"], ABL["contrasts"]
for key, name in (("Zero", "A0_all_benchmark"), ("One", "A1_without_covariance"),
                  ("Two", "A2_serving_plus_covariance"), ("Logit", "e_logit")):
    m(f"abl{key}PR", f(S[name]["pr_auc"], 3))
for key, name in (("ZeroOne", "A0_minus_A1"), ("OneLogit", "A1_minus_logit"),
                  ("TwoLogit", "A2_minus_logit"), ("TwoZero", "A2_minus_A0"),
                  ("ZeroLogit", "A0_minus_logit")):
    m(f"abl{key}", sg(CT[name]["mean"], 3))
    m(f"abl{key}CI", ci(CT[name]["ci95"], 3))
m("ablNcov", str(ABL["feature_counts"]["_covariance"]))
m("ablNall", str(ABL["feature_counts"]["A0_all_benchmark"]))
m("ablNone", str(ABL["feature_counts"]["A1_without_covariance"]))
m("ablNtwo", str(ABL["feature_counts"]["A2_serving_plus_covariance"]))
m("ablReading", ABL["reading"])
m("ablShare", f(100 * CT["A0_minus_A1"]["ci95"][1] / CT["A0_minus_logit"]["mean"], 0))
m("ablCovCap", f(CT["A0_minus_A1"]["ci95"][1], 3))

# --- P6: coherence of the CDM risk (test events) ----------------------------------------------

lo_ = COH["a_low_end"]
m("cohMatched", n(COH["n_matched_raw"]))
m("cohRho", f(COH["selection"]["rho"], 2))
for key, name in (("", "primary_any_later_cdm"), ("Fin", "final_cdm_only"), ("Far", "band_-29.9_-12_any_later"),
                  ("Mid", "band_-12_-8_any_later"), ("Near", "band_-8_-6_any_later"),
                  ("Sent", "sentinel_-30_any_later")):
    b_ = lo_[name]
    m(f"coh{key}N", n(b_["n"]))
    m(f"coh{key}Obs", n(b_["observed"]))
    cm = b_["coherent_max"]
    m(f"coh{key}Max", f(cm, 0) if cm >= 10 else (f(cm, 2) if cm >= 0.1 else sci(cm, 1)))
    if b_["p_chernoff"] < 1:
        m(f"coh{key}P", sci(b_["p_chernoff"], 1))
m("cohRatio", f(lo_["primary_any_later_cdm"]["ratio"], 0))
m("cohReading", lo_["reading"])
hi_ = COH["b_high_end"]
m("cohHighN", n(hi_["final"]["n"]))
m("cohHighRatio", f(hi_["final"]["ratio"], 2))
m("cohHighCI", ci(hi_["final"]["ci95"], 2, signed=False))
m("cohNextRatio", f(hi_["next_cdm"]["ratio"], 2))
m("cohNextCI", ci(hi_["next_cdm"]["ci95"], 2, signed=False))
m("cohStay", f(100 * hi_["final"]["share_still_high_risk_weighted"], 0))
up_ = COH["c_updates"]
m("updPairs", n(up_["pairs"]))
m("updEvents", n(up_["events"]))
m("updMed", f(up_["median_d2"], 2))
m("updShare", f(100 * up_["share_above_chi2_3_p99"], 0))
m("updCI", ci([100 * v for v in up_["ci95"]], 1, signed=False))
# post hoc: the update test in the encounter plane, and robustness of (a)
upp = CPH["c_updates_plane"]
assert upp["pairs"] == up_["pairs"]                                          # the same updates
assert abs(upp["registered_share_above_chi2_3_p99"] - up_["share_above_chi2_3_p99"]) < 1e-12
m("updPlaneShare", f(100 * upp["share_above_chi2_2_p99"], 1))
m("updPlaneCI", ci([100 * v for v in upp["ci95"]], 1, signed=False))
m("updPlaneMed", f(upp["median_d2"], 2))
m("updChiMed", f(upp["median_chi2_2"], 2))
m("updInflate", f(upp["median_ratio_registered_to_plane"], 0))
m("updCos", sci(upp["median_abs_cos_position_velocity"], 1))
rob = CPH["a_robustness"]
assert rob["crossings"] == lo_["primary_any_later_cdm"]["observed"]
assert abs(rob["coherent_max"] - lo_["primary_any_later_cdm"]["coherent_max"]) < 1e-12
m("cohMissions", str(rob["n_missions"]))
m("cohMarkov", f(rob["markov_bound_without_independence"], 2))
pass_ = rob["training_candidates_passing_test_filters"]
m("cohPassLo", f(100 * pass_["band_-12_-8"]["share_eligible"], 0))
m("cohPassHi", f(100 * pass_["band_-29.9_-12"]["share_eligible"], 0))
reg_ = CPH["c_regularization"]
assert abs(reg_["0.999"]["share_above_chi2_2_p99"] - upp["share_above_chi2_2_p99"]) < 1e-12   # the registered clip
shares_ = [reg_[k]["share_above_chi2_2_p99"] for k in ("none", "0.9999", "0.999", "0.99")]
m("regLo", f(100 * min(shares_), 1))
m("regHi", f(100 * max(shares_), 1))
m("regCorr", f(100 * reg_["share_of_correlations_beyond_0.999"], 1))
mah_ = CPH["d_esa_mahalanobis"]
m("mahRot", f(100 * mah_["rotated"]["share_within_1pct"], 0))
m("mahUn", f(100 * mah_["unrotated"]["share_within_1pct"], 0))
tol_ = CPH["a_tolerance_matching"]
for k_ in ("primary_any_later_cdm", "band_-29.9_-12_any_later", "band_-12_-8_any_later", "band_-8_-6_any_later"):
    assert tol_[k_]["n"] == lo_[k_]["n"] and tol_[k_]["observed"] == lo_[k_]["observed"]      # unchanged
assert tol_["sentinel_-30_any_later"]["observed"] == lo_["sentinel_-30_any_later"]["observed"]
assert tol_["n_matched"] - COH["n_matched_raw"] == tol_["sentinel_-30_any_later"]["n"] - lo_["sentinel_-30_any_later"]["n"]
m("tolMatched", n(tol_["n_matched"]))
m("tolExtra", str(tol_["n_matched"] - COH["n_matched_raw"]))
rc_ = COH["d_recompute_check"]
m("recN", n(rc_["rotated"]["n"]))
m("recUnrot", f(rc_["unrotated"]["median_abs_dlog10"], 2))
m("recRot", f(rc_["rotated"]["median_abs_dlog10"], 3))
m("recWithinUnrot", f(100 * rc_["unrotated"]["share_within_0.1"], 0))
m("recWithinRot", f(100 * rc_["rotated"]["share_within_0.1"], 0))
m("recFactor", f(rc_["unrotated"]["median_abs_dlog10"] / rc_["rotated"]["median_abs_dlog10"], 0))

# --- P7-P8: public-data counterparts, and VIGIL's classifier as a probability forecast -------

p7, p8f, p8a = PUB["p7_early_warning"], PUB["p8_forecast"], PUB["p8_false_alarms"]
ps_, pc_ = p7["pooled"]["summary"], p7["pooled"]["contrasts"]
assert abs(ps_["A0"]["pr_auc"] - ABL["summary"]["A0_all_benchmark"]["pr_auc"]) < 1e-12   # same folds as P4
m("pubNfeat", str(PUB["feature_counts"]["PUB"]))
m("pubPR", f(ps_["PUB"]["pr_auc"], 3))
m("pubLivePR", f(ps_["LIVE"]["pr_auc"], 3))
m("pubLogitPR", f(ps_["e_logit"]["pr_auc"], 3))
m("pubRatio", f(ps_["PUB"]["pr_auc"] / ps_["e_logit"]["pr_auc"], 2))
for key, name in (("D", "PUB_minus_logit"), ("ZeroMinus", "A0_minus_PUB"), ("LiveD", "PUB_minus_LIVE"),
                  ("LiveLogit", "LIVE_minus_logit")):
    m(f"pub{key}", sg(pc_[name]["mean"], 3))
    m(f"pub{key}CI", ci(pc_[name]["ci95"], 3))
m("pubShare", f(100 * pc_["PUB_minus_logit"]["mean"] / pc_["A0_minus_logit"]["mean"], 0))
m("pubReading", p7["reading"])
tq = p7["test_events_only"]
m("pubTestEsc", n(tq["n_escalations"]))
m("pubTestD", sg(tq["contrasts"]["PUB_minus_logit"]["mean"], 3))
m("pubTestCI", ci(tq["contrasts"]["PUB_minus_logit"]["ci95"], 3))
m("pubZeroTestD", sg(tq["contrasts"]["A0_minus_logit"]["mean"], 3))
m("pubZeroTestCI", ci(tq["contrasts"]["A0_minus_logit"]["ci95"], 3))
m("pubRecall", f(ps_["PUB"]["recall_at_budget"], 2))
m("pubNeligible", n(PUB["n_eligible"]))
m("pubNhr", n(PUB["n_high_risk_eligible"]))
for key, name in (("Zero", "A0"), ("Pub", "PUB")):
    dll = p8f[name]["d_logloss_vs_calibrated"]
    m(f"ll{key}", f(p8f[name]["logloss"], 4))
    m(f"ll{key}D", sg(dll["mean"], 4))
    m(f"ll{key}CI", ci(dll["ci95"], 4))
    m(f"ll{key}Pct", f(100 * dll["mean"] / p8f["calibrated"]["logloss"], 0))
    dbr = p8f[name]["d_brier_vs_calibrated"]
    m(f"br{key}Drop", f(-100 * dbr["mean"] / p8f["calibrated"]["brier"], 0))
    m(f"br{key}DropCI", ci([-100 * v / p8f["calibrated"]["brier"] for v in dbr["ci95"][::-1]], 0, signed=False))
    m(f"llPct{key}CI", ci([100 * v / p8f["calibrated"]["logloss"] for v in dll["ci95"]], 0))
m("llCal", f(p8f["calibrated"]["logloss"], 4))
m("llZeroDrop", f(-100 * p8f["A0"]["d_logloss_vs_calibrated"]["mean"] / p8f["calibrated"]["logloss"], 0))
m("llPubDrop", f(-100 * p8f["PUB"]["d_logloss_vs_calibrated"]["mean"] / p8f["calibrated"]["logloss"], 0))
m("llReading", p8f["reading"])
fa_ = p8a["false_alarms_removed_at_95pct"]
m("faCalls", n(p8a["n_calls"]))
m("faStay", n(p8a["n_stay_high_risk"]))
for key, name in (("Zero", "A0"), ("Pub", "PUB"), ("Cal", "calibrated")):
    m(f"fa{key}", f(100 * fa_[name]["mean"], 0))
    m(f"fa{key}CI", ci([100 * v for v in fa_[name]["ci95"]], 0, signed=False))
    m(f"auc{key}", f(p8a["roc_auc"][name], 3))
for key, name in (("Zero", "A0"), ("Pub", "PUB")):
    m(f"auc{key}D", sg(p8a[f"d_roc_auc_{name}_vs_calibrated"]["mean"], 3))
    m(f"auc{key}CI", ci(p8a[f"d_roc_auc_{name}_vs_calibrated"]["ci95"], 3))
# what F2 would do with that filter, at the official test's counts (115 TP, 63 FP, 35 FN)
tp_, fp_, fn_ = 115, 63, 35
assert abs(5 * tp_ / (5 * tp_ + 4 * fn_ + fp_) - fid["f2"]["lab"]) < 5e-4
keep_, cut_ = 0.95, fa_["A0"]["mean"]
f2f = 5 * tp_ * keep_ / (5 * tp_ * keep_ + 4 * (fn_ + tp_ * (1 - keep_)) + fp_ * (1 - cut_))
m("ftwoFilter", f(f2f, 3))
pr1 = P1["cv"]["regression"]["hurdle_lrp"]["delta_L_pct_vs_lrp"]
m("poolLD", sg(pr1["mean"], 1))
m("poolLCI", ci(pr1["ci95"], 1))

# post hoc: the latest-risk baselines ranked by the latest risk itself (no pooled per-fold fits)
ew_ = EWP["early_warning"]
assert all(ew_["reproduces_registered_reports"].values())
lat_ = ew_["latest_risk_ranking"]
m("latPR", f(lat_["pr_auc"], 3))
m("latROC", f(lat_["roc_auc"], 3))
m("latRecall", f(lat_["recall_at_budget"], 2))
for key, name in (("Zero", "A0"), ("One", "A1"), ("Two", "A2"), ("Pub", "PUB"), ("Live", "LIVE"), ("Logit", "e_logit")):
    c_, rc_ = ew_["contrasts_vs_latest_risk"][name], ew_["roc_contrasts_vs_latest_risk"][name]
    m(f"lat{key}D", sg(c_["mean"], 3)); m(f"lat{key}CI", ci(c_["ci95"], 3))
    m(f"lat{key}RocD", sg(rc_["mean"], 3)); m(f"lat{key}RocCI", ci(rc_["ci95"], 3))
    m(f"rec{key}", f(ew_["recall_at_budget"][name], 2))
# the sentences that say so: no PR-area gain is significant, every hurdle's ROC gain but the live feed's is
for name in ("A0", "A1", "A2", "PUB", "LIVE"):
    assert ew_["contrasts_vs_latest_risk"][name]["ci95"][0] < 0 < ew_["contrasts_vs_latest_risk"][name]["ci95"][1]
    assert ew_["recall_at_budget"][name] <= lat_["recall_at_budget"]
for name in ("A0", "A1", "A2", "PUB"):
    assert ew_["roc_contrasts_vs_latest_risk"][name]["ci95"][0] > 0
assert ew_["roc_contrasts_vs_latest_risk"]["LIVE"]["ci95"][0] < 0
assert ew_["contrasts_vs_latest_risk"]["e_logit"]["ci95"][1] < 0
assert abs(ew_["pooled_roc_auc"]["A0"] - ABL["summary"]["A0_all_benchmark"]["roc_auc"]) < 1e-12
m("ewRocZero", f(ew_["pooled_roc_auc"]["A0"], 3))
m("latCut", f(100 * (1 - ew_["pooled_pr_auc"]["e_logit"] / lat_["pr_auc"]), 0))
m("latGain", f(100 * (ew_["pooled_pr_auc"]["A0"] / lat_["pr_auc"] - 1), 0))
fa2_ = EWP["false_alarms"]
assert abs(fa2_["pooled_roc_auc"]["A0"] - p8a["roc_auc"]["A0"]) < 1e-12
m("aucLat", f(fa2_["latest_risk_ranking"]["roc_auc"], 3))
m("faLat", f(100 * fa2_["latest_risk_ranking"]["false_alarms_removed_at_95pct"]["mean"], 0))
m("faLatCI", ci([100 * v for v in fa2_["latest_risk_ranking"]["false_alarms_removed_at_95pct"]["ci95"]], 0, signed=False))
for key, name in (("Zero", "A0"), ("Pub", "PUB")):
    m(f"aucLat{key}D", sg(fa2_["d_roc_auc_vs_latest_risk"][name]["mean"], 3))
    m(f"aucLat{key}CI", ci(fa2_["d_roc_auc_vs_latest_risk"][name]["ci95"], 3))
    assert fa2_["d_roc_auc_vs_latest_risk"][name]["ci95"][0] > 0

# post hoc, after the review: P6 counted by mission, and the crossings by chaser type and dilution
rp_ = REVC["primary"]
assert rp_["reproduces_registered"] and rp_["observed"] == COH["a_low_end"]["primary_any_later_cdm"]["observed"]
assert rp_["coherent_max_missions"] <= rp_["coherent_max"] + 1e-12
m("cohBandMissions", n(rp_["n_missions"]))
assert rp_["missions_with_crossing"] == rob["n_missions"]   # the 7 of the robustness check
m("cohMissionsCross", n(rp_["missions_with_crossing"]))
m("cohMissionMax", f(rp_["coherent_max_missions"], 2))
m("cohMissionP", sci(rp_["p_chernoff_missions"]))
bt_ = REVC["by_chaser_type"]
deb_, unk_, pay_ = bt_["DEBRIS"]["observed"], bt_["UNKNOWN"]["observed"], bt_["PAYLOAD"]["observed"]
assert deb_ + unk_ + pay_ == rp_["observed"]          # no other chaser type crossed
m("cohDebObs", n(deb_)); m("cohUnkObs", n(unk_)); m("cohPayObs", n(pay_))
# debris cannot maneuver: its crossings alone, against the coherent maximum for the whole range
assert deb_ > 2 * rp_["coherent_max"]
m("cohDebTimes", f(deb_ / rp_["coherent_max"], 0))
sp_ = REVC["splits"]
assert sp_["dilution_region"]["observed"] + sp_["outside_dilution"]["observed"] == rp_["observed"]
assert sp_["scaling_missing"]["n"] == 0
m("cohDilObs", n(sp_["dilution_region"]["observed"])); m("cohDilN", n(sp_["dilution_region"]["n"]))
m("cohOutObs", n(sp_["outside_dilution"]["observed"])); m("cohOutN", n(sp_["outside_dilution"]["n"]))
m("cohOutRatio", f(sp_["outside_dilution"]["ratio"], 0))
m("cohOutMissionP", sci(sp_["outside_dilution"]["p_chernoff_missions"]))

# post hoc, after the review: corrected repeated k-fold intervals (Nadeau-Bengio / Bouckaert-Frank),
# an isotonic latest-risk baseline, false-alarm thresholds set in the training folds, rotated covariances
rv_ = REVV
assert all(rv_["reproduces_registered"].values())
assert rv_["rotation_coverage"]["unrotated"] == 0
fc_, fa3_, ew3_ = rv_["p8_forecast"], rv_["p8_false_alarms"], rv_["early_warning"]
ll_ = fc_["logloss_mean_per_fold"]


def pct_ci(c, base, d=0):
    return ci([100 * v / base for v in c["ci95"]], d)


for key, name, base in (("Cal", "A0_minus_calibrated", "calibrated"), ("Iso", "A0_minus_isotonic", "isotonic"),
                        ("PubIso", "PUB_minus_isotonic", "isotonic")):
    c_ = fc_[name]
    m(f"revLl{key}Pct", f(100 * c_["mean"] / ll_[base], 0))
    m(f"revLl{key}PctCI", pct_ci(c_, ll_[base]))
    m(f"revLl{key}P", f(c_["p"], 3) if c_["p"] >= 0.001 else sci(c_["p"]))
m("revLlIsoDrop", f(-100 * fc_["A0_minus_isotonic"]["mean"] / ll_["isotonic"], 0))
assert fc_["A0_minus_calibrated"]["ci95"][1] < 0 and fc_["A0_minus_isotonic"]["ci95"][1] < 0
assert fc_["PUB_minus_isotonic"]["ci95"][0] < 0 < fc_["PUB_minus_isotonic"]["ci95"][1]      # "not significant"
assert fc_["isotonic_minus_calibrated"]["ci95"][0] < 0 < fc_["isotonic_minus_calibrated"]["ci95"][1]
tot_ = fa3_["pooled_totals"]
m("revFaZero", f(100 * tot_["A0"]["fa_removed"], 0)); m("revFaKept", f(100 * tot_["A0"]["kept"], 0))
m("revFaPub", f(100 * tot_["PUB"]["fa_removed"], 0)); m("revFaPubKept", f(100 * tot_["PUB"]["kept"], 0))
m("revFaLat", f(100 * tot_["latest"]["fa_removed"], 0)); m("revFaLatKept", f(100 * tot_["latest"]["kept"], 0))
m("revFaZeroCI", ci([100 * v for v in fa3_["fa_removed"]["A0"]["ci95"]], 0, signed=False))
for key, name in (("Zero", "A0_minus_latest"), ("Pub", "PUB_minus_latest")):
    c_ = fa3_[name]
    m(f"revFa{key}D", sg(100 * c_["mean"], 0)); m(f"revFa{key}DCI", ci([100 * v for v in c_["ci95"]], 0))
    m(f"revFa{key}P", f(c_["p"], 3))
assert fa3_["A0_minus_latest"]["ci95"][0] > 0 and fa3_["PUB_minus_latest"]["ci95"][0] > 0
assert fa3_["PUB_minus_latest"]["p"] > 0.01                     # "borderline"
assert tot_["A0"]["kept"] >= 0.95 and tot_["latest"]["kept"] >= 0.94
for key, name in (("Zero", "A0"), ("One", "A1"), ("Two", "A2"), ("Pub", "PUB"), ("Live", "LIVE")):
    a_, b_ = ew3_["pr_vs_latest"][name], ew3_["roc_vs_latest"][name]
    m(f"revPr{key}D", sg(a_["mean"], 3)); m(f"revPr{key}CI", ci(a_["ci95"], 3))
    m(f"revRoc{key}D", sg(b_["mean"], 3)); m(f"revRoc{key}CI", ci(b_["ci95"], 3)); m(f"revRoc{key}P", f(b_["p"], 2))
    assert a_["ci95"][0] < 0 < a_["ci95"][1]                    # no PR-area gain is significant
for name in ("A0", "A1", "PUB"):
    assert ew3_["roc_vs_latest"][name]["ci95"][0] > 0          # these ROC gains survive the correction
for name in ("A2", "LIVE"):
    assert ew3_["roc_vs_latest"][name]["ci95"][0] < 0          # these do not
# the frame error: rotating the chaser's covariance changes no result
for c_ in (fc_["A0R_minus_A0"], fa3_["A0R_minus_A0"], ew3_["A0R_minus_A0"]["pr"], ew3_["A0R_minus_A0"]["roc"]):
    assert c_["ci95"][0] < 0 < c_["ci95"][1]
m("revRotLl", sg(fc_["A0R_minus_A0"]["mean"], 4)); m("revRotLlCI", ci(fc_["A0R_minus_A0"]["ci95"], 4))
m("revRotFa", sg(100 * fa3_["A0R_minus_A0"]["mean"], 1)); m("revRotFaCI", ci([100 * v for v in fa3_["A0R_minus_A0"]["ci95"]], 1))
m("revRotPr", sg(ew3_["A0R_minus_A0"]["pr"]["mean"], 3)); m("revRotPrCI", ci(ew3_["A0R_minus_A0"]["pr"]["ci95"], 3))
m("revRotN", n(rv_["rotation_coverage"]["rotated"]))

# --- tables ----------------------------------------------------------------------------------

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "numbers.tex").write_text(
    "% generated by scripts/paper_assets.py -- do not edit\n"
    + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in MACROS))

ROWS = [("lrp", "Latest-risk baseline (LRP)", "--", R1),
        ("hurdle_lrp", "VIGIL, benchmark features", "R1 primary", R1),
        ("hurdle_lrp_serving", "VIGIL, live-feed features", "R1 primary", R1),
        ("hurdle_lrp_serving_bench", "VIGIL, live feed, test-mix decision", "R2 primary", R2),
        ("hurdle", "Hurdle, free decision", "exploratory", R1),
        ("hurdle_eserving", "Hurdle, VIGIL-E live-feed features", "exploratory", R2),
        ("gbm_quant", "Quantile gradient boosting", "exploratory", R1),
        ("gbm_resid_aug", "Residual boosting, augmented", "exploratory", R1),
        ("sesc", "Organizers' 25-tree recipe", "exploratory", R1)]
lines = []
for key, label, role, rep in ROWS:
    row = sealed(rep, key)
    dd = row["delta_L_pct_vs_lrp"]
    if key == "lrp":
        dl, cc = "--", "--"
    else:
        big = abs(dd["mean"]) >= 100
        dl = sg(dd["mean"], 0 if big else 1)
        c95 = dd["ci975"] if role == "R2 primary" else dd["ci95"]
        cc = ci(c95, 0 if big else 1) + (r"$^{\ddagger}$" if role == "R2 primary" else "")
    L = row["L"]["mean"]
    Ls = f(L, 3) if L < 10 else f(L, 0)
    lines.append(f"{label} & {role} & {Ls} & {f(row['mse_hr']['mean'], 3)} & "
                 f"{f(row['f2']['mean'], 3)} & {dl} & {cc} \\\\")
(OUT / "tab_sealed.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")   # \input + \bottomrule breaks tabular

lines = []
HMIX = ("training_file", "operational", "test")      # the training file's own mix, ESA's, the test's
for cstr in cut:
    r, cells = rows[cstr], []
    for mx in HMIX:
        q = hm_["rows"][cstr][mx]
        cells.append(f"{f(q['lrp_L'], 3)} & {sg(q['delta_L_pct'], 1)} {ci(q['delta_L_pct_ci95'])}")
    lines.append(f"{f(float(cstr), 2)} & {f(r['mean_inputs_per_event'], 1)} & " + " & ".join(cells) + " \\\\")
(OUT / "tab_horizon.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")   # \input + \bottomrule breaks tabular

lines = []
for name, label, nf in (("e_logit", "Logistic regression on the latest risk (registered)", "1"),
                        ("A1_without_covariance", "A1: benchmark features without covariance",
                         str(ABL["feature_counts"]["A1_without_covariance"])),
                        ("A2_serving_plus_covariance", "A2: live-feed features plus covariance",
                         str(ABL["feature_counts"]["A2_serving_plus_covariance"])),
                        ("A0_all_benchmark", "A0: all benchmark features (VIGIL-E)",
                         str(ABL["feature_counts"]["A0_all_benchmark"]))):
    s = S[name]
    lines.append(f"{label} & {nf} & {f(s['pr_auc'], 3)} & {f(s['roc_auc'], 3)} & "
                 f"{f(s['recall_at_budget'], 2)} \\\\")
    if name == "e_logit":
        lines.append(f"Latest risk itself, ranked (post hoc) & 1 & {f(lat_['pr_auc'], 3)} & {f(lat_['roc_auc'], 3)} & "
                     f"{f(lat_['recall_at_budget'], 2)} \\\\")
(OUT / "tab_ablation.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")   # \input + \bottomrule breaks tabular

lines = []
for name, label, nf in (("calibrated", "Latest risk, calibrated (registered)", "1"),
                        ("LIVE", "Live-feed quantities", str(PUB["feature_counts"]["LIVE"])),
                        ("PUB", "Public-data counterparts", str(PUB["feature_counts"]["PUB"])),
                        ("A0", "All CDM fields (VIGIL)", str(PUB["feature_counts"]["A0"]))):
    pr_ = ps_["e_logit" if name == "calibrated" else name]["pr_auc"]
    if name in p8f:
        ll_ = f(p8f[name]["logloss"], 4)
        auc_ = f(p8a["roc_auc"][name], 3)
        fa__ = f(100 * fa_[name]["mean"], 0) + r"\%"
    else:
        ll_ = auc_ = fa__ = "--"
    lines.append(f"{label} & {nf} & {ll_} & {auc_} & {fa__} & {f(pr_, 3)} \\\\")
    if name == "calibrated":
        lines.append(f"Latest risk itself, ranked (post hoc) & 1 & -- & {f(fa2_['latest_risk_ranking']['roc_auc'], 3)} & "
                     f"{f(100 * fa2_['latest_risk_ranking']['false_alarms_removed_at_95pct']['mean'], 0)}\\% & "
                     f"{f(lat_['pr_auc'], 3)} \\\\")
(OUT / "tab_forecast.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")   # \input + \bottomrule breaks tabular

# --- figures -------------------------------------------------------------------------------------

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
INK, BLUE, ORANGE, GREY, RED = "#1d1d1f", "#1f6fb2", "#d9822b", "#8a8f98", "#b23b3b"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


# fig. 2 (fig2_gains.pdf): every "gain over the baseline" on one axis
items = []
for key, rep, label in (("hurdle_lrp", R1, "VIGIL, benchmark features (R1)"),
                        ("hurdle_lrp_serving", R1, "VIGIL, live-feed features (R1)"),
                        ("hurdle_lrp_serving_bench", R2, "VIGIL, test-mix decision (R2)")):
    dd = sealed(rep, key)["delta_L_pct_vs_lrp"]
    lvl = dd["ci975"] if key.endswith("_bench") else dd["ci95"]
    items.append(("pre-registered, sealed", label, dd["mean"], lvl, BLUE, "o"))
items.append(("chosen on the test", "Organizers' LightGBM (1 of 5,000)", -1.4, None, GREY, "D"))
items.append(("chosen on the test", "2019 winner's rules, on the test", c["sesc_delta_L_pct_full"],
              None, GREY, "D"))
items.append(("out of sample", "2019 winner's rules, training file",
              d["bench_mix"]["delta_L_pct"], d["bench_mix"]["delta_L_pct_ci95"], ORANGE, "s"))
items.append(("no information", f"Leaderboard feedback only ({ATT['budget']} tries)",
              k0["full_dl_pct"]["median"], [k0["full_dl_pct"]["p05"], k0["full_dl_pct"]["p95"]], RED, "v"))
fig, ax = plt.subplots(figsize=(6.3, 3.0))
ys = np.arange(len(items))[::-1]
prev = None
for y, (grp, label, x, lo_hi, col, mk) in zip(ys, items):
    if lo_hi:
        ax.plot(lo_hi, [y, y], color=col, lw=2.0, solid_capstyle="butt")
    ax.plot([x], [y], marker=mk, color=col, ms=5.5, mfc=col if lo_hi else "white", mew=1.4)
    lx = x + (2.4 if x >= 0 else -2.4) if abs(x) < 1.5 else x   # keep labels off the zero line
    ax.text(lx, y + 0.24, sg(x, 1).replace(MINUS, "−") + "%", ha="center", va="bottom",
            fontsize=7, color=col)
    if grp != prev:
        ax.text(21.5, y, grp, ha="right", va="center", fontsize=6.8, color="#6e6e73", style="italic")
        if prev is not None:
            ax.axhline(y + 0.5, color="#d2d2d7", lw=0.6)
        prev = grp
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(ys)
ax.set_yticklabels([it[1] for it in items])
ax.set_xlabel("Change in the official loss L against the latest-risk baseline (%; negative is better)")
ax.set_xlim(-28, 22)
ax.set_ylim(-0.6, len(items) - 0.4)
ax.grid(axis="x", color="#ececf0", lw=0.6)
ax.set_axisbelow(True)
save(fig, "fig2_gains.pdf")

# fig. 4 (fig3_horizon.pdf): the information horizon
cx = np.array([float(cc) for cc in cut])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.55))
pct_ = {k: f"{100 * v['prevalence']:.1f}%" for k, v in mx_.items()}
HSTYLE = {"training_file": (GREY, f"training file ({pct_['training_file']} high-risk)", -0.03),
          "operational": (BLUE, f"operational ({pct_['pooled']}, post hoc)", 0.0),
          "test": (ORANGE, f"test ({pct_['test']})", 0.03)}
for mx in HMIX:
    col, lab, off = HSTYLE[mx]
    q = [hm_["rows"][cc][mx] for cc in cut]
    y = np.array([100 * v["lrp_L_ratio_to_2d"] for v in q])
    lo = np.array([100 * v["lrp_L_ratio_to_2d_ci95"][0] for v in q])
    hi = np.array([100 * v["lrp_L_ratio_to_2d_ci95"][1] for v in q])
    a1.errorbar(cx + off, y, yerr=[y - lo, hi - y], color=col, marker="o", ms=3.5, lw=1.4,
                capsize=2, label=lab)
a1.axhline(100, color=INK, lw=0.6, ls=":")
a1.invert_xaxis()
a1.set_ylim(0, 125)
a1.set_xlabel("Cutoff (days before TCA)")
a1.set_ylabel("Baseline's loss (% of its 2-day value)")
a1.set_title("(a) One more day of messages")
a1.legend(frameon=False, loc="lower left")
for mx in HMIX:
    col, lab, off = HSTYLE[mx]
    q = [hm_["rows"][cc][mx] for cc in cut]
    y = np.array([v["delta_L_pct"] for v in q])
    lo = np.array([v["delta_L_pct_ci95"][0] for v in q])
    hi = np.array([v["delta_L_pct_ci95"][1] for v in q])
    a2.errorbar(cx + off, y, yerr=[y - lo, hi - y], color=col, marker="o", ms=3.5, lw=1.2,
                capsize=2, label=lab)
a2.axhline(0, color=INK, lw=0.8)
a2.invert_xaxis()
a2.set_xlabel("Cutoff (days before TCA)")
a2.set_ylabel("VIGIL against baseline (% change in L)")
a2.set_title("(b) The model's edge")
a2.legend(frameon=False, loc="lower right")
fig.tight_layout(w_pad=2.0)
save(fig, "fig3_horizon.pdf")

# fig. 3 (fig4_resolution.pdf): what the test can resolve
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.5), gridspec_kw={"width_ratios": [1.2, 1]})
boot = RES["a_lrp_bootstrap"]["plain"]["L"]
a1.plot([boot["p025"], boot["p975"]], [0, 0], color=GREY, lw=7, alpha=0.3, solid_capstyle="butt")
a1.plot([boot["p05"], boot["p95"]], [0, 0], color=GREY, lw=7, alpha=0.55, solid_capstyle="butt")
a1.plot([fid["L"]["lab"]], [0], marker="|", color=INK, ms=16, mew=2)
a1.text(fid["L"]["lab"], -0.32, "baseline on\nthe real test", ha="center", va="top", fontsize=6.5)
lb = [s_ for _, s_ in compete.LEADERBOARD]
a1.plot(lb, [0.6] * len(lb), "o", color=ORANGE, ms=4, alpha=0.9)
vig = sealed(R2, "hurdle_lrp_serving_bench")["L"]["mean"]
a1.plot([vig], [0.6], "o", color=BLUE, ms=5.5, zorder=5)
a1.annotate("VIGIL (R2)", (vig, 0.6), xytext=(vig + 0.04, 0.32), fontsize=6.5, color=BLUE,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
a1.text(0.553, 0.86, "2019 teams that matched or beat the baseline", color=ORANGE, fontsize=6.6)
a1.text(boot["p025"], -0.9, "the baseline's own 95% range when the test is re-drawn",
        color="#6e6e73", fontsize=6.5)
a1.set_ylim(-1.05, 1.1)
a1.set_yticks([])
a1.spines["left"].set_visible(False)
a1.set_xlabel("Official loss L")
a1.set_title("(a) Scores against the test's own noise")
lab = ["Visible 90%\n(the feedback)", "Full test\n(final ranking)", "Hidden 10%"]
TOP = 45
for i, key in enumerate(("visible_dl_pct", "full_dl_pct", "holdout_dl_pct")):
    dd = k0[key]
    hi = min(dd["p95"], TOP)
    a2.plot([i, i], [dd["p05"], hi], color=RED, lw=2.2, solid_capstyle="butt")
    a2.plot([i], [dd["median"]], "v", color=RED, ms=6)
    if dd["p95"] > TOP:
        a2.annotate("", (i, TOP + 3), (i, TOP - 2), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1))
        a2.text(i + 0.12, TOP - 6, f"to {sg(dd['p95'], 0).replace(MINUS, chr(8722))}%", fontsize=6.5,
                color=RED)
a2.axhline(0, color=INK, lw=0.8)
a2.set_xticks(range(3))
a2.set_xticklabels(lab, fontsize=6.5)
a2.set_xlim(-0.5, 2.6)
a2.set_ylim(-20, TOP + 5)
a2.set_ylabel("Change in L against baseline (%)")
a2.set_title("(b) No information, only feedback")
fig.tight_layout(w_pad=1.5)
save(fig, "fig4_resolution.pdf")

# fig. 6 (fig5_earlywarning.pdf): early warning (P4 and P7 share folds and seeds)
ew_live = P2["cv"]["early_warning"]["e_hurdle_eserving"]["pr_auc"]["mean"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw={"width_ratios": [1.3, 1]})
bars = [("Logit\n(reg.)", S["e_logit"]["pr_auc"], GREY, "////"),
        ("Latest\nrisk", lat_["pr_auc"], GREY, None),
        ("Live\nfeed", ps_["LIVE"]["pr_auc"], GREY, "...."),
        ("Public", ps_["PUB"]["pr_auc"], ORANGE, None),
        ("A1", S["A1_without_covariance"]["pr_auc"], BLUE, "////"),
        ("A2", S["A2_serving_plus_covariance"]["pr_auc"], BLUE, None),
        ("A0", S["A0_all_benchmark"]["pr_auc"], INK, None)]
a1.axhline(lat_["pr_auc"], color=GREY, lw=0.8, ls="--", zorder=0)
for i, (lab_, v, col, hatch) in enumerate(bars):
    a1.bar(i, v, color="white" if hatch else col, edgecolor=col, hatch=hatch, width=0.66, lw=1.0)
    a1.text(i, v + 0.003, f(v, 3), ha="center", va="bottom", fontsize=6.5)
a1.set_xticks(range(len(bars)))
a1.set_xticklabels([b_[0] for b_ in bars], fontsize=6.6)
a1.set_ylabel("Precision-recall area")
a1.set_ylim(0, 0.15)
a1.set_title(f"(a) Pooled CV, {ABL['n_escalations']} escalations")
lc_ = ew_["contrasts_vs_latest_risk"]
cons = [("A0 − A1 (P4 primary)", CT["A0_minus_A1"], BLUE),
        ("Public − logistic (P7 primary)", pc_["PUB_minus_logit"], ORANGE),
        ("A0 − latest risk", lc_["A0"], GREY), ("A1 − latest risk", lc_["A1"], GREY),
        ("A2 − latest risk", lc_["A2"], GREY), ("Public − latest risk", lc_["PUB"], GREY),
        ("Live feed − latest risk", lc_["LIVE"], GREY), ("Logistic − latest risk", lc_["e_logit"], GREY)]
for i, (lab_, cc, col) in enumerate(cons):
    y = len(cons) - 1 - i
    a2.plot(cc["ci95"], [y, y], color=col, lw=2, solid_capstyle="butt")
    a2.plot([cc["mean"]], [y], "o", color=INK if col == GREY else col, ms=4.2)
a2.axhline(len(cons) - 2.5, color="#d2d2d7", lw=0.6)
a2.axvline(0, color=INK, lw=0.8)
a2.set_yticks(range(len(cons)))
a2.set_yticklabels([c_[0] for c_ in cons][::-1], fontsize=6.6)
a2.set_xlabel("Difference in precision-recall area")
a2.set_title("(b) Contrasts, 95% CIs")
fig.tight_layout(w_pad=1.0)
save(fig, "fig5_earlywarning.pdf")
m("eLiveOnlyPR", f(ew_live, 3))

# fig. 5 (fig6_coherence.pdf): is the CDM risk a coherent forecast of itself? (P6, test events)
test_csv = ROOT / "data/kelvins/Collision Avoidance Challenge - Dataset/kelvins_competition_data/test_data.csv"
if test_csv.exists():
    import pandas as pd
    from backend.app.models.lab import data as D
    from backend.app.models.lab import round4 as R4

    tev = D.events_from_frame(pd.read_csv(test_csv), "test")
    d3 = np.concatenate([np.array(R4.update_d2(e.inputs)) for e in tev])          # as pre-registered
    dp = np.concatenate([np.array(R4.update_d2_plane(e.inputs)) for e in tev])    # post hoc, in the plane
    assert len(d3) == up_["pairs"] and len(dp) == upp["pairs"]
    assert abs(np.mean(dp > R4.CHI2_2_P99) - upp["share_above_chi2_2_p99"]) < 1e-12
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    bands = [("Below 10$^{-12}$", "band_-29.9_-12_any_later"), ("10$^{-12}$ to 10$^{-8}$", "band_-12_-8_any_later"),
             ("10$^{-8}$ to 10$^{-6}$", "band_-8_-6_any_later")]
    FLOOR_Y = 1e-5
    for i, (lab_, key) in enumerate(bands):
        b_ = lo_[key]
        obs, cm = b_["observed"], b_["coherent_max"]
        a1.bar(i - 0.2, max(cm, FLOOR_Y), width=0.34, color="white", edgecolor=GREY, hatch="////", lw=0.9)
        a1.bar(i + 0.2, obs, width=0.34, color=RED if obs > cm else BLUE, lw=0)
        if cm >= 10:
            txt = f"{cm:.0f}"
        elif cm >= 0.1:
            txt = f"{cm:.2f}"
        else:
            e_ = int(math.floor(math.log10(cm)))
            txt = f"{cm / 10 ** e_:.1f}×10$^{{{e_}}}$"
        a1.text(i - 0.23, max(cm, FLOOR_Y) * 1.6, txt, ha="center", va="bottom", fontsize=5.8, color="#6e6e73",
                zorder=6)
        a1.text(i + 0.21, obs * 1.6, str(obs), ha="center", va="bottom", fontsize=7, color=INK, zorder=6)
    a1.set_yscale("log")
    a1.set_ylim(FLOOR_Y, 3e5)
    a1.set_xticks(range(len(bands)))
    a1.set_xticklabels([b_[0] for b_ in bands], fontsize=6.8)
    a1.set_xlabel("Pc of the latest message before the cutoff")
    a1.set_ylabel("Events that later reach Pc $\\geq 10^{-6}$")
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(facecolor="white", edgecolor=GREY, hatch="////", label="most a coherent Pc allows"),
                       Patch(facecolor=RED, label="observed (exceeds it)"), Patch(facecolor=BLUE, label="observed (within it)")],
              frameon=False, loc="upper left", fontsize=6.2)
    a1.set_title("(a) Low probabilities that later turn high", fontsize=8)
    xs = np.logspace(-3, 4, 300)
    from scipy.stats import chi2

    def survival(v):
        s = np.sort(v)
        y = 1.0 - np.arange(1, len(s) + 1) / len(s)
        return s[y > 0], y[y > 0]

    a2.loglog(xs, chi2.sf(xs, 2), color=INK, lw=1.1, ls="--", label="coherent bound ($\\chi^2_2$)")
    a2.loglog(*survival(d3), color=GREY, lw=1.0, label="registered (3-D)")
    a2.loglog(*survival(dp), color=RED, lw=1.5, label="encounter plane")
    a2.axvline(R4.CHI2_2_P99, color=INK, lw=0.6, ls=":")
    a2.annotate(f"{f(100 * upp['share_above_chi2_2_p99'], 1)}% of updates lie\nbeyond the 99% line\n(coherent: at most 1%)",
                (R4.CHI2_2_P99, upp["share_above_chi2_2_p99"]), xytext=(2e-3, 1.2e-2), fontsize=6.3, color=RED,
                va="center", arrowprops=dict(arrowstyle="-", color=RED, lw=0.6))
    a2.set_ylim(1e-4, 1.3)
    a2.set_xlim(1e-3, 1e4)
    a2.set_xlabel("Update size $d^2$ (earlier message's covariance)")
    a2.set_ylabel("Share of updates larger")
    a2.set_title(f"(b) {n(up_['pairs']).replace('{,}', ',')} message-to-message updates", fontsize=8)
    a2.legend(frameon=False, loc="lower left", fontsize=6.2, handlelength=1.8)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig6_coherence.pdf")
else:
    print("test_data.csv not found: fig6_coherence.pdf skipped")

# Graphical table of contents for Astrodynamics: at least 80 mm x 50 mm, height/width
# between 5/8 and 1, 600 dpi. Saved at its exact size (no tight bbox).
fig, (g1, g2) = plt.subplots(2, 1, figsize=(90 / 25.4, 60 / 25.4),
                             gridspec_kw={"height_ratios": [3, 2.1]})
gt = []
for key, rep, label in (("hurdle_lrp", R1, "VIGIL, benchmark (R1)"),
                        ("hurdle_lrp_serving", R1, "VIGIL, live feed (R1)"),
                        ("hurdle_lrp_serving_bench", R2, "VIGIL, test mix (R2)")):
    dd = sealed(rep, key)["delta_L_pct_vs_lrp"]
    gt.append((label, dd["mean"], dd["ci975"] if key.endswith("_bench") else dd["ci95"]))
    assert gt[-1][2][0] < 0 < gt[-1][2][1]     # the picture's claim: every interval spans zero
ys = np.arange(len(gt))[::-1]
for y, (label, x, (lo, hi)) in zip(ys, gt):
    g1.plot([lo, hi], [y, y], color=BLUE, lw=1.8, solid_capstyle="butt")
    g1.plot([x], [y], "o", color=BLUE, ms=4)
g1.axvline(0, color=INK, lw=0.8)
g1.set_yticks(ys)
g1.set_yticklabels([g_[0] for g_ in gt], fontsize=6.3)
g1.set_xlim(-16, 8)
g1.set_ylim(-0.6, len(gt) - 0.4)
g1.tick_params(axis="x", labelsize=6)
g1.set_xlabel("Change in the official loss vs. the latest risk (%)", fontsize=6.3, labelpad=1.5)
# thresholds set inside the training folds (post hoc), the rate an operator could expect
fa_v = {"mean": tot_["A0"]["fa_removed"], "ci95": fa3_["fa_removed"]["A0"]["ci95"]}
fa_l = {"mean": tot_["latest"]["fa_removed"], "ci95": fa3_["fa_removed"]["latest"]["ci95"]}
for y, lab, d_, col in ((1, "VIGIL", fa_v, BLUE), (0, "Latest risk alone", fa_l, GREY)):
    v, (lo, hi) = 100 * d_["mean"], [100 * c_ for c_ in d_["ci95"]]
    g2.barh(y, v, height=0.56, color=col)
    g2.plot([lo, hi], [y, y], color=INK, lw=0.8)
    g2.text(hi + 1.2, y, f"{f(v, 0)}%", va="center", fontsize=6.5, color=col)
g2.set_yticks([1, 0])
g2.set_yticklabels(["VIGIL", "Latest risk alone"], fontsize=6.3)
g2.set_xlim(0, 45)
g2.set_ylim(-0.6, 1.6)
g2.tick_params(axis="x", labelsize=6)
g2.set_xlabel("False alarms filtered, keeping 95% of true calls (%)", fontsize=6.3, labelpad=1.5)
for ax_, title in ((g1, "Sealed test: no model beats the latest risk with confidence"),
                   (g2, "Cross-validation: machine learning filters false alarms")):
    # anchored at the figure's left edge: a title over the axes alone would run off the right
    ax_.annotate(title, xy=(0, 1), xycoords=("figure fraction", "axes fraction"), xytext=(3, 5),
                 textcoords="offset points", ha="left", va="bottom", fontsize=6.8, color=INK)
fig.text(0.99, 0.005, "Bars: 95% confidence intervals (R2: 97.5%)", ha="right", va="bottom",
         fontsize=5.2, color="#6e6e73")
fig.tight_layout(h_pad=0.9, rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "gtoc.pdf")
fig.savefig(FIG / "gtoc.png", dpi=600)
plt.close(fig)

# fig. 1 (fig1_events.pdf): two real training events (needs the Kelvins file)
train_csv = ROOT / "data/kelvins/train_data.csv"
if train_csv.exists():
    from backend.app.models.lab import data as D

    evs = [e for e in D.load(str(train_csv))[0] if e.test_eligible]
    esc = sorted([e for e in evs if e.escalates], key=lambda e: (-e.n_cdm_total, e.event_id))[0]
    fade = sorted([e for e in evs if e.latest_risk >= -6 and e.target < -6],
                  key=lambda e: (-e.n_cdm_total, e.event_id))[0]
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.3), sharey=True)
    for ax, ev, title in ((axs[0], esc, "(a) Quiet at the cutoff, high-risk at the end"),
                          (axs[1], fade, "(b) High-risk at the cutoff, harmless at the end")):
        rows_ = ev.inputs + ev.late
        t = np.array([float(r["time_to_tca"]) for r in rows_])
        rk = np.array([float(r["risk"]) for r in rows_])
        floor = rk <= -12
        # the final CDM sits near TCA, so leave room past 0 for its star
        # (a marker centred on the axis edge is cut in half)
        ax.axvspan(2.0, -0.45, color="#f2f2f5", lw=0)
        ax.axhline(-6, color=RED, lw=0.8, ls="--")
        ax.plot(t[~floor], rk[~floor], "-", color=GREY, lw=0.8)
        # every CDM but the final one gets a dot (or a floor triangle); the
        # final CDM is drawn once, as the star, never also under it
        body = np.arange(len(t)) < len(t) - 1
        ax.plot(t[~floor & body], rk[~floor & body], "o", color=INK, ms=3)
        ax.plot(t[floor & body], np.full((floor & body).sum(), -12.6), "v", color=INK, ms=4)
        k_in = len(ev.inputs) - 1
        ax.plot([t[k_in]], [max(rk[k_in], -12.6)], "o", ms=7, mfc="none", mec=BLUE, mew=1.5)
        ax.plot([t[-1]], [max(rk[-1], -12.6)], "*", ms=10, color=ORANGE, clip_on=False, zorder=5)
        ax.text(0.8, -2.95, "hidden from\nthe model", fontsize=6.5, color=GREY, va="top", ha="center")
        ax.set_xlim(7.2, -0.45)
        ax.set_ylim(-13.2, -2.8)
        ax.set_title(title)
        ax.set_xlabel("Days before closest approach")
    axs[0].set_ylabel("Collision risk, log$_{10}$ Pc")
    axs[0].set_yticks([-12, -10, -8, -6, -4])
    axs[0].set_yticklabels(["≤−12", "−10", "−8", "−6", "−4"])
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig1_events.pdf")
    m("figOneEsc", esc.event_id.split(":")[1])
    m("figOneFade", fade.event_id.split(":")[1])
    (OUT / "numbers.tex").write_text(
        "% generated by scripts/paper_assets.py -- do not edit\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in MACROS))
else:
    print("data/kelvins/train_data.csv not found: fig1_events.pdf skipped")

print(f"wrote {len(MACROS)} numbers, 4 tables and the figures")
