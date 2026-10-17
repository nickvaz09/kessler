"""VIGIL Lab Round 3 (backend/app/models/lab/paper.py), on synthetic data."""
import math

import numpy as np

from backend.app.models.lab import compete, data, metric, paper
from backend.app.models.lab.data import HIGH_RISK
from backend.tests.lab_synth import frame


def _events(n=120, seed=3, start_id=0):
    return data.events_from_frame(frame(n, seed=seed, start_id=start_id))


def test_at_cutoff_adds_late_cdms_but_never_the_final_one():
    for ev in _events(60):
        same = paper.at_cutoff(ev, 2.0)
        assert [r["time_to_tca"] for r in same.inputs] == [r["time_to_tca"] for r in ev.inputs]
        for c in (1.75, 1.0):
            seen = paper.at_cutoff(ev, c)
            ttas = [float(r["time_to_tca"]) for r in seen.inputs]
            assert all(t >= c for t in ttas)
            assert all(t > ev.final_tta for t in ttas)       # the target stays hidden
            assert ttas == sorted(ttas, reverse=True)       # still in message order
            assert seen.target == ev.target and seen.test_eligible == ev.test_eligible
            assert len(seen.inputs) >= len(ev.inputs)


def test_covariance_rule_matches_the_preregistered_names():
    cov = ["l_t_sigma_r", "l_c_sigma_tdot", "l_t_ct_r", "l_c_cndot_tdot", "l_t_crdot_n",
           "l_mahalanobis_distance", "l_max_risk_estimate", "l_max_risk_scaling",
           "l_t_position_covariance_det_slog10", "p_m2", "p_proj_risk_trend",
           "h_log_sigma_rate", "h_maha_slope", "h_headroom_mre", "h_sigma_shrink_to_eval"]
    not_cov = ["l_risk", "l_miss_distance", "l_relative_speed", "l_t_cd_area_over_mass",
               "l_c_cr_area_over_mass", "l_c_obs_used", "l_t_span", "h_risk_slope",
               "h_miss_last", "s_pc_log_last", "l_ctype_payload", "l_F10", "l_AP"]
    assert all(paper.is_covariance_feature(c) for c in cov)
    assert not any(paper.is_covariance_feature(c) for c in not_cov)
    sets = paper.feature_sets(cov + not_cov)
    assert set(sets["A1_without_covariance"]) == set(not_cov)
    assert set(sets["A2_serving_plus_covariance"]) == set(cov) | {"s_pc_log_last"}
    assert sets["A0_all_benchmark"] == cov + not_cov


def test_stratified_mask_keeps_each_class_share():
    hr = np.array([True] * 50 + [False] * 450)
    m = paper.stratified_mask(hr, 0.9, np.random.RandomState(0))
    assert m[hr].sum() == 45 and m[~hr].sum() == 405


def test_climb_only_ever_improves_the_visible_score():
    rng = np.random.RandomState(1)
    n = 400
    latest = rng.uniform(-12, -3, n)
    y = np.where(rng.rand(n) < 0.3, latest + rng.normal(0, 0.5, n), -30.0)
    vis = paper.stratified_mask(y >= HIGH_RISK, 0.9, rng)
    none = paper.climb(latest, y, vis, np.random.RandomState(2), k=10, budget=0)
    assert none["accepted"] == 0 and none["full_dl_pct"] == 0.0
    run = paper.climb(latest, y, vis, np.random.RandomState(2), k=10, budget=60)
    assert run["visible_dl_pct"] <= 0.0
    again = paper.climb(latest, y, vis, np.random.RandomState(2), k=10, budget=60)
    assert again == run                                     # seeded, so reproducible


def test_side_reads_what_the_published_cascade_needs():
    evs = _events(40)
    s = paper.side(evs)
    assert len(s["latest"]) == len(evs) == len(s["payload"])
    assert s["payload"].dtype == bool
    pred = compete.sesc_cascade(s)
    assert np.all((pred >= HIGH_RISK) | (pred < HIGH_RISK))  # defined for every event
    assert np.array_equal(metric.lrp(s["latest"]) >= HIGH_RISK,
                          np.array([e.latest_risk >= HIGH_RISK for e in evs]))


def test_nb_correction_shrinks_the_published_t_statistic():
    r = paper.nb_correction()
    assert math.isclose(r["factor"], math.sqrt((1 / 505) / (1 / 505 + 0.25)), rel_tol=1e-12)
    assert abs(r["t_corrected"]) < 1.0 < abs(r["t_naive"])
    assert r["p_corrected"] > 0.05 > r["p_naive"]
    lo, hi = r["f2_gain_ci95_corrected"]
    assert lo < 0 < hi


def test_horizon_runs_and_uses_the_same_events_at_every_cutoff():
    res = paper.horizon(_events(500, seed=11), repeats=1, folds=3, bootstrap=20,
                        cutoffs=(2.0, 1.0), log=lambda *_: None)
    assert set(res["rows"]) == {"2.0", "1.0"}
    assert res["rows"]["2.0"]["lrp_L_ratio_to_2d"] == 1.0
    for row in res["rows"].values():
        assert row["lrp"]["L"] > 0 and len(row["delta_L_pct_ci95"]) == 2
    assert res["rows"]["1.0"]["mean_inputs_per_event"] >= res["rows"]["2.0"]["mean_inputs_per_event"]


def test_resolution_and_attack_run_on_synthetic_data():
    train, test = _events(300, seed=21), _events(200, seed=22, start_id=10_000)
    r = paper.resolution(train, test, bootstrap=30, log=lambda *_: None)
    assert r["full_test"]["n"] == len(test)
    assert r["c_visible_subsets"]["lrp_visible_vs_full_pct"]["n"] == 30
    a = paper.attack(test, runs=5, budget=10, ks=(5,), log=lambda *_: None)
    assert a["by_k"]["5"]["visible_dl_pct"]["p975"] <= 0.0


def test_ablation_runs_and_compares_the_three_feature_sets():
    res = paper.ablation(_events(600, seed=31), repeats=1, folds=3, bootstrap=20,
                         log=lambda *_: None)
    assert set(res["summary"]) == {"A0_all_benchmark", "A1_without_covariance",
                                   "A2_serving_plus_covariance", "e_logit"}
    counts = res["feature_counts"]
    assert counts["A1_without_covariance"] + counts["_covariance"] == counts["A0_all_benchmark"]
    assert set(res["contrasts"]) >= {"A0_minus_A1", "A1_minus_logit", "A2_minus_logit"}
    assert res["reading"] in {"covariance carries the signal", "inconclusive", "other"}
