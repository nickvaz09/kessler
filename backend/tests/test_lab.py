"""VIGIL Lab: protocol fidelity, leakage, metric, physics, end-to-end."""
import json
import math

import numpy as np
import pytest

from backend.app.models.lab import data, features, gate, metric, physics, protocol
from backend.tests.lab_synth import frame


# --- official protocol ----------------------------------------------------------

def _rows(eid, spec):
    return [{"event_id": eid, "time_to_tca": t, "risk": r, "miss_distance": 500.0,
             "relative_speed": 10000.0} for t, r in spec]


def test_inputs_respect_two_day_cutoff_and_target_is_final():
    import pandas as pd

    df = pd.DataFrame(_rows(1, [(5.0, -9.0), (3.0, -7.5), (2.0, -7.0), (1.5, -5.0), (0.4, -4.2)]))
    (ev,) = data.events_from_frame(df)
    assert [r["time_to_tca"] for r in ev.inputs] == [5.0, 3.0, 2.0]  # >= 2 days only
    assert ev.target == -4.2 and ev.latest_risk == -7.0
    assert ev.test_eligible and ev.high_risk and ev.escalates


def test_official_test_filters():
    import pandas as pd

    df = pd.DataFrame(
        _rows(1, [(3.0, -8), (0.5, -8)])            # eligible
        + _rows(2, [(3.0, -8), (1.5, -8)])          # final not within 1 day
        + _rows(3, [(1.9, -8), (0.5, -8)])          # no input CDM at >= 2 days -> dropped
        + _rows(4, [(2.5, -8)]))                    # single CDM
    evs = {e.event_id: e for e in data.events_from_frame(df)}
    assert evs["train:1"].test_eligible
    assert not evs["train:2"].test_eligible
    assert "train:3" not in evs
    assert not evs["train:4"].test_eligible


def test_attach_test_targets_with_risk_only_file():
    import pandas as pd

    inputs = pd.DataFrame(_rows(7, [(4.0, -9.0), (2.2, -8.0)]))
    targets = pd.DataFrame([{"event_id": 7, "true_risk": -5.5}])
    (ev,) = data.events_from_frame(data.attach_test_targets(inputs, targets), "test")
    assert ev.target == -5.5 and ev.final_tta == 0.0 and ev.test_eligible
    assert len(ev.inputs) == 2


# --- metric ---------------------------------------------------------------------

def test_official_score_by_hand():
    truth = np.array([-5.0, -4.0, -7.0, -8.0])
    pred = np.array([-5.5, -6.001, -5.0, -9.0])
    s = metric.official(pred, truth)
    # TP=1 (event 0), FN=1 (event 1), FP=1 (event 2) -> F2 = 5*1/(5*1+4*1+1) = 0.5
    assert s["f2"] == pytest.approx(0.5)
    assert s["mse_hr"] == pytest.approx(((-5.5 + 5) ** 2 + (-6.001 + 4) ** 2) / 2)
    assert s["L"] == pytest.approx(s["mse_hr"] / 0.5)


def test_lrp_and_clip_low_follow_the_organisers_convention():
    lat = np.array([-4.0, -6.0, -6.5, -30.0])
    assert list(metric.lrp(lat)) == [-4.0, -6.0, -6.001, -6.001]
    assert list(metric.clip_low(np.array([-7.0, -5.0]))) == [-6.001, -5.0]


def test_average_precision_and_recall_at_budget():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], bool)
    s = np.arange(10, 0, -1).astype(float)          # perfect order for event 0, 2nd pos at rank 3
    assert metric.average_precision(y, s) == pytest.approx((1 + 2 / 3) / 2)
    assert metric.recall_at_budget(y, s, 0.1) == pytest.approx(0.5)


# --- physics --------------------------------------------------------------------

def _geom_row(miss=800.0, st=300.0, sc=900.0, risk=-6.5):
    return {"relative_position_r": 0.0, "relative_position_t": 0.0, "relative_position_n": miss,
            "relative_velocity_r": 0.0, "relative_velocity_t": 10000.0, "relative_velocity_n": 0.0,
            "t_sigma_r": 0.2 * st, "t_sigma_t": st, "t_sigma_n": 0.3 * st,
            "c_sigma_r": 0.2 * sc, "c_sigma_t": sc, "c_sigma_n": 0.3 * sc, "risk": risk}


def test_dilution_index_sign_and_pcmax_gap():
    tight = physics.encounter_plane(_geom_row(miss=5000.0, st=50, sc=100))
    loose = physics.encounter_plane(_geom_row(miss=100.0, st=3000, sc=9000))
    assert tight["dilution_index"] > 0          # robust region
    assert loose["dilution_index"] < 0          # dilution region
    assert loose["log10_pcmax_gap"] > 0 and tight["log10_pcmax_gap"] >= 0


def test_projected_risk_matches_recomputed_pc_after_shrink():
    """Scaling sigma by s must change log10 Pc exactly as the closed form says."""
    from backend.tests.lab_synth import _risk

    miss, st, sc, s = 700.0, 400.0, 1200.0, 0.5
    v = np.array([0.0, 10000.0, 0.0]); mv = np.array([0.0, 0.0, miss])
    cov = lambda k: np.diag([(0.2 * st * k) ** 2 + (0.2 * sc * k) ** 2, (st * k) ** 2 + (sc * k) ** 2,
                             (0.3 * st * k) ** 2 + (0.3 * sc * k) ** 2])
    r1, _, _ = _risk(mv, v, cov(1.0))
    r2, _, _ = _risk(mv, v, cov(s))
    ph = physics.encounter_plane(_geom_row(miss, st, sc, r1))
    assert physics.projected_risk(r1, ph["m2"], s) == pytest.approx(r2, abs=1e-6)


# --- leakage ---------------------------------------------------------------------

def test_features_ignore_cdms_inside_the_cutoff():
    df = frame(40, seed=5)
    base = features.table(data.events_from_frame(df), "benchmark")
    late = df["time_to_tca"] < data.INPUT_CUTOFF_DAYS
    df2 = df.copy()
    for col in ("risk", "miss_distance", "t_sigma_t", "c_sigma_t", "mahalanobis_distance"):
        df2.loc[late, col] = df2.loc[late, col] * 0.5 - 3.0
    pert = features.table(data.events_from_frame(df2), "benchmark")
    pert = pert.reindex(columns=base.columns)
    assert np.allclose(base.to_numpy(), pert.to_numpy(), equal_nan=True)


def test_serving_track_is_the_production_feature_vector():
    from backend.app.models.features import FEATURE_NAMES

    evs = data.events_from_frame(frame(5, seed=2))
    t = features.table(evs, "serving")
    assert list(t.columns) == [f"s_{n}" for n in FEATURE_NAMES]


# --- end to end -------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_run():
    evs = data.events_from_frame(frame(900, seed=11))
    cv = protocol.cross_validate(evs, repeats=1, folds=3, bootstrap=40, log=lambda *a: None)
    return evs, cv


def test_cv_summary_shapes_and_baseline(small_run):
    evs, cv = small_run
    assert cv["n_events"] == len(evs) and cv["n_eligible"] <= len(evs)
    assert cv["n_trainable"] <= cv["n_events"]
    assert cv["n_escalations"] <= cv["n_escalations_all"]      # scored on eligible only
    lrp = cv["regression"]["lrp"]
    assert lrp["delta_L_pct_vs_lrp"]["mean"] == 0.0
    for k, r in cv["regression"].items():
        assert not np.isnan(r["L"]["mean"]), k   # inf is legitimate (F2 = 0)
        assert r["track"] in ("benchmark", "last", "serving")
        if k != "lrp" and np.isfinite(r["L"]["mean"]):
            d = r["delta_L_pct_vs_lrp"]
            assert d["mde"] is not None and d["mde"] >= 0
    assert {"e_logit", "e_hurdle", "e_serving"} <= set(cv["early_warning"])
    assert cv["regression"]["hurdle_lrp_serving_bench"]["primary"]       # Round 2
    assert not cv["regression"]["hurdle_lrp"]["primary"] and not cv["regression"]["sesc"]["primary"]
    assert cv["early_warning"]["e_hurdle_eserving"]["primary"]
    assert "bench_mix" in cv["regression"]["lrp"]
    assert cv["regression"]["lrp"]["bench_mix"]["delta_L_pct"]["mean"] == 0.0


def test_gate_tests_only_the_preregistered_primaries(small_run):
    from backend.app.models.lab.candidates import PRIMARY

    _, cv = small_run
    g = gate.vigil(cv, None)
    assert set(g) == {"serving", "serving_exportable"}            # round 2: one VIGIL primary
    assert g["serving"]["candidate"] == PRIMARY["vigil_serving"]
    assert g["serving"]["status"] in ("passed", "not_passed", "underpowered")
    # the synthetic set has < 100 high-risk eligible events -> no claim either way
    if cv["n_hr_eligible"] < gate.MIN_HR_EVENTS:
        assert g["serving"]["status"] == "underpowered" and not g["serving"]["passed"]
    ge = gate.vigil_e(cv, None)
    assert set(ge) == {"serving"}
    assert ge["serving"]["candidate"] == PRIMARY["vigil_e_serving"]
    assert ge["serving"]["min_detectable_delta_pr_auc"] is not None
    assert any("97.5% CI" in c["detail"] for c in g["serving"]["checks"])


def _fake_sealed(cv, n_hr, dl_mean, dl_ci):
    import copy

    from backend.app.models.lab.candidates import PRIMARY

    t = {"n_hr": n_hr, "n_escalations": 35, "regression": copy.deepcopy(cv["regression"]),
         "early_warning": copy.deepcopy(cv["early_warning"])}
    t["regression"][PRIMARY["vigil_serving"]]["delta_L_pct_vs_lrp"].update(
        {"mean": dl_mean, "ci95": dl_ci, gate.CI_KEY: dl_ci, "sd_boot": 1.0})
    return t


def test_amended_gate_is_judged_on_the_sealed_test(small_run):
    from backend.app.models.lab.candidates import PRIMARY

    _, cv = small_run
    cv = {**cv, "n_hr_eligible": 66}                        # CV alone: underpowered
    g = gate.vigil(cv, _fake_sealed(cv, 150, -10.0, [-15.0, -5.0]))["serving"]
    assert g["candidate"] == PRIMARY["vigil_serving"] and g["basis"] == "sealed_test"
    assert g["status"] == "passed" and g["passed"] and g["confirmed_on_sealed_test"]
    assert g["min_detectable_delta_pct"] == pytest.approx(gate.CI_Z + 0.8416)   # 97.5% MDE
    assert {c["name"] for c in g["supporting"]} == {"cv_mean_gain", "cv_ci_below_zero"}
    # CV evidence is reported but never blocks
    assert all(c["name"].startswith(("powered", "sealed_")) for c in g["checks"])


def test_amended_gate_requires_power_and_a_clear_sealed_gain(small_run):
    _, cv = small_run
    few = gate.vigil(cv, _fake_sealed(cv, 60, -10.0, [-15.0, -5.0]))["serving"]
    assert few["status"] == "underpowered" and not few["passed"]
    crosses = gate.vigil(cv, _fake_sealed(cv, 150, -10.0, [-18.0, 1.0]))["serving"]
    assert crosses["status"] == "not_passed" and not crosses["confirmed_on_sealed_test"]
    small = gate.vigil(cv, _fake_sealed(cv, 150, -1.0, [-1.8, -0.2]))["serving"]
    assert small["status"] == "not_passed"                  # gain below MIN_GAIN_PCT
    none = gate.vigil(cv, None)["serving"]
    assert none["basis"] == "cv" and none["confirmed_on_sealed_test"] is None
    assert gate.vigil_e(cv, None)["serving"]["basis"] == "cv"


# --- Round 2 ------------------------------------------------------------------------

def test_weighted_official_score():
    truth = np.array([-5.0, -4.0, -7.0, -8.0])
    pred = np.array([-5.5, -6.001, -5.0, -9.0])
    plain = metric.official(pred, truth)
    assert metric.official(pred, truth, weights=np.ones(4)) == pytest.approx(plain)
    # high-risk events x2: TP=2, FN=2, FP=1 -> F2 = 10/(10+8+1); MSE_HR unchanged
    w = np.array([2.0, 2.0, 1.0, 1.0])
    s = metric.official(pred, truth, weights=w)
    assert s["f2"] == pytest.approx(10 / 19) and s["mse_hr"] == pytest.approx(plain["mse_hr"])


def test_label_shift_weights_reach_the_target_mix():
    hr = np.array([True] * 10 + [False] * 990)            # 1% high-risk
    w = metric.label_shift_weights(hr, 150 / 2167)
    assert w[hr].sum() / w.sum() == pytest.approx(150 / 2167)
    assert w.sum() == pytest.approx(len(hr))


def test_benchmark_prior_maps_the_fallback_decision_through_the_odds():
    from backend.app.models.lab import candidates as C

    h = C.Hurdle("hurdle_lrp_serving", mode="lrp", track="serving")
    y = np.array([-5.0] * 4 + [-20.0] * 396)               # 1% high-risk, too few to tune
    h._tune = {"p": np.zeros(400), "m": np.full(400, -5.5), "latest": np.full(400, -9.0), "y": y}
    h.tuned, h.tau, h.alpha = False, C.FALLBACK_TAU, C.FALLBACK_ALPHA
    b = h.with_prior("hurdle_lrp_serving_bench", 150 / 2167)
    k = ((150 / 2167) / (1 - 150 / 2167)) / (0.01 / 0.99)
    odds = (C.FALLBACK_TAU / (1 - C.FALLBACK_TAU)) / k
    assert b.tau == pytest.approx(odds / (1 + odds)) and b.tau < h.tau
    assert b.name == "hurdle_lrp_serving_bench" and h.name == "hurdle_lrp_serving"
    assert h.tau == C.FALLBACK_TAU                          # the base model is untouched


def test_near_miss_labels_come_from_cdms_inside_the_cutoff():
    import pandas as pd

    df = pd.DataFrame(_rows(1, [(5.0, -9.0), (2.5, -8.0), (1.5, -5.5), (0.5, -8.0)])
                      + _rows(2, [(4.0, -9.0), (0.4, -7.0)]))
    b = protocol.Bundle(data.events_from_frame(df))
    assert list(b.y) == [-8.0, -7.0]
    assert list(b.y_aux) == [-5.5, -7.0]                   # crossed -6 at 1.5 d, then fell back
    assert (b.y_aux >= b.y).all()


def test_round2_primaries_are_registered_and_scored(small_run):
    from backend.app.models.lab import candidates as C

    _, cv = small_run
    assert C.PRIMARY["vigil_serving"] in cv["regression"]
    assert C.PRIMARY["vigil_e_serving"] in cv["early_warning"]
    assert C.DERIVED["hurdle_lrp_serving_bench"][0] in protocol.REG_NAMES
    assert cv["regression"]["hurdle_lrp_serving_bench"]["track"] == "serving"
    assert cv["early_warning"]["e_hurdle_eserving"]["track"] == "serving"
    assert "cal_slope" in cv["early_warning"]["e_hurdle_eserving"]


def test_go_no_go_keeps_the_sealed_test_unscored(tmp_path, monkeypatch, small_run):
    import copy

    from backend.app.models.lab import __main__ as cli

    _, cv = small_run
    bad = copy.deepcopy(cv)
    bad["regression"]["hurdle_lrp_serving_bench"]["bench_mix"]["delta_L_pct"]["mean"] = 3.0
    monkeypatch.setattr(protocol, "cross_validate", lambda *a, **k: bad)

    def never(*a, **k):
        raise AssertionError("the sealed test must not be scored after a no-go")
    monkeypatch.setattr(protocol, "sealed_test", never)
    train, test = frame(40, seed=5), frame(20, seed=6, start_id=5_000)
    finals = test.sort_values("time_to_tca").groupby("event_id", as_index=False).first()
    train.to_csv(tmp_path / "train.csv", index=False)
    test[test["time_to_tca"] >= data.INPUT_CUTOFF_DAYS].to_csv(tmp_path / "test.csv", index=False)
    finals[["event_id", "time_to_tca", "risk"]].to_csv(tmp_path / "targets.csv", index=False)
    out = tmp_path / "out"
    assert cli.main(["--train", str(tmp_path / "train.csv"), "--test", str(tmp_path / "test.csv"),
                     "--test-targets", str(tmp_path / "targets.csv"), "--final", "--require-cv-go",
                     "--no-ceiling", "--out", str(out)]) == 0
    run = json.loads((out / "lab_report.json").read_text())
    assert run["test"] is None and run["mode"]["cv_go"]["go"] is False
    assert "go/no-go said no" in (out / "lab_report.md").read_text()


# --- competition mode (2019 rules) -------------------------------------------------

def _side(n=400, seed=0):
    rng = np.random.RandomState(seed)
    latest = np.where(rng.rand(n) < 0.3, rng.uniform(-6, -3, n), rng.uniform(-12, -6.01, n))
    y = np.where(rng.rand(n) < 0.5, latest + rng.normal(0, 0.5, n), -30.0)
    return {"latest": latest, "y": y, "p_b": rng.rand(n), "p_s": rng.rand(n),
            "m_b": np.full(n, -5.5), "m_s": np.full(n, -5.5), "payload": rng.rand(n) < 0.1,
            "t_span": rng.uniform(0, 10, n), "miss": rng.uniform(0, 50000, n)}


def test_competition_baseline_config_is_exactly_lrp():
    from backend.app.models.lab import compete

    s = _side()
    assert np.array_equal(compete.decide(compete.LRP_CFG, s), metric.lrp(s["latest"]))
    assert all(compete.LRP_CFG[k] in v for k, v in compete.GRID.items())   # in the family


def test_competition_split_is_fixed_and_stratified():
    from backend.app.models.lab import compete

    evs = data.events_from_frame(frame(300, seed=8))
    a, b = compete.make_split(evs), compete.make_split(evs)
    assert np.array_equal(a, b)
    hr = np.array([e.high_risk for e in evs])
    assert abs(int(a[hr].sum()) - int((~a)[hr].sum())) <= 1


def test_competition_ledger_enforces_the_budget(tmp_path):
    from backend.app.models.lab import compete

    led = compete.Ledger(tmp_path / "ledger.jsonl", _side(), budget=2)
    led.submit(compete.LRP_CFG, "a")
    led.submit(compete.LRP_CFG, "a")                      # a repeat costs nothing
    led.submit({**compete.LRP_CFG, "tau": 0.3}, "b")
    with pytest.raises(RuntimeError, match="budget"):
        led.submit({**compete.LRP_CFG, "tau": 0.1}, "c")
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 2


def test_competition_placement_on_the_2019_leaderboard():
    from backend.app.models.lab import compete

    assert compete.placement(0.661) == 10 and compete.placement(0.55) == 1
    assert compete.placement(0.70) == 13


def test_near_duplicate_cdm_times_do_not_crash_features():
    import pandas as pd

    rows = frame(1, seed=4)
    first = rows.iloc[[0]].copy()
    dup = first.copy(); dup["time_to_tca"] = first["time_to_tca"] - 1e-5
    dup["t_sigma_t"] = first["t_sigma_t"] * 1.05; dup["c_sigma_t"] = first["c_sigma_t"] * 1.05
    df = pd.concat([first, dup, rows.iloc[1:]], ignore_index=True)
    feats = features.table(data.events_from_frame(df), "benchmark")
    assert np.isfinite(feats["h_sigma_shrink_to_eval"]).all()


def test_ranking_metrics_are_tie_invariant():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0], bool)
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    perm = np.array([0, 4, 3, 2, 1, 9, 8, 7, 6, 5])
    assert metric.average_precision(y, s) == pytest.approx(metric.average_precision(y[perm], s[perm]))
    assert metric.recall_at_budget(y, s, 0.3) == pytest.approx(metric.recall_at_budget(y[perm], s[perm], 0.3))
    # top-3 = the 0.9 event + 2 of 4 tied events holding 1 positive -> 1 + 0.5 of 2
    assert metric.recall_at_budget(y, s, 0.3) == pytest.approx(0.75)


def test_covariance_determinants_fit_in_float32_and_keep_their_order():
    import pandas as pd

    # the real Kelvins files reach 6.7e46 here, beyond float32 (3.4e38)
    dets = [6.73e46, 1.2e39, 655189.3, 10.4, 0.0, -8.2e18]
    df = pd.DataFrame([{"event_id": i, "time_to_tca": 3.0, "risk": -9.0,
                        "t_position_covariance_det": d} for i, d in enumerate(dets)])
    col = features.table(data.events_from_frame(df), "last")["l_t_position_covariance_det_slog10"]
    assert (np.abs(col) < np.finfo(np.float32).max).all()
    assert list(np.argsort(col.to_numpy())) == list(np.argsort(dets))   # monotone
    assert "l_t_position_covariance_det" not in features.table(data.events_from_frame(df), "last")


def test_bundle_fails_fast_on_float32_overflow(monkeypatch):
    evs = data.events_from_frame(frame(20, seed=3))
    real = features.table

    def table(events, track="benchmark"):
        df = real(events, track)
        df.iloc[0, 0] = 1e45
        return df
    monkeypatch.setattr(features, "table", table)
    with pytest.raises(ValueError, match="float32"):
        protocol.Bundle(evs)


def test_exported_booster_is_the_evaluated_model(small_run):
    import xgboost as xgb

    from backend.app.models.lab import candidates as C, export

    evs, _ = small_run
    b = protocol.Bundle(evs)
    tr = np.flatnonzero(b.trainable)
    cand = C.EClassifier("e_serving", track="eserving")
    cand.fit(protocol._ctx(b, tr, "eserving", 0))
    X = b.X["eserving"][tr[:25]]
    served = export._evaluated_booster(cand.model).predict(xgb.DMatrix(X), output_margin=True)
    assert np.allclose(served, cand.margin(X), atol=1e-6)


def test_export_gate_checks_carry_the_model_card_field():
    from backend.app.models.lab import export

    g = export._gate_for_card({"passed": True, "checks": [{"name": "x", "ok": True, "detail": ""},
                                                           {"name": "y", "ok": False, "detail": ""}]})
    assert [c["passed"] for c in g["checks"]] == [True, False]


def test_sealed_targets_require_final_flag():
    from backend.app.models.lab import __main__ as cli

    with pytest.raises(SystemExit):
        cli.main(["--train", "t.csv", "--test", "x.csv", "--test-targets", "y.csv"])


def test_a_sealed_report_is_never_overwritten_by_accident(tmp_path):
    from backend.app.models.lab import __main__ as cli

    (tmp_path / "lab_report.json").write_text(json.dumps({"test": {"n_events": 2167}}))
    with pytest.raises(SystemExit):
        cli.main(["--train", "t.csv", "--out", str(tmp_path)])


def test_report_says_when_there_is_no_sealed_test(small_run):
    from backend.app.models.lab import report

    _, cv = small_run
    run = {"cv": cv, "test": None, "ceiling": None,
           "mode": {"pooled": True, "n_pooled_test_events": 250, "targets_supplied": True},
           "gate_vigil": gate.vigil(cv, None), "gate_vigil_e": gate.vigil_e(cv, None)}
    md = report.markdown(run)
    assert "NO SEALED TEST" in md and "pooled" in md and "CV pool" in md


def test_missing_test_targets_are_an_error():
    import pandas as pd

    inputs = pd.DataFrame(_rows(7, [(4.0, -9.0)]) + _rows(8, [(3.0, -9.0)]))
    targets = pd.DataFrame([{"event_id": 7, "risk": -5.5}])
    with pytest.raises(ValueError, match="no final CDM"):
        data.attach_test_targets(inputs, targets)


def test_inner_split_keeps_groups_together():
    from backend.app.models.lab.candidates import inner_split

    groups = np.repeat(np.arange(50), 3)
    strata = np.repeat(np.arange(50) % 2, 3)
    fit, val = inner_split(len(groups), strata, 0, 0.2, groups=groups)
    assert not set(groups[fit]) & set(groups[val])


def test_hurdle_falls_back_to_fixed_decision_with_few_high_risk_events(small_run):
    from backend.app.models.lab import candidates as C

    evs, _ = small_run
    b = protocol.Bundle(evs)
    idx = np.flatnonzero(b.trainable)[:300]
    h = C.Hurdle("hurdle_lrp", mode="lrp").fit(protocol._ctx(b, idx, "benchmark", 0))
    n_hr = int((b.y[idx][b.eligible[idx]] >= -6).sum())
    if n_hr < C.MIN_HR_FOR_TUNING:
        assert not h.tuned and (h.tau, h.alpha) == (C.FALLBACK_TAU, C.FALLBACK_ALPHA)
    assert len(h.clfs) == C.Hurdle.INNER_FOLDS


def test_cli_end_to_end_with_sealed_test_and_export(tmp_path, monkeypatch):
    from backend.app import config
    from backend.app.models import escalation, risk_model
    from backend.app.models.lab import __main__ as cli

    train = frame(700, seed=21)
    test = frame(250, seed=22, start_id=10_000)
    # official test inputs exclude CDMs inside 2 days; finals ship separately
    finals = test.sort_values("time_to_tca").groupby("event_id", as_index=False).first()
    test_inputs = test[test["time_to_tca"] >= data.INPUT_CUTOFF_DAYS]
    train.to_csv(tmp_path / "train.csv", index=False)
    test_inputs.to_csv(tmp_path / "test.csv", index=False)
    finals[["event_id", "time_to_tca", "risk"]].to_csv(tmp_path / "targets.csv", index=False)

    # force the gates open so the export path is exercised (the real gate
    # logic is tested above); nothing touches production artifacts
    ok = {"passed": True, "status": "passed", "basis": "sealed_test",
          "confirmed_on_sealed_test": True, "checks": [],
          "min_detectable_delta_pct": 1.0, "min_detectable_delta_pr_auc": 0.1}
    monkeypatch.setitem(gate.PRIMARY, "vigil_e_serving", "e_serving")   # the exportable format
    monkeypatch.setattr(gate, "_vigil_check", lambda n, cv, t, adjusted=False: {"candidate": n, **ok})
    monkeypatch.setattr(gate, "_ew_check", lambda n, cv, t, adjusted=False: {"candidate": n, **ok})
    out, exp = tmp_path / "out", tmp_path / "models"
    rc = cli.main(["--train", str(tmp_path / "train.csv"), "--test", str(tmp_path / "test.csv"),
                   "--test-targets", str(tmp_path / "targets.csv"), "--final",
                   "--repeats", "1", "--folds", "3",
                   "--bootstrap", "20", "--no-ceiling", "--out", str(out),
                   "--export", "--export-dir", str(exp)])
    assert rc == 0
    run = json.loads((out / "lab_report.json").read_text())
    assert run["test"]["n_events"] > 0 and "lrp" in run["test"]["regression"]
    assert "reproduces_published" in run["test"]["lrp_fidelity"]
    assert "VIGIL Lab report" in (out / "lab_report.md").read_text()

    # exported artifacts load in the production loaders
    monkeypatch.setattr(config, "RISK_MODEL_FILE", exp / "risk_model.json")
    monkeypatch.setattr(config, "RISK_MODEL_META_FILE", exp / "risk_model.meta.json")
    monkeypatch.setattr(config, "MODEL_DIR", exp)
    risk_model._LOAD_CACHE.clear()
    m = risk_model.RiskModel.load()
    assert m.booster is not None and m.version == "vigil-1.0" and m.target == "residual"
    cdms = [{"message_time": "2026-01-01T00:00:00Z", "miss_distance_m": 400.0,
             "relative_velocity_ms": 9000.0, "probability_of_collision": 1e-5}]
    pc, version = m.predict(cdms, "2026-01-03T00:00:00Z")
    assert 0.0 <= pc <= 1.0 and version == "vigil-1.0"
    escalation._LOAD_CACHE.clear()
    em = escalation.EscalationModel.load()
    assert em is not None and em.version == "vigil-e-1.0"
