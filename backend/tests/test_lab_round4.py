"""VIGIL Lab Round 4 (backend/app/models/lab/round4.py)."""
import math

import numpy as np
import pandas as pd

from backend.app.models.lab import round4 as R4

V_LEO = 1000.0 * math.sqrt(R4.MU_KM3_S2 / 7000.0)


def _row(vrel=(0.0, -2 * V_LEO, 0.0), **over):
    row = {"t_j2k_sma": 7000.0, "t_j2k_ecc": 0.001,
           "relative_velocity_r": vrel[0], "relative_velocity_t": vrel[1], "relative_velocity_n": vrel[2],
           "relative_position_r": 10.0, "relative_position_t": 0.0, "relative_position_n": 200.0,
           "t_sigma_r": 5.0, "t_sigma_t": 50.0, "t_sigma_n": 5.0,
           "c_sigma_r": 20.0, "c_sigma_t": 400.0, "c_sigma_n": 20.0,
           "t_ct_r": -0.5, "t_cn_r": 0.0, "t_cn_t": 0.0, "c_ct_r": -0.5, "c_cn_r": 0.0, "c_cn_t": 0.0,
           "t_span": 2.0, "c_span": 2.0, "risk": -5.0}
    row.update(over)
    return row


def test_head_on_rotation_flips_along_and_cross_track():
    m = R4.chaser_to_target(_row())
    assert np.allclose(m, np.diag([1.0, -1.0, -1.0]), atol=1e-12)


def test_perpendicular_crossing_maps_chaser_along_track_to_target_cross_track():
    m = R4.chaser_to_target(_row(vrel=(0.0, -V_LEO, V_LEO)))
    assert np.allclose(m @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(m.T @ m, np.eye(3), atol=1e-12)


def test_rotation_refused_for_an_eccentric_target_or_missing_orbit():
    assert R4.chaser_to_target(_row(t_j2k_ecc=0.5)) is None
    assert R4.chaser_to_target(_row(t_j2k_ecc=float("nan"))) is None
    assert R4.chaser_to_target(_row(t_j2k_sma=float("nan"))) is None


def test_rotation_keeps_the_combined_covariance_trace_and_moves_the_long_axis():
    row = _row(vrel=(0.0, -V_LEO, V_LEO))
    plain, rot = R4.combined_covariance(row, rotate=False), R4.combined_covariance(row, rotate=True)
    assert np.isclose(np.trace(plain), np.trace(rot))
    assert rot[2, 2] > 100 * plain[2, 2] / 2        # the chaser's 400 m along-track sigma now lies cross-track
    assert np.all(np.linalg.eigvalsh(rot) > 0)


def test_disk_integral_matches_the_small_body_formula():
    s, hbr = 100.0, 1.0
    x = np.array([150.0, 0.0])
    pc = R4.pc_disk(x, np.eye(2) * s * s, hbr)
    approx = hbr ** 2 / (2 * s * s) * math.exp(-0.5 * (x @ x) / (s * s))
    assert abs(pc / approx - 1) < 1e-3


def test_recomputed_pc_is_finite_and_rises_with_a_closer_miss():
    far = R4.recomputed_log10_pc(_row(relative_position_n=400.0))
    near = R4.recomputed_log10_pc(_row(relative_position_n=50.0))
    assert np.isfinite(far) and np.isfinite(near) and near > far
    assert np.isnan(R4.recomputed_log10_pc(_row(t_span=float("nan"))))


def test_update_d2_is_the_mahalanobis_length_of_the_change():
    a = _row()
    b = _row(relative_position_n=230.0)
    c = R4.combined_covariance(a, rotate=True)
    dx = np.array([0.0, 0.0, 30.0])
    assert np.isclose(R4.update_d2([a, b])[0], dx @ np.linalg.solve(c, dx))
    assert R4.update_d2([a]) == []


def test_plane_update_measures_the_change_in_the_encounter_plane():
    a = _row(t_ct_r=-0.9, c_ct_r=0.9)                     # head-on: both are -0.9 in the target's frame
    b = _row(t_ct_r=-0.9, c_ct_r=0.9, relative_position_r=40.0)   # a 30 m radial change, in the plane
    c = R4.combined_covariance(a, rotate=True)
    plane = R4.update_d2_plane([a, b])[0]
    assert np.isclose(plane, 30.0 ** 2 / c[0, 0])         # the plane is R-N, with no R-N correlation here
    assert R4.update_d2([a, b])[0] > 4 * plane            # the 3-D length also counts the R-T correlation
    along = _row(t_ct_r=-0.9, c_ct_r=0.9, relative_position_t=500.0)   # a change along the motion
    assert R4.update_d2_plane([a, along])[0] < 1e-12
    assert R4.update_d2_plane([a]) == []


def test_3d_length_of_an_in_plane_change_is_never_below_the_planar_length():
    rng = np.random.RandomState(0)
    for _ in range(200):
        g = rng.randn(3, 3)
        c = g @ g.T + 1e-3 * np.eye(3)
        m = R4.plane_basis(rng.randn(3))
        x = m.T @ rng.randn(2)                             # a vector in the plane
        assert x @ np.linalg.solve(c, x) >= (m @ x) @ np.linalg.solve(m @ c @ m.T, m @ x) - 1e-9


def test_clip_variant_reproduces_the_registered_covariance():
    from backend.app.models.lab.physics import covariance_rtn
    row = _row(t_ct_r=-0.9995, c_cn_t=0.3)
    for o in "tc":
        assert np.allclose(R4.covariance_rtn_clip(row, o, 0.999), covariance_rtn(row, o))
    assert not np.allclose(R4.covariance_rtn_clip(row, "t", None), covariance_rtn(row, "t"))


def test_tolerance_matching_survives_a_rounding_boundary():
    mid = 3.123456789 + 0.5e-9                   # halfway between two 9-decimal keys
    t_raw, t_test = mid - 1e-12, mid + 1e-12     # one CDM, written with a little float noise
    assert round(t_raw, 9) != round(t_test, 9)   # rounded keys split it
    base = dict(mission_id=1, miss_distance=100.0)
    raw = pd.DataFrame([{**base, "event_id": 7, "time_to_tca": t_raw, "risk": -8.0},
                        {**base, "event_id": 7, "time_to_tca": 0.4, "risk": -6.5}])
    test = pd.DataFrame([{**base, "event_id": 0, "time_to_tca": t_test, "risk": -8.0}])
    targets = pd.DataFrame([{"event_id": 0, "time_to_tca": 0.4, "true_risk": -6.5}])
    assert R4.match_raw(test, targets, raw) == {}
    assert set(R4.match_raw_tol(test, targets, raw)) == {0}


def test_chernoff_bound_and_coherent_bound():
    assert R4.chernoff_p(1, 2.0) == 1.0
    assert R4.chernoff_p(5, 0.01) < 1e-9 and R4.chernoff_p(6, 0.01) < R4.chernoff_p(5, 0.01)
    assert 0 < R4.chernoff_p(3, 1.0) < 1
    b = R4.coherent_bound(np.array([-30.0, -10.0, -7.0, -5.0]), rho=2.0)
    assert np.allclose(b, [2e-24, 2e-4, 0.2, 1.0])


def test_public_rule_keeps_public_quantities_only():
    cols = ["s_pc_latest", "h_risk_slope", "h_log_sigma_rate", "h_headroom_mre", "l_ctype_debris",
            "l_risk", "l_miss_distance", "l_relative_velocity_t", "l_relative_position_r",
            "l_t_sigma_t", "l_c_ct_r", "l_mahalanobis_distance", "l_max_risk_estimate", "l_c_obs_used",
            "l_t_weighted_rms", "l_c_span", "l_c_cr_area_over_mass", "l_c_j2k_inc", "l_F10",
            "l_c_cd_area_over_mass", "p_m2", "l_t_rcs_estimate", "l_c_rcs_estimate"]
    keep = R4.public_columns(cols)
    assert keep == ["s_pc_latest", "h_risk_slope", "l_ctype_debris", "l_risk", "l_miss_distance",
                    "l_relative_velocity_t", "l_c_j2k_inc", "l_F10", "l_c_cd_area_over_mass"]
    X = np.arange(len(cols), dtype=float)[None, :].repeat(3, 0)
    X[:, cols.index("l_c_rcs_estimate")] = [0.05, 0.5, np.nan]
    M, names = R4.public_matrix(X, cols)
    assert names[-2:] == ["l_t_rcs_estimate_class", "l_c_rcs_estimate_class"]
    assert np.array_equal(M[:, -1], [0.0, 1.0, np.nan], equal_nan=True)
    assert np.array_equal(R4.rcs_class([0.0999, 0.1, 0.9999, 1.0]), [0.0, 1.0, 1.0, 2.0])


def test_match_raw_needs_one_raw_event_with_the_published_final():
    base = dict(mission_id=1, miss_distance=100.0)
    raw = pd.DataFrame([
        {**base, "event_id": 7, "time_to_tca": 3.0, "risk": -8.0},
        {**base, "event_id": 7, "time_to_tca": 2.5, "risk": -7.0},
        {**base, "event_id": 7, "time_to_tca": 1.5, "risk": -5.0},
        {**base, "event_id": 7, "time_to_tca": 0.4, "risk": -6.5},
        {**base, "event_id": 9, "time_to_tca": 3.0, "risk": -9.0},
        {**base, "event_id": 9, "time_to_tca": 0.5, "risk": -9.5},
    ])
    test = pd.DataFrame([
        {**base, "event_id": 0, "time_to_tca": 3.0, "risk": -8.0},
        {**base, "event_id": 0, "time_to_tca": 2.5, "risk": -7.0},
        {**base, "event_id": 1, "time_to_tca": 3.0, "risk": -9.0},
    ])
    targets = pd.DataFrame([{"event_id": 0, "time_to_tca": 0.4, "true_risk": -6.5},
                            {"event_id": 1, "time_to_tca": 0.5, "true_risk": -1.0}])   # final differs
    late = R4.match_raw(test, targets, raw)
    assert set(late) == {0}
    assert [r["risk"] for r in late[0]] == [-5.0, -6.5]
