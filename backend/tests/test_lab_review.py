"""Post-hoc review checks: the corrected t-test, the mission-level bound, the
in-fold threshold, the isotonic baseline, the rotated-physics switch and the
class mixes."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.models.lab import physics, review
from backend.app.models.lab import round4 as R4


def test_corrected_t_matches_the_formula():
    d = np.array([0.5, 1.5, 1.0, 2.0, 0.0, 1.0, 1.2, 0.8, 1.1, 0.9])
    out = review.corrected_t(d, folds=5)
    se = math.sqrt(d.var(ddof=1) * (1 / len(d) + 1 / 4))
    assert out["mean"] == pytest.approx(d.mean())
    assert out["t"] == pytest.approx(d.mean() / se)
    lo, hi = out["ci95"]
    assert lo < d.mean() < hi
    # wider than the uncorrected interval, which ignores the shared training data
    assert (hi - lo) > 2 * 1.96 * d.std(ddof=1) / math.sqrt(len(d))


def test_corrected_t_ignores_missing_folds():
    out = review.corrected_t([1.0, float("nan"), 2.0, 3.0], folds=5)
    assert out["n_folds"] == 3


def test_mission_bound_is_a_union_bound_per_mission():
    out = review.mission_bound(["a", "a", "a", "b"], [0.4, 0.5, 0.6, 0.2], [True, True, False, False])
    assert out["n_missions"] == 2
    assert out["missions_with_crossing"] == 1
    assert out["coherent_max_missions"] == pytest.approx(1.0 + 0.2)   # mission a capped at 1
    assert out["crossings_per_crossing_mission"] == [2]
    assert out["p_chernoff_missions"] == pytest.approx(R4.chernoff_p(1, 1.2))


def test_keep_threshold_keeps_95_percent():
    s = np.arange(1, 101, dtype=float)
    thr = review.keep_threshold(s, 0.95)
    assert np.mean(s >= thr) >= 0.95
    assert np.mean(s > thr) < 0.95


def test_isotonic_baseline_is_monotone_and_bounded():
    rng = np.random.RandomState(0)
    lat = rng.uniform(-10, -3, 400)
    y = np.where(rng.uniform(size=400) < 1 / (1 + np.exp(-(lat + 6))), -5.0, -7.0)
    f = review.isotonic_baseline(lat, y)
    grid = np.linspace(-12, -2, 50)
    p = f(grid)
    assert np.all(np.diff(p) >= -1e-12)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_rotated_physics_rotates_only_the_chaser_and_restores():
    row = {"t_sigma_r": 10.0, "t_sigma_t": 200.0, "t_sigma_n": 20.0,
           "c_sigma_r": 15.0, "c_sigma_t": 300.0, "c_sigma_n": 25.0,
           "t_j2k_sma": 7000.0, "t_j2k_ecc": 0.001,
           "relative_velocity_r": 0.0, "relative_velocity_t": -7000.0, "relative_velocity_n": 9000.0}
    orig = physics.covariance_rtn
    plain_t, plain_c = orig(row, "t"), orig(row, "c")
    with review.rotated_physics() as counts:
        rot_t, rot_c = physics.covariance_rtn(row, "t"), physics.covariance_rtn(row, "c")
    assert physics.covariance_rtn is orig
    assert counts == {"rotated": 1, "unrotated": 0}
    np.testing.assert_allclose(rot_t, plain_t)
    assert not np.allclose(rot_c, plain_c)
    np.testing.assert_allclose(np.trace(rot_c), np.trace(plain_c))   # a rotation keeps the trace


def test_class_mixes_pool_the_eligible_events_of_both_files():
    def ev(target, eligible=True):
        return SimpleNamespace(target=target, test_eligible=eligible)
    train = [ev(-5.0)] + [ev(-20.0)] * 9 + [ev(-5.0, eligible=False)]
    test = [ev(-5.0)] * 3 + [ev(-20.0)] * 7
    mx = review.class_mixes(train, test)
    assert mx["training_file"] == {"n": 10, "n_high_risk": 1, "prevalence": 0.1}
    assert mx["test"]["prevalence"] == pytest.approx(0.3)
    assert mx["pooled"] == {"n": 20, "n_high_risk": 4, "prevalence": 0.2}
