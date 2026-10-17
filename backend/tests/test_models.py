"""Tests for the risk-model layer: pc, features, risk_model, train helpers."""
import json
import math
import re

import numpy as np
import pytest

from backend.app import config, db
from backend.app.models import pc, risk_model, train
from backend.app.models.features import FEATURE_NAMES, build_features
from backend.app.models.risk_model import (RiskModel, predict_for_conjunction,
                                           risk_level)

ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

TCA = "2026-07-23T00:00:00Z"
CDMS = [
    {"message_time": "2026-07-20T00:00:00Z", "miss_distance_m": 2000.0,
     "relative_velocity_ms": 14000.0, "probability_of_collision": 1e-6},
    {"message_time": "2026-07-21T00:00:00Z", "miss_distance_m": 1500.0,
     "relative_velocity_ms": 14100.0, "probability_of_collision": None},
    {"message_time": "2026-07-22T00:00:00Z", "miss_distance_m": 1000.0,
     "relative_velocity_ms": 14200.0, "probability_of_collision": 1e-5},
]


# --- pc ----------------------------------------------------------------------

class TestPc:
    def test_sigma_at(self):
        assert pc.sigma_at(0.0) == pytest.approx(config.SIGMA_BASE_M)
        assert pc.sigma_at(1.0) == pytest.approx(
            config.SIGMA_BASE_M + config.SIGMA_GROWTH_M_PER_DAY)
        # never shrinks below the epoch value
        assert pc.sigma_at(-2.0) == pytest.approx(config.SIGMA_BASE_M)

    def test_bounds(self):
        for d in [0.0, 10.0, 500.0, 5000.0, 1e6]:
            for age in [0.0, 0.5, 3.0]:
                value = pc.compute_pc(d, 14000.0, propagation_age_days=age)
                assert 0.0 <= value <= 1.0

    def test_monotonic_decreasing_and_max_at_zero(self):
        distances = [0.0, 50.0, 200.0, 800.0, 2000.0, 8000.0]
        values = [pc.compute_pc(d, 14000.0, propagation_age_days=1.0)
                  for d in distances]
        assert values[0] == max(values)
        for lo, hi in zip(values[1:], values[:-1]):
            assert lo < hi  # strictly decreasing with miss distance

    def test_formula_value(self):
        sigma = math.sqrt(2.0) * pc.sigma_at(1.0)
        d, hbr = 700.0, config.HARD_BODY_RADIUS_M
        expected = (hbr ** 2 / (2 * sigma ** 2)) * math.exp(-d ** 2 / (2 * sigma ** 2))
        assert pc.compute_pc(d, 14000.0, propagation_age_days=1.0) == pytest.approx(expected)

    def test_hard_body_radius_override_and_clamp(self):
        default = pc.compute_pc(100.0, 14000.0)
        explicit = pc.compute_pc(100.0, 14000.0,
                                 hard_body_radius_m=config.HARD_BODY_RADIUS_M)
        assert default == pytest.approx(explicit)
        # enormous HBR pushes the closed form above 1 -> clamped
        assert pc.compute_pc(0.0, 14000.0, propagation_age_days=0.0,
                             hard_body_radius_m=1e9) == 1.0

    def test_deterministic(self):
        a = pc.compute_pc(812.4, 14210.3, propagation_age_days=1.6)
        b = pc.compute_pc(812.4, 14210.3, propagation_age_days=1.6)
        assert a == b


# --- features ----------------------------------------------------------------

class TestFeatures:
    def test_full_sequence(self):
        feats = build_features(CDMS, TCA)
        assert set(feats) == set(FEATURE_NAMES)
        assert feats["miss_distance_m"] == pytest.approx(1000.0)
        assert feats["relative_velocity_ms"] == pytest.approx(14200.0)
        assert feats["pc_latest"] == pytest.approx(1e-5)
        assert feats["time_to_tca_h"] == pytest.approx(24.0)  # from latest CDM
        assert feats["cdm_count"] == 3.0
        assert feats["md_delta"] == pytest.approx(-1000.0)
        assert feats["pc_log_delta"] == pytest.approx(1.0)  # 1e-6 -> 1e-5
        assert feats["md_slope"] == pytest.approx(-500.0)  # m/day, exact linear

    def test_as_of_override(self):
        feats = build_features(CDMS, TCA, as_of_iso="2026-07-22T12:00:00Z")
        assert feats["time_to_tca_h"] == pytest.approx(12.0)

    def test_single_cdm(self):
        feats = build_features([CDMS[0]], TCA)
        assert feats["cdm_count"] == 1.0
        assert feats["md_delta"] == 0.0
        assert feats["pc_log_delta"] == 0.0
        assert feats["md_slope"] == 0.0
        assert feats["pc_latest"] == pytest.approx(1e-6)
        assert feats["time_to_tca_h"] == pytest.approx(72.0)

    def test_single_cdm_null_pc(self):
        cdm = dict(CDMS[0], probability_of_collision=None)
        feats = build_features([cdm], TCA)
        assert feats["pc_latest"] == 0.0
        assert feats["pc_log_delta"] == 0.0

    def test_all_null_pcs(self):
        cdms = [dict(c, probability_of_collision=None) for c in CDMS]
        feats = build_features(cdms, TCA)
        assert feats["pc_latest"] == 0.0
        assert feats["pc_log_delta"] == 0.0
        assert feats["md_delta"] == pytest.approx(-1000.0)  # geometry unaffected

    def test_one_non_null_pc_no_delta(self):
        cdms = [dict(CDMS[0], probability_of_collision=None),
                dict(CDMS[1]), dict(CDMS[2])]
        feats = build_features(cdms, TCA)
        assert feats["pc_latest"] == pytest.approx(1e-5)
        assert feats["pc_log_delta"] == 0.0  # only one positive pc

    def test_empty_sequence(self):
        feats = build_features([], TCA)
        assert set(feats) == set(FEATURE_NAMES)
        assert all(v == 0.0 for v in feats.values())

    def test_unsorted_input_is_sorted(self):
        feats = build_features(list(reversed(CDMS)), TCA)
        assert feats["md_delta"] == pytest.approx(-1000.0)
        assert feats["pc_log_delta"] == pytest.approx(1.0)


# --- risk_model --------------------------------------------------------------

class TestHeuristicPredict:
    def test_extrapolates_by_pc_log_delta(self):
        predicted, version = RiskModel().predict(CDMS, TCA)
        # pc_latest 1e-5 * 10**1.0 = 1e-4
        assert predicted == pytest.approx(1e-4)
        assert version == config.HEURISTIC_MODEL_VERSION

    def test_cap_at_10x(self):
        cdms = [dict(CDMS[0], probability_of_collision=1e-8),
                dict(CDMS[2], probability_of_collision=1e-5)]
        predicted, _ = RiskModel().predict(cdms, TCA)
        assert predicted == pytest.approx(1e-4)  # delta 3 capped at 1 -> 10x

    def test_cap_at_tenth_downward(self):
        cdms = [dict(CDMS[0], probability_of_collision=1e-4),
                dict(CDMS[2], probability_of_collision=1e-6)]
        predicted, _ = RiskModel().predict(cdms, TCA)
        assert predicted == pytest.approx(1e-7)  # delta -2 capped at -1

    def test_no_pc_at_all(self):
        cdms = [dict(c, probability_of_collision=None) for c in CDMS]
        predicted, version = RiskModel().predict(cdms, TCA)
        assert predicted == 0.0
        assert version == config.HEURISTIC_MODEL_VERSION

    def test_clamped_to_one(self):
        cdms = [dict(CDMS[0], probability_of_collision=0.05),
                dict(CDMS[2], probability_of_collision=0.5)]
        predicted, _ = RiskModel().predict(cdms, TCA)
        assert predicted == 1.0  # 0.5 * 10x -> clamp

    def test_load_falls_back_to_heuristic(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "RISK_MODEL_FILE", tmp_path / "absent.json")
        model = RiskModel.load()
        assert model.booster is None
        assert model.version == config.HEURISTIC_MODEL_VERSION


def _import_xgboost():
    """Import xgboost or skip - its native lib raises XGBoostError (not
    ImportError) when libomp is missing, which importorskip does not catch."""
    try:
        import xgboost
        return xgboost
    except Exception as exc:  # ImportError or xgboost.core.XGBoostError
        pytest.skip(f"xgboost unavailable: {exc}")


class TestXgbLoad:
    def test_load_artifact_and_predict(self, tmp_path, monkeypatch):
        xgb = _import_xgboost()
        rng = np.random.RandomState(0)
        X = rng.rand(64, len(FEATURE_NAMES))
        y = rng.uniform(-10.0, -4.0, size=64)  # log10 risk targets
        reg = xgb.XGBRegressor(n_estimators=5, max_depth=2)
        reg.fit(X, y)
        model_file = tmp_path / "risk_model.json"
        meta_file = tmp_path / "risk_model.meta.json"
        reg.get_booster().save_model(str(model_file))
        meta_file.write_text(json.dumps({"version": "xgb-0.1"}))
        monkeypatch.setattr(config, "RISK_MODEL_FILE", model_file)
        monkeypatch.setattr(config, "RISK_MODEL_META_FILE", meta_file)

        model = RiskModel.load()
        assert model.booster is not None
        assert model.version == "xgb-0.1"
        predicted, version = model.predict(CDMS, TCA)
        assert version == "xgb-0.1"
        assert 0.0 <= predicted <= 1.0


class TestRiskLevel:
    @pytest.mark.parametrize("value,expected", [
        (5e-4, "critical"),
        (1e-4, "critical"),      # threshold inclusive
        (9.9e-5, "elevated"),
        (1e-6, "elevated"),      # threshold inclusive
        (9.9e-7, "nominal"),
        (0.0, "nominal"),
        (None, "nominal"),
    ])
    def test_thresholds(self, value, expected):
        assert risk_level(value) == expected


class TestPredictForConjunction:
    @pytest.fixture
    def conn(self, tmp_path):
        conn = db.get_conn(tmp_path / "test.db")
        db.init_db(conn)
        now = "2026-07-19T00:00:00Z"
        conn.execute("INSERT INTO satellites (id, norad_id, name, object_type,"
                     " last_updated) VALUES (1, 25544, 'ISS (ZARYA)', 'PAYLOAD', ?)",
                     (now,))
        conn.execute("INSERT INTO satellites (id, norad_id, name, object_type,"
                     " last_updated) VALUES (2, 34454, 'COSMOS 2251 DEB', 'DEBRIS', ?)",
                     (now,))
        conn.execute("INSERT INTO conjunctions (id, primary_object_id,"
                     " secondary_object_id, time_of_closest_approach,"
                     " first_seen_at, status) VALUES (1, 1, 2, ?, ?, 'active')",
                     (TCA, now))
        # insert out of order: reads must sort ascending by message_time
        for cdm in [CDMS[2], CDMS[0], CDMS[1]]:
            conn.execute(
                "INSERT INTO cdm_messages (conjunction_id, message_time,"
                " miss_distance_m, relative_velocity_ms,"
                " probability_of_collision, source)"
                " VALUES (1, ?, ?, ?, ?, 'screening')",
                (cdm["message_time"], cdm["miss_distance_m"],
                 cdm["relative_velocity_ms"], cdm["probability_of_collision"]))
        conn.commit()
        yield conn
        conn.close()

    def test_predicts_and_persists(self, conn):
        result = predict_for_conjunction(conn, 1, model=RiskModel())
        assert set(result) == {"conjunction_id", "predicted_final_risk",
                               "model_version", "prediction_time",
                               "escalation_prob", "escalation_model_version"}
        # no VIGIL-E artifact passed -> escalation fields are null
        assert result["escalation_prob"] is None
        assert result["escalation_model_version"] is None
        assert result["conjunction_id"] == 1
        assert result["predicted_final_risk"] == pytest.approx(1e-4)
        assert result["model_version"] == config.HEURISTIC_MODEL_VERSION
        assert ISO_Z.match(result["prediction_time"])

        rows = conn.execute("SELECT * FROM risk_predictions"
                            " WHERE conjunction_id = 1").fetchall()
        assert len(rows) == 1
        assert rows[0]["predicted_final_risk"] == pytest.approx(1e-4)
        assert rows[0]["model_version"] == config.HEURISTIC_MODEL_VERSION
        assert rows[0]["prediction_time"] == result["prediction_time"]

    def test_multiple_predictions_append(self, conn):
        predict_for_conjunction(conn, 1, model=RiskModel())
        predict_for_conjunction(conn, 1, model=RiskModel())
        n = conn.execute("SELECT COUNT(*) AS n FROM risk_predictions"
                         " WHERE conjunction_id = 1").fetchone()["n"]
        assert n == 2

    def test_unknown_id_raises(self, conn):
        with pytest.raises(ValueError):
            predict_for_conjunction(conn, 999, model=RiskModel())

    def test_conjunction_without_cdms(self, conn):
        conn.execute("INSERT INTO conjunctions (id, primary_object_id,"
                     " secondary_object_id, time_of_closest_approach,"
                     " first_seen_at, status) VALUES (2, 2, 1, ?, ?, 'active')",
                     (TCA, "2026-07-19T00:00:00Z"))
        conn.commit()
        result = predict_for_conjunction(conn, 2, model=RiskModel())
        assert result["predicted_final_risk"] == 0.0


# --- train helpers -----------------------------------------------------------

class TestTrainDataset:
    def test_build_dataset_synthetic(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({
            "event_id":       [1, 1, 1, 2, 2, 3],
            "time_to_tca":    [5.0, 3.0, 1.0, 4.0, 0.5, 2.0],
            "miss_distance":  [9000.0, 6000.0, 3000.0, 5000.0, 4000.0, 700.0],
            "relative_speed": [12000.0, 12010.0, 12020.0, 9000.0, 9010.0, 7000.0],
            "risk":           [-8.0, -7.0, -6.0, -10.0, -9.5, -5.0],
        })
        X, y, baseline = train.build_dataset(df)
        # event 3 has a single CDM -> skipped
        assert X.shape == (2, len(FEATURE_NAMES))
        assert list(y) == [-6.0, -9.5]  # final CDM risk per event
        assert list(baseline) == [-7.0, -10.0]  # last non-final risk
        i_md = FEATURE_NAMES.index("miss_distance_m")
        i_count = FEATURE_NAMES.index("cdm_count")
        i_ttc = FEATURE_NAMES.index("time_to_tca_h")
        assert X[0][i_md] == pytest.approx(6000.0)  # latest non-final CDM
        assert X[0][i_count] == 2.0
        assert X[0][i_ttc] == pytest.approx(72.0)  # 3 days to TCA
        assert X[1][i_count] == 1.0

    def test_missing_csv_exits_zero(self, tmp_path, capsys):
        rc = train.main(["--data", str(tmp_path / "nope.csv")])
        assert rc == 0
        out = capsys.readouterr().out
        assert "zenodo.org" in out
