"""Glass-box risk explanation tests - hermetic: temp DATA_DIR/DB/model
paths, no network, no live Claude calls. Contract: docs/CONTRACTS.md
'v13 additions' block C: RiskModel.explain, GET /risk/{id}/explain, and the
assistant explain_risk tool.
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend.app import config, db
from backend.app.models import risk_model
from backend.app.models.features import FEATURE_NAMES
from backend.app.models.risk_model import RiskModel

# the shipped hybrid artifact's identity (data/models/risk_model.meta.json)
HYBRID = {"version": "xgb-0.2", "target": "residual_below_threshold"}

# deterministic stand-in for RiskModel.explain output (wiring tests)
CANNED_EXPLANATION = {
    "base_value": -0.12,
    "contributions": [
        {"name": "pc_latest", "value": 1e-08, "contribution": 0.9},
        {"name": "miss_distance_m", "value": 5000.0, "contribution": -0.4},
        {"name": "time_to_tca_h", "value": 30.0, "contribution": 0.1},
    ],
}


def _seed(conn):
    """Three conjunctions spanning the hybrid's mode boundary:
    #1 latest Pc 3.1e-5 (log10 -4.5 >= -6) -> carry_forward;
    #2 latest Pc 1e-8  (log10 -8   <  -6) -> learned_residual;
    #3 all-null Pc (pc_latest 0)          -> learned_residual."""
    conn.executescript(
        """
    INSERT INTO satellites (id, norad_id, name, object_type, last_updated) VALUES
        (1, 25544, 'ISS (ZARYA)', 'PAYLOAD', '2026-07-21T13:02:41Z'),
        (2, 34454, 'COSMOS 2251 DEB', 'DEBRIS', '2026-07-21T13:02:41Z'),
        (3, 43556, 'AEROCUBE 12A', 'PAYLOAD', '2026-07-21T13:02:41Z');

    INSERT INTO conjunctions (id, primary_object_id, secondary_object_id,
                              time_of_closest_approach, first_seen_at, status) VALUES
        (1, 1, 2, '2026-07-23T04:12:09Z', '2026-07-20T10:00:00Z', 'active'),
        (2, 1, 3, '2026-07-23T10:00:00Z', '2026-07-21T06:00:00Z', 'active'),
        (3, 2, 3, '2026-07-24T00:00:00Z', '2026-07-21T08:00:00Z', 'active');

    INSERT INTO cdm_messages (conjunction_id, message_time, miss_distance_m,
                              relative_velocity_ms, probability_of_collision, source) VALUES
        (1, '2026-07-20T10:00:00Z', 1500.0, 14210.3, 1.1e-06, 'screening'),
        (1, '2026-07-21T10:00:00Z', 812.4, 14210.3, 3.1e-05, 'screening'),
        (2, '2026-07-21T06:00:00Z', 6200.0, 12000.0, 4.0e-08, 'screening'),
        (2, '2026-07-21T12:00:00Z', 5000.0, 12000.0, 1.0e-08, 'screening'),
        (3, '2026-07-21T08:00:00Z', 9000.0, 11000.0, NULL, 'screening');
    """
    )
    conn.commit()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "RISK_MODEL_FILE",
                        tmp_path / "models" / "risk_model.json")
    monkeypatch.setattr(config, "RISK_MODEL_META_FILE",
                        tmp_path / "models" / "risk_model.meta.json")
    conn = db.get_conn()
    db.init_db(conn)
    _seed(conn)
    conn.close()
    return tmp_path


@pytest.fixture()
def client(env):
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


def _install_model(monkeypatch, model):
    """RiskModel.load() -> the given prebuilt instance (no artifact I/O)."""
    monkeypatch.setattr(risk_model.RiskModel, "load",
                        classmethod(lambda cls: model))


def _hybrid_stub():
    """A hybrid-target model whose booster is a non-None placeholder - mode
    selection never touches it on the carry_forward path."""
    return RiskModel(booster=object(), **HYBRID)


def _import_xgboost():
    """Import xgboost or skip - its native lib raises XGBoostError (not
    ImportError) when libomp is missing, which importorskip does not catch."""
    try:
        import xgboost
        return xgboost
    except Exception as exc:  # ImportError or xgboost.core.XGBoostError
        pytest.skip(f"xgboost unavailable: {exc}")


def _train_tiny_booster(xgb):
    """A real 8-feature booster small enough to train inside a test."""
    import numpy as np

    rng = np.random.RandomState(7)
    X = rng.rand(96, len(FEATURE_NAMES))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 5] + rng.normal(0.0, 0.05, size=96)
    reg = xgb.XGBRegressor(n_estimators=8, max_depth=2, base_score=0.5)
    reg.fit(X, y)
    return reg.get_booster()


# --- RiskModel.explain -------------------------------------------------------

class TestExplainMethod:
    def test_heuristic_returns_none(self):
        feats = {name: 0.0 for name in FEATURE_NAMES}
        assert RiskModel().explain(feats) is None

    def test_real_booster_shape_sorting_and_additivity(self):
        xgb = _import_xgboost()
        import numpy as np

        booster = _train_tiny_booster(xgb)
        model = RiskModel(booster=booster, **HYBRID)
        feats = {name: 0.1 * (i + 1) for i, name in enumerate(FEATURE_NAMES)}

        out = model.explain(feats)
        assert set(out) == {"base_value", "contributions"}
        rows = out["contributions"]
        assert [set(r) for r in rows] == [
            {"name", "value", "contribution"}] * len(FEATURE_NAMES)
        assert {r["name"] for r in rows} == set(FEATURE_NAMES)
        for row in rows:  # each row echoes its input feature value
            assert row["value"] == pytest.approx(feats[row["name"]])
        magnitudes = [abs(r["contribution"]) for r in rows]
        assert magnitudes == sorted(magnitudes, reverse=True)
        # SHAP additivity: bias + contributions == the booster's raw output
        vec = np.asarray([[feats[n] for n in FEATURE_NAMES]], dtype=float)
        raw = float(booster.predict(xgb.DMatrix(vec))[0])
        total = out["base_value"] + sum(r["contribution"] for r in rows)
        assert total == pytest.approx(raw, abs=1e-4)

    def test_any_failure_returns_none(self):
        xgb = _import_xgboost()
        model = RiskModel(booster=_train_tiny_booster(xgb), **HYBRID)
        assert model.explain({}) is None  # missing feature keys
        assert model.explain({n: None for n in FEATURE_NAMES}) is None


# --- GET /risk/{id}/explain --------------------------------------------------

class TestExplainEndpoint:
    def test_unknown_id_404(self, client):
        response = client.get("/risk/999/explain")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_unavailable_when_no_artifact(self, client):
        # real load() path: nothing at the temp RISK_MODEL_FILE -> heuristic
        response = client.get("/risk/1/explain")
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "unavailable"
        assert body["model_version"] == config.HEURISTIC_MODEL_VERSION
        assert body["threshold_log10_pc"] == pytest.approx(-6.0)
        assert "contributions" not in body and "base_value" not in body
        assert "heuristic" in body["note"]

    def test_carry_forward_at_high_risk(self, client, monkeypatch):
        _install_model(monkeypatch, _hybrid_stub())
        body = client.get("/risk/1/explain").json()
        assert body["mode"] == "carry_forward"
        assert body["model_version"] == HYBRID["version"]
        assert body["threshold_log10_pc"] == pytest.approx(-6.0)
        assert "contributions" not in body and "base_value" not in body
        assert "defers" in body["note"]

    def test_learned_residual_below_threshold(self, client, monkeypatch):
        _install_model(monkeypatch, _hybrid_stub())
        monkeypatch.setattr(risk_model.RiskModel, "explain",
                            lambda self, feats: dict(CANNED_EXPLANATION))
        body = client.get("/risk/2/explain").json()
        assert body["mode"] == "learned_residual"
        assert body["model_version"] == HYBRID["version"]
        assert body["base_value"] == CANNED_EXPLANATION["base_value"]
        assert body["contributions"] == CANNED_EXPLANATION["contributions"]
        assert "residual" in body["note"]

    def test_zero_pc_sequence_is_residual_mode(self, client, monkeypatch):
        # all-null Pc -> pc_latest 0 -> below threshold, never carry_forward
        _install_model(monkeypatch, _hybrid_stub())
        monkeypatch.setattr(risk_model.RiskModel, "explain",
                            lambda self, feats: dict(CANNED_EXPLANATION))
        assert client.get("/risk/3/explain").json()["mode"] == "learned_residual"

    def test_unavailable_when_attribution_fails(self, client, monkeypatch):
        _install_model(monkeypatch, _hybrid_stub())
        monkeypatch.setattr(risk_model.RiskModel, "explain",
                            lambda self, feats: None)
        response = client.get("/risk/2/explain")
        assert response.status_code == 200  # degrade, never 500
        assert response.json()["mode"] == "unavailable"

    def test_end_to_end_with_tiny_real_artifact(self, client):
        xgb = _import_xgboost()
        config.RISK_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        _train_tiny_booster(xgb).save_model(str(config.RISK_MODEL_FILE))
        config.RISK_MODEL_META_FILE.write_text(json.dumps(
            {"version": "xgb-tiny", "target": "residual_below_threshold"}))

        low = client.get("/risk/2/explain").json()
        assert low["mode"] == "learned_residual"
        assert low["model_version"] == "xgb-tiny"
        assert len(low["contributions"]) == len(FEATURE_NAMES)
        assert isinstance(low["base_value"], float)

        high = client.get("/risk/1/explain").json()
        assert high["mode"] == "carry_forward"
        assert high["model_version"] == "xgb-tiny"


# --- assistant explain_risk tool ---------------------------------------------

def _execute_tool(name, args):
    from backend.app.assistant import tools as tools_mod

    conn = db.get_conn()
    try:
        content, is_error = tools_mod.ToolRuntime(conn).execute(name, args)
    finally:
        conn.close()
    return json.loads(content), is_error


class TestExplainTool:
    def test_schema_registered_and_prompt_mentions(self):
        from backend.app.assistant import prompt as prompt_mod
        from backend.app.assistant import tools as tools_mod

        [tool] = [t for t in tools_mod.TOOLS if t["name"] == "explain_risk"]
        assert tool["input_schema"]["required"] == ["conjunction_id"]
        assert tool["input_schema"]["additionalProperties"] is False
        assert set(tool["input_schema"]["properties"]) == {"conjunction_id"}
        assert "WHY" in tool["description"]
        assert "model_version" in tool["description"]
        assert "explain_risk" in tools_mod._HANDLERS

        text = prompt_mod.SYSTEM_PROMPT
        assert "explain_risk" in text
        assert "model_version" in text

    def test_unknown_id_surfaced_as_tool_error(self, env):
        payload, is_error = _execute_tool("explain_risk",
                                          {"conjunction_id": 999})
        assert is_error is True
        assert "not found" in payload["error"]

    def test_missing_argument_is_tool_error(self, env):
        payload, is_error = _execute_tool("explain_risk", {})
        assert is_error is True
        assert "conjunction_id" in payload["error"]

    def test_tool_returns_endpoint_shape(self, env, monkeypatch):
        _install_model(monkeypatch, _hybrid_stub())
        payload, is_error = _execute_tool("explain_risk",
                                          {"conjunction_id": 1})
        assert is_error is False
        assert payload["mode"] == "carry_forward"
        assert payload["model_version"] == HYBRID["version"]
        assert payload["threshold_log10_pc"] == pytest.approx(-6.0)
