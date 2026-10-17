"""Model-card endpoint tests - hermetic: temp DATA_DIR/DB paths, no network,
no real model training (the booster is a stub with a real get_score shape).
Contract: docs/CONTRACTS.md 'v12 additions'.
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend.app import config, db
from backend.app.models import risk_model
from backend.app.models.features import FEATURE_NAMES

# The real shipped metrics (data/models/risk_model.meta.json, xgb-0.2) as a
# fixture - the endpoint must surface these numbers verbatim; the UI hardcodes
# none of them.
META = {
    "version": "xgb-0.2",
    "target": "residual_below_threshold",
    "trained_at": "2026-07-21T18:52:39Z",
    "metrics": {
        "mse_test_all": 18.68046298645857,
        "mse_test_high_risk": 14.620441304391385,
        "baseline_mse_all": 22.59008764001637,
        "baseline_mse_high_risk": 14.872330848032435,
        "n_events": 12929,
        "n_train": 10343,
        "n_test": 2586,
        "n_test_high_risk": 76,
    },
}

CARD_KEYS = {"version", "target", "trained_at", "heuristic", "metrics",
             "feature_importance", "live",
             "early_warning"}  # VIGIL-E section, null without an artifact
LIVE_KEYS = {"predictions_total", "conjunctions_scored", "last_prediction_at"}


class FakeBooster:
    """get_score like a numpy-trained xgboost booster: 'f<i>' keys, gain
    values (models/train.py fits from bare arrays, so names are positional)."""

    def __init__(self, scores):
        self.scores = scores

    def get_score(self, importance_type):
        assert importance_type == "gain"
        return dict(self.scores)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "RISK_MODEL_FILE",
                        tmp_path / "models" / "risk_model.json")
    monkeypatch.setattr(config, "RISK_MODEL_META_FILE",
                        tmp_path / "models" / "risk_model.meta.json")
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


def _write_meta(meta=META):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.RISK_MODEL_META_FILE.write_text(json.dumps(meta))


def _install_booster(monkeypatch, scores, version=META["version"],
                     target=META["target"]):
    """RiskModel.load() -> a model wrapping FakeBooster (no real artifact)."""
    monkeypatch.setattr(
        risk_model.RiskModel, "load",
        classmethod(lambda cls: cls(booster=FakeBooster(scores),
                                    version=version, target=target)))


def test_card_heuristic_when_no_artifact(client):
    resp = client.get("/model/card")
    assert resp.status_code == 200
    card = resp.json()
    assert set(card) == CARD_KEYS
    assert card["version"] == config.HEURISTIC_MODEL_VERSION
    assert card["heuristic"] is True
    assert card["target"] is None
    assert card["trained_at"] is None
    assert card["metrics"] is None
    assert card["feature_importance"] == []
    assert card["early_warning"] is None
    assert set(card["live"]) == LIVE_KEYS
    assert card["live"] == {"predictions_total": 0, "conjunctions_scored": 0,
                            "last_prediction_at": None}


def test_card_full_when_artifact_present(client, monkeypatch):
    _write_meta()
    # gains deliberately unsorted; 'f99' and a junk key must be skipped
    _install_booster(monkeypatch, {
        "f0": 41.5, "f2": 812.9, "f6": 230.4, "f3": 12.0,
        "f99": 5.0, "not-a-feature": 3.0,
    })
    card = client.get("/model/card").json()
    assert card["version"] == META["version"]
    assert card["target"] == META["target"]
    assert card["trained_at"] == META["trained_at"]
    assert card["heuristic"] is False
    assert card["metrics"] == META["metrics"]
    # mapped through FEATURE_NAMES, sorted by gain desc, unknowns dropped
    assert card["feature_importance"] == [
        {"name": FEATURE_NAMES[2], "gain": 812.9},
        {"name": FEATURE_NAMES[6], "gain": 230.4},
        {"name": FEATURE_NAMES[0], "gain": 41.5},
        {"name": FEATURE_NAMES[3], "gain": 12.0},
    ]


def test_card_importance_caps_at_top_12(client, monkeypatch):
    from backend.app.api import model as model_api

    _write_meta()
    # named keys map straight through; more entries than the cap
    scores = {name: float(i + 1) for i, name in enumerate(FEATURE_NAMES)}
    scores.update({f"f{i}": 100.0 + i for i in range(len(FEATURE_NAMES))})
    _install_booster(monkeypatch, scores)
    fi = client.get("/model/card").json()["feature_importance"]
    assert len(fi) == model_api.IMPORTANCE_TOP_N
    gains = [row["gain"] for row in fi]
    assert gains == sorted(gains, reverse=True)


def test_card_corrupt_artifact_degrades_not_500(client):
    # real RiskModel.load() path: meta is fine, booster file is garbage ->
    # heuristic fallback, metrics still served, importances empty
    _write_meta()
    config.RISK_MODEL_FILE.write_bytes(b"this is not an xgboost model")
    resp = client.get("/model/card")
    assert resp.status_code == 200
    card = resp.json()
    assert card["heuristic"] is True
    assert card["metrics"] == META["metrics"]
    assert card["feature_importance"] == []


def test_card_corrupt_meta_degrades_not_500(client):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.RISK_MODEL_META_FILE.write_text("{not json")
    resp = client.get("/model/card")
    assert resp.status_code == 200
    card = resp.json()
    assert card["version"] == config.HEURISTIC_MODEL_VERSION
    assert card["metrics"] is None


def test_card_live_counts_from_seeded_predictions(client):
    conn = db.get_conn(config.DB_PATH)
    try:
        conn.executescript("""
        INSERT INTO satellites (id, norad_id, name, object_type, last_updated) VALUES
            (1, 25544, 'ISS (ZARYA)', 'PAYLOAD', '2026-07-21T13:02:41Z'),
            (2, 34454, 'COSMOS 2251 DEB', 'DEBRIS', '2026-07-21T13:02:41Z'),
            (3, 43556, 'AEROCUBE 12A', 'PAYLOAD', '2026-07-21T13:02:41Z');
        INSERT INTO conjunctions (id, primary_object_id, secondary_object_id,
                                  time_of_closest_approach, first_seen_at) VALUES
            (1, 1, 2, '2026-07-23T04:12:09Z', '2026-07-20T10:00:00Z'),
            (2, 1, 3, '2026-07-22T10:00:00Z', '2026-07-21T06:00:00Z');
        INSERT INTO risk_predictions (conjunction_id, predicted_final_risk,
                                      prediction_time, model_version) VALUES
            (1, 3.1e-05, '2026-07-21T10:00:00Z', 'xgb-0.2'),
            (1, 4.4e-05, '2026-07-21T14:00:00Z', 'xgb-0.2'),
            (2, 1.0e-06, '2026-07-21T12:00:00Z', 'xgb-0.2');
        """)
        conn.commit()
    finally:
        conn.close()

    live = client.get("/model/card").json()["live"]
    assert live == {
        "predictions_total": 3,
        "conjunctions_scored": 2,
        "last_prediction_at": "2026-07-21T14:00:00Z",
    }


def test_benchmark_card_comes_from_the_committed_lab_reports(client):
    import json
    from pathlib import Path

    from backend.app.models.lab import benchmark_card

    r = client.get("/model/benchmark")
    assert r.status_code == 200
    card = r.json()
    assert card["fidelity"]["reproduces_published"] is True
    assert card["vigil"]["L"] < card["esa_model"]["L"] < card["baseline"]["L"]
    # every figure is regenerable from the reports it names (no hand-typed numbers)
    if all((benchmark_card.ROOT / s["path"]).exists() for s in card["sources"]):
        assert json.loads(json.dumps(benchmark_card.build())) == card
    assert Path(benchmark_card.OUT).exists()


def test_benchmark_card_absent_is_a_404_not_a_500(client, monkeypatch):
    from backend.app.api import model as model_api

    monkeypatch.setattr(model_api, "BENCHMARK_CARD", "no/such/card.json")
    assert client.get("/model/benchmark").status_code == 404
