"""VIGIL-E (early-warning escalation classifier) tests.

Every dataset here is SYNTHETIC: a small Kelvins-shaped CSV generated in the
test (columns event_id, time_to_tca [days], miss_distance [m],
relative_speed [m/s], risk [log10 Pc]). Escalating events carry a rising
risk trend, the rest drift flat or down - a planted signal so the training
pipeline can be exercised end to end. None of these numbers describe the
real model; the real artifact is trained on ESA Kelvins by the owner.

Postgres tests run only when KESSLER_TEST_PG_URL points at a server where
the user may CREATE DATABASE (each test gets a fresh database).
"""
import json
import os
import sqlite3
import uuid

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import config, db
from backend.app.models import escalation, risk_model, train_escalation
from backend.app.models.escalation import ESCALATION_FEATURE_NAMES
from backend.app.models.features import FEATURE_NAMES, build_features


# --- synthetic fixture ------------------------------------------------------------


def synthetic_kelvins(n_events=600, seed=0, signal=True) -> pd.DataFrame:
    """Clearly synthetic Kelvins-shaped CDM table (see module docstring)."""
    rng = np.random.RandomState(seed)
    rows = []
    for ev in range(n_events):
        n = rng.randint(4, 10)
        ttca = np.sort(rng.uniform(0.3, 6.5, n))[::-1]
        esc = rng.rand() < 0.2
        start = rng.uniform(-10.5, -7.0)
        slope = rng.uniform(0.6, 1.4) if (esc and signal) else rng.uniform(-0.6, 0.15)
        elapsed = ttca[0] - ttca
        risk = start + slope * elapsed + rng.normal(0, 0.15, n)
        md = rng.uniform(200, 5000) * np.exp(-0.1 * elapsed)
        for k in range(n):
            rows.append({"event_id": ev, "time_to_tca": ttca[k],
                         "miss_distance": md[k],
                         "relative_speed": rng.uniform(1000, 15000),
                         "risk": risk[k]})
    return pd.DataFrame(rows)


@pytest.fixture()
def model_dir(tmp_path, monkeypatch):
    d = tmp_path / "models"
    monkeypatch.setattr(config, "MODEL_DIR", d)
    return d


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train once on the synthetic CSV; yields (model_dir, meta)."""
    root = tmp_path_factory.mktemp("vigil_e")
    csv = root / "train_data.csv"
    synthetic_kelvins().to_csv(csv, index=False)
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "MODEL_DIR", root / "models")
    try:
        assert train_escalation.main(["--data", str(csv)]) == 0
        meta = json.loads(escalation.meta_path().read_text())
    finally:
        mp.undo()
    return root / "models", meta


@pytest.fixture()
def with_artifact(trained, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", trained[0])
    return trained[1]


def _cdms(risks, start="2026-07-20T00:00:00Z", step_h=12.0):
    from datetime import datetime, timedelta, timezone
    t0 = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return [{"message_time": (t0 + timedelta(hours=step_h * i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "miss_distance_m": 1000.0 - 50 * i, "relative_velocity_ms": 12000.0,
             "probability_of_collision": (None if r is None else (0.0 if r == "zero" else 10.0 ** r))}
            for i, r in enumerate(risks)]


# --- features ------------------------------------------------------------------


def test_escalation_features_extend_the_shared_contract():
    cdms = _cdms([-9.0, -8.0, -7.0])
    tca = "2026-07-22T00:00:00Z"
    feats = escalation.escalation_features(cdms, tca)
    assert list(ESCALATION_FEATURE_NAMES[:len(FEATURE_NAMES)]) == FEATURE_NAMES
    assert set(feats) == set(ESCALATION_FEATURE_NAMES)
    base = build_features(cdms, tca)
    for name in FEATURE_NAMES:          # same function in training and serving
        assert feats[name] == base[name]
    assert feats["log10_pc_latest"] == pytest.approx(-7.0)
    # slope +2 decades/day, 24 h to TCA -> projected -5
    assert feats["pc_log_slope"] == pytest.approx(2.0)
    assert feats["pc_log_projected_tca"] == pytest.approx(-5.0)
    assert feats["pc_log_drop_from_max"] == pytest.approx(0.0)


def test_escalation_features_degenerate_sequences():
    tca = "2026-07-22T00:00:00Z"
    empty = escalation.escalation_features([], tca)
    assert empty["log10_pc_latest"] == escalation.LOG10_PC_FLOOR
    assert empty["pc_log_drop_from_max"] == 0.0
    nulls = escalation.escalation_features(_cdms([None, "zero"]), tca)
    assert nulls["log10_pc_latest"] == escalation.LOG10_PC_FLOOR
    assert nulls["pc_log_projected_tca"] == escalation.LOG10_PC_FLOOR
    falling = escalation.escalation_features(_cdms([-5.0, -8.0]), tca)
    assert falling["pc_log_drop_from_max"] == pytest.approx(3.0)
    assert escalation.in_domain(falling)
    assert not escalation.in_domain(escalation.escalation_features(_cdms([-5.5]), tca))


# --- dataset + metrics ----------------------------------------------------------


def test_decision_points_only_below_threshold_and_before_final():
    df = pd.DataFrame([
        # event 1: -8, -7, -5 (final) -> escalates; 2 decision points
        {"event_id": 1, "time_to_tca": 3.0, "miss_distance": 900, "relative_speed": 1e4, "risk": -8.0},
        {"event_id": 1, "time_to_tca": 2.0, "miss_distance": 800, "relative_speed": 1e4, "risk": -7.0},
        {"event_id": 1, "time_to_tca": 1.0, "miss_distance": 700, "relative_speed": 1e4, "risk": -5.0},
        # event 2: -5 then -9 (final): one decision point? no - first is above
        {"event_id": 2, "time_to_tca": 3.0, "miss_distance": 900, "relative_speed": 1e4, "risk": -5.0},
        {"event_id": 2, "time_to_tca": 1.0, "miss_distance": 900, "relative_speed": 1e4, "risk": -9.0},
        # event 3: single CDM -> dropped
        {"event_id": 3, "time_to_tca": 1.0, "miss_distance": 900, "relative_speed": 1e4, "risk": -9.0},
    ])
    data = train_escalation.build_decision_points(df, -6.0)
    assert data["event_ids"] == [1]
    assert list(data["y"]) == [1, 1]
    assert list(data["point_ttca_days"]) == [3.0, 2.0]
    assert data["point_weight"].tolist() == [0.5, 0.5]
    assert data["event_cf_ttca_days"][0] == 1.0
    assert bool(data["event_lead_eligible"][0])
    assert data["X"].shape == (2, len(ESCALATION_FEATURE_NAMES))


def test_lead_time_is_measured_against_carry_forward():
    data = {"event_lead_eligible": np.array([True, True, False]),
            "event_cf_ttca_days": np.array([1.0, 0.5, np.nan])}
    pe = np.array([0, 0, 1, 1, 2])
    tt = np.array([3.0, 2.0, 2.0, 1.0, 2.0])
    prob = np.array([0.1, 0.9, 0.2, 0.3, 0.99])   # event 0 flagged at 2.0 d
    out = train_escalation.lead_time_metrics(prob, pe, tt, data, np.array([0, 1, 2]), 0.5)
    assert out["n_lead_eligible_test"] == 2
    assert out["median_lead_time_h"] == pytest.approx(24.0)      # (2.0-1.0) d
    assert out["median_lead_time_h_all"] == pytest.approx(12.0)  # median(24, 0)
    assert out["frac_flagged_early"] == pytest.approx(0.5)
    assert out["median_hours_before_tca_carry_forward"] == pytest.approx(18.0)


def test_split_is_event_level_seeded_and_disjoint():
    labels = np.array([True] * 20 + [False] * 80)
    a = train_escalation.split_events(labels, 42)
    b = train_escalation.split_events(labels, 42)
    assert all((x == y).all() for x, y in zip(a, b))
    fit, val, test = a
    assert not (set(fit) & set(val) or set(fit) & set(test) or set(val) & set(test))
    assert len(fit) + len(val) + len(test) == 100
    assert labels[test].sum() == 4          # stratified 20%


def _run(pr=0.6, lpr=0.5, lead=10.0, fe=0.4, lfe=0.3, val=True):
    return {"val_has_both_classes": val,
            "metrics": {"pr_auc": pr, "median_lead_time_h": lead, "frac_flagged_early": fe},
            "baselines": {"logistic_latest_risk": {"pr_auc": lpr, "frac_flagged_early": lfe}}}


def test_ship_gate_logic():
    good = [_run() for _ in range(3)]
    assert train_escalation.ship_gate(_run(), good)["passed"]
    for bad in (_run(pr=0.51), _run(lead=float("nan")), _run(lead=0.0),
                _run(fe=0.2), _run(val=False)):
        assert not train_escalation.ship_gate(bad, good)["passed"]
    # one unlucky seed where the model loses to the baseline blocks shipping
    assert not train_escalation.ship_gate(_run(), good[:2] + [_run(pr=0.4)])["passed"]
    names = [c["name"] for c in train_escalation.ship_gate(_run(), good)["checks"]]
    assert names == ["validation_has_both_classes", "pr_auc_beats_logistic",
                     "median_lead_time_positive", "early_recall_not_below_logistic",
                     "seed_check_beats_logistic_every_seed"]


# --- training CLI ------------------------------------------------------------------


def test_missing_csv_exits_zero_and_writes_nothing(model_dir, tmp_path, capsys):
    assert train_escalation.main(["--data", str(tmp_path / "nope.csv")]) == 0
    assert "zenodo" in capsys.readouterr().out.lower()
    assert not model_dir.exists()


def test_training_end_to_end_ships_on_planted_signal(trained):
    d, meta = trained
    assert (d / "vigil_e.json").exists()
    assert meta["version"] == escalation.MODEL_VERSION
    assert meta["features"] == ESCALATION_FEATURE_NAMES
    assert meta["gate"]["passed"] is True
    assert set(meta["metrics"]) == set(escalation.METRIC_KEYS)
    m = meta["metrics"]
    assert 0.0 <= m["roc_auc"] <= 1.0 and 0.0 <= m["pr_auc"] <= 1.0
    assert m["median_lead_time_h"] > 0
    assert m["n_events_test"] > 0 and m["n_escalating_test"] > 0
    assert set(meta["baselines"]) == {"logistic_latest_risk", "carry_forward"}
    assert meta["baselines"]["carry_forward"]["median_lead_time_h"] == 0.0
    assert [s["seed"] for s in meta["seed_check"]] == list(train_escalation.SEED_CHECK)
    assert meta["alert_budget"] == train_escalation.ALERT_BUDGET
    assert 0.0 <= meta["alert_threshold"] <= 1.0
    assert {"a", "b"} <= set(meta["calibration"])


def test_failed_gate_saves_nothing_and_removes_stale_artifact(model_dir, tmp_path, monkeypatch):
    csv = tmp_path / "train_data.csv"
    synthetic_kelvins(n_events=300, seed=3).to_csv(csv, index=False)
    model_dir.mkdir(parents=True)
    escalation.artifact_path().write_text("stale")
    escalation.meta_path().write_text("{}")
    monkeypatch.setattr(train_escalation, "GATE_PR_AUC_MARGIN", 10.0)  # unreachable
    report = tmp_path / "report.json"
    assert train_escalation.main(["--data", str(csv), "--report", str(report)]) == 0
    assert not escalation.artifact_path().exists()
    assert not escalation.meta_path().exists()
    rep = json.loads(report.read_text())
    assert rep["gate"]["passed"] is False


def test_serving_reproduces_training_probabilities(with_artifact):
    model = escalation.EscalationModel.load()
    assert model is not None and model.version == escalation.MODEL_VERSION
    df = synthetic_kelvins(n_events=40, seed=9)
    data = train_escalation.build_decision_points(df, -6.0)
    import xgboost as xgb
    margins = model.booster.predict(xgb.DMatrix(data["X"]), output_margin=True)
    expected = escalation.calibrate(margins, model.cal_a, model.cal_b)
    for row, p in zip(data["X"][:25], expected[:25]):
        feats = dict(zip(ESCALATION_FEATURE_NAMES, row))
        assert model.predict_proba(feats) == pytest.approx(float(p), rel=1e-6)
    expl = model.explain(dict(zip(ESCALATION_FEATURE_NAMES, data["X"][0])))
    total = expl["base_value"] + sum(r["contribution"] for r in expl["contributions"])
    assert total == pytest.approx(float(margins[0]), abs=1e-4)


# --- load / serving null paths ---------------------------------------------------------


def test_load_is_none_without_or_with_corrupt_artifact(model_dir):
    assert escalation.EscalationModel.load() is None
    model_dir.mkdir(parents=True)
    escalation.artifact_path().write_text("not a booster")
    escalation.meta_path().write_text(json.dumps({"version": "vigil-e-0.1"}))
    assert escalation.EscalationModel.load() is None
    assert escalation.score_cdms(None, _cdms([-8.0]), "2026-07-22T00:00:00Z") == (None, None)


def test_score_cdms_declines_above_threshold(with_artifact):
    model = escalation.EscalationModel.load()
    tca = "2026-07-22T00:00:00Z"
    p, v = escalation.score_cdms(model, _cdms([-9.0, -8.0]), tca)
    assert v == escalation.MODEL_VERSION and 0.0 <= p <= 1.0
    assert escalation.score_cdms(model, _cdms([-7.0, -5.0]), tca) == (None, escalation.MODEL_VERSION)


# --- DB migration + pipeline ---------------------------------------------------------------


def test_migration_adds_columns_to_old_sqlite_db(tmp_path):
    path = tmp_path / "old.db"
    raw = sqlite3.connect(path)
    raw.executescript("""
        CREATE TABLE risk_predictions (
            id INTEGER PRIMARY KEY, conjunction_id INTEGER NOT NULL,
            predicted_final_risk REAL NOT NULL, prediction_time TEXT NOT NULL,
            model_version TEXT NOT NULL);
        INSERT INTO risk_predictions VALUES (1, 1, 1e-5, '2026-07-21T00:00:00Z', 'vigil-0.4');
    """)
    raw.commit()
    raw.close()
    conn = db.get_conn(path)
    db.init_db(conn)
    db.init_db(conn)   # idempotent
    cols = db.table_columns(conn, "risk_predictions")
    assert cols[-2:] == ["escalation_prob", "escalation_model_version"]
    row = conn.execute("SELECT escalation_prob, escalation_model_version"
                       " FROM risk_predictions").fetchone()
    assert tuple(row) == (None, None)
    conn.close()


def _seed_events(conn):
    conn.executescript("""
    INSERT INTO satellites (id, norad_id, name, object_type, last_updated) VALUES
        (1, 25544, 'ISS (ZARYA)', 'PAYLOAD', '2026-07-21T00:00:00Z'),
        (2, 34454, 'COSMOS 2251 DEB', 'DEBRIS', '2026-07-21T00:00:00Z');
    INSERT INTO conjunctions (id, primary_object_id, secondary_object_id,
                              time_of_closest_approach, first_seen_at, status) VALUES
        (1, 1, 2, '2026-07-23T00:00:00Z', '2026-07-20T00:00:00Z', 'active'),
        (2, 2, 1, '2026-07-23T06:00:00Z', '2026-07-20T00:00:00Z', 'active');
    INSERT INTO cdm_messages (conjunction_id, message_time, miss_distance_m,
                              relative_velocity_ms, probability_of_collision, source) VALUES
        (1, '2026-07-20T00:00:00Z', 1500.0, 12000.0, 1e-9, 'screening'),
        (1, '2026-07-21T00:00:00Z', 1200.0, 12000.0, 1e-8, 'screening'),
        (2, '2026-07-20T00:00:00Z', 300.0, 12000.0, 1e-5, 'screening'),
        (2, '2026-07-21T00:00:00Z', 250.0, 12000.0, 2e-5, 'screening');
    """)
    conn.commit()


def test_pipeline_predict_active_null_without_artifact(tmp_path, model_dir):
    from backend.app.ingestion import pipeline
    conn = db.get_conn(tmp_path / "k.db")
    db.init_db(conn)
    _seed_events(conn)
    assert pipeline._predict_active(conn) == 2
    rows = conn.execute("SELECT escalation_prob, escalation_model_version"
                        " FROM risk_predictions").fetchall()
    assert [tuple(r) for r in rows] == [(None, None), (None, None)]
    conn.close()


def test_pipeline_predict_active_scores_with_artifact(tmp_path, with_artifact):
    from backend.app.ingestion import pipeline
    conn = db.get_conn(tmp_path / "k.db")
    db.init_db(conn)
    _seed_events(conn)
    assert pipeline._predict_active(conn) == 2
    rows = {r["conjunction_id"]: r for r in conn.execute(
        "SELECT conjunction_id, escalation_prob, escalation_model_version"
        " FROM risk_predictions").fetchall()}
    assert 0.0 <= rows[1]["escalation_prob"] <= 1.0          # below -6: scored
    assert rows[1]["escalation_model_version"] == escalation.MODEL_VERSION
    assert rows[2]["escalation_prob"] is None                # already >= -6
    assert rows[2]["escalation_model_version"] == escalation.MODEL_VERSION
    conn.close()


# --- API shapes ---------------------------------------------------------------------


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "api.db")
    monkeypatch.setattr(config, "RISK_MODEL_FILE", tmp_path / "none.json")
    monkeypatch.setattr(config, "RISK_MODEL_META_FILE", tmp_path / "none.meta.json")
    monkeypatch.delenv("KESSLER_DATABASE_URL", raising=False)
    monkeypatch.delenv("KESSLER_SNAPSHOT", raising=False)
    from backend.app.api import conjunctions as conjunctions_api
    monkeypatch.setattr(conjunctions_api, "now_iso", lambda: "2026-07-21T12:00:00Z")
    conn = db.get_conn(config.DB_PATH)
    db.init_db(conn)
    _seed_events(conn)
    conn.close()

    def score():
        from backend.app.ingestion import pipeline
        c = db.get_conn(config.DB_PATH)
        pipeline._predict_active(c)
        c.close()

    from backend.app.main import app
    with TestClient(app) as client:
        yield client, score


def test_api_null_path_without_artifact(api, model_dir):
    client, score = api
    score()
    items = client.get("/conjunctions").json()["items"]
    assert items and all(i["escalation_prob"] is None
                         and i["escalation_model_version"] is None for i in items)
    detail = client.get("/conjunctions/1").json()
    assert detail["escalation_prob"] is None and "escalation_model_version" in detail
    risk = client.get("/risk/1").json()
    assert risk["escalation_prob"] is None and risk["escalation_model_version"] is None
    card = client.get("/model/card").json()
    assert card["early_warning"] is None
    expl = client.get("/risk/1/escalation").json()
    assert expl["mode"] == "unavailable" and expl["escalation_prob"] is None
    assert client.get("/risk/999/escalation").status_code == 404


def test_api_with_artifact(api, with_artifact):
    client, score = api
    score()
    items = {i["id"]: i for i in client.get("/conjunctions").json()["items"]}
    assert 0.0 <= items[1]["escalation_prob"] <= 1.0
    assert items[1]["escalation_model_version"] == escalation.MODEL_VERSION
    assert items[2]["escalation_prob"] is None
    assert client.get("/conjunctions/1").json()["escalation_prob"] == items[1]["escalation_prob"]
    risk = client.get("/risk/1").json()
    assert risk["escalation_prob"] == items[1]["escalation_prob"]
    assert risk["escalation_model_version"] == escalation.MODEL_VERSION

    ew = client.get("/model/card").json()["early_warning"]
    meta = with_artifact
    assert ew["version"] == meta["version"]
    assert ew["metrics"] == meta["metrics"]           # verbatim from training
    assert ew["gate"] == meta["gate"] and ew["seed_check"] == meta["seed_check"]
    assert ew["alert_threshold"] == meta["alert_threshold"]
    assert ew["baselines"] == meta["baselines"]
    assert ew["live"]["predictions_total"] == 1 and ew["live"]["conjunctions_scored"] == 1
    assert all(r["name"] in ESCALATION_FEATURE_NAMES for r in ew["feature_importance"])

    expl = client.get("/risk/1/escalation").json()
    assert expl["mode"] == "early_warning"
    assert expl["escalation_prob"] == pytest.approx(items[1]["escalation_prob"])
    assert {r["name"] for r in expl["contributions"]} == set(ESCALATION_FEATURE_NAMES)
    assert client.get("/risk/2/escalation").json()["mode"] == "not_applicable"


# --- Postgres ------------------------------------------------------------------------------

PG_URL = os.environ.get("KESSLER_TEST_PG_URL")


@pytest.fixture()
def pg_conn(monkeypatch):
    """A fresh database on KESSLER_TEST_PG_URL's server, dropped afterwards."""
    if not PG_URL:
        pytest.skip("KESSLER_TEST_PG_URL not set (no Postgres server)")
    import psycopg
    name = f"kessler_t_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(PG_URL, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    url = PG_URL.rsplit("/", 1)[0] + "/" + name
    monkeypatch.setenv("KESSLER_DATABASE_URL", url)
    conn = db.get_conn()
    try:
        yield conn
    finally:
        conn.close()
        with psycopg.connect(PG_URL, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE {name} WITH (FORCE)")


def test_pg_new_schema_and_old_table_migration(pg_conn):
    pg_conn.execute("""CREATE TABLE risk_predictions (
        id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
        conjunction_id INTEGER NOT NULL,
        predicted_final_risk DOUBLE PRECISION NOT NULL,
        prediction_time TEXT NOT NULL, model_version TEXT NOT NULL)""")
    pg_conn.commit()
    db.init_db(pg_conn)
    db.init_db(pg_conn)
    cols = db.table_columns(pg_conn, "risk_predictions")
    assert cols[-2:] == ["escalation_prob", "escalation_model_version"]


def test_pg_scoring_and_summary_sql(pg_conn, with_artifact):
    db.init_db(pg_conn)
    for stmt in """
    INSERT INTO satellites (id, norad_id, name, object_type, last_updated) VALUES
        (1, 25544, 'ISS (ZARYA)', 'PAYLOAD', '2026-07-21T00:00:00Z'),
        (2, 34454, 'COSMOS 2251 DEB', 'DEBRIS', '2026-07-21T00:00:00Z');
    INSERT INTO conjunctions (id, primary_object_id, secondary_object_id,
        time_of_closest_approach, first_seen_at, status) VALUES
        (1, 1, 2, '2026-07-23T00:00:00Z', '2026-07-20T00:00:00Z', 'active');
    INSERT INTO cdm_messages (conjunction_id, message_time, miss_distance_m,
        relative_velocity_ms, probability_of_collision, source) VALUES
        (1, '2026-07-20T00:00:00Z', 1500.0, 12000.0, 1e-9, 'screening'),
        (1, '2026-07-21T00:00:00Z', 1200.0, 12000.0, 1e-8, 'screening')
    """.split(";"):
        pg_conn.execute(stmt)
    pg_conn.commit()
    out = risk_model.predict_for_conjunction(
        pg_conn, 1, escalation_model=escalation.EscalationModel.load())
    assert 0.0 <= out["escalation_prob"] <= 1.0
    from backend.app.api import conjunctions as conjunctions_api
    items = conjunctions_api.query_summaries(pg_conn, "c.id = ?", (1,))
    assert items[0]["escalation_prob"] == pytest.approx(out["escalation_prob"])
    from backend.app.api import model as model_api
    ew = model_api._early_warning_section(pg_conn)
    assert ew["live"]["predictions_total"] == 1
