"""API layer tests - hermetic: temp DB, stubbed ingestion/models modules,
no network. Shapes asserted against docs/CONTRACTS.md exactly.
"""
import sys
import types

import pytest
from fastapi.testclient import TestClient

from backend.app import config, db

# real ISS TLE (fixture string; propagate is stubbed so only shape matters)
ISS_TLE1 = "1 25544U 98067A   24007.58333333  .00016717  00000-0  30777-3 0  9994"
ISS_TLE2 = "2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.49560532430089"

STUB_ELEMENTS = {
    "epoch": "2026-07-21T06:11:52Z",
    "inclination_deg": 51.63,
    "raan_deg": 124.11,
    "eccentricity": 0.000731,
    "arg_perigee_deg": 83.2,
    "mean_anomaly_deg": 12.3,
    "mean_motion_rev_day": 15.5021,
    "period_min": 92.89,
    "apogee_km": 421.4,
    "perigee_km": 412.9,
    "semi_major_axis_km": 6795.3,
    "bstar": 3.0777e-4,
}

OBJECT_ITEM_KEYS = {
    "id", "norad_id", "name", "object_type", "intl_designator", "epoch",
    "inclination_deg", "raan_deg", "eccentricity", "arg_perigee_deg",
    "mean_motion_rev_day", "period_min", "apogee_km", "perigee_km",
    "last_updated",
}
SUMMARY_KEYS = {
    "id", "tca", "primary", "secondary", "miss_distance_m",
    "relative_velocity_ms", "pc_latest", "pc_source", "predicted_final_risk",
    "model_version", "risk_level", "cdm_count", "first_seen_at", "status",
    "pc_trend",
    "escalation_prob", "escalation_model_version",  # VIGIL-E (null w/o artifact)
}
SIDE_KEYS = {"id", "norad_id", "name", "object_type"}
FEATURE_KEYS = {
    "miss_distance_m", "relative_velocity_ms", "pc_latest", "time_to_tca_h",
    "cdm_count", "md_delta", "pc_log_delta", "md_slope",
}
STATUS_KEYS = {
    "catalog_size", "active_conjunctions", "elevated_count", "critical_count",
    "last_ingest_at", "model_version", "window_hours", "has_demo_data",
    "auto_refresh_interval_h", "last_run_id",  # scheduler/history keys
    "last_run_error",  # failed refresh surfaced (never a false all-clear)
    "snapshot",  # offline snapshot provenance (null unless loaded from one)
    "space_weather",  # live storm summary (null without NOAA data)
}


def _install_stubs(monkeypatch):
    """Stub concurrently-built modules so tests are hermetic + deterministic."""
    propagate = types.ModuleType("backend.app.ingestion.propagate")
    propagate.orbital_elements = lambda line1, line2: dict(STUB_ELEMENTS)
    propagate.sample_orbit = lambda line1, line2, n=181: [[7000.0, 0.0, 0.0]] * n

    pipeline = types.ModuleType("backend.app.ingestion.pipeline")
    pipeline.RefreshBusyError = type("RefreshBusyError", (RuntimeError,), {})

    def _refresh(conn, demo=False):
        return {
            "fetched": 12, "satellites_new": 2, "tles_added": 5,
            "conjunctions_new": 1, "cdms_added": 1, "predictions_made": 1,
            "duration_s": 0.1, "demo": demo,
        }

    pipeline.refresh = _refresh

    features = types.ModuleType("backend.app.models.features")

    def _build_features(cdms, tca_iso, as_of_iso=None):
        latest = cdms[-1] if cdms else {}
        return {
            "miss_distance_m": latest.get("miss_distance_m"),
            "relative_velocity_ms": latest.get("relative_velocity_ms"),
            "pc_latest": latest.get("probability_of_collision"),
            "time_to_tca_h": 38.2,
            "cdm_count": len(cdms),
            "md_delta": -120.1,
            "pc_log_delta": 1.45,
            "md_slope": -60.2,
        }

    features.build_features = _build_features

    import backend.app.ingestion as ingestion_pkg
    import backend.app.models as models_pkg

    monkeypatch.setitem(sys.modules, "backend.app.ingestion.propagate", propagate)
    monkeypatch.setitem(sys.modules, "backend.app.ingestion.pipeline", pipeline)
    monkeypatch.setitem(sys.modules, "backend.app.models.features", features)
    monkeypatch.setattr(ingestion_pkg, "propagate", propagate, raising=False)
    monkeypatch.setattr(ingestion_pkg, "pipeline", pipeline, raising=False)
    monkeypatch.setattr(models_pkg, "features", features, raising=False)
    return pipeline


def _seed(conn):
    conn.executescript(
        f"""
    INSERT INTO satellites (id, norad_id, name, object_type, intl_designator, last_updated) VALUES
        (1, 25544, 'ISS (ZARYA)', 'PAYLOAD', '1998-067A', '2026-07-21T13:02:41Z'),
        (2, 34454, 'COSMOS 2251 DEB', 'DEBRIS', NULL, '2026-07-21T13:02:41Z'),
        (3, 43556, 'AEROCUBE 12A', 'PAYLOAD', '2018-046G', '2026-07-21T13:02:41Z');

    INSERT INTO tle_history (satellite_id, epoch, line1, line2, ingested_at) VALUES
        (1, '2026-07-20T06:11:52Z', '{ISS_TLE1}', '{ISS_TLE2}', '2026-07-20T13:00:00Z'),
        (1, '2026-07-21T06:11:52Z', '{ISS_TLE1}', '{ISS_TLE2}', '2026-07-21T13:02:41Z'),
        (2, '2026-07-21T02:00:00Z', '{ISS_TLE1}', '{ISS_TLE2}', '2026-07-21T13:02:41Z');

    INSERT INTO conjunctions (id, primary_object_id, secondary_object_id,
                              time_of_closest_approach, first_seen_at, status) VALUES
        (1, 1, 2, '2026-07-23T04:12:09Z', '2026-07-20T10:00:00Z', 'active'),
        (2, 1, 3, '2026-07-22T10:00:00Z', '2026-07-21T06:00:00Z', 'active'),
        (3, 1, 2, '2026-07-18T00:00:00Z', '2026-07-16T00:00:00Z', 'past'),
        (4, 3, 2, '2026-07-22T00:00:00Z', '2026-07-21T07:00:00Z', 'active'),
        (5, 1, 2, '2026-07-23T20:00:00Z', '2026-07-21T09:00:00Z', 'active');

    -- c1 CDMs inserted newest-first to prove ordering is by message_time.
    INSERT INTO cdm_messages (conjunction_id, message_time, miss_distance_m,
                              relative_velocity_ms, probability_of_collision, source) VALUES
        (1, '2026-07-21T10:00:00Z', 812.4, 14210.3, 3.1e-05, 'screening'),
        (1, '2026-07-20T10:00:00Z', 1500.0, 14210.3, 1.1e-06, 'screening'),
        (1, '2026-07-20T22:00:00Z', 1100.0, 14210.3, 8.4e-06, 'demo'),
        (1, '2026-07-21T04:00:00Z', 950.0, 14210.3, NULL, 'screening'),
        (2, '2026-07-21T09:00:00Z', 400.0, 12000.0, 2.0e-04, 'screening'),
        (3, '2026-07-17T00:00:00Z', 5000.0, 10000.0, 1.0e-07, 'screening'),
        (4, '2026-07-21T08:00:00Z', 9000.0, 7000.0, NULL, 'screening');

    INSERT INTO risk_predictions (conjunction_id, predicted_final_risk,
                                  prediction_time, model_version) VALUES
        (1, 3.0e-05, '2026-07-20T23:00:00Z', 'heuristic-0.1'),
        (1, 4.4e-05, '2026-07-21T11:00:00Z', 'heuristic-0.1'),
        (2, 3.0e-04, '2026-07-21T11:00:00Z', 'heuristic-0.1');
    """
    )
    conn.commit()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    _install_stubs(monkeypatch)
    # Dev endpoints on by default; the gate tests flip the env per request.
    monkeypatch.setenv("KESSLER_DEV_ENDPOINTS", "1")
    # Freeze the honest-active cutoff so the fixture conjunctions (TCAs on
    # 2026-07-22/23) always read as active regardless of the real clock.
    from backend.app.api import conjunctions as conjunctions_api
    monkeypatch.setattr(conjunctions_api, "now_iso", lambda: "2026-07-21T13:10:00Z")
    conn = db.get_conn()
    db.init_db(conn)
    _seed(conn)
    conn.close()
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


# --- /status -----------------------------------------------------------------

def test_status_shape_and_counts(client):
    body = client.get("/status").json()
    assert set(body) == STATUS_KEYS
    assert body["catalog_size"] == 3
    assert body["active_conjunctions"] == 4
    assert body["elevated_count"] == 1   # c1
    assert body["critical_count"] == 1   # c2
    assert body["last_ingest_at"] == "2026-07-21T13:02:41Z"
    assert body["model_version"] == "heuristic-0.1"
    assert body["window_hours"] == config.SCREENING_WINDOW_HOURS
    assert body["has_demo_data"] is True


# --- /objects ----------------------------------------------------------------

def test_objects_listing_shape_and_order(client):
    body = client.get("/objects").json()
    assert set(body) == {"count", "items"}
    assert body["count"] == 3
    names = [item["name"] for item in body["items"]]
    assert names == ["AEROCUBE 12A", "COSMOS 2251 DEB", "ISS (ZARYA)"]

    iss = body["items"][2]
    assert set(iss) == OBJECT_ITEM_KEYS
    assert iss["norad_id"] == 25544
    assert iss["intl_designator"] == "1998-067A"
    assert iss["epoch"] == "2026-07-21T06:11:52Z"  # latest TLE epoch
    assert iss["inclination_deg"] == 51.63
    assert iss["apogee_km"] == 421.4
    # extra propagate keys are not leaked into the item
    assert "mean_anomaly_deg" not in iss and "bstar" not in iss


def test_objects_without_tle_have_null_elements(client):
    body = client.get("/objects?search=43556").json()
    assert body["count"] == 1
    item = body["items"][0]
    assert set(item) == OBJECT_ITEM_KEYS
    assert item["epoch"] is None
    assert item["inclination_deg"] is None
    assert item["period_min"] is None


def test_objects_search_numeric_exact(client):
    body = client.get("/objects?search=25544").json()
    assert body["count"] == 1
    assert body["items"][0]["norad_id"] == 25544


def test_objects_search_name_case_insensitive(client):
    body = client.get("/objects?search=cosmos").json()
    assert body["count"] == 1
    assert body["items"][0]["name"] == "COSMOS 2251 DEB"
    assert client.get("/objects?search=zzznope").json() == {"count": 0, "items": []}


def test_objects_pagination(client):
    body = client.get("/objects?limit=1&offset=1").json()
    assert body["count"] == 3  # count is total matching, not page size
    assert len(body["items"]) == 1
    assert body["items"][0]["name"] == "COSMOS 2251 DEB"


def test_object_detail(client):
    body = client.get("/objects/25544").json()
    assert set(body) == OBJECT_ITEM_KEYS | {
        "tle_history", "orbit_path", "conjunctions", "conjunction_count",
        "space_weather"}
    assert body["space_weather"] is None      # no live NOAA data in tests

    history = body["tle_history"]
    assert [set(t) for t in history] == [{"epoch", "line1", "line2", "ingested_at"}] * 2
    assert [t["epoch"] for t in history] == [
        "2026-07-21T06:11:52Z", "2026-07-20T06:11:52Z"]  # newest first
    assert history[0]["line1"] == ISS_TLE1

    assert len(body["orbit_path"]) == 181
    assert body["orbit_path"][0] == [7000.0, 0.0, 0.0]

    # active first, then risk desc: c2 (critical), c1 (elevated), c5 (nominal), past c3
    assert [c["id"] for c in body["conjunctions"]] == [2, 1, 5, 3]
    assert all(set(c) == SUMMARY_KEYS for c in body["conjunctions"])
    assert body["conjunction_count"] == 4  # true total (uncapped here)


def test_object_detail_no_tle(client):
    body = client.get("/objects/43556").json()
    assert body["tle_history"] == []
    assert body["orbit_path"] == []
    # c2 (secondary side, critical) then c4 (primary side, nominal)
    assert [c["id"] for c in body["conjunctions"]] == [2, 4]


def test_object_detail_404(client):
    response = client.get("/objects/99999")
    assert response.status_code == 404
    assert "detail" in response.json()


# --- /conjunctions -----------------------------------------------------------

def test_conjunctions_active_sorted_by_effective_risk(client):
    body = client.get("/conjunctions").json()
    assert set(body) == {"count", "items"}
    assert body["count"] == 4
    # eff pc: c2=3e-4, c1=4.4e-5, c4=0, c5=0 (tie broken by TCA asc)
    assert [item["id"] for item in body["items"]] == [2, 1, 4, 5]
    assert [item["risk_level"] for item in body["items"]] == [
        "critical", "elevated", "nominal", "nominal"]
    assert all(set(item) == SUMMARY_KEYS for item in body["items"])


def test_conjunction_summary_fields(client):
    items = {i["id"]: i for i in client.get("/conjunctions").json()["items"]}
    c1 = items[1]
    assert set(c1["primary"]) == SIDE_KEYS and set(c1["secondary"]) == SIDE_KEYS
    assert c1["primary"]["norad_id"] == 25544
    assert c1["secondary"] == {
        "id": 2, "norad_id": 34454, "name": "COSMOS 2251 DEB",
        "object_type": "DEBRIS"}
    assert c1["tca"] == "2026-07-23T04:12:09Z"
    assert c1["first_seen_at"] == "2026-07-20T10:00:00Z"
    assert c1["status"] == "active"
    # latest CDM values, latest prediction, trend ascending with nulls skipped
    assert c1["miss_distance_m"] == 812.4
    assert c1["relative_velocity_ms"] == 14210.3
    assert c1["pc_latest"] == 3.1e-05
    assert c1["pc_source"] == "assumed"  # no covariance_json on the latest CDM
    assert c1["predicted_final_risk"] == 4.4e-05
    assert c1["model_version"] == "heuristic-0.1"
    assert c1["cdm_count"] == 4
    assert c1["pc_trend"] == [1.1e-06, 8.4e-06, 3.1e-05]

    c5 = items[5]  # no CDMs, no predictions
    assert c5["miss_distance_m"] is None
    assert c5["pc_latest"] is None
    assert c5["pc_source"] == "assumed"  # no CDM at all -> assumed path
    assert c5["predicted_final_risk"] is None
    assert c5["model_version"] is None
    assert c5["cdm_count"] == 0
    assert c5["pc_trend"] == []
    assert c5["risk_level"] == "nominal"


def test_pc_source_covariance_when_latest_cdm_has_covariance(client):
    """pc_source flips to 'covariance' when the LATEST CDM carries one."""
    conn = db.get_conn()
    conn.execute(
        "UPDATE cdm_messages SET covariance_json = ?"
        " WHERE conjunction_id = 1 AND id = ("
        "   SELECT id FROM cdm_messages WHERE conjunction_id = 1"
        "   ORDER BY message_time DESC, id DESC LIMIT 1)",
        ('{"sat1_rtn": [[1,0,0],[0,1,0],[0,0,1]]}',))
    conn.commit()
    conn.close()
    items = {i["id"]: i for i in client.get("/conjunctions").json()["items"]}
    assert items[1]["pc_source"] == "covariance"
    assert items[2]["pc_source"] == "assumed"  # untouched event stays assumed


def test_conjunctions_status_filters(client):
    past = client.get("/conjunctions?status=past").json()
    assert past["count"] == 1
    assert past["items"][0]["id"] == 3

    everything = client.get("/conjunctions?status=all").json()
    assert everything["count"] == 5
    assert [i["id"] for i in everything["items"]] == [2, 1, 3, 4, 5]


def test_conjunctions_risk_filter(client):
    elevated = client.get("/conjunctions?risk=elevated").json()
    assert elevated["count"] == 1
    assert elevated["items"][0]["id"] == 1

    nominal = client.get("/conjunctions?risk=nominal").json()
    assert nominal["count"] == 2
    assert [i["id"] for i in nominal["items"]] == [4, 5]


def test_conjunctions_pagination(client):
    body = client.get("/conjunctions?limit=2&offset=1").json()
    assert body["count"] == 4  # total after filters, not page size
    assert [i["id"] for i in body["items"]] == [1, 4]


def test_conjunction_detail(client):
    body = client.get("/conjunctions/1").json()
    assert set(body) == SUMMARY_KEYS | {"cdms", "risk_predictions"}

    cdms = body["cdms"]
    assert [set(c) for c in cdms] == [{
        "message_time", "miss_distance_m", "relative_velocity_ms",
        "probability_of_collision", "source"}] * 4
    assert [c["message_time"] for c in cdms] == [
        "2026-07-20T10:00:00Z", "2026-07-20T22:00:00Z",
        "2026-07-21T04:00:00Z", "2026-07-21T10:00:00Z"]  # ascending
    assert cdms[2]["probability_of_collision"] is None
    assert cdms[1]["source"] == "demo"

    predictions = body["risk_predictions"]
    assert [set(p) for p in predictions] == [{
        "predicted_final_risk", "prediction_time", "model_version"}] * 2
    assert [p["predicted_final_risk"] for p in predictions] == [3.0e-05, 4.4e-05]


def test_conjunction_detail_404(client):
    response = client.get("/conjunctions/999")
    assert response.status_code == 404
    assert "detail" in response.json()


# --- /risk -------------------------------------------------------------------

def test_risk_endpoint(client):
    body = client.get("/risk/1").json()
    assert set(body) == {
        "conjunction_id", "risk_level", "predicted_final_risk", "pc_latest",
        "model_version", "prediction_time", "features",
        "escalation_prob", "escalation_model_version"}
    assert body["conjunction_id"] == 1
    assert body["risk_level"] == "elevated"
    assert body["predicted_final_risk"] == 4.4e-05
    assert body["pc_latest"] == 3.1e-05
    assert body["model_version"] == "heuristic-0.1"
    assert body["prediction_time"] == "2026-07-21T11:00:00Z"

    features = body["features"]
    assert set(features) == FEATURE_KEYS
    # stub derives these from the cdms passed in -> proves ascending order + keys
    assert features["cdm_count"] == 4
    assert features["miss_distance_m"] == 812.4
    assert features["pc_latest"] == 3.1e-05


def test_risk_without_prediction_falls_back_to_pc_latest(client):
    body = client.get("/risk/2").json()
    assert body["risk_level"] == "critical"

    no_data = client.get("/risk/5").json()  # no CDMs, no predictions
    assert no_data["risk_level"] == "nominal"
    assert no_data["predicted_final_risk"] is None
    assert no_data["pc_latest"] is None
    assert no_data["model_version"] is None
    assert no_data["prediction_time"] is None
    assert set(no_data["features"]) == FEATURE_KEYS


def test_risk_404(client):
    response = client.get("/risk/999")
    assert response.status_code == 404
    assert "detail" in response.json()


# --- /ingest/refresh ---------------------------------------------------------

def test_ingest_refresh(client):
    body = client.post("/ingest/refresh").json()
    assert body["fetched"] == 12
    assert body["demo"] is False

    assert client.post("/ingest/refresh?demo=true").json()["demo"] is True


def test_ingest_refresh_failure_is_502(client, monkeypatch):
    pipeline = sys.modules["backend.app.ingestion.pipeline"]

    def _boom(conn, demo=False):
        raise RuntimeError("celestrak unreachable")

    monkeypatch.setattr(pipeline, "refresh", _boom)
    response = client.post("/ingest/refresh")
    assert response.status_code == 502
    assert "celestrak unreachable" in response.json()["detail"]


def test_ingest_refresh_busy_is_409(client, monkeypatch):
    """pipeline's lock-held signal (RefreshBusyError) maps to a 409."""
    pipeline = sys.modules["backend.app.ingestion.pipeline"]

    def _busy(conn, demo=False):
        raise pipeline.RefreshBusyError("a refresh is already running")

    monkeypatch.setattr(pipeline, "refresh", _busy)
    response = client.post("/ingest/refresh")
    assert response.status_code == 409
    assert response.json() == {"detail": "a refresh is already running"}


def test_ingest_refresh_dev_gate_off_404(client, monkeypatch):
    """Without a truthy KESSLER_DEV_ENDPOINTS the route pretends not to exist."""
    monkeypatch.delenv("KESSLER_DEV_ENDPOINTS", raising=False)
    response = client.post("/ingest/refresh")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not found"}
    monkeypatch.setenv("KESSLER_DEV_ENDPOINTS", "no")  # non-truthy value
    assert client.post("/ingest/refresh").status_code == 404
    # read per request: flipping the env re-enables without a reload
    monkeypatch.setenv("KESSLER_DEV_ENDPOINTS", "1")
    assert client.post("/ingest/refresh").status_code == 200


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_ingest_refresh_dev_gate_truthy_values(client, monkeypatch, value):
    monkeypatch.setenv("KESSLER_DEV_ENDPOINTS", value)
    assert client.post("/ingest/refresh").status_code == 200


def test_conjunctions_early_warning_filter(client, monkeypatch):
    """early_warning=true lists only events whose VIGIL-E probability meets
    the model's own alert threshold; with no model meta the list is empty."""
    from backend.app.api import conjunctions as conjunctions_api

    ids = [it["id"] for it in client.get("/conjunctions").json()["items"]]
    assert ids
    conn = db.get_conn(config.DB_PATH)
    conn.execute("UPDATE risk_predictions SET escalation_prob = 0.9,"
                 " escalation_model_version = 'vigil-e-test'"
                 " WHERE conjunction_id = ?", (ids[0],))
    conn.commit()
    conn.close()

    monkeypatch.setattr(conjunctions_api, "_early_warning_threshold", lambda: (None, None))
    assert client.get("/conjunctions?early_warning=true").json()["count"] == 0

    monkeypatch.setattr(conjunctions_api, "_early_warning_threshold",
                        lambda: (0.5, "vigil-e-test"))
    body = client.get("/conjunctions?early_warning=true").json()
    nominal = {it["id"] for it in client.get("/conjunctions").json()["items"]
               if it["risk_level"] == "nominal"}
    assert [it["id"] for it in body["items"]] == ([ids[0]] if ids[0] in nominal else [])
    # a different (newer) model version: stale probabilities are not shown
    monkeypatch.setattr(conjunctions_api, "_early_warning_threshold",
                        lambda: (0.5, "vigil-e-next"))
    assert client.get("/conjunctions?early_warning=true").json()["count"] == 0
