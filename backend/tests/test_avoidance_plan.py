"""One-click avoidance plan - POST /conjunctions/{id}/avoidance-plan.

Hermetic: temp DB, synthetic TLEs, real SGP4 + the two-body maneuver sandbox.

Geometry: two circular-ish LEO orbits with the same inclination and mean
motion but nodes 4 deg apart, phased so they meet at an orbit crossing ~6.5 h
after the wall clock (miss a few hundred meters, ~0.4 km/s closing speed).
Equal inclination means equal J2 node drift, so the geometry holds over the
12 h the planner looks back. With TCA ~6.5 h out the 1-orbit, 3 h and 6 h
candidate burns are live and the 12 h one is skipped (already past).
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import config, db
from backend.app.ingestion import propagate
from backend.app.models import maneuver as maneuver_mod
from backend.tests.test_ingestion import make_tle

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
EPOCH = datetime.now(timezone.utc).replace(microsecond=0)
EPOCH_ISO = EPOCH.strftime(ISO_FMT)

A_L1, A_L2 = make_tle(92001, 51.6, 100.0, 0.0005, 0.0, 200.00, 15.5, EPOCH)
B_L1, B_L2 = make_tle(92002, 51.6, 104.0, 0.0005, 0.0, 197.51, 15.5, EPOCH)


def _parse(s):
    return datetime.strptime(s, ISO_FMT).replace(tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def tca():
    """Closest approach in the 6-7.6 h window: 1 s scan, then its second."""
    times = [EPOCH + timedelta(seconds=s) for s in range(6 * 3600, int(7.6 * 3600))]
    ra, _ = propagate.positions_at(A_L1, A_L2, times)
    rb, _ = propagate.positions_at(B_L1, B_L2, times)
    d = np.linalg.norm(ra - rb, axis=1)
    i = int(np.argmin(d))
    assert d[i] < 1.0, "fixture geometry drifted: no sub-km approach"
    return times[i]


def _seed(conn, tca):
    tca_iso = tca.strftime(ISO_FMT)
    off_iso = (tca + timedelta(minutes=20)).strftime(ISO_FMT)
    past_iso = (EPOCH - timedelta(hours=1)).strftime(ISO_FMT)
    soon_iso = (datetime.now(timezone.utc) + timedelta(minutes=8)).strftime(ISO_FMT)
    conn.executescript(f"""
    INSERT INTO satellites (id, norad_id, name, object_type, last_updated) VALUES
        (1, 92001, 'PLAN PRIMARY', 'PAYLOAD', '{EPOCH_ISO}'),
        (2, 92002, 'PLAN SECONDARY', 'DEBRIS', '{EPOCH_ISO}'),
        (3, 92003, 'CORRUPT', 'PAYLOAD', '{EPOCH_ISO}');
    INSERT INTO tle_history (satellite_id, epoch, line1, line2, ingested_at) VALUES
        (1, '{EPOCH_ISO}', '{A_L1}', '{A_L2}', '{EPOCH_ISO}'),
        (2, '{EPOCH_ISO}', '{B_L1}', '{B_L2}', '{EPOCH_ISO}'),
        (3, '{EPOCH_ISO}', 'CORRUPT LINE ONE', 'CORRUPT LINE TWO', '{EPOCH_ISO}');
    INSERT INTO conjunctions (id, primary_object_id, secondary_object_id,
                              time_of_closest_approach, first_seen_at, status) VALUES
        (1, 1, 2, '{tca_iso}', '{EPOCH_ISO}', 'active'),
        (2, 1, 2, '{past_iso}', '{EPOCH_ISO}', 'active'),
        (3, 1, 2, '{soon_iso}', '{EPOCH_ISO}', 'active'),
        (4, 1, 2, '{off_iso}', '{EPOCH_ISO}', 'active'),
        (5, 3, 2, '{tca_iso}', '{EPOCH_ISO}', 'active');
    INSERT INTO cdm_messages (conjunction_id, message_time, miss_distance_m,
                              relative_velocity_ms, probability_of_collision) VALUES
        (4, '{EPOCH_ISO}', 150.0, 400.0, 1e-4);
    """)
    conn.commit()


@pytest.fixture()
def client(tmp_path, monkeypatch, tca):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "plan.db")
    conn = db.get_conn()
    db.init_db(conn)
    _seed(conn, tca)
    conn.close()
    from backend.app.main import app

    with TestClient(app) as c:
        yield c


def _plan(client, cid, body=None):
    return client.post(f"/conjunctions/{cid}/avoidance-plan", json=body)


@pytest.fixture()
def plan(client):
    r = _plan(client, 1)
    assert r.status_code == 200, r.text
    return r.json()


def test_plan_shape_and_method_label(plan):
    assert plan["conjunction_id"] == 1
    assert plan["status"] == "burn_recommended"
    assert plan["method"] == ("two-body differential, along-track, "
                              "screening-grade; not an operational maneuver plan")
    for key in ("baseline", "criteria", "recommended", "options", "skipped",
                "assumptions", "tca", "computed_at", "note"):
        assert key in plan
    crit = plan["criteria"]
    assert crit["target_miss_m"] == 1000.0
    assert crit["pc_threshold"] == config.PC_ELEVATED
    assert crit["required_miss_m"] >= max(1000.0, crit["pc_required_miss_m"] or 0)
    assert plan["baseline"]["miss_m"] < crit["required_miss_m"]
    assert plan["baseline"]["safe"] is False


def test_candidate_leads_and_skips(plan, tca):
    labels = [o["lead_label"] for o in plan["options"]]
    assert labels == ["1 orbit", "3 h", "6 h"]
    assert [s["lead_label"] for s in plan["skipped"]] == ["12 h"]
    now = datetime.now(timezone.utc)
    for o in plan["options"]:
        burn = _parse(o["burn_iso"])
        assert abs((tca - burn).total_seconds() - o["lead_s"]) <= 1.0
        assert (burn - now).total_seconds() >= 240   # >= 5 min at plan time
    orbit = plan["options"][0]["lead_s"]
    assert 5400 < orbit < 5800   # 15.5 rev/day ~ 92.9 min


def test_every_option_is_safe_and_minimal(plan, client):
    from backend.app.api import conjunctions as conj_api

    required = plan["criteria"]["required_miss_m"]
    conn = db.get_conn()
    try:
        for o in plan["options"]:
            assert o["reachable"] is True
            assert o["new_miss_m"] >= required - 0.1
            assert o["meets_pc"] is True and o["new_pc"] < config.PC_ELEVATED
            assert o["meets_miss"] is True
            assert o["direction"] in ("prograde", "retrograde")
            assert (o["dv_ms"] > 0) == (o["direction"] == "prograde")
            # minimality: 3 tolerance steps smaller, both signs, falls short
            ctx = conj_api.load_maneuver_context(conn, 1, _parse(o["burn_iso"]))
            smaller = abs(o["dv_ms"]) - 0.003
            for sign in (1.0, -1.0):
                miss = conj_api.evaluate_maneuver(ctx, sign * smaller)["new_miss_m"]
                assert miss < required
    finally:
        conn.close()


def test_recommendation_rule(plan):
    rec = plan["recommended"]
    assert rec is not None and rec in plan["options"]
    smallest = min(abs(o["dv_ms"]) for o in plan["options"])
    assert abs(rec["dv_ms"]) <= smallest + maneuver_mod.PLAN_TIE_MS
    ties = [o for o in plan["options"]
            if abs(o["dv_ms"]) - smallest <= maneuver_mod.PLAN_TIE_MS]
    assert rec["lead_s"] == min(o["lead_s"] for o in ties)
    # longer lead -> smaller burn for an along-track phasing change
    dvs = [abs(o["dv_ms"]) for o in plan["options"]]
    assert dvs[0] > dvs[-1]
    assert rec["propellant_g_per_100kg"] == pytest.approx(
        maneuver_mod.propellant_g(rec["dv_ms"]), abs=0.01)


def test_applying_recommendation_reproduces_via_maneuver(plan, client):
    rec = plan["recommended"]
    r = client.post("/conjunctions/1/maneuver",
                    json={"dv_along_ms": rec["dv_ms"], "burn_iso": rec["burn_iso"]})
    assert r.status_code == 200
    body = r.json()
    # dv is rounded to 0.1 mm/s in the plan: the recompute agrees to meters
    assert body["new_miss_m"] == pytest.approx(rec["new_miss_m"], abs=5.0)
    assert body["new_pc"] < config.PC_ELEVATED


def test_custom_target_raises_required_miss(client, plan):
    r = _plan(client, 1, {"target_miss_m": 5000})
    assert r.status_code == 200
    body = r.json()
    assert body["criteria"]["required_miss_m"] >= 5000
    assert body["recommended"]["new_miss_m"] >= 5000 - 0.1
    assert abs(body["recommended"]["dv_ms"]) > abs(plan["recommended"]["dv_ms"])


def test_unreachable_under_tiny_dv_cap(client):
    r = _plan(client, 1, {"max_dv_ms": 0.001})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "unreachable"
    assert body["recommended"] is None
    assert body["note"]
    assert all(o["reachable"] is False for o in body["options"])


def test_no_burn_needed_when_already_safe(client, monkeypatch):
    monkeypatch.setattr(config, "PC_ELEVATED", 1.0)   # Pc criterion trivially met
    r = _plan(client, 1, {"target_miss_m": 1.0})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "no_burn_needed"
    assert body["recommended"] is None and body["options"] == []


def test_geometry_mismatch_is_not_an_all_clear(client):
    """TLEs say ~thousands of km at the stored time, the latest message says
    150 m: the planner must not report 'no burn needed'."""
    r = _plan(client, 4)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "geometry_mismatch"
    assert body["recommended"] is None
    assert body["baseline"]["screened_miss_m"] == 150.0
    assert body["note"]


def test_errors(client):
    assert _plan(client, 999999).status_code == 404
    past = _plan(client, 2)
    assert past.status_code == 409 and "passed" in past.json()["detail"]
    soon = _plan(client, 3)
    assert soon.status_code == 409 and "too soon" in soon.json()["detail"]
    assert _plan(client, 1, {"target_miss_m": -5}).status_code == 422
    assert _plan(client, 1, {"target_miss_m": 1e9}).status_code == 422
    assert _plan(client, 1, {"max_dv_ms": 9}).status_code == 422
    assert _plan(client, 1, {"max_dv_ms": 0}).status_code == 422
    corrupt = _plan(client, 5)
    assert corrupt.status_code == 422 and "TLE" in corrupt.json()["detail"]


def test_model_layer_validation(tmp_path, tca):
    conn = db.get_conn(tmp_path / "m.db")
    db.init_db(conn)
    _seed(conn, tca)
    try:
        with pytest.raises(ValueError, match="target_miss_m"):
            maneuver_mod.plan_avoidance(conn, 1, target_miss_m=float("nan"))
        with pytest.raises(ValueError, match="max_dv_ms"):
            maneuver_mod.plan_avoidance(conn, 1, max_dv_ms=-1)
        with pytest.raises(LookupError):
            maneuver_mod.plan_avoidance(conn, 424242)
        with pytest.raises(maneuver_mod.PlanConflict):
            maneuver_mod.plan_avoidance(conn, 2)
    finally:
        conn.close()


def test_propellant_helper():
    assert maneuver_mod.propellant_g(0.0) == 0.0
    # 1 m/s on 100 kg at Isp 220 s: m*(1-exp(-1/2157.5)) ~ 46.3 g
    assert maneuver_mod.propellant_g(1.0) == pytest.approx(46.34, abs=0.05)
    assert maneuver_mod.propellant_g(-1.0) == maneuver_mod.propellant_g(1.0)
