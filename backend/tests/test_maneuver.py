"""Maneuver-sandbox tests - hermetic: temp DB, synthetic TLEs, real SGP4 +
two-body propagation, no network.

Geometry: the test_ephemeris crossing pair (same node line, 5 deg inclination
split, small mean-anomaly offset) shifted so the close approach happens ~41
minutes after the wall clock (MA ~200 deg -> 160 deg to the ascending node at
15.5 rev/day). The endpoint validates burn times against the real clock
(>= now - 1 h and >= 5 min before TCA), so a genuinely future TCA is required
for the happy paths.

models/maneuver.py round-trip invariants are covered here too (the module has
no other test home): rv -> elements -> rv reconstruction and the exact
cancellation of the two-body differential at dv = 0.
"""
import math
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

# Crossing pair ~160 deg of mean anomaly before the shared ascending node:
# close approach ~41 min after EPOCH (= wall-clock now), miss ~2.4 km.
A_L1, A_L2 = make_tle(91001, 51.6, 100.0, 0.0005, 0.0, 200.00, 15.5, EPOCH)
B_L1, B_L2 = make_tle(91002, 56.6, 100.0, 0.0005, 0.0, 200.02, 15.5, EPOCH)

RESPONSE_KEYS = {
    "conjunction_id", "dv_along_ms", "burn_iso", "new_miss_m", "new_tca",
    "delta_miss_m", "new_pc", "old", "primary_rel_offset_km", "method",
}


def _parse_iso(s: str) -> datetime:
    return datetime.strptime(s, ISO_FMT).replace(tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def coarse_tca():
    """Coarse TCA: argmin separation on a 30 s grid over the first 70 min."""
    times = [EPOCH + timedelta(seconds=30 * i) for i in range(141)]
    ra, _va = propagate.positions_at(A_L1, A_L2, times)
    rb, _vb = propagate.positions_at(B_L1, B_L2, times)
    i = int(np.argmin(np.linalg.norm(ra - rb, axis=1)))
    assert 0 < i < len(times) - 1
    tca = times[i]
    # the endpoint needs valid burn times: TCA must still be > 5 min out
    assert (tca - datetime.now(timezone.utc)).total_seconds() > 900
    return tca


def _seed_db(conn, tca_iso):
    """Seed the crossing pair + a corrupt-TLE pair (shared by both fixtures)."""
    conn.executescript(
        f"""
    INSERT INTO satellites (id, norad_id, name, object_type, last_updated) VALUES
        (1, 91001, 'MNV PRIMARY',   'PAYLOAD', '{EPOCH_ISO}'),
        (2, 91002, 'MNV SECONDARY', 'DEBRIS',  '{EPOCH_ISO}'),
        (3, 91003, 'CORRUPT SAT',   'PAYLOAD', '{EPOCH_ISO}');

    INSERT INTO tle_history (satellite_id, epoch, line1, line2, ingested_at) VALUES
        (1, '{EPOCH_ISO}', '{A_L1}', '{A_L2}', '{EPOCH_ISO}'),
        (2, '{EPOCH_ISO}', '{B_L1}', '{B_L2}', '{EPOCH_ISO}'),
        (3, '{EPOCH_ISO}', 'CORRUPT LINE ONE', 'CORRUPT LINE TWO', '{EPOCH_ISO}');

    INSERT INTO conjunctions (id, primary_object_id, secondary_object_id,
                              time_of_closest_approach, first_seen_at, status) VALUES
        (1, 1, 2, '{tca_iso}', '{EPOCH_ISO}', 'active'),
        (2, 1, 3, '{tca_iso}', '{EPOCH_ISO}', 'active');
    """
    )
    conn.commit()


@pytest.fixture()
def client(tmp_path, monkeypatch, coarse_tca):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    conn = db.get_conn()
    db.init_db(conn)
    _seed_db(conn, coarse_tca.strftime(ISO_FMT))
    conn.close()
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sandbox_conn(tmp_path, coarse_tca):
    """Direct-connection twin of `client` for models-layer optimizer tests."""
    conn = db.get_conn(tmp_path / "opt.db")
    db.init_db(conn)
    _seed_db(conn, coarse_tca.strftime(ISO_FMT))
    yield conn
    conn.close()


def _post(client, cid, dv, burn_iso):
    return client.post(f"/conjunctions/{cid}/maneuver",
                       json={"dv_along_ms": dv, "burn_iso": burn_iso})


# --- happy path --------------------------------------------------------------

def test_response_shape(client, coarse_tca):
    burn_iso = (coarse_tca - timedelta(seconds=600)).strftime(ISO_FMT)
    response = _post(client, 1, 2.0, burn_iso)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESPONSE_KEYS
    assert body["conjunction_id"] == 1
    assert body["dv_along_ms"] == 2.0
    assert body["burn_iso"] == burn_iso
    assert set(body["old"]) == {"miss_m", "pc"}
    assert body["new_miss_m"] > 0.0
    assert body["old"]["miss_m"] > 0.0
    assert body["delta_miss_m"] == pytest.approx(
        body["new_miss_m"] - body["old"]["miss_m"], abs=0.2)
    assert 0.0 <= body["new_pc"] <= 1.0
    assert 0.0 <= body["old"]["pc"] <= 1.0
    assert "two-body" in body["method"]
    off = body["primary_rel_offset_km"]
    assert len(off) == 3 and all(math.isfinite(c) for c in off)
    # new TCA stays inside the +/-300 s search window around the stored TCA
    dt = abs((_parse_iso(body["new_tca"]) - coarse_tca).total_seconds())
    assert dt <= 300.0


def test_dv_sign_changes_the_encounter(client, coarse_tca):
    """+2 and -2 m/s both move the miss, and move it differently."""
    burn_iso = (coarse_tca - timedelta(seconds=600)).strftime(ISO_FMT)
    plus = _post(client, 1, 2.0, burn_iso).json()
    minus = _post(client, 1, -2.0, burn_iso).json()
    # A 2 m/s along-track burn ~10 min ahead displaces km-scale: both bite.
    assert abs(plus["delta_miss_m"]) > 10.0
    assert abs(minus["delta_miss_m"]) > 10.0
    assert plus["delta_miss_m"] != minus["delta_miss_m"]
    assert plus["new_miss_m"] != minus["new_miss_m"]
    # opposite burns push the primary to opposite sides of the coast path
    dot = sum(p * m for p, m in zip(plus["primary_rel_offset_km"],
                                    minus["primary_rel_offset_km"]))
    assert dot < 0.0
    # both share the same unmaneuvered baseline
    assert plus["old"] == minus["old"]


def test_dv_zero_is_exact_null_maneuver(client, coarse_tca):
    """dv=0: the two two-body propagations cancel exactly."""
    burn_iso = (coarse_tca - timedelta(seconds=600)).strftime(ISO_FMT)
    body = _post(client, 1, 0.0, burn_iso).json()
    assert body["delta_miss_m"] == 0.0
    assert body["new_miss_m"] == body["old"]["miss_m"]
    assert body["primary_rel_offset_km"] == [0.0, 0.0, 0.0]


# --- validation --------------------------------------------------------------

def test_burn_too_close_to_tca_422(client, coarse_tca):
    burn_iso = (coarse_tca - timedelta(seconds=60)).strftime(ISO_FMT)
    response = _post(client, 1, 2.0, burn_iso)
    assert response.status_code == 422
    assert "5 minutes" in response.json()["detail"]


def test_burn_too_far_in_past_422(client):
    burn_iso = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(ISO_FMT)
    response = _post(client, 1, 2.0, burn_iso)
    assert response.status_code == 422
    assert "1 hour" in response.json()["detail"]


def test_dv_out_of_range_422(client, coarse_tca):
    burn_iso = (coarse_tca - timedelta(seconds=600)).strftime(ISO_FMT)
    assert _post(client, 1, 9.0, burn_iso).status_code == 422
    assert _post(client, 1, -5.1, burn_iso).status_code == 422


def test_bad_burn_iso_422(client):
    response = _post(client, 1, 2.0, "not-a-timestamp")
    assert response.status_code == 422
    assert "burn_iso" in response.json()["detail"]


def test_unknown_conjunction_404(client, coarse_tca):
    burn_iso = (coarse_tca - timedelta(seconds=600)).strftime(ISO_FMT)
    assert _post(client, 999999, 2.0, burn_iso).status_code == 404


def test_corrupt_tle_degrades_to_422(client, coarse_tca):
    burn_iso = (coarse_tca - timedelta(seconds=600)).strftime(ISO_FMT)
    response = _post(client, 2, 2.0, burn_iso)
    assert response.status_code == 422
    assert "TLE" in response.json()["detail"]


# --- models/maneuver invariants ----------------------------------------------

def test_rv_elements_round_trip():
    r0, v0 = propagate.positions_at(A_L1, A_L2, [EPOCH])
    r0, v0 = np.asarray(r0[0]), np.asarray(v0[0])
    elements = maneuver_mod.rv_to_elements(r0, v0)
    r1, v1 = maneuver_mod.elements_to_rv(elements)
    assert float(np.linalg.norm(r1 - r0)) < 1e-6   # km: sub-millimeter
    assert float(np.linalg.norm(v1 - v0)) < 1e-9


def test_apply_maneuver_prograde_raises_energy():
    burn_iso = (EPOCH + timedelta(seconds=300)).strftime(ISO_FMT)
    coast = maneuver_mod.apply_maneuver(A_L1, A_L2, burn_iso, 0.0)
    burned = maneuver_mod.apply_maneuver(A_L1, A_L2, burn_iso, 2.0)
    assert burned["a_km"] > coast["a_km"]  # prograde burn raises the orbit
    retro = maneuver_mod.apply_maneuver(A_L1, A_L2, burn_iso, -2.0)
    assert retro["a_km"] < coast["a_km"]


# --- optimize_dv ---------------------------------------------------------------
# Burn ~35 min before TCA: enough lead for km-scale authority per m/s while
# keeping the burn comfortably inside the endpoint's now-1h validity window.

OPT_LEAD_S = 2100
OPT_KEYS = {"conjunction_id", "target_miss_m", "dv_ms", "achieved_miss_m",
            "burn_iso", "evaluations", "reachable", "method"}


def _opt_burn_iso(coarse_tca):
    return (coarse_tca - timedelta(seconds=OPT_LEAD_S)).strftime(ISO_FMT)


def test_optimize_reachable_finds_minimal_dv(sandbox_conn, coarse_tca):
    """Reachable target: achieved >= target at the global min |dv| (both signs).

    Minimality is verified against a 0.05 m/s brute-force scan of both signs
    through the same evaluator the optimizer uses - if the optimizer searched
    only one sign, or stopped short of the crossing, this fails.
    """
    from backend.app.api import conjunctions as conj_api

    burn_iso = _opt_burn_iso(coarse_tca)
    burn = _parse_iso(burn_iso)
    ctx = conj_api.load_maneuver_context(sandbox_conn, 1, burn)
    base = conj_api.evaluate_maneuver(ctx, 0.0)["new_miss_m"]
    reach = max(conj_api.evaluate_maneuver(ctx, 5.0)["new_miss_m"],
                conj_api.evaluate_maneuver(ctx, -5.0)["new_miss_m"])
    assert reach > base + 1000.0  # precondition: km-scale authority at 5 m/s
    target = base + 0.5 * (reach - base)

    res = maneuver_mod.optimize_dv(sandbox_conn, 1, target, burn_iso)
    assert set(res) == OPT_KEYS
    assert res["conjunction_id"] == 1
    assert res["burn_iso"] == burn_iso
    assert res["reachable"] is True
    assert res["achieved_miss_m"] >= target - 0.05  # rounded to 0.1 m
    assert 0.0 < abs(res["dv_ms"]) <= 5.0
    assert res["evaluations"] >= 3
    assert "golden-section" in res["method"]

    # brute-force minimal feasible |dv| over both signs at 0.05 m/s steps
    feasible = [
        u
        for u in np.arange(0.05, 5.0 + 1e-9, 0.05)
        for sign in (1.0, -1.0)
        if conj_api.evaluate_maneuver(ctx, sign * u)["new_miss_m"] >= target
    ]
    assert feasible, "brute force must agree the target is reachable"
    assert abs(res["dv_ms"]) <= min(feasible) + 0.015  # tolerance + rounding


def test_optimize_target_already_met_needs_no_burn(sandbox_conn, coarse_tca):
    res = maneuver_mod.optimize_dv(sandbox_conn, 1, 1.0, _opt_burn_iso(coarse_tca))
    assert res["reachable"] is True
    assert res["dv_ms"] == 0.0
    assert res["evaluations"] == 1
    assert res["achieved_miss_m"] >= 1.0


def test_optimize_unreachable_reports_best(sandbox_conn, coarse_tca):
    """500 km is far beyond +/-5 m/s authority: reachable False, best reported."""
    res = maneuver_mod.optimize_dv(
        sandbox_conn, 1, 500_000.0, _opt_burn_iso(coarse_tca))
    assert res["reachable"] is False
    assert res["achieved_miss_m"] < 500_000.0
    # only dv = 0 and the two sign endpoints get evaluated; best is an endpoint
    assert res["evaluations"] == 3
    assert abs(res["dv_ms"]) == 5.0
    assert res["achieved_miss_m"] > 0.0


def test_optimize_unknown_conjunction_raises_lookup(sandbox_conn, coarse_tca):
    with pytest.raises(LookupError):
        maneuver_mod.optimize_dv(
            sandbox_conn, 999999, 1000.0, _opt_burn_iso(coarse_tca))


def test_optimize_validates_burn_and_target(sandbox_conn, coarse_tca):
    late = (coarse_tca - timedelta(seconds=60)).strftime(ISO_FMT)
    with pytest.raises(ValueError, match="5 minutes"):
        maneuver_mod.optimize_dv(sandbox_conn, 1, 1000.0, late)
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(ISO_FMT)
    with pytest.raises(ValueError, match="1 hour"):
        maneuver_mod.optimize_dv(sandbox_conn, 1, 1000.0, stale)
    with pytest.raises(ValueError, match="target_miss_m"):
        maneuver_mod.optimize_dv(sandbox_conn, 1, 0.0, _opt_burn_iso(coarse_tca))


def test_optimize_corrupt_tle_raises_value_error(sandbox_conn, coarse_tca):
    with pytest.raises(ValueError, match="TLE"):
        maneuver_mod.optimize_dv(
            sandbox_conn, 2, 1000.0, _opt_burn_iso(coarse_tca))
