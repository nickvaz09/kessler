"""Encounter-ephemeris endpoint tests - hermetic: temp DB, synthetic TLEs,
real SGP4 propagation, no network.

Geometry: two make_tle orbits identical except a 5 deg inclination split and a
small mean-anomaly offset, sharing their node line (epoch = now, MA ~350 deg).
They cross near the ascending node ~155 s after epoch with a genuinely steep
separation minimum (~0.66 km/s crossing rate), so centering/refinement
assertions are sharp and deterministic. The conjunction row is seeded at the
COARSE TCA taken from a 30 s grid - the endpoint's 0.05 s re-refinement must
recover the sub-sample TCA and beat the stored (coarse) miss distance.
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import config, db
from backend.app.ingestion import propagate
from backend.app.models import pc as pc_mod
from backend.tests.test_ingestion import make_tle

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

EPOCH = datetime.now(timezone.utc).replace(microsecond=0)
EPOCH_ISO = EPOCH.strftime(ISO_FMT)

# Crossing pair: same node line (same RAAN), 5 deg inclination split, small
# mean-anomaly offset -> close approach at the ascending node (~155 s after
# epoch: 10 deg of mean anomaly at 15.5 rev/day), miss ~2.4 km along-track.
A_L1, A_L2 = make_tle(90001, 51.6, 100.0, 0.0005, 0.0, 350.00, 15.5, EPOCH)
B_L1, B_L2 = make_tle(90002, 56.6, 100.0, 0.0005, 0.0, 350.02, 15.5, EPOCH)

TOP_KEYS = {
    "conjunction_id", "tca", "window_s", "step_s", "start", "gmst_start_rad",
    "earth_rot_rad_s", "primary", "secondary", "tca_state", "sigma_m",
    "latest_message_miss_m", "geometry_consistent",
}
OBJECT_KEYS = {"norad_id", "name", "r"}
TCA_STATE_KEYS = {"sep_m", "rel_vel_ms", "ric_m", "approach_angle_deg", "rel_vel_hat"}
RIC_KEYS = {"radial", "in_track", "cross_track"}
NULL_TCA_STATE = {
    "sep_m": None,
    "rel_vel_ms": None,
    "ric_m": {"radial": None, "in_track": None, "cross_track": None},
    "approach_angle_deg": None,
    "rel_vel_hat": None,
}


def _parse_iso(s: str) -> datetime:
    return datetime.strptime(s, ISO_FMT).replace(tzinfo=timezone.utc)


def _sep_m(body, i: int) -> float:
    px, py, pz = body["primary"]["r"][i]
    sx, sy, sz = body["secondary"]["r"][i]
    return math.dist((px, py, pz), (sx, sy, sz)) * 1000.0


@pytest.fixture(scope="session")
def coarse():
    """(coarse_tca, coarse_miss_m): argmin separation on a 30 s grid.

    Deliberately coarse - with a ~0.66 km/s crossing rate a 30 s grid
    overstates the miss by up to ~10 km, which is exactly what the endpoint's
    0.05 s re-refinement must beat.
    """
    times = [EPOCH + timedelta(seconds=30 * i) for i in range(41)]
    ra, _va = propagate.positions_at(A_L1, A_L2, times)
    rb, _vb = propagate.positions_at(B_L1, B_L2, times)
    d_km = np.linalg.norm(ra - rb, axis=1)
    i = int(np.argmin(d_km))
    assert 0 < i < len(times) - 1  # minimum interior to the search window
    return times[i], float(d_km[i]) * 1000.0


def _seed(conn, coarse_tca_iso: str, coarse_miss_m: float) -> None:
    # Conjunction 4: same pair, stored TCA 20 min off the real crossing, and
    # a latest message claiming 150 m - the TLEs put the pair thousands of km
    # apart there (the avoidance plan's "geometry_mismatch" case).
    off_tca_iso = (datetime.strptime(coarse_tca_iso, ISO_FMT)
                   + timedelta(minutes=20)).strftime(ISO_FMT)
    conn.executescript(
        f"""
    INSERT INTO satellites (id, norad_id, name, object_type, last_updated) VALUES
        (1, 90001, 'SIM PRIMARY',   'PAYLOAD', '{EPOCH_ISO}'),
        (2, 90002, 'SIM SECONDARY', 'DEBRIS',  '{EPOCH_ISO}'),
        (3, 90003, 'CORRUPT SAT',   'PAYLOAD', '{EPOCH_ISO}'),
        (4, 90004, 'NO TLE SAT',    'PAYLOAD', '{EPOCH_ISO}');

    INSERT INTO tle_history (satellite_id, epoch, line1, line2, ingested_at) VALUES
        (1, '{EPOCH_ISO}', '{A_L1}', '{A_L2}', '{EPOCH_ISO}'),
        (2, '{EPOCH_ISO}', '{B_L1}', '{B_L2}', '{EPOCH_ISO}'),
        (3, '{EPOCH_ISO}', 'CORRUPT LINE ONE', 'CORRUPT LINE TWO', '{EPOCH_ISO}');

    INSERT INTO conjunctions (id, primary_object_id, secondary_object_id,
                              time_of_closest_approach, first_seen_at, status) VALUES
        (1, 1, 2, '{coarse_tca_iso}', '{EPOCH_ISO}', 'active'),
        (2, 1, 3, '{coarse_tca_iso}', '{EPOCH_ISO}', 'active'),
        (3, 1, 4, '{coarse_tca_iso}', '{EPOCH_ISO}', 'active'),
        (4, 1, 2, '{off_tca_iso}', '{EPOCH_ISO}', 'active');

    INSERT INTO cdm_messages (conjunction_id, message_time, miss_distance_m,
                              relative_velocity_ms, probability_of_collision, source)
        VALUES (1, '{EPOCH_ISO}', {coarse_miss_m}, 660.0, NULL, 'screening'),
               (4, '{EPOCH_ISO}', 150.0, 660.0, 1e-4, 'screening');
    """
    )
    conn.commit()


@pytest.fixture()
def client(tmp_path, monkeypatch, coarse):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    coarse_tca, coarse_miss_m = coarse
    conn = db.get_conn()
    db.init_db(conn)
    _seed(conn, coarse_tca.strftime(ISO_FMT), coarse_miss_m)
    conn.close()
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def body(client):
    response = client.get("/conjunctions/1/ephemeris?window_s=120")
    assert response.status_code == 200
    return response.json()


# --- shape -------------------------------------------------------------------

def test_shape_and_constants(body):
    assert set(body) == TOP_KEYS
    assert body["conjunction_id"] == 1
    assert body["window_s"] == 120
    assert body["step_s"] == 1.0
    assert body["earth_rot_rad_s"] == 7.2921159e-5
    assert 0.0 <= body["gmst_start_rad"] < 2.0 * math.pi

    assert set(body["primary"]) == OBJECT_KEYS
    assert set(body["secondary"]) == OBJECT_KEYS
    assert body["primary"]["norad_id"] == 90001
    assert body["primary"]["name"] == "SIM PRIMARY"
    assert body["secondary"]["norad_id"] == 90002
    assert body["secondary"]["name"] == "SIM SECONDARY"

    assert set(body["tca_state"]) == TCA_STATE_KEYS
    assert set(body["tca_state"]["ric_m"]) == RIC_KEYS

    # start = tca - window_s (both strings floored to the same sub-second).
    delta = _parse_iso(body["tca"]) - _parse_iso(body["start"])
    assert delta.total_seconds() == body["window_s"]


def test_grid_length_rounding_and_sanity(body):
    n = 2 * body["window_s"] + 1
    assert len(body["primary"]["r"]) == n
    assert len(body["secondary"]["r"]) == n
    for r in (body["primary"]["r"], body["secondary"]["r"]):
        assert all(len(sample) == 3 for sample in r)
        # km TEME rounded to 3 dp, LEO radius band for this synthetic orbit
        for sample in (r[0], r[n // 2], r[-1]):
            assert all(round(c, 3) == c for c in sample)
            radius = math.sqrt(sum(c * c for c in sample))
            assert 6600.0 < radius < 6900.0


# --- refinement + centering --------------------------------------------------

def test_grid_centered_on_minimum_separation(body):
    n = len(body["primary"]["r"])
    center = body["window_s"]
    seps = [_sep_m(body, i) for i in range(n)]
    # The middle sample sits at the refined TCA: it is the grid minimum
    # (within a couple of samples - 3 dp rounding is ~1 m of noise while the
    # per-sample separation change near TCA is ~90 m here).
    assert abs(int(np.argmin(seps)) - center) <= 2
    # ... and matches the reported refined separation up to grid rounding.
    assert abs(seps[center] - body["tca_state"]["sep_m"]) < 3.0


def test_refined_tca_beats_coarse_seed(body, coarse):
    coarse_tca, coarse_miss_m = coarse
    # Sub-sample refinement can only tighten a 30 s grid minimum.
    assert body["tca_state"]["sep_m"] <= coarse_miss_m * 1.001 + 0.5
    # Refined TCA stays near the seeded coarse TCA (within the 30 s grid slop,
    # + 1 s for the whole-second string flooring).
    tca = _parse_iso(body["tca"])
    assert abs((tca - coarse_tca).total_seconds()) <= 16.0
    # Physically: the node crossing is ~155 s after the TLE epoch.
    assert abs((tca - EPOCH).total_seconds() - 155.0) < 30.0


def test_tca_state_geometry(body):
    state = body["tca_state"]
    # crossing geometry: ~2.4 km along-track miss, ~0.66 km/s encounter
    assert 200.0 < state["sep_m"] < 6000.0
    assert 400.0 < state["rel_vel_ms"] < 900.0
    # Angle between the velocity vectors == the 5 deg inclination split.
    assert 4.5 < state["approach_angle_deg"] < 5.5
    # RIC is a rotation of the separation vector: |ric| reconstructs sep_m.
    ric = state["ric_m"]
    ric_norm = math.sqrt(ric["radial"] ** 2 + ric["in_track"] ** 2
                         + ric["cross_track"] ** 2)
    assert ric_norm == pytest.approx(state["sep_m"], rel=0.01)
    # rel_vel_hat is a unit vector in TEME.
    ux, uy, uz = state["rel_vel_hat"]
    assert math.sqrt(ux * ux + uy * uy + uz * uz) == pytest.approx(1.0, abs=1e-3)


def test_sigma_combined_from_tle_ages(body):
    # Both TLE epochs are EPOCH; sigma_m must be the RSS of sigma_at(age)
    # per object with age measured to the refined TCA (documented ASSUMED).
    age_days = (_parse_iso(body["tca"]) - EPOCH).total_seconds() / 86400.0
    expected = math.sqrt(2.0 * pc_mod.sigma_at(age_days) ** 2)
    assert body["sigma_m"] == pytest.approx(expected, abs=0.2)


# --- parameter handling ------------------------------------------------------

def test_window_clamps_and_default(client):
    low = client.get("/conjunctions/1/ephemeris?window_s=10").json()
    assert low["window_s"] == 60
    assert len(low["primary"]["r"]) == 121

    high = client.get("/conjunctions/1/ephemeris?window_s=99999").json()
    assert high["window_s"] == 3600
    assert len(high["primary"]["r"]) == 7201
    assert len(high["secondary"]["r"]) == 7201

    default = client.get("/conjunctions/1/ephemeris").json()
    assert default["window_s"] == 1200
    assert default["step_s"] == 1.0
    assert len(default["primary"]["r"]) == 2401


def test_unknown_conjunction_404(client):
    response = client.get("/conjunctions/999/ephemeris")
    assert response.status_code == 404
    assert "detail" in response.json()


# --- degrade paths (never 500 on a bad TLE) ----------------------------------

def test_corrupt_tle_degrades(client, coarse):
    response = client.get("/conjunctions/2/ephemeris?window_s=120")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == TOP_KEYS
    # Failed object: empty grid; healthy object still fully sampled.
    assert payload["secondary"]["r"] == []
    assert len(payload["primary"]["r"]) == 241
    # No refinement possible: stored TCA echoed, tca_state all nulls.
    assert payload["tca"] == coarse[0].strftime(ISO_FMT)
    assert payload["tca_state"] == NULL_TCA_STATE
    # Both TLE rows exist (epochs are valid), so the ASSUMED sigma still holds.
    assert payload["sigma_m"] is not None and payload["sigma_m"] > 0.0


def test_missing_tle_degrades(client, coarse):
    response = client.get("/conjunctions/3/ephemeris?window_s=120")
    assert response.status_code == 200
    payload = response.json()
    assert payload["secondary"]["r"] == []
    assert len(payload["primary"]["r"]) == 241
    assert payload["tca"] == coarse[0].strftime(ISO_FMT)
    assert payload["tca_state"] == NULL_TCA_STATE
    assert payload["sigma_m"] is None


# --- geometry check vs the latest message -------------------------------------

def test_geometry_consistent_when_message_agrees(body):
    """Conjunction 1: the message miss is the coarse screen of the same TLEs,
    so both clear (or both miss) the plan's required miss together."""
    assert body["latest_message_miss_m"] is not None
    assert body["latest_message_miss_m"] >= body["tca_state"]["sep_m"]
    assert body["geometry_consistent"] is True


def test_geometry_mismatch_flagged(client):
    """Conjunction 4: TLEs say far apart, the latest message says 150 m. The
    ephemeris must flag it with the same verdict the avoidance plan reaches."""
    r = client.get("/conjunctions/4/ephemeris?window_s=60")
    assert r.status_code == 200
    eph = r.json()
    assert eph["latest_message_miss_m"] == 150.0
    assert eph["tca_state"]["sep_m"] > 100_000.0
    assert eph["geometry_consistent"] is False
    plan = client.post("/conjunctions/4/avoidance-plan")
    if plan.status_code == 200:   # the plan needs a TCA far enough ahead
        assert plan.json()["status"] == "geometry_mismatch"


def test_geometry_unknown_without_message_or_state(client):
    """No message (conjunction 3, missing TLE too): the check is None, never
    a false all-clear."""
    eph = client.get("/conjunctions/3/ephemeris?window_s=60").json()
    assert eph["latest_message_miss_m"] is None
    assert eph["geometry_consistent"] is None
