"""Fleet positions endpoint tests - hermetic: temp DB, frozen clock, no network.

Time is frozen near the ISS TLE epoch so real SGP4 propagation is
deterministic and physically sane (radius / speed assertions below).
"""
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.app import config, db
from backend.app.api import fleet

# Real ISS TLE - same fixture constant as test_api.py. Epoch 24007.58333333
# = 2024-01-07T14:00 UTC; the frozen "now" below is 10 h later.
ISS_TLE1 = "1 25544U 98067A   24007.58333333  .00016717  00000-0  30777-3 0  9994"
ISS_TLE2 = "2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.49560532430089"

# Microseconds present on purpose: the endpoint must floor to the second.
FROZEN_NOW = datetime(2024, 1, 8, 0, 0, 0, 123456, tzinfo=timezone.utc)
FROZEN_EPOCH_ISO = "2024-01-08T00:00:00Z"


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FROZEN_NOW


def _seed(conn):
    conn.executescript(
        f"""
    INSERT INTO satellites (id, norad_id, name, object_type, intl_designator, last_updated) VALUES
        (1, 25544, 'ISS (ZARYA)', 'PAYLOAD', '1998-067A', '2024-01-08T00:00:00Z'),
        (2, 34454, 'COSMOS 2251 DEB', 'DEBRIS', NULL, '2024-01-08T00:00:00Z'),
        (3, 40000, 'FALCON 9 R/B', 'ROCKET BODY', NULL, '2024-01-08T00:00:00Z'),
        (4, 50000, 'UNKNOWN OBJ', 'UNKNOWN', NULL, '2024-01-08T00:00:00Z'),
        (5, 60000, 'CORRUPT SAT', 'PAYLOAD', NULL, '2024-01-08T00:00:00Z'),
        (6, 70000, 'NO TLE SAT', 'PAYLOAD', NULL, '2024-01-08T00:00:00Z');

    -- Sat 1 has an older TLE row too: the endpoint must pick the LATEST epoch.
    INSERT INTO tle_history (satellite_id, epoch, line1, line2, ingested_at) VALUES
        (1, '2024-01-06T14:00:00Z', '{ISS_TLE1}', '{ISS_TLE2}', '2024-01-06T15:00:00Z'),
        (1, '2024-01-07T14:00:00Z', '{ISS_TLE1}', '{ISS_TLE2}', '2024-01-07T15:00:00Z'),
        (2, '2024-01-07T14:00:00Z', '{ISS_TLE1}', '{ISS_TLE2}', '2024-01-07T15:00:00Z'),
        (3, '2024-01-07T14:00:00Z', '{ISS_TLE1}', '{ISS_TLE2}', '2024-01-07T15:00:00Z'),
        (4, '2024-01-07T14:00:00Z', '{ISS_TLE1}', '{ISS_TLE2}', '2024-01-07T15:00:00Z'),
        (5, '2024-01-07T14:00:00Z', 'CORRUPT LINE ONE', 'CORRUPT LINE TWO', '2024-01-07T15:00:00Z');

    -- c1 active sat1-sat2: Pc above the elevated threshold -> risk_code 2 on BOTH sides.
    -- c2 active sat1-sat4: Pc above the critical threshold -> risk_code 3 (max wins for sat1).
    -- c3 PAST sat3-sat2: critical Pc but inactive -> must not contribute.
    INSERT INTO conjunctions (id, primary_object_id, secondary_object_id,
                              time_of_closest_approach, first_seen_at, status) VALUES
        (1, 1, 2, '2024-01-09T04:12:09Z', '2024-01-07T10:00:00Z', 'active'),
        (2, 1, 4, '2024-01-09T10:00:00Z', '2024-01-07T11:00:00Z', 'active'),
        (3, 3, 2, '2024-01-05T00:00:00Z', '2024-01-03T00:00:00Z', 'past');

    INSERT INTO cdm_messages (conjunction_id, message_time, miss_distance_m,
                              relative_velocity_ms, probability_of_collision, source) VALUES
        (1, '2024-01-07T22:00:00Z', 812.4, 14210.3, 3.1e-05, 'screening'),
        (2, '2024-01-07T22:00:00Z', 120.0, 12000.0, 2.0e-04, 'screening'),
        (3, '2024-01-04T00:00:00Z', 90.0, 11000.0, 5.0e-04, 'screening');
    """
    )
    conn.commit()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(fleet, "datetime", _FrozenDatetime)
    conn = db.get_conn()
    db.init_db(conn)
    _seed(conn)
    conn.close()
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def body(client):
    response = client.get("/fleet/positions")
    assert response.status_code == 200
    return response.json()


def _by_norad(body):
    return {row[0]: row for row in body["objects"]}


def test_epoch_is_now_floored_to_the_second(body):
    assert set(body) == {"epoch", "count", "objects"}
    assert body["epoch"] == FROZEN_EPOCH_ISO  # microseconds floored, trailing Z
    parsed = datetime.strptime(body["epoch"], "%Y-%m-%dT%H:%M:%SZ")
    assert parsed.replace(tzinfo=timezone.utc) == FROZEN_NOW.replace(microsecond=0)


def test_packed_rows_shape_and_count(body):
    # sats 1-4 propagate; corrupt-TLE sat 5 and TLE-less sat 6 are excluded
    assert body["count"] == 4
    assert len(body["objects"]) == 4
    for row in body["objects"]:
        assert len(row) == 10
        norad, name, type_code, x, y, z, vx, vy, vz, risk_code = row
        assert isinstance(norad, int)
        assert isinstance(name, str)
        assert isinstance(type_code, int)
        assert isinstance(risk_code, int)
        for value in (x, y, z, vx, vy, vz):
            assert isinstance(value, float)
    assert _by_norad(body)[25544][1] == "ISS (ZARYA)"


def test_positions_and_velocities_physical(body):
    for row in body["objects"]:
        x, y, z, vx, vy, vz = row[3:9]
        radius = (x * x + y * y + z * z) ** 0.5
        speed = (vx * vx + vy * vy + vz * vz) ** 0.5
        assert 6600.0 <= radius <= 7100.0   # LEO shell, km TEME
        assert 6.0 <= speed <= 9.0          # km/s


def test_rounding(body):
    for row in body["objects"]:
        assert all(round(value, 3) == value for value in row[3:6])   # km, 3 dp
        assert all(round(value, 4) == value for value in row[6:9])   # km/s, 4 dp


def test_type_code_mapping(body):
    rows = _by_norad(body)
    assert rows[25544][2] == 0   # PAYLOAD
    assert rows[34454][2] == 1   # DEBRIS
    assert rows[40000][2] == 2   # ROCKET BODY
    assert rows[50000][2] == 3   # UNKNOWN


def test_risk_code_mapping(body):
    rows = _by_norad(body)
    # sat1: max(elevated c1, critical c2) -> 3; both sides of an event carry it
    assert rows[25544][9] == 3
    assert rows[34454][9] == 2   # secondary side of elevated c1
    assert rows[50000][9] == 3   # secondary side of critical c2
    # past-critical c3 must not count; sat3 has no active conjunction
    assert rows[40000][9] == 0


def test_corrupt_tle_is_skipped_not_500(body):
    assert 60000 not in _by_norad(body)   # corrupt TLE object silently skipped
