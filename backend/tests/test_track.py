"""Ground-track endpoint + GMST/subpoint helper tests - hermetic: temp DB,
frozen clock, real SGP4 propagation, no network.

Time is frozen 10 h after the ISS TLE epoch so real propagation is
deterministic and physically sane. Sanity anchors: |lat| bounded by the
inclination, LEO altitude band, westward ground-track regression of roughly
one Earth-rotation-per-period (~23 deg), and 6-9 km/s subpoint speed.
"""
import math
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend.app import config, db
from backend.app.api import objects as objects_api
from backend.app.ingestion import propagate

# Real ISS TLE - same fixture constant as test_api.py / test_fleet.py.
# Epoch 24007.58333333 = 2024-01-07T14:00 UTC; frozen "now" is 10 h later.
ISS_TLE1 = "1 25544U 98067A   24007.58333333  .00016717  00000-0  30777-3 0  9994"
ISS_TLE2 = "2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.49560532430089"
ISS_INCLINATION_DEG = 51.64

# Microseconds present on purpose: the endpoint must floor to the second.
FROZEN_NOW = datetime(2024, 1, 8, 0, 0, 0, 123456, tzinfo=timezone.utc)
FROZEN_EPOCH_ISO = "2024-01-08T00:00:00Z"

TRACK_KEYS = {"epoch", "period_min", "samples", "subpoint"}
SUBPOINT_KEYS = {"lat_deg", "lon_deg", "alt_km", "velocity_kms"}
NULLS_SHAPE = {
    "epoch": FROZEN_EPOCH_ISO,
    "period_min": None,
    "samples": [],
    "subpoint": {"lat_deg": None, "lon_deg": None, "alt_km": None, "velocity_kms": None},
}


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FROZEN_NOW


def _wrap180(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _seed(conn):
    conn.executescript(
        f"""
    INSERT INTO satellites (id, norad_id, name, object_type, intl_designator, last_updated) VALUES
        (1, 25544, 'ISS (ZARYA)', 'PAYLOAD', '1998-067A', '2024-01-08T00:00:00Z'),
        (2, 60000, 'CORRUPT SAT', 'PAYLOAD', NULL, '2024-01-08T00:00:00Z'),
        (3, 70000, 'NO TLE SAT', 'PAYLOAD', NULL, '2024-01-08T00:00:00Z');

    -- Sat 1's OLDER row is corrupt on purpose: a track only renders if the
    -- endpoint selects the LATEST-epoch TLE.
    INSERT INTO tle_history (satellite_id, epoch, line1, line2, ingested_at) VALUES
        (1, '2024-01-06T14:00:00Z', 'CORRUPT LINE ONE', 'CORRUPT LINE TWO', '2024-01-06T15:00:00Z'),
        (1, '2024-01-07T14:00:00Z', '{ISS_TLE1}', '{ISS_TLE2}', '2024-01-07T15:00:00Z'),
        (2, '2024-01-07T14:00:00Z', 'CORRUPT LINE ONE', 'CORRUPT LINE TWO', '2024-01-07T15:00:00Z');
    """
    )
    conn.commit()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(objects_api, "datetime", _FrozenDatetime)
    conn = db.get_conn()
    db.init_db(conn)
    _seed(conn)
    conn.close()
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def body(client):
    response = client.get("/objects/25544/track")
    assert response.status_code == 200
    return response.json()


# --- endpoint shape ----------------------------------------------------------

def test_track_shape_and_epoch(body):
    assert set(body) == TRACK_KEYS
    assert body["epoch"] == FROZEN_EPOCH_ISO       # now, floored to the second
    assert 90.0 < body["period_min"] < 96.0        # ISS ~92.9 min
    assert len(body["samples"]) == 240             # default sample count
    assert all(len(sample) == 3 for sample in body["samples"])
    assert set(body["subpoint"]) == SUBPOINT_KEYS


def test_samples_clamped_16_720(client):
    assert len(client.get("/objects/25544/track?samples=5").json()["samples"]) == 16
    assert len(client.get("/objects/25544/track?samples=100000").json()["samples"]) == 720
    assert len(client.get("/objects/25544/track?samples=64").json()["samples"]) == 64


def test_rounding(body):
    for lon, lat, alt in body["samples"]:
        assert round(lon, 2) == lon and round(lat, 2) == lat
        assert round(alt, 1) == alt
    sub = body["subpoint"]
    assert round(sub["lat_deg"], 2) == sub["lat_deg"]
    assert round(sub["lon_deg"], 2) == sub["lon_deg"]
    assert round(sub["alt_km"], 1) == sub["alt_km"]
    assert round(sub["velocity_kms"], 2) == sub["velocity_kms"]


# --- physical sanity anchors -------------------------------------------------

def test_latitude_bounded_by_inclination(body):
    assert all(
        abs(lat) <= ISS_INCLINATION_DEG + 0.5 for _lon, lat, _alt in body["samples"]
    )
    # the track actually reaches high latitude (not stuck at the equator)
    assert max(abs(lat) for _lon, lat, _alt in body["samples"]) > 45.0


def test_longitudes_wrapped(body):
    assert all(-180.0 <= lon < 180.0 for lon, _lat, _alt in body["samples"])


def test_altitude_leo_band(body):
    assert all(350.0 <= alt <= 500.0 for _lon, _lat, alt in body["samples"])


def test_track_regresses_westward_by_earth_rotation(body):
    lons = [sample[0] for sample in body["samples"]]
    # Samples span exactly one period, so first and last share orbital phase:
    # the closing longitude shift is minus Earth's rotation per period
    # (~22.5 deg west for a ~93 min orbit, plus small RAAN drift).
    end_shift = _wrap180(lons[-1] - lons[0])
    assert -30.0 < end_shift < -15.0
    # unwrapped, a prograde track advances ~360 - 23 deg east over one orbit
    total_drift = sum(_wrap180(b - a) for a, b in zip(lons, lons[1:]))
    assert 320.0 < total_drift < 350.0


def test_subpoint_is_first_sample_state(body):
    lon0, lat0, alt0 = body["samples"][0]
    sub = body["subpoint"]
    assert sub["lon_deg"] == lon0
    assert sub["lat_deg"] == lat0
    assert sub["alt_km"] == alt0
    assert 6.0 <= sub["velocity_kms"] <= 9.0


# --- failure shapes ----------------------------------------------------------

def test_unknown_norad_404(client):
    response = client.get("/objects/99999/track")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_corrupt_tle_yields_nulls_shape(client):
    assert client.get("/objects/60000/track").json() == NULLS_SHAPE


def test_missing_tle_yields_nulls_shape(client):
    assert client.get("/objects/70000/track").json() == NULLS_SHAPE


# --- propagate helpers -------------------------------------------------------

def test_gmst_at_j2000():
    # IAU-82 at T=0 (2000-01-01 12:00): GMST = 67310.54841 s = 280.46062 deg.
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert math.isclose(
        propagate.gmst_rad(j2000), math.radians(280.46062), abs_tol=1e-5
    )


def test_gmst_daily_advance_and_range():
    t0 = datetime(2024, 1, 8, 0, 0, 0, tzinfo=timezone.utc)
    # Sidereal time gains ~0.98565 deg per solar day.
    advance = (
        propagate.gmst_rad(t0 + timedelta(days=1)) - propagate.gmst_rad(t0)
    ) % (2.0 * math.pi)
    assert math.isclose(advance, math.radians(0.98565), abs_tol=1e-5)
    for hours in range(0, 48, 7):
        value = propagate.gmst_rad(t0 + timedelta(hours=hours))
        assert 0.0 <= value < 2.0 * math.pi


def test_subpoint_helper_physical():
    sub = propagate.subpoint(
        ISS_TLE1, ISS_TLE2, datetime(2024, 1, 8, 0, 0, 0, tzinfo=timezone.utc)
    )
    assert set(sub) == SUBPOINT_KEYS
    assert abs(sub["lat_deg"]) <= ISS_INCLINATION_DEG + 0.5
    assert -180.0 <= sub["lon_deg"] < 180.0
    assert 350.0 <= sub["alt_km"] <= 500.0
    assert 6.0 <= sub["velocity_kms"] <= 9.0


# --- gzip middleware ---------------------------------------------------------

def test_status_still_serves_with_gzip_accepted(client):
    response = client.get("/status", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert response.json()["catalog_size"] == 3
    # below minimum_size=1024: must not be compressed
    assert response.headers.get("content-encoding") != "gzip"


def test_large_track_response_is_gzipped(client):
    response = client.get(
        "/objects/25544/track?samples=720", headers={"Accept-Encoding": "gzip"}
    )
    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"
    assert len(response.json()["samples"]) == 720  # decompresses cleanly
