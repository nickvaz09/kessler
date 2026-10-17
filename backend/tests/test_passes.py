"""Ground-station pass prediction tests - hermetic: temp DB, frozen clock,
real SGP4 propagation, no network.

Geometry anchors: an equatorial observer directly under a synthetic
zero-inclination orbit (make_tle, 15.5 rev/day, alt ~410 km) must see one
near-zenith pass per SYNODIC period (1/(1/94 min - 1/1436 min) ~ 99.3 min ->
14-15 passes/day), rising due west (the satellite outruns Earth rotation) and
setting due east, each ~6-7 min above 10 deg. A polar observer can never see
that orbit (elevation ~ -40 deg at closest). ISS from a mid-latitude site
yields a handful of passes per day, all ordered aos < tca < los.
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import config, db
from backend.app.api import objects as objects_api
from backend.app.ingestion import propagate
from backend.tests.test_ingestion import make_tle

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Real ISS TLE - same fixture constant as test_track.py / test_fleet.py.
# Epoch 24007.58333333 = 2024-01-07T14:00 UTC; frozen "now" is 10 h later.
ISS_TLE1 = "1 25544U 98067A   24007.58333333  .00016717  00000-0  30777-3 0  9994"
ISS_TLE2 = "2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.49560532430089"

EPOCH = datetime(2024, 1, 8, 0, 0, 0, tzinfo=timezone.utc)
# Microseconds present on purpose: the endpoint must floor to the second.
FROZEN_NOW = datetime(2024, 1, 8, 0, 0, 0, 123456, tzinfo=timezone.utc)
FROZEN_START_ISO = "2024-01-08T00:00:00Z"

# synthetic equatorial orbit: its ground track is the equator
EQ_L1, EQ_L2 = make_tle(90101, 0.0, 0.0, 0.0005, 0.0, 0.0, 15.5, EPOCH)

PASS_KEYS = {"aos", "los", "tca", "max_el_deg", "aos_az_deg", "los_az_deg",
             "duration_s", "track"}
MIDLAT_SITE = {"lat": 40.0, "lon": -75.0}


def _parse_iso(s: str) -> datetime:
    return datetime.strptime(s, ISO_FMT).replace(tzinfo=timezone.utc)


def _assert_track_shape(p: dict) -> None:
    """Sky-view track: [[t_offset_s, az_deg, el_deg]] every 10 s AOS->LOS.

    Offsets are measured from AOS, strictly increasing, on a 10 s grid with
    the LOS sample always closing the track (final gap <= 10 s). Elevation
    is not re-filtered against min_el (a multi-hump pass keeps its dip), so
    only physical bounds apply; az stays in [0, 360).
    """
    track = p["track"]
    assert isinstance(track, list) and len(track) >= 1
    for sample in track:
        assert isinstance(sample, list) and len(sample) == 3
        off, az, el = sample
        assert isinstance(off, int)
        assert 0.0 <= az < 360.0
        assert -90.0 <= el <= 90.0
        assert el <= p["max_el_deg"] + 0.11   # refined peak bounds the grid
    offsets = [s[0] for s in track]
    assert offsets[0] == 0
    assert offsets[-1] == p["duration_s"]
    assert all(b > a for a, b in zip(offsets, offsets[1:]))
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    assert all(gap == 10 for gap in gaps[:-1])
    assert all(0 < gap <= 10 for gap in gaps)


def _assert_pass_shape(p: dict, min_el: float = 10.0) -> None:
    assert set(p) == PASS_KEYS
    aos, tca, los = _parse_iso(p["aos"]), _parse_iso(p["tca"]), _parse_iso(p["los"])
    assert aos < tca < los
    assert p["duration_s"] == (los - aos).total_seconds()
    assert min_el <= p["max_el_deg"] <= 90.0
    assert 0.0 <= p["aos_az_deg"] < 360.0
    assert 0.0 <= p["los_az_deg"] < 360.0
    _assert_track_shape(p)


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FROZEN_NOW


# --- propagate helpers: geometry sanity --------------------------------------

def test_observer_ecef_wgs84():
    # Equator: semi-major axis along x. Pole: polar radius b = a(1 - f).
    np.testing.assert_allclose(
        propagate.observer_ecef(0.0, 0.0, 0.0), [6378.137, 0.0, 0.0], atol=1e-9
    )
    pole = propagate.observer_ecef(90.0, 0.0, 0.0)
    assert pole[2] == pytest.approx(6378.137 * (1.0 - 1.0 / 298.257223563), abs=1e-3)
    assert abs(pole[0]) < 1e-9 and abs(pole[1]) < 1e-9
    # altitude adds along the local vertical (radially at the equator)
    lifted = propagate.observer_ecef(0.0, 0.0, 1000.0)
    assert lifted[0] == pytest.approx(6379.137, abs=1e-9)


def test_look_angles_zenith_and_range_clip():
    # A point straight above the site (along the site vector, TEME == ECEF at
    # GMST-rotated placement) must read elevation 90.
    t = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    g = propagate.gmst_rad(t)
    up_ecef = propagate.observer_ecef(0.0, 0.0, 0.0) * (7000.0 / 6378.137)
    # Rotate ECEF back to TEME (inverse of the z-rotation by GMST).
    up_teme = np.array([[up_ecef[0] * math.cos(g) - up_ecef[1] * math.sin(g),
                         up_ecef[0] * math.sin(g) + up_ecef[1] * math.cos(g),
                         0.0]])
    el, _az = propagate.look_angles(up_teme, [t], 0.0, 0.0, 0.0)
    assert el[0] == pytest.approx(90.0, abs=1e-6)


def test_equatorial_observer_sees_periodic_overhead_passes():
    result = propagate.passes(EQ_L1, EQ_L2, 0.0, 0.0, 0.0, EPOCH, 24)
    # one pass per synodic period (~99.3 min) -> 14-15 in 24 h
    assert 13 <= len(result) <= 16
    for p in result:
        _assert_pass_shape(p)
        assert p["max_el_deg"] > 85.0            # passes through zenith
        assert 300 <= p["duration_s"] <= 600     # ~6.8 min above 10 deg
        # Outruns Earth rotation: rises due west, sets due east.
        assert p["aos_az_deg"] == pytest.approx(270.0, abs=2.0)
        assert p["los_az_deg"] == pytest.approx(90.0, abs=2.0)
    gaps_min = [
        (_parse_iso(b["aos"]) - _parse_iso(a["aos"])).total_seconds() / 60.0
        for a, b in zip(result, result[1:])
    ]
    assert all(95.0 < gap < 104.0 for gap in gaps_min)


def test_polar_observer_never_sees_equatorial_orbit():
    assert propagate.passes(EQ_L1, EQ_L2, 89.9, 0.0, 0.0, EPOCH, 24) == []


def test_higher_threshold_shrinks_passes():
    low = propagate.passes(EQ_L1, EQ_L2, 0.0, 0.0, 0.0, EPOCH, 6, min_el_deg=10)
    high = propagate.passes(EQ_L1, EQ_L2, 0.0, 0.0, 0.0, EPOCH, 6, min_el_deg=60)
    assert len(high) == len(low) > 0             # zenith passes clear both bars
    for lo, hi in zip(low, high):
        assert hi["duration_s"] < lo["duration_s"]
        assert lo["aos"] < hi["aos"] and hi["los"] < lo["los"]


def test_iss_midlat_site_chronological_and_sane():
    result = propagate.passes(
        ISS_TLE1, ISS_TLE2, MIDLAT_SITE["lat"], MIDLAT_SITE["lon"], 0.0, EPOCH, 24
    )
    assert 0 < len(result) <= 8
    for p in result:
        _assert_pass_shape(p)
        assert 60 <= p["duration_s"] <= 900
    aos_list = [p["aos"] for p in result]
    assert aos_list == sorted(aos_list)
    # Passes never overlap: each AOS follows the previous LOS.
    for a, b in zip(result, result[1:]):
        assert a["los"] < b["aos"]


def test_track_sampling_covers_aos_to_los():
    result = propagate.passes(EQ_L1, EQ_L2, 0.0, 0.0, 0.0, EPOCH, 6)
    assert result
    for p in result:
        track = p["track"]
        # 10 s grid from AOS plus the closing LOS sample.
        assert len(track) == math.ceil(p["duration_s"] / 10) + 1
        # Track endpoints are the refined AOS/LOS samples: az matches the
        # summary fields, el sits just above the 10 deg threshold (1 s grid
        # overshoot is bounded by the elevation rate at the crossing).
        assert track[0][1] % 360.0 == pytest.approx(p["aos_az_deg"] % 360.0, abs=0.11)
        assert track[-1][1] % 360.0 == pytest.approx(p["los_az_deg"] % 360.0, abs=0.11)
        assert 10.0 - 0.11 <= track[0][2] <= 13.0
        assert 10.0 - 0.11 <= track[-1][2] <= 13.0
        # The 10 s grid lands within ~5 s of culmination; a zenith pass at
        # ~410 km sweeps <= ~1.1 deg/s there, so the sampled peak stays
        # within a few degrees of the parabola-refined maximum.
        assert max(s[2] for s in track) >= p["max_el_deg"] - 7.0


def test_track_zenith_pass_azimuth_flips_west_to_east():
    # Equatorial overhead pass: azimuth starts due west, ends due east; the
    # elevation profile rises to the peak then falls (single hump).
    p = propagate.passes(EQ_L1, EQ_L2, 0.0, 0.0, 0.0, EPOCH, 3)[0]
    track = p["track"]
    assert track[0][1] == pytest.approx(270.0, abs=2.0)
    assert track[-1][1] == pytest.approx(90.0, abs=2.0)
    peak = max(range(len(track)), key=lambda i: track[i][2])
    els = [s[2] for s in track]
    assert els[:peak + 1] == sorted(els[:peak + 1])
    assert els[peak:] == sorted(els[peak:], reverse=True)


def test_pass_in_progress_at_window_start_clamps_to_edge():
    # Start the window mid-pass (inside the first equatorial pass above):
    # AOS must clamp to the window start, duration truncated accordingly.
    full = propagate.passes(EQ_L1, EQ_L2, 0.0, 0.0, 0.0, EPOCH, 3)
    mid = _parse_iso(full[0]["tca"])
    clipped = propagate.passes(EQ_L1, EQ_L2, 0.0, 0.0, 0.0, mid, 3)
    assert clipped[0]["aos"] == full[0]["tca"]
    assert clipped[0]["los"] == full[0]["los"]
    assert clipped[0]["duration_s"] < full[0]["duration_s"]


def test_propagation_failure_raises_value_error():
    # Decayed/hyperbolic propagation errors surface as ValueError (positions_at)
    # so the endpoint can degrade; garbage lines raise on element parsing.
    with pytest.raises(Exception):
        propagate.passes("CORRUPT LINE ONE", "CORRUPT LINE TWO",
                         0.0, 0.0, 0.0, EPOCH, 1)


# --- endpoint -----------------------------------------------------------------

def _seed(conn):
    conn.executescript(
        f"""
    INSERT INTO satellites (id, norad_id, name, object_type, intl_designator, last_updated) VALUES
        (1, 25544, 'ISS (ZARYA)', 'PAYLOAD', '1998-067A', '2024-01-08T00:00:00Z'),
        (2, 60000, 'CORRUPT SAT', 'PAYLOAD', NULL, '2024-01-08T00:00:00Z'),
        (3, 70000, 'NO TLE SAT', 'PAYLOAD', NULL, '2024-01-08T00:00:00Z');

    INSERT INTO tle_history (satellite_id, epoch, line1, line2, ingested_at) VALUES
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


def _get(client, norad=25544, **params):
    query = {"lat": MIDLAT_SITE["lat"], "lon": MIDLAT_SITE["lon"], **params}
    return client.get(f"/objects/{norad}/passes", params=query)


def test_endpoint_shape_site_echo_and_passes(client):
    response = _get(client, hours=24)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"site", "passes", "note"}
    assert body["site"] == {
        "lat_deg": 40.0, "lon_deg": -75.0, "alt_m": 0.0, "min_el_deg": 10.0,
        "hours": 24, "start": FROZEN_START_ISO,  # now, floored to the second
    }
    assert body["note"] is None
    assert 0 < len(body["passes"]) <= 8
    for p in body["passes"]:
        _assert_pass_shape(p)
        assert p["aos"] >= FROZEN_START_ISO


def test_endpoint_min_el_filters(client):
    loose = _get(client, hours=24).json()["passes"]
    strict = _get(client, hours=24, min_el=45).json()["passes"]
    assert len(strict) < len(loose)
    assert all(p["max_el_deg"] >= 45.0 for p in strict)


def test_endpoint_empty_when_geometry_forbids(client):
    # ISS (51.6 deg inclination) is never visible from the south pole.
    body = _get(client, lat=-89.9, lon=0.0, hours=24).json()
    assert body["passes"] == []
    assert body["note"] is None


def test_unknown_norad_404(client):
    response = _get(client, norad=99999)
    assert response.status_code == 404
    assert "detail" in response.json()


def test_corrupt_tle_degrades_with_note(client):
    response = _get(client, norad=60000)
    assert response.status_code == 200
    body = response.json()
    assert body["passes"] == []
    assert isinstance(body["note"], str) and body["note"]


def test_missing_tle_degrades_with_note(client):
    response = _get(client, norad=70000)
    assert response.status_code == 200
    body = response.json()
    assert body["passes"] == []
    assert isinstance(body["note"], str) and body["note"]


@pytest.mark.parametrize("params", [
    {},                                        # lat+lon required
    {"lon": 0.0},                              # lat required
    {"lat": 0.0},                              # lon required
    {"lat": 90.5, "lon": 0.0},                 # lat out of range
    {"lat": -90.5, "lon": 0.0},
    {"lat": 0.0, "lon": 180.5},                # lon out of range
    {"lat": 0.0, "lon": -180.5},
    {"lat": 0.0, "lon": 0.0, "min_el": 90.1},  # min_el out of range
    {"lat": 0.0, "lon": 0.0, "min_el": -0.1},
    {"lat": 0.0, "lon": 0.0, "hours": 0},      # hours out of range
    {"lat": 0.0, "lon": 0.0, "hours": 169},
    {"lat": 0.0, "lon": 0.0, "alt_m": 10001},  # alt_m out of range
    {"lat": "north", "lon": 0.0},              # non-numeric
])
def test_param_validation_422(client, params):
    assert client.get("/objects/25544/passes", params=params).status_code == 422


def test_boundary_params_accepted(client):
    for params in ({"lat": 90.0, "lon": -180.0}, {"lat": -90.0, "lon": 180.0},
                   {"min_el": 0.0, "hours": 1}, {"min_el": 90.0, "hours": 1}):
        response = _get(client, **{**MIDLAT_SITE, **params})
        assert response.status_code == 200
