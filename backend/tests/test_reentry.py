"""Reentry watch endpoint tests - hermetic: temp DB, frozen clocks, no network.

TLE fixtures are checksum-valid synthetic lines with CRAFTED ndot fields
(line 1 columns 34-43 hold ndot/2 in rev/day^2). The endpoint's decay rate
must reproduce  (2/3) * a * ndot / n  (a from n via mu, i.e. a ~ n^(-2/3)),
and the est_window buckets divide the perigee margin above the 100 km floor
by that rate. Cache behavior is exercised by monkeypatching the module's
wall clock (reentry._now_s), never by sleeping.
"""
import math
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.app import config, db
from backend.app.api import reentry as reentry_api
from backend.app.ingestion import propagate
from backend.tests.test_ingestion import _checksum, make_tle

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

EPOCH = datetime(2024, 1, 7, 14, 0, 0, tzinfo=timezone.utc)
EPOCH_ISO = "2024-01-07T14:00:00Z"
# frozen "now" 10 h after the TLE epoch -> epoch_age_h == 10.0
FROZEN_NOW = datetime(2024, 1, 8, 0, 0, 0, tzinfo=timezone.utc)

ITEM_KEYS = {"norad_id", "name", "perigee_km", "apogee_km", "epoch",
             "epoch_age_h", "decay_rate_km_day", "est_window"}

MU = 398600.4418
RE = 6378.137
TWO_PI = 2.0 * math.pi


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FROZEN_NOW


def _mean_motion_rev_day(perigee_km: float, ecc: float) -> float:
    a = (RE + perigee_km) / (1.0 - ecc)
    return math.sqrt(MU / a**3) * 86400.0 / TWO_PI


def decay_tle(norad: int, perigee_km: float, ndot_half: float,
              ecc: float = 0.001) -> tuple[str, str]:
    """Checksum-valid TLE with the given perigee and ndot/2 field (rev/day^2)."""
    mm = _mean_motion_rev_day(perigee_km, ecc)
    _, l2 = make_tle(norad, 51.6, 0.0, ecc, 0.0, 0.0, mm, EPOCH)
    yy = EPOCH.year % 100
    doy = (EPOCH - datetime(EPOCH.year, 1, 1, tzinfo=timezone.utc)
           ).total_seconds() / 86400.0 + 1.0
    assert abs(ndot_half) < 1.0
    sign = "-" if ndot_half < 0 else " "
    field = sign + f"{abs(ndot_half):.8f}"[1:]        # e.g. ' .00016717'
    l1 = (f"1 {norad:05d}U 24001A   {yy:02d}{doy:012.8f}"
          f" {field}  00000-0  00000-0 0  999")
    l1 += _checksum(l1)
    # The 8-decimal field quantizes ndot/2 to 1e-8.
    assert len(l1) == 69
    assert float(l1[33:43]) == pytest.approx(ndot_half, abs=5.1e-9)
    return l1, l2


def ndot_half_for_rate(perigee_km: float, rate_km_day: float,
                       ecc: float = 0.001) -> float:
    """ndot/2 field value that yields the given SMA-loss rate (km/day):
    invert rate = (2/3) * a * ndot / n  ->  ndot/2 = 0.75 * rate * n / a."""
    a = (RE + perigee_km) / (1.0 - ecc)
    n = _mean_motion_rev_day(perigee_km, ecc)
    return 0.75 * rate_km_day * n / a


def _seed(entries):
    """entries: [(norad, name, line1, line2)] -> satellite + latest TLE."""
    conn = db.get_conn()
    try:
        for norad, name, l1, l2 in entries:
            cur = conn.execute(
                "INSERT INTO satellites (norad_id, name, object_type, last_updated)"
                " VALUES (?, ?, 'PAYLOAD', ?)", (norad, name, EPOCH_ISO))
            conn.execute(
                "INSERT INTO tle_history (satellite_id, epoch, line1, line2,"
                " ingested_at) VALUES (?, ?, ?, ?, ?)",
                (cur.lastrowid, EPOCH_ISO, l1, l2, EPOCH_ISO))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(reentry_api, "datetime", _FrozenDatetime)
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(reentry_api, "_now_s", lambda: clock["t"])
    reentry_api._cache.clear()
    conn = db.get_conn()
    db.init_db(conn)
    conn.close()
    from backend.app.main import app

    with TestClient(app) as test_client:
        test_client.clock = clock
        yield test_client
    reentry_api._cache.clear()


# --- unit: decay-rate conversion + bucket math --------------------------------

def test_ndot_field_parsing_sign_and_scale():
    l1, _ = decay_tle(90001, 200.0, 0.00016717)
    assert reentry_api._ndot_rev_day2(l1) == pytest.approx(2 * 0.00016717)
    l1_neg, _ = decay_tle(90002, 200.0, -0.00016717)
    assert reentry_api._ndot_rev_day2(l1_neg) == pytest.approx(-2 * 0.00016717)


def test_est_window_buckets_from_margin_over_rate():
    # perigee 200 -> margin 100 km above the 100 km floor.
    assert reentry_api._est_window(200.0, 100.0 / 5.0) == "<7 D"     # 5 d
    assert reentry_api._est_window(200.0, 100.0 / 6.99) == "<7 D"
    assert reentry_api._est_window(200.0, 100.0 / 7.01) == "<30 D"
    assert reentry_api._est_window(200.0, 100.0 / 29.9) == "<30 D"
    assert reentry_api._est_window(200.0, 100.0 / 30.1) == "<90 D"
    assert reentry_api._est_window(200.0, 100.0 / 89.9) == "<90 D"
    assert reentry_api._est_window(200.0, 100.0 / 90.1) == ">90 D"
    # non-positive rate: no decay estimate is possible
    assert reentry_api._est_window(200.0, 0.0) == ">90 D"
    assert reentry_api._est_window(200.0, -0.5) == ">90 D"


# --- endpoint ------------------------------------------------------------------

def test_items_shape_filter_and_sort(client):
    _seed([
        (90001, "DECAY LOW", *decay_tle(90001, 150.0, 0.001)),
        (90002, "DECAY MID", *decay_tle(90002, 200.0, 0.0002)),
        (90003, "ABOVE DEFAULT", *decay_tle(90003, 350.0, 0.0001)),
        (90004, "HIGH SAFE", *decay_tle(90004, 800.0, 0.00001)),
    ])
    body = client.get("/reentry").json()
    assert set(body) == {"count", "items"}
    assert body["count"] == 2
    items = body["items"]
    # Default threshold 250: 150 and 200 qualify, sorted by perigee asc.
    assert [i["norad_id"] for i in items] == [90001, 90002]
    for item in items:
        assert set(item) == ITEM_KEYS
        assert item["epoch"] == EPOCH_ISO
        assert item["epoch_age_h"] == pytest.approx(10.0)
        assert item["est_window"] in ("<7 D", "<30 D", "<90 D", ">90 D")
    assert items[0]["perigee_km"] == pytest.approx(150.0, abs=0.2)
    assert items[0]["apogee_km"] > items[0]["perigee_km"]


def test_decay_rate_matches_documented_conversion(client):
    ndot_half = 0.00016717
    l1, l2 = decay_tle(90010, 220.0, ndot_half)
    _seed([(90010, "RATE CHECK", l1, l2)])
    elements = propagate.orbital_elements(l1, l2)
    expected = (2.0 / 3.0) * elements["semi_major_axis_km"] \
        * (2.0 * ndot_half) / elements["mean_motion_rev_day"]
    item = client.get("/reentry").json()["items"][0]
    assert item["decay_rate_km_day"] == pytest.approx(expected, abs=5e-4)
    assert expected > 0.0


def test_bucket_assignment_through_endpoint(client):
    # perigee 200 (margin 100 km): rates 20 / 5 / 2 / 0.5 km/d -> 5/20/50/200 d.
    _seed([
        (90021, "B7", *decay_tle(90021, 200.0, ndot_half_for_rate(200.0, 20.0))),
        (90022, "B30", *decay_tle(90022, 201.0, ndot_half_for_rate(201.0, 5.0))),
        (90023, "B90", *decay_tle(90023, 202.0, ndot_half_for_rate(202.0, 2.0))),
        (90024, "B90P", *decay_tle(90024, 203.0, ndot_half_for_rate(203.0, 0.5))),
        (90025, "NEG", *decay_tle(90025, 204.0, -0.0001)),
    ])
    windows = {i["norad_id"]: i["est_window"]
               for i in client.get("/reentry").json()["items"]}
    assert windows == {90021: "<7 D", 90022: "<30 D", 90023: "<90 D",
                       90024: ">90 D", 90025: ">90 D"}
    # Negative ndot must also surface a negative displayed rate.
    items = {i["norad_id"]: i for i in client.get("/reentry").json()["items"]}
    assert items[90025]["decay_rate_km_day"] < 0.0


def test_threshold_clamps_120_to_400(client):
    _seed([
        (90031, "P115", *decay_tle(90031, 115.0, 0.001)),
        (90032, "P130", *decay_tle(90032, 130.0, 0.001)),
        (90033, "P350", *decay_tle(90033, 350.0, 0.0001)),
        (90034, "P450", *decay_tle(90034, 450.0, 0.0001)),
    ])
    # 50 clamps up to 120: only the 115 km perigee qualifies.
    low = client.get("/reentry", params={"perigee_km": 50}).json()["items"]
    assert [i["norad_id"] for i in low] == [90031]
    # 1000 clamps down to 400: 450 stays excluded.
    high = client.get("/reentry", params={"perigee_km": 1000}).json()["items"]
    assert [i["norad_id"] for i in high] == [90031, 90032, 90033]
    # in-range thresholds are used as-is (strictly-below comparison)
    mid = client.get("/reentry", params={"perigee_km": 355}).json()["items"]
    assert [i["norad_id"] for i in mid] == [90031, 90032, 90033]


def test_cap_100_lowest_perigees(client):
    _seed([(91000 + i, f"SWARM {i}",
            *decay_tle(91000 + i, 150.0 + 0.5 * i, 0.0005))
           for i in range(105)])
    body = client.get("/reentry").json()
    assert body["count"] == 105  # true match total, reported before the cap
    items = body["items"]
    assert len(items) == 100
    perigees = [i["perigee_km"] for i in items]
    assert perigees == sorted(perigees)
    # The five highest-perigee seeds fell off the capped list.
    assert {i["norad_id"] for i in items} == {91000 + i for i in range(100)}


def test_cache_10min_keyed_by_threshold(client):
    _seed([(90041, "FIRST", *decay_tle(90041, 200.0, 0.0005))])
    t0 = client.clock["t"]
    first = client.get("/reentry").json()["items"]
    assert [i["norad_id"] for i in first] == [90041]

    # New data lands after the scan...
    _seed([(90042, "SECOND", *decay_tle(90042, 180.0, 0.0005))])
    # ...within the TTL the same threshold serves the cached snapshot,
    client.clock["t"] = t0 + 599.0
    assert [i["norad_id"] for i in client.get("/reentry").json()["items"]] \
        == [90041]
    # ...a DIFFERENT threshold is its own cache key (fresh scan sees both),
    other = client.get("/reentry", params={"perigee_km": 300}).json()["items"]
    assert [i["norad_id"] for i in other] == [90042, 90041]
    # ...and past the TTL the default threshold recomputes.
    client.clock["t"] = t0 + 600.0
    assert [i["norad_id"] for i in client.get("/reentry").json()["items"]] \
        == [90042, 90041]


def test_empty_catalog_and_no_matches(client):
    assert client.get("/reentry").json() == {"count": 0, "items": []}
    reentry_api._cache.clear()
    _seed([(90051, "HIGH", *decay_tle(90051, 800.0, 0.0001))])
    assert client.get("/reentry").json() == {"count": 0, "items": []}


def test_unparseable_tle_skipped_not_500(client):
    _seed([
        (90061, "GOOD", *decay_tle(90061, 200.0, 0.0005)),
        (90062, "CORRUPT", "CORRUPT LINE ONE", "CORRUPT LINE TWO"),
    ])
    response = client.get("/reentry")
    assert response.status_code == 200
    assert [i["norad_id"] for i in response.json()["items"]] == [90061]


def test_non_numeric_threshold_422(client):
    assert client.get("/reentry", params={"perigee_km": "low"}).status_code == 422
