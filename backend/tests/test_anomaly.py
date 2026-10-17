"""Anomaly-scan tests - hermetic: temp DB, hand-built TLE histories, no network.

The scan reads elements straight from fixed TLE columns, so fixtures build
line pairs with exactly controlled mean motion (-> SMA), inclination and
n-dot. Element math helpers below invert the scan's own a(n) relation so each
case can dial in a precise delta-SMA in km.
"""
import math
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend.app import config, db
from backend.app.models import anomaly

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

NOW = datetime.now(timezone.utc).replace(microsecond=0)
NOW_ISO = NOW.strftime(ISO_FMT)

BASE_MM = 15.5  # rev/day (~a = 6768 km)


# --- TLE fixture builders ------------------------------------------------------

def _cs(line: str) -> str:
    """TLE checksum: sum of digits, '-' counts as 1, mod 10."""
    total = 0
    for ch in line:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return str(total % 10)


def _tle(norad: int, epoch: datetime, mm: float, inc: float = 51.6,
         ndot_half: float = 0.0) -> tuple[str, str]:
    """Checksum-valid TLE pair with controlled mm / inc / n-dot fields."""
    yy = epoch.year % 100
    doy = (epoch - datetime(epoch.year, 1, 1, tzinfo=timezone.utc)
           ).total_seconds() / 86400.0 + 1.0
    mag = abs(ndot_half)
    assert mag < 1.0
    field = ("-" if ndot_half < 0 else " ") + "." + f"{mag:.8f}"[2:]
    l1 = (f"1 {norad:05d}U 24001A   {yy:02d}{doy:012.8f}"
          f" {field}  00000-0  00000-0 0  999")
    l1 += _cs(l1)
    l2 = (f"2 {norad:05d} {inc:8.4f} {100.0:8.4f} 0001000 {0.0:8.4f}"
          f" {0.0:8.4f} {mm:11.8f}    0")
    l2 += _cs(l2)
    assert len(l1) == 69 and len(l2) == 69
    return l1, l2


def _a_from_mm(mm: float) -> float:
    n_rad_s = mm * 2.0 * math.pi / 86400.0
    return (anomaly.MU_KM3_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)


def _mm_for_a(a_km: float) -> float:
    n_rad_s = math.sqrt(anomaly.MU_KM3_S2 / a_km ** 3)
    return n_rad_s * 86400.0 / (2.0 * math.pi)


def _mm_for_delta(delta_km: float) -> float:
    """Mean motion whose SMA is BASE_MM's SMA + delta_km."""
    return _mm_for_a(_a_from_mm(BASE_MM) + delta_km)


def _ndot_half_for_drag(drop_km: float, dt_days: float) -> float:
    """n-dot/2 field whose drag term explains an SMA drop of drop_km over dt.

    Inverts expected = -(2/3)(a/n) * ndot * dt (the scan's documented formula).
    """
    a1 = _a_from_mm(BASE_MM)
    ndot = drop_km * 3.0 * BASE_MM / (2.0 * a1 * dt_days)
    return ndot / 2.0


def _conn(tmp_path):
    conn = db.get_conn(tmp_path / "anomaly.db")
    db.init_db(conn)
    return conn


def _seed_sat(conn, norad: int, name: str, tles) -> None:
    """tles: [(epoch_dt, line1, line2)] appended as the satellite's history."""
    cur = conn.execute(
        "INSERT INTO satellites (norad_id, name, object_type, last_updated)"
        " VALUES (?, ?, 'PAYLOAD', ?)", (norad, name, NOW_ISO))
    sat_id = cur.lastrowid
    for epoch, l1, l2 in tles:
        conn.execute(
            "INSERT INTO tle_history (satellite_id, epoch, line1, line2,"
            " ingested_at) VALUES (?, ?, ?, ?, ?)",
            (sat_id, epoch.strftime(ISO_FMT), l1, l2, NOW_ISO))
    conn.commit()


def _pair(conn, norad, name, hours_apart, mm1, mm2, inc1=51.6, inc2=51.6,
          ndot_half=0.0, last_age_h=2.0):
    """Two-TLE history: epochs now-(last_age_h+hours_apart) and now-last_age_h."""
    e2 = NOW - timedelta(hours=last_age_h)
    e1 = e2 - timedelta(hours=hours_apart)
    _seed_sat(conn, norad, name, [
        (e1, *_tle(norad, e1, mm1, inc1, ndot_half)),
        (e2, *_tle(norad, e2, mm2, inc2, ndot_half)),
    ])


# --- element extraction --------------------------------------------------------

def test_tle_mean_elements_roundtrip():
    l1, l2 = _tle(90001, NOW, BASE_MM, inc=53.21, ndot_half=0.00123456)
    a_km, inc, mm, ndot = anomaly.tle_mean_elements(l1, l2)
    assert mm == pytest.approx(BASE_MM, abs=1e-8)
    assert inc == pytest.approx(53.21, abs=1e-4)
    assert ndot == pytest.approx(2 * 0.00123456, abs=1e-8)
    assert a_km == pytest.approx(_a_from_mm(BASE_MM), abs=1e-6)
    # dialing in a +1 km SMA via mean motion round-trips through the scan math
    _l1b, l2b = _tle(90001, NOW, _mm_for_delta(1.0))
    a2, _, _, _ = anomaly.tle_mean_elements(_l1b, l2b)
    assert a2 - a_km == pytest.approx(1.0, abs=1e-4)


# --- flag rules ----------------------------------------------------------------

def test_sma_raise_flagged(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 90010, 'RAISER', 8.0, BASE_MM, _mm_for_delta(1.0))
    items = anomaly.scan_tle_deltas(conn, since_hours=48)
    conn.close()
    assert len(items) == 1
    item = items[0]
    assert item["kind"] == "sma_raise"
    assert item["norad_id"] == 90010
    assert item["name"] == "RAISER"
    assert item["delta_sma_km"] == pytest.approx(1.0, abs=0.01)
    assert item["dt_hours"] == pytest.approx(8.0, abs=0.1)
    assert item["from_epoch"] < item["to_epoch"]


def test_drag_consistent_drop_not_flagged(tmp_path):
    """A 0.6 km drop the TLE's own n-dot fully explains stays quiet."""
    conn = _conn(tmp_path)
    _pair(conn, 90011, 'DECAYER', 12.0, BASE_MM, _mm_for_delta(-0.6),
          ndot_half=_ndot_half_for_drag(0.6, 0.5))
    items = anomaly.scan_tle_deltas(conn, since_hours=48)
    conn.close()
    assert items == []


def test_beyond_drag_drop_flagged(tmp_path):
    """The same 0.6 km drop with a near-zero drag term is a maneuver signature."""
    conn = _conn(tmp_path)
    _pair(conn, 90012, 'DROPPER', 12.0, BASE_MM, _mm_for_delta(-0.6),
          ndot_half=0.00000500)
    items = anomaly.scan_tle_deltas(conn, since_hours=48)
    conn.close()
    assert len(items) == 1
    item = items[0]
    assert item["kind"] == "sma_drop"
    assert item["delta_sma_km"] == pytest.approx(-0.6, abs=0.01)
    # the drag-expected drop over the interval is tiny vs the observed one
    assert abs(item["expected_drag_dsma_km"]) < 0.1


def test_drop_within_2x_drag_not_flagged(tmp_path):
    """Drop of 1.5x the drag-expected amount: inside the 2x tolerance."""
    conn = _conn(tmp_path)
    _pair(conn, 90013, 'NOISY DECAYER', 12.0, BASE_MM, _mm_for_delta(-0.9),
          ndot_half=_ndot_half_for_drag(0.6, 0.5))
    items = anomaly.scan_tle_deltas(conn, since_hours=48)
    conn.close()
    assert items == []


def test_inclination_change_flagged(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 90014, 'PLANE CHANGER', 8.0, BASE_MM, BASE_MM,
          inc1=51.60, inc2=51.62)
    items = anomaly.scan_tle_deltas(conn, since_hours=48)
    conn.close()
    assert len(items) == 1
    item = items[0]
    assert item["kind"] == "inclination"
    assert item["delta_inc_deg"] == pytest.approx(0.02, abs=1e-3)
    assert abs(item["delta_sma_km"]) < 0.01


def test_sub_threshold_deltas_not_flagged(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 90015, 'QUIET', 8.0, BASE_MM, _mm_for_delta(0.3),
          inc1=51.600, inc2=51.605)
    items = anomaly.scan_tle_deltas(conn, since_hours=48)
    conn.close()
    assert items == []


def test_raise_and_inclination_yield_two_items(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 90016, 'BIG BURN', 8.0, BASE_MM, _mm_for_delta(1.0),
          inc1=51.60, inc2=51.65)
    items = anomaly.scan_tle_deltas(conn, since_hours=48)
    conn.close()
    assert {item["kind"] for item in items} == {"sma_raise", "inclination"}
    assert all(item["norad_id"] == 90016 for item in items)


def test_corrupt_tles_skipped(tmp_path):
    conn = _conn(tmp_path)
    _seed_sat(conn, 90017, 'GARBAGE', [
        (NOW - timedelta(hours=10), 'CORRUPT LINE ONE', 'CORRUPT LINE TWO'),
        (NOW - timedelta(hours=2), 'CORRUPT LINE 1B', 'CORRUPT LINE 2B'),
    ])
    items = anomaly.scan_tle_deltas(conn, since_hours=48)
    conn.close()
    assert items == []


# --- windowing, sorting, capping ------------------------------------------------

def test_window_filtering(tmp_path):
    """A pair straddling the window edge only flags once both TLEs are inside."""
    conn = _conn(tmp_path)
    _pair(conn, 90018, 'OLD BURN', 70.0, BASE_MM, _mm_for_delta(1.0),
          last_age_h=2.0)  # epochs now-72h and now-2h
    assert anomaly.scan_tle_deltas(conn, since_hours=48) == []
    wide = anomaly.scan_tle_deltas(conn, since_hours=96)
    conn.close()
    assert len(wide) == 1 and wide[0]["kind"] == "sma_raise"


def test_successive_pairs_within_history(tmp_path):
    """Three TLEs: only the pair with the raise flags, differenced successively."""
    conn = _conn(tmp_path)
    e1 = NOW - timedelta(hours=20)
    e2 = NOW - timedelta(hours=10)
    e3 = NOW - timedelta(hours=2)
    mm_hi = _mm_for_delta(1.0)
    _seed_sat(conn, 90019, 'STEPPER', [
        (e1, *_tle(90019, e1, BASE_MM)),
        (e2, *_tle(90019, e2, BASE_MM)),           # no change
        (e3, *_tle(90019, e3, mm_hi)),             # +1 km on the last hop
    ])
    items = anomaly.scan_tle_deltas(conn, since_hours=48)
    conn.close()
    assert len(items) == 1
    assert items[0]["from_epoch"] == e2.strftime(ISO_FMT)
    assert items[0]["to_epoch"] == e3.strftime(ISO_FMT)


def test_sorted_by_magnitude_and_capped_at_50(tmp_path):
    conn = _conn(tmp_path)
    for i in range(55):
        _pair(conn, 91100 + i, f'BURNER {i}', 8.0, BASE_MM,
              _mm_for_delta(0.6 + 0.01 * i))
    items = anomaly.scan_tle_deltas(conn, since_hours=48)
    conn.close()
    assert len(items) == 50  # cap
    mags = [abs(item["delta_sma_km"]) for item in items]
    assert mags == sorted(mags, reverse=True)
    assert mags[0] == pytest.approx(0.6 + 0.01 * 54, abs=0.01)
    # the 5 smallest raises fell off the capped list
    assert mags[-1] == pytest.approx(0.6 + 0.01 * 5, abs=0.01)


# --- GET /anomalies -------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "api.db")
    conn = db.get_conn()
    db.init_db(conn)
    _pair(conn, 90020, 'API RAISER', 8.0, BASE_MM, _mm_for_delta(1.0))
    conn.close()
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_get_anomalies_shape(client):
    body = client.get("/anomalies").json()
    assert set(body) == {"hours", "count", "method", "items"}
    assert body["hours"] == 48
    assert body["count"] == len(body["items"]) == 1
    item = body["items"][0]
    assert item["kind"] == "sma_raise"
    assert item["norad_id"] == 90020
    assert "heuristic" in body["method"]


def test_get_anomalies_hours_clamped(client):
    assert client.get("/anomalies?hours=1").json()["hours"] == 6
    assert client.get("/anomalies?hours=6").json()["hours"] == 6
    assert client.get("/anomalies?hours=99999").json()["hours"] == 168
    assert client.get("/anomalies?hours=72").json()["hours"] == 72


def test_get_anomalies_bad_hours_422(client):
    assert client.get("/anomalies?hours=abc").status_code == 422
