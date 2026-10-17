"""Offline tests for the ingestion layer: TLE parsing, propagation sanity,
conjunction screening on synthetic geometry, and event recording dedupe.

No network access; TLEs are fixture constants or synthesized with valid
checksums at a chosen epoch. DBs are temp files under tmp_path.
"""
from datetime import datetime, timedelta, timezone

import pytest

from backend.app import db
from backend.app.ingestion import celestrak, propagate, screening
from backend.app.ingestion.screening import ConjunctionEvent

# canonical real ISS TLE (sgp4 reference element set, epoch 2008-09-20)
ISS_NAME = "ISS (ZARYA)"
ISS_L1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

# Real Cosmos-2251 collision debris TLE (fixture for parse/classification).
DEB_NAME = "COSMOS 2251 DEB"
DEB_L1 = "1 34427U 93036RU  24001.43908906  .00001240  00000-0  56370-3 0  9992"
DEB_L2 = "2 34427  74.0355 285.1245 0032622 292.7837  67.0056 14.34120736769414"

RB_NAME = "SL-16 R/B"

GROUP_TEXT = f"""{ISS_NAME}
{ISS_L1}
{ISS_L2}

{DEB_NAME}
{DEB_L1}
{DEB_L2}

{RB_NAME}
1 22803U 93059B   24001.50000000  .00000100  00000-0  10000-3 0  9997
2 22803  71.0100 100.0000 0020000 100.0000 260.0000 14.20000000 12345
"""


def _checksum(body: str) -> str:
    total = 0
    for ch in body:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return str(total % 10)


def make_tle(norad: int, incl: float, raan: float, ecc: float, argp: float,
             ma: float, mm: float, epoch: datetime) -> tuple[str, str]:
    """Build a checksum-valid TLE line pair for synthetic screening geometry."""
    yy = epoch.year % 100
    doy = (epoch - datetime(epoch.year, 1, 1, tzinfo=timezone.utc)
           ).total_seconds() / 86400.0 + 1.0
    l1 = (f"1 {norad:05d}U 24001A   {yy:02d}{doy:012.8f}"
          f"  .00000000  00000-0  00000-0 0  999")
    l1 += _checksum(l1)
    ecc_str = f"{ecc:.7f}"[2:]
    l2 = (f"2 {norad:05d} {incl:8.4f} {raan:8.4f} {ecc_str} {argp:8.4f}"
          f" {ma:8.4f} {mm:11.8f}    0")
    l2 += _checksum(l2)
    assert len(l1) == 69 and len(l2) == 69
    return l1, l2


def _make_conn(tmp_path):
    conn = db.get_conn(tmp_path / "test.db")
    db.init_db(conn)
    return conn


def _seed_satellites(conn, entries):
    """entries: [(norad, name, line1, line2)] -> insert satellite + latest TLE."""
    now = datetime.now(timezone.utc).strftime(celestrak.ISO_FMT)
    for norad, name, l1, l2 in entries:
        cur = conn.execute(
            "INSERT INTO satellites (norad_id, name, object_type, last_updated)"
            " VALUES (?, ?, 'PAYLOAD', ?)", (norad, name, now))
        conn.execute(
            "INSERT INTO tle_history (satellite_id, epoch, line1, line2, ingested_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (cur.lastrowid, celestrak.tle_epoch(l1).strftime(celestrak.ISO_FMT),
             l1, l2, now))
    conn.commit()


# --- celestrak parsing -------------------------------------------------------

def test_parse_tle_text_records():
    records = celestrak.parse_tle_text(GROUP_TEXT)
    assert len(records) == 3
    iss, deb, rb = records

    assert iss["norad_id"] == 25544
    assert iss["name"] == ISS_NAME
    assert iss["object_type"] == "PAYLOAD"
    assert iss["intl_designator"] == "1998-067A"
    assert iss["line1"] == ISS_L1
    assert iss["line2"] == ISS_L2
    assert iss["epoch"] == "2008-09-20T12:25:40Z"

    assert deb["norad_id"] == 34427
    assert deb["name"] == DEB_NAME  # trailing whitespace stripped
    assert deb["object_type"] == "DEBRIS"
    assert deb["intl_designator"] == "1993-036RU"
    assert deb["epoch"].startswith("2024-01-01T")

    assert rb["norad_id"] == 22803
    assert rb["object_type"] == "ROCKET BODY"


def test_tle_epoch_year_windowing():
    # yy < 57 -> 20xx, else 19xx
    ep_2008 = celestrak.tle_epoch(ISS_L1)
    assert ep_2008.year == 2008
    l1_1993 = "1 22803U 93059B   93250.50000000  .00000100  00000-0  10000-3 0  9997"
    assert celestrak.tle_epoch(l1_1993).year == 1993
    expected = datetime(2008, 9, 20, 12, 25, 40, tzinfo=timezone.utc)
    assert abs((ep_2008 - expected).total_seconds()) < 1.0


def test_merge_records_keeps_newest_epoch():
    old = {"norad_id": 1, "epoch": "2024-01-01T00:00:00Z", "name": "OLD"}
    new = {"norad_id": 1, "epoch": "2024-06-01T00:00:00Z", "name": "NEW"}
    merged = celestrak.merge_records([old, new])
    assert merged[1]["name"] == "NEW"
    merged = celestrak.merge_records([new, old])
    assert merged[1]["name"] == "NEW"


def test_upsert_catalog_and_tle_dedupe(tmp_path):
    conn = _make_conn(tmp_path)
    records = {r["norad_id"]: r for r in celestrak.parse_tle_text(GROUP_TEXT)}
    stats = celestrak.upsert_catalog(conn, records)
    assert stats == {"fetched": 3, "satellites_new": 3, "tles_added": 3}

    # same records again: no new satellites, no new TLE rows (epoch UNIQUE)
    stats2 = celestrak.upsert_catalog(conn, records)
    assert stats2 == {"fetched": 3, "satellites_new": 0, "tles_added": 0}
    n_tles = conn.execute("SELECT COUNT(*) AS n FROM tle_history").fetchone()["n"]
    assert n_tles == 3
    conn.close()


# --- propagation -------------------------------------------------------------

def test_orbital_elements_iss_sanity():
    el = propagate.orbital_elements(ISS_L1, ISS_L2)
    assert el["inclination_deg"] == pytest.approx(51.64, abs=0.1)
    assert 90.0 < el["period_min"] < 95.0
    assert el["eccentricity"] == pytest.approx(0.0006703, abs=1e-7)
    assert 250.0 < el["perigee_km"] < el["apogee_km"] < 500.0
    assert 6600.0 < el["semi_major_axis_km"] < 6800.0
    assert el["mean_motion_rev_day"] == pytest.approx(15.72, abs=0.05)
    assert el["epoch"] == "2008-09-20T12:25:40Z"


def test_positions_at_shapes_and_radius():
    epoch = celestrak.tle_epoch(ISS_L1)
    times = [epoch + timedelta(minutes=10 * i) for i in range(6)]
    r, v = propagate.positions_at(ISS_L1, ISS_L2, times)
    assert r.shape == (6, 3) and v.shape == (6, 3)
    for row in r:
        radius = (row[0] ** 2 + row[1] ** 2 + row[2] ** 2) ** 0.5
        assert 6600.0 < radius < 6800.0  # LEO altitude in km


def test_sample_orbit_closes_after_one_period():
    path = propagate.sample_orbit(ISS_L1, ISS_L2, n=181)
    assert len(path) == 181
    assert all(len(p) == 3 for p in path)
    first, last = path[0], path[-1]
    gap = sum((a - b) ** 2 for a, b in zip(first, last)) ** 0.5
    assert gap < 100.0  # returns near the start after exactly one period


# --- screening ---------------------------------------------------------------

def test_screening_finds_synthetic_close_pair(tmp_path):
    conn = _make_conn(tmp_path)
    epoch = datetime.now(timezone.utc)
    # Two near-identical LEO orbits offset slightly in mean anomaly: a small
    # persistent along-track separation (~6 km), well under the threshold.
    a_l1, a_l2 = make_tle(90001, 51.6, 100.0, 0.0005, 0.0, 10.00, 15.5, epoch)
    b_l1, b_l2 = make_tle(90002, 51.6, 100.0, 0.0005, 0.0, 10.05, 15.5, epoch)
    # A distant object ~800 km higher: never within threshold of the pair.
    c_l1, c_l2 = make_tle(90003, 98.0, 200.0, 0.0010, 0.0, 50.00, 13.0, epoch)
    _seed_satellites(conn, [
        (90001, "PRIMARY SAT", a_l1, a_l2),
        (90002, "NEIGHBOR SAT", b_l1, b_l2),
        (90003, "FAR SAT", c_l1, c_l2),
    ])

    events = screening.screen(conn, {90001}, window_hours=1,
                              threshold_km=15.0, step_s=60.0, min_rel_vel_ms=0.0)
    pairs = {(ev.primary_norad, ev.secondary_norad) for ev in events}
    assert (90001, 90002) in pairs
    assert all(sec != 90003 for _p, sec in pairs)

    ev = next(e for e in events
              if (e.primary_norad, e.secondary_norad) == (90001, 90002))
    assert 100.0 < ev.miss_distance_m < 15000.0
    assert 0.0 <= ev.relative_velocity_ms < 1000.0  # co-orbital pair drifts slowly
    assert ev.tca.tzinfo is not None
    now = datetime.now(timezone.utc)
    assert now - timedelta(minutes=5) <= ev.tca <= now + timedelta(hours=1, minutes=5)
    conn.close()


def test_screening_catches_fast_crossing_between_samples(tmp_path):
    """Aliasing regression: a ~10.8 km/s crossing whose true miss is ~6 km
    but whose nearest 60 s sample is >15 km away. The sampled-distance gate
    alone misses it; the closed-form segment-minimum gate must catch it."""
    import numpy as np
    from sgp4.api import Satrec, jday

    conn = _make_conn(tmp_path)
    epoch = datetime.now(timezone.utc)
    # Same shell and node, 10 deg vs 100 deg inclination: both reach the
    # ascending node ~23 min in, 0.28 deg of mean anomaly apart.
    a_l1, a_l2 = make_tle(90001, 10.0, 100.0, 0.0005, 0.0, 270.00, 15.5, epoch)
    b_l1, b_l2 = make_tle(90002, 100.0, 100.0, 0.0005, 0.0, 270.28, 15.5, epoch)
    _seed_satellites(conn, [
        (90001, "PRIMARY SAT", a_l1, a_l2),
        (90002, "CROSSING SAT", b_l1, b_l2),
    ])

    # self-check the geometry: every 60 s sample stays outside 15 km
    jd, fr = jday(epoch.year, epoch.month, epoch.day, epoch.hour,
                  epoch.minute, epoch.second + epoch.microsecond / 1e6)
    secs = np.arange(0.0, 3600.0, 60.0)
    frs = fr + secs / 86400.0
    jds = np.full_like(frs, jd)
    _, ra, _ = Satrec.twoline2rv(a_l1, a_l2).sgp4_array(jds, frs)
    _, rb, _ = Satrec.twoline2rv(b_l1, b_l2).sgp4_array(jds, frs)
    assert np.linalg.norm(ra - rb, axis=1).min() > 15.0

    events = screening.screen(conn, {90001}, window_hours=1,
                              threshold_km=15.0, step_s=60.0)
    ev = next(e for e in events
              if (e.primary_norad, e.secondary_norad) == (90001, 90002))
    assert 3000.0 < ev.miss_distance_m < 10000.0   # true miss ~6.1 km
    assert ev.relative_velocity_ms > 6000.0
    conn.close()


def test_screening_primary_pair_norad_ascending_once(tmp_path):
    conn = _make_conn(tmp_path)
    epoch = datetime.now(timezone.utc)
    a_l1, a_l2 = make_tle(90001, 51.6, 100.0, 0.0005, 0.0, 10.00, 15.5, epoch)
    b_l1, b_l2 = make_tle(90002, 51.6, 100.0, 0.0005, 0.0, 10.05, 15.5, epoch)
    _seed_satellites(conn, [
        (90001, "PRIMARY A", a_l1, a_l2),
        (90002, "PRIMARY B", b_l1, b_l2),
    ])
    # Both objects are primaries: exactly one event, norad-ascending ordering.
    events = screening.screen(conn, {90001, 90002}, window_hours=1,
                              threshold_km=15.0, step_s=60.0, min_rel_vel_ms=0.0)
    assert len(events) == 1
    assert (events[0].primary_norad, events[0].secondary_norad) == (90001, 90002)
    conn.close()


# --- record_events dedupe ----------------------------------------------------

def test_record_events_dedupe(tmp_path):
    conn = _make_conn(tmp_path)
    epoch = datetime.now(timezone.utc)
    a_l1, a_l2 = make_tle(90001, 51.6, 100.0, 0.0005, 0.0, 10.00, 15.5, epoch)
    b_l1, b_l2 = make_tle(90002, 51.6, 100.0, 0.0005, 0.0, 10.05, 15.5, epoch)
    _seed_satellites(conn, [
        (90001, "PRIMARY SAT", a_l1, a_l2),
        (90002, "NEIGHBOR SAT", b_l1, b_l2),
    ])

    now = datetime.now(timezone.utc).replace(microsecond=0)
    now_iso = now.strftime(celestrak.ISO_FMT)
    tca = now + timedelta(hours=2)

    ev = ConjunctionEvent(90001, 90002, tca, 5000.0, 10.0)
    stats1 = screening.record_events(conn, [ev], now_iso)
    assert stats1 == {"conjunctions_new": 1, "cdms_added": 1}

    # Same pair, TCA shifted by 10 min (< 30 min tolerance): same event - 
    # TCA updated in place, no new conjunction, one more CDM.
    tca2 = tca + timedelta(minutes=10)
    ev2 = ConjunctionEvent(90001, 90002, tca2, 4800.0, 11.0)
    later_iso = (now + timedelta(hours=1)).strftime(celestrak.ISO_FMT)
    stats2 = screening.record_events(conn, [ev2], later_iso)
    assert stats2 == {"conjunctions_new": 0, "cdms_added": 1}

    conjs = conn.execute(
        "SELECT time_of_closest_approach FROM conjunctions").fetchall()
    assert len(conjs) == 1
    assert conjs[0]["time_of_closest_approach"] == tca2.strftime(celestrak.ISO_FMT)
    n_cdms = conn.execute("SELECT COUNT(*) AS n FROM cdm_messages").fetchone()["n"]
    assert n_cdms == 2

    # same pair but TCA 5 h away (> tolerance): a genuinely new conjunction
    ev3 = ConjunctionEvent(90001, 90002, tca + timedelta(hours=5), 9000.0, 12.0)
    stats3 = screening.record_events(conn, [ev3], later_iso)
    assert stats3 == {"conjunctions_new": 1, "cdms_added": 1}
    n_conjs = conn.execute("SELECT COUNT(*) AS n FROM conjunctions").fetchone()["n"]
    assert n_conjs == 2

    # CDM rows carry the screening source and real geometry.
    row = conn.execute(
        "SELECT * FROM cdm_messages ORDER BY id LIMIT 1").fetchone()
    assert row["source"] == "screening"
    assert row["miss_distance_m"] == 5000.0
    assert row["relative_velocity_ms"] == 10.0
    conn.close()


# --- refresh lock ------------------------------------------------------------

def test_refresh_busy_while_another_thread_holds_lock(tmp_path):
    """A refresh raises RefreshBusyError while a concurrent one is in flight -
    and the rejected call writes no refresh_runs row."""
    import threading

    from backend.app.ingestion import pipeline

    conn = _make_conn(tmp_path)
    held = threading.Event()
    release = threading.Event()

    def holder():
        with pipeline._refresh_lock:
            held.set()
            release.wait(timeout=10.0)

    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(timeout=10.0)
    try:
        with pytest.raises(pipeline.RefreshBusyError):
            pipeline.refresh(conn)
    finally:
        release.set()
        thread.join(timeout=10.0)
    n_runs = conn.execute(
        "SELECT COUNT(*) AS n FROM refresh_runs").fetchone()["n"]
    assert n_runs == 0
    assert issubclass(pipeline.RefreshBusyError, RuntimeError)
    conn.close()


def test_refresh_releases_lock_on_success_and_failure(tmp_path, monkeypatch):
    from backend.app.ingestion import pipeline

    conn = _make_conn(tmp_path)
    monkeypatch.setattr(
        pipeline, "_run_cycle",
        lambda conn, demo, full_catalog: {"fetched": 0})
    assert pipeline.refresh(conn) == {"fetched": 0}
    assert not pipeline._refresh_lock.locked()

    def boom(conn, demo, full_catalog):
        raise RuntimeError("cycle blew up")

    monkeypatch.setattr(pipeline, "_run_cycle", boom)
    with pytest.raises(RuntimeError, match="cycle blew up"):
        pipeline.refresh(conn)
    assert not pipeline._refresh_lock.locked()  # a raise never wedges the lock
    conn.close()
