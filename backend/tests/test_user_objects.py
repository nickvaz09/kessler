"""User-fleet endpoint tests - hermetic: temp DB, no network, local app.

The router is mounted on a locally built FastAPI app (not backend.app.main)
so these tests cannot race the concurrently owned main.py wiring. Screening
tests plant a real close approach: the user object and a catalog 'crosser'
share an ascending node and reach it simultaneously (same mean motion, same
node, 10 degrees before the node at epoch) in planes 78 degrees apart - a
genuine ~9.7 km/s crossing encounter, not a co-moving pair.
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sgp4.api import Satrec

from backend.app import config, db
from backend.app.api import user_objects
from backend.app.ingestion import celestrak, propagate, screening

# Canonical real ISS TLE (sgp4 reference element set, epoch 2008-09-20) - 
# same fixture as test_ingestion.py.
ISS_L1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
# Real Cosmos-2251 debris line 2 (norad 34427): valid format, wrong object.
DEB_L2 = "2 34427  74.0355 285.1245 0032622 292.7837  67.0056 14.34120736769414"

ITEM_KEYS = {"id", "name", "norad_id", "note", "added_at", "epoch",
             "inclination_deg", "apogee_km", "perigee_km", "period_min"}
EVENT_KEYS = {"secondary", "tca", "miss_distance_m", "relative_velocity_ms",
              "pc", "risk_level"}


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
    """Checksum-valid TLE pair for synthetic geometry (as in test_ingestion)."""
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


def _seed_satellites(conn, entries):
    """entries: [(norad, name, line1, line2)] -> satellite + latest TLE row."""
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


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("KESSLER_DATABASE_URL", raising=False)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    conn = db.get_conn()
    db.init_db(conn)
    conn.close()
    app = FastAPI()
    app.include_router(user_objects.router)
    return TestClient(app)


def _conn():
    conn = db.get_conn()
    return conn


# --- CRUD --------------------------------------------------------------------

def test_post_valid_tle_creates_item(client):
    response = client.post("/user-objects", json={"line1": ISS_L1, "line2": ISS_L2})
    assert response.status_code == 201
    item = response.json()
    assert set(item) == ITEM_KEYS
    assert item["id"] == 1
    assert item["name"] == "USER OBJECT 25544"       # default name
    assert item["norad_id"] == 25544                 # decoded from line 1
    assert item["note"] is None
    assert item["epoch"] == "2008-09-20T12:25:40Z"   # TLE epoch, not now
    assert item["inclination_deg"] == pytest.approx(51.64, abs=0.1)
    assert 90.0 < item["period_min"] < 95.0
    assert 250.0 < item["perigee_km"] < item["apogee_km"] < 500.0
    # added_at is a well-formed ISO Z instant
    datetime.strptime(item["added_at"], "%Y-%m-%dT%H:%M:%SZ")


def test_post_with_name_and_note(client):
    response = client.post("/user-objects", json={
        "name": "MY BIRD", "note": "ops test", "line1": ISS_L1, "line2": ISS_L2})
    assert response.status_code == 201
    item = response.json()
    assert item["name"] == "MY BIRD"
    assert item["note"] == "ops test"


def test_list_returns_items_in_id_order(client):
    assert client.get("/user-objects").json() == {"items": []}
    client.post("/user-objects", json={"line1": ISS_L1, "line2": ISS_L2})
    client.post("/user-objects", json={
        "name": "SECOND", "line1": ISS_L1, "line2": ISS_L2})
    body = client.get("/user-objects").json()
    assert [item["id"] for item in body["items"]] == [1, 2]
    assert all(set(item) == ITEM_KEYS for item in body["items"])
    assert body["items"][1]["name"] == "SECOND"


def test_delete_then_404(client):
    item = client.post(
        "/user-objects", json={"line1": ISS_L1, "line2": ISS_L2}).json()
    response = client.delete(f"/user-objects/{item['id']}")
    assert response.status_code == 204
    assert client.get("/user-objects").json() == {"items": []}
    assert client.delete(f"/user-objects/{item['id']}").status_code == 404


def test_post_name_and_note_length_bounds(client):
    # over the caps: 422 from body validation, nothing stored
    over_name = client.post("/user-objects", json={
        "name": "N" * 65, "line1": ISS_L1, "line2": ISS_L2})
    assert over_name.status_code == 422
    over_note = client.post("/user-objects", json={
        "note": "x" * 501, "line1": ISS_L1, "line2": ISS_L2})
    assert over_note.status_code == 422
    assert client.get("/user-objects").json() == {"items": []}
    # exactly at the caps: accepted
    at_caps = client.post("/user-objects", json={
        "name": "N" * 64, "note": "x" * 500,
        "line1": ISS_L1, "line2": ISS_L2})
    assert at_caps.status_code == 201
    assert at_caps.json()["name"] == "N" * 64


def test_post_at_fleet_cap_413(client, monkeypatch):
    monkeypatch.setattr(user_objects, "_fleet_count",
                        lambda conn: user_objects.USER_OBJECTS_MAX)
    response = client.post(
        "/user-objects", json={"line1": ISS_L1, "line2": ISS_L2})
    assert response.status_code == 413
    assert str(user_objects.USER_OBJECTS_MAX) in response.json()["detail"]
    assert client.get("/user-objects").json() == {"items": []}  # nothing stored


# --- TLE validation ----------------------------------------------------------

@pytest.mark.parametrize("line1,line2,expected", [
    ("hello world", ISS_L2, "line 1"),          # not a TLE at all
    (ISS_L1, "not a tle", "line 2"),            # bad second line
    (ISS_L2, ISS_L1, "line 1"),                 # swapped lines
    (ISS_L1, ISS_L2[:40], "line 2"),            # truncated line 2
    (ISS_L1, DEB_L2, "line 2"),                 # NORAD ids disagree
    (ISS_L1[:20] + "XX" + ISS_L1[22:], ISS_L2, "line 1"),  # corrupt epoch field
])
def test_post_bad_tle_422_names_failing_line(client, line1, line2, expected):
    response = client.post("/user-objects", json={"line1": line1, "line2": line2})
    assert response.status_code == 422
    assert expected in response.json()["detail"]
    assert client.get("/user-objects").json() == {"items": []}  # nothing stored


def test_post_zero_mean_motion_422(client):
    epoch = datetime.now(timezone.utc)
    l1, l2 = make_tle(90001, 51.6, 100.0, 0.0005, 0.0, 10.0, 0.0, epoch)
    response = client.post("/user-objects", json={"line1": l1, "line2": l2})
    assert response.status_code == 422
    assert "line 2" in response.json()["detail"]


def test_post_unpropagatable_tle_422(client):
    # ecc 0.9 at 15.5 rev/day puts perigee deep inside the Earth: the TLE
    # parses but the epoch test propagation fails (sgp4 'decayed').
    epoch = datetime.now(timezone.utc)
    l1, l2 = make_tle(90001, 51.6, 100.0, 0.9, 0.0, 10.0, 15.5, epoch)
    response = client.post("/user-objects", json={"line1": l1, "line2": l2})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "line 2" in detail and "sgp4 error" in detail


# --- On-demand screening -----------------------------------------------------

def _seed_crossing_catalog(epoch):
    """Catalog with two node-crossers and one far sat (see module docstring).

    CROSSER A shares the user object's node arrival time exactly; CROSSER B
    trails A by 0.10 deg mean anomaly (~12 km along-track), so its approach
    is genuinely wider - two events with distinct misses for the sort check.
    FAR SAT lives ~800 km higher and never comes near.
    """
    return [
        (80001, "CROSSER A", *make_tle(80001, 130.0, 40.0, 0.0005, 0.0, 350.00, 15.5, epoch)),
        (80002, "CROSSER B", *make_tle(80002, 130.0, 40.0, 0.0005, 0.0, 349.90, 15.5, epoch)),
        (80003, "FAR SAT",   *make_tle(80003, 98.0, 200.0, 0.0010, 0.0, 50.00, 13.0, epoch)),
    ]


@pytest.fixture()
def screened(client):
    """POST a user object into a planted-crossing catalog and screen it."""
    epoch = datetime.now(timezone.utc)
    conn = _conn()
    _seed_satellites(conn, _seed_crossing_catalog(epoch))
    conn.close()
    u_l1, u_l2 = make_tle(90500, 51.6, 40.0, 0.0005, 0.0, 350.0, 15.5, epoch)
    item = client.post(
        "/user-objects", json={"line1": u_l1, "line2": u_l2}).json()
    response = client.post(f"/user-objects/{item['id']}/screen?window_hours=6")
    assert response.status_code == 200
    return item, response.json()


def test_screen_finds_planted_close_approach(screened):
    _item, body = screened
    assert set(body) == {"screened_at", "window_hours", "events"}
    assert body["window_hours"] == 6
    datetime.strptime(body["screened_at"], "%Y-%m-%dT%H:%M:%SZ")

    secondaries = [ev["secondary"]["norad_id"] for ev in body["events"]]
    assert 80001 in secondaries and 80002 in secondaries
    assert 80003 not in secondaries                  # far sat never close
    assert all(n < user_objects.SYNTHETIC_NORAD_BASE for n in secondaries)

    now = datetime.now(timezone.utc)
    for ev in body["events"]:
        assert set(ev) == EVENT_KEYS
        assert set(ev["secondary"]) == {"norad_id", "name", "object_type"}
        assert 0.0 < ev["miss_distance_m"] < 15000.0
        assert ev["relative_velocity_ms"] > 1000.0   # true crossing encounter
        tca = datetime.strptime(ev["tca"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
        assert now - timedelta(minutes=5) <= tca <= now + timedelta(hours=6, minutes=5)
    by_norad = {ev["secondary"]["norad_id"]: ev for ev in body["events"]}
    assert by_norad[80001]["secondary"]["name"] == "CROSSER A"
    # A shares the node arrival exactly; B trails along-track -> wider miss.
    assert by_norad[80001]["miss_distance_m"] < by_norad[80002]["miss_distance_m"]


def test_screen_events_sorted_by_pc_desc_with_consistent_levels(screened):
    _item, body = screened
    pcs = [ev["pc"] if ev["pc"] is not None else 0.0 for ev in body["events"]]
    assert len(pcs) >= 2
    assert all(0.0 <= pc <= 1.0 for pc in pcs)
    assert pcs == sorted(pcs, reverse=True)
    for ev in body["events"]:
        pc = ev["pc"] if ev["pc"] is not None else 0.0
        expected = ("critical" if pc >= config.PC_CRITICAL
                    else "elevated" if pc >= config.PC_ELEVATED else "nominal")
        assert ev["risk_level"] == expected


def test_screen_persists_nothing(screened):
    item, body = screened
    assert body["events"]                            # the screen did find events
    conn = _conn()
    sats = conn.execute("SELECT COUNT(*) AS n FROM satellites").fetchone()["n"]
    synth = conn.execute(
        "SELECT COUNT(*) AS n FROM satellites WHERE norad_id >= ?",
        (user_objects.SYNTHETIC_NORAD_BASE,)).fetchone()["n"]
    conjs = conn.execute("SELECT COUNT(*) AS n FROM conjunctions").fetchone()["n"]
    cdms = conn.execute("SELECT COUNT(*) AS n FROM cdm_messages").fetchone()["n"]
    kept = conn.execute("SELECT COUNT(*) AS n FROM user_objects").fetchone()["n"]
    conn.close()
    assert sats == 3       # the 3 seeded catalog objects only
    assert synth == 0      # synthetic norad never written to satellites
    assert conjs == 0 and cdms == 0   # on-demand results are not stored
    assert kept == 1 and item["id"] == 1


def test_screen_window_clamp(client, monkeypatch):
    item = client.post(
        "/user-objects", json={"line1": ISS_L1, "line2": ISS_L2}).json()
    seen = {}

    def fake_screen(conn, primaries, window_hours=None, extra=None, **kwargs):
        seen["window_hours"] = window_hours
        seen["primaries"] = primaries
        seen["extra"] = extra
        return []

    monkeypatch.setattr(screening, "screen", fake_screen)
    low = client.post(f"/user-objects/{item['id']}/screen?window_hours=0.01")
    assert low.status_code == 200
    assert low.json()["window_hours"] == 1 == seen["window_hours"]  # min clamp
    high = client.post(f"/user-objects/{item['id']}/screen?window_hours=99999")
    assert high.status_code == 200
    assert high.json()["window_hours"] == 168 == seen["window_hours"]  # max
    assert low.json()["events"] == []
    # the injected extra carries the synthetic norad, which is the sole primary
    syn = user_objects.SYNTHETIC_NORAD_BASE + item["id"]
    assert seen["primaries"] == {syn}
    assert seen["extra"] == [
        {"norad_id": syn, "line1": ISS_L1, "line2": ISS_L2}]


def test_screen_unknown_id_404(client):
    response = client.post("/user-objects/999/screen")
    assert response.status_code == 404


# --- screen() extra-injection unit behavior ----------------------------------

def test_screen_extra_bad_tle_skipped_no_crash(client):
    conn = _conn()
    events = screening.screen(
        conn, {900001}, window_hours=1,
        extra=[{"norad_id": 900001, "line1": "garbage", "line2": "lines"}])
    conn.close()
    assert events == []   # unparseable extra skipped -> no primaries -> empty


# --- Launch-window scan --------------------------------------------------------

MU_KM3_S2 = 398600.4418
SCAN_KEYS = {"days", "method", "duration_s"}
DAY_KEYS = {"date", "events", "min_miss_km", "max_pc"}


def _fit_crosser(user_sat: Satrec, target: datetime, norad: int,
                 incl_deg: float, dmm: float,
                 base_mm: float = 15.5) -> tuple[str, str]:
    """TLE for a catalog object that passes within a few km of the user object
    at ``target`` - and only there within a two-week horizon.

    Construction: the crosser's TLE epoch is at target, so its mean elements
    place it without secular error; the plane (raan) is solved to contain the
    user's actual SGP4 position at target, the perigee (argp, ma=0) is put at
    that point, and eccentricity absorbs the semi-major-axis difference from
    ``dmm``. A short fixed-point iteration against SGP4 itself then cancels
    the J2 short-period offsets (radial -> ecc, along-track -> argp,
    out-of-plane -> raan). ``dmm`` (rev/day, negative so ecc compensation
    stays positive) desynchronizes the periods: adjacent revolutions misalign
    by ~|dmm|*360 s of arrival time (tens of km at crossing speed), and the
    synodic wrap-around is 1/|dmm| days >> 14 - so no other approach in the
    scan window comes near the capture radius.
    """
    jd, fr = propagate.jday_arrays([target])
    err_code, r_u, v_u = user_sat.sgp4(float(jd[0]), float(fr[0]))
    assert err_code == 0
    r_u = np.array(r_u)
    r_u_norm = float(np.linalg.norm(r_u))
    ru_hat = r_u / r_u_norm

    mm2 = base_mm + dmm
    n2 = mm2 * 2.0 * math.pi / 86400.0
    a2 = (MU_KM3_S2 / (n2 * n2)) ** (1.0 / 3.0)
    inc = math.radians(incl_deg)
    # Solve raan so the plane normal is perpendicular to the target direction.
    alpha = math.atan2(ru_hat[1], ru_hat[0])
    rho = math.hypot(ru_hat[0], ru_hat[1])
    q = -ru_hat[2] * math.cos(inc) / (rho * math.sin(inc))
    assert abs(q) <= 1.0, "pick a steeper crosser inclination"
    raan_deg = math.degrees(alpha + math.asin(q)) % 360.0
    ecc = max(1.0 - r_u_norm / a2, 1e-5)   # perigee radius == target radius

    line1 = line2 = None
    for _ in range(8):
        inc_r = math.radians(incl_deg)
        raan_r = math.radians(raan_deg)
        n_hat = np.array([math.sin(inc_r) * math.sin(raan_r),
                          -math.sin(inc_r) * math.cos(raan_r),
                          math.cos(inc_r)])
        node_hat = np.array([math.cos(raan_r), math.sin(raan_r), 0.0])
        m_hat = np.cross(n_hat, node_hat)
        argp = math.degrees(
            math.atan2(float(ru_hat @ m_hat), float(ru_hat @ node_hat))) % 360.0
        line1, line2 = make_tle(norad, incl_deg, raan_deg, ecc, argp,
                                0.0, mm2, target)
        sat = Satrec.twoline2rv(line1, line2)
        err_code, r_c, v_c = sat.sgp4(float(jd[0]), float(fr[0]))
        assert err_code == 0
        r_c = np.array(r_c)
        err = r_c - r_u
        if float(np.linalg.norm(err)) < 1.5:
            break
        # radial -> eccentricity (perigee sits at the encounter point)
        ecc = min(max(ecc + (float(np.linalg.norm(r_c)) - r_u_norm) / a2,
                      1e-5), 0.02)
        # along-track -> argp: rotate the reference direction by the signed
        # in-plane error angle so the recomputed argp shifts by du and the
        # crosser (at perigee, ma=0) lands du further along its orbit.
        rc_hat = r_c / np.linalg.norm(r_c)
        du = math.atan2(float(np.cross(rc_hat, ru_hat) @ n_hat),
                        float(rc_hat @ ru_hat))
        ru_hat = (ru_hat * math.cos(du)
                  + np.cross(n_hat, ru_hat) * math.sin(du))
        ru_hat = ru_hat / np.linalg.norm(ru_hat)
        # out-of-plane -> node rotation about z (lever = (z x n) . r_u)
        oop = float(err @ n_hat)
        lever = float(np.cross([0.0, 0.0, 1.0], n_hat) @ r_u)
        if abs(lever) > 500.0:
            raan_deg = (raan_deg + math.degrees(oop / lever)) % 360.0

    final_sat = Satrec.twoline2rv(line1, line2)
    err_code, r_c, v_c = final_sat.sgp4(float(jd[0]), float(fr[0]))
    miss_km = float(np.linalg.norm(np.array(r_c) - r_u))
    rel_kms = float(np.linalg.norm(np.array(v_c) - np.array(v_u)))
    assert err_code == 0 and miss_km < 10.0, f"plant failed: {miss_km:.1f} km"
    assert rel_kms > 1.0   # genuine crossing, far above the co-moving gate
    return line1, line2


def test_window_scan_buckets_planted_days(client):
    now = datetime.now(timezone.utc)
    u_l1, u_l2 = make_tle(90500, 51.6, 40.0, 0.0005, 0.0, 350.0, 15.5, now)
    user_sat = Satrec.twoline2rv(u_l1, u_l2)
    # Encounters planted at 12:00Z on day 2 and day 5 (well inside their UTC
    # dates, so a +/-60 s coarse-minimum can never straddle a bucket edge).
    date2 = (now + timedelta(days=2)).date()
    date5 = (now + timedelta(days=5)).date()
    t2 = datetime(date2.year, date2.month, date2.day, 12, tzinfo=timezone.utc)
    t5 = datetime(date5.year, date5.month, date5.day, 12, tzinfo=timezone.utc)
    c2 = _fit_crosser(user_sat, t2, 80001, 100.0, -0.025)
    c5 = _fit_crosser(user_sat, t5, 80002, 80.0, -0.03)
    conn = _conn()
    _seed_satellites(conn, [(80001, "DAY2 CROSSER", *c2),
                            (80002, "DAY5 CROSSER", *c5)])
    conn.close()
    item = client.post(
        "/user-objects", json={"line1": u_l1, "line2": u_l2}).json()

    response = client.post(f"/user-objects/{item['id']}/window-scan",
                           json={"days": 7})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == SCAN_KEYS
    assert body["method"] == "coarse-120s"
    assert isinstance(body["duration_s"], (int, float))
    assert body["duration_s"] >= 0.0

    days = body["days"]
    assert all(set(d) == DAY_KEYS for d in days)
    dates = [datetime.strptime(d["date"], "%Y-%m-%d").date() for d in days]
    assert len(days) == 8                          # 7-day window -> 8 UTC dates
    assert dates == [dates[0] + timedelta(days=k) for k in range(len(dates))]
    assert dates[0] <= now.date() + timedelta(days=1)   # starts at scan 'now'

    # The coarse metric is the rectilinear segment minimum (no refinement):
    # its deterministic bound is the capture radius, not the threshold.
    capture_km = (config.SCREENING_THRESHOLD_KM
                  + screening.COARSE_CAPTURE_MARGIN_KM)
    by_date = {d["date"]: d for d in days}
    # Planted days must carry their encounter. Counts are >= 1, not == 1:
    # orbiting pairs re-approach every synodic period, so the *other*
    # planted pair can legitimately dip under the capture radius on the
    # same day (time-of-run dependent geometry).
    for planted in (date2, date5):
        bucket = by_date[planted.isoformat()]
        assert bucket["events"] >= 1
        assert 0.0 <= bucket["min_miss_km"] < capture_km
        assert 0.0 < bucket["max_pc"] <= 1.0
    # bucket self-consistency everywhere: events and stats agree
    for d in days:
        if d["events"] == 0:
            assert d["min_miss_km"] is None and d["max_pc"] is None
        else:
            assert d["events"] <= 2                # only two pairs exist
            assert 0.0 <= d["min_miss_km"] < capture_km


def test_window_scan_persists_nothing(client):
    now = datetime.now(timezone.utc)
    u_l1, u_l2 = make_tle(90500, 51.6, 40.0, 0.0005, 0.0, 350.0, 15.5, now)
    item = client.post(
        "/user-objects", json={"line1": u_l1, "line2": u_l2}).json()
    assert client.post(f"/user-objects/{item['id']}/window-scan",
                       json={"days": 3}).status_code == 200
    conn = _conn()
    sats = conn.execute("SELECT COUNT(*) AS n FROM satellites").fetchone()["n"]
    conjs = conn.execute("SELECT COUNT(*) AS n FROM conjunctions").fetchone()["n"]
    cdms = conn.execute("SELECT COUNT(*) AS n FROM cdm_messages").fetchone()["n"]
    conn.close()
    assert sats == 0 and conjs == 0 and cdms == 0


def test_window_scan_default_days_and_bounds(client):
    item = client.post(
        "/user-objects", json={"line1": ISS_L1, "line2": ISS_L2}).json()
    # no body -> default 14 days -> 15 UTC-date buckets (empty catalog: all 0)
    response = client.post(f"/user-objects/{item['id']}/window-scan")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == SCAN_KEYS and body["method"] == "coarse-120s"
    assert len(body["days"]) == 15
    assert all(d["events"] == 0 and d["min_miss_km"] is None
               and d["max_pc"] is None for d in body["days"])
    # Boundary values are accepted...
    low = client.post(f"/user-objects/{item['id']}/window-scan",
                      json={"days": 3})
    assert low.status_code == 200 and len(low.json()["days"]) == 4
    high = client.post(f"/user-objects/{item['id']}/window-scan",
                       json={"days": 14})
    assert high.status_code == 200 and len(high.json()["days"]) == 15


@pytest.mark.parametrize("days", [2, 15, 0, -1, 7.5, "week"])
def test_window_scan_out_of_range_days_422(client, days):
    item = client.post(
        "/user-objects", json={"line1": ISS_L1, "line2": ISS_L2}).json()
    response = client.post(f"/user-objects/{item['id']}/window-scan",
                           json={"days": days})
    assert response.status_code == 422


def test_window_scan_unknown_id_404(client):
    response = client.post("/user-objects/999/window-scan", json={"days": 7})
    assert response.status_code == 404
    assert client.post("/user-objects/999/window-scan").status_code == 404
