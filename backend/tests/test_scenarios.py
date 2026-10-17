"""Time Machine scenarios - hermetic: temp DB, shipped scenario file, no network.

Covers the reconstructed Iridium 33 / Cosmos 2251 scenario end to end: the
live screening code detects it at the historical epoch, the ephemeris has the
/conjunctions/{id}/ephemeris shape, the illustrative breakup cloud behaves
(counts, spreading, determinism), the live-catalog band count, 404s, the
archived-file preference and the gp_history parsing used by
scripts/fetch_scenario_tles.py.
"""
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import config, db, scenarios
from backend.app.ingestion import screening
from backend.app.scenarios import archive, breakup, replay

SID = "iridium-cosmos-2009"
EVENT = datetime(2009, 2, 10, 16, 56, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[2]
BASE_FILE = ROOT / "backend" / "app" / "scenarios" / "iridium_cosmos_2009.json"


def _base():
    return json.loads(BASE_FILE.read_text(encoding="utf-8"))


def _with_epoch(line1: str, yyddd: str) -> str:
    """Replace a TLE line-1 epoch field (cols 19-32)."""
    return line1[:18] + yyddd + line1[32:]


# One in-band payload, one out-of-band payload, one cataloged collision
# fragment in band, one corrupt row (skipped). Mean motion drives altitude:
# 14.28 rev/day ~ 790 km; 15.5 rev/day ~ 420 km.
IN_BAND_L2 = "2 40001  86.4000 122.7790 0002000  90.0000  10.0269 14.28339205    09"
LOW_L2 = "2 40002  51.6400 208.9163 0006317  69.9862  25.2906 15.49560532430089"
FRAG_L2 = "2 34454  74.0000  22.5554 0016000  90.0000 249.8413 14.25313483    09"
L1 = "1 40001U 15001A   24007.58333333  .00000000  00000-0  00000-0 0  9990"


def _seed(conn):
    conn.executescript(f"""
    INSERT INTO satellites (id, norad_id, name, object_type, intl_designator, last_updated) VALUES
        (1, 40001, 'IN BAND SAT', 'PAYLOAD', NULL, '2024-01-08T00:00:00Z'),
        (2, 40002, 'LOW SAT', 'PAYLOAD', NULL, '2024-01-08T00:00:00Z'),
        (3, 34454, 'COSMOS 2251 DEB', 'DEBRIS', NULL, '2024-01-08T00:00:00Z'),
        (4, 40004, 'CORRUPT', 'PAYLOAD', NULL, '2024-01-08T00:00:00Z');
    INSERT INTO tle_history (satellite_id, epoch, line1, line2, ingested_at) VALUES
        (1, '2024-01-07T14:00:00Z', '{L1}', '{IN_BAND_L2}', '2024-01-07T15:00:00Z'),
        (2, '2024-01-07T14:00:00Z', '{L1}', '{LOW_L2}', '2024-01-07T15:00:00Z'),
        (3, '2024-01-07T14:00:00Z', '{L1}', '{FRAG_L2}', '2024-01-07T15:00:00Z'),
        (4, '2024-01-07T14:00:00Z', 'BAD', 'BAD', '2024-01-07T15:00:00Z');
    """)
    conn.commit()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    conn = db.get_conn()
    db.init_db(conn)
    _seed(conn)
    conn.close()
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


def _clear_scenario_caches():
    from backend.app.api import scenarios as api

    replay._cache.clear()
    api._band_cache.clear()
    api._cloud_cache.clear()


@pytest.fixture()
def reconstruction(monkeypatch):
    """Pin the loader to the RECONSTRUCTED scenario file. The repo ships the
    archived twin (real Space-Track element sets), which would otherwise win;
    these tests exercise the reconstruction's known geometry."""
    _clear_scenario_caches()
    monkeypatch.setattr(scenarios, "resolve_file", lambda base: base)
    yield
    _clear_scenario_caches()


@pytest.fixture(scope="module")
def detail():
    from backend.app.api import scenarios as api

    with pytest.MonkeyPatch.context() as mp:        # the reconstruction, as above
        _clear_scenario_caches()
        mp.setattr(scenarios, "resolve_file", lambda base: base)
        out = api.get_scenario(SID)
    _clear_scenario_caches()
    return out


# --- list / detail ----------------------------------------------------------

def test_list_scenarios(client, reconstruction):
    res = client.get("/scenarios")
    assert res.status_code == 200
    body = res.json()
    assert body["count"] >= 1
    item = next(s for s in body["scenarios"] if s["id"] == SID)
    assert item["provenance"] == "reconstructed"
    assert item["event"]["time_utc"] == "2009-02-10T16:56:00Z"


def test_detail_shape_and_provenance(client, reconstruction):
    res = client.get(f"/scenarios/{SID}")
    assert res.status_code == 200
    body = res.json()
    for key in ("id", "title", "provenance", "provenance_label", "provenance_note",
                "replay_hint", "event", "objects", "sources", "screening",
                "encounter"):
        assert key in body
    assert body["provenance"] == "reconstructed"
    assert "fetch_scenario_tles.py" in body["replay_hint"]
    norads = {o["norad_id"]: o for o in body["objects"]}
    assert set(norads) == {24946, 22675}
    assert norads[24946]["role"] == "primary"
    assert norads[24946]["elements"]["inclination_deg"] == pytest.approx(86.4, abs=0.01)
    assert norads[22675]["elements"]["inclination_deg"] == pytest.approx(74.0, abs=0.01)
    assert body["event"]["source"].startswith("NASA Orbital Debris Quarterly News")
    assert "_file" not in body


def test_detail_tracks_are_sgp4_at_the_detecting_run(detail):
    tr = detail["tracks"]
    assert tr["start"] == detail["screening"]["detection"]["run_at"]
    assert tr["step_s"] == 30
    for side in ("primary", "secondary"):
        pts = np.array(tr[side])
        assert pts.shape == (241, 3)
        radius = np.linalg.norm(pts, axis=1)
        assert np.all((radius > 6378 + 700) & (radius < 6378 + 900))


def test_screening_detects_reconstructed_conjunction(detail):
    scr = detail["screening"]
    assert scr["detected"] is True
    det = scr["detection"]
    assert det["miss_distance_m"] < 1000.0
    assert 11_500.0 < det["relative_velocity_ms"] < 11_900.0
    tca = datetime.fromisoformat(det["tca"].replace("Z", "+00:00"))
    assert abs((tca - EVENT).total_seconds()) < 60.0
    # first hourly run whose 72 h window contains the event
    assert det["run_at"] == "2009-02-07T17:00:00Z"
    assert 70.0 < det["hours_before_tca"] <= 72.0
    assert det["risk_level"] in ("elevated", "critical")
    assert 0.0 < det["pc"] <= 1.0
    loc = det["location"]
    # The reconstruction's elements were fitted with a spherical-Earth
    # altitude, so on the WGS-84 ellipsoid it sits ~20 km above the published
    # 789 km; the archived real data (test below) lands on 789.
    assert loc["alt_km"] == pytest.approx(808.4, abs=2.0)
    assert 65.0 < loc["lat_deg"] < 80.0 and 60.0 < loc["lon_deg"] < 140.0


def test_screening_history_pc_rises_as_lead_shrinks(detail):
    hist = detail["screening"]["history"]
    assert len(hist) >= 5 and all(h["detected"] for h in hist)
    leads = [h["hours_before_tca"] for h in hist]
    assert leads == sorted(leads, reverse=True)
    pcs = [h["pc"] for h in hist]
    assert pcs == sorted(pcs)          # sigma shrinks with propagation age


def test_replay_uses_the_live_screening_code(monkeypatch):
    scn = scenarios.get_scenario(SID)
    calls = []
    real = screening.screen

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(screening, "screen", spy)
    run = replay.run_screen(scn, datetime(2009, 2, 9, 12, 0, tzinfo=timezone.utc))
    assert calls and calls[0]["start"].year == 2009
    assert len(calls[0]["catalog"]) == 2
    assert run["detected"] and run["miss_distance_m"] < 1000.0
    # Pc is the live screening Pc at the record_events propagation age.
    age = run["hours_before_tca"] / 24.0
    assert run["pc"] == pytest.approx(
        screening._compute_pc(run["miss_distance_m"], run["relative_velocity_ms"], age),
        rel=1e-3)


def test_run_before_window_does_not_detect():
    scn = scenarios.get_scenario(SID)
    run = replay.run_screen(scn, datetime(2009, 2, 7, 15, 0, tzinfo=timezone.utc))
    assert run["detected"] is False


# --- ephemeris --------------------------------------------------------------

def test_ephemeris_shape_matches_conjunction_ephemeris(client):
    res = client.get(f"/scenarios/{SID}/ephemeris", params={"window_s": 300})
    assert res.status_code == 200
    body = res.json()
    for key in ("conjunction_id", "tca", "window_s", "step_s", "start",
                "gmst_start_rad", "earth_rot_rad_s", "primary", "secondary",
                "tca_state", "sigma_m"):
        assert key in body
    assert body["scenario_id"] == SID
    assert body["window_s"] == 300 and body["step_s"] == 1.0
    for side in ("primary", "secondary"):
        assert len(body[side]["r"]) == 601
        assert all(len(p) == 3 for p in body[side]["r"])
    mid_p = np.array(body["primary"]["r"][300])
    mid_s = np.array(body["secondary"]["r"][300])
    assert np.linalg.norm(mid_p - mid_s) < 12.0      # km: within 1 s of TCA
    st = body["tca_state"]
    assert st["sep_m"] < 1000.0
    assert 11_500.0 < st["rel_vel_ms"] < 11_900.0
    assert 95.0 < st["approach_angle_deg"] < 110.0


def test_ephemeris_window_is_clamped(client):
    body = client.get(f"/scenarios/{SID}/ephemeris", params={"window_s": 5}).json()
    assert body["window_s"] == 60
    assert len(body["primary"]["r"]) == 121


# --- debris -----------------------------------------------------------------

def test_debris_shape_counts_and_labels(client, reconstruction):
    res = client.get(f"/scenarios/{SID}/debris", params={"hours": 24})
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == breakup.MODEL_LABEL
    assert "simplified" in body["model"]
    counts = body["counts"]
    expected = round(0.1 * 1460 ** 0.75 * 0.1 ** -1.71)
    assert counts["modeled_total_ge_10cm"] == expected
    assert sum(counts["by_parent"].values()) == expected
    assert counts["by_parent"]["22675"] > counts["by_parent"]["24946"]  # heavier
    assert counts["drawn"] == len(body["fragments"]) == len(body["positions"])
    assert all(len(r) == len(body["fragment_fields"]) for r in body["fragments"])
    assert counts["in_orbit_modeled"] + counts["reentered_modeled"] == counts["generated"]
    assert body["energy"]["catastrophic"] is True
    assert body["time_utc"] == "2009-02-11T16:56:00Z"
    lo, hi = body["altitude_band_km"]
    assert lo < 789.0 < hi


def test_debris_live_catalog_band_overlap(client):
    live = client.get(f"/scenarios/{SID}/debris").json()["live_catalog"]
    assert live["catalog_objects"] == 3            # corrupt TLE skipped
    assert live["objects_in_band"] == 2            # in-band sat + fragment
    assert live["payloads_in_band"] == 1
    assert live["collision_fragments_cataloged"] == 1
    assert "not a risk" in live["label"]


def test_debris_limit_subsamples(client):
    body = client.get(f"/scenarios/{SID}/debris", params={"limit": 100}).json()
    assert body["counts"]["drawn"] <= 100
    assert body["counts"]["modeled_total_ge_10cm"] > 1000


def test_debris_cloud_spreads_over_time(client):
    def spread(hours):
        body = client.get(f"/scenarios/{SID}/debris", params={"hours": hours}).json()
        pts = np.array([p for p in body["positions"] if p is not None])
        return float(np.mean(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))

    s0 = spread(0.0)
    s1 = spread(0.05)      # 3 minutes: a streak
    s2 = spread(48.0)      # 2 days: a ring
    assert s0 < 1.0        # every fragment starts at the collision point
    assert s0 < s1 < s2


def test_debris_hours_validation(client):
    assert client.get(f"/scenarios/{SID}/debris", params={"hours": -1}).status_code == 422


@pytest.mark.parametrize("path", ["/scenarios/nope", "/scenarios/nope/ephemeris",
                                  "/scenarios/nope/debris"])
def test_unknown_scenario_404(client, path):
    res = client.get(path)
    assert res.status_code == 404
    assert "nope" in res.json()["detail"]


# --- breakup model units -------------------------------------------------------

def test_fragment_count_and_energy():
    assert breakup.fragment_count(1460.0) == round(0.1 * 1460 ** 0.75 * 0.1 ** -1.71)
    e = breakup.specific_energy_j_per_g(560.0, 900.0, 11_700.0)
    assert e == pytest.approx(0.5 * 560 * 11_700 ** 2 / 900_000, rel=1e-9)
    assert e > breakup.CATASTROPHIC_J_PER_G


def test_j2_rates_regress_node_for_prograde_orbits():
    raan_dot, argp_dot, m_dot = breakup.secular_rates(
        np.array([7167.0]), np.array([0.001]), np.radians([74.0]))
    assert raan_dot[0] < 0.0
    deg_per_day = math.degrees(raan_dot[0]) * 86400
    assert -2.5 < deg_per_day < -1.0
    assert m_dot[0] == pytest.approx(math.sqrt(398600.4418 / 7167.0 ** 3), rel=0.01)


def test_generation_is_deterministic_and_starts_at_parent():
    r = [3000.0, 1000.0, 6300.0]
    v = [-6.0, 4.0, 2.0]
    parents = [{"norad_id": 1, "name": "a", "mass_kg": 560, "r": r, "v": v},
               {"norad_id": 2, "name": "b", "mass_kg": 900, "r": r,
                "v": [5.0, 5.0, -1.0]}]
    g1 = breakup.generate(parents, 11_700.0, seed=7)
    g2 = breakup.generate(parents, 11_700.0, seed=7)
    assert np.array_equal(g1["fragments"], g2["fragments"])
    pos = breakup.positions(g1["fragments"], 0.0)
    ok = np.all(np.isfinite(pos), axis=1)
    assert ok.sum() > 0
    assert np.allclose(pos[ok], np.array(r)[None, :], atol=1e-3)


def test_non_catastrophic_breakup_yields_no_fragments():
    parents = [{"norad_id": 1, "name": "a", "mass_kg": 1, "r": [7000, 0, 0],
                "v": [0, 7.5, 0]},
               {"norad_id": 2, "name": "b", "mass_kg": 10000, "r": [7000, 0, 0],
                "v": [0, 7.5, 0.01]}]
    g = breakup.generate(parents, 10.0)
    assert g["energy"]["catastrophic"] is False
    assert g["counts"]["modeled_total_ge_10cm"] == 0


def test_geodetic_altitude_on_the_ellipsoid():
    a = replay._WGS84_A_KM
    b = a * math.sqrt(1.0 - replay._WGS84_E2)
    lat, h = replay.geodetic_lat_alt([a + 789.0, 0.0, 0.0])      # equator
    assert lat == pytest.approx(0.0, abs=1e-9) and h == pytest.approx(789.0, abs=1e-6)
    lat, h = replay.geodetic_lat_alt([0.0, 0.0, b + 789.0])      # north pole
    assert lat == pytest.approx(90.0) and h == pytest.approx(789.0, abs=1e-6)
    # a sphere of the equatorial radius reads ~20 km low at 72.5 deg N
    phi = math.radians(72.5)
    n = a / math.sqrt(1.0 - replay._WGS84_E2 * math.sin(phi) ** 2)
    r = [(n + 789.0) * math.cos(phi), 0.0, (n * (1.0 - replay._WGS84_E2) + 789.0) * math.sin(phi)]
    lat, h = replay.geodetic_lat_alt(r)
    assert lat == pytest.approx(72.5, abs=1e-6) and h == pytest.approx(789.0, abs=1e-6)
    assert 15.0 < h - (math.sqrt(sum(c * c for c in r)) - a) < 25.0


ARCHIVED_FILE = scenarios.SCENARIO_DIR / ("iridium_cosmos_2009" + scenarios.ARCHIVED_SUFFIX)


@pytest.mark.skipif(not ARCHIVED_FILE.exists(), reason="archived element sets not fetched")
def test_archived_real_data_reproduces_the_published_event():
    """The shipped archive (Space-Track gp_history, epochs before the event)
    replayed through the live screening code must land on what NASA published:
    ~16:56 UTC, ~789 km over northern Siberia - and inside the 72 h look-ahead."""
    from backend.app.api import scenarios as api

    _clear_scenario_caches()
    body = api.get_scenario(SID)
    _clear_scenario_caches()
    assert body["provenance"] == "archived"
    det = body["screening"]["detection"]
    tca = datetime.fromisoformat(det["tca"].replace("Z", "+00:00"))
    assert abs((tca - EVENT).total_seconds()) < 60.0
    assert 70.0 < det["hours_before_tca"] <= 72.0
    assert det["risk_level"] in ("elevated", "critical")
    assert det["miss_distance_m"] < 1000.0
    loc = det["location"]
    assert loc["alt_km"] == pytest.approx(789.0, abs=3.0)
    assert 72.0 < loc["lat_deg"] < 73.0


# --- archived element sets ----------------------------------------------------------

def _history_fixture():
    base = _base()
    i33 = next(o for o in base["objects"] if o["norad_id"] == 24946)
    c51 = next(o for o in base["objects"] if o["norad_id"] == 22675)
    return [
        {"NORAD_CAT_ID": "24946", "OBJECT_NAME": "IRIDIUM 33",
         "EPOCH": "2009-02-08 04:12:00.123456",
         "TLE_LINE1": _with_epoch(i33["line1"], "09039.17500000"),
         "TLE_LINE2": i33["line2"]},
        {"NORAD_CAT_ID": "24946", "OBJECT_NAME": "IRIDIUM 33",
         "EPOCH": "2009-02-07 11:43:12",
         "TLE_LINE1": i33["line1"], "TLE_LINE2": i33["line2"]},
        # duplicate epoch: first one wins, no double entry
        {"NORAD_CAT_ID": "24946", "OBJECT_NAME": "IRIDIUM 33",
         "EPOCH": "2009-02-07 11:43:12",
         "TLE_LINE1": i33["line1"], "TLE_LINE2": i33["line2"]},
        # after the collision: must be dropped
        {"NORAD_CAT_ID": "24946", "OBJECT_NAME": "IRIDIUM 33",
         "EPOCH": "2009-02-11 01:00:00",
         "TLE_LINE1": _with_epoch(i33["line1"], "09042.04166667"),
         "TLE_LINE2": i33["line2"]},
        {"NORAD_CAT_ID": "22675", "OBJECT_NAME": "COSMOS 2251",
         "EPOCH": "2009-02-07T09:12:36", "TLE_LINE1": c51["line1"],
         "TLE_LINE2": c51["line2"]},
        # malformed rows: truncated line, wrong norad in the lines, junk
        {"NORAD_CAT_ID": "22675", "EPOCH": "2009-02-08 00:00:00",
         "TLE_LINE1": "1 22675U", "TLE_LINE2": c51["line2"]},
        {"NORAD_CAT_ID": "22675", "EPOCH": "2009-02-08 00:00:00",
         "TLE_LINE1": i33["line1"], "TLE_LINE2": i33["line2"]},
        {"NORAD_CAT_ID": None, "EPOCH": "garbage"},
        "not a dict",
    ]


def test_parse_history_rows_filters_and_sorts():
    out = archive.parse_history_rows(_history_fixture(), before=EVENT,
                                     norads=[24946, 22675])
    assert set(out) == {24946, 22675}
    assert [r["epoch"] for r in out[24946]] == ["2009-02-07T11:43:12Z",
                                                "2009-02-08T04:12:00Z"]
    assert len(out[22675]) == 1
    with pytest.raises(ValueError):
        archive.parse_history_rows({"error": "nope"}, before=EVENT)


def test_history_url_is_one_gp_history_query():
    url = archive.history_url([22675, 24946], datetime(2009, 2, 3), datetime(2009, 2, 11))
    assert "/class/gp_history/" in url
    assert "/NORAD_CAT_ID/22675,24946/" in url
    assert "/EPOCH/2009-02-03--2009-02-11/" in url


def test_build_archived_sets_provenance():
    base = _base()
    elsets = archive.parse_history_rows(_history_fixture(), before=EVENT)
    out = archive.build_archived(base, elsets,
                                 fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert out["provenance"] == "archived"
    assert "reconstruction" not in out
    assert out["archive"]["elset_counts"] == {"24946": 2, "22675": 1}
    for obj in out["objects"]:
        assert "line1" not in obj and obj["elsets"]
    with pytest.raises(ValueError):
        archive.build_archived(base, {24946: elsets[24946]})


def test_loader_prefers_archived_file(tmp_path, monkeypatch):
    base = _base()
    (tmp_path / "iridium_cosmos_2009.json").write_text(json.dumps(base))
    monkeypatch.setattr(scenarios, "SCENARIO_DIR", tmp_path)
    assert scenarios.get_scenario(SID)["provenance"] == "reconstructed"
    elsets = archive.parse_history_rows(_history_fixture(), before=EVENT)
    archived = archive.build_archived(base, elsets)
    (tmp_path / "iridium_cosmos_2009.archived.json").write_text(json.dumps(archived))
    scn = scenarios.get_scenario(SID)
    assert scn["provenance"] == "archived"
    assert len(scenarios.list_scenarios()) == 1     # the twin is not listed twice
    i33 = scenarios.primary(scn)
    # each run only sees element sets published before it
    assert scenarios.elset_at(i33, "2009-02-07T12:00:00Z")["epoch"] == "2009-02-07T11:43:12Z"
    assert scenarios.elset_at(i33, "2009-02-09T00:00:00Z")["epoch"] == "2009-02-08T04:12:00Z"
    assert scenarios.elset_at(i33, "2009-02-01T00:00:00Z") is None


def test_malformed_scenario_file_is_skipped(tmp_path, monkeypatch):
    (tmp_path / "broken.json").write_text('{"id": "x"}')
    (tmp_path / "iridium_cosmos_2009.json").write_text(json.dumps(_base()))
    monkeypatch.setattr(scenarios, "SCENARIO_DIR", tmp_path)
    assert [s["id"] for s in scenarios.list_scenarios()] == [SID]


def test_fetch_script_end_to_end_with_mock_transport(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "fetch_scenario_tles", ROOT / "scripts" / "fetch_scenario_tles.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setenv("SPACETRACK_IDENTITY", "user@example.com")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "hunter2")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url)))
        if request.url.path.endswith("/ajaxauth/login"):
            return httpx.Response(200, text='""')
        assert "/class/gp_history/" in str(request.url)
        return httpx.Response(200, json=_history_fixture())

    out = tmp_path / "out.archived.json"
    assert mod.main(["--out", str(out)], transport=httpx.MockTransport(handler)) == 0
    data = json.loads(out.read_text())
    assert data["provenance"] == "archived"
    assert [m for m, _u in seen] == ["POST", "GET"]    # one login, one query
    assert all("hunter2" not in u for _m, u in seen)


def test_archived_sets_respect_publication_time():
    """A set whose epoch precedes a run but which Space-Track published after
    it must not be visible to that run (no look-ahead)."""
    from backend.app import scenarios as scn_pkg

    rows = [r for r in _history_fixture() if isinstance(r, dict)]
    for r in rows:
        if r.get("EPOCH") == "2009-02-07T09:12:36":
            r["CREATION_DATE"] = "2009-02-07 15:00:00"
    out = archive.parse_history_rows(rows, before=EVENT, norads=[22675])
    rec = out[22675][0]
    assert rec["published"] == "2009-02-07T15:00:00Z"
    obj = {"elsets": [rec]}
    assert scn_pkg.elset_at(obj, "2009-02-07T12:00:00Z") is None     # epoch passed, not yet published
    assert scn_pkg.elset_at(obj, "2009-02-07T16:00:00Z") is rec
