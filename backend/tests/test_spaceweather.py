"""Space weather: NOAA parsing, the small physics, and fail-safe routes.

No network: the NOAA feeds are replaced with recorded-shape fixtures."""
import math

import pytest
from fastapi.testclient import TestClient

from backend.app import spaceweather as sw


def test_kp_to_ap_follows_the_bartels_table():
    assert sw.kp_to_ap(0) == 0 and sw.kp_to_ap(5) == 48 and sw.kp_to_ap(9) == 400
    assert sw.kp_to_ap(7) == 132 and 39 < sw.kp_to_ap(4.8) < 48


def test_flare_classes():
    assert sw.flare_class(1.55e-6) == "C1.6"
    assert sw.flare_class(2.8e-4) == "X2.8"
    assert sw.flare_class(3e-8) == "A3.0" and sw.flare_class(None) is None


def test_magnetopause_is_pushed_in_by_fast_dense_wind():
    calm = sw.magnetopause_re(5, 400, 0)["standoff_re"]
    storm = sw.magnetopause_re(22, 830, -10)["standoff_re"]
    assert 9.5 < calm < 11.5 and storm < 7 < calm


def test_decay_and_along_track_error_by_hand():
    # 3U CubeSat, B = 2.2 * 0.03 / 4 = 0.0165 m^2/kg, rho 5e-13 kg/m^3, 500 km
    d = sw.decay_rate_m_day(5e-13, 0.0165, 500)
    a = (sw.RE_KM + 500) * 1e3
    assert d == pytest.approx(5e-13 * 0.0165 * math.sqrt(sw.MU * a) * 86400)
    assert 30 < d < 45
    # the error grows with the square of time
    e24, e48 = (sw.along_track_error_m(d, 500, h) for h in (24, 48))
    assert e48 == pytest.approx(4 * e24)


def test_storms_raise_density_most_at_higher_altitudes():
    pytest.importorskip("pymsis")
    q = sw.densities([400, 800], 130, 130, 4)
    g5 = sw.densities([400, 800], 130, 130, 400)
    r400, r800 = g5[0] / q[0], g5[1] / q[1]
    assert 1.5 < r400 < r800 < 25          # the high thermosphere responds most


FIX = {
    "kp": [{"time_tag": "2026-10-10T12:00:00", "Kp": 5.33}, {"time_tag": "2026-10-10T15:00:00", "Kp": 7.0}],
    "kp_forecast": [{"time_tag": "2026-10-11T00:00:00", "kp": 6.0, "observed": "predicted"}],
    "scales": {"0": {"DateStamp": "2026-10-10", "R": {"Scale": "0", "Text": "none"},
                     "S": {"Scale": "0", "Text": "none"}, "G": {"Scale": "3", "Text": "strong"}}},
    "xray": [{"time_tag": "2026-10-10T19:00:00Z", "energy": "0.1-0.8nm", "flux": 1.5e-6}],
    "wind_speed": [{"proton_speed": 830, "time_tag": "2026-10-10T19:08:00Z"}],
    "wind_mag": [{"bt": 19, "bz_gsm": -2, "time_tag": "2026-10-10T19:08:00Z"}],
    "wind_plasma": [{"active": True, "proton_density": 22.0, "time_tag": "2026-10-10T19:08:00"}],
    "f107": [{"time_tag": "2026-10-10T17:00:00", "flux": 132.0},
             {"time_tag": "2026-10-10T12:00:00", "flux": 999.0},
             {"time_tag": "2026-10-09T17:00:00", "flux": 120.0}],
    "aurora": {"Observation Time": "t", "Forecast Time": "f",
               "coordinates": [[10, 60, 40], [10, 58, 12], [10, 50, 3], [10, -61, 20]]},
}


def test_build_parses_every_feed_and_dedupes_daily_flux():
    out = sw._build(FIX, {k: "ok" for k in FIX})
    assert out["scales"]["G"] == {"scale": 3.0, "text": "strong"}
    assert out["flare"]["class"] == "C1.5" and out["kp"]["now"] == 7.0
    assert out["solar_wind"]["density_cc"] == 22.0 and out["solar_wind"]["standoff_re"] < 8
    assert out["f107"]["now"] == 132.0 and out["f107"]["mean"] == 126.0   # one value per day
    assert out["aurora"]["north_edge"] == [(10, 58)] and out["aurora"]["south_edge"] == [(10, -61)]


def test_routes_degrade_and_validate(monkeypatch):
    from backend.app.main import app

    monkeypatch.setenv("KESSLER_SPACEWEATHER", "1")       # exercise the real fetch path
    sw._bg.update(running=False, fail_t=0.0)
    monkeypatch.setattr(sw, "_fetch_all", lambda: ({}, {k: "unavailable (ConnectError)" for k in sw.FEEDS}))
    sw._cache.update(t=0.0, data=None)
    with TestClient(app) as c:
        r = c.get("/spaceweather")
        assert r.status_code == 200 and r.json()["stale"] is True
        assert "scenarios" not in r.json()            # live data only, no what-ifs
        assert c.get("/spaceweather/orbit").status_code == 503
        assert c.get("/spaceweather/orbit?alt_km=5").status_code == 422
        assert c.get("/spaceweather/impact").json()["available"] is False


def test_orbit_effects_use_the_live_conditions(monkeypatch):
    """With live data the calculator applies the current Kp to the orbit."""
    pytest.importorskip("pymsis")
    from backend.app.main import app
    monkeypatch.setenv("KESSLER_SPACEWEATHER", "1")
    monkeypatch.setattr(sw, "storm_table", lambda *a, **k: None)
    monkeypatch.setattr(sw, "_fetch_all", lambda: (dict(FIX), {k: "ok" for k in FIX}))
    sw._cache.update(t=0.0, data=None)
    sw._bg.update(running=False, fail_t=0.0)
    with TestClient(app) as c:
        o = c.get("/spaceweather/orbit?alt_km=550").json()
        assert o["kp"] == 7.0 and o["density_ratio"] > 1.5
        assert o["decay_m_day"] > o["decay_quiet_m_day"]
    sw._cache.update(t=0.0, data=None)


# --- the storm layer: Kp record, drift model, live access ---------------------

from datetime import datetime, timedelta, timezone  # noqa: E402

ISS1 = "1 25544U 98067A   26283.50000000  .00020000  00000-0  35000-3 0  9990"
ISS2 = "2 25544  51.6400 200.0000 0005000  90.0000 270.0000 15.50000000000000"


def _flat_ctx(kp_by_interval, start, ratio_at_kp):
    """A StormContext with a hand-made ratio table: ratio(kp) at every
    altitude, and 3-hourly Kp intervals from `start`."""
    series = [[(start + timedelta(hours=3 * k)).strftime("%Y-%m-%dT%H:%M:%SZ"), kp, "observed"]
              for k, kp in enumerate(kp_by_interval)]
    table = [[ratio_at_kp(lvl)] * len(sw.ALTS_KM) for lvl in sw.KP_LEVELS]
    return sw.StormContext({"updated": "u", "kp": {"now": kp_by_interval[-1]},
                            "kp_series": series,
                            "storm": {"kp_levels": sw.KP_LEVELS, "alts_km": sw.ALTS_KM,
                                      "ratio": table}})


def test_kp_series_merges_observed_then_forecast_in_order():
    obs = [("2026-10-10T12:00:00", 5.33), ("2026-10-10T15:00:00", 7.0)]
    fc = [("2026-10-10T15:00:00", 6.0, "estimated"),      # observed wins
          ("2026-10-10T18:00:00", 6.33, "estimated"),
          ("2026-10-11T00:00:00", 4.0, "predicted"),
          ("2026-10-10T21:00:00", 5.0, "predicted")]
    rows = sw._kp_series(obs, fc)
    assert [r[0][11:13] for r in rows] == ["12", "15", "18", "21", "00"]
    assert rows[1] == ["2026-10-10T15:00:00Z", 7.0, "observed"]
    assert rows[-1][2] == "predicted"


def test_drag_profile_reads_the_tle_drag_term():
    a, alt, ndot, epoch_s, perigee, apogee = sw.drag_profile(ISS1, ISS2)
    # TLE field .00020000 is ndot/2 in rev/day^2 -> true ndot in rad/s^2
    assert ndot == pytest.approx(2 * 0.0002 * 2 * math.pi / 86400 ** 2, rel=1e-9)
    assert 380 < perigee < alt < apogee < 450
    assert datetime.fromtimestamp(epoch_s, timezone.utc).date().isoformat() == "2026-10-10"


def _epoch_dt(prof):
    return datetime.fromtimestamp(prof[3], timezone.utc)


def test_drift_matches_the_closed_form_for_a_storm_after_the_fit():
    prof = sw.drag_profile(ISS1, ISS2)
    ep = _epoch_dt(prof)
    # quiet through the TLE's 36 h fit window, then a steady storm (R = 2.5)
    ctx = _flat_ctx([1.0] * 12 + [7.0] * 12, ep - timedelta(hours=36),
                    lambda kp: 1.0 if kp < 7 else 2.5)
    T = 24 * 3600.0
    got = ctx.drift_m(prof, prof[3], prof[3] + T)
    # constant extra ndot (R/R_fit - 1 = 1.5): a * 1.5 * ndot * T^2 / 2
    assert got == pytest.approx(prof[0] * 1.5 * prof[2] * T * T / 2, rel=1e-9)
    # quadratic in time; zero for no span or above the drag cap
    assert ctx.drift_m(prof, prof[3], prof[3] + 2 * T) == pytest.approx(4 * got, rel=1e-9)
    assert ctx.drift_m(prof, prof[3] + T, prof[3] + T) == 0.0
    high = (prof[0], 1500.0, prof[2], prof[3], 1490.0, 1510.0)
    assert ctx.drift_m(high, prof[3], prof[3] + T) == 0.0


def test_a_tle_fitted_inside_a_steady_storm_already_carries_it():
    prof = sw.drag_profile(ISS1, ISS2)
    ep = _epoch_dt(prof)
    ctx = _flat_ctx([7.0] * 24, ep - timedelta(hours=36), lambda kp: 1.0 if kp < 7 else 2.5)
    assert ctx.drift_m(prof, prof[3], prof[3] + 86400.0) == pytest.approx(0.0, abs=1e-6)
    assert ctx.fit_ratio(prof) == pytest.approx(2.5)


def test_a_storm_that_ends_after_the_fit_also_moves_the_object():
    prof = sw.drag_profile(ISS1, ISS2)
    ep = _epoch_dt(prof)
    # fitted during the storm, then calm: the object decays slower than
    # predicted (R/R_fit - 1 = -0.6), reported as a magnitude
    ctx = _flat_ctx([7.0] * 12 + [1.0] * 12, ep - timedelta(hours=36),
                    lambda kp: 1.0 if kp < 7 else 2.5)
    T = 24 * 3600.0
    got = ctx.drift_m(prof, prof[3], prof[3] + T)
    assert got == pytest.approx(prof[0] * 0.6 * prof[2] * T * T / 2, rel=1e-9)


def test_drift_only_counts_the_storm_part_of_the_span():
    prof = sw.drag_profile(ISS1, ISS2)
    ep = _epoch_dt(prof)
    # quiet fit, quiet for 12 h after the epoch, then the storm: drift at
    # 24 h equals a 12 h steady storm
    ctx = _flat_ctx([1.0] * 16 + [7.0] * 4, ep - timedelta(hours=36),
                    lambda kp: 1.0 if kp < 7 else 2.5)
    half = 12 * 3600.0
    got = ctx.drift_m(prof, prof[3], prof[3] + 2 * half)
    assert got == pytest.approx(prof[0] * 1.5 * prof[2] * half * half / 2, rel=1e-9)


def test_drift_is_capped_where_a_public_tle_says_little():
    prof = sw.drag_profile(ISS1, ISS2)
    ep = _epoch_dt(prof)
    ctx = _flat_ctx([1.0] * 12 + [9.0] * 80, ep - timedelta(hours=36),
                    lambda kp: 1.0 if kp < 9 else 9.0)
    assert ctx.drift_m(prof, prof[3], prof[3] + 30 * 86400.0) == sw.DRIFT_CAP_M


def test_ratio_interpolates_kp_in_thirds():
    t0 = datetime(2026, 10, 10, tzinfo=timezone.utc)
    ctx = _flat_ctx([5.0], t0, lambda kp: 1.0 + kp)
    assert ctx.ratio(5.0, 550) == pytest.approx(6.0)
    assert ctx.ratio(5.5, 550) == pytest.approx(6.5)
    assert ctx.ratio(99, 550) == pytest.approx(10.0)       # clamped to Kp 9


def test_storm_table_is_quiet_at_kp1_and_climbs():
    pytest.importorskip("pymsis")
    tab = sw.storm_table(130, 130, datetime(2026, 10, 10, tzinfo=timezone.utc))
    ratio = tab["ratio"]
    i550 = sw.ALTS_KM.index(550)
    assert ratio[3][i550] == pytest.approx(1.0)          # Kp 1 is the reference
    col = [row[i550] for row in ratio]
    assert col == sorted(col) and 2.0 < col[21] < 4.0     # Kp 7: a few times denser


def test_summary_labels_and_g_scale():
    assert [sw.g_from_kp(k) for k in (3.0, 4.67, 5.0, 6.33, 6.67, 9.0)] == [0, 1, 1, 2, 3, 5]
    out = sw.summary(sw._build(FIX, {k: "ok" for k in FIX}))
    assert out["level"] == "storm" and out["label"] == "G3 strong storm" and out["kp"] == 7.0
    assert out["flare"] == "C1.5"
    assert sw.summary({"updated": None}) is None


def test_peek_never_blocks_and_starts_one_background_refresh(monkeypatch):
    import threading
    monkeypatch.setenv("KESSLER_SPACEWEATHER", "1")
    gate = threading.Event()
    calls = []

    def slow_fetch():
        calls.append(1)
        gate.wait(5)
        return FIX, {k: "ok" for k in FIX}

    monkeypatch.setattr(sw, "_fetch_all", slow_fetch)
    monkeypatch.setattr(sw, "storm_table", lambda *a, **k: None)
    sw._cache.update(t=0.0, data=None)
    sw._bg.update(running=False, fail_t=0.0)
    assert sw.peek() is None and sw.peek() is None        # cold: no wait, one fetch
    gate.set()
    for _ in range(100):
        if sw._cache["data"] is not None and not sw._bg["running"]:
            break
        threading.Event().wait(0.02)
    assert len(calls) == 1 and sw.peek()["kp"]["now"] == 7.0
    sw._cache.update(t=0.0, data=None)


def test_switched_off_means_unavailable_never_network(monkeypatch):
    monkeypatch.setenv("KESSLER_SPACEWEATHER", "0")
    monkeypatch.setattr(sw, "_fetch_all", lambda: pytest.fail("must not fetch"))
    sw._cache.update(t=0.0, data=None)
    assert sw.peek() is None
    assert sw.conditions()["stale"] is True and sw.storm_context() is None


def test_conjunctions_carry_the_storm_layer(monkeypatch, tmp_path):
    """attach_space_weather: widened when the storm drift beats the sigma."""
    from backend.app import config, db
    from backend.app.api import conjunctions as conj_api
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "sw.db")
    conn = db.get_conn()
    db.init_db(conn)
    conn.execute("INSERT INTO satellites (id, norad_id, name, object_type, last_updated) VALUES"
                 " (1, 25544, 'ISS', 'PAYLOAD', 't'), (2, 99999, 'DEB', 'DEBRIS', 't')")
    for sid, epoch in ((1, "2026-10-10T12:00:00Z"), (2, "2026-10-10T12:00:00Z")):
        conn.execute("INSERT INTO tle_history (satellite_id, epoch, line1, line2, ingested_at)"
                     " VALUES (?, ?, ?, ?, ?)", (sid, epoch, ISS1, ISS2, epoch))
    conn.commit()
    t0 = datetime(2026, 10, 9, tzinfo=timezone.utc)        # 36 h before the TLE epoch
    storm = _flat_ctx([1.0] * 12 + [8.0] * 24, t0, lambda kp: 1.0 if kp < 7 else 3.0)
    item = {"id": 7, "tca": "2026-10-12T12:00:00Z",
            "primary": {"id": 1}, "secondary": {"id": 2}}
    conj_api.attach_space_weather(conn, [item], storm)
    sw_ = item["space_weather"]
    assert sw_["drift_m"] > sw_["sigma_m"] > 0 and sw_["widened"] is True
    assert sw_["sigma_storm_m"] == pytest.approx(math.hypot(sw_["sigma_m"], sw_["drift_m"]), abs=1)
    quiet = _flat_ctx([1.0] * 36, t0, lambda kp: 1.0)
    item2 = {"id": 8, "tca": "2026-10-12T12:00:00Z", "primary": {"id": 1}, "secondary": {"id": 2}}
    conj_api.attach_space_weather(conn, [item2], quiet)
    assert item2["space_weather"]["drift_m"] == 0 and item2["space_weather"]["widened"] is False
    no_data = {"id": 9, "tca": "2026-10-12T12:00:00Z", "primary": {"id": 1}, "secondary": {"id": 2}}
    conj_api.attach_space_weather(conn, [no_data])        # no live data: key stays absent
    assert "space_weather" not in no_data
    conn.close()


def test_storm_flags_mark_what_an_operator_should_recheck(monkeypatch, tmp_path):
    """'watch' = already elevated and widened; 'may_escalate' = nominal but
    the widened sigma allows an elevated Pc; a far pair stays unflagged."""
    from backend.app import config, db
    from backend.app.api import conjunctions as conj_api
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "flags.db")
    conn = db.get_conn()
    db.init_db(conn)
    conn.execute("INSERT INTO satellites (id, norad_id, name, object_type, last_updated) VALUES"
                 " (1, 25544, 'ISS', 'PAYLOAD', 't'), (2, 99999, 'DEB', 'DEBRIS', 't')")
    for sid in (1, 2):
        conn.execute("INSERT INTO tle_history (satellite_id, epoch, line1, line2, ingested_at)"
                     " VALUES (?, '2026-10-10T12:00:00Z', ?, ?, 't')", (sid, ISS1, ISS2))
    conn.commit()
    monkeypatch.setattr(sw, "pair_drift_m", lambda *a, **k: 4000.0)
    ctx = _flat_ctx([7.0], datetime(2026, 10, 10, tzinfo=timezone.utc), lambda kp: 2.0)
    base = {"tca": "2026-10-12T12:00:00Z", "primary": {"id": 1}, "secondary": {"id": 2}}
    items = [dict(base, id=1, miss_distance_m=5000.0, risk_level="nominal"),
             dict(base, id=2, miss_distance_m=200.0, risk_level="elevated"),
             dict(base, id=3, miss_distance_m=14000.0, risk_level="nominal")]
    conj_api.attach_space_weather(conn, items, ctx)
    flags = [i["space_weather"]["flag"] for i in items]
    assert flags == ["may_escalate", "watch", None]
    assert items[0]["space_weather"]["pc_storm"] >= config.PC_ELEVATED
    assert [i["id"] for i in conj_api.storm_flagged(items)] == [2, 1]
    conn.close()


def test_a_flaky_feed_keeps_its_last_good_copy_and_kp_falls_back(monkeypatch):
    monkeypatch.setenv("KESSLER_SPACEWEATHER", "1")
    monkeypatch.setattr(sw, "storm_table", lambda *a, **k: None)
    sw._cache.update(t=0.0, data=None)
    sw._raw_last.clear()
    monkeypatch.setattr(sw, "_fetch_all", lambda: (dict(FIX), {k: "ok" for k in FIX}))
    assert sw.conditions(force=True)["kp"]["now"] == 7.0
    broken = {k: v for k, v in FIX.items() if k != "kp"}
    monkeypatch.setattr(sw, "_fetch_all", lambda: (dict(broken),
                        {**{k: "ok" for k in broken}, "kp": "unavailable (JSONDecodeError)"}))
    d = sw.conditions(force=True)
    assert d["kp"]["now"] == 7.0 and "kept the copy" in d["feeds"]["kp"]
    # no copy at all: Kp comes from the forecast product's observed rows
    sw._raw_last.clear()
    fc = [{"time_tag": "2026-10-10T15:00:00", "kp": 6.67, "observed": "observed"},
          {"time_tag": "2099-01-01T00:00:00", "kp": 9.0, "observed": "predicted"}]
    out = sw._build({**broken, "kp_forecast": fc}, {k: "ok" for k in broken})
    assert out["kp"]["now"] == 6.67
    # and with no Kp anywhere the summary says nothing rather than "quiet"
    out = sw._build({k: v for k, v in broken.items() if k != "kp_forecast"}, {})
    assert out.get("kp") is None and sw.summary(out) is None
    sw._cache.update(t=0.0, data=None)
    sw._raw_last.clear()
