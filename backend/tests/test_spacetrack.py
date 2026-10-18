"""Offline tests for Space-Track cdm_public ingestion.

No network: HTTP behavior is exercised through httpx.MockTransport clients and
fixture JSON rows; DBs are temp files under tmp_path. The velocity policy under
test is documented in backend/app/ingestion/spacetrack.py: cdm_public has no
relative-speed column, so a message is stored only when its velocity is
message-supplied or reusable from the matched conjunction's latest CDM -
otherwise the row is skipped and counted (skipped_no_velocity).
"""
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from backend.app import db
from backend.app.ingestion import celestrak, pipeline, spacetrack

NOW = datetime.now(timezone.utc).replace(microsecond=0)
NOW_ISO = NOW.strftime(celestrak.ISO_FMT)
TCA = NOW + timedelta(hours=10)


def _st_date(dt: datetime, sep: str = " ", frac: bool = False) -> str:
    s = dt.strftime(f"%Y-%m-%d{sep}%H:%M:%S")
    return s + ".000000" if frac else s


def raw_row(**overrides) -> dict:
    """A realistic cdm_public JSON row (all values strings, like the API)."""
    row = {
        "CDM_ID": "418254001",
        "CREATED": _st_date(NOW - timedelta(hours=2), sep=" ", frac=True),
        "EMERGENCY_REPORTABLE": "N",
        "TCA": _st_date(TCA, sep="T", frac=True),
        "MIN_RNG": "812",
        "PC": "0.0000310000",
        "SAT_1_ID": "25544",
        "SAT_1_NAME": "ISS (ZARYA)",
        "SAT_2_ID": "34454",
        "SAT_2_NAME": "COSMOS 2251 DEB",
    }
    row.update(overrides)
    return row


def _make_conn(tmp_path):
    conn = db.get_conn(tmp_path / "test.db")
    db.init_db(conn)
    return conn


def _seed_satellite(conn, norad: int, name: str) -> int:
    cur = conn.execute(
        "INSERT INTO satellites (norad_id, name, object_type, last_updated)"
        " VALUES (?, ?, 'PAYLOAD', ?)", (norad, name, NOW_ISO))
    return cur.lastrowid


def _seed_conjunction(conn, p_id: int, s_id: int, tca: datetime,
                      velocity: float | None = 14210.3) -> int:
    cur = conn.execute(
        "INSERT INTO conjunctions (primary_object_id, secondary_object_id,"
        " time_of_closest_approach, first_seen_at, status)"
        " VALUES (?, ?, ?, ?, 'active')",
        (p_id, s_id, tca.strftime(celestrak.ISO_FMT), NOW_ISO))
    conj_id = cur.lastrowid
    if velocity is not None:
        conn.execute(
            "INSERT INTO cdm_messages (conjunction_id, message_time,"
            " miss_distance_m, relative_velocity_ms, probability_of_collision,"
            " source) VALUES (?, ?, 900.0, ?, 2.5e-05, 'screening')",
            (conj_id, NOW_ISO, velocity))
    conn.commit()
    return conj_id


# --- normalize_row -----------------------------------------------------------

def test_normalize_row_full_mapping():
    rec = spacetrack.normalize_row(raw_row(EMERGENCY_REPORTABLE="Y"))
    assert rec is not None
    assert rec["cdm_id"] == 418254001
    # CREATED with space separator + fractional seconds -> whole-second ISO Z.
    assert rec["message_time"] == (NOW - timedelta(hours=2)).strftime(celestrak.ISO_FMT)
    # TCA with 'T' separator + fractional seconds.
    assert rec["tca"] == TCA.strftime(celestrak.ISO_FMT)
    assert rec["miss_distance_m"] == pytest.approx(812.0)  # MIN_RNG already m
    assert rec["relative_speed_ms"] is None  # cdm_public has no such column
    assert rec["probability_of_collision"] == pytest.approx(3.1e-05)
    assert rec["sat1_norad"] == 25544
    assert rec["sat1_name"] == "ISS (ZARYA)"
    assert rec["sat2_norad"] == 34454
    assert rec["sat2_name"] == "COSMOS 2251 DEB"
    assert rec["emergency_reportable"] is True


def test_normalize_row_tolerates_missing_pc():
    for pc_value in (None, "", "   ", "not-a-number"):
        rec = spacetrack.normalize_row(raw_row(PC=pc_value))
        assert rec is not None
        assert rec["probability_of_collision"] is None
    # out-of-range PC is demoted to None, row kept
    rec = spacetrack.normalize_row(raw_row(PC="1.5"))
    assert rec is not None and rec["probability_of_collision"] is None


def test_normalize_row_drops_malformed():
    assert spacetrack.normalize_row(raw_row(MIN_RNG=None)) is None
    assert spacetrack.normalize_row(raw_row(MIN_RNG="")) is None
    assert spacetrack.normalize_row(raw_row(MIN_RNG="-1.0")) is None
    assert spacetrack.normalize_row(raw_row(TCA="not a date")) is None
    assert spacetrack.normalize_row(raw_row(CREATED=None)) is None
    assert spacetrack.normalize_row(raw_row(SAT_2_ID="")) is None
    assert spacetrack.normalize_row({}) is None


# --- login / fetch (MockTransport, no network) -------------------------------

def test_login_posts_env_credentials(monkeypatch):
    monkeypatch.setenv("SPACETRACK_IDENTITY", "ops_user")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "hunter2-secret")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, text='""')

    with httpx.Client(transport=httpx.MockTransport(handler),
                      follow_redirects=True) as client:
        spacetrack.login(client)
    assert len(seen) == 1
    req = seen[0]
    assert req.method == "POST"
    assert req.url == spacetrack.LOGIN_URL
    body = req.read().decode()
    assert "identity=ops_user" in body and "password=hunter2-secret" in body


def test_login_failure_raises_without_leaking_password(monkeypatch):
    monkeypatch.setenv("SPACETRACK_IDENTITY", "ops_user")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "hunter2-secret")

    def handler(request):
        return httpx.Response(200, json={"Login": "Failed"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError) as err:
            spacetrack.login(client)
    assert "hunter2-secret" not in str(err.value)
    assert "ops_user" not in str(err.value)


def test_login_requires_env_credentials(monkeypatch):
    monkeypatch.delenv("SPACETRACK_IDENTITY", raising=False)
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)
    with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200))) as client:
        with pytest.raises(RuntimeError, match="SPACETRACK_IDENTITY"):
            spacetrack.login(client)


def test_fetch_cdms_single_query_etiquette_and_normalization():
    rows = [raw_row(),
            raw_row(CDM_ID="418254002", MIN_RNG="1400", SAT_2_ID="99999",
                    SAT_2_NAME="FENGYUN 1C DEB"),
            raw_row(CDM_ID="bad", TCA="garbage")]  # malformed -> dropped
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=rows)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        cdms = spacetrack.fetch_cdms(client=client)

    # One query, correct class/window/order/limit per Space-Track etiquette.
    assert len(seen) == 1
    path = seen[0].url.path
    assert "/basicspacedata/query/class/cdm_public" in path
    assert "/TCA/>now" in path
    assert "orderby/CREATED desc" in path
    assert f"limit/{spacetrack.QUERY_LIMIT}" in path
    # two valid rows normalized, malformed row dropped
    assert len(cdms) == 2
    assert {c["cdm_id"] for c in cdms} == {418254001, 418254002}
    assert cdms[1]["miss_distance_m"] == pytest.approx(1400.0)


def test_fetch_cdms_custom_limit():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=[])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert spacetrack.fetch_cdms(client=client, limit=25) == []
    assert "limit/25" in seen[0].url.path


# --- ingest_cdms -------------------------------------------------------------

def test_ingest_appends_to_matched_conjunction_reusing_velocity(tmp_path):
    conn = _make_conn(tmp_path)
    iss = _seed_satellite(conn, 25544, "ISS (ZARYA)")
    deb = _seed_satellite(conn, 34454, "COSMOS 2251 DEB")
    conj_id = _seed_conjunction(conn, iss, deb, TCA, velocity=14210.3)

    # TCA 10 min off the stored one: inside the 30-min same-event tolerance.
    cdm = spacetrack.normalize_row(
        raw_row(TCA=_st_date(TCA + timedelta(minutes=10), sep="T")))
    stats = spacetrack.ingest_cdms(conn, [cdm])
    assert stats["cdms_added"] == 1
    assert stats["conjunctions_matched"] == 1
    assert stats["conjunctions_new"] == 0
    assert stats["satellites_created"] == 0
    assert stats["skipped_no_velocity"] == 0

    row = conn.execute(
        "SELECT * FROM cdm_messages WHERE source = 'spacetrack'").fetchone()
    assert row["conjunction_id"] == conj_id
    assert row["miss_distance_m"] == pytest.approx(812.0)
    assert row["relative_velocity_ms"] == pytest.approx(14210.3)  # reused
    assert row["probability_of_collision"] == pytest.approx(3.1e-05)
    assert row["message_time"] == cdm["message_time"]
    # the screener's refined TCA is authoritative - not rewritten
    tca_db = conn.execute(
        "SELECT time_of_closest_approach FROM conjunctions WHERE id = ?",
        (conj_id,)).fetchone()[0]
    assert tca_db == TCA.strftime(celestrak.ISO_FMT)
    conn.close()


def test_ingest_matches_reversed_pair_order(tmp_path):
    conn = _make_conn(tmp_path)
    iss = _seed_satellite(conn, 25544, "ISS (ZARYA)")
    deb = _seed_satellite(conn, 34454, "COSMOS 2251 DEB")
    conj_id = _seed_conjunction(conn, iss, deb, TCA)

    # Space-Track lists the pair the other way round.
    cdm = spacetrack.normalize_row(raw_row(
        SAT_1_ID="34454", SAT_1_NAME="COSMOS 2251 DEB",
        SAT_2_ID="25544", SAT_2_NAME="ISS (ZARYA)"))
    stats = spacetrack.ingest_cdms(conn, [cdm])
    assert stats["cdms_added"] == 1 and stats["conjunctions_matched"] == 1
    row = conn.execute(
        "SELECT conjunction_id FROM cdm_messages WHERE source = 'spacetrack'"
    ).fetchone()
    assert row["conjunction_id"] == conj_id
    conn.close()


def test_ingest_dedupes_on_conjunction_and_message_time(tmp_path):
    conn = _make_conn(tmp_path)
    iss = _seed_satellite(conn, 25544, "ISS (ZARYA)")
    deb = _seed_satellite(conn, 34454, "COSMOS 2251 DEB")
    _seed_conjunction(conn, iss, deb, TCA)

    cdm = spacetrack.normalize_row(raw_row())
    assert spacetrack.ingest_cdms(conn, [cdm])["cdms_added"] == 1
    # Refetching the same message later must not duplicate it.
    stats = spacetrack.ingest_cdms(conn, [cdm])
    assert stats["cdms_added"] == 0
    assert stats["skipped_duplicate"] == 1
    n = conn.execute("SELECT COUNT(*) AS n FROM cdm_messages"
                     " WHERE source = 'spacetrack'").fetchone()["n"]
    assert n == 1
    conn.close()


def test_ingest_dedupes_republished_cdm_minutes_apart(tmp_path):
    """Space-Track republishes each CDM from the other object's perspective a
    few minutes later with identical miss/Pc - it must not store twice. A
    genuinely NEW update in the same window (different Pc) must store."""
    conn = _make_conn(tmp_path)
    iss = _seed_satellite(conn, 25544, "ISS (ZARYA)")
    deb = _seed_satellite(conn, 34454, "COSMOS 2251 DEB")
    _seed_conjunction(conn, iss, deb, TCA)

    first = spacetrack.normalize_row(raw_row())
    assert spacetrack.ingest_cdms(conn, [first])["cdms_added"] == 1

    republished = spacetrack.normalize_row(raw_row(
        CREATED=_st_date(NOW - timedelta(hours=2) + timedelta(minutes=5),
                         sep=" ", frac=True),
        SAT_1_ID="34454", SAT_1_NAME="COSMOS 2251 DEB",
        SAT_2_ID="25544", SAT_2_NAME="ISS (ZARYA)"))
    stats = spacetrack.ingest_cdms(conn, [republished])
    assert stats["cdms_added"] == 0
    assert stats["skipped_duplicate"] == 1

    real_update = spacetrack.normalize_row(raw_row(
        CREATED=_st_date(NOW - timedelta(hours=2) + timedelta(minutes=10),
                         sep=" ", frac=True),
        PC="0.0000450000"))
    assert spacetrack.ingest_cdms(conn, [real_update])["cdms_added"] == 1
    n = conn.execute("SELECT COUNT(*) AS n FROM cdm_messages"
                     " WHERE source = 'spacetrack'").fetchone()["n"]
    assert n == 2
    conn.close()


def test_operator_cdm_takes_precedence_over_newer_screening_row(tmp_path):
    """Screening rows are stamped 'now', Space-Track rows with CREATED (always
    earlier). The board and the model must still use the operator's numbers
    rather than letting our assumed-sigma estimate mask them."""
    from backend.app.api import conjunctions as conjunctions_api
    from backend.app.models import risk_model

    conn = _make_conn(tmp_path)
    iss = _seed_satellite(conn, 25544, "ISS (ZARYA)")
    deb = _seed_satellite(conn, 34454, "COSMOS 2251 DEB")
    conj_id = _seed_conjunction(conn, iss, deb, TCA)   # screening: 900 m, 2.5e-5

    assert spacetrack.ingest_cdms(
        conn, [spacetrack.normalize_row(raw_row())])["cdms_added"] == 1

    item = conjunctions_api.query_summaries(conn, "c.id = ?", (conj_id,))[0]
    assert item["miss_distance_m"] == 812.0
    assert item["pc_latest"] == 3.1e-05
    scored = risk_model.scored_cdms(conn, conj_id)
    assert [c["miss_distance_m"] for c in scored] == [812.0]
    conn.close()


def test_ingest_skips_unknown_pair_without_velocity(tmp_path):
    """cdm_public supplies no velocity; a pair with no matching conjunction has
    none to reuse -> the row is skipped and counted, nothing is written."""
    conn = _make_conn(tmp_path)
    stats = spacetrack.ingest_cdms(conn, [spacetrack.normalize_row(raw_row())])
    assert stats == {"fetched": 1, "cdms_added": 0, "conjunctions_new": 0,
                     "conjunctions_matched": 0, "satellites_created": 0,
                     "tles_fetched": 0, "skipped_no_velocity": 1, "skipped_duplicate": 0}
    for table in ("satellites", "conjunctions", "cdm_messages"):
        n = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        assert n == 0
    conn.close()


def test_ingest_skips_matched_conjunction_with_no_cdms(tmp_path):
    conn = _make_conn(tmp_path)
    iss = _seed_satellite(conn, 25544, "ISS (ZARYA)")
    deb = _seed_satellite(conn, 34454, "COSMOS 2251 DEB")
    _seed_conjunction(conn, iss, deb, TCA, velocity=None)  # no CDM history
    stats = spacetrack.ingest_cdms(conn, [spacetrack.normalize_row(raw_row())])
    assert stats["cdms_added"] == 0 and stats["skipped_no_velocity"] == 1
    conn.close()


def test_ingest_message_supplied_velocity_creates_minimal_rows(tmp_path):
    """A normalized CDM that carries its own relative speed (a fuller CDM class
    than cdm_public) can mint unknown satellites and a new conjunction."""
    conn = _make_conn(tmp_path)
    cdm = spacetrack.normalize_row(raw_row(SAT_1_ID="90001", SAT_1_NAME=None,
                                           SAT_2_ID="90002", SAT_2_NAME=None))
    cdm["relative_speed_ms"] = 12345.6
    stats = spacetrack.ingest_cdms(conn, [cdm])
    assert stats["cdms_added"] == 1
    assert stats["conjunctions_new"] == 1
    assert stats["satellites_created"] == 2
    assert stats["skipped_no_velocity"] == 0

    sats = conn.execute(
        "SELECT norad_id, name, object_type FROM satellites ORDER BY norad_id"
    ).fetchall()
    assert [(s["norad_id"], s["name"], s["object_type"]) for s in sats] == [
        (90001, "OBJECT 90001", "UNKNOWN"), (90002, "OBJECT 90002", "UNKNOWN")]
    conj = conn.execute("SELECT * FROM conjunctions").fetchone()
    assert conj["time_of_closest_approach"] == cdm["tca"]
    assert conj["status"] == "active"
    row = conn.execute("SELECT * FROM cdm_messages").fetchone()
    assert row["source"] == "spacetrack"
    assert row["relative_velocity_ms"] == pytest.approx(12345.6)

    # A second message for the same event now matches and reuses the velocity.
    later = spacetrack.normalize_row(raw_row(
        SAT_1_ID="90001", SAT_2_ID="90002",
        CREATED=_st_date(NOW - timedelta(hours=1)),
        TCA=_st_date(TCA + timedelta(minutes=5), sep="T")))
    stats2 = spacetrack.ingest_cdms(conn, [later])
    assert stats2["cdms_added"] == 1 and stats2["conjunctions_matched"] == 1
    rows = conn.execute("SELECT relative_velocity_ms FROM cdm_messages"
                        " ORDER BY message_time").fetchall()
    assert [r["relative_velocity_ms"] for r in rows] == \
        pytest.approx([12345.6, 12345.6])
    conn.close()


def test_ensure_satellite_creates_minimal_row_once(tmp_path):
    conn = _make_conn(tmp_path)
    sat_id = spacetrack.ensure_satellite(conn, 87654)
    assert spacetrack.ensure_satellite(conn, 87654) == sat_id  # idempotent
    row = conn.execute("SELECT * FROM satellites WHERE norad_id = 87654").fetchone()
    assert row["name"] == "OBJECT 87654"
    assert row["object_type"] == "UNKNOWN"
    conn.close()


# --- pipeline.refresh hook ---------------------------------------------------

@pytest.fixture()
def offline_celestrak(monkeypatch):
    """refresh() must not hit the network: every group fetch returns nothing."""
    monkeypatch.setattr(celestrak, "fetch_group",
                        lambda group, client=None: [])


def test_pipeline_skips_spacetrack_without_creds(tmp_path, monkeypatch,
                                                 offline_celestrak):
    monkeypatch.delenv("SPACETRACK_IDENTITY", raising=False)
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)

    def bomb(*args, **kwargs):
        raise AssertionError("fetch_cdms must not be called without creds")
    monkeypatch.setattr(spacetrack, "fetch_cdms", bomb)

    conn = _make_conn(tmp_path)
    stats = pipeline.refresh(conn)
    assert stats["spacetrack_cdms"] == 0
    conn.close()


def test_pipeline_ingests_spacetrack_with_creds(tmp_path, monkeypatch,
                                                offline_celestrak):
    monkeypatch.setenv("SPACETRACK_IDENTITY", "ops_user")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "hunter2-secret")

    conn = _make_conn(tmp_path)
    iss = _seed_satellite(conn, 25544, "ISS (ZARYA)")
    deb = _seed_satellite(conn, 34454, "COSMOS 2251 DEB")
    _seed_conjunction(conn, iss, deb, TCA, velocity=14210.3)

    fixture_cdms = [spacetrack.normalize_row(raw_row())]
    # refresh_cdms owns login + fetch; stand in for both, keep real ingest.
    monkeypatch.setattr(
        spacetrack, "refresh_cdms",
        lambda c: spacetrack.ingest_cdms(c, fixture_cdms))

    stats = pipeline.refresh(conn)
    assert stats["spacetrack_cdms"] == 1
    n = conn.execute("SELECT COUNT(*) AS n FROM cdm_messages"
                     " WHERE source = 'spacetrack'").fetchone()["n"]
    assert n == 1
    conn.close()


def test_pipeline_skips_spacetrack_cdms_when_switched_off(tmp_path, monkeypatch,
                                                          offline_celestrak):
    monkeypatch.setenv("SPACETRACK_IDENTITY", "ops_user")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "hunter2-secret")
    monkeypatch.setenv("KESSLER_SPACETRACK_CDMS", "0")
    def must_not_run(*args, **kwargs):
        raise AssertionError("CDMs were pulled with KESSLER_SPACETRACK_CDMS=0")
    monkeypatch.setattr(spacetrack, "refresh_cdms", must_not_run)

    conn = _make_conn(tmp_path)
    stats = pipeline.refresh(conn)
    assert stats["spacetrack_cdms"] == 0
    conn.close()


def test_pipeline_spacetrack_failure_is_nonfatal(tmp_path, monkeypatch, capsys,
                                                 offline_celestrak):
    monkeypatch.setenv("SPACETRACK_IDENTITY", "ops_user")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "hunter2-secret")

    def boom(*args, **kwargs):
        raise RuntimeError("Space-Track login failed (credentials rejected)")
    monkeypatch.setattr(spacetrack, "fetch_cdms", boom)

    conn = _make_conn(tmp_path)
    stats = pipeline.refresh(conn)  # must not raise
    assert stats["spacetrack_cdms"] == 0
    err = capsys.readouterr().err
    assert "Space-Track CDM ingestion failed" in err
    assert "hunter2-secret" not in err
    conn.close()


# --- fetch_full_catalog (gp class, EPOCH > now-30) ---------------------------

def gp_row(norad: int, name: str = "ISS (ZARYA)", otype: str = "PAYLOAD",
           **overrides) -> dict:
    """A realistic Space-Track gp JSON row (values are strings, like the API).

    The TLE lines are stored, never propagated, by upsert_catalog - plausible
    fixed strings are sufficient.
    """
    row = {
        "NORAD_CAT_ID": str(norad),
        "OBJECT_NAME": name,
        "OBJECT_ID": "1998-067A",
        "OBJECT_TYPE": otype,
        "EPOCH": _st_date(NOW - timedelta(days=2), sep="T", frac=True),
        "TLE_LINE1": f"1 {norad:05d}U 98067A   26200.50000000  .00016717"
                     "  00000-0  10270-3 0  9000",
        "TLE_LINE2": f"2 {norad:05d}  51.6400 208.9163 0006317  69.9862"
                     " 254.3157 15.49560000000000",
    }
    row.update(overrides)
    return row


def test_fetch_full_catalog_single_query_and_upsert_shape(tmp_path):
    rows = [gp_row(25544),
            gp_row(48274, name="ODD OBJECT", otype="TBA"),  # -> UNKNOWN
            gp_row(11111, TLE_LINE2=None)]                  # malformed -> drop
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=rows)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        records = spacetrack.fetch_full_catalog(client=client)

    # one query against gp with the 30-day epoch window + the shared predicates.
    assert len(seen) == 1
    path = seen[0].url.path
    assert "/basicspacedata/query/class/gp" in path
    assert "/EPOCH/>now-30" in path
    assert f"/predicates/{spacetrack.GP_PREDICATES}" in path
    assert "/format/json" in path

    # celestrak-record shape, keyed by norad, malformed row dropped.
    assert set(records) == {25544, 48274}
    rec = records[25544]
    assert set(rec) == {"name", "object_type", "intl_designator", "epoch",
                        "line1", "line2"}
    assert rec["name"] == "ISS (ZARYA)"
    assert rec["object_type"] == "PAYLOAD"
    assert rec["intl_designator"] == "1998-067A"
    assert rec["epoch"] == (NOW - timedelta(days=2)).strftime(celestrak.ISO_FMT)
    assert rec["line1"].startswith("1 25544U")
    assert records[48274]["object_type"] == "UNKNOWN"  # unrecognized type

    # The records feed celestrak.upsert_catalog unchanged.
    conn = _make_conn(tmp_path)
    up = celestrak.upsert_catalog(conn, records)
    assert up == {"fetched": 2, "satellites_new": 2, "tles_added": 2}
    conn.close()


def test_fetch_full_catalog_rejects_unexpected_shape():
    with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"error": "nope"}))) as client:
        with pytest.raises(RuntimeError, match="unexpected gp shape"):
            spacetrack.fetch_full_catalog(client=client)


# --- covariance enrichment (restricted cdm class) ----------------------------
# Free accounts lack the SSA agreement: the live class is inaccessible (an
# authenticated probe on 2026-07-23 got HTTP 400 "Your Class Does Not Exist").
# Everything here is offline: MockTransport + fixture rows shaped like the
# documented cdm-class columns.

def cov_row(**overrides) -> dict:
    row = {
        "CDM_ID": "418254001",
        "CREATION_DATE": _st_date(NOW - timedelta(hours=2), sep=" ", frac=True),
        "TCA": _st_date(TCA, sep="T", frac=True),
        "SAT1_OBJECT_DESIGNATOR": "25544",
        "SAT2_OBJECT_DESIGNATOR": "34454",
        "SAT1_CR_R": "2500.0", "SAT1_CT_R": "120.0", "SAT1_CT_T": "640000.0",
        "SAT1_CN_R": "-35.0", "SAT1_CN_T": "80.0", "SAT1_CN_N": "10000.0",
        "SAT2_CR_R": "900.0", "SAT2_CT_R": "-60.0", "SAT2_CT_T": "250000.0",
        "SAT2_CN_R": "12.0", "SAT2_CN_T": "-40.0", "SAT2_CN_N": "4900.0",
    }
    row.update(overrides)
    return row


SAT1_RTN = [[2500.0, 120.0, -35.0],
            [120.0, 640000.0, 80.0],
            [-35.0, 80.0, 10000.0]]
SAT2_RTN = [[900.0, -60.0, 12.0],
            [-60.0, 250000.0, -40.0],
            [12.0, -40.0, 4900.0]]


@pytest.fixture()
def covariance_access(monkeypatch):
    """Isolate the module-level denial flag (auto-restored)."""
    monkeypatch.setattr(spacetrack, "_COVARIANCE_ACCESS_DENIED", False)


def test_normalize_covariance_row_builds_symmetric_rtn(covariance_access):
    rec = spacetrack.normalize_covariance_row(cov_row())
    assert rec is not None
    assert rec["cdm_id"] == 418254001
    assert rec["message_time"] == (NOW - timedelta(hours=2)).strftime(celestrak.ISO_FMT)
    assert rec["tca"] == TCA.strftime(celestrak.ISO_FMT)
    assert rec["sat1_norad"] == 25544 and rec["sat2_norad"] == 34454
    assert rec["sat1_rtn"] == SAT1_RTN
    assert rec["sat2_rtn"] == SAT2_RTN
    for matrix in (rec["sat1_rtn"], rec["sat2_rtn"]):  # lower-triangle mirrored
        for a in range(3):
            for b in range(3):
                assert matrix[a][b] == matrix[b][a]


def test_normalize_covariance_row_drops_incomplete():
    # A partial covariance is useless: any missing component drops the row.
    assert spacetrack.normalize_covariance_row(cov_row(SAT1_CN_N=None)) is None
    assert spacetrack.normalize_covariance_row(cov_row(SAT2_CR_R="")) is None
    assert spacetrack.normalize_covariance_row(cov_row(SAT2_CT_T="n/a")) is None
    assert spacetrack.normalize_covariance_row(cov_row(TCA="garbage")) is None
    assert spacetrack.normalize_covariance_row(cov_row(CREATION_DATE=None)) is None
    assert spacetrack.normalize_covariance_row(
        cov_row(SAT1_OBJECT_DESIGNATOR="")) is None
    assert spacetrack.normalize_covariance_row({}) is None
    # CDM_ID stays optional.
    rec = spacetrack.normalize_covariance_row(cov_row(CDM_ID=None))
    assert rec is not None and rec["cdm_id"] is None


def test_fetch_covariances_queries_restricted_class_once(covariance_access):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=[cov_row(), {"CDM_ID": "junk"}])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        rows = spacetrack.fetch_covariances(client)
    assert len(seen) == 1
    path = seen[0].url.path
    assert "/basicspacedata/query/class/cdm/" in path
    assert "/TCA/>now" in path
    assert f"/predicates/{spacetrack.CDM_COV_PREDICATES}" in path
    assert len(rows) == 1  # malformed row dropped
    assert rows[0]["sat1_rtn"] == SAT1_RTN


@pytest.mark.parametrize("denial", [
    httpx.Response(403, text="Forbidden"),
    httpx.Response(401, text="Unauthorized"),
    # Observed live (free account, 2026-07-23): the restricted class is
    # masked as nonexistent rather than forbidden.
    httpx.Response(400, json={"error": "Your Class Does Not Exist "}),
])
def test_fetch_covariances_denied_flags_and_never_retries(
        covariance_access, capsys, denial):
    seen = []

    def handler(request):
        seen.append(request)
        return denial

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert spacetrack.fetch_covariances(client) is None       # denied
        assert spacetrack._COVARIANCE_ACCESS_DENIED is True
        assert spacetrack.fetch_covariances(client) is None       # flag path
    assert len(seen) == 1  # never retried within the process
    err = capsys.readouterr().err
    assert err.count("SSA") == 1  # exactly one stderr line
    assert "covariance enrichment disabled" in err


def test_fetch_covariances_transient_error_raises_without_flag(covariance_access):
    with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(503, text="maintenance"))) as client:
        with pytest.raises(httpx.HTTPStatusError):
            spacetrack.fetch_covariances(client)
    assert spacetrack._COVARIANCE_ACCESS_DENIED is False  # transient, no flag


def test_apply_covariances_updates_matched_cdm(tmp_path):
    import json as jsonlib

    conn = _make_conn(tmp_path)
    iss = _seed_satellite(conn, 25544, "ISS (ZARYA)")
    deb = _seed_satellite(conn, 34454, "COSMOS 2251 DEB")
    conj_id = _seed_conjunction(conn, iss, deb, TCA, velocity=14210.3)
    cdm = spacetrack.normalize_row(raw_row())
    assert spacetrack.ingest_cdms(conn, [cdm])["cdms_added"] == 1

    cov = spacetrack.normalize_covariance_row(cov_row())
    assert spacetrack.apply_covariances(conn, [cov], cdms=[cdm]) == 1

    row = conn.execute(
        "SELECT covariance_json FROM cdm_messages WHERE conjunction_id = ?"
        " AND source = 'spacetrack'", (conj_id,)).fetchone()
    stored = jsonlib.loads(row["covariance_json"])
    assert stored == {"sat1_rtn": SAT1_RTN, "sat2_rtn": SAT2_RTN,
                      "source": "spacetrack-cdm"}
    # The screening message is never touched.
    screening_row = conn.execute(
        "SELECT covariance_json FROM cdm_messages WHERE source = 'screening'"
    ).fetchone()
    assert screening_row["covariance_json"] is None
    conn.close()


def test_apply_covariances_matches_by_cdm_id(tmp_path):
    """A covariance row whose CDM_ID appears in the ingested batch adopts that
    message's pair/TCA/message_time - here the row's own TCA drifted beyond
    the 30-min tolerance, so only the CDM_ID path can match it."""
    import json as jsonlib

    conn = _make_conn(tmp_path)
    iss = _seed_satellite(conn, 25544, "ISS (ZARYA)")
    deb = _seed_satellite(conn, 34454, "COSMOS 2251 DEB")
    _seed_conjunction(conn, iss, deb, TCA, velocity=14210.3)
    cdm = spacetrack.normalize_row(raw_row())
    spacetrack.ingest_cdms(conn, [cdm])

    cov = spacetrack.normalize_covariance_row(cov_row(
        TCA=_st_date(TCA + timedelta(minutes=45), sep="T"),
        CREATION_DATE=_st_date(NOW - timedelta(hours=2), frac=True)))
    assert spacetrack.apply_covariances(conn, [cov], cdms=[cdm]) == 1
    row = conn.execute("SELECT covariance_json FROM cdm_messages"
                       " WHERE source = 'spacetrack'").fetchone()
    assert jsonlib.loads(row["covariance_json"])["source"] == "spacetrack-cdm"
    conn.close()


def test_apply_covariances_pair_tca_fallback_targets_latest(tmp_path):
    """Without a CDM_ID match (cdms=None) the pair+TCA path matches the
    conjunction; a message_time absent from the stored messages falls back to
    the latest source='spacetrack' message."""
    import json as jsonlib

    conn = _make_conn(tmp_path)
    iss = _seed_satellite(conn, 25544, "ISS (ZARYA)")
    deb = _seed_satellite(conn, 34454, "COSMOS 2251 DEB")
    _seed_conjunction(conn, iss, deb, TCA, velocity=14210.3)
    older = spacetrack.normalize_row(raw_row(
        CREATED=_st_date(NOW - timedelta(hours=6))))
    newer = spacetrack.normalize_row(raw_row())
    spacetrack.ingest_cdms(conn, [older, newer])

    cov = spacetrack.normalize_covariance_row(cov_row(
        CDM_ID=None, CREATION_DATE=_st_date(NOW - timedelta(minutes=30))))
    assert spacetrack.apply_covariances(conn, [cov]) == 1
    rows = conn.execute(
        "SELECT message_time, covariance_json FROM cdm_messages"
        " WHERE source = 'spacetrack' ORDER BY message_time").fetchall()
    assert rows[0]["covariance_json"] is None          # older untouched
    assert rows[1]["covariance_json"] is not None      # latest got it
    assert jsonlib.loads(rows[1]["covariance_json"])["sat1_rtn"] == SAT1_RTN
    conn.close()


def test_apply_covariances_skips_unmatched(tmp_path):
    conn = _make_conn(tmp_path)
    # Unknown pair entirely.
    cov = spacetrack.normalize_covariance_row(cov_row())
    assert spacetrack.apply_covariances(conn, [cov]) == 0
    # Known pair, but TCA outside the record_events tolerance.
    iss = _seed_satellite(conn, 25544, "ISS (ZARYA)")
    deb = _seed_satellite(conn, 34454, "COSMOS 2251 DEB")
    _seed_conjunction(conn, iss, deb, TCA + timedelta(hours=3))
    assert spacetrack.apply_covariances(conn, [cov]) == 0
    n = conn.execute("SELECT COUNT(*) AS n FROM cdm_messages"
                     " WHERE covariance_json IS NOT NULL").fetchone()["n"]
    assert n == 0
    conn.close()


def test_ingest_derives_velocity_via_sgp4_for_unmatched_pair(tmp_path):
    """Policy step 3: an unmatched pair whose objects both have TLEs gets its
    encounter velocity from SGP4 propagation at TCA - the row is stored and a
    new conjunction is minted; nothing is fabricated."""
    from backend.tests.test_ingestion import make_tle

    conn = _make_conn(tmp_path)
    iss = _seed_satellite(conn, 25544, "ISS (ZARYA)")
    deb = _seed_satellite(conn, 34454, "COSMOS 2251 DEB")
    for sat_id, norad, incl in ((iss, 25544, 51.6), (deb, 34454, 97.6)):
        l1, l2 = make_tle(norad, incl, 100.0, 0.0005, 0.0, 10.0, 15.5, NOW)
        conn.execute(
            "INSERT INTO tle_history (satellite_id, epoch, line1, line2,"
            " ingested_at) VALUES (?, ?, ?, ?, ?)",
            (sat_id, NOW_ISO, l1, l2, NOW_ISO))
    conn.commit()

    stats = spacetrack.ingest_cdms(conn, [spacetrack.normalize_row(raw_row())])
    assert stats["cdms_added"] == 1
    assert stats["conjunctions_new"] == 1
    assert stats["skipped_no_velocity"] == 0
    row = conn.execute(
        "SELECT relative_velocity_ms, miss_distance_m, source"
        " FROM cdm_messages").fetchone()
    assert row["source"] == "spacetrack"
    assert 0.0 < row["relative_velocity_ms"] < 16000.0
    assert row["miss_distance_m"] == pytest.approx(812.0)
    conn.close()
