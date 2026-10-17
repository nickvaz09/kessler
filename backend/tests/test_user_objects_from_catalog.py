"""POST /user-objects/from-catalog - copy a catalog TLE into the user fleet.

Hermetic like test_user_objects.py (temp DB, local app, no network); reuses
its TLE builder, seeding helper and planted-crossing catalog.
"""
from datetime import datetime, timezone

from backend.app.api import user_objects
from backend.tests.test_user_objects import (  # noqa: F401  (client fixture)
    ITEM_KEYS, _conn, _seed_crossing_catalog, _seed_satellites, client,
    make_tle,
)


def _catalog_snapshot():
    conn = _conn()
    sats = [tuple(r) for r in conn.execute(
        "SELECT id, norad_id, name, object_type, last_updated FROM satellites"
        " ORDER BY id")]
    tles = [tuple(r) for r in conn.execute(
        "SELECT id, satellite_id, epoch, line1, line2, ingested_at"
        " FROM tle_history ORDER BY id")]
    conn.close()
    return sats, tles


def _seed_user_sat(epoch):
    """The 'user' satellite lives in the catalog (norad 90500) next to the
    planted crossers, so from-catalog + screen must find the crossers."""
    l1, l2 = make_tle(90500, 51.6, 40.0, 0.0005, 0.0, 350.0, 15.5, epoch)
    conn = _conn()
    _seed_satellites(conn, _seed_crossing_catalog(epoch)
                     + [(90500, "KESSLERSAT 1", l1, l2)])
    conn.close()
    return l1, l2


def test_from_catalog_copies_latest_tle(client):
    l1, l2 = _seed_user_sat(datetime.now(timezone.utc))
    before = _catalog_snapshot()
    response = client.post("/user-objects/from-catalog", json={"norad_id": 90500})
    assert response.status_code == 201
    item = response.json()
    assert set(item) == ITEM_KEYS | {"created"}
    assert item["created"] is True
    assert item["name"] == "KESSLERSAT 1"            # catalog name by default
    assert item["norad_id"] == 90500
    assert item["inclination_deg"] is not None
    conn = _conn()
    row = conn.execute(
        "SELECT line1, line2 FROM user_objects WHERE id = ?",
        (item["id"],)).fetchone()
    conn.close()
    assert (row["line1"], row["line2"]) == (l1, l2)  # exact latest TLE copy
    assert _catalog_snapshot() == before              # catalog untouched


def test_from_catalog_uses_newest_tle(client):
    now = datetime.now(timezone.utc)
    old_l1, old_l2 = make_tle(90600, 97.5, 10.0, 0.001, 0.0, 0.0, 15.1,
                              now.replace(year=now.year - 1))
    new_l1, new_l2 = make_tle(90600, 97.5, 12.0, 0.001, 0.0, 0.0, 15.1, now)
    conn = _conn()
    _seed_satellites(conn, [(90600, "TWO EPOCHS", old_l1, old_l2)])
    sat_id = conn.execute(
        "SELECT id FROM satellites WHERE norad_id = 90600").fetchone()["id"]
    conn.execute(
        "INSERT INTO tle_history (satellite_id, epoch, line1, line2, ingested_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (sat_id, now.strftime("%Y-%m-%dT%H:%M:%SZ"), new_l1, new_l2,
         now.strftime("%Y-%m-%dT%H:%M:%SZ")))
    conn.commit()
    conn.close()
    item = client.post(
        "/user-objects/from-catalog", json={"norad_id": 90600}).json()
    conn = _conn()
    row = conn.execute("SELECT line1 FROM user_objects WHERE id = ?",
                       (item["id"],)).fetchone()
    conn.close()
    assert row["line1"] == new_l1


def test_from_catalog_is_idempotent_per_element_set(client):
    _seed_user_sat(datetime.now(timezone.utc))
    first = client.post("/user-objects/from-catalog", json={"norad_id": 90500})
    again = client.post("/user-objects/from-catalog", json={"norad_id": 90500})
    assert first.status_code == 201 and again.status_code == 200
    assert again.json()["created"] is False
    assert again.json()["id"] == first.json()["id"]
    assert len(client.get("/user-objects").json()["items"]) == 1


def test_from_catalog_custom_name_and_note(client):
    _seed_user_sat(datetime.now(timezone.utc))
    item = client.post("/user-objects/from-catalog", json={
        "norad_id": 90500, "name": "OUR CUBESAT", "note": "demo"}).json()
    assert item["name"] == "OUR CUBESAT" and item["note"] == "demo"


def test_from_catalog_unknown_norad_404(client):
    response = client.post("/user-objects/from-catalog", json={"norad_id": 12345})
    assert response.status_code == 404
    assert "12345" in response.json()["detail"]


def test_from_catalog_no_tle_422(client):
    conn = _conn()
    conn.execute(
        "INSERT INTO satellites (norad_id, name, object_type, last_updated)"
        " VALUES (77777, 'NO TLE', 'PAYLOAD', '2026-01-01T00:00:00Z')")
    conn.commit()
    conn.close()
    response = client.post("/user-objects/from-catalog", json={"norad_id": 77777})
    assert response.status_code == 422
    assert "no TLE" in response.json()["detail"]


def test_from_catalog_rejects_bad_ids(client):
    for bad in (-1, 2 ** 63, "abc"):
        response = client.post("/user-objects/from-catalog", json={"norad_id": bad})
        assert response.status_code == 422


def test_from_catalog_fleet_full_413(client, monkeypatch):
    _seed_user_sat(datetime.now(timezone.utc))
    monkeypatch.setattr(user_objects, "USER_OBJECTS_MAX", 0)
    response = client.post("/user-objects/from-catalog", json={"norad_id": 90500})
    assert response.status_code == 413


def test_from_catalog_then_screen_skips_own_catalog_copy(client):
    """The copy screens like any user object: the planted crossers are found,
    and the object's own catalog entry (same orbit, co-moving) is not an event."""
    _seed_user_sat(datetime.now(timezone.utc))
    item = client.post(
        "/user-objects/from-catalog", json={"norad_id": 90500}).json()
    body = client.post(
        f"/user-objects/{item['id']}/screen?window_hours=6").json()
    secondaries = {ev["secondary"]["norad_id"] for ev in body["events"]}
    assert {80001, 80002} <= secondaries
    assert 90500 not in secondaries


def test_from_catalog_refreshes_same_satellite_in_place(client):
    """After the catalog gets a newer element set, re-adding the satellite
    updates the existing user object rather than creating a second one."""
    now = datetime.now(timezone.utc)
    _seed_user_sat(now.replace(year=now.year - 1))
    first = client.post("/user-objects/from-catalog", json={"norad_id": 90500}).json()
    new_l1, new_l2 = make_tle(90500, 51.6, 42.0, 0.0005, 0.0, 10.0, 15.5, now)
    conn = _conn()
    sat_id = conn.execute(
        "SELECT id FROM satellites WHERE norad_id = 90500").fetchone()["id"]
    conn.execute(
        "INSERT INTO tle_history (satellite_id, epoch, line1, line2, ingested_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (sat_id, now.strftime("%Y-%m-%dT%H:%M:%SZ"), new_l1, new_l2,
         now.strftime("%Y-%m-%dT%H:%M:%SZ")))
    conn.commit()
    conn.close()
    again = client.post("/user-objects/from-catalog", json={"norad_id": 90500})
    assert again.status_code == 200
    body = again.json()
    assert body["id"] == first["id"] and body["updated"] is True
    assert len(client.get("/user-objects").json()["items"]) == 1
    conn = _conn()
    row = conn.execute("SELECT line1 FROM user_objects WHERE id = ?",
                       (first["id"],)).fetchone()
    conn.close()
    assert row["line1"] == new_l1
