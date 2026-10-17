"""Watch Mode tests - hermetic: temp DB + DATA_DIR, FakeClient injected
through service.watch_pass's `client` parameter (the test_assistant idiom).
No live Claude API calls, no network. Contract: docs/CONTRACTS.md
'v13 additions' section A.
"""
import json
import re
import sys
import types

import pytest
from fastapi.testclient import TestClient

from backend.app import config, db
from backend.app.api import assistant as assistant_api
from backend.app.api import watchlog as watchlog_api
from backend.app.assistant import service

# Real ISS TLE fixture strings (parsed by the real propagate helpers).
ISS_TLE1 = "1 25544U 98067A   24007.58333333  .00016717  00000-0  30777-3 0  9994"
ISS_TLE2 = "2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.49560532430089"


# --- FakeClient mimicking anthropic response shapes (create-only) -------------

def text_block(text):
    return types.SimpleNamespace(type="text", text=text)


def thinking_block():
    return types.SimpleNamespace(type="thinking", thinking="")


def tool_use(block_id, name, tool_input):
    return types.SimpleNamespace(
        type="tool_use", id=block_id, name=name, input=tool_input
    )


def resp(blocks, stop="end_turn"):
    return types.SimpleNamespace(content=blocks, stop_reason=stop)


class FakeMessages:
    def __init__(self, script, cycle=False):
        self.script = list(script)
        self.cycle = cycle
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.cycle:
            return self.script[0]
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script, cycle=False):
        self.messages = FakeMessages(script, cycle=cycle)


def _raising_client(exc_factory):
    class RaisingMessages:
        def create(self, **kwargs):
            raise exc_factory()

    client = types.SimpleNamespace()
    client.messages = RaisingMessages()
    return client


def _make_api_error(cls, status_code, message):
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    return cls(message, response=response, body=None)


# --- fixtures -----------------------------------------------------------------

def _seed(conn):
    conn.executescript(
        f"""
    INSERT INTO satellites (id, norad_id, name, object_type, intl_designator, last_updated) VALUES
        (1, 25544, 'ISS (ZARYA)', 'PAYLOAD', '1998-067A', '2026-07-21T13:02:41Z'),
        (2, 34454, 'COSMOS 2251 DEB', 'DEBRIS', NULL, '2026-07-21T13:02:41Z'),
        (3, 43556, 'AEROCUBE 12A', 'PAYLOAD', '2018-046G', '2026-07-21T13:02:41Z');

    INSERT INTO tle_history (satellite_id, epoch, line1, line2, ingested_at) VALUES
        (1, '2026-07-21T06:11:52Z', '{ISS_TLE1}', '{ISS_TLE2}', '2026-07-21T13:02:41Z'),
        (2, '2026-07-21T02:00:00Z', '{ISS_TLE1}', '{ISS_TLE2}', '2026-07-21T13:02:41Z');

    INSERT INTO conjunctions (id, primary_object_id, secondary_object_id,
                              time_of_closest_approach, first_seen_at, status) VALUES
        (1, 1, 2, '2026-07-23T04:12:09Z', '2026-07-20T10:00:00Z', 'active'),
        (2, 1, 3, '2026-07-22T10:00:00Z', '2026-07-21T06:00:00Z', 'active');

    INSERT INTO cdm_messages (conjunction_id, message_time, miss_distance_m,
                              relative_velocity_ms, probability_of_collision, source) VALUES
        (1, '2026-07-21T10:00:00Z', 812.4, 14210.3, 3.1e-05, 'screening'),
        (2, '2026-07-21T09:00:00Z', 400.0, 12000.0, 2.0e-04, 'screening');

    INSERT INTO risk_predictions (conjunction_id, predicted_final_risk,
                                  prediction_time, model_version) VALUES
        (1, 4.4e-05, '2026-07-21T11:00:00Z', 'heuristic-0.1'),
        (2, 3.0e-04, '2026-07-21T11:00:00Z', 'heuristic-0.1');
    """
    )
    conn.commit()


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    # Freeze the honest-active cutoff so the fixture conjunctions always read
    # as active regardless of the real clock.
    from backend.app.api import conjunctions as conjunctions_api
    monkeypatch.setattr(conjunctions_api, "now_iso",
                        lambda: "2026-07-21T13:10:00Z")
    conn = db.get_conn()
    db.init_db(conn)
    _seed(conn)
    conn.close()
    return tmp_path


@pytest.fixture()
def http_client(seeded_db):
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


WATCH_ENTRY = {
    "summary_md": "- **CRITICAL 0 / ELEVATED 1**: conj 2 (heuristic-0.1)\n"
                  "- No changes since last run.",
    "spoken": "One elevated conjunction on the board. No critical events "
              "this shift.",
}


def _watch_client(final=None, rounds=()):
    """FakeClient scripted with optional tool_use rounds, then a
    structured-outputs style final turn (first text block carries JSON)."""
    script = list(rounds)
    text = final if isinstance(final, str) else json.dumps(WATCH_ENTRY)
    script.append(resp([thinking_block(), text_block(text)]))
    return FakeClient(script)


def _row_count(conn):
    return conn.execute("SELECT COUNT(*) FROM watchlog").fetchone()[0]


# --- schema -------------------------------------------------------------------

def test_watchlog_table_in_both_dialects():
    assert "CREATE TABLE IF NOT EXISTS watchlog" in db.SCHEMA
    assert "CREATE TABLE IF NOT EXISTS watchlog" in db.SCHEMA_PG
    assert "watchlog" in db._ID_TABLES  # pg lastrowid support


def test_existing_db_gains_watchlog_via_init_db(tmp_path):
    # Simulate an older DB: init, drop the new table, re-run init_db - the
    # CREATE TABLE IF NOT EXISTS path recreates it without touching the rest.
    path = tmp_path / "old.db"
    conn = db.get_conn(path)
    try:
        db.init_db(conn)
        conn.execute("DROP TABLE watchlog")
        conn.commit()
        db.init_db(conn)
        conn.execute(
            "INSERT INTO watchlog (created_at, model, summary_md, spoken)"
            " VALUES ('2026-08-17T00:00:00Z', 'm', 's', 'sp')"
        )
        conn.commit()
        assert _row_count(conn) == 1
    finally:
        conn.close()


# --- watch_pass ---------------------------------------------------------------

def test_watch_pass_runs_loop_and_inserts_row(seeded_db):
    fake = _watch_client(rounds=[
        resp(
            [
                tool_use("tu_1", "get_status", {}),
                tool_use("tu_2", "list_conjunctions", {"limit": 5}),
            ],
            stop="tool_use",
        ),
    ])
    conn = db.get_conn()
    try:
        entry = service.watch_pass(conn, client=fake)
        assert entry is not None
        assert set(entry) == {"id", "created_at", "model", "summary_md",
                              "spoken"}
        assert entry["model"] == "claude-opus-5-5"
        assert entry["summary_md"] == WATCH_ENTRY["summary_md"]
        assert entry["spoken"] == WATCH_ENTRY["spoken"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                            entry["created_at"])

        row = conn.execute(
            "SELECT * FROM watchlog WHERE id = ?", (entry["id"],)
        ).fetchone()
        assert row["summary_md"] == WATCH_ENTRY["summary_md"]
        assert row["spoken"] == WATCH_ENTRY["spoken"]
        meta = json.loads(row["meta_json"])
        assert meta["effort"] == "low"
        assert meta["rounds"] == 2
    finally:
        conn.close()

    # chat-loop mechanics: assistant turn appended, all tool_results in one
    # user message, real DATA handlers executed against the seeded DB
    second = fake.messages.calls[1]["messages"]
    assert second[0] == {"role": "user",
                         "content": service.WATCH_USER_MESSAGE}
    assert second[-2]["role"] == "assistant"
    tool_msg = second[-1]
    assert tool_msg["role"] == "user"
    assert [b["tool_use_id"] for b in tool_msg["content"]] == ["tu_1", "tu_2"]
    status_payload = json.loads(tool_msg["content"][0]["content"])
    assert status_payload["catalog_size"] == 3
    assert status_payload["active_conjunctions"] == 2
    listing = json.loads(tool_msg["content"][1]["content"])
    assert listing["count"] == 2


def test_watch_pass_request_shape_data_tools_only(seeded_db):
    fake = _watch_client()
    conn = db.get_conn()
    try:
        assert service.watch_pass(conn, client=fake) is not None
    finally:
        conn.close()

    call = fake.messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == service.MAX_TOKENS
    assert call["thinking"] == {"type": "adaptive"}
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in call

    # structured outputs: strict JSON schema, effort default 'low'
    assert call["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": service.WATCHLOG_SCHEMA},
    }
    schema = call["output_config"]["format"]["schema"]
    assert schema["required"] == ["summary_md", "spoken"]
    assert schema["additionalProperties"] is False

    # DATA tools only, in deterministic TOOLS order - never UI-action,
    # report, or consent-gated tools
    names = [t["name"] for t in call["tools"]]
    assert names == list(service.WATCH_TOOL_NAMES)
    banned_tools = {
        "navigate_user", "select_on_globe", "set_globe_filter",
        "set_globe_speed", "set_table_filter", "fill_field",
        "show_trajectory", "show_encounter", "open_simulator",
        "create_report", "run_pipeline_refresh", "screen_user_object",
    }
    assert banned_tools.isdisjoint(names)
    for reviewed in ("get_status", "get_changes", "list_conjunctions",
                     "get_anomalies"):
        assert reviewed in names

    # frozen cached system prompt first, frozen addendum after it
    assert call["system"][0]["text"] is service.prompt.SYSTEM_PROMPT
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    addendum = call["system"][1]["text"]
    assert addendum.startswith("You are writing the change-of-watch log entry")
    assert "1500" in addendum and "450" in addendum
    # headless: one frozen user message, no volatile envelope
    assert call["messages"] == [
        {"role": "user", "content": service.WATCH_USER_MESSAGE}
    ]


def test_watch_pass_effort_env_override(seeded_db, monkeypatch):
    monkeypatch.setenv("KESSLER_AI_EFFORT", "medium")
    fake = _watch_client()
    conn = db.get_conn()
    try:
        assert service.watch_pass(conn, client=fake) is not None
    finally:
        conn.close()
    assert fake.messages.calls[0]["output_config"]["effort"] == "medium"


def test_watch_pass_keyless_returns_none(seeded_db, monkeypatch, capsys):
    from fastapi import HTTPException

    def no_credentials():
        raise HTTPException(status_code=503, detail=service.AUTH_DETAIL)

    monkeypatch.setattr(service, "_default_client", no_credentials)
    conn = db.get_conn()
    try:
        assert service.watch_pass(conn) is None
        assert _row_count(conn) == 0
    finally:
        conn.close()
    assert "[watch] pass failed" in capsys.readouterr().err


def test_watch_pass_claude_errors_swallowed(seeded_db, capsys):
    import anthropic
    import httpx

    factories = [
        lambda: _make_api_error(anthropic.AuthenticationError, 401, "bad key"),
        lambda: _make_api_error(anthropic.RateLimitError, 429, "slow down"),
        lambda: _make_api_error(anthropic.APIStatusError, 500, "server broke"),
        lambda: anthropic.APIConnectionError(
            request=httpx.Request("POST",
                                  "https://api.anthropic.com/v1/messages")),
    ]
    conn = db.get_conn()
    try:
        for factory in factories:
            assert service.watch_pass(conn,
                                      client=_raising_client(factory)) is None
        assert _row_count(conn) == 0
    finally:
        conn.close()
    assert "[watch] pass failed" in capsys.readouterr().err


def test_watch_pass_refusal_returns_none(seeded_db):
    conn = db.get_conn()
    try:
        assert service.watch_pass(
            conn, client=FakeClient([resp([], stop="refusal")])
        ) is None
        assert _row_count(conn) == 0
    finally:
        conn.close()


def test_watch_pass_malformed_output_returns_none(seeded_db):
    cases = [
        "this is not json {",
        json.dumps({"spoken": "x"}),                      # missing summary_md
        json.dumps({"summary_md": "x"}),                  # missing spoken
        json.dumps({"summary_md": "", "spoken": "x"}),    # empty string
        json.dumps({"summary_md": 3, "spoken": "x"}),     # wrong type
        json.dumps(["summary_md", "spoken"]),             # not an object
    ]
    conn = db.get_conn()
    try:
        for payload in cases:
            assert service.watch_pass(
                conn, client=_watch_client(final=payload)
            ) is None
        assert _row_count(conn) == 0
    finally:
        conn.close()


def test_watch_pass_iteration_cap_is_6(seeded_db):
    fake = FakeClient(
        [resp([tool_use("tu_1", "get_status", {})], stop="tool_use")],
        cycle=True,
    )
    conn = db.get_conn()
    try:
        assert service.watch_pass(conn, client=fake) is None
        assert _row_count(conn) == 0
    finally:
        conn.close()
    assert len(fake.messages.calls) == 6


# --- scheduler hook -----------------------------------------------------------

def test_scheduler_hook_guarded_by_watch_mode_env(seeded_db, monkeypatch):
    from backend.app import main as main_mod

    calls = []
    monkeypatch.setattr(service, "watch_pass",
                        lambda conn, client=None: calls.append(conn) or None)
    conn = db.get_conn()
    try:
        monkeypatch.delenv("KESSLER_WATCH_MODE", raising=False)
        main_mod._watch_pass_after_refresh(conn)
        monkeypatch.setenv("KESSLER_WATCH_MODE", "0")
        main_mod._watch_pass_after_refresh(conn)
        assert calls == []  # default OFF: never burns tokens

        monkeypatch.setenv("KESSLER_WATCH_MODE", "1")
        main_mod._watch_pass_after_refresh(conn)
        assert calls == [conn]
    finally:
        conn.close()


def test_scheduled_refresh_watch_passes_only_after_success(seeded_db,
                                                          monkeypatch):
    from backend.app import main as main_mod
    import backend.app.ingestion as ingestion_pkg

    order = []
    pipeline = types.ModuleType("backend.app.ingestion.pipeline")
    pipeline.RefreshBusyError = type("RefreshBusyError", (RuntimeError,), {})
    pipeline.refresh = lambda conn, demo=False: order.append("refresh") or {}
    monkeypatch.setitem(sys.modules, "backend.app.ingestion.pipeline", pipeline)
    monkeypatch.setattr(ingestion_pkg, "pipeline", pipeline, raising=False)
    monkeypatch.setenv("KESSLER_WATCH_MODE", "1")
    monkeypatch.setattr(service, "watch_pass",
                        lambda conn, client=None: order.append("watch") or None)

    main_mod._run_scheduled_refresh()
    assert order == ["refresh", "watch"]

    order.clear()

    def failing_refresh(conn, demo=False):
        order.append("refresh")
        raise RuntimeError("fetch failed")

    pipeline.refresh = failing_refresh
    main_mod._run_scheduled_refresh()  # error swallowed by the scheduler
    assert order == ["refresh"]  # failed refresh: no watch pass


# --- GET /watchlog ------------------------------------------------------------

def _insert_entry(conn, created_at, summary="- all quiet",
                  spoken="All quiet."):
    conn.execute(
        "INSERT INTO watchlog (created_at, model, summary_md, spoken,"
        " meta_json) VALUES (?, 'claude-opus-4-8', ?, ?, NULL)",
        (created_at, summary, spoken),
    )


def test_get_watchlog_empty_is_200(http_client):
    response = http_client.get("/watchlog")
    assert response.status_code == 200
    assert response.json() == {"entries": []}


def test_get_watchlog_newest_first_shape_and_limits(http_client):
    conn = db.get_conn()
    try:
        _insert_entry(conn, "2026-08-15T06:00:00Z", summary="- oldest")
        _insert_entry(conn, "2026-08-16T06:00:00Z", summary="- middle")
        _insert_entry(conn, "2026-08-17T06:00:00Z", summary="- newest")
        conn.commit()
    finally:
        conn.close()

    body = http_client.get("/watchlog").json()
    assert [e["summary_md"] for e in body["entries"]] == [
        "- newest", "- middle", "- oldest"]
    for entry in body["entries"]:
        # meta_json stays internal - entries expose exactly the contract keys
        # plus the tts playback signature over the spoken line
        assert set(entry) == {"id", "created_at", "model", "summary_md",
                              "spoken", "spoken_sig"}
        assert entry["spoken_sig"] == assistant_api.tts_sign(entry["spoken"])

    # explicit limit
    body = http_client.get("/watchlog?limit=1").json()
    assert [e["summary_md"] for e in body["entries"]] == ["- newest"]
    # cap 50: an oversized limit clamps instead of erroring
    assert http_client.get("/watchlog?limit=999").status_code == 200
    assert len(http_client.get("/watchlog?limit=999").json()["entries"]) == 3
    # floor: nonsense low values clamp to 1
    assert len(http_client.get("/watchlog?limit=0").json()["entries"]) == 1


def test_get_watchlog_default_limit_is_10(http_client):
    conn = db.get_conn()
    try:
        for i in range(12):
            _insert_entry(conn, f"2026-08-17T06:00:{i:02d}Z",
                          summary=f"- entry {i}")
        conn.commit()
    finally:
        conn.close()
    body = http_client.get("/watchlog").json()
    assert len(body["entries"]) == 10
    assert body["entries"][0]["summary_md"] == "- entry 11"


# --- POST /watchlog/run -------------------------------------------------------

def test_run_watchlog_returns_new_entry(http_client, monkeypatch):
    fake = _watch_client()
    monkeypatch.setattr(service, "_default_client", lambda: fake)

    response = http_client.post("/watchlog/run")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"id", "created_at", "model", "summary_md",
                         "spoken", "spoken_sig"}
    assert body["summary_md"] == WATCH_ENTRY["summary_md"]
    assert body["spoken"] == WATCH_ENTRY["spoken"]
    assert body["model"] == "claude-opus-5-5"

    listed = http_client.get("/watchlog").json()["entries"]
    assert [e["id"] for e in listed] == [body["id"]]


def test_run_watchlog_keyless_is_503_with_detail(http_client, monkeypatch):
    from fastapi import HTTPException

    def no_credentials():
        raise HTTPException(status_code=503, detail=service.AUTH_DETAIL)

    monkeypatch.setattr(service, "_default_client", no_credentials)
    response = http_client.post("/watchlog/run")
    assert response.status_code == 503
    assert response.json()["detail"] == watchlog_api.RUN_FAIL_DETAIL
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
    assert http_client.get("/watchlog").json() == {"entries": []}


def test_run_watchlog_claude_failure_is_503_not_500(http_client, monkeypatch):
    import anthropic

    fake = _raising_client(
        lambda: _make_api_error(anthropic.APIStatusError, 500, "server broke")
    )
    monkeypatch.setattr(service, "_default_client", lambda: fake)
    response = http_client.post("/watchlog/run")
    assert response.status_code == 503
    assert "detail" in response.json()
