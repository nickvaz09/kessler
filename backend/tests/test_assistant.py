"""Kessler AI backend tests - hermetic: temp DB + DATA_DIR, FakeClient
injected through service.chat's `client` parameter. No live Claude API calls,
no network. Contract: docs/CONTRACTS.md 'Kessler AI - backend'.
"""
import json
import sys
import types

import pytest
from fastapi.testclient import TestClient

from backend.app import config, db
from backend.app.assistant import report, service

# Real ISS TLE fixture strings (parsed by the real propagate helpers).
ISS_TLE1 = "1 25544U 98067A   24007.58333333  .00016717  00000-0  30777-3 0  9994"
ISS_TLE2 = "2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.49560532430089"


# --- FakeClient mimicking anthropic response shapes --------------------------

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


def stream_error(exc, deltas=()):
    """Script entry for FakeMessages.stream: the streamed turn yields the
    given text deltas, then raises exc mid-stream."""
    return types.SimpleNamespace(kind="stream_error", exc=exc, deltas=list(deltas))


class FakeStream:
    """Mimics anthropic's MessageStreamManager (0.117): a context manager
    whose handle exposes a text_stream iterator and get_final_message()."""

    def __init__(self, item):
        self._item = item

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    @property
    def text_stream(self):
        item = self._item

        def deltas():
            if getattr(item, "kind", None) == "stream_error":
                yield from item.deltas
                raise item.exc
            for block in getattr(item, "content", None) or []:
                if getattr(block, "type", None) == "text" and block.text:
                    yield block.text

        return deltas()

    def get_final_message(self):
        if getattr(self._item, "kind", None) == "stream_error":
            raise self._item.exc
        return self._item


class FakeMessages:
    def __init__(self, script, cycle=False):
        self.script = list(script)
        self.cycle = cycle
        self.calls = []

    def _next(self, kwargs):
        self.calls.append(kwargs)
        if self.cycle:
            return self.script[0]
        return self.script.pop(0)

    def create(self, **kwargs):
        return self._next(kwargs)

    def stream(self, **kwargs):
        return FakeStream(self._next(kwargs))


class FakeClient:
    def __init__(self, script, cycle=False):
        self.messages = FakeMessages(script, cycle=cycle)


# --- fixtures ----------------------------------------------------------------

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
        (2, 1, 3, '2026-07-22T10:00:00Z', '2026-07-21T06:00:00Z', 'active'),
        (3, 1, 2, '2026-07-18T00:00:00Z', '2026-07-16T00:00:00Z', 'past');

    INSERT INTO cdm_messages (conjunction_id, message_time, miss_distance_m,
                              relative_velocity_ms, probability_of_collision, source) VALUES
        (1, '2026-07-20T10:00:00Z', 1500.0, 14210.3, 1.1e-06, 'screening'),
        (1, '2026-07-21T10:00:00Z', 812.4, 14210.3, 3.1e-05, 'screening'),
        (2, '2026-07-21T09:00:00Z', 400.0, 12000.0, 2.0e-04, 'screening'),
        (3, '2026-07-17T00:00:00Z', 5000.0, 10000.0, 1.0e-07, 'screening');

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
    # Freeze the honest-active cutoff so the fixture conjunctions (TCAs on
    # 2026-07-22/23) always read as active regardless of the real clock.
    from backend.app.api import conjunctions as conjunctions_api
    monkeypatch.setattr(conjunctions_api, "now_iso", lambda: "2026-07-21T13:10:00Z")
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


CONTEXT = {"page": "dashboard", "url": "/", "selected_norad": 25544}
USER = [{"role": "user", "content": "How do things look?"}]


def _chat(fake, messages=None, context=None):
    return service.chat(messages or list(USER), context or dict(CONTEXT),
                        client=fake)


# --- request construction ----------------------------------------------------

def test_end_turn_reply_and_request_shape(seeded_db):
    fake = FakeClient([resp([thinking_block(), text_block("All nominal.")])])
    result = _chat(fake)

    assert result["reply"] == "All nominal."
    assert result["actions"] == []
    assert result["report_url"] is None
    assert result["model"] == "claude-opus-5-5"

    call = fake.messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == 16000
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}  # default
    # never sampled params on this model family
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in call
    # frozen cached system prompt
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "KESSLER AI" in call["system"][0]["text"]
    assert len(call["tools"]) == 27  # + plan_avoidance; + get_space_weather
    # volatile context in the FINAL user message envelope, not the system
    envelope = call["messages"][-1]["content"]
    assert envelope.startswith("<context>")
    assert "page=dashboard" in envelope and "selected_norad=25544" in envelope
    assert "live status: catalog 3 objects" in envelope
    assert envelope.endswith("How do things look?")
    # cache safety: no volatile values leak into the frozen system prompt
    assert "selected_norad=25544" not in call["system"][0]["text"]
    assert "live status:" not in call["system"][0]["text"]
    assert call["system"][0]["text"] is service.prompt.SYSTEM_PROMPT


# --- tool loop ---------------------------------------------------------------

def test_tool_loop_feeds_result_back(seeded_db):
    fake = FakeClient([
        resp([tool_use("tu_1", "get_status", {})], stop="tool_use"),
        resp([text_block("3 objects tracked; 2 active conjunctions.")]),
    ])
    result = _chat(fake)
    assert result["reply"] == "3 objects tracked; 2 active conjunctions."

    second = fake.messages.calls[1]["messages"]
    # appended: assistant turn with the tool_use, then one user tool_result msg
    assert second[-2]["role"] == "assistant"
    tool_results = second[-1]
    assert tool_results["role"] == "user"
    [result_block] = tool_results["content"]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "tu_1"
    payload = json.loads(result_block["content"])
    assert payload["catalog_size"] == 3
    assert payload["active_conjunctions"] == 2
    assert "is_error" not in result_block


def test_parallel_tool_results_in_one_message(seeded_db):
    fake = FakeClient([
        resp(
            [
                tool_use("tu_a", "get_status", {}),
                tool_use("tu_b", "list_conjunctions", {"risk": "critical"}),
            ],
            stop="tool_use",
        ),
        resp([text_block("done")]),
    ])
    _chat(fake)

    second = fake.messages.calls[1]["messages"]
    tool_msg = second[-1]
    assert tool_msg["role"] == "user"
    assert [b["tool_use_id"] for b in tool_msg["content"]] == ["tu_a", "tu_b"]
    assert all(b["type"] == "tool_result" for b in tool_msg["content"])
    listing = json.loads(tool_msg["content"][1]["content"])
    assert listing["count"] == 1
    assert listing["items"][0]["id"] == 2
    assert listing["items"][0]["risk_level"] == "critical"


def test_data_tool_get_conjunction_detail(seeded_db):
    fake = FakeClient([
        resp([tool_use("tu_1", "get_conjunction", {"id": 1})], stop="tool_use"),
        resp([text_block("ok")]),
    ])
    _chat(fake)
    payload = json.loads(
        fake.messages.calls[1]["messages"][-1]["content"][0]["content"]
    )
    assert payload["id"] == 1
    assert len(payload["cdms"]) == 2
    assert payload["risk_predictions"][0]["model_version"] == "heuristic-0.1"


def test_unknown_and_missing_entities_are_error_results(seeded_db):
    fake = FakeClient([
        resp(
            [
                tool_use("tu_1", "no_such_tool", {}),
                tool_use("tu_2", "get_object", {"norad": 99999}),
            ],
            stop="tool_use",
        ),
        resp([text_block("ok")]),
    ])
    _chat(fake)
    blocks = fake.messages.calls[1]["messages"][-1]["content"]
    assert blocks[0]["is_error"] is True
    assert "unknown tool" in json.loads(blocks[0]["content"])["error"]
    assert blocks[1]["is_error"] is True
    assert "not found" in json.loads(blocks[1]["content"])["error"]


def test_run_pipeline_refresh_maps_to_pipeline(seeded_db, monkeypatch):
    monkeypatch.setenv("KESSLER_DEV_ENDPOINTS", "1")
    pipeline = types.ModuleType("backend.app.ingestion.pipeline")
    seen = {}

    def _refresh(conn, demo=False):
        seen["demo"] = demo
        return {"fetched": 1, "duration_s": 0.1}

    pipeline.refresh = _refresh
    import backend.app.ingestion as ingestion_pkg

    monkeypatch.setitem(sys.modules, "backend.app.ingestion.pipeline", pipeline)
    monkeypatch.setattr(ingestion_pkg, "pipeline", pipeline, raising=False)

    fake = FakeClient([
        resp([tool_use("tu_1", "run_pipeline_refresh", {"demo": True})],
             stop="tool_use"),
        resp([text_block("refreshed")]),
    ])
    _chat(fake)
    assert seen == {"demo": True}
    payload = json.loads(
        fake.messages.calls[1]["messages"][-1]["content"][0]["content"]
    )
    assert payload["fetched"] == 1


def test_run_pipeline_refresh_gated_like_ingest_endpoint(seeded_db, monkeypatch):
    """Without KESSLER_DEV_ENDPOINTS the chat cannot start a refresh (the
    same gate as POST /ingest/refresh) - it gets a tool error instead."""
    monkeypatch.delenv("KESSLER_DEV_ENDPOINTS", raising=False)
    import backend.app.ingestion.pipeline as real_pipeline
    called = []
    monkeypatch.setattr(real_pipeline, "refresh",
                        lambda conn, demo=False: called.append(demo))
    fake = FakeClient([
        resp([tool_use("tu_1", "run_pipeline_refresh", {"demo": True})],
             stop="tool_use"),
        resp([text_block("cannot refresh")]),
    ])
    _chat(fake)
    assert called == []
    block = fake.messages.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert "disabled" in json.loads(block["content"])["error"]


def test_leading_assistant_turn_folded_into_first_user_turn(seeded_db):
    """SHIFT BRIEF puts an assistant greeting first; the API rejects a
    transcript that opens with an assistant turn, so it is folded in."""
    fake = FakeClient([resp([text_block("ok")])])
    service.chat(
        [{"role": "assistant", "content": "SHIFT BRIEF: 3 elevated events."},
         {"role": "user", "content": "tell me more"}],
        dict(CONTEXT), client=fake)
    sent = fake.messages.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["user"]
    assert "SHIFT BRIEF: 3 elevated events." in sent[0]["content"]
    assert sent[0]["content"].rstrip().endswith("tell me more")


def test_get_passes_reads_site_from_context(seeded_db):
    from backend.app.assistant import tools as tools_mod

    rt = tools_mod.ToolRuntime(db.get_conn(), {"site": {"lat": 40.0, "lon": -75.0}})
    out, is_error = rt.execute("get_passes", {"norad": 25544})
    # no AttributeError: the site reaches the tool (the object may or may not
    # have passes in the fixture window)
    assert "AttributeError" not in out
    rt_nosite = tools_mod.ToolRuntime(db.get_conn())
    out, is_error = rt_nosite.execute("get_passes", {"norad": 25544})
    assert is_error is True and "AttributeError" not in out


# --- UI actions --------------------------------------------------------------

def test_ui_actions_collected_in_order(seeded_db):
    fake = FakeClient([
        resp(
            [
                tool_use("tu_1", "set_table_filter", {"risk": "critical"}),
                tool_use("tu_2", "select_on_globe", {"norad": 25544}),
                tool_use("tu_3", "set_globe_speed", {"mult": 600}),
                tool_use("tu_4", "navigate_user",
                         {"url": "/object.html?id=25544"}),
            ],
            stop="tool_use",
        ),
        resp([text_block("Focused the display.")]),
    ])
    result = _chat(fake)
    assert result["actions"] == [
        {"type": "set_table_filter", "risk": "critical"},
        {"type": "select_on_globe", "norad": 25544},
        {"type": "set_globe_speed", "mult": 600},
        {"type": "navigate", "url": "/object.html?id=25544"},
    ]
    # UI tools ack with {"ok": true}
    blocks = fake.messages.calls[1]["messages"][-1]["content"]
    assert all(json.loads(b["content"]) == {"ok": True} for b in blocks)


def test_navigate_rejects_absolute_and_protocol_relative_urls(seeded_db):
    fake = FakeClient([
        resp(
            [
                tool_use("tu_1", "navigate_user", {"url": "https://evil.example/x"}),
                tool_use("tu_2", "navigate_user", {"url": "//evil.example/x"}),
                # browsers read a backslash as a slash: protocol-relative too
                tool_use("tu_3", "navigate_user", {"url": "/\\evil.example/x"}),
                tool_use("tu_4", "navigate_user", {"url": "/\t/evil.example/x"}),
            ],
            stop="tool_use",
        ),
        resp([text_block("ok")]),
    ])
    result = _chat(fake)
    assert result["actions"] == []
    blocks = fake.messages.calls[1]["messages"][-1]["content"]
    assert all(b["is_error"] is True for b in blocks)


def test_fill_field_validates_and_queues_set_field(seeded_db):
    fake = FakeClient([
        resp(
            [
                tool_use("tu_1", "fill_field",
                         {"field_id": "conj-search", "value": "ISS"}),
                tool_use("tu_2", "fill_field",
                         {"field_id": "bad id!", "value": "x"}),
                tool_use("tu_3", "fill_field",
                         {"field_id": "add-note", "value": "x" * 501}),
            ],
            stop="tool_use",
        ),
        resp([text_block("ok")]),
    ])
    result = _chat(fake)
    # only the valid call queues an action; bad id / oversized value error
    assert result["actions"] == [
        {"type": "set_field", "field_id": "conj-search", "value": "ISS"}
    ]
    blocks = fake.messages.calls[1]["messages"][-1]["content"]
    assert [bool(b.get("is_error")) for b in blocks] == [False, True, True]


def test_globe_filter_and_trajectory_validation(seeded_db):
    fake = FakeClient([
        resp(
            [
                tool_use("tu_1", "set_globe_filter",
                         {"types": ["DEBRIS"], "risk_only": True}),
                tool_use("tu_2", "show_trajectory",
                         {"norad": 25544, "partner_norad": 34454}),
                tool_use("tu_3", "set_globe_speed", {"mult": 42}),
            ],
            stop="tool_use",
        ),
        resp([text_block("ok")]),
    ])
    result = _chat(fake)
    assert result["actions"] == [
        {"type": "set_globe_filter", "types": ["DEBRIS"], "risk_only": True},
        {"type": "show_trajectory", "norad": 25544, "partner_norad": 34454},
    ]
    blocks = fake.messages.calls[1]["messages"][-1]["content"]
    assert blocks[2]["is_error"] is True  # mult 42 rejected


# --- loop termination --------------------------------------------------------

def test_iteration_cap(seeded_db):
    fake = FakeClient(
        [resp([tool_use("tu_1", "get_status", {})], stop="tool_use")],
        cycle=True,
    )
    result = _chat(fake)
    assert len(fake.messages.calls) == 12
    assert "cap" in result["reply"]


def test_refusal_returns_terse_decline(seeded_db):
    fake = FakeClient([resp([], stop="refusal")])
    result = _chat(fake)
    assert result["reply"] == service.REFUSAL_REPLY
    assert result["actions"] == []


def test_pause_turn_resends_with_assistant_content(seeded_db):
    paused = resp([text_block("working...")], stop="pause_turn")
    fake = FakeClient([paused, resp([text_block("final answer")])])
    result = _chat(fake)
    assert result["reply"] == "final answer"
    second = fake.messages.calls[1]["messages"]
    assert second[-1]["role"] == "assistant"
    assert second[-1]["content"] is paused.content


# --- reports -----------------------------------------------------------------

REPORT_SECTIONS = [
    {
        "heading": "Assessment",
        "body": "**High** interest event with `Pc` rising.\n\n"
                "- monitor closely\n- re-screen at next pass\n\n"
                "<script>alert(1)</script>",
    }
]


def test_create_report_tool_writes_file_and_sets_report_url(seeded_db):
    fake = FakeClient([
        resp(
            [tool_use("tu_1", "create_report", {
                "title": "Conjunction 1 briefing",
                "subject": {"type": "conjunction", "id": 1},
                "sections": REPORT_SECTIONS,
            })],
            stop="tool_use",
        ),
        resp([text_block("Report ready.")]),
    ])
    result = _chat(fake)
    assert result["report_url"] is not None
    assert result["report_url"].startswith("/reports/")
    uid = result["report_url"].rsplit("/", 1)[1]
    path = seeded_db / "reports" / f"{uid}.html"
    assert path.is_file()


def test_report_html_content_real_data_and_escaped_narrative(seeded_db):
    conn = db.get_conn()
    try:
        url = report.create_report(
            conn, "Conjunction 1 briefing",
            {"type": "conjunction", "id": 1}, REPORT_SECTIONS,
        )
    finally:
        conn.close()
    uid = url.rsplit("/", 1)[1]
    page = (seeded_db / "reports" / f"{uid}.html").read_text()
    # real DB data
    assert "ISS (ZARYA)" in page and "COSMOS 2251 DEB" in page
    assert "2026-07-23T04:12:09Z" in page          # TCA
    assert "heuristic-0.1" in page                 # model version cited
    assert "3.10e-05" in page                      # latest CDM Pc
    # narrative: escaped first, then minimal markdown
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<strong>High</strong>" in page
    assert "<code>Pc</code>" in page
    assert "<li>monitor closely</li>" in page
    # print affordance + print light theme, dark tokens inline
    assert "PRINT / SAVE AS PDF" in page and "window.print()" in page
    assert "@media print" in page and "--k-bg:#0a0d11" in page
    # self-contained: no external requests - the only http token is the SVG
    # xmlns inside the favicon's inline data: URI (a namespace, not a fetch)
    stripped = page.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "http://" not in stripped and "https://" not in stripped


def test_create_report_retention_prunes_oldest(seeded_db, monkeypatch):
    """create_report keeps only the newest REPORTS_KEEP files by mtime."""
    import os

    monkeypatch.setattr(report, "REPORTS_KEEP", 3)
    reports_dir = seeded_db / "reports"
    conn = db.get_conn()
    try:
        urls = [report.create_report(conn, f"Overview {i}",
                                     {"type": "overview"}, [])
                for i in range(3)]
        # Backdate the first two so mtime ordering is unambiguous.
        for age, url in enumerate(urls[:2], start=1):
            path = reports_dir / f"{url.rsplit('/', 1)[1]}.html"
            stamp = path.stat().st_mtime - 1000.0 * age
            os.utime(path, (stamp, stamp))
        newest = report.create_report(
            conn, "Overview 3", {"type": "overview"}, [])
    finally:
        conn.close()
    kept = {p.name for p in reports_dir.glob("*.html")}
    assert len(kept) == 3
    assert f"{newest.rsplit('/', 1)[1]}.html" in kept
    assert f"{urls[2].rsplit('/', 1)[1]}.html" in kept
    assert f"{urls[0].rsplit('/', 1)[1]}.html" in kept
    assert f"{urls[1].rsplit('/', 1)[1]}.html" not in kept  # oldest pruned
    report._prune_reports(seeded_db / "does-not-exist")  # never raises


def test_report_unknown_subject_is_tool_error(seeded_db):
    fake = FakeClient([
        resp(
            [tool_use("tu_1", "create_report", {
                "title": "x",
                "subject": {"type": "conjunction", "id": 999},
                "sections": [{"heading": "h", "body": "b"}],
            })],
            stop="tool_use",
        ),
        resp([text_block("ok")]),
    ])
    result = _chat(fake)
    assert result["report_url"] is None
    block = fake.messages.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert "not found" in json.loads(block["content"])["error"]


def test_get_report_route_serves_html(seeded_db, http_client):
    conn = db.get_conn()
    try:
        url = report.create_report(conn, "Overview", {"type": "overview"}, [
            {"heading": "Summary", "body": "All quiet."},
        ])
    finally:
        conn.close()
    response = http_client.get(url)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "SYSTEM STATUS" in response.text


def test_reports_route_rejects_non_uuid_and_traversal(http_client):
    assert http_client.get("/reports/not-a-uuid").status_code == 404
    # encoded traversal decodes into the path param; uuid parse rejects it
    assert http_client.get("/reports/..%2F..%2Fsecrets").status_code == 404
    # well-formed but nonexistent uuid
    assert http_client.get(
        "/reports/00000000-0000-4000-8000-000000000000"
    ).status_code == 404


# --- HTTP endpoint -----------------------------------------------------------

def test_chat_endpoint_shape(http_client, monkeypatch):
    fake = FakeClient([resp([text_block("All nominal.")])])
    monkeypatch.setattr(service, "_default_client", lambda: fake)
    response = http_client.post(
        "/assistant/chat",
        json={"messages": [{"role": "user", "content": "status?"}],
              "context": {"page": "alerts", "url": "/alerts.html"}},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"reply", "actions", "report_url", "model"}
    assert body["reply"] == "All nominal."
    assert "page=alerts" in fake.messages.calls[0]["messages"][-1]["content"]


def test_chat_endpoint_rejects_empty_messages(http_client):
    response = http_client.post(
        "/assistant/chat", json={"messages": [], "context": {}}
    )
    assert response.status_code == 422


def _raising_client(exc_factory):
    class RaisingMessages:
        def create(self, **kwargs):
            raise exc_factory()

        def stream(self, **kwargs):
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


def test_auth_error_maps_to_503_with_contract_detail(http_client, monkeypatch):
    import anthropic

    fake = _raising_client(
        lambda: _make_api_error(anthropic.AuthenticationError, 401, "bad key")
    )
    monkeypatch.setattr(service, "_default_client", lambda: fake)
    response = http_client.post(
        "/assistant/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_rate_limit_maps_to_429(http_client, monkeypatch):
    import anthropic

    fake = _raising_client(
        lambda: _make_api_error(anthropic.RateLimitError, 429, "slow down")
    )
    monkeypatch.setattr(service, "_default_client", lambda: fake)
    response = http_client.post(
        "/assistant/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 429


def test_api_status_error_maps_to_502(seeded_db):
    import anthropic

    fake = _raising_client(
        lambda: _make_api_error(anthropic.APIStatusError, 500, "server broke")
    )
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _chat(fake)
    assert excinfo.value.status_code == 502
    # the raw upstream body (request ids, billing text) must never reach the
    # UI - only the status code and a pointer to the server log
    assert "server broke" not in excinfo.value.detail
    assert "HTTP 500" in excinfo.value.detail


def test_api_credit_exhaustion_maps_to_actionable_detail(seeded_db):
    import anthropic

    fake = _raising_client(lambda: _make_api_error(
        anthropic.APIStatusError, 400,
        "Your credit balance is too low to access the Anthropic API."))
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _chat(fake)
    assert excinfo.value.status_code == 502
    assert "credits are exhausted" in excinfo.value.detail
    assert "request_id" not in excinfo.value.detail


# --- show_encounter ----------------------------------------------------------

ENCOUNTER_ACTION = {
    "type": "show_encounter",
    "conjunction_id": 1,
    "primary_norad": 25544,
    "secondary_norad": 34454,
    "tca": "2026-07-23T04:12:09Z",
}


def test_show_encounter_queues_contract_action_and_unknown_id_errors(seeded_db):
    fake = FakeClient([
        resp(
            [
                tool_use("tu_1", "show_encounter", {"conjunction_id": 1}),
                tool_use("tu_2", "show_encounter", {"conjunction_id": 999}),
            ],
            stop="tool_use",
        ),
        resp([text_block("ok")]),
    ])
    result = _chat(fake)
    assert result["actions"] == [ENCOUNTER_ACTION]

    blocks = fake.messages.calls[1]["messages"][-1]["content"]
    # tool result grounds the model: pair norads + names + tca from the DB
    payload = json.loads(blocks[0]["content"])
    assert "is_error" not in blocks[0]
    assert payload["primary_name"] == "ISS (ZARYA)"
    assert payload["secondary_name"] == "COSMOS 2251 DEB"
    assert payload["primary_norad"] == 25544
    assert payload["secondary_norad"] == 34454
    assert payload["tca"] == "2026-07-23T04:12:09Z"
    # unknown conjunction -> is_error result, no action queued
    assert blocks[1]["is_error"] is True
    assert "not found" in json.loads(blocks[1]["content"])["error"]


# --- open_simulator ----------------------------------------------------------

def test_open_simulator_tool_schema_present(seeded_db):
    fake = FakeClient([resp([text_block("ok")])])
    _chat(fake)
    tools = fake.messages.calls[0]["tools"]
    [schema] = [t for t in tools if t["name"] == "open_simulator"]
    assert schema["input_schema"]["required"] == ["conjunction_id"]
    assert schema["input_schema"]["additionalProperties"] is False
    assert schema["input_schema"]["properties"]["conjunction_id"]["type"] == "integer"
    assert "SIMULATE" in schema["description"]


def test_open_simulator_queues_action_including_past_events(seeded_db):
    # conjunction 3 is status='past' - replaying past events is legitimate
    fake = FakeClient([
        resp(
            [
                tool_use("tu_1", "open_simulator", {"conjunction_id": 1}),
                tool_use("tu_2", "open_simulator", {"conjunction_id": 3}),
            ],
            stop="tool_use",
        ),
        resp([text_block("Simulator up.")]),
    ])
    result = _chat(fake)
    assert result["actions"] == [
        {"type": "open_simulator", "conjunction_id": 1},
        {"type": "open_simulator", "conjunction_id": 3},
    ]
    blocks = fake.messages.calls[1]["messages"][-1]["content"]
    # tool result grounds the model: pair names + norads + tca from the DB
    payload = json.loads(blocks[0]["content"])
    assert "is_error" not in blocks[0]
    assert payload["primary_name"] == "ISS (ZARYA)"
    assert payload["secondary_name"] == "COSMOS 2251 DEB"
    assert payload["primary_norad"] == 25544
    assert payload["secondary_norad"] == 34454
    assert payload["tca"] == "2026-07-23T04:12:09Z"
    past = json.loads(blocks[1]["content"])
    assert "is_error" not in blocks[1]
    assert past["status"] == "past"


def test_open_simulator_unknown_id_is_tool_error(seeded_db):
    fake = FakeClient([
        resp([tool_use("tu_1", "open_simulator", {"conjunction_id": 999})],
             stop="tool_use"),
        resp([text_block("ok")]),
    ])
    result = _chat(fake)
    assert result["actions"] == []
    block = fake.messages.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert "not found" in json.loads(block["content"])["error"]


# --- streaming: service ------------------------------------------------------

def _stream(fake, messages=None, context=None):
    return list(service.chat_stream(messages or list(USER),
                                    context or dict(CONTEXT), client=fake))


def test_stream_event_sequence_across_two_round_loop(seeded_db):
    fake = FakeClient([
        resp(
            [
                text_block("Checking."),
                tool_use("tu_1", "get_status", {}),
                tool_use("tu_2", "select_on_globe", {"norad": 25544}),
            ],
            stop="tool_use",
        ),
        resp([text_block("Focused the globe.")]),
    ])
    events = _stream(fake)
    # Full ordering: per round text deltas first, then a tool event per
    # tool_use as it executes, with the action event at execution time
    # (immediately after the tool that queued it) - not batched at the end.
    assert events == [
        {"type": "text", "text": "Checking."},
        {"type": "tool", "name": "get_status"},
        {"type": "tool", "name": "select_on_globe"},
        {"type": "action", "action": {"type": "select_on_globe", "norad": 25544}},
        {"type": "text", "text": "\n\n"},  # paragraph break after a tool round
        {"type": "text", "text": "Focused the globe."},
        {"type": "done", "report_url": None, "model": "claude-opus-5-5"},
    ]
    # loop mechanics unchanged: all tool_results in one user message
    tool_msg = fake.messages.calls[1]["messages"][-1]
    assert tool_msg["role"] == "user"
    assert [b["tool_use_id"] for b in tool_msg["content"]] == ["tu_1", "tu_2"]


def test_stream_request_shape_and_high_effort_default(seeded_db):
    fake = FakeClient([resp([text_block("All nominal.")])])
    events = _stream(fake)
    assert events[-1] == {"type": "done", "report_url": None,
                          "model": "claude-opus-5-5"}
    # the request dict is unchanged apart from being passed to stream()
    call = fake.messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == 16000
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in call
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert len(call["tools"]) == 27  # + plan_avoidance; + get_space_weather
    assert call["messages"][-1]["content"].startswith("<context>")


def test_effort_env_var_still_overrides(seeded_db, monkeypatch):
    monkeypatch.setenv("KESSLER_AI_EFFORT", "low")
    fake = FakeClient([resp([text_block("ok")]),
                       resp([text_block("ok")])])
    _chat(fake)
    assert fake.messages.calls[0]["output_config"] == {"effort": "low"}
    assert _stream(fake)[-1]["type"] == "done"
    assert fake.messages.calls[1]["output_config"] == {"effort": "low"}


def test_stream_show_encounter_action_emitted_at_execution_time(seeded_db):
    fake = FakeClient([
        resp([tool_use("tu_1", "show_encounter", {"conjunction_id": 1})],
             stop="tool_use"),
        resp([text_block("Watching it.")]),
    ])
    events = _stream(fake)
    assert events == [
        {"type": "tool", "name": "show_encounter"},
        {"type": "action", "action": ENCOUNTER_ACTION},
        {"type": "text", "text": "Watching it."},
        {"type": "done", "report_url": None, "model": "claude-opus-5-5"},
    ]


def test_stream_open_simulator_action_emitted_at_execution_time(seeded_db):
    fake = FakeClient([
        resp([tool_use("tu_1", "open_simulator", {"conjunction_id": 1})],
             stop="tool_use"),
        resp([text_block("Simulator open.")]),
    ])
    events = _stream(fake)
    assert events == [
        {"type": "tool", "name": "open_simulator"},
        {"type": "action",
         "action": {"type": "open_simulator", "conjunction_id": 1}},
        {"type": "text", "text": "Simulator open."},
        {"type": "done", "report_url": None, "model": "claude-opus-5-5"},
    ]


def test_stream_error_mid_stream_one_event_then_close(seeded_db):
    import anthropic

    fake = FakeClient([stream_error(
        _make_api_error(anthropic.APIStatusError, 500, "server broke"),
        deltas=["Partial "],
    )])
    events = _stream(fake)
    # partial text kept, one typed error event, clean close - no done event
    assert events == [
        {"type": "text", "text": "Partial "},
        {"type": "error",
         "detail": "Claude API error (HTTP 500) — details in the server log."},
    ]


def test_stream_error_mapping_matches_nonstreaming_semantics(seeded_db):
    import anthropic
    import httpx

    cases = [
        (lambda: _make_api_error(anthropic.AuthenticationError, 401, "bad key"),
         service.AUTH_DETAIL),
        (lambda: _make_api_error(anthropic.RateLimitError, 429, "slow down"),
         "Claude API rate limited — retry shortly."),
        (lambda: anthropic.APIConnectionError(
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")),
         "Cannot reach the Claude API (network error)."),
    ]
    for factory, detail in cases:
        fake = _raising_client(factory)
        events = list(service.chat_stream(list(USER), dict(CONTEXT), client=fake))
        assert events == [{"type": "error", "detail": detail}]


def test_stream_refusal(seeded_db):
    fake = FakeClient([resp([], stop="refusal")])
    assert _stream(fake) == [
        {"type": "text", "text": service.REFUSAL_REPLY},
        {"type": "done", "report_url": None, "model": "claude-opus-5-5"},
    ]


def test_stream_pause_turn_resends_with_assistant_content(seeded_db):
    paused = resp([text_block("working...")], stop="pause_turn")
    fake = FakeClient([paused, resp([text_block("final answer")])])
    events = _stream(fake)
    assert events == [
        {"type": "text", "text": "working..."},
        {"type": "text", "text": "final answer"},
        {"type": "done", "report_url": None, "model": "claude-opus-5-5"},
    ]
    assert fake.messages.calls[1]["messages"][-1]["content"] is paused.content


def test_stream_iteration_cap(seeded_db):
    fake = FakeClient(
        [resp([tool_use("tu_1", "get_status", {})], stop="tool_use")],
        cycle=True,
    )
    events = _stream(fake)
    assert len(fake.messages.calls) == 12
    assert events[-2]["type"] == "text" and "cap" in events[-2]["text"]
    assert events[-1] == {"type": "done", "report_url": None,
                          "model": "claude-opus-5-5"}


def test_stream_done_carries_report_url(seeded_db):
    fake = FakeClient([
        resp([tool_use("tu_1", "create_report", {
            "title": "Overview brief",
            "subject": {"type": "overview"},
            "sections": [{"heading": "Summary", "body": "All quiet."}],
        })], stop="tool_use"),
        resp([text_block("Report ready.")]),
    ])
    events = _stream(fake)
    done = events[-1]
    assert done["type"] == "done"
    assert done["report_url"].startswith("/reports/")
    # create_report is not a UI action - no action event
    assert [e for e in events if e["type"] == "action"] == []


def test_stream_rejects_empty_messages_before_first_byte(seeded_db):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        service.chat_stream([], {}, client=FakeClient([]))
    assert excinfo.value.status_code == 400


# --- streaming: endpoint -----------------------------------------------------

def _sse_events(body: str) -> list[dict]:
    frames = [chunk for chunk in body.split("\n\n") if chunk]
    assert all(frame.startswith("data: ") for frame in frames)
    return [json.loads(frame[len("data: "):]) for frame in frames]


def test_stream_endpoint_sse_framing(http_client, monkeypatch):
    fake = FakeClient([
        resp([tool_use("tu_1", "set_globe_speed", {"mult": 60})],
             stop="tool_use"),
        resp([text_block("Sped up.")]),
    ])
    monkeypatch.setattr(service, "_default_client", lambda: fake)
    response = http_client.post(
        "/assistant/chat/stream",
        json={"messages": [{"role": "user", "content": "faster"}],
              "context": {"page": "dashboard", "url": "/"}},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _sse_events(response.text) == [
        {"type": "tool", "name": "set_globe_speed"},
        {"type": "action", "action": {"type": "set_globe_speed", "mult": 60}},
        {"type": "text", "text": "Sped up."},
        {"type": "done", "report_url": None, "model": "claude-opus-5-5"},
    ]


def test_stream_endpoint_error_event_in_body(http_client, monkeypatch):
    import anthropic

    fake = FakeClient([stream_error(
        _make_api_error(anthropic.APIStatusError, 500, "server broke"))])
    monkeypatch.setattr(service, "_default_client", lambda: fake)
    response = http_client.post(
        "/assistant/chat/stream",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200  # stream already open - error is an event
    assert _sse_events(response.text) == [
        {"type": "error",
         "detail": "Claude API error (HTTP 500) — details in the server log."},
    ]


def test_stream_endpoint_auth_failure_before_stream_is_503(http_client,
                                                           monkeypatch):
    from fastapi import HTTPException

    def no_credentials():
        raise HTTPException(status_code=503, detail=service.AUTH_DETAIL)

    monkeypatch.setattr(service, "_default_client", no_credentials)
    response = http_client.post(
        "/assistant/chat/stream",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_stream_endpoint_rejects_empty_messages(http_client):
    response = http_client.post(
        "/assistant/chat/stream", json={"messages": [], "context": {}}
    )
    assert response.status_code == 422


# --- optimize_maneuver + get_anomalies tools -----------------------------------

def test_v7_tool_schemas_and_prompt_mentions(seeded_db):
    from backend.app.assistant import prompt as prompt_mod
    from backend.app.assistant import tools as tools_mod

    names = [t["name"] for t in tools_mod.TOOLS]
    assert len(names) == 27 and len(set(names)) == 27
    assert set(tools_mod._HANDLERS) == set(names)  # every tool executes

    [opt] = [t for t in tools_mod.TOOLS if t["name"] == "optimize_maneuver"]
    assert opt["input_schema"]["required"] == ["conjunction_id"]
    assert opt["input_schema"]["additionalProperties"] is False
    assert set(opt["input_schema"]["properties"]) == {
        "conjunction_id", "target_miss_m", "burn_iso"}
    assert "TWO-BODY" in opt["description"]

    [anom] = [t for t in tools_mod.TOOLS if t["name"] == "get_anomalies"]
    assert anom["input_schema"]["required"] == []
    assert anom["input_schema"]["additionalProperties"] is False
    assert set(anom["input_schema"]["properties"]) == {"hours"}
    assert "HEURISTIC" in anom["description"]

    # prompt.py mandates the two-body caveat for optimizer reporting
    text = prompt_mod.SYSTEM_PROMPT
    assert "optimize_maneuver" in text and "get_anomalies" in text
    assert "TWO-BODY" in text and "reachable: false" in text


def test_optimize_maneuver_defaults_and_passthrough(seeded_db, monkeypatch):
    from backend.app.models import maneuver as maneuver_mod

    seen = {}

    def fake_optimize_dv(conn, conjunction_id, target_miss_m, burn_iso):
        seen.update(conjunction_id=conjunction_id,
                    target_miss_m=target_miss_m, burn_iso=burn_iso)
        return {"dv_ms": 0.42, "achieved_miss_m": 1010.0,
                "burn_iso": burn_iso, "evaluations": 17,
                "method": "golden-section two-body sandbox"}

    monkeypatch.setattr(maneuver_mod, "optimize_dv", fake_optimize_dv,
                        raising=False)
    fake = FakeClient([
        resp([tool_use("tu_1", "optimize_maneuver", {"conjunction_id": 1})],
             stop="tool_use"),
        resp([text_block("ok")]),
    ])
    _chat(fake)
    # contract defaults: target 1000 m, burn at stored TCA - 6 h
    assert seen == {"conjunction_id": 1, "target_miss_m": 1000.0,
                    "burn_iso": "2026-07-22T22:12:09Z"}
    block = fake.messages.calls[1]["messages"][-1]["content"][0]
    assert "is_error" not in block
    payload = json.loads(block["content"])
    assert payload["dv_ms"] == 0.42
    assert payload["achieved_miss_m"] == 1010.0
    assert payload["evaluations"] == 17
    assert payload["reachable"] is True
    assert "two-body" in payload["approximation"]


def test_optimize_maneuver_explicit_args_and_unreachable(seeded_db,
                                                         monkeypatch):
    from backend.app.models import maneuver as maneuver_mod

    calls = []

    def fake_optimize_dv(conn, conjunction_id, target_miss_m, burn_iso):
        calls.append((conjunction_id, target_miss_m, burn_iso))
        # optimize_dv contract: unreachable is a NORMAL result carrying
        # reachable: false plus the best found within +/-5 m/s
        return {"dv_ms": -5.0, "achieved_miss_m": 3200.0,
                "burn_iso": burn_iso, "evaluations": 24,
                "reachable": False, "method": "golden-section"}

    monkeypatch.setattr(maneuver_mod, "optimize_dv", fake_optimize_dv,
                        raising=False)
    fake = FakeClient([
        resp([tool_use("tu_1", "optimize_maneuver", {
            "conjunction_id": 1,
            "target_miss_m": 50000,
            "burn_iso": "2026-07-23T01:00:00Z",
        })], stop="tool_use"),
        resp([text_block("ok")]),
    ])
    _chat(fake)
    assert calls == [(1, 50000.0, "2026-07-23T01:00:00Z")]
    block = fake.messages.calls[1]["messages"][-1]["content"][0]
    # unreachable is a legitimate ANSWER, not a tool failure
    assert "is_error" not in block
    payload = json.loads(block["content"])
    assert payload["reachable"] is False
    assert payload["achieved_miss_m"] == 3200.0
    assert "two-body" in payload["approximation"]


def test_optimize_maneuver_value_error_is_tool_error(seeded_db, monkeypatch):
    from backend.app.models import maneuver as maneuver_mod

    def fake_optimize_dv(conn, conjunction_id, target_miss_m, burn_iso):
        raise ValueError("burn must not be more than 1 hour in the past")

    monkeypatch.setattr(maneuver_mod, "optimize_dv", fake_optimize_dv,
                        raising=False)
    fake = FakeClient([
        resp([tool_use("tu_1", "optimize_maneuver", {
            "conjunction_id": 1, "burn_iso": "2026-07-23T01:00:00Z",
        })], stop="tool_use"),
        resp([text_block("ok")]),
    ])
    _chat(fake)
    block = fake.messages.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    error = json.loads(block["content"])["error"]
    assert "rejected" in error and "1 hour in the past" in error


def test_optimize_maneuver_validation_and_unavailable(seeded_db, monkeypatch):
    from backend.app.models import maneuver as maneuver_mod

    for name in ("optimize_dv", "optimize_maneuver", "optimize"):
        monkeypatch.delattr(maneuver_mod, name, raising=False)
    fake = FakeClient([
        resp(
            [
                tool_use("tu_1", "optimize_maneuver", {"conjunction_id": 999}),
                tool_use("tu_2", "optimize_maneuver",
                         {"conjunction_id": 1, "target_miss_m": -3}),
                tool_use("tu_3", "optimize_maneuver", {"conjunction_id": 1}),
            ],
            stop="tool_use",
        ),
        resp([text_block("ok")]),
    ])
    _chat(fake)
    blocks = fake.messages.calls[1]["messages"][-1]["content"]
    assert blocks[0]["is_error"] is True
    assert "not found" in json.loads(blocks[0]["content"])["error"]
    assert blocks[1]["is_error"] is True
    assert "positive" in json.loads(blocks[1]["content"])["error"]
    assert blocks[2]["is_error"] is True
    assert "unavailable" in json.loads(blocks[2]["content"])["error"]


def test_get_anomalies_defaults_clamp_and_wrap(seeded_db, monkeypatch):
    anomaly = types.ModuleType("backend.app.models.anomaly")
    windows = []

    def scan_tle_deltas(conn, since_hours=48):
        windows.append(since_hours)
        return [{"norad_id": 25544, "name": "ISS (ZARYA)",
                 "kind": "sma_raise", "delta_sma_km": 1.2}]

    anomaly.scan_tle_deltas = scan_tle_deltas
    monkeypatch.setitem(sys.modules, "backend.app.models.anomaly", anomaly)

    fake = FakeClient([
        resp(
            [
                tool_use("tu_1", "get_anomalies", {}),
                tool_use("tu_2", "get_anomalies", {"hours": 9999}),
            ],
            stop="tool_use",
        ),
        resp([text_block("ok")]),
    ])
    _chat(fake)
    assert windows == [48, 168]  # default 48, clamp cap 168
    blocks = fake.messages.calls[1]["messages"][-1]["content"]
    payload = json.loads(blocks[0]["content"])
    assert "is_error" not in blocks[0]
    assert payload["hours"] == 48
    assert payload["count"] == 1
    assert payload["items"][0]["kind"] == "sma_raise"
    assert "not confirmed maneuvers" in payload["note"]


def test_get_anomalies_unavailable_is_clear_tool_error(seeded_db, monkeypatch):
    # neither the models module nor the api module resolves -> ToolError
    monkeypatch.setitem(sys.modules, "backend.app.models.anomaly", None)
    monkeypatch.setitem(sys.modules, "backend.app.api.anomalies", None)
    fake = FakeClient([
        resp([tool_use("tu_1", "get_anomalies", {})], stop="tool_use"),
        resp([text_block("ok")]),
    ])
    _chat(fake)
    block = fake.messages.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert "unavailable" in json.loads(block["content"])["error"]


# --- narration ----------------------------------------------------------------

from backend.app.api import assistant as assistant_api  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_narrate_cache():
    assistant_api._narrate_cache.clear()
    yield
    assistant_api._narrate_cache.clear()


NARRATION_BEATS = [
    {"t_rel_s": -1200, "text": "Acquisition. Range closing."},
    {"t_rel_s": 0, "text": "TCA. Minimum separation."},
    {"t_rel_s": 60, "text": "Separation opening."},
]


def _narration_client(payload=None):
    """FakeClient whose single scripted turn is a structured-outputs style
    response: first content block is text carrying JSON."""
    text = payload if isinstance(payload, str) else json.dumps(
        {"beats": NARRATION_BEATS if payload is None else payload}
    )
    return FakeClient([resp([thinking_block(), text_block(text)])])


def test_narrate_parses_beats_and_single_structured_call(seeded_db):
    fake = _narration_client()
    result = assistant_api.narrate(1, client=fake)

    assert result["model"] == "claude-opus-5-5"
    sig = assistant_api.tts_sign
    assert result["beats"] == [
        {"t_rel_s": -1200.0, "text": "Acquisition. Range closing.",
         "sig": sig("Acquisition. Range closing.")},
        {"t_rel_s": 0.0, "text": "TCA. Minimum separation.",
         "sig": sig("TCA. Minimum separation.")},
        {"t_rel_s": 60.0, "text": "Separation opening.",
         "sig": sig("Separation opening.")},
    ]
    # one messages.create call, no tool loop
    assert len(fake.messages.calls) == 1
    call = fake.messages.calls[0]
    assert "tools" not in call
    assert call["max_tokens"] == 2000
    assert call["output_config"]["effort"] == "medium"
    fmt = call["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["required"] == ["beats"]
    assert fmt["schema"]["additionalProperties"] is False
    beat_schema = fmt["schema"]["properties"]["beats"]["items"]
    assert beat_schema["required"] == ["t_rel_s", "text"]
    assert beat_schema["additionalProperties"] is False
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in call
    # its own small frozen system prompt - not the chat prompt
    assert call["system"][0]["text"] is service.prompt.NARRATION_SYSTEM_PROMPT
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "KESSLER AI, the operations assistant" not in call["system"][0]["text"]
    # event summary + ephemeris tca_state travel in the user message
    [message] = call["messages"]
    assert message["role"] == "user"
    assert "ISS (ZARYA)" in message["content"]
    assert '"tca_state"' in message["content"]
    assert '"window_s": 1200' in message["content"]


def test_narrate_effort_env_override(seeded_db, monkeypatch):
    monkeypatch.setenv("KESSLER_AI_EFFORT", "low")
    fake = _narration_client()
    assistant_api.narrate(1, client=fake)
    assert fake.messages.calls[0]["output_config"]["effort"] == "low"


def test_narrate_cache_hit_skips_client(seeded_db):
    fake = _narration_client()
    first = assistant_api.narrate(1, client=fake)
    assert len(fake.messages.calls) == 1

    # second call: served from the 10-minute cache - client never touched
    untouchable = FakeClient([])  # any call would IndexError
    second = assistant_api.narrate(1, client=untouchable)
    assert second == first
    assert len(fake.messages.calls) == 1

    # expiry: an aged-out entry recomputes
    expires, payload = assistant_api._narrate_cache[1]
    assistant_api._narrate_cache[1] = (0.0, payload)
    third = assistant_api.narrate(1, client=_narration_client())
    assert third == first


def test_narrate_unknown_conjunction_is_404_and_uncached(seeded_db):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        assistant_api.narrate(999, client=FakeClient([]))
    assert excinfo.value.status_code == 404
    assert "not found" in excinfo.value.detail
    assert 999 not in assistant_api._narrate_cache


def test_narrate_malformed_json_degrades_to_502_and_uncached(seeded_db):
    from fastapi import HTTPException

    cases = [
        "this is not json {",                      # unparseable
        json.dumps({"lines": []}),                 # missing beats array
        json.dumps({"beats": []}),                 # empty beats
        json.dumps({"beats": [{"t_rel_s": "soon", "text": "x"}]}),  # bad shape
        json.dumps({"beats": [{"t_rel_s": 0}]}),   # missing text
    ]
    for payload in cases:
        with pytest.raises(HTTPException) as excinfo:
            assistant_api.narrate(1, client=_narration_client(payload))
        assert excinfo.value.status_code == 502
        assert "malformed narration JSON" in excinfo.value.detail
        assert 1 not in assistant_api._narrate_cache  # failures never cached


def test_narrate_error_mapping_identical_to_chat(seeded_db):
    import anthropic
    import httpx
    from fastapi import HTTPException

    cases = [
        (lambda: _make_api_error(anthropic.AuthenticationError, 401, "bad key"),
         503, service.AUTH_DETAIL),
        (lambda: _make_api_error(anthropic.RateLimitError, 429, "slow down"),
         429, "rate limited"),
        (lambda: _make_api_error(anthropic.APIStatusError, 500, "server broke"),
         502, "HTTP 500"),
        (lambda: anthropic.APIConnectionError(
            request=httpx.Request("POST",
                                  "https://api.anthropic.com/v1/messages")),
         502, "network error"),
    ]
    for factory, status_code, detail_fragment in cases:
        with pytest.raises(HTTPException) as excinfo:
            assistant_api.narrate(1, client=_raising_client(factory))
        assert excinfo.value.status_code == status_code
        assert detail_fragment in excinfo.value.detail


def test_narrate_refusal_is_502(seeded_db):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        assistant_api.narrate(1, client=FakeClient([resp([], stop="refusal")]))
    assert excinfo.value.status_code == 502
    assert "declined" in excinfo.value.detail


def test_narrate_endpoint_shape_and_404(http_client, monkeypatch):
    fake = _narration_client()
    monkeypatch.setattr(service, "_default_client", lambda: fake)

    response = http_client.post("/assistant/narrate",
                                json={"conjunction_id": 1})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"beats", "model"}
    assert body["model"] == "claude-opus-5-5"
    assert body["beats"][0] == {
        "t_rel_s": -1200.0,
        "text": "Acquisition. Range closing.",
        "sig": assistant_api.tts_sign("Acquisition. Range closing."),
    }
    assert len(fake.messages.calls) == 1

    # cache hit: no second API call
    again = http_client.post("/assistant/narrate", json={"conjunction_id": 1})
    assert again.status_code == 200 and again.json() == body
    assert len(fake.messages.calls) == 1

    missing = http_client.post("/assistant/narrate",
                               json={"conjunction_id": 999})
    assert missing.status_code == 404

    bad = http_client.post("/assistant/narrate", json={})
    assert bad.status_code == 422


def test_tools_catalog_endpoint_matches_tools(seeded_db):
    from backend.app.assistant import tools as tools_mod
    from backend.app.main import app

    with TestClient(app) as client:
        body = client.get("/assistant/tools").json()
    assert body["count"] == len(tools_mod.TOOLS)
    assert [i["name"] for i in body["items"]] == [t["name"] for t in tools_mod.TOOLS]
    fams = {i["name"]: i["family"] for i in body["items"]}
    assert fams["open_simulator"] == "ui" and fams["plan_avoidance"] == "data"
    assert fams["create_report"] == "report"
    assert tools_mod.UI_ACTION_TOOL_NAMES <= set(fams)


# --- get_space_weather ------------------------------------------------------------

def test_get_space_weather_tool_schema_and_unavailable_error(seeded_db, monkeypatch):
    from backend.app import spaceweather as sw
    from backend.app.assistant import prompt as prompt_mod
    from backend.app.assistant import tools as tools_mod
    [t] = [t for t in tools_mod.TOOLS if t["name"] == "get_space_weather"]
    assert t["input_schema"]["required"] == [] and "norad" in t["input_schema"]["properties"]
    assert "get_space_weather" in prompt_mod.SYSTEM_PROMPT
    sw._cache.update(t=0.0, data=None)                # no live data (fetch is off in tests)
    text, is_error = tools_mod.ToolRuntime(db.get_conn()).execute("get_space_weather", {})
    assert is_error and "unavailable" in text.lower()
