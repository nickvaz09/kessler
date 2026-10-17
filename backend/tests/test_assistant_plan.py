"""Kessler AI - plan_avoidance tool and open_simulator apply_plan.

FakeClient pattern from test_assistant (no network, no API key). The
end-to-end check runs the real planner through ToolRuntime on the
test_avoidance_plan geometry.
"""
import json
from datetime import datetime, timedelta, timezone

from backend.app import db
from backend.app.assistant import prompt as prompt_mod
from backend.app.assistant import tools as tools_mod
from backend.tests.test_assistant import (  # noqa: F401  (fixtures)
    FakeClient, _chat, resp, seeded_db, text_block, tool_use)
from backend.tests import test_avoidance_plan as plan_fx
from backend.tests.test_avoidance_plan import tca  # noqa: F401  (fixture)

FAKE_PLAN = {
    "conjunction_id": 1,
    "tca": "2026-07-23T04:12:09Z",
    "computed_at": "2026-07-21T13:10:00Z",
    "status": "burn_recommended",
    "note": None,
    "baseline": {"miss_m": 812.4, "pc": 3e-5, "tca": "2026-07-23T04:12:09Z",
                 "safe": False, "screened_miss_m": 812.4},
    "criteria": {"target_miss_m": 1000.0, "pc_threshold": 1e-6,
                 "pc_required_miss_m": 1700.0, "required_miss_m": 1700.0,
                 "max_dv_ms": 5.0, "rule": "long prose"},
    "recommended": {"lead_label": "6 h", "lead_s": 21600.0,
                    "burn_iso": "2026-07-22T22:12:09Z", "dv_ms": 0.18,
                    "direction": "prograde", "new_miss_m": 1720.0,
                    "delta_miss_m": 900.0, "new_tca": "2026-07-23T04:12:09Z",
                    "new_pc": 8e-7, "reachable": True, "meets_miss": True,
                    "meets_pc": True, "propellant_g_per_100kg": 8.3,
                    "evaluations": 24},
    "options": [],
    "skipped": [],
    "assumptions": {},
    "method": "two-body differential, along-track, screening-grade; "
              "not an operational maneuver plan",
    "method_detail": "x",
}
FAKE_PLAN["options"] = [FAKE_PLAN["recommended"]]


def test_plan_avoidance_schema_and_prompt():
    [schema] = [t for t in tools_mod.TOOLS if t["name"] == "plan_avoidance"]
    assert schema["input_schema"]["required"] == ["conjunction_id"]
    assert schema["input_schema"]["additionalProperties"] is False
    assert set(schema["input_schema"]["properties"]) == {
        "conjunction_id", "target_miss_m"}
    assert "TWO-BODY" in schema["description"]
    assert "geometry_mismatch" in schema["description"]
    [sim] = [t for t in tools_mod.TOOLS if t["name"] == "open_simulator"]
    assert sim["input_schema"]["properties"]["apply_plan"]["type"] == "boolean"
    assert sim["input_schema"]["required"] == ["conjunction_id"]
    text = prompt_mod.SYSTEM_PROMPT
    assert "plan_avoidance" in text and "apply_plan" in text
    assert "NEVER call that an all-clear" in text


def test_plan_avoidance_tool_passthrough_and_trim(seeded_db, monkeypatch):
    from backend.app.models import maneuver as maneuver_mod

    seen = {}

    def fake_plan(conn, conjunction_id, target_miss_m=None, max_dv_ms=None):
        seen.update(cid=conjunction_id, target=target_miss_m)
        return json.loads(json.dumps(FAKE_PLAN))

    monkeypatch.setattr(maneuver_mod, "plan_avoidance", fake_plan)
    fake = FakeClient([
        resp([tool_use("tu_1", "plan_avoidance",
                       {"conjunction_id": 1, "target_miss_m": 2000})],
             stop="tool_use"),
        resp([text_block("Fire 0.18 m/s prograde.")]),
    ])
    result = _chat(fake)
    assert seen == {"cid": 1, "target": 2000.0}
    assert result["actions"] == []   # a data tool queues no UI action
    block = fake.messages.calls[1]["messages"][-1]["content"][0]
    assert "is_error" not in block
    payload = json.loads(block["content"])
    assert payload["status"] == "burn_recommended"
    assert payload["recommended"]["dv_ms"] == 0.18
    assert payload["method"].startswith("two-body differential")
    assert "rule" not in payload["criteria"]
    assert "recommendation_rule" in payload and "fuel_context" in payload
    assert set(payload["options"][0]) == {
        "lead_label", "burn_iso", "dv_ms", "direction", "new_miss_m",
        "new_pc", "reachable", "propellant_g_per_100kg"}


def test_plan_avoidance_tool_errors(seeded_db):
    # seeded TCAs (July 2026) are in the past relative to the real clock:
    # the planner's PlanConflict surfaces as an is_error tool result.
    fake = FakeClient([
        resp([tool_use("tu_1", "plan_avoidance", {"conjunction_id": 1}),
              tool_use("tu_2", "plan_avoidance", {"conjunction_id": 999}),
              tool_use("tu_3", "plan_avoidance",
                       {"conjunction_id": 1, "target_miss_m": "far"})],
             stop="tool_use"),
        resp([text_block("ok")]),
    ])
    _chat(fake)
    blocks = fake.messages.calls[1]["messages"][-1]["content"]
    assert all(b["is_error"] is True for b in blocks)
    errs = [json.loads(b["content"])["error"] for b in blocks]
    assert "passed" in errs[0]
    assert "not found" in errs[1]
    assert "number" in errs[2]


def test_open_simulator_apply_plan(seeded_db):
    conn = db.get_conn()
    future = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT INTO conjunctions (id, primary_object_id, secondary_object_id,"
        " time_of_closest_approach, first_seen_at, status)"
        " VALUES (9, 1, 2, ?, '2026-07-20T00:00:00Z', 'active')", (future,))
    conn.commit()
    conn.close()
    fake = FakeClient([
        resp([tool_use("tu_1", "open_simulator",
                       {"conjunction_id": 9, "apply_plan": True}),
              tool_use("tu_2", "open_simulator",
                       {"conjunction_id": 3, "apply_plan": True}),
              tool_use("tu_3", "open_simulator",
                       {"conjunction_id": 9, "apply_plan": "yes"})],
             stop="tool_use"),
        resp([text_block("Plan loaded.")]),
    ])
    result = _chat(fake)
    assert result["actions"] == [
        {"type": "open_simulator", "conjunction_id": 9, "apply_plan": True}]
    blocks = fake.messages.calls[1]["messages"][-1]["content"]
    assert json.loads(blocks[0]["content"])["apply_plan"] is True
    assert blocks[1]["is_error"] is True   # past event: no plan to apply
    assert "passed" in json.loads(blocks[1]["content"])["error"]
    assert blocks[2]["is_error"] is True


def test_plan_avoidance_tool_runs_real_planner(tmp_path, tca):  # noqa: F811
    conn = db.get_conn(tmp_path / "tool.db")
    db.init_db(conn)
    plan_fx._seed(conn, tca)
    try:
        rt = tools_mod.ToolRuntime(conn)
        out, is_error = rt.execute("plan_avoidance", {"conjunction_id": 1})
        assert is_error is False
        payload = json.loads(out)
        assert payload["status"] == "burn_recommended"
        rec = payload["recommended"]
        assert rec["new_miss_m"] >= payload["criteria"]["required_miss_m"] - 0.1
        assert rec["new_pc"] < payload["criteria"]["pc_threshold"]
        assert [o["lead_label"] for o in payload["options"]] == [
            "1 orbit", "3 h", "6 h"]
        assert rt.actions == []
        out, is_error = rt.execute("plan_avoidance", {"conjunction_id": 2})
        assert is_error is True and "passed" in json.loads(out)["error"]
    finally:
        conn.close()
