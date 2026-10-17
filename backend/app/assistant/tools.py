"""Kessler AI tool surface: hand-written Claude tool schemas + executor.

DATA tools reuse the SQL/shape helpers from the REST layer (api/conjunctions,
api/objects, api/risk, api/status) so the assistant always sees the same
numbers as the UI. UI-ACTION tools validate their input, append one action to
the ordered per-request queue, and return {"ok": true}. create_report renders
via assistant/report.py.

TOOLS is a DETERMINISTIC, ordered list - it participates in the prompt-cache
prefix and must not be reordered or mutated per request. Every schema sets
additionalProperties: false and an explicit required list, and every
description states when the tool should be called (this measurably improves
triggering).

data tools (get_changes, get_passes, simulate_maneuver, list_user_objects,
screen_user_object) call backend logic built in parallel from the same
contract (api/changes, propagate.passes, the conjunction maneuver endpoint,
api/user_objects). They resolve those callables lazily at execution time
(importlib + tolerant signature binding, mirroring the lazy-import pattern
used across backend/app/api) so this module always imports cleanly and a
not-yet-deployed dependency surfaces as a clear ToolError, never a crash.
"""
import importlib
import inspect
import json
import types
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from backend.app.api import conjunctions as conjunctions_api
from backend.app.api import objects as objects_api
from backend.app.api import risk as risk_api
from backend.app.api import status as status_api
from backend.app.assistant import report as report_mod


class ToolError(Exception):
    """Tool-level failure surfaced to the model as an is_error tool_result."""


GLOBE_TYPES = ("PAYLOAD", "DEBRIS", "RB", "UNKNOWN")
RISK_LEVELS = ("critical", "elevated", "nominal")
SPEED_MULTS = (1, 60, 600)

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
PASSES_DEFAULT_HOURS = 48        # GET /objects/{norad}/passes default
PASSES_MAX_HOURS = 168
SCREEN_DEFAULT_WINDOW_HOURS = 72  # POST /user-objects/{id}/screen default
SCREEN_MAX_WINDOW_HOURS = 168
DV_ALONG_MAX_MS = 5.0            # contract: dv_along_ms in -5..5
DEFAULT_BURN_LEAD_H = 6          # contract: default burn at TCA - 6 h
OPTIMIZE_DEFAULT_TARGET_MISS_M = 1000.0  # optimize_maneuver default target
ANOMALY_DEFAULT_HOURS = 48       # GET /anomalies?hours default
ANOMALY_MAX_HOURS = 168

NO_SITE_GUIDANCE = (
    "No ground site is set. Ask the user to set one first: open any object "
    "page and enter the site latitude/longitude in the PASSES panel — the "
    "saved site is sent along with every assistant request automatically."
)

MANEUVER_METHOD_NOTE = (
    "two-body post-burn approximation: SGP4 state at the burn, delta-v "
    "applied along the velocity vector, Kepler propagation afterwards "
    "(drag/J2 ignored) — an hours-scale estimate, not an operational product"
)

ANOMALY_HEURISTIC_NOTE = (
    "element-delta heuristic over public TLEs (|dSMA| >= 0.5 km or "
    "dInc >= 0.01 deg between successive TLEs; drag-consistent SMA decay "
    "excluded) — maneuver CANDIDATES, not confirmed maneuvers"
)

TOOLS = [
    # --- DATA tools ----------------------------------------------------------
    {
        "name": "get_status",
        "description": (
            "Fetch live platform status: catalog size, active conjunction "
            "count, elevated/critical counts, last ingest time, and the "
            "active risk-model version. Call this whenever the user asks how "
            "things look overall, how many objects are tracked, whether data "
            "is fresh, or which model is running."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_conjunctions",
        "description": (
            "List conjunction events sorted by effective risk (desc), then "
            "TCA. Call this when the user asks about upcoming or close "
            "approaches, the riskiest events, alerts, or anything plural "
            "about conjunctions. Prefer status/risk/search filters over "
            "fetching everything."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "past", "all"],
                    "description": "Event status filter; default 'active' "
                                   "(upcoming, inside the screening window).",
                },
                "risk": {
                    "type": "string",
                    "enum": ["critical", "elevated", "nominal"],
                    "description": "Only events at this risk level.",
                },
                "search": {
                    "type": "string",
                    "description": "Case-insensitive match against either "
                                   "object's name or exact NORAD id.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Max items to return (default 20, cap 50).",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_conjunction",
        "description": (
            "Fetch one conjunction event in full: summary plus complete CDM "
            "history and risk-prediction history. Call this when the user "
            "asks about a specific conjunction by id, or after "
            "list_conjunctions when per-event detail is needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Conjunction id."},
            },
            "required": ["id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_object",
        "description": (
            "Fetch one catalog object by NORAD id: catalog record, latest "
            "orbital elements, TLE count, and its conjunctions. Call this "
            "when the user asks about a specific satellite or debris object. "
            "If you only have a name, call search_objects first to resolve "
            "the NORAD id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "norad": {"type": "integer", "description": "NORAD catalog id."},
            },
            "required": ["norad"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_objects",
        "description": (
            "Search the catalog by name substring (case-insensitive) or "
            "exact NORAD id. Call this to resolve a name the user gave into "
            "NORAD ids before object-specific tools or UI actions, or to "
            "check whether an object is tracked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Name fragment, or digits for an exact "
                                   "NORAD id match.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Max items to return (default 10, cap 20).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_risk",
        "description": (
            "Fetch the latest risk prediction for one conjunction plus the "
            "model input features behind it (miss distance, time to TCA, "
            "CDM deltas, slope). Call this when the user asks WHY a risk "
            "level is what it is, about the predicted risk, or about model "
            "features. Cite the returned model_version in your reply."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "conjunction_id": {"type": "integer",
                                   "description": "Conjunction id."},
            },
            "required": ["conjunction_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "explain_risk",
        "description": (
            "Glass-box explanation of one conjunction's predicted risk: "
            "per-feature SHAP contributions of the trained model, or the "
            "carry-forward note when the hybrid defers to the observed Pc. "
            "Call this whenever the user asks WHY a risk number or "
            "prediction is what it is, what drives it, or which features "
            "matter. Report the top contributions with their sign and "
            "always name the returned model_version."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "conjunction_id": {"type": "integer",
                                   "description": "Conjunction id."},
            },
            "required": ["conjunction_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_pipeline_refresh",
        "description": (
            "Re-run the full ingestion pipeline: CelesTrak fetch, SGP4 "
            "conjunction screening, and risk predictions. BLOCKS FOR ROUGHLY "
            "TWO MINUTES. Call this ONLY when the user explicitly asked for "
            "a data refresh / re-ingest — never proactively, and never just "
            "because data looks stale."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "demo": {
                    "type": "boolean",
                    "description": "Also backfill synthetic demo CDM "
                                   "sequences (default false).",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_changes",
        "description": (
            "Diff the two most recent ingestion runs: new conjunctions, "
            "risk-level changes (with Pc before/after), and resolved events. "
            "Call this when the user asks what changed, what is new since "
            "the last refresh, or for a since-last-run delta. Returns empty "
            "sections when fewer than two completed runs exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_anomalies",
        "description": (
            "Scan recent TLE history for maneuver-like element changes: "
            "satellites whose semi-major axis jumped at least 0.5 km or "
            "whose inclination changed at least 0.01 deg between successive "
            "TLEs (SMA drops consistent with the TLE's own drag term are "
            "excluded). Call this when the user asks who maneuvered, about "
            "orbit raises or lowerings, station-keeping activity, or unusual "
            "element changes in the catalog. Results are an element-delta "
            "HEURISTIC on public TLEs, not confirmed maneuvers — say so "
            "when reporting them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 168,
                    "description": "Look-back window in hours "
                                   "(default 48, cap 168).",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_passes",
        "description": (
            "Predict ground-station passes of one object over the user's "
            "configured ground site: AOS/LOS/TCA times (UTC), max elevation, "
            "and duration. Call this when the user asks when an object is "
            "visible, overhead, or passes over their site. The site comes "
            "from the page context automatically (the user sets it on an "
            "object page); if none is set this tool errors with guidance — "
            "relay that guidance to the user."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "norad": {"type": "integer", "description": "NORAD catalog id."},
                "hours": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 168,
                    "description": "Look-ahead window in hours "
                                   "(default 48, cap 168).",
                },
            },
            "required": ["norad"],
            "additionalProperties": False,
        },
    },
    {
        "name": "simulate_maneuver",
        "description": (
            "What-if avoidance maneuver for one conjunction: apply an "
            "along-track delta-v to the PRIMARY object at the burn time and "
            "recompute the encounter — new miss distance, new TCA, and new "
            "Pc versus the unmaneuvered values. Call this when the user asks "
            "what a burn / delta-v / avoidance maneuver would do to an "
            "event. Results use a TWO-BODY post-burn approximation (drag/J2 "
            "ignored) — say so when reporting them. This tool only reports "
            "numbers; suggest open_simulator to visualize the encounter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "conjunction_id": {"type": "integer",
                                   "description": "Conjunction id."},
                "dv_along_ms": {
                    "type": "number",
                    "minimum": -5,
                    "maximum": 5,
                    "description": "Along-track delta-v in m/s, -5..5 "
                                   "(positive = prograde).",
                },
                "burn_iso": {
                    "type": "string",
                    "description": "Burn time, ISO-8601 UTC "
                                   "('2026-07-25T18:00:00Z'). Default: TCA "
                                   "minus 6 hours. Must be at least 5 "
                                   "minutes before TCA and not more than "
                                   "1 hour in the past.",
                },
            },
            "required": ["conjunction_id", "dv_along_ms"],
            "additionalProperties": False,
        },
    },
    {
        "name": "optimize_maneuver",
        "description": (
            "Find the SMALLEST along-track delta-v (searching -5..+5 m/s) "
            "that raises a conjunction's miss distance to at least the "
            "target. Call this when the user asks how much delta-v is "
            "needed, for the minimum or cheapest burn, or to plan an "
            "avoidance maneuver for an event. Results use a TWO-BODY "
            "post-burn approximation (drag/J2 ignored) — ALWAYS state that "
            "caveat when reporting them. When the target is unreachable "
            "within plus/minus 5 m/s the result carries reachable: false "
            "with the best achievable miss — report that honestly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "conjunction_id": {"type": "integer",
                                   "description": "Conjunction id."},
                "target_miss_m": {
                    "type": "number",
                    "minimum": 1,
                    "description": "Required post-burn miss distance in "
                                   "meters (default 1000).",
                },
                "burn_iso": {
                    "type": "string",
                    "description": "Burn time, ISO-8601 UTC "
                                   "('2026-07-25T18:00:00Z'). Default: TCA "
                                   "minus 6 hours. Must be at least 5 "
                                   "minutes before TCA and not more than "
                                   "1 hour in the past.",
                },
            },
            "required": ["conjunction_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "plan_avoidance",
        "description": (
            "One-click avoidance plan for one conjunction: for burn times "
            "1 orbit, 3 h, 6 h and 12 h before closest approach (those still "
            "at least 5 minutes away), finds the SMALLEST along-track burn "
            "that lifts the miss distance to the target AND drops the "
            "assumed-sigma Pc below the elevated threshold, and recommends "
            "one (smallest delta-v; ties go to the later burn). Call this "
            "when the user asks to plan, recommend, or propose an avoidance "
            "maneuver, asks 'what should we do' about a risky event, or "
            "wants the cheapest safe burn. Prefer it over optimize_maneuver "
            "unless the user names a specific burn time. Results are "
            "screening-grade TWO-BODY estimates, not an operational "
            "maneuver plan — say so. Statuses: burn_recommended, "
            "no_burn_needed, unreachable, geometry_mismatch (report the "
            "note; never call a mismatch an all-clear)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "conjunction_id": {"type": "integer",
                                   "description": "Conjunction id."},
                "target_miss_m": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 100000,
                    "description": "Required post-burn miss distance in "
                                   "meters (default 1000; the Pc "
                                   "criterion may require more).",
                },
            },
            "required": ["conjunction_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_user_objects",
        "description": (
            "List the user's own fleet: objects added with user-supplied "
            "TLEs on the MY FLEET page (/fleet.html), with derived orbital "
            "elements. These are kept separate from the public catalog. "
            "Call this when the user asks about their fleet or their own "
            "objects, or before screen_user_object when you only have a "
            "name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "screen_user_object",
        "description": (
            "Run on-demand conjunction screening of ONE user-fleet object "
            "against the live catalog. BLOCKS FOR ROUGHLY 20 SECONDS. "
            "Results are computed fresh and never stored. Call this ONLY "
            "when the user explicitly asked to screen their object — never "
            "proactively. Resolve the fleet object id with "
            "list_user_objects first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "User-fleet object id (from "
                                   "list_user_objects — NOT a NORAD id).",
                },
                "window_hours": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 168,
                    "description": "Screening look-ahead window in hours "
                                   "(default 72).",
                },
            },
            "required": ["id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_space_weather",
        "description": (
            "Live space weather and what it does to orbits: the NOAA storm "
            "level (G/R/S scales), Kp now and forecast, the latest solar "
            "flare, the solar wind, how much denser the upper atmosphere is "
            "than on a quiet day at LEO altitudes (NASA/NRL MSIS 2.1), and "
            "the active close approaches the storm widens (flag 'watch' = "
            "already elevated/critical; 'may_escalate' = nominal but the "
            "widened uncertainty allows an elevated Pc). Pass norad to get one object's storm effect: "
            "its decay now vs a quiet day and how far a storm-blind "
            "prediction drifts after 24 h and 72 h. Call this when the user "
            "asks about solar flares, storms, the Sun, aurora, drag, decay "
            "or whether space weather affects a satellite or a close "
            "approach. Storm drift widens uncertainty; it never changes the "
            "stored Pc or risk level — say so if asked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "norad": {
                    "type": "integer",
                    "description": "Optional NORAD id: include this "
                                   "object's storm effect.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    # --- UI-ACTION tools -----------------------------------------------------
    {
        "name": "navigate_user",
        "description": (
            "Queue a browser navigation to a RELATIVE Kessler URL: '/' "
            "(dashboard, accepts ?sel=<norad>&risk=<level>&search=<text>; "
            "'/?view=spaceweather' opens its live space-weather view), "
            "'/object.html?id=<norad>[&conj=<id>]', '/alerts.html', "
            "'/fleet.html' (the user's MY FLEET page), '/model.html' (the "
            "VIGIL model card — training data, metrics, feature "
            "importances), or '/wall.html' (the ops wall). Call "
            "this when the user asks to go somewhere, or when a globe/table "
            "action is needed but the current page is NOT the dashboard — "
            "fold the intent into the URL params. The frontend applies "
            "navigation last, after all other queued actions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Relative URL starting with '/'. Absolute "
                                   "URLs and external hosts are rejected.",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    {
        "name": "select_on_globe",
        "description": (
            "Select and highlight an object on the dashboard globe (orbit "
            "line + selected-object panel). ONLY works when the current page "
            "is the dashboard; elsewhere call navigate_user with "
            "'/?sel=<norad>' instead. Call this when the user asks to show, "
            "highlight, or find a specific object on the globe."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "norad": {"type": "integer", "description": "NORAD catalog id."},
            },
            "required": ["norad"],
            "additionalProperties": False,
        },
    },
    {
        "name": "set_globe_filter",
        "description": (
            "Filter which object types are visible on the dashboard globe "
            "and/or restrict it to risk-flagged objects. ONLY works on the "
            "dashboard; elsewhere navigate_user to '/'. Call this when the "
            "user asks to show only debris, payloads, rocket bodies, or "
            "'only the risky ones'. Provide types and/or risk_only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "types": {
                    "type": "array",
                    "items": {"type": "string",
                              "enum": ["PAYLOAD", "DEBRIS", "RB", "UNKNOWN"]},
                    "description": "Object types to keep visible.",
                },
                "risk_only": {
                    "type": "boolean",
                    "description": "Show only objects with active "
                                   "elevated/critical conjunctions.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "set_globe_speed",
        "description": (
            "Set the dashboard globe simulation speed multiplier: 1 (real "
            "time), 60, or 600. ONLY works on the dashboard. Call this when "
            "the user asks to speed up or slow down the orbital display."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mult": {"type": "integer", "enum": [1, 60, 600],
                         "description": "Sim-time multiplier."},
            },
            "required": ["mult"],
            "additionalProperties": False,
        },
    },
    {
        "name": "set_table_filter",
        "description": (
            "Filter the dashboard conjunction table by risk level and/or "
            "search text (matches either object's name or NORAD id). ONLY "
            "works on the dashboard; elsewhere call navigate_user with "
            "'/?risk=<level>&search=<text>'. Call this when the user asks to "
            "filter or find events in the table. risk='all' clears the risk "
            "filter. Provide risk and/or search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "risk": {
                    "type": "string",
                    "enum": ["critical", "elevated", "nominal", "all"],
                    "description": "Risk level to filter to; 'all' clears.",
                },
                "search": {
                    "type": "string",
                    "description": "Search text; empty string clears.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "fill_field",
        "description": (
            "Type into any input field in the Kessler UI on the user's "
            "behalf: sets the field's value and fires the same events as a "
            "keystroke, so the page reacts exactly as if the user typed it. "
            "Known field ids — dashboard: 'globe-search' (globe search box), "
            "'conj-search' (conjunction table search), 'conj-risk' (risk "
            "dropdown: critical|elevated|nominal|all); MY FLEET: 'add-name' "
            "(object name), 'add-tle' (TLE lines textarea), 'add-note' "
            "(note); object page: 'site-lat', 'site-lon' (ground-site "
            "coordinates). The field must exist on the user's CURRENT page — "
            "navigate_user there first if needed (fill_field is applied "
            "before any queued navigation, so fill only after the user is "
            "already on the right page, or in a later turn). For the "
            "dashboard table specifically, prefer set_table_filter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "field_id": {
                    "type": "string",
                    "description": "DOM id of the field (letters, digits, "
                                   "'-', '_').",
                },
                "value": {
                    "type": "string",
                    "description": "Text to enter (<= 500 chars; empty "
                                   "string clears the field).",
                },
            },
            "required": ["field_id", "value"],
            "additionalProperties": False,
        },
    },
    {
        "name": "show_trajectory",
        "description": (
            "Open a trajectory modal (3D orbit plot + ground track) for one "
            "object, optionally overlaying a conjunction partner's orbit. "
            "Works on ANY page. Call this when the user asks to see or "
            "visualize an orbit, trajectory, encounter geometry, or ground "
            "track without leaving the current page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "norad": {"type": "integer",
                          "description": "NORAD id of the object to show."},
                "partner_norad": {
                    "type": "integer",
                    "description": "Optional NORAD id of the conjunction "
                                   "partner to overlay.",
                },
            },
            "required": ["norad"],
            "additionalProperties": False,
        },
    },
    {
        "name": "show_encounter",
        "description": (
            "Play a conjunction encounter on the dashboard globe: selects "
            "both objects, draws TCA markers with a connecting line, and "
            "jumps the sim clock to just before closest approach. Call this "
            "when the user wants to SEE, SHOW, or WATCH a conjunction / "
            "close approach / encounter happen on the globe. ONLY works "
            "when the current page is the dashboard; elsewhere call "
            "navigate_user with '/?conj=<id>' instead. Requires the "
            "conjunction id — resolve it with list_conjunctions first if "
            "you only have object names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "conjunction_id": {"type": "integer",
                                   "description": "Conjunction id."},
            },
            "required": ["conjunction_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "open_simulator",
        "description": (
            "Open the full-screen encounter simulator for one conjunction: "
            "a time-scrubbing relative-motion display of the two objects "
            "around TCA with separation, RIC geometry, and per-second "
            "playback. Works on ANY page. Call this when the user wants to "
            "SIMULATE, watch, replay, step through, or measure a close "
            "approach in detail — including PAST events (replaying a past "
            "conjunction is legitimate). Requires the conjunction id — "
            "resolve it with list_conjunctions first if you only have "
            "object names. For a quick look on the dashboard globe, prefer "
            "show_encounter instead. Set apply_plan true (upcoming events "
            "only) to open it with the recommended avoidance burn loaded "
            "and replayed as a before/after — use this after plan_avoidance "
            "when the user wants to see the plan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "conjunction_id": {"type": "integer",
                                   "description": "Conjunction id."},
                "apply_plan": {
                    "type": "boolean",
                    "description": "Load the recommended avoidance burn "
                                   "into the simulator (default false).",
                },
            },
            "required": ["conjunction_id"],
            "additionalProperties": False,
        },
    },
    # --- report tool ---------------------------------------------------------
    {
        "name": "create_report",
        "description": (
            "Render a standalone, printable HTML report combining live "
            "database tables for the subject with your narrative sections. "
            "Call this ONLY when the user asks for a report, briefing, "
            "assessment document, or something to print/share — not for "
            "ordinary answers. Fetch the relevant data with DATA tools first "
            "and ground the narrative in it. Returns {\"url\"}; include that "
            "link in your reply."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string",
                          "description": "Report title."},
                "subject": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["conjunction", "object", "overview"],
                            "description": "What the data tables cover.",
                        },
                        "id": {
                            "type": "integer",
                            "description": "Conjunction id for "
                                           "type=conjunction; NORAD id for "
                                           "type=object; omit for overview.",
                        },
                    },
                    "required": ["type"],
                    "additionalProperties": False,
                },
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "body": {
                                "type": "string",
                                "description": "Narrative text. Minimal "
                                               "markdown only: paragraphs, "
                                               "**bold**, `code`, '- ' lists.",
                            },
                        },
                        "required": ["heading", "body"],
                        "additionalProperties": False,
                    },
                    "description": "Your analysis sections, grounded in data "
                                   "fetched this turn.",
                },
            },
            "required": ["title", "subject", "sections"],
            "additionalProperties": False,
        },
    },
]


# Tools that drive the interface (the rest read data, run analyses or write
# reports). Served by GET /assistant/tools so the About page never drifts.
UI_ACTION_TOOL_NAMES = frozenset({
    "navigate_user", "select_on_globe", "set_globe_filter", "set_globe_speed",
    "set_table_filter", "fill_field", "show_trajectory", "show_encounter",
    "open_simulator",
})


# --- argument helpers --------------------------------------------------------

def _require_int(args: dict, key: str) -> int:
    value = args.get(key)
    if value is None:
        raise ToolError(f"missing required argument: {key}")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ToolError(f"{key} must be an integer")


def _optional_int(args: dict, key: str):
    value = args.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ToolError(f"{key} must be an integer")


def _clamped_limit(args: dict, default: int, cap: int) -> int:
    value = _optional_int(args, "limit")
    return default if value is None else max(1, min(cap, value))


def _satellite_exists(conn, norad: int) -> bool:
    return conn.execute(
        "SELECT 1 FROM satellites WHERE norad_id = ?", (norad,)
    ).fetchone() is not None


def _require_number(args: dict, key: str) -> float:
    value = args.get(key)
    if value is None:
        raise ToolError(f"missing required argument: {key}")
    if isinstance(value, bool):
        raise ToolError(f"{key} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ToolError(f"{key} must be a number")


def _parse_iso_utc(value: str, key: str) -> datetime:
    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except ValueError:
        raise ToolError(
            f"{key} must be ISO-8601 UTC like 2026-07-25T18:00:00Z"
        )


# --- tolerant dispatch -------------------------------------------------------
# The data tools call into other modules lazily. Callables are resolved against
# a short list of conventional names and invoked by parameter-name binding, so
# a renamed or missing module/function degrades to a clear ToolError, never an
# import crash.

def _optional_module(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _first_callable(module, names):
    if module is None:
        return None
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def _call_supported(fn, available: dict):
    """Call fn filling parameters by name from `available`; every parameter
    not provided must carry a default. Known parameter names (conn, the
    contract's query/path params) are always passed explicitly so FastAPI
    Query/Depends default sentinels never leak into a direct call."""
    sig = inspect.signature(fn)
    kwargs = {}
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        if name in available:
            kwargs[name] = available[name]
        elif param.default is inspect.Parameter.empty:
            raise ToolError(
                f"cannot call {getattr(fn, '__name__', 'helper')}: "
                f"unfilled required parameter '{name}'"
            )
    return fn(**kwargs)


def _site_from_context(context) -> tuple[float, float, float]:
    """(lat_deg, lon_deg, alt_m) from context['site']; ToolError guidance
    (set a site on any object page) when absent or unusable."""
    site = context.get("site") if isinstance(context, dict) else None
    if not isinstance(site, dict):
        raise ToolError(NO_SITE_GUIDANCE)
    lat = site.get("lat") if site.get("lat") is not None else site.get("lat_deg")
    lon = site.get("lon") if site.get("lon") is not None else site.get("lon_deg")
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        raise ToolError(NO_SITE_GUIDANCE)
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ToolError(
            "ground site out of range: lat must be -90..90, lon -180..180"
        )
    try:
        alt_m = float(site.get("alt_m") or 0.0)
    except (TypeError, ValueError):
        alt_m = 0.0
    return lat, lon, alt_m


# --- DATA tool handlers ------------------------------------------------------

def _tool_get_status(rt, args):
    return status_api.get_status(rt.conn)


def _tool_list_conjunctions(rt, args):
    status = args.get("status") or "active"
    if status not in ("active", "past", "all"):
        raise ToolError("status must be one of active|past|all")
    risk = args.get("risk")
    if risk is not None and risk not in RISK_LEVELS:
        raise ToolError("risk must be one of critical|elevated|nominal")
    where, params = ("", ()) if status == "all" else ("c.status = ?", (status,))
    items = conjunctions_api.query_summaries(rt.conn, where, params)
    if risk:
        items = [i for i in items if i["risk_level"] == risk]
    search = (args.get("search") or "").strip().lower()
    if search:
        def matches(item):
            for side in (item["primary"], item["secondary"]):
                if search in side["name"].lower() or search == str(side["norad_id"]):
                    return True
            return False
        items = [i for i in items if matches(i)]
    items.sort(key=conjunctions_api.sort_key)
    limit = _clamped_limit(args, default=20, cap=50)
    return {"count": len(items), "items": items[:limit]}


def _tool_get_conjunction(rt, args):
    cid = _require_int(args, "id")
    try:
        return conjunctions_api.get_conjunction(cid, rt.conn)
    except HTTPException as exc:
        raise ToolError(str(exc.detail))


def _tool_get_object(rt, args):
    norad = _require_int(args, "norad")
    row = rt.conn.execute(
        objects_api._OBJECTS_SQL + " WHERE sat.norad_id = ?", (norad,)
    ).fetchone()
    if row is None:
        raise ToolError(f"Object {norad} not found")
    item = objects_api._item_from_row(row)
    item["tle_count"] = rt.conn.execute(
        "SELECT COUNT(*) FROM tle_history WHERE satellite_id = ?", (row["id"],)
    ).fetchone()[0]
    conj = conjunctions_api.query_summaries(
        rt.conn,
        "(c.primary_object_id = ? OR c.secondary_object_id = ?)",
        (row["id"], row["id"]),
    )
    conj.sort(key=lambda i: (0 if i["status"] == "active" else 1,
                             *conjunctions_api.sort_key(i)))
    item["conjunctions"] = conj
    return item


def _tool_search_objects(rt, args):
    query = (args.get("query") or "").strip()
    if not query:
        raise ToolError("query must be non-empty")
    limit = _clamped_limit(args, default=10, cap=20)
    where, params = objects_api._search_filter(query)
    rows = rt.conn.execute(
        objects_api._OBJECTS_SQL + f" WHERE {where}"
        " ORDER BY sat.name COLLATE NOCASE ASC, sat.norad_id ASC LIMIT ?",
        (*params, limit),
    ).fetchall()
    return {"count": len(rows),
            "items": [objects_api._item_from_row(row) for row in rows]}


def _tool_get_risk(rt, args):
    cid = _require_int(args, "conjunction_id")
    try:
        return risk_api.get_risk(cid, rt.conn)
    except HTTPException as exc:
        raise ToolError(str(exc.detail))


def _tool_explain_risk(rt, args):
    cid = _require_int(args, "conjunction_id")
    try:
        return risk_api.explain_risk(cid, rt.conn)
    except HTTPException as exc:
        raise ToolError(str(exc.detail))


def _tool_run_pipeline_refresh(rt, args):
    # Same gate as POST /ingest/refresh: a public chat must not be able to
    # start a multi-minute refresh or write demo CDMs into the live DB.
    from backend.app.api.ingest import _dev_endpoints_enabled

    if not _dev_endpoints_enabled():
        raise ToolError(
            "refresh is disabled on this server (KESSLER_DEV_ENDPOINTS is off);"
            " data refreshes run on the operator's schedule")
    from backend.app.ingestion import pipeline

    return pipeline.refresh(rt.conn, demo=bool(args.get("demo", False)))


_CHANGES_LIST_CAP = 25


def _tool_get_changes(rt, args):
    changes_api = _optional_module("backend.app.api.changes")
    fn = _first_callable(changes_api, ("get_changes", "compute_changes",
                                       "changes"))
    if fn is None:
        raise ToolError("run-to-run changes are unavailable "
                        "(backend.app.api.changes is not deployed)")
    try:
        result = _call_supported(fn, {"conn": rt.conn, "since_run": "latest-1"})
    except HTTPException as exc:
        raise ToolError(str(exc.detail))
    # A big refresh can diff THOUSANDS of conjunctions; serializing them all
    # into one tool_result once produced a 2.6M-token prompt (a hard API
    # reject). Cap every list, keep the true totals.
    if isinstance(result, dict):
        for key, value in list(result.items()):
            if isinstance(value, list) and len(value) > _CHANGES_LIST_CAP:
                result[key] = value[:_CHANGES_LIST_CAP]
                result[f"{key}_total"] = len(value)
                result[f"{key}_truncated"] = len(value) - _CHANGES_LIST_CAP
    return result


def _tool_get_passes(rt, args):
    norad = _require_int(args, "norad")
    lat, lon, alt_m = _site_from_context(rt.context)
    hours = _optional_int(args, "hours")
    hours = (PASSES_DEFAULT_HOURS if hours is None
             else max(1, min(PASSES_MAX_HOURS, hours)))
    row = rt.conn.execute(
        objects_api._OBJECTS_SQL + " WHERE sat.norad_id = ?", (norad,)
    ).fetchone()
    if row is None:
        raise ToolError(f"Object {norad} not found")
    if not (row["line1"] and row["line2"]):
        raise ToolError(f"Object {norad} has no TLE stored — "
                        "cannot predict passes")
    from backend.app.ingestion import propagate  # lazy, like api/objects.py

    passes_fn = getattr(propagate, "passes", None)
    if not callable(passes_fn):
        raise ToolError("pass prediction is unavailable "
                        "(propagate.passes is not deployed)")
    start = datetime.now(timezone.utc).replace(microsecond=0)
    try:
        # Contract-pinned signature: passes(line1, line2, lat_deg, lon_deg,
        # alt_m, start, hours, min_el_deg=10, step_s=30).
        events = passes_fn(row["line1"], row["line2"], lat, lon, alt_m,
                           start, hours)
    except ValueError as exc:
        raise ToolError(f"pass prediction failed: {exc}")
    return {
        "norad_id": norad,
        "name": row["name"],
        "site": {"lat_deg": lat, "lon_deg": lon, "alt_m": alt_m},
        "start": start.strftime(ISO_FMT),
        "hours": hours,
        "count": len(events),
        "passes": events,
    }


def _maneuver_body(annotation, dv: float, burn_iso: str):
    """Request-body object for a maneuver route parameter. A pydantic model
    annotation is constructed properly (its field validation still runs);
    an unannotated body-ish parameter gets an attribute namespace."""
    if isinstance(annotation, type) and (hasattr(annotation, "model_fields")
                                         or hasattr(annotation, "__fields__")):
        try:
            return annotation(dv_along_ms=dv, burn_iso=burn_iso)
        except Exception as exc:
            raise ToolError(f"maneuver request rejected: {exc}")
    return types.SimpleNamespace(dv_along_ms=dv, burn_iso=burn_iso)


def _call_maneuver(fn, cid: int, dv: float, burn_iso: str, conn):
    """Invoke the maneuver logic by parameter-name binding (see the tolerant
    dispatch note above): flat params, a pydantic body, or a namespace body
    all bind; anything else unfillable is a ToolError."""
    flat = {"conjunction_id": cid, "cid": cid, "id": cid,
            "dv_along_ms": dv, "dv": dv,
            "burn_iso": burn_iso, "burn_time": burn_iso,
            "conn": conn}
    body_names = ("body", "payload", "request", "req", "maneuver")
    sig = inspect.signature(fn)
    kwargs = {}
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        if name in flat:
            kwargs[name] = flat[name]
        elif (name in body_names
              or (isinstance(param.annotation, type)
                  and hasattr(param.annotation, "model_fields"))):
            kwargs[name] = _maneuver_body(param.annotation, dv, burn_iso)
        elif param.default is inspect.Parameter.empty:
            raise ToolError(
                f"cannot call {getattr(fn, '__name__', 'maneuver')}: "
                f"unfilled required parameter '{name}'"
            )
    return fn(**kwargs)


def _tool_simulate_maneuver(rt, args):
    cid = _require_int(args, "conjunction_id")
    dv = _require_number(args, "dv_along_ms")
    if not (-DV_ALONG_MAX_MS <= dv <= DV_ALONG_MAX_MS):
        raise ToolError(
            f"dv_along_ms must be within -{DV_ALONG_MAX_MS:g}.."
            f"{DV_ALONG_MAX_MS:g} m/s"
        )
    row = rt.conn.execute(
        "SELECT time_of_closest_approach AS tca FROM conjunctions"
        " WHERE id = ?",
        (cid,),
    ).fetchone()
    if row is None:
        raise ToolError(f"Conjunction {cid} not found")
    burn_iso = args.get("burn_iso")
    if burn_iso is None:
        tca = _parse_iso_utc(row["tca"], "stored TCA")
        burn_iso = (tca - timedelta(hours=DEFAULT_BURN_LEAD_H)).strftime(ISO_FMT)
    else:
        _parse_iso_utc(burn_iso, "burn_iso")  # validate before dispatch
        burn_iso = str(burn_iso)
    fn = _first_callable(conjunctions_api,
                         ("run_maneuver", "post_maneuver", "simulate_maneuver",
                          "compute_maneuver", "maneuver"))
    if fn is None:
        raise ToolError("maneuver computation is unavailable "
                        "(the conjunction maneuver endpoint is not deployed)")
    try:
        result = _call_maneuver(fn, cid, dv, burn_iso, rt.conn)
    except HTTPException as exc:
        raise ToolError(str(exc.detail))
    out = dict(result) if isinstance(result, dict) else {"result": result}
    out.setdefault("conjunction_id", cid)
    out.setdefault("dv_along_ms", dv)
    out.setdefault("burn_iso", burn_iso)
    out["method"] = MANEUVER_METHOD_NOTE
    return out


def _tool_optimize_maneuver(rt, args):
    cid = _require_int(args, "conjunction_id")
    target = args.get("target_miss_m")
    if target is None:
        target = OPTIMIZE_DEFAULT_TARGET_MISS_M
    else:
        if isinstance(target, bool):
            raise ToolError("target_miss_m must be a number")
        try:
            target = float(target)
        except (TypeError, ValueError):
            raise ToolError("target_miss_m must be a number")
        if target <= 0:
            raise ToolError("target_miss_m must be positive")
    row = rt.conn.execute(
        "SELECT time_of_closest_approach AS tca FROM conjunctions"
        " WHERE id = ?",
        (cid,),
    ).fetchone()
    if row is None:
        raise ToolError(f"Conjunction {cid} not found")
    burn_iso = args.get("burn_iso")
    if burn_iso is None:
        tca = _parse_iso_utc(row["tca"], "stored TCA")
        burn_iso = (tca - timedelta(hours=DEFAULT_BURN_LEAD_H)).strftime(ISO_FMT)
    else:
        _parse_iso_utc(burn_iso, "burn_iso")  # validate before dispatch
        burn_iso = str(burn_iso)
    # Contract-pinned signature (models/maneuver.py, built in parallel):
    # optimize_dv(conn, conjunction_id, target_miss_m, burn_iso) ->
    # {dv_ms, achieved_miss_m, burn_iso, evaluations, reachable, method}.
    # An unreachable target is a NORMAL result carrying reachable: false and
    # the best found; ValueError means burn/TLE/propagation problems.
    maneuver_mod = _optional_module("backend.app.models.maneuver")
    fn = _first_callable(maneuver_mod,
                         ("optimize_dv", "optimize_maneuver", "optimize"))
    if fn is None:
        raise ToolError("maneuver optimization is unavailable "
                        "(models.maneuver.optimize_dv is not deployed)")
    try:
        result = _call_supported(fn, {
            "conn": rt.conn,
            "conjunction_id": cid, "cid": cid, "id": cid,
            "target_miss_m": target, "target": target,
            "burn_iso": burn_iso, "burn_time": burn_iso,
        })
    except HTTPException as exc:
        raise ToolError(str(exc.detail))
    except ValueError as exc:
        raise ToolError(f"maneuver optimization rejected: {exc}")
    out = dict(result) if isinstance(result, dict) else {"result": result}
    out.setdefault("conjunction_id", cid)
    out.setdefault("target_miss_m", target)
    out.setdefault("burn_iso", burn_iso)
    out.setdefault("reachable", True)
    out["approximation"] = MANEUVER_METHOD_NOTE
    return out


def _tool_plan_avoidance(rt, args):
    cid = _require_int(args, "conjunction_id")
    target = args.get("target_miss_m")
    if target is not None:
        if isinstance(target, bool):
            raise ToolError("target_miss_m must be a number")
        try:
            target = float(target)
        except (TypeError, ValueError):
            raise ToolError("target_miss_m must be a number")
    from backend.app.models import maneuver as maneuver_mod

    try:
        plan = maneuver_mod.plan_avoidance(rt.conn, cid, target_miss_m=target)
    except LookupError as exc:
        raise ToolError(str(exc))
    except ValueError as exc:   # includes PlanConflict (passed / too soon)
        raise ToolError(f"avoidance plan unavailable: {exc}")
    # The model needs the decision, not the prose rule twice or the search
    # bookkeeping: trim to what it should report.
    out = {k: plan[k] for k in ("conjunction_id", "tca", "status", "note",
                                "baseline", "recommended", "method")}
    crit = dict(plan["criteria"] or {})
    crit.pop("rule", None)
    out["criteria"] = crit
    out["recommendation_rule"] = ("smallest |dv| among safe options; ties "
                                  "within 0.01 m/s go to the later burn")
    out["options"] = [
        {k: o[k] for k in ("lead_label", "burn_iso", "dv_ms", "direction",
                           "new_miss_m", "new_pc", "reachable",
                           "propellant_g_per_100kg")}
        for o in plan["options"]
    ]
    out["fuel_context"] = ("propellant_g_per_100kg assumes Isp 220 s "
                           "(hydrazine class) and a 100 kg spacecraft")
    return out


def _tool_get_anomalies(rt, args):
    hours = _optional_int(args, "hours")
    hours = (ANOMALY_DEFAULT_HOURS if hours is None
             else max(1, min(ANOMALY_MAX_HOURS, hours)))
    # Contract: models/anomaly.py scan_tle_deltas(conn, since_hours=48) (built
    # in parallel); the GET /anomalies handler in api/anomalies.py wraps the
    # same logic - try the model first, degrade to the API module.
    fn = _first_callable(_optional_module("backend.app.models.anomaly"),
                         ("scan_tle_deltas", "scan_deltas", "scan"))
    if fn is None:
        fn = _first_callable(_optional_module("backend.app.api.anomalies"),
                             ("get_anomalies", "list_anomalies", "anomalies"))
    if fn is None:
        raise ToolError("anomaly scanning is unavailable "
                        "(models.anomaly / api.anomalies is not deployed)")
    try:
        result = _call_supported(fn, {"conn": rt.conn,
                                      "since_hours": hours, "hours": hours})
    except HTTPException as exc:
        raise ToolError(str(exc.detail))
    if isinstance(result, list):
        out = {"items": result}
    elif isinstance(result, dict):
        out = dict(result)
    else:
        out = {"items": result}
    out.setdefault("hours", hours)
    items = out.get("items")
    if isinstance(items, list):
        out.setdefault("count", len(items))
    out["note"] = ANOMALY_HEURISTIC_NOTE
    return out


def _norad_from_line1(line1) -> int | None:
    """NORAD id from TLE line-1 columns 3-7; None when undecodable."""
    try:
        return int(str(line1)[2:7])
    except (TypeError, ValueError):
        return None


def _user_object_item(row) -> dict:
    """GET /user-objects item shape (contract): stored fields + elements
    derived from the stored TLE, degrading to nulls like api/objects."""
    item = {
        "id": row["id"],
        "name": row["name"],
        "norad_id": _norad_from_line1(row["line1"]),
        "note": row["note"],
        "added_at": row["added_at"],
        "epoch": None,
        "inclination_deg": None,
        "apogee_km": None,
        "perigee_km": None,
        "period_min": None,
    }
    try:
        from backend.app.ingestion import propagate

        elements = propagate.orbital_elements(row["line1"], row["line2"])
        for key in ("epoch", "inclination_deg", "apogee_km", "perigee_km",
                    "period_min"):
            item[key] = elements.get(key)
    except Exception:
        pass  # unparseable stored TLE: keep null elements
    return item


def _tool_list_user_objects(rt, args):
    try:
        rows = rt.conn.execute(
            "SELECT id, name, line1, line2, note, added_at FROM user_objects"
            " ORDER BY id ASC"
        ).fetchall()
    except Exception as exc:
        raise ToolError(f"user fleet is unavailable: {exc}")
    items = [_user_object_item(row) for row in rows]
    return {"count": len(items), "items": items}


def _tool_screen_user_object(rt, args):
    uid = _require_int(args, "id")
    window_hours = _optional_int(args, "window_hours")
    window_hours = (SCREEN_DEFAULT_WINDOW_HOURS if window_hours is None
                    else max(1, min(SCREEN_MAX_WINDOW_HOURS, window_hours)))
    try:
        row = rt.conn.execute(
            "SELECT id FROM user_objects WHERE id = ?", (uid,)
        ).fetchone()
    except Exception as exc:
        raise ToolError(f"user fleet is unavailable: {exc}")
    if row is None:
        raise ToolError(f"User object {uid} not found")
    user_objects_api = _optional_module("backend.app.api.user_objects")
    fn = _first_callable(user_objects_api,
                         ("screen_user_object", "screen_object", "run_screen",
                          "screen"))
    if fn is None:
        raise ToolError("on-demand fleet screening is unavailable "
                        "(backend.app.api.user_objects is not deployed)")
    try:
        return _call_supported(fn, {
            "id": uid, "user_object_id": uid, "object_id": uid, "uid": uid,
            "window_hours": window_hours, "conn": rt.conn,
        })
    except HTTPException as exc:
        raise ToolError(str(exc.detail))


# --- UI-ACTION tool handlers -------------------------------------------------

def _tool_navigate_user(rt, args):
    url = str(args.get("url") or "")
    # Browsers treat a backslash like a slash, so "/\\evil.example" is
    # protocol-relative too; control characters are stripped by URL parsers.
    if (not url.startswith("/") or url[1:2] in ("/", "\\")
            or any(ord(ch) < 0x20 for ch in url)):
        raise ToolError("url must be a relative path starting with '/' "
                        "(no absolute URLs, no external hosts)")
    rt.actions.append({"type": "navigate", "url": url})
    return {"ok": True}


def _tool_select_on_globe(rt, args):
    norad = _require_int(args, "norad")
    if not _satellite_exists(rt.conn, norad):
        raise ToolError(f"Object {norad} is not in the catalog")
    rt.actions.append({"type": "select_on_globe", "norad": norad})
    return {"ok": True}


def _tool_set_globe_filter(rt, args):
    types = args.get("types")
    risk_only = args.get("risk_only")
    if types is None and risk_only is None:
        raise ToolError("provide types and/or risk_only")
    action = {"type": "set_globe_filter"}
    if types is not None:
        if not isinstance(types, list) or not types:
            raise ToolError("types must be a non-empty array")
        normalized = [str(t).upper() for t in types]
        bad = [t for t in normalized if t not in GLOBE_TYPES]
        if bad:
            raise ToolError(
                f"unknown object types: {bad}; allowed: {list(GLOBE_TYPES)}")
        action["types"] = normalized
    if risk_only is not None:
        action["risk_only"] = bool(risk_only)
    rt.actions.append(action)
    return {"ok": True}


def _tool_set_globe_speed(rt, args):
    mult = _require_int(args, "mult")
    if mult not in SPEED_MULTS:
        raise ToolError(f"mult must be one of {list(SPEED_MULTS)}")
    rt.actions.append({"type": "set_globe_speed", "mult": mult})
    return {"ok": True}


def _tool_set_table_filter(rt, args):
    risk = args.get("risk")
    search = args.get("search")
    if risk is None and search is None:
        raise ToolError("provide risk and/or search")
    action = {"type": "set_table_filter"}
    if risk is not None:
        if risk not in (*RISK_LEVELS, "all"):
            raise ToolError("risk must be one of critical|elevated|nominal|all")
        action["risk"] = risk
    if search is not None:
        action["search"] = str(search)
    rt.actions.append(action)
    return {"ok": True}


_FIELD_ID_MAX = 64
_FIELD_VALUE_MAX = 500


def _tool_fill_field(rt, args):
    field_id = str(args.get("field_id") or "")
    if not field_id or len(field_id) > _FIELD_ID_MAX or not all(
        c.isalnum() or c in "-_" for c in field_id
    ):
        raise ToolError("field_id must be 1-64 chars of letters, digits, "
                        "'-' or '_'")
    value = args.get("value")
    if not isinstance(value, str):
        raise ToolError("value must be a string")
    if len(value) > _FIELD_VALUE_MAX:
        raise ToolError(f"value must be <= {_FIELD_VALUE_MAX} chars")
    rt.actions.append({"type": "set_field", "field_id": field_id,
                       "value": value})
    return {"ok": True}


def _tool_show_trajectory(rt, args):
    norad = _require_int(args, "norad")
    if not _satellite_exists(rt.conn, norad):
        raise ToolError(f"Object {norad} is not in the catalog")
    partner = _optional_int(args, "partner_norad")
    if partner is not None and not _satellite_exists(rt.conn, partner):
        raise ToolError(f"Partner object {partner} is not in the catalog")
    action = {"type": "show_trajectory", "norad": norad}
    if partner is not None:
        action["partner_norad"] = partner
    rt.actions.append(action)
    return {"ok": True}


def _tool_show_encounter(rt, args):
    cid = _require_int(args, "conjunction_id")
    row = rt.conn.execute(
        """
        SELECT c.time_of_closest_approach AS tca,
               p.norad_id AS primary_norad,   p.name AS primary_name,
               s.norad_id AS secondary_norad, s.name AS secondary_name
        FROM conjunctions c
        JOIN satellites p ON p.id = c.primary_object_id
        JOIN satellites s ON s.id = c.secondary_object_id
        WHERE c.id = ?
        """,
        (cid,),
    ).fetchone()
    if row is None:
        raise ToolError(f"Conjunction {cid} not found")
    rt.actions.append({
        "type": "show_encounter",
        "conjunction_id": cid,
        "primary_norad": row["primary_norad"],
        "secondary_norad": row["secondary_norad"],
        "tca": row["tca"],
    })
    return {
        "ok": True,
        "conjunction_id": cid,
        "primary_norad": row["primary_norad"],
        "primary_name": row["primary_name"],
        "secondary_norad": row["secondary_norad"],
        "secondary_name": row["secondary_name"],
        "tca": row["tca"],
    }


def _tool_open_simulator(rt, args):
    cid = _require_int(args, "conjunction_id")
    # Existence check only - active or past: replaying a past encounter in
    # the simulator is a legitimate use (the ephemeris endpoint reconstructs
    # the window around TCA either way).
    row = rt.conn.execute(
        """
        SELECT c.time_of_closest_approach AS tca, c.status,
               p.norad_id AS primary_norad,   p.name AS primary_name,
               s.norad_id AS secondary_norad, s.name AS secondary_name
        FROM conjunctions c
        JOIN satellites p ON p.id = c.primary_object_id
        JOIN satellites s ON s.id = c.secondary_object_id
        WHERE c.id = ?
        """,
        (cid,),
    ).fetchone()
    if row is None:
        raise ToolError(f"Conjunction {cid} not found")
    apply_plan = args.get("apply_plan")
    if apply_plan is not None and not isinstance(apply_plan, bool):
        raise ToolError("apply_plan must be a boolean")
    action = {"type": "open_simulator", "conjunction_id": cid}
    if apply_plan:
        tca = _parse_iso_utc(row["tca"], "stored TCA")
        if tca <= datetime.now(timezone.utc):
            raise ToolError("apply_plan needs an upcoming event; this "
                            "closest approach has already passed")
        action["apply_plan"] = True
    rt.actions.append(action)
    return {
        "ok": True,
        "apply_plan": bool(apply_plan),
        "conjunction_id": cid,
        "primary_norad": row["primary_norad"],
        "primary_name": row["primary_name"],
        "secondary_norad": row["secondary_norad"],
        "secondary_name": row["secondary_name"],
        "tca": row["tca"],
        "status": row["status"],
    }


# --- report tool handler -----------------------------------------------------

def _tool_create_report(rt, args):
    sections = args.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ToolError("sections must be a non-empty array of {heading, body}")
    try:
        url = report_mod.create_report(
            rt.conn,
            str(args.get("title") or "Kessler report"),
            args.get("subject") or {},
            sections,
        )
    except ValueError as exc:
        raise ToolError(str(exc))
    rt.report_url = url
    return {"url": url}


SPACE_WEATHER_METHOD_NOTE = (
    "storm drift = each object's own TLE drag term (ndot) scaled by the "
    "MSIS density ratio at its altitude, integrated over NOAA's observed "
    "and forecast Kp from TLE epoch; reported as extra 1-sigma along-track "
    "uncertainty (cautious: a TLE fitted during the storm already holds "
    "part of it)"
)


def _tool_get_space_weather(rt, args):
    from backend.app import spaceweather as sw
    data = sw.peek() or sw.conditions()
    summary = sw.summary(data)
    if summary is None:
        raise ToolError("Live space weather is unavailable right now (the "
                        "NOAA feeds are unreachable or switched off).")
    th = data.get("thermosphere") or {}
    out = {
        "summary": summary,
        "source": "NOAA Space Weather Prediction Center; NASA/NRL MSIS 2.1",
        "kp_forecast": (data.get("kp_forecast") or [])[:8],
        "density_ratio_by_altitude_km": {
            str(h): round(sw._interp(h, th["alts_km"], th["ratio_now"]), 2)
            for h in (300, 400, 500, 600, 700, 800)} if th.get("ratio_now") else None,
        "method": SPACE_WEATHER_METHOD_NOTE,
    }
    ctx = sw.storm_context(data)
    norad = _optional_int(args, "norad")
    if norad is not None:
        row = rt.conn.execute(
            objects_api._OBJECTS_SQL + " WHERE sat.norad_id = ?", (norad,)).fetchone()
        if row is None:
            raise ToolError(f"Object {norad} not found")
        out["object"] = {"norad_id": norad, "name": row["name"],
                         "effects": sw.object_effects(row["line1"], row["line2"], ctx)}
    if ctx is not None:
        items = conjunctions_api.active_storm_items(rt.conn, ctx)
        flagged = conjunctions_api.storm_flagged(items)
        out["close_approaches"] = {
            "active": len(items),
            "widened": sum(1 for i in items if (i.get("space_weather") or {}).get("widened")),
            "flagged": len(flagged),
            "flag_meaning": {"watch": "already elevated/critical; the storm widens its uncertainty",
                             "may_escalate": "nominal now; the widened uncertainty allows an elevated Pc"},
            "top": [{"id": i["id"], "primary": i["primary"]["name"],
                     "secondary": i["secondary"]["name"], "tca": i["tca"],
                     "risk_level": i["risk_level"],
                     "miss_distance_m": round(i["miss_distance_m"]),
                     **{k: i["space_weather"][k] for k in
                        ("flag", "drift_m", "sigma_m", "sigma_storm_m", "pc_storm")}}
                    for i in flagged[:5]],
        }
    return out


_HANDLERS = {
    "get_status": _tool_get_status,
    "get_space_weather": _tool_get_space_weather,
    "get_anomalies": _tool_get_anomalies,
    "optimize_maneuver": _tool_optimize_maneuver,
    "plan_avoidance": _tool_plan_avoidance,
    "list_conjunctions": _tool_list_conjunctions,
    "get_conjunction": _tool_get_conjunction,
    "get_object": _tool_get_object,
    "search_objects": _tool_search_objects,
    "get_risk": _tool_get_risk,
    "explain_risk": _tool_explain_risk,
    "run_pipeline_refresh": _tool_run_pipeline_refresh,
    "get_changes": _tool_get_changes,
    "get_passes": _tool_get_passes,
    "simulate_maneuver": _tool_simulate_maneuver,
    "list_user_objects": _tool_list_user_objects,
    "screen_user_object": _tool_screen_user_object,
    "navigate_user": _tool_navigate_user,
    "select_on_globe": _tool_select_on_globe,
    "set_globe_filter": _tool_set_globe_filter,
    "set_globe_speed": _tool_set_globe_speed,
    "set_table_filter": _tool_set_table_filter,
    "fill_field": _tool_fill_field,
    "show_trajectory": _tool_show_trajectory,
    "show_encounter": _tool_show_encounter,
    "open_simulator": _tool_open_simulator,
    "create_report": _tool_create_report,
}


class ToolRuntime:
    """Per-request tool executor: DB access + ordered UI-action queue.

    execute() returns (json_string, is_error) ready to drop into a
    tool_result block. Tool failures never raise out of here - they surface
    to the model as is_error results so the loop keeps going. (The Claude
    API call itself is wrapped separately, in service._create_message.)
    """

    def __init__(self, conn, context: dict | None = None):
        self.conn = conn
        self.context: dict = context if isinstance(context, dict) else {}
        self.actions: list[dict] = []
        self.report_url: str | None = None

    def execute(self, name: str, args: dict) -> tuple[str, bool]:
        handler = _HANDLERS.get(name)
        if handler is None:
            return json.dumps({"error": f"unknown tool: {name}"}), True
        try:
            result = handler(self, args if isinstance(args, dict) else {})
        except ToolError as exc:
            return json.dumps({"error": str(exc)}), True
        except Exception as exc:  # tool bug/bad data - report, don't crash loop
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"}), True
        return json.dumps(result, default=str), False
