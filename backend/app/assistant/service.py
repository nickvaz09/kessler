"""Kessler AI chat service: the server-side agentic tool loop.

chat(messages, context, client=None) drives the Claude API until end_turn
(or refusal / iteration cap) and returns {reply, actions, report_url, model}.
chat_stream(...) runs the same loop but returns a generator of typed events
({type: text|tool|action|error|done}) for the SSE endpoint - each turn uses
client.messages.stream(**request), forwarding text_stream deltas live and
reading tool blocks from get_final_message(). The Anthropic client is
constructed lazily inside the request path - never at import - and tests
inject a FakeClient through the `client` parameter.

Prompt-cache discipline: the system prompt (frozen, cache_control ephemeral)
and the deterministic tool list form the stable prefix; all volatile
per-request context (page/url/selection + a one-line live status snapshot) is
rendered into the FINAL user message envelope, never the system prompt.

watch_pass(conn, client=None) is the headless change-of-watch pass: the
same loop against read-only DATA tools only, strict-JSON structured output
inserted into the watchlog table - it returns None instead of raising, so a
failed pass never breaks the scheduled-refresh caller.
"""
import json
import os
import sys
from datetime import datetime, timezone

from fastapi import HTTPException

from backend.app import db
from backend.app.api import status as status_api
from backend.app.assistant import prompt, tools

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "high"
MAX_TOKENS = 16000
MAX_ITERATIONS = 12

AUTH_DETAIL = (
    "Kessler AI needs an Anthropic API key: set ANTHROPIC_API_KEY "
    "(or `ant auth login`) and restart."
)
CAP_NOTE = (
    "[Kessler AI: stopped at the tool-iteration cap (12) — the answer above "
    "may be partial.]"
)
REFUSAL_REPLY = "Sorry — I can't help with that request."


def _default_client():
    """Zero-arg client: resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN /
    an `ant auth login` profile. Constructed per request, never at import."""
    import anthropic

    try:
        return anthropic.Anthropic()
    except TypeError:
        # The SDK raises TypeError (not AuthenticationError) when no
        # credential source resolves at construction time.
        raise HTTPException(status_code=503, detail=AUTH_DETAIL)


def _status_snapshot(conn) -> str:
    """One-line live status for the user-message envelope; never fatal."""
    try:
        s = status_api.get_status(conn)
        return (
            f"catalog {s['catalog_size']} objects · "
            f"{s['active_conjunctions']} active conjunctions "
            f"({s['elevated_count']} elevated / {s['critical_count']} critical)"
            f" · model {s['model_version']}"
            f" · last ingest {s['last_ingest_at'] or 'never'}"
        )
    except Exception:
        return "unavailable"


def _context_envelope(context: dict, conn) -> str:
    parts = [f"page={context.get('page') or 'unknown'}",
             f"url={context.get('url') or '?'}"]
    if context.get("selected_norad"):
        parts.append(f"selected_norad={context['selected_norad']}")
    return (
        "<context>\n"
        f"client: {' · '.join(parts)}\n"
        f"live status: {_status_snapshot(conn)}\n"
        "</context>"
    )


def _build_messages(messages: list[dict], context: dict, conn) -> list[dict]:
    """Copy the transcript and wrap the last user message with the volatile
    context envelope (cache-safe placement: end of the prompt)."""
    loop = [{"role": m["role"], "content": m["content"]} for m in messages
            if m.get("content")]
    # The API rejects a transcript that opens with an assistant turn - which
    # is exactly what the client sends after a SHIFT BRIEF greeting. Fold any
    # leading assistant text into the first user turn as quoted context.
    lead = []
    while loop and loop[0]["role"] == "assistant":
        lead.append(loop.pop(0)["content"])
    if lead:
        note = ("<earlier_assistant_message>\n" + "\n\n".join(lead)
                + "\n</earlier_assistant_message>")
        if loop:
            loop[0] = {"role": "user", "content": f"{note}\n\n{loop[0]['content']}"}
        else:
            loop.append({"role": "user", "content": note})
    envelope = _context_envelope(context, conn)
    for i in range(len(loop) - 1, -1, -1):
        if loop[i]["role"] == "user":
            loop[i]["content"] = f"{envelope}\n\n{loop[i]['content']}"
            return loop
    loop.append({"role": "user", "content": envelope})
    return loop


def _text_of(response) -> str:
    blocks = getattr(response, "content", None) or []
    return "\n".join(
        block.text
        for block in blocks
        if getattr(block, "type", None) == "text" and block.text
    ).strip()


def _base_request(model: str, effort: str) -> dict:
    """The per-turn request dict shared by both loops (messages added per
    iteration; the streaming path passes it to client.messages.stream)."""
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
        # never pass temperature/top_p/top_k (400 on this model family).
        "system": [
            {
                "type": "text",
                "text": prompt.SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": tools.TOOLS,
    }


def _status_error_detail(exc) -> str:
    """Operator-facing line for an APIStatusError. exc.message carries the
    raw upstream JSON body (request ids, billing text) - that belongs in the
    server log, never in the UI. The one case worth naming is exhausted
    credits, because the operator can act on it."""
    print(f"[kessler-ai] Claude API error: {exc}", file=sys.stderr)
    if "credit balance is too low" in str(exc.message):
        return ("Claude API credits are exhausted — add credits to the "
                "Anthropic account to restore Kessler AI.")
    return f"Claude API error (HTTP {exc.status_code}) — details in the server log."


def _create_message(client, request: dict):
    """One API call, typed exception chain - never a bare except."""
    import anthropic

    try:
        return client.messages.create(**request)
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=503, detail=AUTH_DETAIL)
    except TypeError:
        # SDK raises TypeError from _validate_headers when no credential
        # source (env var / profile) resolves for the request.
        raise HTTPException(status_code=503, detail=AUTH_DETAIL)
    except anthropic.RateLimitError:
        raise HTTPException(
            status_code=429, detail="Claude API rate limited — retry shortly."
        )
    except anthropic.APIStatusError as exc:
        raise HTTPException(status_code=502, detail=_status_error_detail(exc))
    except anthropic.APIConnectionError:
        raise HTTPException(
            status_code=502, detail="Cannot reach the Claude API (network error)."
        )


def chat(messages: list[dict], context: dict | None, client=None) -> dict:
    """Run the agentic loop and return {reply, actions, report_url, model}."""
    if not messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")
    if client is None:
        client = _default_client()
    model = os.environ.get("KESSLER_AI_MODEL", DEFAULT_MODEL)
    effort = os.environ.get("KESSLER_AI_EFFORT", DEFAULT_EFFORT)

    conn = db.get_conn()
    try:
        runtime = tools.ToolRuntime(conn, context or {})
        loop_messages = _build_messages(messages, context or {}, conn)
        base_request = _base_request(model, effort)

        reply = None
        response = None
        for _ in range(MAX_ITERATIONS):
            response = _create_message(
                client, {**base_request, "messages": loop_messages}
            )
            stop = getattr(response, "stop_reason", None)

            if stop == "tool_use":
                # Run every tool_use block; return all results in one user
                # message (splitting them degrades parallel tool calling).
                loop_messages.append(
                    {"role": "assistant", "content": response.content}
                )
                results = []
                for block in response.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    content, is_error = runtime.execute(
                        block.name, block.input or {}
                    )
                    result = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                    }
                    if is_error:
                        result["is_error"] = True
                    results.append(result)
                loop_messages.append({"role": "user", "content": results})
                continue

            if stop == "pause_turn":
                # Server-side pause: re-send with the paused turn appended.
                loop_messages.append(
                    {"role": "assistant", "content": response.content}
                )
                continue

            if stop == "refusal":
                reply = REFUSAL_REPLY
            else:  # end_turn / max_tokens - take the text we have
                reply = _text_of(response)
            break

        if reply is None:  # iteration cap reached
            text = _text_of(response) if response is not None else ""
            reply = f"{text}\n\n{CAP_NOTE}" if text else CAP_NOTE

        return {
            "reply": reply,
            "actions": runtime.actions,
            "report_url": runtime.report_url,
            "model": model,
        }
    finally:
        conn.close()


# --- streaming ---------------------------------------------------------------

def _stream_error_detail(exc) -> str:
    """The _create_message exception mapping, rendered as an event detail
    string for mid-stream failures (the response has already started, so an
    HTTPException can no longer change the status code). isinstance order
    matters: AuthenticationError and RateLimitError subclass APIStatusError.
    """
    import anthropic

    if isinstance(exc, (anthropic.AuthenticationError, TypeError)):
        return AUTH_DETAIL
    if isinstance(exc, anthropic.RateLimitError):
        return "Claude API rate limited — retry shortly."
    if isinstance(exc, anthropic.APIStatusError):
        return _status_error_detail(exc)
    return "Cannot reach the Claude API (network error)."


def chat_stream(messages: list[dict], context: dict | None, client=None):
    """Run the agentic loop, returning a generator of typed event dicts:

        {type:'text', text}     live text deltas, forwarded as they stream
        {type:'tool', name}     a tool is about to execute
        {type:'action', action} a UI action was queued, at tool-execution time
        {type:'done', report_url, model}   terminal success event
        {type:'error', detail}  terminal failure event, then clean close

    Validation and client construction happen HERE, before the first byte is
    sent, so they surface as plain HTTPExceptions; everything raised while
    the generator runs becomes one error event followed by a clean close.
    """
    if not messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")
    if client is None:
        client = _default_client()
    model = os.environ.get("KESSLER_AI_MODEL", DEFAULT_MODEL)
    effort = os.environ.get("KESSLER_AI_EFFORT", DEFAULT_EFFORT)
    return _stream_events(client, model, effort, messages, context or {})


def _stream_events(client, model, effort, messages, context):
    import anthropic

    conn = db.get_conn()
    try:
        runtime = tools.ToolRuntime(conn, context)
        loop_messages = _build_messages(messages, context, conn)
        base_request = _base_request(model, effort)

        streamed_text = False
        break_before_text = False
        for _ in range(MAX_ITERATIONS):
            # Per-turn streaming call: the request dict is unchanged apart
            # from being passed to stream(). text_stream deltas forward
            # live; tool blocks come from the accumulated final message.
            try:
                with client.messages.stream(
                    **{**base_request, "messages": loop_messages}
                ) as stream:
                    iter_text = False
                    for delta in stream.text_stream:
                        if delta:
                            if break_before_text and not iter_text:
                                # First text after a tool round: break the
                                # paragraph so "Filtered.Opening..." never runs
                                # together (pause_turn continues mid-text).
                                yield {"type": "text", "text": "\n\n"}
                                break_before_text = False
                            iter_text = True
                            streamed_text = True
                            yield {"type": "text", "text": delta}
                    response = stream.get_final_message()
            except (anthropic.APIStatusError, anthropic.APIConnectionError,
                    TypeError) as exc:
                # Same typed chain as _create_message - one error event,
                # then close cleanly.
                yield {"type": "error", "detail": _stream_error_detail(exc)}
                return
            stop = getattr(response, "stop_reason", None)

            if stop == "tool_use":
                break_before_text = streamed_text
                loop_messages.append(
                    {"role": "assistant", "content": response.content}
                )
                results = []
                for block in response.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    yield {"type": "tool", "name": block.name}
                    queued = len(runtime.actions)
                    content, is_error = runtime.execute(
                        block.name, block.input or {}
                    )
                    # emit actions the moment the tool that queued them ran
                    for action in runtime.actions[queued:]:
                        yield {"type": "action", "action": action}
                    result = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                    }
                    if is_error:
                        result["is_error"] = True
                    results.append(result)
                loop_messages.append({"role": "user", "content": results})
                continue

            if stop == "pause_turn":
                loop_messages.append(
                    {"role": "assistant", "content": response.content}
                )
                continue

            if stop == "refusal":
                yield {"type": "text", "text": REFUSAL_REPLY}
            break
        else:  # iteration cap reached without a terminal stop
            note = f"\n\n{CAP_NOTE}" if streamed_text else CAP_NOTE
            yield {"type": "text", "text": note}

        yield {"type": "done", "report_url": runtime.report_url, "model": model}
    finally:
        conn.close()


# --- watch mode ---------------------------------------------------------------

WATCH_MAX_ITERATIONS = 6
WATCH_DEFAULT_EFFORT = "low"

# Read-only DATA tools only: a headless pass must never queue UI actions or
# write reports, and the consent-gated / long-blocking data tools
# (run_pipeline_refresh, screen_user_object), the browser-context one
# (get_passes needs the user's ground site), and the maneuver what-ifs stay
# out of reach entirely. Filtered from TOOLS so order stays deterministic.
WATCH_TOOL_NAMES = (
    "get_status", "list_conjunctions", "get_conjunction", "get_object",
    "search_objects", "get_risk", "get_changes", "get_anomalies",
)
WATCH_TOOLS = [t for t in tools.TOOLS if t["name"] in WATCH_TOOL_NAMES]

# Frozen (nothing interpolated) so repeated watch passes share THEIR OWN
# cache prefix run-to-run. (No prefix is shared with chat regardless of
# system-block order: the cache prefix starts at the tools array, and the
# watch pass uses a filtered tool list.)
WATCH_SYSTEM_ADDENDUM = (
    "You are writing the change-of-watch log entry for the incoming shift. "
    "Review the live picture with your tools before writing — get_status, "
    "get_changes for what moved since the last run, the top active "
    "conjunctions via list_conjunctions, and get_anomalies — then reply "
    "with ONLY the JSON object. summary_md: at most 1500 characters of "
    "terse watch-officer markdown bullets — lead with critical/elevated "
    "events, then changes since the last run, then anomalies; cite "
    "model_version for any risk figure. spoken: at most 450 characters of "
    "plain sentences for text-to-speech — no markdown, no shorthand that "
    "reads badly aloud."
)
WATCH_USER_MESSAGE = "Write the change-of-watch log entry for this shift."

# The 1500/450-char caps are prompt constraints - string maxLength is not
# enforced by structured outputs.
WATCHLOG_SCHEMA = {
    "type": "object",
    "properties": {
        "summary_md": {"type": "string"},
        "spoken": {"type": "string"},
    },
    "required": ["summary_md", "spoken"],
    "additionalProperties": False,
}


def _watch_request(model: str, effort: str) -> dict:
    """Chat's base request narrowed for the headless pass: filtered DATA
    tool list, structured-outputs format, frozen addendum after the cached
    system block."""
    request = _base_request(model, effort)
    request["tools"] = WATCH_TOOLS
    request["output_config"] = {
        "effort": effort,
        "format": {"type": "json_schema", "schema": WATCHLOG_SCHEMA},
    }
    request["system"] = [
        *request["system"],
        {"type": "text", "text": WATCH_SYSTEM_ADDENDUM},
    ]
    return request


def _parse_watch_entry(text: str) -> tuple[str, str]:
    """(summary_md, spoken) from the structured-outputs JSON; ValueError on
    anything malformed (json.loads raises a ValueError subclass itself)."""
    parsed = json.loads(text) if text else None
    if not isinstance(parsed, dict):
        raise ValueError("watch output is not a JSON object")
    summary_md = parsed.get("summary_md")
    spoken = parsed.get("spoken")
    if not isinstance(summary_md, str) or not summary_md.strip():
        raise ValueError("watch output missing summary_md")
    if not isinstance(spoken, str) or not spoken.strip():
        raise ValueError("watch output missing spoken")
    # The 450-char cap is prompt-only (structured outputs cannot enforce
    # maxLength) and the model habitually overruns it. /assistant/tts caps
    # text at 600, and the client slices to 600 before sending - store what
    # will actually be spoken so the read-time signature matches.
    return summary_md, spoken[:600]


def watch_pass(conn, client=None) -> dict | None:
    """one headless change-of-watch pass, inserted into watchlog.

    Returns the inserted entry ({id, created_at, model, summary_md, spoken})
    or None on any failure - no credentials, Claude API errors, refusal,
    iteration cap, malformed output. Never raises: the scheduled-refresh
    caller (and the /watchlog/run endpoint) must survive a failed pass.
    Iteration cap 6; effort KESSLER_AI_EFFORT or 'low'.
    """
    model = os.environ.get("KESSLER_AI_MODEL", DEFAULT_MODEL)
    effort = os.environ.get("KESSLER_AI_EFFORT", WATCH_DEFAULT_EFFORT)
    try:
        if client is None:
            client = _default_client()
        runtime = tools.ToolRuntime(conn)
        loop_messages = [{"role": "user", "content": WATCH_USER_MESSAGE}]
        base_request = _watch_request(model, effort)

        entry_text = None
        rounds = 0
        for _ in range(WATCH_MAX_ITERATIONS):
            response = _create_message(
                client, {**base_request, "messages": loop_messages}
            )
            rounds += 1
            stop = getattr(response, "stop_reason", None)

            if stop == "tool_use":
                # Same loop mechanics as chat: run every tool_use block,
                # return all results in one user message.
                loop_messages.append(
                    {"role": "assistant", "content": response.content}
                )
                results = []
                for block in response.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    content, is_error = runtime.execute(
                        block.name, block.input or {}
                    )
                    result = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                    }
                    if is_error:
                        result["is_error"] = True
                    results.append(result)
                loop_messages.append({"role": "user", "content": results})
                continue

            if stop == "pause_turn":
                loop_messages.append(
                    {"role": "assistant", "content": response.content}
                )
                continue

            if stop == "refusal":
                raise ValueError("the model refused the watch pass")
            entry_text = _text_of(response)
            break

        if entry_text is None:
            raise ValueError(
                f"iteration cap ({WATCH_MAX_ITERATIONS}) reached without "
                "a log entry"
            )
        summary_md, spoken = _parse_watch_entry(entry_text)

        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cursor = conn.execute(
            "INSERT INTO watchlog (created_at, model, summary_md, spoken,"
            " meta_json) VALUES (?, ?, ?, ?, ?)",
            (created_at, model, summary_md, spoken,
             json.dumps({"effort": effort, "rounds": rounds})),
        )
        conn.commit()
        return {
            "id": cursor.lastrowid,
            "created_at": created_at,
            "model": model,
            "summary_md": summary_md,
            "spoken": spoken,
        }
    except Exception as exc:  # contract: a failed pass never breaks the caller
        print(f"[watch] pass failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None
