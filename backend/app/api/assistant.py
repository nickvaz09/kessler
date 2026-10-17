"""Kessler AI endpoints.

POST /assistant/chat         - server-side Claude agentic loop (assistant.service)
POST /assistant/chat/stream  - same loop as SSE: each event is one
                               `data: <json>\\n\\n` frame, media type
                               text/event-stream (event types in
                               service.chat_stream's docstring).
POST /assistant/narrate      - mission-audio caption beats for one
                               conjunction: one structured-outputs
                               messages.create call (no tool loop), parsed
                               from the first text block, cached in-process
                               for 10 minutes per conjunction id.
POST /assistant/tts          - narration audio for one beat via ElevenLabs
                               (server-side ELEVENLABS_API_KEY, never sent
                               to the browser). Synthesized MP3s are cached
                               on disk keyed by (model, voice, text) so a
                               beat is only ever billed once.
GET  /reports/{id}           - serve a generated report; the path component
                               must parse as a UUID (filenames are rebuilt
                               from the parsed value, so traversal is
                               impossible).
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import uuid
from typing import Literal

import httpx

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.app import config, db
from backend.app.api import conjunctions as conjunctions_api
from backend.app.assistant import prompt, service

router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    context: dict | None = None


@router.get("/assistant/tools")
def list_tools():
    """The assistant's tool catalog (names, one-line purpose, family) - no
    key needed; it describes capabilities, it does not call Claude."""
    from backend.app.assistant import tools as tools_mod

    items = []
    for t in tools_mod.TOOLS:
        name = t["name"]
        family = ("ui" if name in tools_mod.UI_ACTION_TOOL_NAMES
                  else "report" if name == "create_report" else "data")
        desc = (t.get("description") or "").strip().split("\n")[0]
        items.append({"name": name, "family": family, "summary": desc[:200]})
    return {"count": len(items), "items": items}


@router.post("/assistant/chat")
def assistant_chat(req: ChatRequest):
    return service.chat(
        [m.model_dump() for m in req.messages], req.context or {}
    )


@router.post("/assistant/chat/stream")
def assistant_chat_stream(req: ChatRequest):
    # Validation + client construction happen inside chat_stream before the
    # generator is returned, so credential/validation failures are plain
    # HTTPExceptions (no bytes sent yet). Mid-stream failures arrive as one
    # {type:'error'} event from the generator, then a clean close.
    events = service.chat_stream(
        [m.model_dump() for m in req.messages], req.context or {}
    )

    def sse():
        for event in events:
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- narration ----------------------------------------------------------------

NARRATE_MAX_TOKENS = 2000
NARRATE_DEFAULT_EFFORT = "medium"   # KESSLER_AI_EFFORT still overrides
NARRATE_WINDOW_S = 1200             # matches the simulator's default window
NARRATE_CACHE_TTL_S = 600.0         # contract: 10-minute in-process cache

MALFORMED_DETAIL = "Kessler AI returned malformed narration JSON"

# Structured-outputs schema (contract): {beats:[{t_rel_s: number, text:
# string}]}. additionalProperties false + required everywhere; the 8-14 beat
# count and the -window..+60 s range are prompt constraints (array
# min/maxItems are unsupported by structured outputs).
NARRATION_BEATS_SCHEMA = {
    "type": "object",
    "properties": {
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "t_rel_s": {"type": "number"},
                    "text": {"type": "string"},
                },
                "required": ["t_rel_s", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["beats"],
    "additionalProperties": False,
}

# conjunction_id -> (monotonic expiry, {"beats", "model"}). Failures are
# never cached; entries are small (8-14 beats) so no eviction is needed.
_narrate_cache: dict[int, tuple[float, dict]] = {}

_EVENT_SUMMARY_KEYS = (
    "id", "tca", "primary", "secondary", "miss_distance_m",
    "relative_velocity_ms", "pc_latest", "predicted_final_risk",
    "model_version", "risk_level", "cdm_count", "status",
)
_EPHEMERIS_SUMMARY_KEYS = ("tca", "window_s", "step_s", "tca_state", "sigma_m")


class NarrateRequest(BaseModel):
    # int64 bounds mirror the /conjunctions path params - an oversized id
    # must 422, not overflow sqlite into a 500
    conjunction_id: int = Field(ge=0, le=conjunctions_api.INT64_MAX)


def _narrate_user_message(event: dict, ephem: dict) -> str:
    """The volatile per-event payload: conjunction summary + ephemeris
    tca_state (grids dropped - the model narrates from the TCA geometry).
    Lives in the user message so NARRATION_SYSTEM_PROMPT stays frozen."""
    event_summary = {k: event.get(k) for k in _EVENT_SUMMARY_KEYS}
    ephem_summary = {k: ephem.get(k) for k in _EPHEMERIS_SUMMARY_KEYS}
    return (
        "Narrate this conjunction encounter as caption beats.\n\n"
        "<event>\n"
        f"{json.dumps(event_summary, default=str)}\n"
        "</event>\n"
        "<ephemeris>\n"
        f"{json.dumps(ephem_summary, default=str)}\n"
        "</ephemeris>"
    )


def _parse_beats(response) -> list[dict]:
    """Beats from a structured-outputs response: the first text content block
    carries valid JSON matching NARRATION_BEATS_SCHEMA. Anything else (refusal,
    no text block, unparseable JSON, wrong shape) degrades to a 502 with a
    detail - never a crash, never a cached failure."""
    if getattr(response, "stop_reason", None) == "refusal":
        raise HTTPException(
            status_code=502, detail="Kessler AI declined to narrate this event."
        )
    text = next(
        (block.text
         for block in (getattr(response, "content", None) or [])
         if getattr(block, "type", None) == "text"
         and getattr(block, "text", None)),
        None,
    )
    if not text:
        raise HTTPException(
            status_code=502, detail=f"{MALFORMED_DETAIL}: no text content"
        )
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"{MALFORMED_DETAIL}: {exc}")
    beats = parsed.get("beats") if isinstance(parsed, dict) else None
    if not isinstance(beats, list) or not beats:
        raise HTTPException(
            status_code=502, detail=f"{MALFORMED_DETAIL}: missing beats array"
        )
    clean = []
    for beat in beats:
        t_rel = beat.get("t_rel_s") if isinstance(beat, dict) else None
        if (isinstance(t_rel, bool) or not isinstance(t_rel, (int, float))
                or not isinstance(beat.get("text"), str)):
            raise HTTPException(
                status_code=502,
                detail=f"{MALFORMED_DETAIL}: bad beat shape",
            )
        clean.append({"t_rel_s": float(t_rel), "text": beat["text"]})
    return clean


def narrate(conjunction_id: int, client=None) -> dict:
    """{beats, model} for one conjunction via one messages.create call.

    No tool loop. Model/effort resolve through the same env override path as
    chat (defaults: chat model, effort 'medium'); the Claude call reuses
    service._create_message so the typed error mapping (503 auth / 429 rate
    limit / 502 API+network) is identical to chat. Successful results are
    cached in-process for 10 minutes per conjunction id - a cache hit
    returns without touching the client or the DB.
    """
    now = time.monotonic()
    cached = _narrate_cache.get(conjunction_id)
    if cached is not None and cached[0] > now:
        return cached[1]

    conn = db.get_conn()
    try:
        # 404 for unknown ids propagates from get_conjunction. The ephemeris
        # endpoint never 500s on bad TLEs - tca_state degrades to nulls.
        event = conjunctions_api.get_conjunction(conjunction_id, conn)
        ephem = conjunctions_api.get_ephemeris(
            conjunction_id=conjunction_id, window_s=NARRATE_WINDOW_S, conn=conn
        )
    finally:
        conn.close()

    if client is None:
        client = service._default_client()
    model = os.environ.get("KESSLER_AI_MODEL", service.DEFAULT_MODEL)
    effort = os.environ.get("KESSLER_AI_EFFORT", NARRATE_DEFAULT_EFFORT)

    request = {
        "model": model,
        "max_tokens": NARRATE_MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        # never pass temperature/top_p/top_k (400 on this model family).
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": NARRATION_BEATS_SCHEMA},
        },
        "system": [
            {
                "type": "text",
                "text": prompt.NARRATION_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {"role": "user", "content": _narrate_user_message(event, ephem)}
        ],
    }
    response = service._create_message(client, request)
    beats = [{**b, "sig": tts_sign(b["text"])} for b in _parse_beats(response)]
    payload = {"beats": beats, "model": model}
    _narrate_cache[conjunction_id] = (now + NARRATE_CACHE_TTL_S, payload)
    return payload


@router.post("/assistant/narrate")
def assistant_narrate(req: NarrateRequest):
    return narrate(req.conjunction_id)


# --- narration voice (ElevenLabs) ---------------------------------------------

TTS_MAX_CHARS = 600                 # beats are one caption line each
TTS_TIMEOUT_S = 30.0
# Quality tier, not turbo - turbo reads flat/robotic for mission audio.
TTS_MODEL_ID = "eleven_multilingual_v2"
# 'Sarah' - a default My-Voices voice, usable on free-tier keys. (Library
# voices like 'Rachel' 402 for free accounts via the API.)
TTS_DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
# Calm, unhurried delivery: mid stability (expressive but steady), no style
# exaggeration, slowed to 0.9 so numbers land clearly.
TTS_VOICE_SETTINGS = {
    "stability": 0.45,
    "similarity_boost": 0.8,
    "style": 0.0,
    "use_speaker_boost": True,
    "speed": 0.9,
}
TTS_NO_KEY_DETAIL = (
    "Voice synthesis unavailable: set ELEVENLABS_API_KEY in the server "
    "environment (free tier at elevenlabs.io works)."
)

# Caption text is terse mission shorthand; the SPOKEN text expands it so the
# voice reads units and notation as words, not letters. Captions on screen
# keep the shorthand - only the synthesized string differs.
_SPOKEN_REWRITES = [
    (re.compile(r"(?<=\d)\s*km/s\b"), " kilometers per second"),
    (re.compile(r"(?<=\d)\s*m/s\b"), " meters per second"),
    (re.compile(r"(?<=\d)\s*km\b"), " kilometers"),
    (re.compile(r"(?<=\d)\s*m\b"), " meters"),
    (re.compile(r"(?<=\d)\s*deg\b"), " degrees"),
    (re.compile(r"\b(\d+(?:\.\d+)?)e-(\d+)\b"),
     r"\1 times ten to the minus \2"),
    (re.compile(r"\bPc\b"), "collision probability"),
    (re.compile(r"±"), "plus or minus"),
]


def spoken_text(text: str) -> str:
    for rx, rep in _SPOKEN_REWRITES:
        text = rx.sub(rep, text)
    return text


# Synthesis is restricted to SERVER-ORIGINATED text: narrate beats and
# watchlog spoken lines carry an HMAC over a per-process secret, and
# /assistant/tts refuses anything unsigned. Without this, a hosted deploy
# is an anonymous TTS proxy burning the server's ElevenLabs key on
# arbitrary client strings (each unique string also cached to disk forever).
_TTS_SECRET = secrets.token_bytes(32)


def tts_sign(text: str) -> str:
    return hmac.new(_TTS_SECRET, text.encode("utf-8"), hashlib.sha256).hexdigest()


TTS_PRUNE_KEEP = 200               # newest cached clips kept on disk


def _prune_tts_cache(d) -> None:
    """Cap data/tts to TTS_PRUNE_KEEP newest clips + drop stale .part files.

    Best-effort: never raises into the request path.
    """
    try:
        clips = sorted(d.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
        for p in clips[:-TTS_PRUNE_KEEP] if len(clips) > TTS_PRUNE_KEEP else []:
            p.unlink(missing_ok=True)
        cutoff = time.time() - 3600
        for p in d.glob("*.part"):
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
    except OSError:
        pass


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=TTS_MAX_CHARS)
    sig: str = Field(min_length=64, max_length=64)


def _tts_cache_path(voice_id: str, text: str) -> "os.PathLike":
    settings = json.dumps(TTS_VOICE_SETTINGS, sort_keys=True)
    key = hashlib.sha256(
        f"{TTS_MODEL_ID}|{voice_id}|{settings}|{text}".encode()
    ).hexdigest()
    return config.DATA_DIR / "tts" / f"{key}.mp3"


@router.post("/assistant/tts")
def assistant_tts(req: TtsRequest):
    if not hmac.compare_digest(tts_sign(req.text), req.sig):
        raise HTTPException(
            status_code=403,
            detail="Text is not server-originated (missing or invalid "
                   "signature) — synthesis is restricted to narration "
                   "beats and watchlog briefs.",
        )
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail=TTS_NO_KEY_DETAIL)
    voice_id = (
        os.environ.get("ELEVENLABS_VOICE_ID", "").strip() or TTS_DEFAULT_VOICE_ID
    )
    speech = spoken_text(req.text)
    path = _tts_cache_path(voice_id, speech)
    if not path.is_file():
        try:
            resp = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": api_key},
                json={
                    "text": speech,
                    "model_id": TTS_MODEL_ID,
                    "voice_settings": TTS_VOICE_SETTINGS,
                },
                timeout=TTS_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Voice synthesis failed: {exc.__class__.__name__}",
            )
        if resp.status_code != 200:
            # status only - the body may describe the account
            raise HTTPException(
                status_code=502,
                detail=f"Voice synthesis failed: ElevenLabs HTTP {resp.status_code}",
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(path)   # atomic publish - concurrent requests never
                            # see a half-written file
        _prune_tts_cache(path.parent)
    return FileResponse(path, media_type="audio/mpeg")


@router.get("/reports/{report_id}")
def get_report(report_id: str):
    try:
        canonical = str(uuid.UUID(report_id))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=404, detail="Report not found")
    path = config.DATA_DIR / "reports" / f"{canonical}.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="text/html")
