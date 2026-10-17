"""Narration voice endpoint tests - hermetic: temp DATA_DIR, httpx.post
monkeypatched (no network, no real ElevenLabs calls). Contract:
docs/CONTRACTS.md 'v9 revisions'.
"""
import types

import pytest
from fastapi.testclient import TestClient

from backend.app import config
from backend.app.api import assistant as assistant_api

MP3_BYTES = b"\xff\xf3ID3-not-really-an-mp3"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


class FakePost:
    """Records calls; serves a canned ElevenLabs response."""

    def __init__(self, status_code=200, content=MP3_BYTES, exc=None):
        self.calls = []
        self.status_code = status_code
        self.content = content
        self.exc = exc

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.exc is not None:
            raise self.exc
        return types.SimpleNamespace(
            status_code=self.status_code, content=self.content
        )


def _signed(text):
    return {"text": text, "sig": assistant_api.tts_sign(text)}


def test_tts_403_on_unsigned_or_foreign_text(client, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    bad = client.post("/assistant/tts",
                      json={"text": "attack string", "sig": "0" * 64})
    assert bad.status_code == 403
    missing = client.post("/assistant/tts", json={"text": "x"})
    assert missing.status_code == 422   # sig is required


def test_tts_503_without_key(client):
    resp = client.post("/assistant/tts", json=_signed("closest approach"))
    assert resp.status_code == 503
    assert "ELEVENLABS_API_KEY" in resp.json()["detail"]


def test_tts_validates_text(client, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    assert client.post("/assistant/tts", json=_signed("")).status_code == 422
    too_long = "x" * (assistant_api.TTS_MAX_CHARS + 1)
    assert client.post("/assistant/tts", json=_signed(too_long)).status_code == 422


def test_spoken_text_expands_mission_shorthand():
    assert assistant_api.spoken_text(
        "Range 812 m, closing at 14.6 km/s. Pc 3.4e-3."
    ) == ("Range 812 meters, closing at 14.6 kilometers per second. "
          "collision probability 3.4 times ten to the minus 3.")
    # words that merely end in a unit letter stay untouched
    assert assistant_api.spoken_text("TCA. Minimum separation 93 m.") == \
        "TCA. Minimum separation 93 meters."
    assert assistant_api.spoken_text("beam") == "beam"


def test_tts_synthesizes_and_caches(client, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    fake = FakePost()
    monkeypatch.setattr(assistant_api.httpx, "post", fake)

    r1 = client.post("/assistant/tts", json=_signed("one kilometre and closing"))
    assert r1.status_code == 200
    assert r1.headers["content-type"].startswith("audio/mpeg")
    assert r1.content == MP3_BYTES
    (url, kwargs) = fake.calls[0]
    assert assistant_api.TTS_DEFAULT_VOICE_ID in url
    assert kwargs["headers"]["xi-api-key"] == "k"
    assert kwargs["json"]["model_id"] == assistant_api.TTS_MODEL_ID
    assert kwargs["json"]["voice_settings"] == assistant_api.TTS_VOICE_SETTINGS

    # same text again: served from data/tts/ without a second upstream call
    r2 = client.post("/assistant/tts", json=_signed("one kilometre and closing"))
    assert r2.status_code == 200 and r2.content == MP3_BYTES
    assert len(fake.calls) == 1
    cached = list((config.DATA_DIR / "tts").glob("*.mp3"))
    assert len(cached) == 1 and cached[0].read_bytes() == MP3_BYTES


def test_tts_voice_override_changes_cache_key(client, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "customvoice123")
    fake = FakePost()
    monkeypatch.setattr(assistant_api.httpx, "post", fake)
    assert client.post("/assistant/tts", json=_signed("beat")).status_code == 200
    assert "customvoice123" in fake.calls[0][0]
    assert assistant_api._tts_cache_path("customvoice123", "beat") != (
        assistant_api._tts_cache_path(assistant_api.TTS_DEFAULT_VOICE_ID, "beat")
    )


def test_tts_upstream_error_is_502_without_body(client, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    fake = FakePost(status_code=401, content=b'{"detail":"account info"}')
    monkeypatch.setattr(assistant_api.httpx, "post", fake)
    resp = client.post("/assistant/tts", json=_signed("beat"))
    assert resp.status_code == 502
    assert "401" in resp.json()["detail"]
    assert "account info" not in resp.text
    assert not list((config.DATA_DIR / "tts").glob("*"))   # failures never cached


def test_tts_network_error_is_502(client, monkeypatch):
    import httpx as httpx_mod

    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    fake = FakePost(exc=httpx_mod.ConnectError("boom"))
    monkeypatch.setattr(assistant_api.httpx, "post", fake)
    resp = client.post("/assistant/tts", json=_signed("beat"))
    assert resp.status_code == 502
    assert "ConnectError" in resp.json()["detail"]
