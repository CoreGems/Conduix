"""Offline tests for speech output (step 14, /v1/audio/speech).

Pure helpers directly; the route through the real SDK + backend.py against
tests/fake_app_server.py (which speaks the thread/realtime methods), with a
fake WebRTC peer standing in for aiortc.
"""
import array
import io
import math
import time
import wave

import openai
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from conduix import voice
from conduix.backend import (
    Backend,
    RealtimeAnswer,
    RealtimeClosed,
    RealtimeError,
    RealtimeStarted,
    RealtimeTranscript,
    map_realtime,
)
from conduix.config import settings
from tests.test_backend_fake_server import FAKE, stats

TONE = array.array("h", [int(8000 * math.sin(2 * math.pi * 440 * i / voice.SAMPLE_RATE))
                         for i in range(voice.SAMPLE_RATE // 5)]).tobytes()  # 200 ms


class FakePeer:
    """Stands in for AiortcPeer: every turn "records" a 200 ms tone."""
    instances: list["FakePeer"] = []
    silent = False

    def __init__(self):
        self.answer = None
        self.closed = False
        FakePeer.instances.append(self)

    async def offer(self):
        return "fake-offer"

    async def accept(self, sdp):
        self.answer = sdp

    async def wait_ready(self):
        assert self.answer is not None

    def now(self):
        return time.monotonic()

    def audio_between(self, t0, t1):
        return voice.silence(0.1) + (b"" if FakePeer.silent else TONE) + voice.silence(0.1)

    async def close(self):
        self.closed = True


# --- pure helpers --------------------------------------------------------------

def test_map_realtime():
    assert map_realtime("thread/realtime/started", {"threadId": "t"}) == RealtimeStarted()
    assert map_realtime("thread/realtime/sdp", {"sdp": "v=0"}) == RealtimeAnswer("v=0")
    assert map_realtime("thread/realtime/transcript/done",
                        {"role": "assistant", "text": " Hi."}) == RealtimeTranscript("assistant", " Hi.")
    assert map_realtime("thread/realtime/error", {"message": "x"}) == RealtimeError("x")
    assert map_realtime("thread/realtime/closed", {"reason": "requested"}) == RealtimeClosed("requested")
    assert map_realtime("thread/realtime/transcript/delta", {"delta": "H"}) is None


def test_split_text():
    assert voice.split_text("  One.   Two!\nThree? ") == ["One. Two! Three?"]
    long = "Short one. " + "word " * 80 + "end."
    chunks = voice.split_text(long)
    assert all(len(c) <= voice.CHUNK_CHARS for c in chunks) and len(chunks) == 3
    assert " ".join(chunks).split() == long.split()


def test_similarity():
    assert voice.similarity(" Hello from Conduix.", "Hello from Conduix") == 1.0
    assert voice.similarity("Mhm.", "Hello from Conduix. This is a test.") < 0.5


def test_resolve_voice():
    assert voice.resolve_voice("juniper") == "juniper"
    assert voice.resolve_voice("alloy") == settings().speech_voice
    assert voice.resolve_voice(None) == settings().speech_voice
    with pytest.raises(voice.APIError) as ei:
        voice.resolve_voice("nobody")
    assert ei.value.param == "voice" and "juniper" in ei.value.message


def test_trim_silence():
    pcm = voice.silence(0.5) + TONE + voice.silence(0.5)
    trimmed = voice.trim_silence(pcm)
    assert len(TONE) <= len(trimmed) < len(TONE) + 2 * voice.SAMPLE_RATE * 2 * 0.15
    assert voice.trim_silence(voice.silence(1)) == b""


def test_encode_wav():
    with wave.open(io.BytesIO(voice.encode(TONE, "wav"))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, voice.SAMPLE_RATE)
        assert w.readframes(w.getnframes()) == TONE


@pytest.mark.skipif(not voice.available(), reason="needs conduix[voice]")
@pytest.mark.parametrize("fmt", ["mp3", "opus", "aac", "flac"])
def test_encode_compressed(fmt):
    import av

    with av.open(io.BytesIO(voice.encode(TONE, fmt))) as c:
        assert sum(f.samples for f in c.decode(audio=0)) > 0


def test_prompt_hints():
    p = voice._prompt("cheerful", 0.5)
    assert "How to speak: cheerful" in p and "very slowly" in p
    assert voice._prompt(None, 1.0) == voice.PROMPT


async def test_failure_without_full_window_is_502(monkeypatch):
    async def no_limits():
        return {"primary": {"used_percent": 3, "resets_at": 1}}
    monkeypatch.setattr(voice.backend, "rate_limits", no_limits)
    err = await voice._failure('{"error": {"message": "Invalid SDP offer.", "code": "x"}}')
    assert (err.status, err.code) == (502, "speech_failed") and "Invalid SDP offer." in err.message


# --- the route, against the fake app-server --------------------------------------

@pytest.fixture
def speech(monkeypatch):
    from conduix.app import app
    from conduix.sessions import SessionManager

    fresh = Backend(launch_args=FAKE)
    for mod in ("conduix.app", "conduix.sessions", "conduix.routes.responses",
                "conduix.routes.chat", "conduix.routes.models", "conduix.routes.images",
                "conduix.voice"):
        monkeypatch.setattr(f"{mod}.backend", fresh)
    for mod in ("conduix.app", "conduix.routes.responses", "conduix.routes.sessions"):
        monkeypatch.setattr(f"{mod}.manager", SessionManager())
    monkeypatch.setattr(settings(), "speech", True)
    monkeypatch.setattr(voice, "AiortcPeer", FakePeer)
    monkeypatch.setattr(voice, "available", lambda: True)
    monkeypatch.setattr(voice, "TIMING", voice.Timing(startup=5, settle=0, turn_base=5,
                                                      turn_per_char=0, tail=0, gap=0.1))
    FakePeer.instances, FakePeer.silent = [], False
    with TestClient(app) as http:
        client = OpenAI(base_url="http://testserver/v1", api_key="x", http_client=http,
                        max_retries=0)
        yield client, fresh, http


def stats_sync(http, be: Backend) -> dict:
    """fake/stats, run on the app's event loop (the TestClient's portal)."""
    return http.portal.call(stats, be)


def test_speech_wav(speech):
    client, be, http = speech
    r = client.audio.speech.with_raw_response.create(
        model="tts-1", voice="juniper", input="Hello there. Second sentence.",
        response_format="wav")
    assert r.headers["content-type"] == "audio/wav" and r.headers["x-conduix-voice"] == "juniper"
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getframerate() == voice.SAMPLE_RATE and w.getnframes() >= len(TONE) // 2
    peer = FakePeer.instances[-1]
    assert peer.answer == "fake-answer:fake-offer" and peer.closed
    assert be._realtime == {}  # the session unregistered itself


def test_speech_reaches_codex_as_v3_webrtc(speech, monkeypatch):
    client, be, http = speech
    seen = {}
    real_open = be.open_realtime

    async def spy(**kw):
        session = await real_open(**kw)
        seen["thread"] = session.thread_id
        return session
    monkeypatch.setattr(be, "open_realtime", spy)
    client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input="One. Two.",
                               instructions="calm", response_format="pcm")
    s = stats_sync(http, be)
    start = s["realtime_start"][-1]
    assert start["version"] == "v3" and start["outputModality"] == "audio"
    assert start["transport"] == {"type": "webrtc", "sdp": "fake-offer"}
    assert start["voice"] == settings().speech_voice and "calm" in start["prompt"]
    assert start["includeStartupContext"] is False
    assert s["realtime_text"][-1] == voice.SAY + "One. Two."
    assert seen["thread"] in s["realtime_stopped"] and seen["thread"] in s["unsubscribed"]
    params = s["thread_start_params"][-1]
    assert params["ephemeral"] is True and params["config"]["features.shell_tool"] is False


def test_long_input_is_spoken_in_chunks(speech):
    client, be, http = speech
    text = " ".join(f"Sentence number {i} is here." for i in range(20))
    pcm = client.audio.speech.create(model="tts-1", voice="cove", input=text,
                                     response_format="pcm").content
    texts = stats_sync(http, be)["realtime_text"]
    assert len(texts) == len(voice.split_text(text)) > 1
    assert len(pcm) >= len(texts) * len(TONE)


def test_mismatch_retries_once(speech):
    client, be, http = speech
    client.audio.speech.create(model="tts-1", voice="cove", input="ONCE then right",
                               response_format="pcm")
    assert len(stats_sync(http, be)["realtime_text"]) == 2


def test_mismatch_twice_is_502(speech):
    client, be, http = speech
    with pytest.raises(openai.InternalServerError) as ei:
        client.audio.speech.create(model="tts-1", voice="cove", input="MUMBLE please",
                                   response_format="pcm")
    assert ei.value.status_code == 502 and ei.value.body["code"] == "speech_mismatch"
    assert len(stats_sync(http, be)["realtime_text"]) == 2


def test_no_audio_is_502(speech):
    client, _, _ = speech
    FakePeer.silent = True
    with pytest.raises(openai.InternalServerError) as ei:
        client.audio.speech.create(model="tts-1", voice="cove", input="Hello",
                                   response_format="pcm")
    assert ei.value.body["code"] == "speech_mismatch"


def test_realtime_error_with_full_window_is_429(speech):
    client, _, _ = speech  # the fake reports the usage window at 100%
    with pytest.raises(openai.RateLimitError) as ei:
        client.audio.speech.create(model="tts-1", voice="cove", input="RTERROR",
                                   response_format="pcm")
    assert ei.value.body["code"] == "usage_limit_exceeded"
    assert ei.value.response.headers["x-should-retry"] == "false"


def test_bad_requests(speech):
    client, _, _ = speech
    for kw, param in [({"voice": "nobody"}, "voice"), ({"response_format": "ogg"}, "response_format"),
                      ({"speed": 9.0}, "speed"), ({"input": " "}, "input"),
                      ({"input": "x" * 5000}, "input")]:
        args = {"model": "tts-1", "voice": "cove", "input": "Hi", **kw}
        with pytest.raises(openai.BadRequestError) as ei:
            client.audio.speech.create(**args)
        assert ei.value.body["param"] == param
    with pytest.raises(openai.BadRequestError) as ei:
        client.audio.speech.create(model="tts-1", voice="cove", input="Hi", stream_format="sse")
    assert ei.value.body["param"] == "stream_format"
    assert FakePeer.instances == []  # rejected before any session


def test_disabled_is_501(speech, monkeypatch):
    client, _, _ = speech
    monkeypatch.setattr(settings(), "speech", False)
    with pytest.raises(openai.APIStatusError) as ei:
        client.audio.speech.create(model="tts-1", voice="cove", input="Hi")
    assert ei.value.status_code == 501 and ei.value.body["code"] == "speech_disabled"


def test_missing_aiortc_is_501(speech, monkeypatch):
    client, _, _ = speech
    monkeypatch.setattr(voice, "available", lambda: False)
    with pytest.raises(openai.APIStatusError) as ei:
        client.audio.speech.create(model="tts-1", voice="cove", input="Hi")
    assert ei.value.status_code == 501 and ei.value.body["code"] == "speech_unavailable"
