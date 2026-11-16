"""Offline tests for transcription (step 15, /v1/audio/transcriptions).

Pure helpers directly; the route through the real SDK + backend.py against
tests/fake_app_server.py, with a fake WebRTC peer that "plays" the upload by
telling the fake server what to hear (fake/realtime/hear).
"""
import array
import math
import time

import openai
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from openai.types.audio import Transcription, TranscriptionVerbose
from pydantic import BaseModel, ConfigDict

from conduix import voice
from conduix.backend import Backend
from conduix.config import settings
from conduix.voice import Span
from tests.test_backend_fake_server import FAKE, stats

TONE = array.array("h", [int(8000 * math.sin(2 * math.pi * 440 * i / voice.SAMPLE_RATE))
                         for i in range(voice.SAMPLE_RATE)]).tobytes()  # 1 s at 24 kHz
WAV = voice.encode(TONE, "wav")


class _Any(BaseModel):
    model_config = ConfigDict(extra="allow")


class FakeListenPeer:
    """Stands in for AiortcPeer(source=...): playing makes the fake server
    hear `heard` (and, with `data_channel`, reports the turn and its words
    the way the real data channel does)."""
    backend: Backend
    heard = "Hello there, ChatGPT world."
    data_channel = True
    lag_ms = 0  # how late the "recognizer" stamps words (live: ~700 ms)
    instances: list["FakeListenPeer"] = []

    def __init__(self, source=None):
        self.source = source
        self.heard_turns, self.heard_words = [], []
        self.open_user_turns = 0
        self.played_at = None
        self.offset_ms = 0.0
        self.closed = False
        FakeListenPeer.instances.append(self)

    async def offer(self):
        return "fake-offer"

    async def accept(self, sdp):
        self.answer = sdp

    async def wait_ready(self):
        pass

    async def play(self):
        text = FakeListenPeer.heard
        if FakeListenPeer.data_channel and text and "RTERROR" not in text:
            lag = FakeListenPeer.lag_ms
            self.heard_turns.append(Span(100 + lag, 900 + lag, " " + text))
            pieces = [" Hello", " there", ",", " Chat", "GPT", " world", "."]
            self.heard_words = [Span(100 * i + lag, 100 * i + 100 + lag, p)
                                for i, p in enumerate(pieces)]
        await self.backend.codex._client.request("fake/realtime/hear", {"text": text},
                                                 response_model=_Any)
        self.played_at = time.monotonic()

    async def close(self):
        self.closed = True


# --- pure helpers --------------------------------------------------------------

def test_merge_words():
    pieces = [Span(0, 200, " Hi"), Span(200, 400, ", this"), Span(400, 600, " Chat"),
              Span(600, 800, "GPT"), Span(800, 1000, " plan.")]
    assert [(w.text, w.start_ms, w.end_ms) for w in voice.merge_words(pieces)] == [
        ("Hi", 0, 400), ("this", 0, 400), ("ChatGPT", 400, 800), ("plan", 800, 1000)]


def test_subtitles():
    tr = voice.Transcript("One. Two.", 3.0, [Span(0, 1500, "One."), Span(1500, 3000, "Two.")], [])
    assert voice.subtitles(tr, "srt") == (
        "1\n00:00:00,000 --> 00:00:01,500\nOne.\n\n2\n00:00:01,500 --> 00:00:03,000\nTwo.\n")
    assert voice.subtitles(tr, "vtt").startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nOne.")


def test_shift_clamps_to_the_audio():
    spans = voice._shift([Span(300, 900, "x"), Span(5000, 6000, "y")], 200, 1000)
    assert [(s.start_ms, s.end_ms) for s in spans] == [(100, 700), (1000, 1000)]


def test_speech_bounds():
    pcm48 = voice.decode_audio(voice.encode(voice.silence(0.5) + TONE + voice.silence(0.5), "wav"))
    start, end = voice.speech_bounds(pcm48)
    assert 490 <= start <= 510 and 1590 <= end <= 1620  # 100 ms pad after the speech
    assert voice.speech_bounds(bytes(9600)) == (None, None)


def test_listen_prompt_hints():
    p = voice._listen_prompt("uk", "Conduix, WebRTC")
    assert p.startswith(voice.LISTEN_PROMPT) and "uk" in p and "Conduix, WebRTC" in p


@pytest.mark.skipif(not voice.available(), reason="needs conduix[voice]")
@pytest.mark.parametrize("fmt", ["wav", "mp3", "opus", "flac"])
def test_decode_audio(fmt):
    pcm = voice.decode_audio(voice.encode(TONE, fmt))
    assert abs(voice.pcm_seconds(pcm) - 1.0) < 0.1  # 48 kHz mono now


@pytest.mark.skipif(not voice.available(), reason="needs conduix[voice]")
def test_decode_audio_rejects_garbage():
    with pytest.raises(voice.APIError) as ei:
        voice.decode_audio(b"not audio at all")
    assert (ei.value.status, ei.value.code, ei.value.param) == (400, "invalid_audio", "file")


# --- the route, against the fake app-server --------------------------------------

@pytest.fixture
def tx(monkeypatch):
    from conduix.app import app
    from conduix.sessions import SessionManager

    fresh = Backend(launch_args=FAKE)
    for mod in ("conduix.app", "conduix.sessions", "conduix.routes.responses",
                "conduix.routes.chat", "conduix.routes.models", "conduix.routes.images",
                "conduix.voice"):
        monkeypatch.setattr(f"{mod}.backend", fresh)
    for mod in ("conduix.app", "conduix.routes.responses", "conduix.routes.sessions"):
        monkeypatch.setattr(f"{mod}.manager", SessionManager())
    monkeypatch.setattr(settings(), "transcription", True)
    monkeypatch.setattr(voice, "AiortcPeer", FakeListenPeer)
    monkeypatch.setattr(voice, "transcription_available", lambda: True)
    monkeypatch.setattr(voice, "TIMING", voice.Timing(startup=5, listen_quiet=0.5,
                                                      listen_extra=5))
    FakeListenPeer.backend, FakeListenPeer.instances = fresh, []
    FakeListenPeer.heard, FakeListenPeer.data_channel = "Hello there, ChatGPT world.", True
    FakeListenPeer.lag_ms = 0
    with TestClient(app) as http:
        client = OpenAI(base_url="http://testserver/v1", api_key="x", http_client=http,
                        max_retries=0)
        yield client, fresh, http


def fake_stats(http, be):
    return http.portal.call(stats, be)


def test_json(tx):
    client, be, http = tx
    r = client.audio.transcriptions.create(model="whisper-1", file=("a.wav", WAV))
    Transcription.model_validate(r.model_dump())
    assert r.text == "Hello there, ChatGPT world."
    assert r.usage.type == "duration" and r.usage.seconds == 1
    peer = FakeListenPeer.instances[-1]
    assert peer.closed and abs(voice.pcm_seconds(peer.source) - 1.0) < 0.1
    s = fake_stats(http, be)
    start = s["realtime_start"][-1]
    assert start["version"] == "v3" and start["transport"]["type"] == "webrtc"
    assert start["prompt"].startswith(voice.LISTEN_PROMPT) and "voice" not in start
    assert start["threadId"] in s["realtime_stopped"] and start["threadId"] in s["unsubscribed"]
    assert be._realtime == {}


def test_language_and_prompt_reach_the_voice(tx):
    client, be, http = tx
    client.audio.transcriptions.create(model="whisper-1", file=("a.wav", WAV), language="en",
                                       prompt="Spell it Conduix.")
    prompt = fake_stats(http, be)["realtime_start"][-1]["prompt"]
    assert "en" in prompt and "Spell it Conduix." in prompt


def test_text(tx):
    client, _, _ = tx
    r = client.audio.transcriptions.create(model="whisper-1", file=("a.wav", WAV),
                                           response_format="text")
    assert r.strip() == "Hello there, ChatGPT world."


def test_verbose_json_with_words(tx):
    client, _, _ = tx
    r = client.audio.transcriptions.create(model="whisper-1", file=("a.wav", WAV),
                                           response_format="verbose_json",
                                           timestamp_granularities=["word", "segment"])
    TranscriptionVerbose.model_validate(r.model_dump())
    assert r.text == "Hello there, ChatGPT world." and r.language == "unknown"
    assert abs(r.duration - 1.0) < 0.1
    assert [(s.start, s.end, s.text) for s in r.segments] == [(0.1, 0.9, "Hello there, ChatGPT world.")]
    assert [w.word for w in r.words] == ["Hello", "there", "ChatGPT", "world"]
    assert (r.words[2].start, r.words[2].end) == (0.3, 0.5)


def test_verbose_json_without_words(tx):
    client, _, _ = tx
    r = client.audio.transcriptions.create(model="whisper-1", file=("a.wav", WAV),
                                           response_format="verbose_json", language="en")
    assert r.words is None and r.language == "en" and len(r.segments) == 1


@pytest.mark.parametrize("fmt,head", [("srt", "1\n00:00:00,100 --> 00:00:00,900\n"),
                                      ("vtt", "WEBVTT\n\n00:00:00.100 --> 00:00:00.900\n")])
def test_subtitle_formats(tx, fmt, head):
    client, _, _ = tx
    r = client.audio.transcriptions.create(model="whisper-1", file=("a.wav", WAV),
                                           response_format=fmt)
    assert r.startswith(head) and "Hello there, ChatGPT world." in r


def test_late_word_stamps_are_calibrated_on_the_audio(tx):
    client, _, _ = tx
    FakeListenPeer.lag_ms = 700
    r = client.audio.transcriptions.create(model="whisper-1", file=("a.wav", WAV),
                                           response_format="verbose_json",
                                           timestamp_granularities=["word"])
    # Speech starts at 0 in the tone, so everything moves back by 700 ms.
    assert (r.words[0].start, r.segments[0].start, r.segments[0].end) == (0.0, 0.1, 0.9)


def test_falls_back_to_codex_transcripts(tx):
    client, _, _ = tx
    FakeListenPeer.data_channel = False
    r = client.audio.transcriptions.create(model="whisper-1", file=("a.wav", WAV))
    assert r.text == "Hello there, ChatGPT world."


def test_silence_is_empty_text(tx):
    client, _, _ = tx
    FakeListenPeer.heard = ""
    r = client.audio.transcriptions.create(model="whisper-1", file=("a.wav", WAV))
    assert r.text == ""


def test_delegated_turn_is_interrupted(tx):
    client, be, http = tx
    FakeListenPeer.heard = "Please DELEGATE this to Codex."
    r = client.audio.transcriptions.create(model="whisper-1", file=("a.wav", WAV))
    assert r.text == "Please DELEGATE this to Codex."
    assert len(fake_stats(http, be)["interrupted"]) == 1
    assert be.blocked_items == {"delegatedTurn": 1}
    assert http.get("/health").json()["blocked_agent_items"] == {"delegatedTurn": 1}


def test_realtime_error_with_full_window_is_429(tx):
    client, _, _ = tx  # the fake reports the usage window at 100%
    FakeListenPeer.heard = "RTERROR"
    with pytest.raises(openai.RateLimitError) as ei:
        client.audio.transcriptions.create(model="whisper-1", file=("a.wav", WAV))
    assert ei.value.body["code"] == "usage_limit_exceeded"


def test_bad_requests(tx, monkeypatch):
    client, _, http = tx
    for kw, param in [({"response_format": "xml"}, "response_format"),
                      ({"timestamp_granularities": ["word"]}, "timestamp_granularities"),
                      ({"response_format": "verbose_json", "timestamp_granularities": ["char"]},
                       "timestamp_granularities")]:
        with pytest.raises(openai.BadRequestError) as ei:
            client.audio.transcriptions.create(model="whisper-1", file=("a.wav", WAV), **kw)
        assert ei.value.body["param"] == param
    r = http.post("/v1/audio/transcriptions", data={"model": "whisper-1", "stream": "true"},
                  files={"file": ("a.wav", WAV)})
    assert r.status_code == 400 and r.json()["error"]["param"] == "stream"
    r = http.post("/v1/audio/transcriptions", data={"model": "whisper-1"},
                  files={"file": ("a.wav", b"garbage")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_audio"
    r = http.post("/v1/audio/transcriptions", data={"model": "whisper-1"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "file"
    r = http.post("/v1/audio/transcriptions", json={"model": "whisper-1"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "file"
    monkeypatch.setattr(settings(), "transcription_max_seconds", 0)
    r = http.post("/v1/audio/transcriptions", data={"model": "whisper-1"},
                  files={"file": ("a.wav", WAV)})
    assert r.status_code == 400 and "limit is 0s" in r.json()["error"]["message"]
    assert FakeListenPeer.instances == []  # rejected before any session


def test_disabled_is_501(tx, monkeypatch):
    client, _, _ = tx
    monkeypatch.setattr(settings(), "transcription", False)
    with pytest.raises(openai.APIStatusError) as ei:
        client.audio.transcriptions.create(model="whisper-1", file=("a.wav", WAV))
    assert ei.value.status_code == 501 and ei.value.body["code"] == "transcription_disabled"


def test_missing_dependencies_is_501(tx, monkeypatch):
    client, _, _ = tx
    monkeypatch.setattr(voice, "transcription_available", lambda: False)
    with pytest.raises(openai.APIStatusError) as ei:
        client.audio.transcriptions.create(model="whisper-1", file=("a.wav", WAV))
    assert ei.value.status_code == 501 and ei.value.body["code"] == "transcription_unavailable"
