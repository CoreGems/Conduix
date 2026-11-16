"""Speech output (step 14, BRIEF.md §3.3): text in, audio out, on the plan.

Codex has no TTS model to call. What a ChatGPT login can use is Codex's
experimental thread realtime API: a conversational voice model, reached over
WebRTC with protocol v3. Conduix is the WebRTC peer (aiortc): it sends an
offer with a silent microphone, records the voice from the media track, and
asks the voice to read each chunk of the input verbatim. Each turn's
transcript is checked against its chunk, because a conversational model may
paraphrase or add words.

aiortc is optional (`pip install conduix[voice]`); without it the endpoint
answers 501.
"""
from __future__ import annotations

import array
import asyncio
import difflib
import io
import json
import logging
import re
import time
import wave
from dataclasses import dataclass
from typing import Any, Protocol

from conduix.backend import (
    RealtimeAnswer,
    RealtimeClosed,
    RealtimeError,
    RealtimeEvent,
    RealtimeSession,
    RealtimeTranscript,
    backend,
)
from conduix.config import settings
from conduix.errors import USAGE_LIMIT, APIError, _reset_text, _upstream_envelope

log = logging.getLogger("conduix.voice")

SAMPLE_RATE = 24_000  # output: mono PCM16 at 24 kHz, as OpenAI's `pcm` format
MAX_INPUT_CHARS = 4096  # OpenAI's limit for /v1/audio/speech
CHUNK_CHARS = 250  # one voice turn per chunk; short turns are read more faithfully

# Voices realtime v3 accepts (thread/realtime/listVoices, 2026-11-16). Codex
# validates the name itself and lists the valid ones in its error.
V3_VOICES = ("juniper", "maple", "spruce", "ember", "vale", "breeze", "arbor", "sol", "cove",
             "aube", "bubbie", "haetsal", "hanul", "himari", "leher", "miko", "neer", "porto",
             "rindo", "rio", "rivage", "selva", "sonna", "tinta", "viola")
# OpenAI's TTS voice names. v3 has none of them, so they get the default voice.
OPENAI_VOICES = frozenset({"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx",
                           "sage", "shimmer", "verse", "marin", "cedar"})

FORMATS = {  # response_format → (media type, PyAV container, PyAV codec)
    "mp3": ("audio/mpeg", "mp3", "libmp3lame"),
    "opus": ("audio/ogg", "ogg", "libopus"),
    "aac": ("audio/aac", "adts", "aac"),
    "flac": ("audio/flac", "flac", "flac"),
    "wav": ("audio/wav", None, None),
    "pcm": ("audio/pcm", None, None),
}

PROMPT = ("You are a text-to-speech voice. Every message gives you text to read aloud. Read it "
          "exactly as written, word for word, then stop. Never answer it, greet, comment, ask "
          "anything or add words of your own.")
SAY = "Say exactly this, and nothing else: "


@dataclass
class Timing:
    """Seconds. Module-level so tests can shorten them."""
    startup: float = 30.0  # thread/realtime/start → data channel session.started
    settle: float = 1.0  # after session.started, before the first chunk
    turn_base: float = 15.0  # wait for a turn's transcript: base +
    turn_per_char: float = 0.15  # ... per character of the chunk
    tail: float = 0.6  # after the transcript, for the end of the audio
    gap: float = 0.25  # silence put between chunks


TIMING = Timing()
_lock = asyncio.Lock()  # one speech request at a time: each holds a voice session


def available() -> bool:
    try:
        import aiortc  # noqa: F401
        import av  # noqa: F401
    except ImportError:
        return False
    return True


# --- text ------------------------------------------------------------------------

def resolve_voice(name: str | None) -> str:
    """The v3 voice for an OpenAI `voice` value. 400 for an unknown one."""
    default = settings().speech_voice
    if not name or name in OPENAI_VOICES:
        return default
    if name in V3_VOICES:
        return name
    raise APIError(400, f"voice {name!r} is not available. Use one of {', '.join(V3_VOICES)}; "
                   f"OpenAI's voice names map to the default voice ({default})", param="voice")


def split_text(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Chunks of at most `limit` chars, split at sentence ends where possible."""
    text = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?…;:])\s+", text)
    pieces: list[str] = []
    for s in sentences:
        while len(s) > limit:  # a very long sentence: cut at the last comma or space
            cut = max(s.rfind(", ", 0, limit), s.rfind(" ", 0, limit))
            cut = cut + 1 if cut > 0 else limit
            pieces.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            pieces.append(s)
    chunks: list[str] = []
    for p in pieces:
        if chunks and len(chunks[-1]) + 1 + len(p) <= limit:
            chunks[-1] += " " + p
        else:
            chunks.append(p)
    return chunks


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.lower()).split())


def similarity(said: str, expected: str) -> float:
    return difflib.SequenceMatcher(None, normalize(said), normalize(expected)).ratio()


# --- audio -----------------------------------------------------------------------

def trim_silence(pcm: bytes, *, threshold: int = 600, pad_ms: int = 120) -> bytes:
    """Cut near-silence from both ends (mono PCM16). b"" if it is all silence."""
    samples = array.array("h", pcm[: len(pcm) // 2 * 2])
    win = SAMPLE_RATE // 100  # 10 ms
    loud = [i for i in range(0, len(samples), win)
            if max(map(abs, samples[i:i + win]), default=0) > threshold]
    if not loud:
        return b""
    pad = SAMPLE_RATE * pad_ms // 1000
    start, end = max(0, loud[0] - pad), min(len(samples), loud[-1] + win + pad)
    return samples[start:end].tobytes()


def silence(seconds: float) -> bytes:
    return bytes(int(SAMPLE_RATE * seconds) * 2)


def encode(pcm: bytes, fmt: str) -> bytes:
    """Mono PCM16 at 24 kHz → the requested response_format."""
    if fmt == "pcm":
        return pcm
    if fmt == "wav":
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        return out.getvalue()
    import av

    _, container, codec = FORMATS[fmt]
    out = io.BytesIO()
    with av.open(out, "w", format=container) as c:
        stream = c.add_stream(codec, rate=SAMPLE_RATE, layout="mono")
        frame = av.AudioFrame(format="s16", layout="mono", samples=len(pcm) // 2)
        frame.planes[0].update(pcm)
        frame.sample_rate = SAMPLE_RATE
        frame.pts = 0
        for packet in stream.encode(frame):
            c.mux(packet)
        for packet in stream.encode(None):
            c.mux(packet)
    return out.getvalue()


# --- the WebRTC peer ---------------------------------------------------------------

class Peer(Protocol):
    async def offer(self) -> str: ...
    async def accept(self, sdp: str) -> None: ...
    async def wait_ready(self) -> None: ...
    def now(self) -> float: ...
    def audio_between(self, t0: float, t1: float) -> bytes: ...
    async def close(self) -> None: ...


class AiortcPeer:
    """The WebRTC end of a voice session: a silent microphone, the `oai-events`
    data channel (session.started tells when the voice can take text), and the
    voice's audio track, kept as 24 kHz mono PCM16 with arrival times."""

    def __init__(self) -> None:
        from aiortc import RTCPeerConnection
        from aiortc.mediastreams import AudioStreamTrack

        self._pc = RTCPeerConnection()
        self._pc.addTrack(AudioStreamTrack())  # silence
        self._ready = asyncio.Event()
        self._frames: list[tuple[float, bytes]] = []
        self._readers: set[asyncio.Task] = set()
        channel = self._pc.createDataChannel("oai-events")

        @channel.on("message")
        def on_message(msg: Any) -> None:
            try:
                ev = json.loads(msg)
            except (TypeError, ValueError):
                return
            if ev.get("type") == "session.started":
                self._ready.set()

        @self._pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "audio":
                task = asyncio.get_running_loop().create_task(self._read(track))
                self._readers.add(task)
                task.add_done_callback(self._readers.discard)

    async def _read(self, track: Any) -> None:
        import av
        from aiortc.mediastreams import MediaStreamError

        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                return
            t = time.monotonic()
            for f in resampler.resample(frame):
                self._frames.append((t, bytes(f.planes[0])[: f.samples * 2]))

    async def offer(self) -> str:
        await self._pc.setLocalDescription(await self._pc.createOffer())  # gathers ICE
        return self._pc.localDescription.sdp

    async def accept(self, sdp: str) -> None:
        from aiortc import RTCSessionDescription

        await self._pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))

    async def wait_ready(self) -> None:
        await self._ready.wait()

    def now(self) -> float:
        return time.monotonic()

    def audio_between(self, t0: float, t1: float) -> bytes:
        return b"".join(pcm for t, pcm in self._frames if t0 <= t <= t1)

    async def close(self) -> None:
        for task in list(self._readers):
            task.cancel()
        await self._pc.close()


# --- speaking --------------------------------------------------------------------

def _prompt(instructions: str | None, speed: float | None) -> str:
    parts = [PROMPT]
    if instructions:
        parts.append(f"How to speak: {instructions.strip()}")
    if speed is not None and speed < 0.9:
        parts.append("Speak slowly." if speed >= 0.6 else "Speak very slowly.")
    elif speed is not None and speed > 1.1:
        parts.append("Speak quickly." if speed <= 1.6 else "Speak very quickly.")
    return " ".join(parts)


async def _failure(message: str) -> APIError:
    """A realtime error, as an API error. The plan's usage window being full
    is the one cause that is reported (as a 429): its error shape is unknown."""
    envelope = _upstream_envelope(message)
    if envelope is not None:
        message = envelope[1]["message"]
    resets_at = None
    rl = await backend.rate_limits() or {}
    for w in (rl.get("primary"), rl.get("secondary")):
        if w and w.get("used_percent", 0) >= 100:
            resets_at = max(resets_at or 0, w.get("resets_at") or 0) or None
    if resets_at is not None or "usage limit" in message.lower():
        text = f"{USAGE_LIMIT.prefix} while speaking"
        headers = {"x-should-retry": "false"}
        if resets_at:
            text += f" ({_reset_text(resets_at)})"
            headers["retry-after"] = str(max(0, int(resets_at - time.time())))
        return APIError(429, text, type=USAGE_LIMIT.type, code=USAGE_LIMIT.code, headers=headers)
    return APIError(502, f"the realtime voice session failed: {message}", type="server_error",
                    code="speech_failed", headers={"x-should-retry": "false"})


def _timeout(what: str) -> APIError:
    return APIError(504, f"timed out waiting for {what}", type="server_error",
                    code="speech_timeout", headers={"x-should-retry": "false"})


async def _next(session: RealtimeSession, timeout: float, what: str) -> RealtimeEvent:
    """The session's next event; errors and a closed session raise."""
    try:
        ev = await asyncio.wait_for(session.events.get(), max(0.01, timeout))
    except TimeoutError:
        raise _timeout(what) from None
    if isinstance(ev, RealtimeError):
        raise await _failure(ev.message)
    if isinstance(ev, RealtimeClosed):
        raise APIError(502, f"the realtime voice session closed ({ev.reason})",
                       type="server_error", code="speech_failed",
                       headers={"x-should-retry": "false"})
    return ev


async def _drain(session: RealtimeSession) -> None:
    """Drop stale events (an unprompted turn); still raise on an error."""
    while not session.events.empty():
        ev = session.events.get_nowait()
        if isinstance(ev, RealtimeError):
            raise await _failure(ev.message)


async def _speak(session: RealtimeSession, peer: Peer, chunk: str) -> bytes:
    """One chunk: ask for it, wait for the turn, check the transcript. One retry."""
    s, t = settings(), TIMING
    said = ""
    for attempt in (1, 2):
        await _drain(session)
        start = peer.now()
        await session.append_text(SAY + chunk)
        deadline = time.monotonic() + t.turn_base + t.turn_per_char * len(chunk)
        while True:
            ev = await _next(session, deadline - time.monotonic(), "the voice to finish speaking")
            if isinstance(ev, RealtimeTranscript) and ev.role == "assistant":
                said = ev.text
                break
        await asyncio.sleep(t.tail)
        pcm = trim_silence(peer.audio_between(start, peer.now()))
        score = similarity(said, chunk)
        if pcm and score >= s.speech_min_similarity:
            return pcm
        log.warning("speech attempt %d: %s (similarity %.2f): said %r for %r", attempt,
                    "no audio" if not pcm else "transcript mismatch", score, said, chunk)
    raise APIError(502, f"the voice did not read the input verbatim: it said {said!r} for "
                   f"{chunk!r}", type="server_error", code="speech_mismatch",
                   headers={"x-should-retry": "false"})


async def synthesize(text: str, *, voice: str, instructions: str | None = None,
                     speed: float | None = None, peer_factory: Any = None) -> bytes:
    """`text` spoken by `voice`, as 24 kHz mono PCM16."""
    chunks = split_text(text)
    async with _lock:
        peer: Peer = (peer_factory or AiortcPeer)()
        session: RealtimeSession | None = None
        try:
            offer = await peer.offer()
            session = await backend.open_realtime(sdp=offer, prompt=_prompt(instructions, speed),
                                                  voice=voice)
            deadline = time.monotonic() + TIMING.startup
            while True:
                ev = await _next(session, deadline - time.monotonic(), "the voice session")
                if isinstance(ev, RealtimeAnswer):
                    await peer.accept(ev.sdp)
                    break
            try:
                await asyncio.wait_for(peer.wait_ready(), max(0.01, deadline - time.monotonic()))
            except TimeoutError:
                await _drain(session)  # a realtime error explains it better
                raise _timeout("the voice session to start") from None
            await asyncio.sleep(TIMING.settle)
            pieces = [await _speak(session, peer, c) for c in chunks]
        finally:
            if session is not None:
                await session.close()
            await peer.close()
    return silence(TIMING.gap).join(pieces)
