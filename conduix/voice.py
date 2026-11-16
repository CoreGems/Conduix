"""Speech output (step 14, BRIEF.md §3.3) and transcription (step 15, §3.4),
on the plan, through one Codex realtime voice session per request.

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
import math
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
    # Transcription: after playback ends, how long without an open user turn
    # before the transcript is final, and how long past the audio's own
    # duration to wait at most.
    listen_quiet: float = 2.0
    listen_extra: float = 30.0


TIMING = Timing()
_lock = asyncio.Lock()  # one speech request at a time: each holds a voice session


def available() -> bool:
    try:
        import aiortc  # noqa: F401
        import av  # noqa: F401
    except ImportError:
        return False
    return True


def transcription_available() -> bool:
    """Speech's dependencies plus python-multipart (for the upload)."""
    try:
        import python_multipart  # noqa: F401
    except ImportError:
        return False
    return available()


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

INPUT_RATE, INPUT_FRAME = 48_000, 960  # what the microphone track sends: 20 ms frames


@dataclass
class Span:
    """A piece of what the voice heard, on the session's timeline (ms)."""
    start_ms: int
    end_ms: int
    text: str


class Peer(Protocol):
    async def offer(self) -> str: ...
    async def accept(self, sdp: str) -> None: ...
    async def wait_ready(self) -> None: ...
    def now(self) -> float: ...
    def audio_between(self, t0: float, t1: float) -> bytes: ...
    async def close(self) -> None: ...


class ListeningPeer(Peer, Protocol):
    """A peer whose microphone plays audio in (transcription)."""
    heard_turns: list[Span]  # finished user turns (data channel turn.done)
    heard_words: list[Span]  # input_transcript.added pieces
    open_user_turns: int
    played_at: float | None  # when playback ended (monotonic), None before
    offset_ms: float  # session timeline at the moment playback started

    async def play(self) -> None: ...


def _source_track(pcm: bytes) -> Any:
    """A microphone track: silence until play(), then `pcm` (48 kHz mono
    PCM16) in real time, then silence. Audio sent before the session is
    listening is lost (BRIEF.md §3.4), hence the wait."""
    import fractions

    import av
    from aiortc.mediastreams import MediaStreamTrack

    class SourceTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            self.playing = False
            self.finished_at: float | None = None
            self._pos = 0
            self._start: float | None = None
            self._ts = 0

        async def recv(self) -> Any:
            if self._start is None:
                self._start = time.monotonic()
            else:
                self._ts += INPUT_FRAME
                wait = self._start + self._ts / INPUT_RATE - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
            size = INPUT_FRAME * 2
            chunk = b""
            if self.playing:
                chunk = pcm[self._pos: self._pos + size]
                self._pos += size
                if self._pos >= len(pcm) and self.finished_at is None:
                    self.finished_at = time.monotonic()
            frame = av.AudioFrame(format="s16", layout="mono", samples=INPUT_FRAME)
            frame.planes[0].update(chunk + bytes(size - len(chunk)))
            frame.sample_rate = INPUT_RATE
            frame.pts = self._ts
            frame.time_base = fractions.Fraction(1, INPUT_RATE)
            return frame

    return SourceTrack()


class AiortcPeer:
    """The WebRTC end of a voice session: a microphone (silent, or playing
    `source` for transcription), the `oai-events` data channel
    (session.started tells when the voice can take input; user turns and
    their words are what it heard), and the voice's audio track, kept as
    24 kHz mono PCM16 with arrival times."""

    def __init__(self, source: bytes | None = None) -> None:
        from aiortc import RTCPeerConnection
        from aiortc.mediastreams import AudioStreamTrack

        self._pc = RTCPeerConnection()
        self._source = _source_track(source) if source is not None else None
        self._pc.addTrack(self._source or AudioStreamTrack())  # else silence
        self._ready = asyncio.Event()
        self._session_started_at = 0.0
        self._frames: list[tuple[float, bytes]] = []
        self._readers: set[asyncio.Task] = set()
        self.heard_turns: list[Span] = []
        self.heard_words: list[Span] = []
        self.open_user_turns = 0
        self.offset_ms = 0.0
        channel = self._pc.createDataChannel("oai-events")

        @channel.on("message")
        def on_message(msg: Any) -> None:
            try:
                ev = json.loads(msg)
            except (TypeError, ValueError):
                return
            self._on_event(ev)

        @self._pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "audio" and self._source is None:  # a listener ignores replies
                task = asyncio.get_running_loop().create_task(self._read(track))
                self._readers.add(task)
                task.add_done_callback(self._readers.discard)

    def _on_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("type")
        if kind == "session.started":
            self._session_started_at = time.monotonic()
            self._ready.set()
            return
        turn = ev.get("turn") if isinstance(ev.get("turn"), dict) else {}
        if kind == "turn.created" and turn.get("role") == "user":
            self.open_user_turns += 1
        elif kind == "turn.done" and turn.get("role") == "user":
            self.open_user_turns = max(0, self.open_user_turns - 1)
            self.heard_turns.append(Span(turn.get("start_ms", 0), turn.get("end_ms", 0),
                                         turn.get("transcript", "")))
        elif kind == "input_transcript.added":
            item = ev.get("item") if isinstance(ev.get("item"), dict) else {}
            self.heard_words.append(Span(ev.get("start_ms", 0), ev.get("end_ms", 0),
                                         item.get("text", "")))

    async def play(self) -> None:
        if self._source is not None:
            # The data channel's timeline starts when audio starts flowing
            # (found live: measuring from session.started put every word
            # ~0.9 s late), so measure from the track's first frame.
            origin = self._source._start or self._session_started_at
            self.offset_ms = (time.monotonic() - origin) * 1000
            self._source.playing = True

    @property
    def played_at(self) -> float | None:
        return self._source.finished_at if self._source is not None else None

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


# --- transcription (step 15, BRIEF.md §3.4) ---------------------------------------

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # OpenAI's limit for /v1/audio/transcriptions

LISTEN_PROMPT = ("You are a transcription microphone. Only listen. Never reply, answer, greet, "
                 "comment, delegate or hand anything off.")


def decode_audio(data: bytes) -> bytes:
    """Any audio PyAV can read → 48 kHz mono PCM16, the microphone's format.
    400 `invalid_audio` if it can't be decoded or holds no audio."""
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=INPUT_RATE)
    pcm = bytearray()
    try:
        with av.open(io.BytesIO(data)) as c:
            if not c.streams.audio:
                raise ValueError("no audio stream")
            for frame in c.decode(audio=0):
                for f in resampler.resample(frame):
                    pcm += bytes(f.planes[0])[: f.samples * 2]
        for f in resampler.resample(None):
            pcm += bytes(f.planes[0])[: f.samples * 2]
    except (av.FFmpegError, ValueError, IndexError) as e:
        raise APIError(400, f"could not decode the audio file: {e}", param="file",
                       code="invalid_audio") from None
    if not pcm:
        raise APIError(400, "the audio file holds no audio", param="file", code="invalid_audio")
    return bytes(pcm)


def pcm_seconds(pcm: bytes) -> float:
    return len(pcm) / 2 / INPUT_RATE


@dataclass
class Transcript:
    text: str
    duration: float  # seconds of audio
    segments: list[Span]  # one per user turn, ms from the start of the audio
    words: list[Span]


def _listen_prompt(language: str | None, prompt: str | None) -> str:
    parts = [LISTEN_PROMPT]
    if language:
        parts.append(f"The audio is in this language: {language}.")
    if prompt:
        parts.append(f"Context and spellings for what you will hear: {prompt.strip()}")
    return " ".join(parts)


def _shift(spans: list[Span], offset_ms: float, duration_ms: int) -> list[Span]:
    """Session timeline → time in the audio (clamped to it)."""
    def at(ms: float) -> int:
        return int(min(max(0.0, ms - offset_ms), duration_ms))
    return [Span(at(s.start_ms), at(s.end_ms), s.text) for s in spans]


def merge_words(pieces: list[Span]) -> list[Span]:
    """Transcript pieces (" Hi", ", this", " Chat", "GPT") → words without
    punctuation. A piece not starting with a space continues the last word."""
    words: list[Span] = []
    for p in pieces:
        if words and p.text and not p.text[0].isspace():
            last = words[-1]
            words[-1] = Span(last.start_ms, p.end_ms, last.text + p.text)
        else:
            words.append(Span(p.start_ms, p.end_ms, p.text))
    out = []
    for w in words:
        for token in w.text.split():
            token = token.strip(".,!?;:\"'()[]…—-")
            if token:
                out.append(Span(w.start_ms, w.end_ms, token))
    return out


async def transcribe(pcm: bytes, *, language: str | None = None, prompt: str | None = None,
                     peer_factory: Any = None) -> Transcript:
    """What a realtime voice session hears when `pcm` (48 kHz mono PCM16) is
    played into its microphone in real time."""
    duration = pcm_seconds(pcm)
    t = TIMING
    async with _lock:
        peer: ListeningPeer = (peer_factory or AiortcPeer)(source=pcm)
        session: RealtimeSession | None = None
        codex_heard: list[str] = []  # the same turns via Codex (no timings)
        try:
            offer = await peer.offer()
            session = await backend.open_realtime(sdp=offer, prompt=_listen_prompt(language, prompt))
            deadline = time.monotonic() + t.startup
            while True:
                ev = await _next(session, deadline - time.monotonic(), "the voice session")
                if isinstance(ev, RealtimeAnswer):
                    await peer.accept(ev.sdp)
                    break
            try:
                await asyncio.wait_for(peer.wait_ready(), max(0.01, deadline - time.monotonic()))
            except TimeoutError:
                await _drain(session)
                raise _timeout("the voice session to start") from None
            await peer.play()
            give_up = time.monotonic() + duration + t.listen_extra
            while True:
                played = peer.played_at
                if (played is not None and time.monotonic() - played >= t.listen_quiet
                        and peer.open_user_turns == 0
                        and len(codex_heard) >= len(peer.heard_turns)):
                    break
                if time.monotonic() > give_up:
                    log.warning("transcription gave up: played=%s open_user_turns=%d "
                                "data-channel turns=%d codex transcripts=%d words=%d",
                                played is not None, peer.open_user_turns,
                                len(peer.heard_turns), len(codex_heard), len(peer.heard_words))
                    if played is None or (not peer.heard_turns and not codex_heard):
                        raise _timeout("the transcript")
                    log.warning("transcription: a user turn was still open %.0fs after the "
                                "audio ended; returning what was heard", t.listen_extra)
                    break
                try:
                    ev = await asyncio.wait_for(session.events.get(), 0.2)
                except TimeoutError:
                    continue
                if isinstance(ev, RealtimeError):
                    raise await _failure(ev.message)
                if isinstance(ev, RealtimeClosed):
                    raise APIError(502, f"the realtime voice session closed ({ev.reason})",
                                   type="server_error", code="speech_failed",
                                   headers={"x-should-retry": "false"})
                if isinstance(ev, RealtimeTranscript) and ev.role == "user":
                    codex_heard.append(ev.text)
        finally:
            if session is not None:
                await session.close()
            await peer.close()

    duration_ms = int(duration * 1000)
    onset, end = speech_bounds(pcm)
    # The data channel stamps words when the recognizer emits them, which
    # lags the audio (~0.7 s live). Calibrate on the audio itself: the first
    # word starts where speech starts.
    first = min((w.start_ms for w in peer.heard_words), default=None)
    if first is None and peer.heard_turns:
        first = peer.heard_turns[0].start_ms
    lag = peer.offset_ms
    if first is not None and onset is not None:
        lag = min(max(peer.offset_ms, first - onset), peer.offset_ms + MAX_LAG_MS)
    if peer.heard_turns:
        segments = _shift(peer.heard_turns, lag, duration_ms)
        if end is not None:  # a turn closes after the trailing silence: end it with the speech
            segments[-1] = Span(segments[-1].start_ms,
                                max(segments[-1].start_ms, min(segments[-1].end_ms, end)),
                                segments[-1].text)
    else:  # no data-channel turns: Codex's transcripts, without timings
        segments = [Span(0, duration_ms, text) for text in codex_heard]
    segments = [Span(s.start_ms, s.end_ms, s.text.strip()) for s in segments if s.text.strip()]
    words = merge_words(_shift(peer.heard_words, lag, duration_ms))
    return Transcript(" ".join(s.text for s in segments), duration, segments, words)


MAX_LAG_MS = 2000  # never shift timings by more than this past the measured offset


def speech_bounds(pcm: bytes, *, threshold: int = 600, pad_ms: int = 100) -> tuple[int | None, int | None]:
    """(start, end) of non-silence in 48 kHz mono PCM16, in ms; (None, None)
    if it is all silence. Coarse: 10 ms windows."""
    samples = array.array("h", pcm[: len(pcm) // 2 * 2])
    win = INPUT_RATE // 100
    loud = [i for i in range(0, len(samples), win)
            if max(map(abs, samples[i:i + win]), default=0) > threshold]
    if not loud:
        return None, None
    to_ms = 1000 / INPUT_RATE
    total = int(len(samples) * to_ms)
    return int(loud[0] * to_ms), min(total, int((loud[-1] + win) * to_ms) + pad_ms)


def _clock(ms: int, sep: str) -> str:
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def subtitles(tr: Transcript, fmt: str) -> str:
    """The segments as SRT or WebVTT."""
    sep = "," if fmt == "srt" else "."
    blocks = []
    for i, s in enumerate(tr.segments, 1):
        timing = f"{_clock(s.start_ms, sep)} --> {_clock(max(s.end_ms, s.start_ms + 1), sep)}"
        blocks.append(f"{i}\n{timing}\n{s.text}" if fmt == "srt" else f"{timing}\n{s.text}")
    body = "\n\n".join(blocks)
    return (body + "\n") if fmt == "srt" else f"WEBVTT\n\n{body}\n"


def verbose_json(tr: Transcript, language: str | None, word_timestamps: bool) -> dict[str, Any]:
    """OpenAI's `verbose_json` (TranscriptionVerbose). Scores the realtime
    session doesn't report are zero."""
    out: dict[str, Any] = {
        "task": "transcribe",
        "language": language or "unknown",
        "duration": round(tr.duration, 3),
        "text": tr.text,
        "segments": [{
            "id": i, "seek": 0, "start": s.start_ms / 1000, "end": s.end_ms / 1000,
            "text": s.text, "tokens": [], "temperature": 0.0, "avg_logprob": 0.0,
            "compression_ratio": 0.0, "no_speech_prob": 0.0,
        } for i, s in enumerate(tr.segments)],
        "usage": usage(tr),
    }
    if word_timestamps:
        out["words"] = [{"word": w.text, "start": w.start_ms / 1000, "end": w.end_ms / 1000}
                        for w in tr.words]
    return out


def usage(tr: Transcript) -> dict[str, Any]:
    return {"type": "duration", "seconds": math.ceil(tr.duration)}
