"""POST /v1/audio/speech and /v1/audio/transcriptions — OpenAI's audio API
over a Codex thread realtime voice session.

Both experimental and off by default (`CONDUIX_SPEECH=1`,
`CONDUIX_TRANSCRIPTION=1`). See conduix/voice.py and BRIEF.md §3.3–3.4.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict

from conduix import voice
from conduix.config import settings
from conduix.errors import APIError

router = APIRouter(prefix="/v1", tags=["audio"])


class SpeechRequest(BaseModel):
    """`model` (tts-1, gpt-4o-mini-tts, ...) is accepted and ignored: the
    voice is always Codex's realtime model."""

    model_config = ConfigDict(extra="allow")

    input: str
    model: str | None = None
    voice: str | dict[str, Any] | None = None
    instructions: str | None = None
    response_format: str = "mp3"
    speed: float | None = None
    stream_format: str | None = None


def _unavailable(message: str, code: str) -> APIError:
    return APIError(501, message, type="server_error", code=code,
                    headers={"x-should-retry": "false"})


@router.post("/audio/speech")
async def create_speech(req: SpeechRequest) -> Response:
    if not settings().speech:
        raise _unavailable("speech output is turned off; start Conduix with CONDUIX_SPEECH=1 "
                           "(experimental)", "speech_disabled")
    if not voice.available():
        raise _unavailable("speech output needs aiortc: pip install 'conduix[voice]'",
                           "speech_unavailable")
    if not req.input.strip():
        raise APIError(400, "input must not be empty", param="input")
    if len(req.input) > voice.MAX_INPUT_CHARS:
        raise APIError(400, f"input is longer than {voice.MAX_INPUT_CHARS} characters",
                       param="input")
    if req.response_format not in voice.FORMATS:
        raise APIError(400, f"response_format must be one of {', '.join(voice.FORMATS)}",
                       param="response_format")
    if req.stream_format not in (None, "audio"):
        raise APIError(400, "streaming speech (stream_format 'sse') is not supported",
                       param="stream_format")
    if req.speed is not None and not 0.25 <= req.speed <= 4.0:
        raise APIError(400, "speed must be between 0.25 and 4.0", param="speed")
    name = req.voice.get("id") if isinstance(req.voice, dict) else req.voice
    v3_voice = voice.resolve_voice(name)

    pcm = await voice.synthesize(req.input, voice=v3_voice, instructions=req.instructions,
                                 speed=req.speed)
    media_type = voice.FORMATS[req.response_format][0]
    return Response(voice.encode(pcm, req.response_format), media_type=media_type,
                    headers={"x-conduix-voice": v3_voice})


TRANSCRIPTION_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")


def _field(form: Any, name: str) -> str | None:
    value = form.get(name)
    return value if isinstance(value, str) and value != "" else None


@router.post("/audio/transcriptions")
async def create_transcription(request: Request) -> Response:
    """Multipart, as OpenAI's: `file` plus form fields. `model`,
    `temperature`, `chunking_strategy` and `include[]` are accepted and
    ignored. The form is parsed here rather than declared, so the server
    still starts without python-multipart."""
    s = settings()
    if not s.transcription:
        raise _unavailable("transcription is turned off; start Conduix with "
                           "CONDUIX_TRANSCRIPTION=1 (experimental)", "transcription_disabled")
    if not voice.transcription_available():
        raise _unavailable("transcription needs aiortc and python-multipart: "
                           "pip install 'conduix[voice]'", "transcription_unavailable")
    if not request.headers.get("content-type", "").startswith("multipart/form-data"):
        raise APIError(400, "send the audio as multipart/form-data with a `file` field",
                       param="file")
    async with request.form() as form:
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            raise APIError(400, "`file` is required: the audio to transcribe", param="file")
        data = await upload.read()
        fmt = _field(form, "response_format") or "json"
        language = _field(form, "language")
        prompt = _field(form, "prompt")
        stream = (_field(form, "stream") or "false").lower() == "true"
        granularities = [g for g in form.getlist("timestamp_granularities[]")
                         + form.getlist("timestamp_granularities") if isinstance(g, str)]

    if len(data) > voice.MAX_UPLOAD_BYTES:
        raise APIError(400, "file is larger than 25 MB", param="file")
    if fmt not in TRANSCRIPTION_FORMATS:
        raise APIError(400, f"response_format must be one of {', '.join(TRANSCRIPTION_FORMATS)}",
                       param="response_format")
    if stream:
        raise APIError(400, "streaming transcription is not supported", param="stream")
    bad = [g for g in granularities if g not in ("word", "segment")]
    if bad:
        raise APIError(400, f"timestamp_granularities must be 'word' or 'segment', not {bad[0]!r}",
                       param="timestamp_granularities")
    if granularities and fmt != "verbose_json":
        raise APIError(400, "timestamp_granularities needs response_format 'verbose_json'",
                       param="timestamp_granularities")

    pcm = voice.decode_audio(data)
    seconds = voice.pcm_seconds(pcm)
    if seconds > s.transcription_max_seconds:
        raise APIError(400, f"the audio is {seconds:.0f}s long; the limit is "
                       f"{s.transcription_max_seconds}s (CONDUIX_TRANSCRIPTION_MAX_SECONDS): "
                       "it is transcribed in real time", param="file")

    tr = await voice.transcribe(pcm, language=language, prompt=prompt)
    if fmt == "text":
        return PlainTextResponse(tr.text + "\n" if tr.text else "")
    if fmt in ("srt", "vtt"):
        media = "application/x-subrip" if fmt == "srt" else "text/vtt"
        return PlainTextResponse(voice.subtitles(tr, fmt), media_type=media)
    if fmt == "verbose_json":
        return JSONResponse(voice.verbose_json(tr, language, "word" in granularities))
    return JSONResponse({"text": tr.text, "usage": voice.usage(tr)})
