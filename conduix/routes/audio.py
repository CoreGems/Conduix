"""POST /v1/audio/speech — OpenAI text-to-speech over Codex thread realtime.

Experimental and off by default (`CONDUIX_SPEECH=1`). See conduix/voice.py
and BRIEF.md §3.3 for how the audio is made.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import Response
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
