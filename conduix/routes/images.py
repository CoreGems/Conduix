"""POST /v1/images/generations — the OpenAI Images API over Codex.

Codex has no image model to call directly. The request runs as a Responses
turn with the image_generation tool on (`routes.responses`): the chat model
is told to generate one image, and Codex's built-in image tool makes it. The
turn is stopped as soon as the image arrives, and the image is returned in
the Images API shape.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from conduix.backend import ImageGenerationCall, backend
from conduix.errors import APIError, from_turn_error, image_limit_error
from conduix.routes.responses import plan, run_events
from conduix.schema import ResponseCreateRequest, image_info
from conduix.streaming import ResponseStream

router = APIRouter(prefix="/v1", tags=["images"])

INSTRUCTIONS = ("Generate exactly one image for the user's prompt with your image generation "
                "tool. Do not ask clarifying questions. If you can't generate it, say why in "
                "one sentence.")


class ImageGenerateRequest(BaseModel):
    """Unknown OpenAI fields (`quality`, `output_format`, `moderation`, `user`,
    ...) are accepted and ignored: Codex's image tool has no setting for them."""

    model_config = ConfigDict(extra="allow")

    prompt: str
    model: str | None = None
    n: int | None = None
    size: str | None = None
    background: str | None = None
    response_format: str | None = None
    stream: bool | None = None


def _no_image(message: str) -> APIError:
    # Not retried by the openai SDK: every attempt would spend plan quota.
    return APIError(502, message, type="server_error", code="no_image_generated",
                    headers={"x-should-retry": "false"})


def _turn_input(req: ImageGenerateRequest) -> str:
    """The prompt, with the size and background asked for as a hint: the
    model writes the image request itself, so they can't be enforced."""
    hints = []
    if req.size and req.size != "auto":
        hints.append(f"size {req.size}")
    if req.background in ("transparent", "opaque"):
        hints.append(f"{req.background} background")
    return req.prompt + (f"\n\n(Requested: {', '.join(hints)}.)" if hints else "")


@router.post("/images/generations")
async def generate_image(req: ImageGenerateRequest) -> dict[str, Any]:
    if not req.prompt.strip():
        raise APIError(400, "prompt must not be empty", param="prompt")
    if req.n not in (None, 1):
        raise APIError(400, "only n=1 is supported", param="n")
    if req.response_format not in (None, "b64_json"):
        raise APIError(400, "only response_format 'b64_json' is supported: images are returned "
                       "as base64, never as a URL", param="response_format")
    if req.stream:
        raise APIError(400, "streaming is not supported for image generation", param="stream")

    # `model` names an image model in OpenAI's API (gpt-image-1, ...). Here a
    # chat model drives the image tool: one of the plan's models if named,
    # otherwise the default.
    driver = req.model if any(m["id"] == req.model for m in backend.cached_models) else None
    p = plan(ResponseCreateRequest(
        model=driver, input=_turn_input(req), instructions=INSTRUCTIONS,
        tools=[{"type": "image_generation"}], store=False,
    ))
    rs = ResponseStream(model=backend.model_name(p.model))
    image: ImageGenerationCall | None = None
    events = run_events(p, rs)
    try:
        async for ev in events:
            rs.feed(ev)
            if isinstance(ev, ImageGenerationCall) and ev.status != "in_progress":
                image = ev
                break  # the image is in; closing `events` stops the turn here
    finally:
        await events.aclose()

    if rs.failure is not None:
        raise from_turn_error(rs.failure)
    if image is None:
        said = rs.output_text.strip()
        raise _no_image("the model did not generate an image" + (f": {said[:300]}" if said else ""))
    if image.status != "completed":
        if image.limit_exceeded:
            raise image_limit_error(image.resets_at)
        raise _no_image("image generation failed")

    data = {"b64_json": image.result}
    if image.revised_prompt:
        data["revised_prompt"] = image.revised_prompt
    output_format, size = image_info(image.result)
    out: dict[str, Any] = {"created": int(time.time()), "data": [data],
                           "output_format": output_format}
    if size:
        out["size"] = size
    if image.transparent_background is not None:
        out["background"] = "transparent" if image.transparent_background else "opaque"
    return out
