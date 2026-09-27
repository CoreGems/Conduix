"""Request models and input normalization for the OpenAI-compatible routes.

Response *output* shapes come from `openai.types` (validated in tests); the
request side is modelled here because Conduix accepts a subset and adds the
`session_id` extension.
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from conduix.backend import ImagePart
from conduix.errors import APIError


class ReasoningParam(BaseModel):
    model_config = ConfigDict(extra="ignore")
    effort: str | None = None
    summary: Literal["auto", "concise", "detailed", "none"] | None = None


class ResponseCreateRequest(BaseModel):
    """POST /v1/responses. Unknown OpenAI fields are accepted and ignored
    (`temperature`, `top_p`, `max_output_tokens`, `user`, ...): Codex has no
    knob for them."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    stream: bool = False
    previous_response_id: str | None = None
    session_id: str | None = None  # Conduix extension (send via extra_body)
    reasoning: ReasoningParam | None = None
    text: dict[str, Any] | None = None
    store: bool = True
    metadata: dict[str, str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None


# --- input normalization -----------------------------------------------------

ROLES = ("user", "assistant", "system", "developer")


@dataclass
class Msg:
    role: str  # user | assistant | developer ("system" is folded into developer)
    parts: list[str | ImagePart]  # text and images, in order

    @property
    def has_images(self) -> bool:
        return any(isinstance(p, ImagePart) for p in self.parts)

    def to_item(self) -> dict[str, Any]:
        """As a raw Responses API item, for `thread/inject_items`."""
        kind = "output_text" if self.role == "assistant" else "input_text"
        return {"type": "message", "role": self.role, "content": [
            {"type": "input_image", "image_url": p.url} if isinstance(p, ImagePart)
            else {"type": kind, "text": p}
            for p in self.parts
        ]}


def _bad(message: str, code: str | None = None) -> APIError:
    return APIError(400, message, param="input", code=code)


IMAGE_TYPES = ("image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp")
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # OpenAI's per-image limit


def _image(url: Any) -> ImagePart:
    """Validate an image reference: base64 data URLs only (remote URLs are
    deprecated upstream in Codex)."""
    if not isinstance(url, str) or not url:
        raise _bad("image_url must be a string", "invalid_image_url")
    if url.startswith(("http://", "https://")):
        raise _bad("remote image URLs are not supported; download the image and send it as "
                   "a base64 data URL (data:image/png;base64,...)", "invalid_image_url")
    header, sep, payload = url.partition(",")
    if not (header.startswith("data:") and header.endswith(";base64") and sep):
        raise _bad("image_url must be a base64 data URL (data:image/png;base64,...)",
                   "invalid_image_url")
    media_type = header[len("data:"):-len(";base64")].lower()
    if media_type not in IMAGE_TYPES:
        raise _bad(f"unsupported image type {media_type!r}; use png, jpeg, gif or webp",
                   "unsupported_image_media_type")
    try:
        size = len(base64.b64decode(payload, validate=True))
    except (binascii.Error, ValueError):
        raise _bad("image data is not valid base64", "invalid_base64_image") from None
    if size == 0:
        raise _bad("image data is empty", "empty_image_file")
    if size > MAX_IMAGE_BYTES:
        raise _bad(f"image is {size // (1024 * 1024)} MB; the limit is 20 MB", "image_too_large")
    return ImagePart(url)


def _parts(content: Any) -> list[str | ImagePart]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        raise _bad("message content must be a string or a list of content parts")
    out: list[str | ImagePart] = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in ("input_text", "output_text", "text"):
            out.append(part.get("text", ""))
        elif kind == "refusal":
            out.append(part.get("refusal", ""))
        elif kind == "input_image":  # Responses: {"image_url": "data:..."}
            if part.get("file_id"):
                raise _bad("file_id images are not supported; send a base64 data URL",
                           "invalid_image_url")
            out.append(_image(part.get("image_url")))
        elif kind == "image_url":  # Chat: {"image_url": {"url": "data:..."}}
            ref = part.get("image_url")
            out.append(_image(ref.get("url") if isinstance(ref, dict) else ref))
        else:
            raise _bad(f"content part type {kind!r} is not supported")
    return out


def parse_input(input: str | list[dict[str, Any]]) -> list[Msg]:
    """Normalize Responses `input` to messages.

    Accepts a string, easy messages ({role, content}), `message` items, and
    items copied from an earlier response's `output` (reasoning items there
    are dropped: they can't be replayed).
    """
    if isinstance(input, str):
        return [Msg("user", [input])]
    msgs = []
    for item in input:
        kind = item.get("type", "message")
        if kind == "reasoning":
            continue
        if kind != "message":
            if kind in ("function_call", "function_call_output"):
                raise _bad("function tools are not supported yet (planned for v1.1)")
            raise _bad(f"input item type {kind!r} is not supported")
        role = item.get("role")
        if role not in ROLES:
            raise _bad(f"unknown message role {role!r}")
        msg = Msg("developer" if role == "system" else role, _parts(item.get("content")))
        if msg.has_images and msg.role != "user":
            raise _bad("only user messages can contain images")
        msgs.append(msg)
    return msgs


def split_turn(msgs: list[Msg]) -> tuple[list[Msg], list[Msg]]:
    """(history to inject, new turn): the new turn is the trailing user messages."""
    i = len(msgs)
    while i > 0 and msgs[i - 1].role == "user":
        i -= 1
    if i == len(msgs):
        raise _bad("input must end with a user message")
    return msgs[:i], msgs[i:]
