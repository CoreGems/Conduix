"""Request models and input normalization for the OpenAI-compatible routes.

Response *output* shapes come from `openai.types` (validated in tests); the
request side is modelled here because Conduix accepts a subset and adds the
`session_id` extension.
"""
from __future__ import annotations

import base64
import binascii
import json
import re
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


@dataclass
class FunctionCall:
    """A function call the model made earlier (history)."""
    call_id: str
    name: str
    arguments: str
    role = "assistant"

    def to_item(self) -> dict[str, Any]:
        return {"type": "function_call", "call_id": self.call_id, "name": self.name,
                "arguments": self.arguments}


@dataclass
class FunctionOutput:
    """The client's result for a function call."""
    call_id: str
    output: str | list[dict[str, Any]]  # text, or input_text / input_image parts
    role = "tool"

    def to_item(self) -> dict[str, Any]:
        return {"type": "function_call_output", "call_id": self.call_id, "output": self.output}


Item = Msg | FunctionCall | FunctionOutput


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


def _tool_output(output: Any) -> str | list[dict[str, Any]]:
    if isinstance(output, str):
        return output
    if not isinstance(output, list):
        raise _bad("function_call_output.output must be a string or a list of content parts")
    parts: list[dict[str, Any]] = []
    for p in _parts(output):
        parts.append({"type": "input_image", "image_url": p.url} if isinstance(p, ImagePart)
                     else {"type": "input_text", "text": p})
    return parts


def parse_input(input: str | list[dict[str, Any]]) -> list[Item]:
    """Normalize Responses `input` to messages and function-call items.

    Accepts a string, easy messages ({role, content}), `message` items,
    `function_call` / `function_call_output` items, and items copied from an
    earlier response's `output` (reasoning and web_search_call items there are
    dropped: they can't be replayed).
    """
    if isinstance(input, str):
        return [Msg("user", [input])]
    msgs: list[Item] = []
    for item in input:
        kind = item.get("type", "message")
        if kind in ("reasoning", "web_search_call"):
            continue
        if kind == "function_call":
            args = item.get("arguments", "")
            msgs.append(FunctionCall(item.get("call_id") or "", item.get("name") or "",
                                     args if isinstance(args, str) else json.dumps(args)))
            continue
        if kind == "function_call_output":
            if not item.get("call_id"):
                raise _bad("function_call_output needs a call_id")
            msgs.append(FunctionOutput(item["call_id"], _tool_output(item.get("output", ""))))
            continue
        if kind != "message":
            raise _bad(f"input item type {kind!r} is not supported")
        role = item.get("role")
        if role not in ROLES:
            raise _bad(f"unknown message role {role!r}")
        msg = Msg("developer" if role == "system" else role, _parts(item.get("content")))
        if msg.has_images and msg.role != "user":
            raise _bad("only user messages can contain images")
        msgs.append(msg)
    return msgs


def split_turn(msgs: list[Item]) -> tuple[list[Item], list[Msg]]:
    """(history to inject, new turn): the new turn is the trailing user
    messages. It may be empty when the input ends with function call output:
    the model then continues from the tool results."""
    i = len(msgs)
    while i > 0 and isinstance(msgs[i - 1], Msg) and msgs[i - 1].role == "user":
        i -= 1
    if i == len(msgs) and not isinstance(msgs[-1] if msgs else None, FunctionOutput):
        raise _bad("input must end with a user message or function_call_output")
    return msgs[:i], msgs[i:]


# --- tools -------------------------------------------------------------------

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def parse_tools(tools: list[dict[str, Any]] | None, tool_choice: Any = None,
                ) -> tuple[list[dict[str, Any]], bool]:
    """Responses `tools` → (Codex dynamic tool specs, web_search enabled).

    `tool_choice: "none"` drops every tool. "required" or a specific function
    can't be enforced through Codex, so they are treated as "auto".
    """
    if tool_choice == "none" or not tools:
        return [], False
    specs: list[dict[str, Any]] = []
    web_search = False
    for i, tool in enumerate(tools):
        kind = tool.get("type")
        if kind == "function":
            name = tool.get("name")
            if not isinstance(name, str) or not _NAME.match(name):
                raise APIError(400, f"invalid function name {name!r} (a-z, A-Z, 0-9, _ and -, "
                               "up to 64)", param=f"tools[{i}].name")
            if any(s["name"] == name for s in specs):
                raise APIError(400, f"duplicate function name {name!r}", param=f"tools[{i}].name")
            params = tool.get("parameters") or {"type": "object", "properties": {}}
            specs.append({"type": "function", "name": name,
                          "description": tool.get("description") or "", "inputSchema": params})
        elif isinstance(kind, str) and kind.startswith("web_search"):
            web_search = True
        else:
            raise APIError(400, f"tool type {kind!r} is not supported; use function or "
                           "web_search", param=f"tools[{i}].type")
    return specs, web_search


def check_tool_outputs(items: list[Item], history: list[dict[str, Any]] = ()) -> None:
    """Every function_call_output must answer a function call in the history."""
    known = {it["call_id"] for it in history if it.get("type") == "function_call"}
    known |= {m.call_id for m in items if isinstance(m, FunctionCall)}
    for m in items:
        if isinstance(m, FunctionOutput) and m.call_id not in known:
            raise _bad(f"no tool call found for function call output with call_id {m.call_id!r}")
