"""Request models and input normalization for the OpenAI-compatible routes.

Response *output* shapes come from `openai.types` (validated in tests); the
request side is modelled here because Conduix accepts a subset and adds the
`session_id` extension.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

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
    texts: list[str]

    def to_item(self) -> dict[str, Any]:
        """As a raw Responses API item, for `thread/inject_items`."""
        kind = "output_text" if self.role == "assistant" else "input_text"
        return {"type": "message", "role": self.role,
                "content": [{"type": kind, "text": t} for t in self.texts]}


def _bad(message: str) -> APIError:
    return APIError(400, message, param="input")


def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        raise _bad("message content must be a string or a list of content parts")
    out = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in ("input_text", "output_text", "text"):
            out.append(part.get("text", ""))
        elif kind == "refusal":
            out.append(part.get("refusal", ""))
        elif kind == "input_image":
            raise _bad("image input is not supported yet")
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
        msgs.append(Msg("developer" if role == "system" else role, _texts(item.get("content"))))
    return msgs


def split_turn(msgs: list[Msg]) -> tuple[list[Msg], list[Msg]]:
    """(history to inject, new turn): the new turn is the trailing user messages."""
    i = len(msgs)
    while i > 0 and msgs[i - 1].role == "user":
        i -= 1
    if i == len(msgs):
        raise _bad("input must end with a user message")
    return msgs[:i], msgs[i:]
