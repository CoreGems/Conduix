"""POST /v1/chat/completions — a translation layer over /v1/responses.

The request is converted to a Responses request and runs through the same
plan / sessions / error mapping (`routes.responses`); the Responses events
are then translated to `chat.completion` / `chat.completion.chunk`. Chat has
no response ids, so nothing is stored for `previous_response_id`; `messages`
carries the history, or use the `session_id` extension.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from conduix.backend import Event, UnsupportedEffortError, backend
from conduix.errors import APIError, from_exception, from_turn_error
from conduix.routes.responses import plan, run_events
from conduix.schema import ResponseCreateRequest
from conduix.streaming import (
    ChatStream,
    ResponseStream,
    chat_completion,
    collect_response,
    encode_chat_sse,
    stream_response,
)

log = logging.getLogger("conduix.chat")

router = APIRouter(prefix="/v1", tags=["chat"])


class ChatCompletionRequest(BaseModel):
    """Unknown OpenAI fields (`temperature`, `max_tokens`, `top_p`, ...) are
    accepted and ignored: Codex has no knob for them."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[dict[str, Any]]
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    response_format: dict[str, Any] | None = None
    session_id: str | None = None  # Conduix extension (send via extra_body)
    n: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None


def _input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat messages as Responses input items (content shapes are shared)."""
    items = []
    for m in messages:
        role = m.get("role")
        if role == "tool" or m.get("tool_calls") or m.get("function_call"):
            raise APIError(400, "function tools are not supported yet (planned for v1.1)",
                           param="messages")
        items.append({"role": role, "content": m.get("content") or ""})
    return items


def _text(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    if not response_format:
        return None
    kind = response_format.get("type")
    if kind == "json_schema":
        spec = response_format.get("json_schema") or {}
        return {"format": {"type": "json_schema", **spec}}
    return {"format": {"type": kind}}  # "text" passes; "json_object" gets the 400


def to_responses_request(req: ChatCompletionRequest) -> ResponseCreateRequest:
    if req.n not in (None, 1):
        raise APIError(400, "only n=1 is supported", param="n")
    if req.tools:
        raise APIError(400, "function tools are not supported yet (planned for v1.1)",
                       param="tools")
    return ResponseCreateRequest(
        model=req.model,
        input=_input(req.messages),
        stream=req.stream,
        session_id=req.session_id,
        reasoning={"effort": req.reasoning_effort} if req.reasoning_effort else None,
        text=_text(req.response_format),
        store=False,  # chat has no response ids to continue from
    )


def _chat_param(param: str | None) -> str | None:
    """Responses parameter names as the chat request spells them."""
    return {
        "input": "messages",
        "reasoning.effort": "reasoning_effort",
        "text.format.type": "response_format.type",
        "text.format.schema": "response_format.json_schema.schema",
    }.get(param, param)


async def _sse(rs: ResponseStream, chat: ChatStream,
               events: AsyncIterator[Event]) -> AsyncIterator[str]:
    try:
        async with aclosing(stream_response(rs, events)) as stream:
            async for ev in stream:
                if ev["type"] == "error":
                    err = (from_turn_error(rs.failure) if rs.failure is not None else
                           APIError(500, ev["message"], type="server_error", code=ev["code"],
                                    param=ev["param"]))
                    err.param = _chat_param(err.param)
                    yield encode_chat_sse(err.body())  # the openai SDK raises on this
                    break
                for chunk in chat.feed(ev):
                    yield encode_chat_sse(chunk)
    except Exception as exc:  # noqa: BLE001 - never drop the connection silently
        err = from_exception(exc)
        if err.status >= 500:
            log.exception("chat stream failed")
        yield encode_chat_sse(err.body())
    finally:
        await events.aclose()
    yield encode_chat_sse("[DONE]")


@router.post("/chat/completions")
async def create_chat_completion(req: ChatCompletionRequest):
    try:
        p = plan(to_responses_request(req))
    except APIError as exc:
        exc.param = _chat_param(exc.param)
        raise
    except UnsupportedEffortError as exc:
        raise APIError(400, str(exc), param="reasoning_effort",
                       code="unsupported_value") from exc
    model = backend.model_name(p.model)
    rs = ResponseStream(model=model, session_id=req.session_id)
    events = run_events(p, rs)

    if req.stream:
        include_usage = bool((req.stream_options or {}).get("include_usage"))
        chat = ChatStream(model=model, include_usage=include_usage)
        return StreamingResponse(
            _sse(rs, chat, events), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        resp = await collect_response(rs, events)
    finally:
        await events.aclose()
    if resp["status"] == "failed":
        err = from_turn_error(rs.failure)
        err.param = _chat_param(err.param)
        raise err
    return chat_completion(resp)
