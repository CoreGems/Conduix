"""POST /v1/responses — the OpenAI Responses API over Codex.

Three ways to hold a conversation (CONDUIX_API_USEAGE_GUIDE.md §6):

  * Stateless: a fresh (implicit) session; earlier messages in `input` are
    injected into its thread as real items, the trailing user message(s) run
    as the turn.
  * `previous_response_id`: if that response is still the latest turn on its
    thread, continue that thread. Otherwise (a branch, the thread was
    evicted, or `instructions` changed) rebuild the history from the
    response store onto a fresh thread.
  * `session_id` (Conduix extension): run on that session's thread.

Every completed response is stored (unless `store: false`) so any of them
can be continued with `previous_response_id`.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
import json
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from conduix.backend import Event, ToolCall, TurnDone, backend
from conduix.errors import APIError, from_exception, from_turn_error, not_found
from conduix.responses_store import StoredResponse, store
from conduix.schema import (
    FunctionCall,
    FunctionOutput,
    Item,
    Msg,
    ResponseCreateRequest,
    check_tool_outputs,
    parse_input,
    parse_tools,
    split_turn,
)
from conduix.sessions import Session, manager
from conduix.streaming import ResponseStream, collect_response, encode_sse, stream_response

log = logging.getLogger("conduix.responses")

router = APIRouter(prefix="/v1", tags=["responses"])


@dataclass
class Plan:
    req: ResponseCreateRequest
    model: str | None  # passed to Codex; None → its default
    effort: str | None
    summary: str | None
    output_schema: dict[str, Any] | None
    history: list[Item]  # earlier items in `input`, to inject
    new: list[Msg]  # the trailing user message(s): this turn (may be empty)
    prev: StoredResponse | None
    tools: list[dict[str, Any]] = field(default_factory=list)  # Codex dynamic tool specs
    web_search: bool = False
    tool_called: bool = False  # set when the turn ended on a function call

    @property
    def tools_key(self) -> str:
        return json.dumps([self.tools, self.web_search], sort_keys=True) if (
            self.tools or self.web_search) else ""


def _has_images(item: Item) -> bool:
    if isinstance(item, Msg):
        return item.has_images
    if isinstance(item, FunctionOutput) and isinstance(item.output, list):
        return any(p.get("type") == "input_image" for p in item.output)
    return False


def _output_schema(text: dict[str, Any] | None) -> dict[str, Any] | None:
    fmt = (text or {}).get("format") or {}
    kind = fmt.get("type", "text")
    if kind == "text":
        return None
    if kind == "json_schema":
        if not isinstance(fmt.get("schema"), dict):
            raise APIError(400, "text.format.schema is required", param="text.format.schema")
        return fmt["schema"]
    raise APIError(400, f"text.format type {kind!r} is not supported; use json_schema",
                   param="text.format.type")


def plan(req: ResponseCreateRequest) -> Plan:
    """Validate the request. Every 4xx is raised here, before any streaming."""
    tools, web_search = parse_tools(req.tools, req.tool_choice)
    if req.session_id and (tools or web_search):
        # A tool call ends the turn and needs a fresh thread to resume on,
        # which a long-lived session thread can't give.
        raise APIError(400, "tools can't be combined with session_id; use previous_response_id "
                       "or resend the history", param="tools")
    if req.session_id and req.previous_response_id:
        raise APIError(400, "use either session_id or previous_response_id, not both",
                       param="previous_response_id")

    effort = req.reasoning.effort if req.reasoning else None
    summary = req.reasoning.summary if req.reasoning else None
    requested_model = req.model
    if req.session_id:
        sess = manager.get(req.session_id)
        if sess is None:
            raise not_found(f"session {req.session_id!r} not found (expired or server restarted)",
                            param="session_id")
        # The session's model / effort are defaults; the request's own win.
        requested_model = requested_model or sess.model
        effort = effort or sess.effort
    model = backend.resolve_model(requested_model, effort)
    msgs = parse_input(req.input)
    history, new = split_turn(msgs)
    if any(_has_images(m) for m in msgs) and not backend.supports_images(model):
        raise APIError(400, f"model {backend.model_name(model)!r} does not accept image input",
                       param="model", code="unsupported_value")

    prev = None
    if req.previous_response_id:
        prev = store.get(req.previous_response_id)
        try:
            prev and store.history(prev.id)
        except KeyError:
            prev = None
        if prev is None:
            raise not_found(
                f"response {req.previous_response_id!r} not found (expired or server "
                "restarted); resend the conversation history instead",
                param="previous_response_id",
            )
    check_tool_outputs(msgs, store.history(prev.id) if prev else [])
    return Plan(req, model, effort, summary, _output_schema(req.text), history, new, prev,
                tools=tools, web_search=web_search)


def _output_items(rs: ResponseStream) -> list[dict[str, Any]]:
    """The response's output as replayable history (messages and function calls)."""
    out = []
    for item in rs.response["output"]:
        if item["type"] == "message":
            texts = [c["text"] for c in item["content"] if c["type"] == "output_text"]
            out.append(Msg("assistant", texts).to_item())
        elif item["type"] == "function_call":
            out.append(FunctionCall(item["call_id"], item["name"], item["arguments"]).to_item())
    return out


async def _turn(
    p: Plan, rs: ResponseStream, sess: Session, *,
    base: list[dict[str, Any]], own: list[dict[str, Any]], parent_id: str | None,
) -> AsyncIterator[Event]:
    """Run one turn on a session the caller holds. `base` is history that
    already belongs to earlier stored responses; `own` is this request's."""
    await backend.inject_items(sess.thread, base + own)
    sess.head_response_id = None  # the thread is about to move on
    turn_input = [part for m in p.new for part in m.parts]
    async with aclosing(backend.run_turn(
        sess.thread, turn_input, model=p.model, effort=p.effort,
        summary=p.summary, output_schema=p.output_schema, allow_web_search=p.web_search,
    )) as events:
        async for ev in events:
            yield ev
            if isinstance(ev, ToolCall):
                # Hand the call to the client. Leaving the loop closes
                # run_turn, which interrupts the turn (Codex never gets a
                # result); the output comes back in a later request and is
                # replayed onto a fresh thread.
                p.tool_called = True
                break
    if p.tool_called:
        yield TurnDone("completed")

    # The consumer has fed every event to `rs` by the time we resume here.
    if rs.response["status"] == "completed" and p.req.store:
        store.add(StoredResponse(
            id=rs.id, session_id=sess.id, parent_id=parent_id,
            items=own + [m.to_item() for m in p.new] + _output_items(rs),
            instructions=p.req.instructions,
        ))
        if not p.tool_called:  # an interrupted thread can't be continued
            sess.head_response_id = rs.id


async def run_events(p: Plan, rs: ResponseStream) -> AsyncIterator[Event]:
    req = p.req
    own = [m.to_item() for m in p.history]

    if req.session_id:
        async with manager.use(req.session_id) as sess:
            if req.instructions and req.instructions != sess.instructions:
                # Thread instructions are fixed at start; carry a change as a
                # developer message instead.
                own = [Msg("developer", [req.instructions]).to_item()] + own
            async with aclosing(_turn(p, rs, sess, base=[], own=own,
                                      parent_id=sess.head_response_id)) as events:
                async for ev in events:
                    yield ev
        return

    base: list[dict[str, Any]] = []
    if p.prev is not None:
        sess = manager.get(p.prev.session_id)
        # Only implicit sessions: continuing an explicit one here would add
        # turns to a user's session behind its session_id.
        if (sess is not None and sess.implicit and sess.head_response_id == p.prev.id
                and sess.instructions == req.instructions and sess.tools_key == p.tools_key):
            async with manager.use(sess.id) as sess:
                if sess.head_response_id == p.prev.id:  # nobody moved it while we waited
                    async with aclosing(_turn(p, rs, sess, base=[], own=own,
                                              parent_id=p.prev.id)) as events:
                        async for ev in events:
                            yield ev
                    return
        log.info("rebuilding history of %s onto a new thread", p.prev.id)
        base = store.history(p.prev.id)

    sess = await manager.create(model=p.model, instructions=req.instructions, implicit=True,
                                tools=p.tools, web_search=p.web_search, tools_key=p.tools_key)
    try:
        async with manager.use(sess.id) as sess:
            async with aclosing(_turn(p, rs, sess, base=base, own=own,
                                      parent_id=p.prev.id if p.prev else None)) as events:
                async for ev in events:
                    yield ev
    finally:
        if not req.store or p.tool_called:
            await manager.delete(sess.id)


async def _sse(rs: ResponseStream, events: AsyncIterator[Event]) -> AsyncIterator[str]:
    try:
        async with aclosing(stream_response(rs, events)) as stream:
            async for e in stream:
                yield encode_sse(e)
    except Exception as exc:  # noqa: BLE001 - never drop the connection silently
        err = from_exception(exc)
        if err.status >= 500:
            log.exception("stream failed")
        yield encode_sse(rs.error_event(err.message, code=err.code, param=err.param))
    finally:
        await events.aclose()


@router.post("/responses")
async def create_response(req: ResponseCreateRequest):
    p = plan(req)
    rs = ResponseStream(
        model=backend.model_name(p.model),
        instructions=req.instructions,
        previous_response_id=req.previous_response_id,
        reasoning={"effort": p.effort, "summary": p.summary},
        metadata=req.metadata,
        text=req.text,
        session_id=req.session_id,
    )
    events = run_events(p, rs)
    if req.stream:
        return StreamingResponse(
            _sse(rs, events), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        resp = await collect_response(rs, events)
    finally:
        await events.aclose()
    if resp["status"] == "failed":
        raise from_turn_error(rs.failure)
    return resp
