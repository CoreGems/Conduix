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
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from openai_codex import TextInput

from conduix.backend import Event, backend
from conduix.errors import APIError, not_found
from conduix.responses_store import StoredResponse, store
from conduix.schema import Msg, ResponseCreateRequest, parse_input, split_turn
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
    history: list[Msg]  # earlier messages in `input`, to inject
    new: list[Msg]  # the trailing user message(s): this turn
    prev: StoredResponse | None


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
    if req.tools:
        raise APIError(400, "function tools are not supported yet (planned for v1.1)",
                       param="tools")
    if req.session_id and req.previous_response_id:
        raise APIError(400, "use either session_id or previous_response_id, not both",
                       param="previous_response_id")

    effort = req.reasoning.effort if req.reasoning else None
    summary = req.reasoning.summary if req.reasoning else None
    model = backend.resolve_model(req.model, effort)
    history, new = split_turn(parse_input(req.input))

    if req.session_id and manager.get(req.session_id) is None:
        raise not_found(f"session {req.session_id!r} not found (expired or server restarted)",
                        param="session_id")
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
    return Plan(req, model, effort, summary, _output_schema(req.text), history, new, prev)


def _assistant_items(rs: ResponseStream) -> list[dict[str, Any]]:
    return [
        Msg("assistant", [c["text"] for c in item["content"] if c["type"] == "output_text"]).to_item()
        for item in rs.response["output"] if item["type"] == "message"
    ]


async def _turn(
    p: Plan, rs: ResponseStream, sess: Session, *,
    base: list[dict[str, Any]], own: list[dict[str, Any]], parent_id: str | None,
) -> AsyncIterator[Event]:
    """Run one turn on a session the caller holds. `base` is history that
    already belongs to earlier stored responses; `own` is this request's."""
    await backend.inject_items(sess.thread, base + own)
    sess.head_response_id = None  # the thread is about to move on
    turn_input = [TextInput(t) for m in p.new for t in m.texts]
    async with aclosing(backend.run_turn(
        sess.thread, turn_input, model=p.model, effort=p.effort,
        summary=p.summary, output_schema=p.output_schema,
    )) as events:
        async for ev in events:
            yield ev

    # The consumer has fed every event to `rs` by the time we resume here.
    if rs.response["status"] == "completed" and p.req.store:
        store.add(StoredResponse(
            id=rs.id, session_id=sess.id, parent_id=parent_id,
            items=own + [m.to_item() for m in p.new] + _assistant_items(rs),
            instructions=p.req.instructions,
        ))
        sess.head_response_id = rs.id


async def _events(p: Plan, rs: ResponseStream) -> AsyncIterator[Event]:
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
                and sess.instructions == req.instructions):
            async with manager.use(sess.id) as sess:
                if sess.head_response_id == p.prev.id:  # nobody moved it while we waited
                    async with aclosing(_turn(p, rs, sess, base=[], own=own,
                                              parent_id=p.prev.id)) as events:
                        async for ev in events:
                            yield ev
                    return
        log.info("rebuilding history of %s onto a new thread", p.prev.id)
        base = store.history(p.prev.id)

    sess = await manager.create(model=p.model, instructions=req.instructions, implicit=True)
    try:
        async with manager.use(sess.id) as sess:
            async with aclosing(_turn(p, rs, sess, base=base, own=own,
                                      parent_id=p.prev.id if p.prev else None)) as events:
                async for ev in events:
                    yield ev
    finally:
        if not req.store:
            await manager.delete(sess.id)


async def _sse(rs: ResponseStream, events: AsyncIterator[Event]) -> AsyncIterator[str]:
    try:
        async with aclosing(stream_response(rs, events)) as stream:
            async for e in stream:
                yield encode_sse(e)
    except APIError as exc:
        yield encode_sse(rs.error_event(exc.message, code=exc.code, param=exc.param))
    except Exception as exc:  # noqa: BLE001 - never drop the connection silently
        log.exception("stream failed")
        yield encode_sse(rs.error_event(f"{type(exc).__name__}: {exc}", code="server_error"))
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
    events = _events(p, rs)
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
        # step 8 maps quota/auth failures to 429/401
        raise APIError(500, resp["error"]["message"], type="server_error", code="server_error")
    return resp
