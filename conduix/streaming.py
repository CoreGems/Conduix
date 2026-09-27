"""Synthesize OpenAI Responses API events from backend events.

Conduit forwards Anthropic SSE as-is; Conduix has to build OpenAI's from
Codex notifications. `ResponseStream` is a pure state machine: feed it
backend events, get back Responses stream events (plain dicts, validated
against the `openai` package's types in tests) and, at the end, the final
`Response` object. It knows nothing about HTTP, so the SSE route and the
non-streaming collector share it.

Event order for one message (same as api.openai.com):
  response.created, response.in_progress,
  response.output_item.added, response.content_part.added,
  response.output_text.delta × N, response.output_text.done,
  response.content_part.done, response.output_item.done,
  response.completed | response.failed | response.incomplete
Reasoning items add response.reasoning_summary_part.added /
reasoning_summary_text.delta / .done / part.done in place of the text events.

Chat Completions chunks (step 9) will be derived from the same events.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from conduix.backend import (
    Event,
    MessageDone,
    MessageStarted,
    ReasoningDone,
    ReasoningStarted,
    ReasoningSummaryDelta,
    TextDelta,
    ToolCall,
    TurnDone,
    TurnError,
    Usage,
    WebSearchCall,
)

from conduix.errors import resolve

log = logging.getLogger("conduix.streaming")


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


@dataclass
class _Message:
    index: int
    id: str
    phase: str | None
    text: str = ""
    done: bool = False


@dataclass
class _WebSearch:
    index: int
    id: str
    done: bool = False


@dataclass
class _Reasoning:
    index: int
    id: str
    parts: dict[int, str] = field(default_factory=dict)  # summary_index → text so far
    done: bool = False


class ResponseStream:
    def __init__(
        self,
        *,
        model: str,
        response_id: str | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        reasoning: dict[str, Any] | None = None,
        metadata: dict[str, str] | None = None,
        text: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> None:
        self.id = response_id or new_response_id()
        self.response: dict[str, Any] = {
            "id": self.id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "background": False,
            "error": None,
            "incomplete_details": None,
            "instructions": instructions,
            "metadata": metadata or {},
            "model": model,
            "output": [],
            "parallel_tool_calls": False,  # no function tools until v1.1
            "previous_response_id": previous_response_id,
            "reasoning": reasoning or {"effort": None, "summary": None},
            "temperature": None,
            "text": text or {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "usage": None,
        }
        if session_id is not None:
            self.response["session_id"] = session_id  # Conduix extension
        self._seq = 0
        self._items: dict[str, _Message | _Reasoning | _WebSearch] = {}
        self.tool_calls: list[dict[str, Any]] = []  # function_call items, in order
        self._error: TurnError | None = None
        self.failure: TurnError | None = None  # set when the turn failed
        self.finished = False

    # --- plumbing ------------------------------------------------------------

    def _emit(self, type_: str, **fields: Any) -> dict[str, Any]:
        ev = {"type": type_, "sequence_number": self._seq, **fields}
        self._seq += 1
        return ev

    def _snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.response)

    @property
    def output_text(self) -> str:
        return "".join(
            c["text"]
            for item in self.response["output"] if item["type"] == "message"
            for c in item["content"] if c["type"] == "output_text"
        )

    # --- items ---------------------------------------------------------------

    @staticmethod
    def _message_item(m: _Message, status: str) -> dict[str, Any]:
        content = [] if status == "in_progress" else [_text_part(m.text)]
        item = {"id": m.id, "type": "message", "role": "assistant",
                "status": status, "content": content}
        if m.phase:
            item["phase"] = m.phase
        return item

    @staticmethod
    def _reasoning_item(r: _Reasoning) -> dict[str, Any]:
        return {"id": r.id, "type": "reasoning",
                "summary": [_summary_part(r.parts[i]) for i in sorted(r.parts)]}

    def _add_item(self, item: dict[str, Any]) -> int:
        self.response["output"].append(item)
        return len(self.response["output"]) - 1

    def _start_message(self, ev: MessageStarted) -> list[dict]:
        m = _Message(index=len(self.response["output"]), id=ev.item_id, phase=ev.phase)
        self._items[m.id] = m
        item = self._message_item(m, "in_progress")
        self._add_item(item)
        return [
            self._emit("response.output_item.added", output_index=m.index, item=copy.deepcopy(item)),
            self._emit("response.content_part.added", item_id=m.id, output_index=m.index,
                       content_index=0, part=_text_part("")),
        ]

    def _finish_message(self, m: _Message, final_text: str | None, status: str) -> list[dict]:
        out = []
        if final_text is not None:
            if not m.text and final_text:
                # Codex sent the whole text without deltas; stream it as one.
                out.append(self._text_delta(m, final_text))
            elif final_text != m.text:
                log.warning("message %s: deltas differ from final text; using final text", m.id)
            m.text = final_text
        m.done = True
        item = self._message_item(m, status)
        self.response["output"][m.index] = item
        out += [
            self._emit("response.output_text.done", item_id=m.id, output_index=m.index,
                       content_index=0, text=m.text, logprobs=[]),
            self._emit("response.content_part.done", item_id=m.id, output_index=m.index,
                       content_index=0, part=_text_part(m.text)),
            self._emit("response.output_item.done", output_index=m.index, item=copy.deepcopy(item)),
        ]
        return out

    def _text_delta(self, m: _Message, delta: str) -> dict:
        m.text += delta
        return self._emit("response.output_text.delta", item_id=m.id, output_index=m.index,
                          content_index=0, delta=delta, logprobs=[])

    def _start_reasoning(self, ev: ReasoningStarted) -> list[dict]:
        r = _Reasoning(index=len(self.response["output"]), id=ev.item_id)
        self._items[r.id] = r
        item = self._reasoning_item(r)
        self._add_item(item)
        return [self._emit("response.output_item.added", output_index=r.index,
                           item=copy.deepcopy(item))]

    def _summary_delta(self, r: _Reasoning, idx: int, delta: str) -> list[dict]:
        out = []
        if idx not in r.parts:
            r.parts[idx] = ""
            out.append(self._emit("response.reasoning_summary_part.added", item_id=r.id,
                                  output_index=r.index, summary_index=idx,
                                  part=_summary_part("")))
        r.parts[idx] += delta
        out.append(self._emit("response.reasoning_summary_text.delta", item_id=r.id,
                              output_index=r.index, summary_index=idx, delta=delta))
        return out

    def _finish_reasoning(self, r: _Reasoning, summary: list[str] | None) -> list[dict]:
        out = []
        # Codex's final summary wins; parts it never streamed are sent whole.
        for idx, text in enumerate(summary or []):
            if idx not in r.parts:
                out += self._summary_delta(r, idx, text)
            r.parts[idx] = text
        for idx in sorted(r.parts):
            out += [
                self._emit("response.reasoning_summary_text.done", item_id=r.id,
                           output_index=r.index, summary_index=idx, text=r.parts[idx]),
                self._emit("response.reasoning_summary_part.done", item_id=r.id,
                           output_index=r.index, summary_index=idx,
                           part=_summary_part(r.parts[idx])),
            ]
        r.done = True
        item = self._reasoning_item(r)
        self.response["output"][r.index] = item
        out.append(self._emit("response.output_item.done", output_index=r.index,
                              item=copy.deepcopy(item)))
        return out

    def _function_call(self, ev: ToolCall) -> list[dict]:
        item = {"id": f"fc_{uuid.uuid4().hex}", "type": "function_call", "status": "completed",
                "call_id": ev.call_id, "name": ev.name, "arguments": ev.arguments}
        index = self._add_item(item)
        self.tool_calls.append(item)
        started = {**item, "status": "in_progress", "arguments": ""}
        return [
            self._emit("response.output_item.added", output_index=index, item=started),
            self._emit("response.function_call_arguments.delta", item_id=item["id"],
                       output_index=index, delta=ev.arguments),
            self._emit("response.function_call_arguments.done", item_id=item["id"],
                       output_index=index, arguments=ev.arguments),
            self._emit("response.output_item.done", output_index=index, item=copy.deepcopy(item)),
        ]

    def _web_search(self, ev: WebSearchCall) -> list[dict]:
        ws = self._items.get(ev.item_id)
        if not isinstance(ws, _WebSearch):
            ws = _WebSearch(index=len(self.response["output"]), id=ev.item_id)
            self._items[ws.id] = ws
            self._add_item({"id": ws.id, "type": "web_search_call", "status": "in_progress",
                            "action": _web_action(ev.action)})
            out = [
                self._emit("response.output_item.added", output_index=ws.index,
                           item=copy.deepcopy(self.response["output"][ws.index])),
                self._emit("response.web_search_call.in_progress", item_id=ws.id,
                           output_index=ws.index),
                self._emit("response.web_search_call.searching", item_id=ws.id,
                           output_index=ws.index),
            ]
        else:
            out = []
        if ev.status == "in_progress":
            return out
        ws.done = True
        item = self.response["output"][ws.index]
        item["status"] = ev.status
        if ev.action:
            item["action"] = _web_action(ev.action)
        if ev.status == "completed":
            out.append(self._emit("response.web_search_call.completed", item_id=ws.id,
                                  output_index=ws.index))
        out.append(self._emit("response.output_item.done", output_index=ws.index,
                              item=copy.deepcopy(item)))
        return out

    # --- public --------------------------------------------------------------

    def start(self) -> list[dict]:
        return [
            self._emit("response.created", response=self._snapshot()),
            self._emit("response.in_progress", response=self._snapshot()),
        ]

    def error_event(self, message: str, *, code: str | None = None,
                    param: str | None = None) -> dict:
        """A terminal `error` event for failures outside the turn itself."""
        self.finished = True
        return self._emit("error", code=code, message=message, param=param)

    def feed(self, ev: Event) -> list[dict]:
        if self.finished:
            return []
        if isinstance(ev, MessageStarted):
            return self._start_message(ev)
        if isinstance(ev, ReasoningStarted):
            return self._start_reasoning(ev)

        if isinstance(ev, TextDelta):
            m = self._items.get(ev.item_id)
            if not isinstance(m, _Message):
                m_start = self._start_message(MessageStarted(ev.item_id))
                return m_start + [self._text_delta(self._items[ev.item_id], ev.delta)]
            return [self._text_delta(m, ev.delta)]

        if isinstance(ev, MessageDone):
            out = []
            if not isinstance(self._items.get(ev.item_id), _Message):
                out = self._start_message(MessageStarted(ev.item_id))
            return out + self._finish_message(self._items[ev.item_id], ev.text, "completed")

        if isinstance(ev, ReasoningSummaryDelta):
            r = self._items.get(ev.item_id)
            if not isinstance(r, _Reasoning):
                out = self._start_reasoning(ReasoningStarted(ev.item_id))
                return out + self._summary_delta(self._items[ev.item_id], ev.summary_index, ev.delta)
            return self._summary_delta(r, ev.summary_index, ev.delta)

        if isinstance(ev, ReasoningDone):
            out = []
            if not isinstance(self._items.get(ev.item_id), _Reasoning):
                out = self._start_reasoning(ReasoningStarted(ev.item_id))
            return out + self._finish_reasoning(self._items[ev.item_id], ev.summary)

        if isinstance(ev, ToolCall):
            return self._function_call(ev)

        if isinstance(ev, WebSearchCall):
            return self._web_search(ev)

        if isinstance(ev, Usage):
            self.response["usage"] = {
                "input_tokens": ev.input_tokens,
                "input_tokens_details": {
                    "cached_tokens": ev.cached_input_tokens,
                    "cache_write_tokens": ev.cache_write_input_tokens,
                },
                "output_tokens": ev.output_tokens,
                "output_tokens_details": {"reasoning_tokens": ev.reasoning_output_tokens},
                "total_tokens": ev.input_tokens + ev.output_tokens,
            }
            return []

        if isinstance(ev, TurnError):
            self._error = ev
            return []

        if isinstance(ev, TurnDone):
            return self._finish(ev)

        return []

    def _finish(self, ev: TurnDone) -> list[dict]:
        out = []
        # Close anything Codex left open (a failed or interrupted turn).
        for item in self._items.values():
            if item.done:
                continue
            if isinstance(item, _Message):
                out += self._finish_message(item, None, "incomplete")
            elif isinstance(item, _WebSearch):
                out += self._web_search(WebSearchCall(item.id, "incomplete"))
            else:
                out += self._finish_reasoning(item, None)

        self.finished = True
        if ev.status == "completed":
            self.response["status"] = "completed"
            self.response["completed_at"] = int(time.time())
            return out + [self._emit("response.completed", response=self._snapshot())]
        if ev.status == "interrupted":
            self.response["status"] = "incomplete"
            return out + [self._emit("response.incomplete", response=self._snapshot())]

        # Prefer the turn's own error; an earlier error notification may carry
        # the codexErrorInfo when turn/completed doesn't.
        err = ev.error or self._error or TurnError("codex turn failed")
        early = self._error
        if err.codex_error_info is None and early is not None and early.codex_error_info:
            err = dataclasses.replace(err, codex_error_info=early.codex_error_info,
                                      resets_at=err.resets_at or early.resets_at)
        self.failure = err
        kind, message, param = resolve(err)
        self.response["status"] = "failed"
        self.response["error"] = {"code": kind.response_code, "message": message}
        return out + [
            # Clients are told to watch for `error` (usage guide §11); the
            # terminal response.failed then carries the final response.
            self._emit("error", code=kind.code, message=message, param=param),
            self._emit("response.failed", response=self._snapshot()),
        ]


def _web_action(action: dict[str, Any]) -> dict[str, Any]:
    """Codex webSearch action → OpenAI web_search_call action."""
    kind = action.get("type")
    if kind == "openPage":
        return {"type": "open_page", "url": action.get("url")}
    if kind == "findInPage":
        return {"type": "find_in_page", "url": action.get("url") or "",
                "pattern": action.get("pattern") or ""}
    return {"type": "search", "query": action.get("query") or ""}


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}


def _summary_part(text: str) -> dict[str, Any]:
    return {"type": "summary_text", "text": text}


# --- drivers -----------------------------------------------------------------

async def stream_response(rs: ResponseStream, events: AsyncIterator[Event]) -> AsyncIterator[dict]:
    """Yield Responses stream events for one turn, always ending in a terminal event."""
    for e in rs.start():
        yield e
    async for ev in events:
        for e in rs.feed(ev):
            yield e
    if not rs.finished:
        for e in rs.feed(TurnDone("failed", TurnError("codex turn ended without completing"))):
            yield e


async def collect_response(rs: ResponseStream, events: AsyncIterator[Event]) -> dict[str, Any]:
    """Non-streaming: run the turn to the end and return the final Response."""
    async for _ in stream_response(rs, events):
        pass
    return rs.response


def encode_sse(event: dict[str, Any]) -> str:
    """One Responses stream event as an SSE frame (`event:` = its type)."""
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


# --- Chat Completions (step 9) ------------------------------------------------
#
# Derived from the Responses events above rather than from backend events, so
# ordering, failure handling and item bookkeeping live in one place. Chat has
# no reasoning items: they are dropped. Several message items (rare in
# chat-only mode) are joined with a blank line.

def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def chat_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if not usage:
        return None
    return {
        "prompt_tokens": usage["input_tokens"],
        "completion_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "prompt_tokens_details": {
            "cached_tokens": usage["input_tokens_details"]["cached_tokens"],
            "cache_write_tokens": usage["input_tokens_details"]["cache_write_tokens"],
        },
        "completion_tokens_details": {
            "reasoning_tokens": usage["output_tokens_details"]["reasoning_tokens"],
        },
    }


def _message_texts(response: dict[str, Any]) -> list[str]:
    return [
        "".join(c["text"] for c in item["content"] if c["type"] == "output_text")
        for item in response["output"] if item["type"] == "message"
    ]


def _chat_tool_call(item: dict[str, Any]) -> dict[str, Any]:
    return {"id": item["call_id"], "type": "function",
            "function": {"name": item["name"], "arguments": item["arguments"]}}


def chat_completion(response: dict[str, Any], *, completion_id: str | None = None) -> dict[str, Any]:
    """Non-streaming: a finished Responses `response` as a `chat.completion`."""
    calls = [_chat_tool_call(item) for item in response["output"] if item["type"] == "function_call"]
    texts = _message_texts(response)
    message = {"role": "assistant", "content": "\n\n".join(texts) if texts or not calls else None,
               "refusal": None, "annotations": []}
    if calls:
        message["tool_calls"] = calls
    out = {
        "id": completion_id or new_completion_id(),
        "object": "chat.completion",
        "created": response["created_at"],
        "model": response["model"],
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if calls else "stop",
            "logprobs": None,
        }],
        "usage": chat_usage(response["usage"]),
        "system_fingerprint": None,
    }
    if "session_id" in response:
        out["session_id"] = response["session_id"]
    return out


class ChatStream:
    """Responses stream events in, `chat.completion.chunk` dicts out.

    A Responses `error` event is not translated here: the route turns it into
    OpenAI's in-stream error payload, `{"error": {...}}`.
    """

    def __init__(self, *, model: str, include_usage: bool = False,
                 completion_id: str | None = None) -> None:
        self.id = completion_id or new_completion_id()
        self.created = int(time.time())
        self.model = model
        self.include_usage = include_usage
        self._messages = 0
        self._tool_calls: dict[str, int] = {}  # function_call item id → tool_calls index

    def _chunk(self, delta: dict[str, Any] | None, finish_reason: str | None = None,
               **extra: Any) -> dict[str, Any]:
        chunk = {
            "id": self.id, "object": "chat.completion.chunk", "created": self.created,
            "model": self.model, "system_fingerprint": None,
            "choices": [] if delta is None else [{
                "index": 0, "delta": delta, "finish_reason": finish_reason, "logprobs": None,
            }],
            **extra,
        }
        if self.include_usage and "usage" not in extra:
            chunk["usage"] = None  # as OpenAI does on every chunk but the last
        return chunk

    def feed(self, ev: dict[str, Any]) -> list[dict[str, Any]]:
        kind = ev["type"]
        if kind == "response.created":
            return [self._chunk({"role": "assistant", "content": "", "refusal": None})]
        if kind == "response.output_item.added" and ev["item"]["type"] == "message":
            self._messages += 1
            if self._messages > 1:
                return [self._chunk({"content": "\n\n"})]
        elif kind == "response.output_item.added" and ev["item"]["type"] == "function_call":
            item = ev["item"]
            index = self._tool_calls[item["id"]] = len(self._tool_calls)
            return [self._chunk({"tool_calls": [{
                "index": index, "id": item["call_id"], "type": "function",
                "function": {"name": item["name"], "arguments": ""},
            }]})]
        elif kind == "response.function_call_arguments.delta":
            return [self._chunk({"tool_calls": [{
                "index": self._tool_calls[ev["item_id"]], "function": {"arguments": ev["delta"]},
            }]})]
        elif kind == "response.output_text.delta":
            return [self._chunk({"content": ev["delta"]})]
        elif kind in ("response.completed", "response.incomplete"):
            reason = "tool_calls" if self._tool_calls else "stop"
            out = [self._chunk({}, finish_reason=reason)]
            if self.include_usage:
                out.append(self._chunk(None, usage=chat_usage(ev["response"]["usage"])))
            return out
        return []


def encode_chat_sse(payload: dict[str, Any] | str) -> str:
    """Chat Completions frames carry no `event:` line; the stream ends with [DONE]."""
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n"
