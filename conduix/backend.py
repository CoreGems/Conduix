"""The only module that touches the openai-codex SDK.

Owns the shared AsyncCodex (one `codex app-server` child per server), pins
every thread to chat-only defaults, and maps Codex notifications to the small
event types below so the rest of Conduix never sees SDK types. A breaking SDK
release should only need changes here.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai_codex import (  # noqa: F401 - error types re-exported for errors.py
    ApprovalMode,
    AsyncCodex,
    AsyncThread,
    CodexConfig,
    CodexError,
    ImageInput,
    JsonRpcError,
    RunInput,
    Sandbox,
    ServerBusyError,
    TextInput,
    TransportClosedError,
)

from pydantic import BaseModel, ConfigDict

from conduix.config import settings

log = logging.getLogger("conduix.backend")

BASE_INSTRUCTIONS = (
    "You are a helpful assistant answering over a plain chat API. "
    "You have no tools, no shell and no filesystem. Answer directly."
)


def base_instructions(*, tools: bool = False, web_search: bool = False,
                      image_generation: bool = False) -> str:
    """The chat-only persona, telling the model which tools it really has.

    Found live: with BASE_INSTRUCTIONS ("You have no tools") the model refuses
    to call declared functions, so a thread with tools needs its own wording.
    """
    if not tools and not web_search and not image_generation:
        return BASE_INSTRUCTIONS
    parts = ["You are a helpful assistant answering over a chat API. "
             "You have no shell and no filesystem."]
    if tools:
        parts.append("Call the provided functions when they help answer.")
    if web_search:
        parts.append("You can search the web; cite the sources you use.")
    if image_generation:
        parts.append("You can generate images.")
    return " ".join(parts)

# Every agentic feature Codex 0.157 turns on by default. Accepted without
# errors in the step-2 probe (BRIEF.md §3.1).
CHAT_ONLY_CONFIG: dict[str, Any] = {
    "web_search": "disabled",
    "features.shell_tool": False,
    "features.unified_exec": False,
    "features.view_image": False,
    "features.apps": False,
    "features.plugins": False,
    "features.multi_agent": False,
    "features.browser_use": False,
    "features.computer_use": False,
    "features.image_generation": False,
    "features.sleep_tool": False,
    "features.tool_suggest": False,
    "features.skill_search": False,
    "features.goals": False,
    "features.hooks": False,
}

# Keys that would make Codex bill an API account instead of the ChatGPT plan.
API_KEY_VARS = ("OPENAI_API_KEY", "CODEX_API_KEY")

# Item types that mean Codex acted as an agent. With CHAT_ONLY_CONFIG they
# should never appear; if one does, it is logged and dropped. (`webSearch` and
# `imageGeneration` are allowed only on turns that asked for them;
# `dynamicToolCall` items mirror the client's own function calls, which arrive
# as ToolCall events instead.)
AGENTIC_ITEM_TYPES = frozenset({
    "commandExecution", "fileChange", "mcpToolCall",
    "collabAgentToolCall", "subAgentActivity", "imageView", "sleep",
})

# `item/tool/call` server requests are re-routed into their turn's
# notification stream under this method (see Backend._reader_loop).
TOOL_CALL_METHOD = "conduix/toolCall"


@dataclass(frozen=True, slots=True)
class ImagePart:
    """An image in turn input: a validated base64 `data:image/...` URL."""
    url: str


# A turn's input: text and image parts, in order.
TurnInput = list[str | ImagePart]


def _to_run_input(parts: TurnInput | RunInput) -> RunInput:
    if isinstance(parts, list) and all(isinstance(p, (str, ImagePart)) for p in parts):
        return [ImageInput(p.url) if isinstance(p, ImagePart) else TextInput(p) for p in parts]
    return parts


class BillingGuardError(RuntimeError):
    """Codex is not logged in with a ChatGPT account."""


class UnknownModelError(ValueError):
    pass


class UnsupportedEffortError(ValueError):
    pass


# --- internal events ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MessageStarted:
    item_id: str
    phase: str | None = None


@dataclass(frozen=True, slots=True)
class TextDelta:
    item_id: str
    delta: str


@dataclass(frozen=True, slots=True)
class MessageDone:
    item_id: str
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningStarted:
    item_id: str


@dataclass(frozen=True, slots=True)
class ReasoningSummaryDelta:
    item_id: str
    summary_index: int
    delta: str


@dataclass(frozen=True, slots=True)
class ReasoningDone:
    item_id: str
    summary: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    cache_write_input_tokens: int = 0


@dataclass(frozen=True, slots=True)
class TurnError:
    message: str
    # Raw `codexErrorInfo` (a string code or a dict carrying httpStatusCode).
    # Kept unparsed until step 8 records real quota/auth shapes.
    codex_error_info: Any = None
    additional_details: str | None = None
    resets_at: int | None = None  # filled in by run_turn for a plan-quota error


def is_quota_error(err: TurnError) -> bool:
    return err.codex_error_info == "usageLimitExceeded"


@dataclass(frozen=True, slots=True)
class TokenCount:
    """Raw `thread/tokenUsage/updated`: `last` model call and thread `total`.
    run_turn turns these into per-turn Usage (a turn with tool calls makes
    several model calls, each with its own update)."""
    last: Usage
    total: Usage


@dataclass(frozen=True, slots=True)
class ToolCall:
    """The model called a client-defined function (`item/tool/call`).

    Codex waits for the result; Conduix never sends one. The consumer ends
    its response with a function_call item, run_turn interrupts the turn, and
    the client's output comes back in a later request, replayed onto a fresh
    thread.
    """
    call_id: str
    name: str
    arguments: str  # JSON text, as the OpenAI APIs carry it
    request_id: Any = None


@dataclass(frozen=True, slots=True)
class WebSearchCall:
    item_id: str
    status: str  # "in_progress" | "completed"
    action: dict[str, Any] = field(default_factory=dict)  # {"type": "search", "query"} | ...


@dataclass(frozen=True, slots=True)
class ImageGenerationCall:
    """Codex's built-in image generation. Found live (step 13): one
    `imageGeneration` item, started with an empty result and completed about
    20 s later with the whole image. There are no partial images."""
    item_id: str
    status: str  # "in_progress" | "completed" | "failed"
    result: str = ""  # the image as base64 (PNG), once completed
    revised_prompt: str | None = None
    transparent_background: bool | None = None
    # Codex also writes the image under ~/.codex/generated_images; run_turn
    # deletes that copy.
    saved_path: str | None = None
    limit_exceeded: bool = False  # failed on a usage limit (`usageLimitExceeded`)
    resets_at: int | None = None  # when that limit resets (unix seconds), if known


@dataclass(frozen=True, slots=True)
class BlockedItem:
    """Codex started an agentic item (shell, file edit, tool call) despite
    chat-only mode. run_turn counts and drops it; it never reaches clients."""
    kind: str


@dataclass(frozen=True, slots=True)
class TurnDone:
    status: str  # "completed" | "interrupted" | "failed"
    error: TurnError | None = None


Event = (
    MessageStarted | TextDelta | MessageDone
    | ReasoningStarted | ReasoningSummaryDelta | ReasoningDone
    | ToolCall | WebSearchCall | ImageGenerationCall
    | Usage | TurnError | TurnDone
)


def _usage(d: dict[str, Any] | None) -> Usage:
    d = d or {}
    return Usage(
        input_tokens=d.get("input_tokens", 0),
        cached_input_tokens=d.get("cached_input_tokens", 0),
        output_tokens=d.get("output_tokens", 0),
        reasoning_output_tokens=d.get("reasoning_output_tokens", 0),
        cache_write_input_tokens=d.get("cache_write_input_tokens") or 0,
    )


def usage_minus(a: Usage, b: Usage) -> Usage:
    return Usage(*(max(0, x - y) for x, y in zip(dataclasses.astuple(a), dataclasses.astuple(b))))


def _turn_error(err: dict[str, Any]) -> TurnError:
    return TurnError(
        message=err.get("message", ""),
        codex_error_info=err.get("codex_error_info"),
        additional_details=err.get("additional_details"),
    )


def map_notification(method: str, payload: dict[str, Any]) -> list[Event]:
    """Map one Codex notification (payload dumped to snake_case) to events.

    Pure, so it is unit-tested against payloads recorded by the probe.
    """
    if method in ("item/started", "item/completed"):
        item = payload.get("item") or {}
        kind, item_id = item.get("type"), item.get("id", "")
        started = method == "item/started"
        if kind == "agentMessage":
            if started:
                return [MessageStarted(item_id, item.get("phase"))]
            return [MessageDone(item_id, item.get("text", ""))]
        if kind == "reasoning":
            if started:
                return [ReasoningStarted(item_id)]
            return [ReasoningDone(item_id, list(item.get("summary") or []))]
        if kind == "webSearch":
            return [WebSearchCall(item_id, "in_progress" if started else "completed",
                                  dict(item.get("action") or {}))]
        if kind == "imageGeneration":
            if started:
                return [ImageGenerationCall(item_id, "in_progress")]
            failure = item.get("failure") or {}
            ok = bool(item.get("result")) and not failure
            return [ImageGenerationCall(
                item_id, "completed" if ok else "failed",
                result=item["result"] if ok else "",
                revised_prompt=item.get("revised_prompt"),
                transparent_background=item.get("transparent_background"),
                saved_path=item.get("saved_path"),
                limit_exceeded=failure.get("type") == "usageLimitExceeded",
                resets_at=failure.get("resets_at"),
            )]
        if kind in AGENTIC_ITEM_TYPES and started:
            return [BlockedItem(kind)]
        return []

    if method == TOOL_CALL_METHOD:
        args = payload.get("arguments")
        return [ToolCall(
            call_id=payload.get("callId", ""),
            name=payload.get("tool", ""),
            arguments=args if isinstance(args, str) else json.dumps(args if args is not None else {}),
            request_id=payload.get("requestId"),
        )]

    if method == "item/agentMessage/delta":
        return [TextDelta(payload.get("item_id", ""), payload.get("delta", ""))]

    if method == "item/reasoning/summaryTextDelta":
        return [ReasoningSummaryDelta(
            payload.get("item_id", ""),
            payload.get("summary_index", 0),
            payload.get("delta", ""),
        )]

    if method == "thread/tokenUsage/updated":
        tu = payload.get("token_usage") or {}
        return [TokenCount(_usage(tu.get("last")), _usage(tu.get("total") or tu.get("last")))]

    if method == "error":
        if payload.get("will_retry"):
            log.info("codex retrying after error: %s", (payload.get("error") or {}).get("message"))
            return []
        return [_turn_error(payload.get("error") or {})]

    if method == "turn/completed":
        turn = payload.get("turn") or {}
        err = turn.get("error")
        return [TurnDone(turn.get("status", "completed"), _turn_error(err) if err else None)]

    return []


# --- thread realtime (speech output, step 14; BRIEF.md §3.3) ---------------------

# The only transport and protocol version a ChatGPT login can use: websocket
# needs API-key auth, and the AVAS voice backend refuses v1 and v2.
REALTIME_VERSION = "v3"


@dataclass(frozen=True, slots=True)
class RealtimeStarted:
    pass


@dataclass(frozen=True, slots=True)
class RealtimeAnswer:
    """The WebRTC SDP answer for the offer sent with thread/realtime/start."""
    sdp: str


@dataclass(frozen=True, slots=True)
class RealtimeTranscript:
    """One finished voice turn: what the voice said (or heard, role user)."""
    role: str
    text: str


@dataclass(frozen=True, slots=True)
class RealtimeError:
    message: str


@dataclass(frozen=True, slots=True)
class RealtimeClosed:
    reason: str


RealtimeEvent = RealtimeStarted | RealtimeAnswer | RealtimeTranscript | RealtimeError | RealtimeClosed


def map_realtime(method: str, params: dict[str, Any]) -> RealtimeEvent | None:
    """Map one raw (camelCase) thread/realtime/* notification. Pure."""
    if method == "thread/realtime/started":
        return RealtimeStarted()
    if method == "thread/realtime/sdp":
        return RealtimeAnswer(params.get("sdp", ""))
    if method == "thread/realtime/transcript/done":
        return RealtimeTranscript(params.get("role", ""), params.get("text", ""))
    if method == "thread/realtime/error":
        return RealtimeError(params.get("message", "") or "realtime session failed")
    if method == "thread/realtime/closed":
        return RealtimeClosed(params.get("reason", "") or "closed")
    return None  # deltas, items: not needed


class RealtimeSession:
    """A realtime voice session on its own chat-only thread (Backend.open_realtime).

    Codex holds the session; the client is the WebRTC peer and gets the audio
    on its media track. Events arrive on `events`.
    """

    def __init__(self, backend: Backend, thread_id: str) -> None:
        self._backend = backend
        self.thread_id = thread_id
        self.events: asyncio.Queue[RealtimeEvent] = asyncio.Queue()

    async def _request(self, method: str, params: dict[str, Any]) -> None:
        await self._backend.codex._client.request(method, {"threadId": self.thread_id, **params},
                                                  response_model=_AnyResult)

    async def append_text(self, text: str) -> None:
        """Send text to the voice model as a user message."""
        await self._request("thread/realtime/appendText", {"text": text})

    async def close(self) -> None:
        """Stop the session and unload its thread. Best effort, never raises."""
        try:
            await asyncio.wait_for(self._request("thread/realtime/stop", {}), timeout=5)
        except Exception as e:  # noqa: BLE001
            log.warning("could not stop realtime on thread %s: %s", self.thread_id, e)
        self._backend._realtime.pop(self.thread_id, None)
        await self._backend.close_thread(self.thread_id)


class _AnyResult(BaseModel):
    """The realtime methods' (empty) responses: not in the SDK's types."""
    model_config = ConfigDict(extra="allow")


def _dump(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True, warnings=False)
    params = getattr(obj, "params", None)  # UnknownNotification
    return params if isinstance(params, dict) else {}


def discard_saved_image(path: str | None) -> None:
    """Delete Codex's on-disk copy of a generated image. Best effort.

    The client gets the image in the response, so nothing would read the
    file again. Only `<codex home>/generated_images/<thread>/<item>` is
    touched.
    """
    if not path:
        return
    file = Path(path)
    if file.parent.parent.name != "generated_images":
        log.warning("left a generated image where it was, in an unexpected place: %s", path)
        return
    try:
        file.unlink(missing_ok=True)
    except OSError as e:
        log.warning("could not remove generated image %s: %s", path, e)
        return
    try:
        file.parent.rmdir()  # the thread's folder, once it is empty
    except OSError:
        pass  # it still holds another image of this thread


def scrub_api_keys() -> list[str]:
    """Remove API-key env vars from this process so Codex bills the plan.

    CodexConfig.env is merged *over* os.environ, so it can override a key but
    not remove it; the Codex child inherits whatever this process holds.
    """
    return [k for k in API_KEY_VARS if os.environ.pop(k, None) is not None]


# --- backend -----------------------------------------------------------------

class Backend:
    def __init__(self, *, launch_args: tuple[str, ...] | None = None) -> None:
        # launch_args replaces `codex app-server` (tests run a fake app-server).
        self._launch_args = launch_args
        self._codex: AsyncCodex | None = None
        # Agentic items Codex started despite CHAT_ONLY_CONFIG, by type.
        # Expected to stay empty; shown on /health.
        self.blocked_items: dict[str, int] = {}
        self.account_type: str | None = None
        self.plan_type: str | None = None
        self.codex_version: str | None = None
        self._models: list[dict[str, Any]] = []
        self._bg: set[asyncio.Task] = set()
        # thread id → callback for its thread/realtime/* notifications (called
        # on the reader thread). See open_realtime.
        self._realtime: dict[str, Any] = {}
        self.started_at = 0  # unix seconds; also the `created` time /v1/models reports

    @property
    def started(self) -> bool:
        return self._codex is not None

    async def start(self) -> None:
        s = settings()
        for k in scrub_api_keys():
            log.warning("removed %s from the environment; usage bills to the ChatGPT plan", k)
        s.workspace_dir.mkdir(parents=True, exist_ok=True)

        codex = AsyncCodex(CodexConfig(
            codex_bin=s.codex_bin, cwd=str(s.workspace_dir),
            launch_args_override=self._launch_args,
        ))
        self._install_reader_loop(codex)
        await codex.__aenter__()
        try:
            acct = (await codex.account()).account
            info = acct.root if acct is not None else None
            if info is None or info.type != "chatgpt":
                raise BillingGuardError(
                    f"Codex account is {getattr(info, 'type', 'not logged in')!r}, not 'chatgpt'. "
                    "Run `codex login` with your ChatGPT account."
                )
            self._codex = codex
            # The plan's model list is fixed for the process; cache it for validation.
            self._models = await self.models()
        except BaseException:
            self._codex = None
            await codex.close()
            raise

        self.account_type = info.type
        self.started_at = int(time.time())
        self.plan_type = getattr(info.plan_type, "value", str(info.plan_type))
        server = codex.metadata.serverInfo
        # serverInfo.version is a user-agent string: "0.157.1 (Windows ...) ..."
        self.codex_version = server.version.split()[0] if server and server.version else None
        log.info("codex %s ready: chatgpt plan %r", self.codex_version, self.plan_type)

    def _install_reader_loop(self, codex: AsyncCodex) -> None:
        """Replace the SDK's stdout reader loop (before the process starts).

        The SDK answers every server-initiated request at once from its single
        reader thread, and its default answer *accepts* command and file-change
        approvals. Conduix needs two things it can't do:
          * decline every approval (defence in depth: approvals are already
            off, the sandbox read-only and the shell/file tools disabled);
          * not answer `item/tool/call` at all. It is re-routed into the
            turn's notification stream (TOOL_CALL_METHOD) so run_turn yields a
            ToolCall in order with the rest of the turn.
          * hand `thread/realtime/*` notifications of a speech session to
            that session (open_realtime) instead of the SDK's global queue,
            which nothing reads.
        Everything else is routed exactly as the SDK's own loop does
        (openai_codex/client.py, CodexClient._reader_loop, 0.157.1).
        """
        sync = codex._client._sync

        def reader_loop() -> None:
            try:
                while True:
                    msg = sync._read_message()
                    method = msg.get("method")
                    if method is not None and "id" in msg:
                        reply = self._server_request(sync, msg)
                        if reply is not None:
                            sync._write_message(reply)
                        continue
                    if method is not None:
                        if isinstance(method, str) and method.startswith("thread/realtime/"):
                            params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
                            listener = self._realtime.get(params.get("threadId", ""))
                            if listener is not None:
                                listener(method, params)
                                continue
                        if isinstance(method, str):
                            sync._router.route_notification(
                                sync._coerce_notification(method, msg.get("params")))
                        continue
                    sync._router.route_response(msg)
            except BaseException as exc:  # noqa: BLE001 - mirrors the SDK loop
                sync._router.fail_all(exc)

        sync._reader_loop = reader_loop

    def _server_request(self, sync: Any, msg: dict[str, Any]) -> dict[str, Any] | None:
        """Answer (or defer) a request from app-server. Runs on the reader thread."""
        method, rid = msg["method"], msg["id"]
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        if method == "item/tool/call":
            sync._router.route_notification(sync._coerce_notification(
                TOOL_CALL_METHOD, {**params, "requestId": rid}))
            return None  # deferred: never answered, the turn gets interrupted
        if method.endswith("requestApproval"):
            self.blocked_items["approval"] = self.blocked_items.get("approval", 0) + 1
            log.warning("declined codex approval request %s", method)
            return {"id": rid, "result": {"decision": "decline"}}
        log.warning("codex sent an unhandled request %s; answered with an error", method)
        return {"id": rid, "error": {"code": -32601, "message": f"conduix does not handle {method}"}}

    async def stop(self) -> None:
        codex, self._codex = self._codex, None
        if codex is None:
            return
        try:
            await codex.close()
        except Exception as e:  # noqa: BLE001 - shutdown must not raise
            # e.g. OSError closing stdin of an app-server that already died
            # (the SDK's close() doesn't expect that on Windows).
            log.warning("closing codex app-server: %s: %s", type(e).__name__, e)

    @property
    def codex(self) -> AsyncCodex:
        if self._codex is None:
            raise RuntimeError("backend not started")
        return self._codex

    async def status(self) -> dict[str, Any]:
        if self._codex is None:
            return {"codex": "not_started"}
        try:
            acct = (await asyncio.wait_for(self._codex.account(), timeout=5)).account
        except Exception as e:  # noqa: BLE001 - health must not raise
            return {"codex": "error", "detail": f"{type(e).__name__}: {e}"}
        info = acct.root if acct is not None else None
        rl = await self.rate_limits() or {}
        usage = {
            name: {k: w.get(k) for k in ("used_percent", "window_duration_mins", "resets_at")}
            for name in ("primary", "secondary") if (w := rl.get(name))
        }
        return {
            "codex": "ok",
            "codex_version": self.codex_version,
            "logged_in": info is not None,
            "account_type": getattr(info, "type", None),
            "plan_type": getattr(getattr(info, "plan_type", None), "value", None),
            "usage": usage or None,
            "blocked_agent_items": dict(self.blocked_items),
        }

    async def models(self) -> list[dict[str, Any]]:
        resp = await self.codex.models()
        return [
            {
                "id": m.id,
                "is_default": m.is_default,
                "default_effort": m.default_reasoning_effort.value,
                "efforts": [o.reasoning_effort.value for o in m.supported_reasoning_efforts],
                "input_modalities": [getattr(x, "value", x) for x in (m.input_modalities or [])],
            }
            for m in resp.data
        ]

    @property
    def cached_models(self) -> list[dict[str, Any]]:
        """The plan's models as read at startup."""
        return list(self._models)

    def supports_images(self, model: str | None) -> bool:
        model = model or settings().default_model
        entry = next((m for m in self._models
                      if (m["id"] == model if model else m["is_default"])), None)
        return entry is None or "image" in entry["input_modalities"]

    def resolve_model(self, model: str | None, effort: str | None = None) -> str | None:
        """Validate a model id and effort against the plan's cached model list.

        Returns the model id to pass to Codex (None → Codex's default). Raises
        UnknownModelError / UnsupportedEffortError for a 400.
        """
        model = model or settings().default_model
        if not self._models:  # not started (unit tests) → nothing to check against
            return model
        if model is None:
            entry = next((m for m in self._models if m["is_default"]), None)
        else:
            entry = next((m for m in self._models if m["id"] == model), None)
            if entry is None:
                ids = ", ".join(m["id"] for m in self._models)
                raise UnknownModelError(f"model {model!r} is not available on this plan ({ids})")
        if effort is not None and entry is not None and effort not in entry["efforts"]:
            raise UnsupportedEffortError(
                f"effort {effort!r} is not supported by {entry['id']} "
                f"(supported: {', '.join(entry['efforts'])})"
            )
        return model

    async def rate_limits(self) -> dict[str, Any] | None:
        """The plan's usage windows (`account/rateLimits/read`), or None.

        Read-only. The same RPC family can *consume* the account's free
        rate-limit-reset credits; Conduix never does that.
        """
        if self._codex is None:
            return None
        from openai_codex.generated.v2_all import GetAccountRateLimitsResponse

        try:
            r = await asyncio.wait_for(self._codex._client.request(
                "account/rateLimits/read", None, response_model=GetAccountRateLimitsResponse,
            ), timeout=5)
        except Exception as e:  # noqa: BLE001 - informational only
            log.warning("could not read rate limits: %s", e)
            return None
        return r.rate_limits.model_dump(mode="json", exclude_none=True)

    async def quota_resets_at(self) -> int | None:
        """When the exhausted usage window resets (unix seconds), if known."""
        rl = await self.rate_limits()
        windows = [w for w in ((rl or {}).get("primary"), (rl or {}).get("secondary")) if w]
        full = [w for w in windows if w.get("used_percent", 0) >= 100]
        resets = [w["resets_at"] for w in (full or windows) if w.get("resets_at")]
        return max(resets) if resets else None

    def model_name(self, model: str | None) -> str:
        """The id to report for `model` (None → the plan's default model)."""
        model = model or settings().default_model
        if model:
            return model
        return next((m["id"] for m in self._models if m["is_default"]), "codex-default")

    async def inject_items(self, thread: AsyncThread, items: list[dict[str, Any]]) -> None:
        """Append Responses API items to a thread's model-visible history.

        Used to replay conversation history as real user/assistant messages.
        Raw `thread/inject_items` request: the SDK has no wrapper for it.
        """
        if not items:
            return
        from openai_codex.generated.v2_all import ThreadInjectItemsResponse

        await self.codex._client.request(
            "thread/inject_items", {"threadId": thread.id, "items": items},
            response_model=ThreadInjectItemsResponse,
        )

    async def close_thread(self, thread_id: str) -> None:
        """Unload a thread from app-server. Best effort.

        The SDK has no wrapper for `thread/unsubscribe`, so this goes through
        its raw typed request (as does inject_items).
        """
        if self._codex is None:
            return
        from openai_codex.generated.v2_all import ThreadUnsubscribeResponse

        try:
            await self._codex._client.request(
                "thread/unsubscribe", {"threadId": thread_id},
                response_model=ThreadUnsubscribeResponse,
            )
        except Exception as e:  # noqa: BLE001 - closing must not raise
            log.warning("could not unsubscribe thread %s: %s", thread_id, e)

    async def start_thread(
        self,
        *,
        model: str | None = None,
        developer_instructions: str | None = None,
        ephemeral: bool = True,
        tools: list[dict[str, Any]] | None = None,
        web_search: bool = False,
        image_generation: bool = False,
    ) -> AsyncThread:
        """Start a chat-only thread.

        `tools` are Codex dynamic tool specs ({type: "function", name,
        description, inputSchema}); they are fixed for the thread's life.
        `web_search` and `image_generation` turn Codex's own web search and
        image generation back on for this thread.
        """
        s = settings()
        config = dict(CHAT_ONLY_CONFIG)
        if web_search:
            config["web_search"] = s.web_search_mode
        if image_generation:
            config["features.image_generation"] = True
        kwargs = dict(
            approval_mode=ApprovalMode.deny_all,
            sandbox=Sandbox.read_only,
            cwd=str(s.workspace_dir),
            base_instructions=base_instructions(tools=bool(tools), web_search=web_search,
                                                image_generation=image_generation),
            developer_instructions=developer_instructions or s.default_instructions,
            ephemeral=ephemeral,
            model=model or s.default_model,
            config=config,
        )
        if not tools:
            return await self.codex.thread_start(**kwargs)

        # `dynamicTools` is accepted by app-server (verified live) but missing
        # from the SDK's ThreadStartParams, so build the same params and add it.
        from openai_codex._approval_mode import _approval_mode_settings
        from openai_codex._sandbox import _sandbox_mode
        from openai_codex.generated.v2_all import ThreadStartParams, ThreadStartResponse

        policy, reviewer = _approval_mode_settings(kwargs.pop("approval_mode"))
        params = ThreadStartParams(
            approval_policy=policy, approvals_reviewer=reviewer,
            sandbox=_sandbox_mode(kwargs.pop("sandbox")), **kwargs,
        ).model_dump(mode="json", by_alias=True, exclude_none=True)
        params["dynamicTools"] = tools
        started = await self.codex._client.request(
            "thread/start", params, response_model=ThreadStartResponse)
        return AsyncThread(self.codex, started.thread.id)

    async def open_realtime(self, *, sdp: str, prompt: str,
                            voice: str | None = None) -> RealtimeSession:
        """Start a realtime voice session (audio out) over WebRTC.

        `sdp` is the caller's WebRTC offer; the answer arrives as a
        RealtimeAnswer event, then the voice speaks on the media track. The
        thread is chat-only, ephemeral and used for nothing else. The caller
        must close() the session.
        """
        s = settings()
        thread = await self.codex.thread_start(
            approval_mode=ApprovalMode.deny_all,
            sandbox=Sandbox.read_only,
            cwd=str(s.workspace_dir),
            base_instructions=prompt,
            ephemeral=True,
            config=dict(CHAT_ONLY_CONFIG),
        )
        session = RealtimeSession(self, thread.id)
        loop = asyncio.get_running_loop()

        def deliver(method: str, params: dict[str, Any]) -> None:  # on the reader thread
            ev = map_realtime(method, params)
            if ev is not None:
                loop.call_soon_threadsafe(session.events.put_nowait, ev)

        self._realtime[thread.id] = deliver
        params: dict[str, Any] = {
            "outputModality": "audio",
            "transport": {"type": "webrtc", "sdp": sdp},
            "version": REALTIME_VERSION,
            "prompt": prompt,
            "includeStartupContext": False,
        }
        if voice:
            params["voice"] = voice
        try:
            await session._request("thread/realtime/start", params)
        except BaseException:
            await session.close()
            raise
        return session

    async def run_turn(
        self,
        thread: AsyncThread,
        input: TurnInput | RunInput,
        *,
        model: str | None = None,
        effort: str | None = None,
        summary: str | None = None,
        output_schema: dict[str, Any] | None = None,
        allow_web_search: bool = False,
        allow_image_generation: bool = False,
    ) -> AsyncIterator[Event]:
        """Run one turn and yield its events, ending with TurnDone.

        Usage events are cumulative for the turn (all its model calls). If the
        consumer stops early (client disconnect, or a ToolCall it hands to the
        client), the turn is interrupted so Codex doesn't keep spending plan
        quota.
        """
        handle = await thread.turn(
            _to_run_input(input), model=model, effort=effort or settings().default_effort, summary=summary,
            output_schema=output_schema,
        )
        done = False
        before: Usage | None = None  # thread total before this turn
        try:
            async for n in handle.stream():
                for ev in map_notification(n.method, _dump(n.payload)):
                    if isinstance(ev, WebSearchCall) and not allow_web_search:
                        ev = BlockedItem("webSearch")
                    if isinstance(ev, ImageGenerationCall):
                        discard_saved_image(ev.saved_path)
                        if not allow_image_generation:
                            if ev.status != "in_progress":
                                continue  # counted once, when it started
                            ev = BlockedItem("imageGeneration")
                    if isinstance(ev, BlockedItem):
                        self.blocked_items[ev.kind] = self.blocked_items.get(ev.kind, 0) + 1
                        log.warning("codex started a %s item despite chat-only mode; dropped",
                                    ev.kind)
                        continue
                    if isinstance(ev, TokenCount):
                        if before is None:
                            before = usage_minus(ev.total, ev.last)
                        ev = usage_minus(ev.total, before)
                    done = done or isinstance(ev, TurnDone)
                    yield await self._with_reset_time(ev)
        finally:
            if not done:
                # A separate task: on client disconnect Starlette cancels via
                # an anyio cancel scope, which would cancel an await here too.
                task = asyncio.get_running_loop().create_task(self._interrupt(handle))
                self._bg.add(task)
                task.add_done_callback(self._bg.discard)

    async def _with_reset_time(self, ev: Event) -> Event:
        """Attach the quota reset time to a plan-quota error."""
        if isinstance(ev, TurnError) and is_quota_error(ev):
            return dataclasses.replace(ev, resets_at=await self.quota_resets_at())
        if isinstance(ev, TurnDone) and ev.error and is_quota_error(ev.error):
            err = dataclasses.replace(ev.error, resets_at=await self.quota_resets_at())
            return dataclasses.replace(ev, error=err)
        return ev

    @staticmethod
    async def _interrupt(handle: Any) -> None:
        try:
            await asyncio.wait_for(handle.interrupt(), timeout=5)
            log.info("interrupted turn %s (consumer went away)", handle.id)
        except Exception as e:  # noqa: BLE001 - best effort
            log.warning("could not interrupt turn %s: %s", handle.id, e)


backend = Backend()
