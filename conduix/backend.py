"""The only module that touches the openai-codex SDK.

Owns the shared AsyncCodex (one `codex app-server` child per server), pins
every thread to chat-only defaults, and maps Codex notifications to the small
event types below so the rest of Conduix never sees SDK types. A breaking SDK
release should only need changes here.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
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

from conduix.config import settings

log = logging.getLogger("conduix.backend")

BASE_INSTRUCTIONS = (
    "You are a helpful assistant answering over a plain chat API. "
    "You have no tools, no shell and no filesystem. Answer directly."
)

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
# should never appear; if one does, it is logged and dropped.
AGENTIC_ITEM_TYPES = frozenset({
    "commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall",
    "collabAgentToolCall", "subAgentActivity", "webSearch", "imageView",
    "imageGeneration", "sleep",
})


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
class TurnDone:
    status: str  # "completed" | "interrupted" | "failed"
    error: TurnError | None = None


Event = (
    MessageStarted | TextDelta | MessageDone
    | ReasoningStarted | ReasoningSummaryDelta | ReasoningDone
    | Usage | TurnError | TurnDone
)


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
        if kind in AGENTIC_ITEM_TYPES and started:
            log.warning("codex produced a %s item despite chat-only mode; dropped", kind)
        return []

    if method == "item/agentMessage/delta":
        return [TextDelta(payload.get("item_id", ""), payload.get("delta", ""))]

    if method == "item/reasoning/summaryTextDelta":
        return [ReasoningSummaryDelta(
            payload.get("item_id", ""),
            payload.get("summary_index", 0),
            payload.get("delta", ""),
        )]

    if method == "thread/tokenUsage/updated":
        last = (payload.get("token_usage") or {}).get("last") or {}
        return [Usage(
            input_tokens=last.get("input_tokens", 0),
            cached_input_tokens=last.get("cached_input_tokens", 0),
            output_tokens=last.get("output_tokens", 0),
            reasoning_output_tokens=last.get("reasoning_output_tokens", 0),
            cache_write_input_tokens=last.get("cache_write_input_tokens") or 0,
        )]

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


def _dump(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    params = getattr(obj, "params", None)  # UnknownNotification
    return params if isinstance(params, dict) else {}


def scrub_api_keys() -> list[str]:
    """Remove API-key env vars from this process so Codex bills the plan.

    CodexConfig.env is merged *over* os.environ, so it can override a key but
    not remove it; the Codex child inherits whatever this process holds.
    """
    return [k for k in API_KEY_VARS if os.environ.pop(k, None) is not None]


# --- backend -----------------------------------------------------------------

class Backend:
    def __init__(self) -> None:
        self._codex: AsyncCodex | None = None
        self.account_type: str | None = None
        self.plan_type: str | None = None
        self.codex_version: str | None = None
        self._models: list[dict[str, Any]] = []
        self._bg: set[asyncio.Task] = set()
        self.started_at = 0  # unix seconds; also the `created` time /v1/models reports

    @property
    def started(self) -> bool:
        return self._codex is not None

    async def start(self) -> None:
        s = settings()
        for k in scrub_api_keys():
            log.warning("removed %s from the environment; usage bills to the ChatGPT plan", k)
        s.workspace_dir.mkdir(parents=True, exist_ok=True)

        codex = AsyncCodex(CodexConfig(codex_bin=s.codex_bin, cwd=str(s.workspace_dir)))
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

    async def stop(self) -> None:
        codex, self._codex = self._codex, None
        if codex is not None:
            await codex.close()

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
    ) -> AsyncThread:
        s = settings()
        return await self.codex.thread_start(
            approval_mode=ApprovalMode.deny_all,
            sandbox=Sandbox.read_only,
            cwd=str(s.workspace_dir),
            base_instructions=BASE_INSTRUCTIONS,
            developer_instructions=developer_instructions or s.default_instructions,
            ephemeral=ephemeral,
            model=model or s.default_model,
            config=CHAT_ONLY_CONFIG,
        )

    async def run_turn(
        self,
        thread: AsyncThread,
        input: TurnInput | RunInput,
        *,
        model: str | None = None,
        effort: str | None = None,
        summary: str | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[Event]:
        """Run one turn and yield its events, ending with TurnDone.

        If the consumer stops early (client disconnect), the turn is
        interrupted so Codex doesn't keep spending plan quota.
        """
        handle = await thread.turn(
            _to_run_input(input), model=model, effort=effort or settings().default_effort, summary=summary,
            output_schema=output_schema,
        )
        done = False
        try:
            async for n in handle.stream():
                for ev in map_notification(n.method, _dump(n.payload)):
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
