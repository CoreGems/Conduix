"""OpenAI-shaped errors: {"error": {"message", "type", "param", "code"}}.

Codex failures are classified from its own `codexErrorInfo` enum (the
app-server protocol schema in openai_codex.generated.v2_all), never by
matching message text:

  usageLimitExceeded     → 429, plan quota; message carries the reset time,
                           `x-should-retry: false` stops the openai SDK
                           retrying something that can't clear until reset
  rateLimitExceeded      → 429, transient
  serverOverloaded       → 503
  unauthorized           → 401, `codex login` needed
  contextWindowExceeded  → 400 context_length_exceeded
  policy / badRequest    → 400
  connection failures    → by their httpStatusCode (429/401), else 502
  anything else          → 500

When the upstream OpenAI API itself rejects the request (e.g. an invalid
json_schema), Codex passes its error envelope through as the message text:
{"type":"error","error":{"type","code","message","param"},"status":400}.
That is OpenAI's documented shape, so it is unwrapped and passed through with
its own status, code and param.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai_codex import CodexError, JsonRpcError, ServerBusyError, TransportClosedError

from conduix.backend import TurnError, UnknownModelError, UnsupportedEffortError


class APIError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status, self.message, self.type, self.param, self.code = (
            status, message, type, param, code
        )
        self.headers = headers or {}

    def body(self) -> dict:
        return {"error": {
            "message": self.message, "type": self.type, "param": self.param, "code": self.code,
        }}


def not_found(message: str, *, param: str | None = None) -> APIError:
    return APIError(404, message, param=param, code="not_found")


# --- Codex error classification -------------------------------------------------

@dataclass(frozen=True)
class Kind:
    status: int
    type: str
    code: str
    # `Response.error.code` must be one of OpenAI's ResponseError literals.
    response_code: str
    prefix: str = ""


USAGE_LIMIT = Kind(429, "rate_limit_exceeded", "usage_limit_exceeded", "rate_limit_exceeded",
                   "ChatGPT plan usage limit reached")
RATE_LIMIT = Kind(429, "rate_limit_exceeded", "rate_limit_exceeded", "rate_limit_exceeded",
                  "Rate limited upstream; retry shortly")
OVERLOADED = Kind(503, "server_error", "server_overloaded", "server_error",
                  "Upstream overloaded; retry shortly")
UNAUTHORIZED = Kind(401, "authentication_error", "codex_unauthorized", "server_error",
                    "Codex is not authorized; run `codex login` on the server")
UPSTREAM = Kind(502, "server_error", "upstream_error", "server_error",
                "Codex could not reach the model")
SERVER = Kind(500, "server_error", "server_error", "server_error")

_BY_CODE = {
    "usageLimitExceeded": USAGE_LIMIT,
    "rateLimitExceeded": RATE_LIMIT,
    "serverOverloaded": OVERLOADED,
    "unauthorized": UNAUTHORIZED,
    "contextWindowExceeded": Kind(400, "invalid_request_error", "context_length_exceeded",
                                  "invalid_prompt", "Conversation exceeds the model's context window"),
    "sessionBudgetExceeded": Kind(400, "invalid_request_error", "session_budget_exceeded",
                                  "invalid_prompt", "Conversation exceeds Codex's session budget"),
    "cyberPolicy": Kind(400, "invalid_request_error", "content_policy_violation",
                        "invalid_prompt"),
    "misalignmentPolicyViolation": Kind(400, "invalid_request_error",
                                        "misalignment_policy_violation",
                                        "misalignment_policy_violation"),
    "badRequest": Kind(400, "invalid_request_error", "invalid_request", "invalid_prompt"),
}


def _by_http_status(status: int | None) -> Kind:
    if status == 429:
        return RATE_LIMIT
    if status in (401, 403):
        return UNAUTHORIZED
    return UPSTREAM


def classify(info: Any) -> Kind:
    """Kind for a raw `codexErrorInfo`: a string code, or a one-key dict like
    {"httpConnectionFailed": {"http_status_code": 502}}."""
    if isinstance(info, str):
        return _BY_CODE.get(info, SERVER)
    if isinstance(info, dict) and len(info) == 1:
        detail = next(iter(info.values()))
        if isinstance(detail, dict):
            return _by_http_status(detail.get("http_status_code", detail.get("httpStatusCode")))
    return SERVER


def _reset_text(resets_at: int) -> str:
    when = datetime.fromtimestamp(resets_at, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"resets at {when}"


def _upstream_envelope(message: str) -> tuple[int | None, dict[str, Any]] | None:
    """(status, inner error) if `message` is an OpenAI API error envelope."""
    if not message.lstrip().startswith("{"):
        return None
    try:
        body = json.loads(message)
    except ValueError:
        return None
    inner = body.get("error") if isinstance(body, dict) else None
    if not isinstance(inner, dict) or not isinstance(inner.get("message"), str):
        return None
    status = body.get("status")
    return (status if isinstance(status, int) else None), inner


def _response_code_for(status: int) -> str:
    return {429: "rate_limit_exceeded"}.get(status, "invalid_prompt" if status < 500 else "server_error")


def resolve(err: TurnError) -> tuple[Kind, str, str | None]:
    """(kind, client-facing message, param) for a failed turn."""
    envelope = _upstream_envelope(err.message or "")
    if envelope is not None:
        status, inner = envelope
        kind = classify(err.codex_error_info)
        if status is not None:
            kind = Kind(status, inner.get("type") or kind.type, inner.get("code") or kind.code,
                        _response_code_for(status))
        elif inner.get("code"):
            kind = replace(kind, code=inner["code"])
        return kind, inner["message"], inner.get("param")

    kind = classify(err.codex_error_info)
    message = ": ".join(x for x in (kind.prefix, err.message) if x) or "Codex turn failed"
    if kind is USAGE_LIMIT and err.resets_at:
        message += f" ({_reset_text(err.resets_at)})"
    return kind, message, None


def describe(err: TurnError) -> tuple[Kind, str]:
    kind, message, _ = resolve(err)
    return kind, message


def from_turn_error(err: TurnError) -> APIError:
    kind, message, param = resolve(err)
    headers = {}
    if kind is USAGE_LIMIT:
        headers["x-should-retry"] = "false"
        if err.resets_at:
            headers["retry-after"] = str(max(0, int(err.resets_at - time.time())))
    return APIError(kind.status, message, type=kind.type, code=kind.code, param=param,
                    headers=headers)


def from_exception(exc: BaseException) -> APIError:
    """An exception from the SDK / app-server (not a failed turn)."""
    if isinstance(exc, APIError):
        return exc
    if isinstance(exc, ServerBusyError):
        return APIError(503, f"{OVERLOADED.prefix}: {exc.message}", type="server_error",
                        code=OVERLOADED.code)
    if isinstance(exc, JsonRpcError):
        data = exc.data if isinstance(exc.data, dict) else {}
        info = data.get("codexErrorInfo", data.get("codex_error_info"))
        if info is not None:
            return from_turn_error(TurnError(exc.message, info))
        return APIError(502, f"Codex rejected the request: {exc.message}",
                        type="server_error", code="upstream_error")
    if isinstance(exc, TransportClosedError):
        return APIError(503, "the Codex app-server process went away; restart Conduix",
                        type="server_error", code="codex_unavailable")
    if isinstance(exc, CodexError):
        return APIError(502, f"Codex error: {exc}", type="server_error", code="upstream_error")
    return APIError(500, f"{type(exc).__name__}: {exc}", type="server_error",
                    code="server_error")


def install(app: FastAPI) -> None:
    def respond(err: APIError) -> JSONResponse:
        return JSONResponse(err.body(), status_code=err.status, headers=err.headers)

    @app.exception_handler(APIError)
    async def _api_error(_req: Request, exc: APIError) -> JSONResponse:
        return respond(exc)

    @app.exception_handler(CodexError)
    async def _codex_error(_req: Request, exc: CodexError) -> JSONResponse:
        return respond(from_exception(exc))

    @app.exception_handler(UnknownModelError)
    async def _unknown_model(_req: Request, exc: UnknownModelError) -> JSONResponse:
        return respond(APIError(400, str(exc), param="model", code="model_not_found"))

    @app.exception_handler(UnsupportedEffortError)
    async def _bad_effort(_req: Request, exc: UnsupportedEffortError) -> JSONResponse:
        return respond(APIError(400, str(exc), param="reasoning.effort",
                                code="unsupported_value"))

    @app.exception_handler(RequestValidationError)
    async def _validation(_req: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = [str(x) for x in first.get("loc", ()) if x != "body"]
        return respond(APIError(400, first.get("msg", "invalid request"),
                                param=".".join(loc) or None))
