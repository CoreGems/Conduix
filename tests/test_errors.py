"""Offline tests for error classification (errors.py) and quota reset lookup."""
import json
import time

import pytest
from openai_codex import JsonRpcError, ServerBusyError, TransportClosedError

from conduix.backend import Backend, TurnDone, TurnError
from conduix.errors import APIError, classify, describe, from_exception, from_turn_error


@pytest.mark.parametrize("info, status, code, response_code", [
    ("usageLimitExceeded", 429, "usage_limit_exceeded", "rate_limit_exceeded"),
    ("rateLimitExceeded", 429, "rate_limit_exceeded", "rate_limit_exceeded"),
    ("serverOverloaded", 503, "server_overloaded", "server_error"),
    ("unauthorized", 401, "codex_unauthorized", "server_error"),
    ("contextWindowExceeded", 400, "context_length_exceeded", "invalid_prompt"),
    ("misalignmentPolicyViolation", 400, "misalignment_policy_violation",
     "misalignment_policy_violation"),
    ("internalServerError", 500, "server_error", "server_error"),
    ("somethingNewInAFutureCodex", 500, "server_error", "server_error"),
    (None, 500, "server_error", "server_error"),
    ({"httpConnectionFailed": {"http_status_code": 429}}, 429, "rate_limit_exceeded",
     "rate_limit_exceeded"),
    ({"responseStreamDisconnected": {"httpStatusCode": 401}}, 401, "codex_unauthorized",
     "server_error"),
    ({"responseTooManyFailedAttempts": {"http_status_code": 502}}, 502, "upstream_error",
     "server_error"),
    ({"httpConnectionFailed": {}}, 502, "upstream_error", "server_error"),
])
def test_classify(info, status, code, response_code):
    kind = classify(info)
    assert (kind.status, kind.code, kind.response_code) == (status, code, response_code)


def test_response_codes_are_valid_openai_literals():
    from typing import get_args

    from openai.types.responses import ResponseError

    allowed = set(get_args(ResponseError.model_fields["code"].annotation))
    from conduix import errors
    for kind in [*errors._BY_CODE.values(), errors.UPSTREAM, errors.SERVER]:
        assert kind.response_code in allowed, kind


def test_quota_message_and_headers():
    resets = int(time.time()) + 3600
    err = from_turn_error(TurnError("You've hit your usage limit.", "usageLimitExceeded",
                                    resets_at=resets))
    assert err.status == 429 and err.code == "usage_limit_exceeded"
    assert err.message.startswith("ChatGPT plan usage limit reached: You've hit your usage limit.")
    assert "resets at" in err.message and "UTC" in err.message
    assert err.headers["x-should-retry"] == "false"
    assert 3590 <= int(err.headers["retry-after"]) <= 3600


def test_quota_without_reset_time():
    err = from_turn_error(TurnError("limit", "usageLimitExceeded"))
    assert "resets at" not in err.message and "retry-after" not in err.headers


def test_transient_errors_stay_retryable():
    assert "x-should-retry" not in from_turn_error(TurnError("slow down", "rateLimitExceeded")).headers


def test_describe_without_message():
    assert describe(TurnError("", None)) [1] == "Codex turn failed"


def test_from_exception():
    assert from_exception(ServerBusyError(-32001, "busy", "server_overloaded")).status == 503
    assert from_exception(TransportClosedError("gone")).code == "codex_unavailable"
    rpc = JsonRpcError(-32000, "limit", {"codexErrorInfo": "usageLimitExceeded"})
    assert from_exception(rpc).status == 429
    assert from_exception(JsonRpcError(-32000, "nope")).status == 502
    assert from_exception(ValueError("x")).status == 500
    same = APIError(404, "x")
    assert from_exception(same) is same


# --- quota reset lookup (backend) ------------------------------------------------

async def test_quota_resets_at_prefers_exhausted_window(monkeypatch):
    b = Backend()

    async def fake_limits():
        return {"primary": {"used_percent": 100, "resets_at": 500},
                "secondary": {"used_percent": 40, "resets_at": 900}}

    monkeypatch.setattr(b, "rate_limits", fake_limits)
    assert await b.quota_resets_at() == 500


async def test_quota_resets_at_falls_back_to_any_window(monkeypatch):
    b = Backend()

    async def fake_limits():
        return {"primary": {"used_percent": 4, "resets_at": 1791131515}}

    monkeypatch.setattr(b, "rate_limits", fake_limits)
    assert await b.quota_resets_at() == 1791131515


async def test_quota_resets_at_unknown(monkeypatch):
    b = Backend()

    async def none():
        return None

    monkeypatch.setattr(b, "rate_limits", none)
    assert await b.quota_resets_at() is None


async def test_quota_errors_get_reset_time_attached(monkeypatch):
    b = Backend()

    async def resets():
        return 1234

    monkeypatch.setattr(b, "quota_resets_at", resets)
    quota = TurnError("limit", "usageLimitExceeded")
    assert (await b._with_reset_time(quota)).resets_at == 1234
    done = await b._with_reset_time(TurnDone("failed", quota))
    assert done.error.resets_at == 1234
    other = TurnError("boom", "internalServerError")
    assert (await b._with_reset_time(other)) is other


# Captured live 2026-09-27: an invalid json_schema rejected by the upstream API.
UPSTREAM_400 = json.dumps({
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "code": "invalid_json_schema",
        "message": "Invalid schema for response_format 'codex_output_schema': "
                   "'strng' is not valid under any of the given schemas.",
        "param": "text.format.schema",
    },
    "status": 400,
}, indent=2)


def test_upstream_envelope_is_unwrapped():
    err = from_turn_error(TurnError(UPSTREAM_400, None))
    assert (err.status, err.type, err.code, err.param) == (
        400, "invalid_request_error", "invalid_json_schema", "text.format.schema")
    assert err.message.startswith("Invalid schema for response_format")


def test_upstream_envelope_response_code_is_valid():
    kind, message = describe(TurnError(UPSTREAM_400, "badRequest"))
    assert kind.response_code == "invalid_prompt" and "{" not in message


def test_non_envelope_json_like_text_is_left_alone():
    err = from_turn_error(TurnError("{not json", "internalServerError"))
    assert err.status == 500 and err.message.endswith("{not json")
