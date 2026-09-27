"""Offline tests of conduix/backend.py + the real openai-codex SDK, against a
fake app-server speaking the real JSON-RPC protocol (tests/fake_app_server.py).

These cover what the fake-backend tests can't: the SDK calls, the exact
thread/turn parameters that reach Codex (the chat-only lockdown), the raw
RPCs (inject_items, unsubscribe, rateLimits/read), interrupts, and what
happens when the app-server dies.
"""
import asyncio
import sys
from pathlib import Path

import openai
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from conduix import backend as backend_mod
from conduix.backend import (
    CHAT_ONLY_CONFIG,
    Backend,
    BillingGuardError,
    ImagePart,
    MessageDone,
    MessageStarted,
    TextDelta,
    TransportClosedError,
    TurnDone,
    TurnError,
    Usage,
)

FAKE = (sys.executable, str(Path(__file__).parent / "fake_app_server.py"))


class Stats(BaseModel):
    model_config = ConfigDict(extra="allow")


async def stats(be: Backend) -> dict:
    return (await be.codex._client.request("fake/stats", None, response_model=Stats)).model_dump()


@pytest.fixture
async def be():
    b = Backend(launch_args=FAKE)
    await b.start()
    try:
        yield b
    finally:
        await b.stop()


async def turn(be, thread, text, **kw):
    return [ev async for ev in be.run_turn(thread, [text] if isinstance(text, str) else text, **kw)]


# --- startup / billing guard -------------------------------------------------------

async def test_start_reads_account_and_models(be):
    assert be.account_type == "chatgpt" and be.plan_type == "plus"
    assert be.codex_version == "0.157.1"
    assert [m["id"] for m in be.cached_models][:2] == ["gpt-6-astra", "gpt-6-sol"]
    status = await be.status()
    assert status["codex"] == "ok" and status["usage"]["primary"]["used_percent"] == 100
    assert status["blocked_agent_items"] == {}


async def test_billing_guard_refuses_no_login(monkeypatch):
    monkeypatch.setenv("FAKE_ACCOUNT", "none")
    b = Backend(launch_args=FAKE)
    with pytest.raises(BillingGuardError):
        await b.start()
    assert not b.started


async def test_api_key_never_reaches_codex(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    b = Backend(launch_args=FAKE)
    await b.start()
    try:
        assert (await stats(b))["api_key_in_env"] is False
    finally:
        await b.stop()


# --- chat-only lockdown reaches Codex -------------------------------------------------

async def test_thread_start_parameters(be):
    await be.start_thread(model="gpt-6-sol", developer_instructions="be brief")
    [p] = (await stats(be))["thread_start_params"]
    assert p["approvalPolicy"] == "never"  # ApprovalMode.deny_all
    assert p["sandbox"] == "read-only"
    assert p["ephemeral"] is True
    assert p["config"] == CHAT_ONLY_CONFIG
    assert p["baseInstructions"] == backend_mod.BASE_INSTRUCTIONS
    assert p["developerInstructions"] == "be brief"
    assert p["model"] == "gpt-6-sol"
    assert Path(p["cwd"]).name == "workspace"


async def test_turn_parameters(be):
    thread = await be.start_thread()
    schema = {"type": "object"}
    await turn(be, thread, "hi", model="gpt-6-sol", effort="high", summary="auto",
               output_schema=schema)
    [p] = (await stats(be))["turn_start_params"]
    assert (p["model"], p["effort"], p["summary"], p["outputSchema"]) == (
        "gpt-6-sol", "high", "auto", schema)


# --- turns ----------------------------------------------------------------------

async def test_turn_event_sequence(be):
    thread = await be.start_thread()
    events = await turn(be, thread, "hi")
    assert [type(e) for e in events] == [MessageStarted, TextDelta, TextDelta, MessageDone,
                                         Usage, TurnDone]
    assert events[3].text == "ctx=user:hi" and events[-1].status == "completed"
    assert events[4] == Usage(100, 80, 5, 0, 0)


async def test_inject_items_become_history(be):
    thread = await be.start_thread()
    await be.inject_items(thread, [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "A"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "B"}]},
    ])
    events = await turn(be, thread, "C")
    assert events[3].text == "ctx=user:A;assistant:B;user:C"


async def test_image_parts_reach_codex_as_images(be):
    thread = await be.start_thread()
    events = await turn(be, thread, ["look", ImagePart("data:image/png;base64,AAAA")])
    assert events[3].text == "ctx=user:look;user:<image 26>"


async def test_agentic_items_are_blocked_and_counted(be):
    thread = await be.start_thread()
    events = await turn(be, thread, "TOOL please run dir")
    assert events[-1].status == "completed"
    assert be.blocked_items == {"commandExecution": 1}
    assert (await be.status())["blocked_agent_items"] == {"commandExecution": 1}


async def test_retryable_error_is_dropped(be):
    thread = await be.start_thread()
    events = await turn(be, thread, "RETRY")
    assert not any(isinstance(e, TurnError) for e in events)
    assert events[-1].status == "completed"


async def test_quota_error_gets_reset_time_from_rate_limits(be):
    thread = await be.start_thread()
    events = await turn(be, thread, "QUOTA")
    errors = [e for e in events if isinstance(e, TurnError)]
    assert errors and errors[0].codex_error_info == "usageLimitExceeded"
    assert errors[0].resets_at == 1791131515
    assert events[-1].status == "failed" and events[-1].error.resets_at == 1791131515


async def test_consumer_leaving_interrupts_the_turn(be):
    thread = await be.start_thread()
    gen = be.run_turn(thread, ["SLOW"])
    async for ev in gen:
        if isinstance(ev, TextDelta):
            break
    await gen.aclose()
    for _ in range(100):
        if (await stats(be))["interrupted"]:
            break
        await asyncio.sleep(0.05)
    assert len((await stats(be))["interrupted"]) == 1


async def test_close_thread_unsubscribes(be):
    thread = await be.start_thread()
    await be.close_thread(thread.id)
    assert (await stats(be))["unsubscribed"] == [thread.id]


async def test_app_server_crash_mid_turn_raises_not_hangs(be):
    thread = await be.start_thread()
    with pytest.raises(TransportClosedError):
        await asyncio.wait_for(turn(be, thread, "CRASH"), timeout=10)
    assert (await be.status())["codex"] == "error"  # /health turns degraded
    await be.stop()  # must not raise although the process is gone


# --- the whole app, offline -----------------------------------------------------------

@pytest.fixture
def app_client(monkeypatch):
    from conduix.app import app
    from conduix.responses_store import ResponseStore
    from conduix.routes import responses as responses_mod
    from conduix.sessions import SessionManager

    fresh = Backend(launch_args=FAKE)
    for mod in ("conduix.app", "conduix.sessions", "conduix.routes.responses",
                "conduix.routes.chat", "conduix.routes.models"):
        monkeypatch.setattr(f"{mod}.backend", fresh)
    mgr = SessionManager()
    for mod in ("conduix.app", "conduix.routes.responses", "conduix.routes.sessions"):
        monkeypatch.setattr(f"{mod}.manager", mgr)
    monkeypatch.setattr(responses_mod, "store", ResponseStore())
    with TestClient(app) as http:  # runs the lifespan: real Backend.start() on the fake
        yield OpenAI(base_url="http://testserver/v1", api_key="x", http_client=http,
                     max_retries=0), http


def test_full_stack_offline(app_client):
    client, http = app_client
    assert http.get("/health").json()["status"] == "ok"
    assert client.models.list().data[0].id == "gpt-6-astra"

    r1 = client.responses.create(model="gpt-6-astra", input="A")
    assert r1.output_text == "ctx=user:A"
    r2 = client.responses.create(model="gpt-6-astra", input="B", previous_response_id=r1.id)
    assert r2.output_text == "ctx=user:A;assistant:ctx=user:A;user:B"

    with client.responses.stream(model="gpt-6-astra", input="hi") as s:
        assert s.get_final_response().output_text == "ctx=user:hi"

    chunks = list(client.chat.completions.create(
        model="gpt-6-astra", stream=True, messages=[{"role": "system", "content": "S"},
                                                    {"role": "user", "content": "hi"}]))
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "ctx=developer:S;user:hi"

    with pytest.raises(openai.RateLimitError) as ei:
        client.responses.create(model="gpt-6-astra", input="QUOTA")
    assert "(resets at 2026-10-04 16:31 UTC)" in ei.value.message
