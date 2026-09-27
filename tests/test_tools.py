"""Offline tests for function tools and web search (step 12 / v1.1).

Route level: the fake backend from test_responses_route (its answers dump the
thread's model-visible history; "CALL" makes it call the thread's first tool).
Backend level: the real SDK + backend.py against tests/fake_app_server.py.
"""
import json
import sys
from pathlib import Path

import openai
import pytest
from openai.types.responses import Response, ResponseStreamEvent
from pydantic import TypeAdapter

from conduix.backend import (
    Backend,
    MessageDone,
    ToolCall,
    TurnDone,
    Usage,
    WebSearchCall,
    base_instructions,
)
from tests.test_backend_fake_server import FAKE, stats
from tests.test_responses_route import FakeBackend, client, fb, http, sse_events  # noqa: F401

EVENTS = TypeAdapter(ResponseStreamEvent)
M = "gpt-6-astra"
WEATHER = {"type": "function", "name": "get_weather", "description": "Weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                          "required": ["city"]}}
CHAT_WEATHER = {"type": "function", "function": {k: v for k, v in WEATHER.items() if k != "type"}}


# --- Responses API: function calls ------------------------------------------------

def test_function_call_ends_the_response(client, fb):
    r = client.responses.create(model=M, input="CALL", tools=[WEATHER])
    Response.model_validate(r.model_dump())
    assert r.status == "completed" and r.output_text == ""
    [call] = r.output
    assert call.type == "function_call" and call.name == "get_weather"
    assert json.loads(call.arguments) == {"city": "Oslo"} and call.call_id.startswith("call_")
    thread = fb.threads[-1]
    assert thread.interrupted  # the Codex turn was stopped, never given a result
    assert thread.tools == [{"type": "function", "name": "get_weather",
                             "description": "Weather for a city.",
                             "inputSchema": WEATHER["parameters"]}]
    assert fb.manager.list() == []  # the interrupted thread is discarded


def test_tool_output_resumes_on_a_fresh_thread(client, fb):
    r1 = client.responses.create(model=M, input="CALL", tools=[WEATHER])
    call = r1.output[0]
    r2 = client.responses.create(model=M, tools=[WEATHER], previous_response_id=r1.id, input=[
        {"type": "function_call_output", "call_id": call.call_id, "output": "17C cloudy"}])
    assert len(fb.threads) == 2
    assert r2.output_text == (f'ctx=user:CALL;call:get_weather{{"city": "Oslo"}}#{call.call_id};'
                              f"tool:17C cloudy#{call.call_id}")  # empty turn input after the tool


def test_full_history_with_tool_items_works_without_previous_id(client):
    r = client.responses.create(model=M, tools=[WEATHER], input=[
        {"role": "user", "content": "Weather in Oslo?"},
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city":"Oslo"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "-3C snow"},
        {"role": "user", "content": "And tomorrow?"},
    ])
    assert r.output_text == ('ctx=user:Weather in Oslo?;call:get_weather{"city":"Oslo"}#c1;'
                             "tool:-3C snow#c1;user:And tomorrow?")


def test_output_for_unknown_call_is_400(client):
    with pytest.raises(openai.BadRequestError) as ei:
        client.responses.create(model=M, tools=[WEATHER], input=[
            {"role": "user", "content": "hi"},
            {"type": "function_call_output", "call_id": "nope", "output": "x"}])
    assert "call_id 'nope'" in ei.value.message


def test_function_call_stream_events(http):
    r = http.post("/v1/responses", json={"model": M, "input": "CALL", "tools": [WEATHER],
                                         "stream": True})
    events = sse_events(r.text)
    for e in events:
        EVENTS.validate_python(e)
    assert [e["type"] for e in events] == [
        "response.created", "response.in_progress", "response.output_item.added",
        "response.function_call_arguments.delta", "response.function_call_arguments.done",
        "response.output_item.done", "response.completed",
    ]


def test_openai_stream_helper_with_function_call(client):
    with client.responses.stream(model=M, input="CALL", tools=[WEATHER]) as s:
        final = s.get_final_response()
    assert final.output[0].type == "function_call" and final.output[0].name == "get_weather"


def test_tool_choice_none_starts_no_tools(client, fb):
    r = client.responses.create(model=M, input="CALL", tools=[WEATHER], tool_choice="none")
    assert r.output_text == "ctx=user:CALL" and fb.threads[-1].tools == []


def test_same_tools_continue_the_thread_changed_tools_rebuild(client, fb):
    r1 = client.responses.create(model=M, input="A", tools=[WEATHER])
    r2 = client.responses.create(model=M, input="B", tools=[WEATHER], previous_response_id=r1.id)
    assert len(fb.threads) == 1
    other = {**WEATHER, "name": "get_time"}
    client.responses.create(model=M, input="C", tools=[other], previous_response_id=r2.id)
    assert len(fb.threads) == 2


def test_tools_with_session_id_is_400(client, http):
    sid = http.post("/v1/sessions", json={}).json()["session_id"]
    with pytest.raises(openai.BadRequestError) as ei:
        client.responses.create(model=M, input="A", tools=[WEATHER], extra_body={"session_id": sid})
    assert ei.value.body["param"] == "tools"


# --- Responses API: web search ------------------------------------------------------

def test_web_search(client, fb, http):
    r = client.responses.create(model=M, input="SEARCH", tools=[{"type": "web_search"}])
    ws = r.output[0]
    assert ws.type == "web_search_call" and ws.status == "completed"
    assert ws.action.type == "open_page" and ws.action.url == "https://x.test"
    assert fb.threads[-1].web_search and fb.turns[-1]["allow_web_search"] is True
    events = sse_events(http.post("/v1/responses", json={
        "model": M, "input": "SEARCH", "tools": [{"type": "web_search_preview"}], "stream": True}).text)
    for e in events:
        EVENTS.validate_python(e)
    assert "response.web_search_call.completed" in [e["type"] for e in events]


def test_no_web_search_unless_asked(client, fb):
    client.responses.create(model=M, input="SEARCH")
    assert not fb.threads[-1].web_search and fb.turns[-1]["allow_web_search"] is False


# --- Chat Completions ----------------------------------------------------------------

def test_chat_tool_call_and_resume(client, fb):
    c1 = client.chat.completions.create(model=M, tools=[CHAT_WEATHER],
                                        messages=[{"role": "user", "content": "CALL"}])
    choice = c1.choices[0]
    assert choice.finish_reason == "tool_calls" and choice.message.content is None
    [tc] = choice.message.tool_calls
    assert tc.function.name == "get_weather" and json.loads(tc.function.arguments) == {"city": "Oslo"}

    c2 = client.chat.completions.create(model=M, tools=[CHAT_WEATHER], messages=[
        {"role": "user", "content": "CALL"},
        choice.message.model_dump(exclude_none=True),
        {"role": "tool", "tool_call_id": tc.id, "content": "17C"},
    ])
    assert c2.choices[0].message.content == (
        f'ctx=user:CALL;call:get_weather{{"city": "Oslo"}}#{tc.id};tool:17C#{tc.id}')
    assert fb.manager.list() == []


def test_chat_tool_call_stream(client):
    chunks = list(client.chat.completions.create(
        model=M, tools=[CHAT_WEATHER], stream=True, messages=[{"role": "user", "content": "CALL"}]))
    calls = [tc for c in chunks for tc in (c.choices[0].delta.tool_calls or []) if c.choices]
    assert calls[0].index == 0 and calls[0].id and calls[0].function.name == "get_weather"
    assert "".join(tc.function.arguments or "" for tc in calls) == '{"city": "Oslo"}'
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_chat_web_search_options(client, fb):
    client.chat.completions.create(model=M, web_search_options={},
                                   messages=[{"role": "user", "content": "SEARCH"}])
    assert fb.threads[-1].web_search


# --- backend + real SDK against the fake app-server ------------------------------------

@pytest.fixture
async def be():
    b = Backend(launch_args=FAKE)
    await b.start()
    try:
        yield b
    finally:
        await b.stop()


SPEC = {"type": "function", "name": "get_weather", "description": "d",
        "inputSchema": {"type": "object", "properties": {}}}


async def test_thread_with_tools_keeps_the_lockdown(be):
    await be.start_thread(tools=[SPEC], web_search=True)
    [p] = (await stats(be))["thread_start_params"]
    assert p["dynamicTools"] == [SPEC]
    assert p["approvalPolicy"] == "never" and p["sandbox"] == "read-only"
    assert p["config"]["web_search"] == "live" and p["config"]["features.shell_tool"] is False
    assert p["baseInstructions"] == base_instructions(tools=True, web_search=True)
    assert "Call the provided functions" in p["baseInstructions"]


async def test_tool_call_request_becomes_an_event_and_is_never_answered(be):
    thread = await be.start_thread(tools=[SPEC])
    events = []
    async for ev in be.run_turn(thread, ["CALLTOOL"]):
        events.append(ev)
        if isinstance(ev, ToolCall):
            break
    call = events[-1]
    assert (call.call_id, call.name, json.loads(call.arguments)) == ("exec-1", "get_weather",
                                                                     {"city": "Kyiv"})
    import asyncio
    for _ in range(100):
        s = await stats(be)
        if s["interrupted"]:
            break
        await asyncio.sleep(0.05)
    assert len(s["interrupted"]) == 1
    assert not any(r["id"].startswith("srv-") and r["id"] != "srv-approve"
                   for r in s["client_replies"])  # no result was sent for the call


async def test_approval_requests_are_declined(be):
    thread = await be.start_thread()
    events = [ev async for ev in be.run_turn(thread, ["APPROVE"])]
    assert events[-1].status == "completed"
    replies = (await stats(be))["client_replies"]
    assert {"id": "srv-approve", "result": {"decision": "decline"}} in replies
    assert be.blocked_items == {"approval": 1}


async def test_unknown_server_requests_get_an_error(be):
    thread = await be.start_thread()
    await_all = [ev async for ev in be.run_turn(thread, ["WEIRD"])]
    assert await_all[-1].status == "completed"
    [reply] = [r for r in (await stats(be))["client_replies"] if r["id"] == "srv-weird"]
    assert reply["error"]["code"] == -32601


async def test_usage_is_per_turn_across_model_calls(be):
    thread = await be.start_thread()
    await_first = [ev async for ev in be.run_turn(thread, ["hi"])]
    events = [ev async for ev in be.run_turn(thread, ["TWICE"])]
    usages = [e for e in events if isinstance(e, Usage)]
    # two model calls in this turn, and none of the first turn's usage
    assert usages[-1] == Usage(200, 160, 10, 0, 0)
    assert [e for e in await_first if isinstance(e, Usage)][-1] == Usage(100, 80, 5, 0, 0)


async def test_web_search_items_only_when_enabled(be):
    on = await be.start_thread(web_search=True)
    events = [ev async for ev in be.run_turn(on, ["SEARCH"], allow_web_search=True)]
    searches = [e for e in events if isinstance(e, WebSearchCall)]
    assert [s.status for s in searches] == ["in_progress", "completed"]
    assert searches[-1].action == {"type": "openPage", "url": "https://www.python.org/"}
    assert isinstance(events[-1], TurnDone) and any(isinstance(e, MessageDone) for e in events)
    assert be.blocked_items == {}
