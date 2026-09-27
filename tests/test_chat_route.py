"""Offline tests for POST /v1/chat/completions, driven by the real `openai` client.

Reuses the fake backend from test_responses_route: its answers dump the
thread's model-visible history ("ctx=user:A;assistant:B;...").
"""
import openai
import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from tests.test_responses_route import FakeBackend, client, fb, http, sse_events  # noqa: F401

M = "gpt-6-astra"


def user(text):
    return [{"role": "user", "content": text}]


def raw_stream(http, **body):
    r = http.post("/v1/chat/completions", json={"model": M, "stream": True, **body})
    lines = [l for l in r.text.splitlines() if l]
    return r, lines


def test_create(client):
    c = client.chat.completions.create(model=M, messages=user("hi"))
    ChatCompletion.model_validate(c.model_dump())
    assert c.id.startswith("chatcmpl-") and c.object == "chat.completion"
    assert c.choices[0].message.content == "ctx=user:hi"
    assert c.choices[0].finish_reason == "stop"
    assert c.usage.prompt_tokens == 100 and c.usage.prompt_tokens_details.cached_tokens == 80
    assert c.model == M


def test_system_and_history_become_real_messages(client):
    c = client.chat.completions.create(model=M, messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": [{"type": "text", "text": "C"}]},
    ])
    assert c.choices[0].message.content == "ctx=developer:sys;user:A;assistant:B;user:C"


def test_stream(client):
    chunks = list(client.chat.completions.create(model=M, messages=user("hi"), stream=True))
    for ch in chunks:
        ChatCompletionChunk.model_validate(ch.model_dump())
    assert chunks[0].choices[0].delta.role == "assistant"
    assert "".join(ch.choices[0].delta.content or "" for ch in chunks) == "ctx=user:hi"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert all(ch.usage is None for ch in chunks)
    assert len({ch.id for ch in chunks}) == 1


def test_stream_include_usage(client):
    chunks = list(client.chat.completions.create(
        model=M, messages=user("hi"), stream=True, stream_options={"include_usage": True}))
    assert chunks[-1].choices == [] and chunks[-1].usage.total_tokens == 105
    assert chunks[-2].choices[0].finish_reason == "stop"


def test_raw_stream_framing(http):
    r, lines = raw_stream(http, messages=user("hi"))
    assert r.headers["content-type"].startswith("text/event-stream")
    assert all(l.startswith("data: ") for l in lines)  # no `event:` lines in chat SSE
    assert lines[-1] == "data: [DONE]"


def test_effort_and_json_schema_reach_codex(client, fb):
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    client.chat.completions.create(
        model=M, messages=user("hi"), reasoning_effort="high",
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": schema}})
    assert fb.turns[-1]["effort"] == "high" and fb.turns[-1]["output_schema"] == schema


def test_nothing_is_kept_between_stateless_calls(client, fb):
    client.chat.completions.create(model=M, messages=user("hi"))
    assert fb.manager.list() == []


def test_session_id(client, http, fb):
    sid = http.post("/v1/sessions", json={}).json()["session_id"]
    client.chat.completions.create(model=M, messages=user("A"), extra_body={"session_id": sid})
    c = client.chat.completions.create(model=M, messages=user("B"), extra_body={"session_id": sid})
    assert len(fb.threads) == 1
    assert c.choices[0].message.content.startswith("ctx=user:A;assistant:")
    assert c.model_extra["session_id"] == sid


def test_ignored_params_are_accepted(client):
    c = client.chat.completions.create(model=M, messages=user("hi"), temperature=0.2,
                                       max_tokens=5, top_p=0.9, user="u", seed=1)
    assert c.choices[0].message.content == "ctx=user:hi"


@pytest.mark.parametrize("kwargs, param", [
    ({"messages": user("A"), "tools": [{"type": "custom", "custom": {"name": "f"}}]}, "tools[0].type"),
    ({"messages": [{"role": "user", "content": "A"},
                   {"role": "tool", "content": "x", "tool_call_id": "1"}]}, "messages"),  # unknown call
    ({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]}]}, "messages"),
    ({"messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]},
     "messages"),
    ({"messages": user("A"), "n": 2}, "n"),
    ({"messages": user("A"), "response_format": {"type": "xml"}}, "response_format.type"),
    ({"messages": user("A"), "response_format": {"type": "json_schema", "json_schema": {
        "name": "e", "schema": {"type": "object", "properties": {}, "additionalProperties": False}}}},
     "response_format.json_schema.schema"),
    ({"messages": user("A"), "reasoning_effort": "bogus"}, "reasoning_effort"),
])
def test_bad_requests_are_400(client, kwargs, param):
    with pytest.raises(openai.BadRequestError) as ei:
        client.chat.completions.create(model=M, **kwargs)
    assert ei.value.body["param"] == param


def test_quota_is_429_and_not_retried(http, fb):
    retrying = OpenAI(base_url="http://testserver/v1", api_key="x", http_client=http)
    with pytest.raises(openai.RateLimitError) as ei:
        retrying.chat.completions.create(model=M, messages=user("QUOTA"))
    assert len(fb.turns) == 1 and ei.value.code == "usage_limit_exceeded"


def test_quota_while_streaming_raises_in_the_sdk(client):
    stream = client.chat.completions.create(model=M, messages=user("QUOTA"), stream=True)
    with pytest.raises(openai.APIError) as ei:
        list(stream)
    assert ei.value.code == "usage_limit_exceeded" and "resets at" in ei.value.message


def test_dead_app_server_while_streaming(http):
    _, lines = raw_stream(http, messages=user("DEAD"))
    assert '"codex_unavailable"' in lines[-2] and lines[-1] == "data: [DONE]"
