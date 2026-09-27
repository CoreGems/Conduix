"""Offline tests for POST /v1/responses, driven by the real `openai` client.

The fake backend keeps each thread's model-visible history and answers with
a dump of it ("ctx=user:A;assistant:B;..."), so tests can assert exactly what
the model would have seen.
"""
import json
from types import SimpleNamespace

import openai
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from conduix import sessions as sessions_mod
from conduix.app import app
from conduix.backend import (
    MessageDone,
    MessageStarted,
    TextDelta,
    TurnDone,
    TurnError,
    UnknownModelError,
    Usage,
)
from conduix.responses_store import ResponseStore
from conduix.routes import responses as responses_mod
from conduix.routes import sessions as routes_sessions_mod
from conduix.sessions import SessionManager


class FakeThread:
    def __init__(self, tid, instructions):
        self.id = tid
        self.instructions = instructions
        self.items = []  # (role, text)


class FakeBackend:
    def __init__(self):
        self.threads = []
        self.closed = []
        self.turns = []  # kwargs of each run_turn

    def resolve_model(self, model, effort=None):
        if model == "nope":
            raise UnknownModelError("unknown model")
        return model

    def model_name(self, model):
        return model or "gpt-6-astra"

    async def start_thread(self, *, model=None, developer_instructions=None):
        t = FakeThread(f"thr_{len(self.threads)}", developer_instructions)
        self.threads.append(t)
        return t

    async def close_thread(self, thread_id):
        self.closed.append(thread_id)

    async def inject_items(self, thread, items):
        for it in items:
            thread.items += [(it["role"], c["text"]) for c in it["content"]]

    async def run_turn(self, thread, input, **kw):
        self.turns.append(kw)
        texts = [i.text for i in input]
        thread.items += [("user", t) for t in texts]
        if "BOOM" in texts:
            raise RuntimeError("transport closed")
        if "FAIL" in texts:
            yield TurnError("usage limit reached")
            yield TurnDone("failed")
            return
        reply = "ctx=" + ";".join(f"{r}:{t}" for r, t in thread.items)
        thread.items.append(("assistant", reply))
        yield MessageStarted("msg_1", "final_answer")
        yield TextDelta("msg_1", reply[:5])
        yield TextDelta("msg_1", reply[5:])
        yield MessageDone("msg_1", reply)
        yield Usage(100, 80, 5, 0)
        yield TurnDone("completed")


@pytest.fixture
def fb(monkeypatch):
    fb = FakeBackend()
    mgr = SessionManager()
    monkeypatch.setattr(sessions_mod, "backend", fb)
    monkeypatch.setattr(responses_mod, "backend", fb)
    monkeypatch.setattr(responses_mod, "manager", mgr)
    monkeypatch.setattr(routes_sessions_mod, "manager", mgr)
    monkeypatch.setattr(responses_mod, "store", ResponseStore())
    fb.manager = mgr
    return fb


@pytest.fixture
def http(fb):
    return TestClient(app)  # no `with`: the lifespan (real Codex) doesn't run


@pytest.fixture
def client(http):
    return OpenAI(base_url="http://testserver/v1", api_key="x", http_client=http, max_retries=0)


def sse_events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


# --- basics ------------------------------------------------------------------

def test_create(client):
    r = client.responses.create(model="gpt-6-astra", input="hi")
    assert r.id.startswith("resp_") and r.status == "completed"
    assert r.output_text == "ctx=user:hi"
    assert r.model == "gpt-6-astra"
    assert r.usage.input_tokens_details.cached_tokens == 80


def test_default_model_is_reported(client):
    assert client.responses.create(input="hi").model == "gpt-6-astra"


def test_stream(client):
    with client.responses.stream(model="gpt-6-astra", input="hi") as s:
        deltas = [e.delta for e in s if e.type == "response.output_text.delta"]
        final = s.get_final_response()
    assert "".join(deltas) == "ctx=user:hi" == final.output_text


def test_effort_summary_and_schema_reach_codex(client, fb):
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    r = client.responses.create(
        model="gpt-6-sol", input="hi", reasoning={"effort": "high", "summary": "auto"},
        text={"format": {"type": "json_schema", "name": "x", "schema": schema}},
    )
    assert fb.turns[-1] == {"model": "gpt-6-sol", "effort": "high", "summary": "auto",
                            "output_schema": schema}
    assert r.reasoning.effort == "high"


def test_instructions_become_thread_instructions(client, fb):
    client.responses.create(input="hi", instructions="be terse")
    assert fb.threads[-1].instructions == "be terse"


# --- stateless history ---------------------------------------------------------

def test_history_is_injected_as_real_messages(client):
    r = client.responses.create(input=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": [{"type": "input_text", "text": "C"}]},
    ])
    assert r.output_text == "ctx=developer:sys;user:A;assistant:B;user:C"


def test_output_items_from_a_previous_response_can_be_resent(client):
    r1 = client.responses.create(input="A")
    history = [{"role": "user", "content": "A"}] + [o.model_dump() for o in r1.output]
    history.append({"role": "user", "content": "B"})
    r2 = client.responses.create(input=history)
    assert r2.output_text == f"ctx=user:A;assistant:{r1.output_text};user:B"


# --- previous_response_id --------------------------------------------------------

def test_previous_response_continues_same_thread(client, fb):
    r1 = client.responses.create(input="A")
    r2 = client.responses.create(input="B", previous_response_id=r1.id)
    assert len(fb.threads) == 1
    assert r2.output_text == f"ctx=user:A;assistant:{r1.output_text};user:B"
    assert r2.previous_response_id == r1.id


def test_branching_rebuilds_the_right_history(client, fb):
    r1 = client.responses.create(input="A")
    client.responses.create(input="B", previous_response_id=r1.id)
    r3 = client.responses.create(input="C", previous_response_id=r1.id)  # branch from r1
    assert len(fb.threads) == 2
    assert r3.output_text == f"ctx=user:A;assistant:{r1.output_text};user:C"


def test_continuing_after_thread_was_evicted(client, http, fb):
    r1 = client.responses.create(input="A")
    for s in fb.manager.list():
        assert http.delete(f"/v1/sessions/{s.id}").status_code == 200
    assert fb.manager.list() == []
    r2 = client.responses.create(input="B", previous_response_id=r1.id)
    assert r2.output_text == f"ctx=user:A;assistant:{r1.output_text};user:B"


def test_changed_instructions_rebuild_with_new_instructions(client, fb):
    r1 = client.responses.create(input="A", instructions="one")
    client.responses.create(input="B", previous_response_id=r1.id, instructions="two")
    assert [t.instructions for t in fb.threads] == ["one", "two"]


def test_unknown_previous_response_is_404(client):
    with pytest.raises(openai.NotFoundError) as ei:
        client.responses.create(input="B", previous_response_id="resp_nope")
    assert ei.value.body["param"] == "previous_response_id"


def test_store_false_is_not_kept(client, fb):
    r1 = client.responses.create(input="A", store=False)
    assert fb.manager.list() == []
    with pytest.raises(openai.NotFoundError):
        client.responses.create(input="B", previous_response_id=r1.id)


# --- session_id ----------------------------------------------------------------

def test_session_turns_share_a_thread(client, http, fb):
    sid = http.post("/v1/sessions", json={"instructions": "sys"}).json()["session_id"]
    client.responses.create(input="A", extra_body={"session_id": sid})
    r2 = client.responses.create(input="B", extra_body={"session_id": sid})
    assert len(fb.threads) == 1 and fb.threads[0].instructions == "sys"
    assert r2.output_text.startswith("ctx=user:A;assistant:") and r2.output_text.endswith(";user:B")
    assert r2.model_extra["session_id"] == sid


def test_previous_response_does_not_advance_a_users_session(client, http, fb):
    sid = http.post("/v1/sessions", json={}).json()["session_id"]
    r1 = client.responses.create(input="A", extra_body={"session_id": sid})
    client.responses.create(input="B", previous_response_id=r1.id)
    assert fb.threads[0].items == [("user", "A"), ("assistant", r1.output_text)]


def test_implicit_sessions_are_hidden(client, http):
    client.responses.create(input="A")
    assert http.get("/v1/sessions").json()["data"] == []


def test_unknown_session_is_404(client):
    with pytest.raises(openai.NotFoundError):
        client.responses.create(input="A", extra_body={"session_id": "sess_nope"})


def test_session_and_previous_together_is_400(client):
    with pytest.raises(openai.BadRequestError):
        client.responses.create(input="A", previous_response_id="resp_x",
                                extra_body={"session_id": "sess_x"})


# --- validation ------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, param", [
    ({"input": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]}, "input"),
    ({"input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}]}, "input"),
    ({"input": "A", "tools": [{"type": "function", "name": "f", "parameters": {}}]}, "tools"),
    ({"input": "A", "text": {"format": {"type": "json_object"}}}, "text.format.type"),
    ({"input": "A", "model": "nope"}, "model"),
])
def test_bad_requests_are_400(client, kwargs, param):
    with pytest.raises(openai.BadRequestError) as ei:
        client.responses.create(**kwargs)
    assert ei.value.body["param"] == param


# --- failures ------------------------------------------------------------------

def test_failed_turn_non_streaming_is_500_and_not_stored(client, fb):
    with pytest.raises(openai.InternalServerError) as ei:
        client.responses.create(input="FAIL")
    assert "usage limit" in ei.value.body["message"]


def test_failed_turn_streaming_ends_with_response_failed(http):
    r = http.post("/v1/responses", json={"input": "FAIL", "stream": True})
    events = sse_events(r.text)
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"]["message"] == "usage limit reached"


def test_exception_mid_stream_becomes_error_event(http):
    r = http.post("/v1/responses", json={"input": "BOOM", "stream": True})
    events = sse_events(r.text)
    assert [e["type"] for e in events[:2]] == ["response.created", "response.in_progress"]
    assert events[-1]["type"] == "error" and "transport closed" in events[-1]["message"]
    assert [e["sequence_number"] for e in events] == list(range(len(events)))


def test_failed_turn_breaks_the_session_head(client, http, fb):
    sid = http.post("/v1/sessions", json={}).json()["session_id"]
    r1 = client.responses.create(input="A", extra_body={"session_id": sid})
    with pytest.raises(openai.InternalServerError):
        client.responses.create(input="FAIL", extra_body={"session_id": sid})
    assert fb.manager.get(sid).head_response_id is None
    # r1 is still continuable; it rebuilds, so the failed turn isn't in its history.
    r3 = client.responses.create(input="C", previous_response_id=r1.id)
    assert r3.output_text == f"ctx=user:A;assistant:{r1.output_text};user:C"
