"""Offline tests for sessions.py and the /v1/sessions routes (fake backend)."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from conduix import sessions as sessions_mod
from conduix.app import app
from conduix.backend import Backend, UnknownModelError, UnsupportedEffortError
from conduix.errors import APIError
from conduix.routes import sessions as routes_mod
from conduix.sessions import SessionManager


class FakeBackend:
    def __init__(self):
        self.started = []
        self.closed = []

    def resolve_model(self, model, effort=None):
        if model == "nope":
            raise UnknownModelError("unknown model")
        if effort == "bogus":
            raise UnsupportedEffortError("unsupported effort")
        return model

    async def start_thread(self, *, model=None, developer_instructions=None):
        t = SimpleNamespace(id=f"thr_{len(self.started)}")
        self.started.append((t.id, model, developer_instructions))
        return t

    async def close_thread(self, thread_id):
        self.closed.append(thread_id)


@pytest.fixture
def fake(monkeypatch):
    fb = FakeBackend()
    monkeypatch.setattr(sessions_mod, "backend", fb)
    return fb


def limits(monkeypatch, *, max_sessions=100, idle=1800):
    monkeypatch.setattr(sessions_mod, "settings", lambda: SimpleNamespace(
        max_sessions=max_sessions, session_idle_timeout_s=idle))


async def settle(mgr):
    await asyncio.gather(*list(mgr._closing))


async def test_create_get_list_delete(fake):
    mgr = SessionManager()
    s = await mgr.create(model="gpt-6-astra", effort="low", instructions="be brief")
    assert s.id.startswith("sess_")
    assert fake.started == [("thr_0", "gpt-6-astra", "be brief")]
    assert mgr.get(s.id) is s and mgr.list() == [s]

    assert await mgr.delete(s.id) is True
    assert mgr.get(s.id) is None
    await settle(mgr)
    assert fake.closed == ["thr_0"] and s.closed
    assert await mgr.delete(s.id) is False


async def test_use_unknown_session_is_404(fake):
    mgr = SessionManager()
    with pytest.raises(APIError) as ei:
        async with mgr.use("sess_missing"):
            pass
    assert ei.value.status == 404 and ei.value.param == "session_id"


async def test_use_serializes_turns_and_counts(fake):
    mgr = SessionManager()
    s = await mgr.create()
    order = []

    async def turn(name, hold):
        async with mgr.use(s.id):
            order.append(f"{name}+")
            await asyncio.sleep(hold)
            order.append(f"{name}-")

    await asyncio.gather(turn("a", 0.05), turn("b", 0))
    assert order == ["a+", "a-", "b+", "b-"]
    assert s.turn_count == 2


async def test_delete_while_busy_waits_for_turn_and_rejects_waiters(fake):
    mgr = SessionManager()
    s = await mgr.create()
    entered, release = asyncio.Event(), asyncio.Event()

    async def running_turn():
        async with mgr.use(s.id):
            entered.set()
            await release.wait()

    async def queued_turn():
        async with mgr.use(s.id):
            pass

    t1 = asyncio.create_task(running_turn())
    await entered.wait()
    t2 = asyncio.create_task(queued_turn())  # got the session, now waiting on the lock
    await asyncio.sleep(0)

    assert await mgr.delete(s.id)
    await asyncio.sleep(0.01)
    assert fake.closed == []  # not cut off mid-turn

    release.set()
    await t1
    with pytest.raises(APIError) as ei:
        await t2
    assert ei.value.status == 404
    await settle(mgr)
    assert fake.closed == ["thr_0"]


async def test_cap_evicts_least_recently_used_idle(fake, monkeypatch):
    limits(monkeypatch, max_sessions=2)
    mgr = SessionManager()
    a = await mgr.create()
    b = await mgr.create()
    # Explicit times: time.time() ticks ~15 ms on Windows, so real ones tie.
    a.last_used_at, b.last_used_at = 200, 100
    c = await mgr.create()
    assert {x.id for x in mgr.list()} == {a.id, c.id}
    await settle(mgr)
    assert fake.closed == [b.thread.id]


async def test_cap_with_all_busy_is_503(fake, monkeypatch):
    limits(monkeypatch, max_sessions=1)
    mgr = SessionManager()
    a = await mgr.create()
    async with mgr.use(a.id):
        with pytest.raises(APIError) as ei:
            await mgr.create()
    assert ei.value.status == 503
    assert fake.closed == ["thr_1"]  # the new thread isn't leaked
    assert [x.id for x in mgr.list()] == [a.id]


async def test_sweep_expires_idle_but_not_busy(fake, monkeypatch):
    limits(monkeypatch, idle=10)
    mgr = SessionManager()
    idle, busy, fresh = await mgr.create(), await mgr.create(), await mgr.create()
    idle.last_used_at = busy.last_used_at = 0
    async with busy.lock:
        assert mgr.sweep(now=100) == [idle.id]
    assert {x.id for x in mgr.list()} == {busy.id, fresh.id}


async def test_stop_closes_everything(fake):
    mgr = SessionManager()
    await mgr.start()
    await mgr.create()
    await mgr.create()
    await mgr.stop()
    assert mgr.list() == [] and sorted(fake.closed) == ["thr_0", "thr_1"]


async def test_unknown_model_or_effort_starts_no_thread(fake):
    mgr = SessionManager()
    with pytest.raises(UnknownModelError):
        await mgr.create(model="nope")
    with pytest.raises(UnsupportedEffortError):
        await mgr.create(effort="bogus")
    assert fake.started == []


# --- backend.resolve_model ---------------------------------------------------

def _backend_with_models():
    b = Backend()
    b._models = [
        {"id": "gpt-6-astra", "is_default": True, "efforts": ["low", "medium", "ultra"]},
        {"id": "gpt-5.5", "is_default": False, "efforts": ["low", "xhigh"]},
    ]
    return b


def test_resolve_model():
    b = _backend_with_models()
    assert b.resolve_model("gpt-5.5", "xhigh") == "gpt-5.5"
    assert b.resolve_model(None, "ultra") is None  # checked against the default model
    with pytest.raises(UnsupportedEffortError):
        b.resolve_model("gpt-5.5", "ultra")
    with pytest.raises(UnsupportedEffortError):
        b.resolve_model(None, "max")
    with pytest.raises(UnknownModelError):
        b.resolve_model("gpt-4o")


# --- routes ------------------------------------------------------------------

@pytest.fixture
def client(fake, monkeypatch):
    monkeypatch.setattr(routes_mod, "manager", SessionManager())
    return TestClient(app)  # no `with`: lifespan (real Codex) doesn't run


def test_routes_roundtrip(client):
    r = client.post("/v1/sessions", json={})
    assert r.status_code == 200
    sid = r.json()["session_id"]

    listed = client.get("/v1/sessions").json()
    assert listed["object"] == "list"
    assert [x["session_id"] for x in listed["data"]] == [sid]

    r = client.delete(f"/v1/sessions/{sid}")
    assert r.json() == {"id": sid, "object": "session.deleted", "deleted": True}


def test_create_without_body(client):
    assert client.post("/v1/sessions").status_code == 200


def test_delete_unknown_is_openai_404(client):
    r = client.delete("/v1/sessions/sess_nope")
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert r.json()["error"]["param"] == "session_id"


def test_bad_effort_is_openai_400(client):
    r = client.post("/v1/sessions", json={"effort": "bogus"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "reasoning.effort"


def test_bad_body_is_openai_400(client):
    r = client.post("/v1/sessions", json={"model": 123})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "model"
