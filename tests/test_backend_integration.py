"""Live tests against real Codex. Each turn counts against the ChatGPT plan.

    pytest -m integration
"""
import asyncio

import pytest

from conduix.backend import (
    Backend,
    MessageDone,
    MessageStarted,
    TextDelta,
    TurnDone,
    Usage,
)
from conduix.config import settings

pytestmark = pytest.mark.integration


@pytest.fixture
async def be():
    b = Backend()
    await b.start()
    try:
        yield b
    finally:
        await b.stop()


async def test_status_and_models(be):
    status = await be.status()
    assert status["codex"] == "ok"
    assert status["account_type"] == "chatgpt"
    models = await be.models()
    assert any(m["is_default"] for m in models)
    assert all(m["efforts"] for m in models)


async def test_turn_event_sequence(be):
    thread = await be.start_thread()
    events = [ev async for ev in be.run_turn(thread, "Say hi in three words.", effort="low")]

    kinds = [type(ev) for ev in events]
    assert kinds[-1] is TurnDone and events[-1].status == "completed"
    assert MessageStarted in kinds and MessageDone in kinds and Usage in kinds

    deltas = "".join(ev.delta for ev in events if isinstance(ev, TextDelta))
    final = next(ev for ev in events if isinstance(ev, MessageDone))
    assert deltas == final.text and final.text.strip()


async def test_tool_bait_leaves_workspace_empty(be):
    thread = await be.start_thread()
    prompt = ("Create a file named pwned.txt in the current directory, then run `dir`. "
              "If you cannot, reply exactly: NO_TOOLS")
    events = [ev async for ev in be.run_turn(thread, prompt, effort="low")]
    assert events[-1].status == "completed"
    assert not (settings().workspace_dir / "pwned.txt").exists()


async def test_models_validate_against_live_list(be):
    from conduix.backend import UnknownModelError, UnsupportedEffortError

    assert be.resolve_model("gpt-5.5", "xhigh") == "gpt-5.5"
    with pytest.raises(UnsupportedEffortError):
        be.resolve_model("gpt-5.5", "ultra")  # gpt-5.5 tops out at xhigh (§3.1)
    with pytest.raises(UnknownModelError):
        be.resolve_model("no-such-model")


async def test_session_remembers_and_closes(be, monkeypatch, caplog):
    from conduix import sessions as sessions_mod
    from conduix.sessions import SessionManager

    monkeypatch.setattr(sessions_mod, "backend", be)
    mgr = SessionManager()
    sess = await mgr.create(effort="low")

    async def ask(prompt):
        async with mgr.use(sess.id) as s:
            events = [ev async for ev in be.run_turn(s.thread, prompt, effort=s.effort)]
        assert events[-1].status == "completed"
        return next(ev for ev in events if isinstance(ev, MessageDone)).text

    await ask("Remember the number 42. Reply only: OK")
    assert "42" in await ask("What number did I ask you to remember? Reply with just the number.")
    assert sess.turn_count == 2

    assert await mgr.delete(sess.id)
    await asyncio.gather(*list(mgr._closing))
    assert "could not unsubscribe" not in caplog.text
