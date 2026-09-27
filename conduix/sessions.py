"""SessionManager: one Codex thread per `session_id`.

Rules (CONDUIX_API_USEAGE_GUIDE.md §6C):
  * One turn at a time per session. `use()` holds `Session.lock` for the
    whole turn, so a second request on the same session waits its turn.
  * Idle sessions expire after `session_idle_timeout_s`; a session that is
    mid-turn is never swept or evicted.
  * At most `max_sessions`; creating one more evicts the least recently used
    idle session.
  * In memory only: a restart loses every session.

Everything runs on one event loop, and the dict is only touched between
awaits, so it needs no lock of its own.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from openai_codex import AsyncThread

from conduix.backend import backend
from conduix.config import settings
from conduix.errors import APIError, not_found

log = logging.getLogger("conduix.sessions")

SWEEP_INTERVAL_S = 60


@dataclass
class Session:
    id: str
    thread: AsyncThread
    model: str | None = None  # session defaults; a request's own values win
    effort: str | None = None
    instructions: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    turn_count: int = 0
    closed: bool = False
    # Created by /v1/responses for a request without session_id; hidden from
    # GET /v1/sessions but swept and capped like any other session.
    implicit: bool = False
    # Last completed response on this thread; None once the thread holds a
    # turn that no stored response describes (failed turn, store=false).
    head_response_id: str | None = None

    @property
    def busy(self) -> bool:
        return self.lock.locked()


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._sweeper: asyncio.Task | None = None
        self._closing: set[asyncio.Task] = set()  # background closes of busy sessions

    async def start(self) -> None:
        if self._sweeper is None:
            self._sweeper = asyncio.create_task(self._sweep_loop())

    async def stop(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            try:
                await self._sweeper
            except asyncio.CancelledError:
                pass
            self._sweeper = None
        sessions, self._sessions = list(self._sessions.values()), {}
        # Shutdown doesn't wait for in-flight turns; Codex closes right after.
        for sess in sessions:
            sess.closed = True
            await backend.close_thread(sess.thread.id)
        for task in list(self._closing):
            task.cancel()

    async def create(
        self,
        *,
        model: str | None = None,
        effort: str | None = None,
        instructions: str | None = None,
        implicit: bool = False,
    ) -> Session:
        model = backend.resolve_model(model, effort)
        thread = await backend.start_thread(model=model, developer_instructions=instructions)

        # No awaits from here to the insert, so the cap check can't race.
        s = settings()
        if len(self._sessions) >= s.max_sessions:
            idle = [x for x in self._sessions.values() if not x.busy]
            if not idle:
                await backend.close_thread(thread.id)
                raise APIError(
                    503, f"session limit ({s.max_sessions}) reached and every session is busy",
                    type="server_error", code="session_limit",
                )
            oldest = min(idle, key=lambda x: x.last_used_at)
            log.info("session cap reached; evicting %s", oldest.id)
            self._discard(oldest)

        sess = Session(
            id=f"sess_{uuid.uuid4().hex}", thread=thread,
            model=model, effort=effort, instructions=instructions, implicit=implicit,
        )
        self._sessions[sess.id] = sess
        return sess

    def get(self, sid: str) -> Session | None:
        return self._sessions.get(sid)

    def list(self, *, include_implicit: bool = True) -> list[Session]:
        return [s for s in self._sessions.values() if include_implicit or not s.implicit]

    async def delete(self, sid: str) -> bool:
        sess = self._sessions.get(sid)
        if sess is None:
            return False
        self._discard(sess)
        return True

    @asynccontextmanager
    async def use(self, sid: str) -> AsyncIterator[Session]:
        """Hold a session for one turn. Waits while another turn runs on it."""
        sess = self._sessions.get(sid)
        if sess is None:
            raise not_found(f"session {sid!r} not found (expired or server restarted)",
                            param="session_id")
        async with sess.lock:
            if sess.closed:  # deleted while this request waited for the lock
                raise not_found(f"session {sid!r} was deleted", param="session_id")
            sess.last_used_at = time.time()
            try:
                yield sess
            finally:
                sess.turn_count += 1
                sess.last_used_at = time.time()

    def _discard(self, sess: Session) -> None:
        """Remove now (new requests get 404); unload the thread once it's idle."""
        self._sessions.pop(sess.id, None)
        # Set before _close queues on the lock: the lock is FIFO, so requests
        # already waiting on it run first and must see the session as gone.
        sess.closed = True
        task = asyncio.create_task(self._close(sess))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    async def _close(self, sess: Session) -> None:
        # Waits for an in-flight turn to finish rather than cutting it off.
        async with sess.lock:
            await backend.close_thread(sess.thread.id)

    def sweep(self, now: float | None = None) -> list[str]:
        cutoff = (now or time.time()) - settings().session_idle_timeout_s
        stale = [x for x in self._sessions.values() if not x.busy and x.last_used_at < cutoff]
        for sess in stale:
            self._discard(sess)
        if stale:
            log.info("expired %d idle session(s)", len(stale))
        return [x.id for x in stale]

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(SWEEP_INTERVAL_S)
            self.sweep()


manager = SessionManager()
