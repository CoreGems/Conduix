"""Session lifecycle endpoints — Conduix's extension over the OpenAI API."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from conduix.errors import not_found
from conduix.sessions import Session, manager

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    # All optional defaults for the session; each request can still override
    # model and effort. `instructions` is fixed once the thread starts.
    model: str | None = None
    effort: str | None = None
    instructions: str | None = None


class SessionInfo(BaseModel):
    session_id: str
    object: str = "session"
    model: str | None
    effort: str | None
    created_at: int
    last_used_at: int
    turn_count: int
    busy: bool

    @classmethod
    def of(cls, s: Session) -> "SessionInfo":
        return cls(
            session_id=s.id, model=s.model, effort=s.effort,
            created_at=int(s.created_at), last_used_at=int(s.last_used_at),
            turn_count=s.turn_count, busy=s.busy,
        )


class SessionList(BaseModel):
    object: str = "list"
    data: list[SessionInfo]


@router.get("", response_model=SessionList)
async def list_sessions() -> SessionList:
    return SessionList(data=[SessionInfo.of(s) for s in manager.list()])


@router.post("", response_model=SessionInfo)
async def create_session(req: CreateSessionRequest | None = None) -> SessionInfo:
    req = req or CreateSessionRequest()
    s = await manager.create(model=req.model, effort=req.effort, instructions=req.instructions)
    return SessionInfo.of(s)


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict:
    if not await manager.delete(session_id):
        raise not_found(f"session {session_id!r} not found", param="session_id")
    return {"id": session_id, "object": "session.deleted", "deleted": True}
