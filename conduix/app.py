"""Conduix FastAPI app — OpenAI-compatible local API."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from conduix import __version__
from conduix import errors
from conduix.backend import backend
from conduix.config import settings
from conduix.routes.models import router as models_router
from conduix.routes.responses import router as responses_router
from conduix.routes.sessions import router as sessions_router
from conduix.sessions import manager

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Fails startup (BillingGuardError) if Codex isn't logged in with ChatGPT.
    await backend.start()
    await manager.start()
    try:
        yield
    finally:
        await manager.stop()
        await backend.stop()


app = FastAPI(
    title="Conduix",
    version=__version__,
    description="OpenAI-compatible local API, powered by the openai-codex SDK.",
    lifespan=lifespan,
)
errors.install(app)
app.include_router(responses_router)
app.include_router(models_router)
app.include_router(sessions_router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, Any]:
    status = await backend.status()
    return {
        "status": "ok" if status.get("codex") == "ok" else "degraded",
        "version": __version__,
        **status,
    }


def main() -> None:
    import uvicorn

    s = settings()
    uvicorn.run("conduix.app:app", host=s.host, port=s.port)
