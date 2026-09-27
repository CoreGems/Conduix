"""Conduix FastAPI app — OpenAI-compatible local API."""
from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI

from conduix import __version__

app = FastAPI(
    title="Conduix",
    version=__version__,
    description="OpenAI-compatible local API, powered by the openai-codex SDK.",
)


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str | None]:
    # Codex login / plan status is added in step 3, once backend.py owns AsyncCodex.
    return {
        "status": "ok",
        "version": __version__,
        "openai_codex": _pkg_version("openai-codex"),
    }


def main() -> None:
    import uvicorn

    uvicorn.run(
        "conduix.app:app",
        host=os.environ.get("CONDUIX_HOST", "127.0.0.1"),
        port=int(os.environ.get("CONDUIX_PORT", "8766")),
    )
