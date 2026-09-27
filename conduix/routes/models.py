"""GET /v1/models — the models the logged-in ChatGPT plan can use.

OpenAI's Model shape plus Conduix extras a client needs to pick `effort`.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from conduix.backend import backend
from conduix.errors import not_found

router = APIRouter(prefix="/v1/models", tags=["models"])


def _model(m: dict[str, Any], created: int) -> dict[str, Any]:
    return {
        "id": m["id"],
        "object": "model",
        "created": created,
        "owned_by": "openai",
        "is_default": m["is_default"],
        "default_effort": m["default_effort"],
        "efforts": m["efforts"],
        "input_modalities": m["input_modalities"],
    }


@router.get("")
async def list_models() -> dict[str, Any]:
    created = backend.started_at
    return {"object": "list", "data": [_model(m, created) for m in backend.cached_models]}


@router.get("/{model_id}")
async def retrieve_model(model_id: str) -> dict[str, Any]:
    for m in backend.cached_models:
        if m["id"] == model_id:
            return _model(m, backend.started_at)
    raise not_found(f"model {model_id!r} is not available on this plan", param="model")
