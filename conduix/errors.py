"""OpenAI-shaped errors: {"error": {"message", "type", "param", "code"}}.

Step 8 adds the quota / auth / upstream mapping on top of this.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from conduix.backend import UnknownModelError, UnsupportedEffortError


class APIError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status, self.message, self.type, self.param, self.code = (
            status, message, type, param, code
        )

    def body(self) -> dict:
        return {"error": {
            "message": self.message, "type": self.type, "param": self.param, "code": self.code,
        }}


def not_found(message: str, *, param: str | None = None) -> APIError:
    return APIError(404, message, param=param, code="not_found")


def install(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_req: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(UnknownModelError)
    async def _unknown_model(_req: Request, exc: UnknownModelError) -> JSONResponse:
        err = APIError(400, str(exc), param="model", code="model_not_found")
        return JSONResponse(err.body(), status_code=400)

    @app.exception_handler(UnsupportedEffortError)
    async def _bad_effort(_req: Request, exc: UnsupportedEffortError) -> JSONResponse:
        err = APIError(400, str(exc), param="reasoning.effort", code="unsupported_value")
        return JSONResponse(err.body(), status_code=400)

    @app.exception_handler(RequestValidationError)
    async def _validation(_req: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = [str(x) for x in first.get("loc", ()) if x != "body"]
        err = APIError(400, first.get("msg", "invalid request"), param=".".join(loc) or None)
        return JSONResponse(err.body(), status_code=400)
