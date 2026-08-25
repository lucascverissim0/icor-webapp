"""FastAPI application factory for the local planner service."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from icor.api.planner import router
from icor.api.schemas import FieldError, ProblemResponse
from icor.application.planner import PlannerRepository, PlannerService
from icor.infrastructure.demo_planner_repository import DemoPlannerRepository

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE = ROOT / "data" / "demo" / "planner-v1.json"
DEFAULT_WEB_ORIGIN = "http://127.0.0.1:5173"


def _problem(request: Request, *, code: str, message: str, status_code: int) -> JSONResponse:
    body = ProblemResponse(
        code=code,
        message=message,
        correlation_id=request.state.correlation_id,
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _local_origin(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("ICOR_WEB_ORIGIN must be a local HTTP origin")
    return value.rstrip("/")


def create_app(repository: PlannerRepository | None = None) -> FastAPI:
    app = FastAPI(
        title="ICOR Planner API",
        version="1.0.0",
        description="Local demonstration contract for windshield demand planning.",
    )
    selected_repository = repository or DemoPlannerRepository.from_path(DEFAULT_FIXTURE)
    app.state.planner_service = PlannerService(selected_repository)

    @app.middleware("http")
    async def correlation_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        request.state.correlation_id = uuid4().hex
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = request.state.correlation_id
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        field_errors = [
            FieldError(
                field=".".join(str(part) for part in item["loc"] if part not in {"query", "path"}),
                message=item["msg"],
            )
            for item in error.errors()
        ]
        body = ProblemResponse(
            code="invalid_request",
            message="One or more request values are invalid.",
            correlation_id=request.state.correlation_id,
            field_errors=field_errors,
        )
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception) -> JSONResponse:
        LOGGER.error(
            "Planner request failed correlation_id=%s error_type=%s",
            request.state.correlation_id,
            type(error).__name__,
        )
        return _problem(
            request,
            code="internal_error",
            message="The planner service could not complete the request.",
            status_code=500,
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[_local_origin(os.getenv("ICOR_WEB_ORIGIN", DEFAULT_WEB_ORIGIN))],
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["Accept", "Content-Type"],
    )
    app.include_router(router)
    return app
