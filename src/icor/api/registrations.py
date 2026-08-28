"""Fail-closed HTTP routes for canonical official registration evidence."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from icor.api.schemas import (
    ProblemResponse,
    RegistrationPageResponse,
    RegistrationSummaryResponse,
)
from icor.application.registrations import (
    RegistrationQuery,
    RegistrationService,
    RegistrationUnavailableError,
)

router = APIRouter(prefix="/api/v1/registrations", tags=["official registrations"])


def _service(request: Request) -> RegistrationService | None:
    return request.app.state.registration_service


def _unavailable(request: Request) -> JSONResponse:
    problem = ProblemResponse(
        code="registration_data_unavailable",
        message="Verified official registration data is not available.",
        correlation_id=request.state.correlation_id,
    )
    return JSONResponse(status_code=503, content=problem.model_dump(mode="json"))


@router.get(
    "/summary",
    response_model=RegistrationSummaryResponse,
    response_model_exclude_none=True,
    responses={503: {"model": ProblemResponse}},
)
def summary(request: Request) -> RegistrationSummaryResponse | JSONResponse:
    service = _service(request)
    if service is None:
        return _unavailable(request)
    try:
        return RegistrationSummaryResponse.model_validate(service.summary())
    except RegistrationUnavailableError:
        return _unavailable(request)


@router.get(
    "/ranking",
    response_model=RegistrationPageResponse,
    responses={422: {"model": ProblemResponse}, 503: {"model": ProblemResponse}},
)
def ranking(
    request: Request,
    geography: str = "EU27",
    year: int = 2024,
    search: Annotated[str | None, Query(max_length=100)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
) -> RegistrationPageResponse | JSONResponse:
    service = _service(request)
    if service is None:
        return _unavailable(request)
    try:
        result = service.ranking(
            RegistrationQuery(
                geography=geography,
                year=year,
                search=search,
                page=page,
                page_size=page_size,
            )
        )
    except RegistrationUnavailableError:
        return _unavailable(request)
    return RegistrationPageResponse.model_validate(result)
