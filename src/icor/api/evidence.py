"""Local-only HTTP routes for reviewing sealed source-evidence candidates."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from icor.api.schemas import (
    EvidenceObservationPageResponse,
    EvidenceSummaryResponse,
    ProblemResponse,
)
from icor.application.evidence_review import EvidenceObservationQuery, EvidenceReviewService
from icor.domain.evidence import MappingStatus, Measure

router = APIRouter(prefix="/api/v1/evidence", tags=["source evidence"])


def _service(request: Request) -> EvidenceReviewService | None:
    return request.app.state.evidence_service


def _unavailable(request: Request) -> JSONResponse:
    body = ProblemResponse(
        code="evidence_unavailable",
        message="No verified source-evidence candidate is configured for local review.",
        correlation_id=request.state.correlation_id,
    )
    return JSONResponse(status_code=503, content=body.model_dump(mode="json"))


@router.get(
    "/summary",
    response_model=EvidenceSummaryResponse,
    responses={503: {"model": ProblemResponse}},
)
def summary(request: Request) -> EvidenceSummaryResponse | JSONResponse:
    service = _service(request)
    if service is None:
        return _unavailable(request)
    return EvidenceSummaryResponse.model_validate(service.summary())


@router.get(
    "/observations",
    response_model=EvidenceObservationPageResponse,
    responses={422: {"model": ProblemResponse}, 503: {"model": ProblemResponse}},
)
def observations(
    request: Request,
    release_id: Annotated[str | None, Query(max_length=80)] = None,
    geography: Annotated[str | None, Query(max_length=80)] = None,
    measure: Measure | None = None,
    mapping_status: MappingStatus | None = None,
    search: Annotated[str | None, Query(max_length=100)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
) -> EvidenceObservationPageResponse | JSONResponse:
    service = _service(request)
    if service is None:
        return _unavailable(request)
    result = service.list_observations(
        EvidenceObservationQuery(
            release_id=release_id,
            geography=geography,
            measure=measure.value if measure else None,
            mapping_status=mapping_status.value if mapping_status else None,
            search=search,
            page=page,
            page_size=page_size,
        )
    )
    return EvidenceObservationPageResponse.model_validate(result)
