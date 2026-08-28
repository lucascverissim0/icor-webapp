"""Thin HTTP routes for opportunity and production-coverage use cases."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import JSONResponse

from icor.api.schemas import (
    DeleteCoverageResponse,
    OpportunityDrillDownResponse,
    OpportunityPageResponse,
    ProblemResponse,
    ProductionCoverageRequest,
    ProductionCoverageResponse,
)
from icor.application.coverage import (
    CanonicalCoverageError,
    CoverageNotFoundError,
    CreateCoverageCommand,
    DuplicateCoverageError,
    ProductionCoverageService,
)
from icor.application.opportunities import (
    OpportunityGroupBy,
    OpportunityQuery,
    OpportunityService,
)

router = APIRouter()
PROBLEM_RESPONSES = {
    404: {"model": ProblemResponse},
    409: {"model": ProblemResponse},
    422: {"model": ProblemResponse},
    500: {"model": ProblemResponse},
}


def _opportunity_service(request: Request) -> OpportunityService | None:
    return request.app.state.opportunity_service


def _coverage_service(request: Request) -> ProductionCoverageService | None:
    return request.app.state.coverage_service


def _query(
    group_by: OpportunityGroupBy,
    market: list[str] | None,
    horizon: list[int] | None,
) -> OpportunityQuery:
    return OpportunityQuery(
        group_by=group_by,
        markets=tuple(market or ()),
        horizons=tuple(horizon or ()),
    )


def _problem(
    request: Request, *, status_code: int, code: str, message: str
) -> JSONResponse:
    body = ProblemResponse(
        code=code,
        message=message,
        correlation_id=request.state.correlation_id,
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


@router.get(
    "/api/v1/opportunities",
    response_model=OpportunityPageResponse,
    responses={422: {"model": ProblemResponse}, 500: {"model": ProblemResponse}},
)
def opportunities(
    request: Request,
    group_by: OpportunityGroupBy = OpportunityGroupBy.BRAND,
    market: Annotated[list[str] | None, Query()] = None,
    horizon: Annotated[list[int] | None, Query()] = None,
) -> OpportunityPageResponse | JSONResponse:
    service = _opportunity_service(request)
    if service is None:
        return _snapshot_unavailable(request)
    result = service.list(_query(group_by, market, horizon))
    return OpportunityPageResponse.model_validate(result)


@router.get(
    "/api/v1/opportunities/{group_id}/configurations",
    response_model=list[OpportunityDrillDownResponse],
    responses=PROBLEM_RESPONSES,
)
def opportunity_configurations(
    group_id: str,
    request: Request,
    group_by: OpportunityGroupBy = OpportunityGroupBy.BRAND,
    market: Annotated[list[str] | None, Query()] = None,
    horizon: Annotated[list[int] | None, Query()] = None,
) -> list[OpportunityDrillDownResponse] | JSONResponse:
    service = _opportunity_service(request)
    if service is None:
        return _snapshot_unavailable(request)
    rows = service.drill_down(
        group_id, _query(group_by, market, horizon)
    )
    if not rows:
        return _problem(
            request,
            status_code=404,
            code="opportunity_not_found",
            message="The requested opportunity was not found.",
        )
    return [OpportunityDrillDownResponse.model_validate(row) for row in rows]


@router.get(
    "/api/v1/production-coverage",
    response_model=list[ProductionCoverageResponse],
    responses={500: {"model": ProblemResponse}},
)
def production_coverage(
    request: Request,
) -> list[ProductionCoverageResponse] | JSONResponse:
    service = _coverage_service(request)
    if service is None:
        return _snapshot_unavailable(request)
    return [
        ProductionCoverageResponse.model_validate(row)
        for row in service.list_all()
    ]


def _command(payload: ProductionCoverageRequest) -> CreateCoverageCommand:
    return CreateCoverageCommand(
        match_type=payload.match_type,
        configuration_id=payload.configuration_id,
        brand=payload.brand,
        model=payload.model,
        model_year=payload.model_year,
        note=payload.note,
    )


def _mutation_error(request: Request, error: Exception) -> JSONResponse:
    if isinstance(error, CanonicalCoverageError):
        return _problem(
            request,
            status_code=422,
            code="invalid_canonical_coverage",
            message=str(error),
        )
    if isinstance(error, DuplicateCoverageError):
        return _problem(
            request,
            status_code=409,
            code="duplicate_coverage",
            message="Production coverage already exists for this canonical identity.",
        )
    return _problem(
        request,
        status_code=404,
        code="coverage_not_found",
        message="The requested production coverage was not found.",
    )


@router.post(
    "/api/v1/production-coverage",
    response_model=ProductionCoverageResponse,
    status_code=status.HTTP_201_CREATED,
    responses=PROBLEM_RESPONSES,
)
def create_production_coverage(
    payload: ProductionCoverageRequest, request: Request
) -> ProductionCoverageResponse | JSONResponse:
    service = _coverage_service(request)
    if service is None:
        return _snapshot_unavailable(request)
    try:
        saved = service.create(_command(payload))
    except (CanonicalCoverageError, DuplicateCoverageError) as error:
        return _mutation_error(request, error)
    return ProductionCoverageResponse.model_validate(saved)


@router.put(
    "/api/v1/production-coverage/{coverage_id}",
    response_model=ProductionCoverageResponse,
    responses=PROBLEM_RESPONSES,
)
def update_production_coverage(
    coverage_id: str, payload: ProductionCoverageRequest, request: Request
) -> ProductionCoverageResponse | JSONResponse:
    service = _coverage_service(request)
    if service is None:
        return _snapshot_unavailable(request)
    try:
        saved = service.update(coverage_id, _command(payload))
    except (
        CanonicalCoverageError,
        DuplicateCoverageError,
        CoverageNotFoundError,
    ) as error:
        return _mutation_error(request, error)
    return ProductionCoverageResponse.model_validate(saved)


@router.delete(
    "/api/v1/production-coverage/{coverage_id}",
    response_model=DeleteCoverageResponse,
    responses=PROBLEM_RESPONSES,
)
def delete_production_coverage(
    coverage_id: str, request: Request
) -> DeleteCoverageResponse | JSONResponse:
    service = _coverage_service(request)
    if service is None:
        return _snapshot_unavailable(request)
    try:
        service.delete(coverage_id)
    except CoverageNotFoundError as error:
        return _mutation_error(request, error)
    return DeleteCoverageResponse(coverage_id=coverage_id, deleted=True)


def _snapshot_unavailable(request: Request) -> JSONResponse:
    return _problem(
        request,
        status_code=503,
        code="planning_snapshot_unavailable",
        message="No verified active planning snapshot is available.",
    )
