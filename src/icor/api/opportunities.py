"""Thin HTTP routes for opportunity and production-coverage use cases."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import JSONResponse

from icor.api.schemas import (
    DeleteCoverageResponse,
    OpportunityContributionResponse,
    OpportunityDrillDownResponse,
    OpportunityFleetEstimateResponse,
    OpportunityPageResponse,
    OpportunityRowResponse,
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
    OpportunitySort,
)

router = APIRouter()
PROBLEM_RESPONSES = {
    404: {"model": ProblemResponse},
    409: {"model": ProblemResponse},
    422: {"model": ProblemResponse},
    503: {"model": ProblemResponse},
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
    page: int = 1,
    page_size: int = 25,
    text: str = "",
    sort: OpportunitySort = OpportunitySort.SCORE,
) -> OpportunityQuery:
    # The detail routes resolve one group that the caller already holds, so they
    # leave text and sort at their defaults: narrowing or reordering a lookup of
    # a known row could only hide it.
    return OpportunityQuery(
        group_by=group_by,
        markets=tuple(market or ()),
        horizons=tuple(horizon or ()),
        text=text,
        sort=sort,
        page=page,
        page_size=page_size,
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
    responses={
        422: {"model": ProblemResponse},
        500: {"model": ProblemResponse},
        503: {"model": ProblemResponse},
    },
)
def opportunities(
    request: Request,
    group_by: OpportunityGroupBy = OpportunityGroupBy.BRAND,
    market: Annotated[list[str] | None, Query()] = None,
    horizon: Annotated[list[int] | None, Query()] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
    q: Annotated[str, Query(max_length=64)] = "",
    sort: OpportunitySort = OpportunitySort.SCORE,
) -> OpportunityPageResponse | JSONResponse:
    if (
        getattr(request.app.state, "client_release", False)
        and group_by is not OpportunityGroupBy.MODEL_YEAR
    ):
        return _problem(
            request,
            status_code=422,
            code="client_release_scope",
            message="The verified client release supports model-year opportunities only.",
        )
    service = _opportunity_service(request)
    if service is None:
        return _snapshot_unavailable(request)
    result = service.list(
        _query(group_by, market, horizon, page, page_size, q, sort)
    )
    return OpportunityPageResponse.model_validate(result)


@router.get(
    "/api/v1/opportunities/{group_id}",
    response_model=OpportunityRowResponse,
    responses=PROBLEM_RESPONSES,
)
def opportunity_detail(
    group_id: str,
    request: Request,
    group_by: OpportunityGroupBy = OpportunityGroupBy.BRAND,
    market: Annotated[list[str] | None, Query()] = None,
    horizon: Annotated[list[int] | None, Query()] = None,
) -> OpportunityRowResponse | JSONResponse:
    if (
        getattr(request.app.state, "client_release", False)
        and group_by is not OpportunityGroupBy.MODEL_YEAR
    ):
        return _problem(
            request,
            status_code=422,
            code="client_release_scope",
            message="The verified client release supports model-year opportunities only.",
        )
    service = _opportunity_service(request)
    if service is None:
        return _snapshot_unavailable(request)
    row = service.get(group_id, _query(group_by, market, horizon))
    if row is None:
        return _problem(
            request,
            status_code=404,
            code="opportunity_not_found",
            message="The requested opportunity was not found.",
        )
    return OpportunityRowResponse.model_validate(row)


@router.get(
    "/api/v1/opportunities/{group_id}/fleet",
    response_model=list[OpportunityFleetEstimateResponse],
    responses=PROBLEM_RESPONSES,
)
def opportunity_fleet(
    group_id: str,
    request: Request,
    group_by: OpportunityGroupBy = OpportunityGroupBy.BRAND,
    market: Annotated[list[str] | None, Query()] = None,
    horizon: Annotated[list[int] | None, Query()] = None,
) -> list[OpportunityFleetEstimateResponse] | JSONResponse:
    if (
        getattr(request.app.state, "client_release", False)
        and group_by is not OpportunityGroupBy.MODEL_YEAR
    ):
        return _problem(
            request,
            status_code=422,
            code="client_release_scope",
            message="The verified client release supports model-year opportunities only.",
        )
    service = _opportunity_service(request)
    if service is None:
        return _snapshot_unavailable(request)
    rows = service.fleet_estimates(group_id, _query(group_by, market, horizon))
    if not rows:
        return _problem(
            request,
            status_code=404,
            code="opportunity_not_found",
            message="The requested opportunity was not found.",
        )
    return [OpportunityFleetEstimateResponse.model_validate(row) for row in rows]


@router.get(
    "/api/v1/opportunities/{group_id}/contributions",
    response_model=list[OpportunityContributionResponse],
    responses=PROBLEM_RESPONSES,
)
def opportunity_contributions(
    group_id: str,
    request: Request,
    group_by: OpportunityGroupBy = OpportunityGroupBy.BRAND,
    market: Annotated[list[str] | None, Query()] = None,
    horizon: Annotated[list[int] | None, Query()] = None,
) -> list[OpportunityContributionResponse] | JSONResponse:
    if (
        getattr(request.app.state, "client_release", False)
        and group_by is not OpportunityGroupBy.MODEL_YEAR
    ):
        return _problem(
            request,
            status_code=422,
            code="client_release_scope",
            message="The verified client release supports model-year opportunities only.",
        )
    service = _opportunity_service(request)
    if service is None:
        return _snapshot_unavailable(request)
    rows = service.contributions(group_id, _query(group_by, market, horizon))
    if not rows:
        return _problem(
            request,
            status_code=404,
            code="opportunity_not_found",
            message="The requested opportunity was not found.",
        )
    return [OpportunityContributionResponse.model_validate(row) for row in rows]


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
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 100,
) -> list[OpportunityDrillDownResponse] | JSONResponse:
    if (
        getattr(request.app.state, "client_release", False)
        and group_by is not OpportunityGroupBy.MODEL_YEAR
    ):
        return _problem(
            request,
            status_code=422,
            code="client_release_scope",
            message="The verified client release supports model-year opportunities only.",
        )
    service = _opportunity_service(request)
    if service is None:
        return _snapshot_unavailable(request)
    rows = service.drill_down(
        group_id, _query(group_by, market, horizon), page, page_size
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
    responses={500: {"model": ProblemResponse}, 503: {"model": ProblemResponse}},
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
