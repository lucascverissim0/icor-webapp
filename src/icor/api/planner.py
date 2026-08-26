"""Thin HTTP routes for planner use cases."""

from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from icor.api.schemas import (
    HealthResponse,
    PlannerOptionsResponse,
    PlannerPageResponse,
    PlanningConfigurationResponse,
    ProblemResponse,
)
from icor.application.planner import PlannerService
from icor.domain.planner import (
    EvidenceStatus,
    PlannerQuery,
    SortDirection,
    SortField,
)

router = APIRouter()
PROBLEM_RESPONSES = {
    422: {"model": ProblemResponse},
    500: {"model": ProblemResponse},
}


def _service(request: Request) -> PlannerService:
    return request.app.state.planner_service


def _correlation_id(request: Request) -> str:
    return request.state.correlation_id


@router.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    options = _service(request).options()
    return HealthResponse(
        status="ok",
        fixture_ready=True,
        data_version=options.scenario.data_version,
    )


@router.get(
    "/api/v1/planner/options",
    response_model=PlannerOptionsResponse,
    responses=PROBLEM_RESPONSES,
)
def planner_options(request: Request) -> PlannerOptionsResponse:
    return PlannerOptionsResponse.model_validate(_service(request).options())


@router.get(
    "/api/v1/planner/configurations",
    response_model=PlannerPageResponse,
    responses=PROBLEM_RESPONSES,
)
def configurations(
    request: Request,
    market: Annotated[list[str] | None, Query()] = None,
    horizon: Annotated[list[int] | None, Query()] = None,
    brand: Annotated[list[str] | None, Query()] = None,
    model: Annotated[list[str] | None, Query()] = None,
    evidence: Annotated[list[EvidenceStatus] | None, Query()] = None,
    sort: SortField = SortField.BASE_DEMAND,
    direction: SortDirection = SortDirection.DESC,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
) -> PlannerPageResponse:
    query = PlannerQuery(
        markets=tuple(market or ()),
        horizons=tuple(horizon or ()),
        brands=tuple(brand or ()),
        models=tuple(model or ()),
        evidence=tuple(evidence or ()),
        sort=sort,
        direction=direction,
        page=page,
        page_size=page_size,
    )
    return PlannerPageResponse.model_validate(_service(request).search(query))


@router.get(
    "/api/v1/planner/configurations/{configuration_id}",
    response_model=PlanningConfigurationResponse,
    responses={
        404: {"model": ProblemResponse},
        422: {"model": ProblemResponse},
        500: {"model": ProblemResponse},
    },
)
def configuration_detail(
    configuration_id: str,
    request: Request,
) -> PlanningConfigurationResponse | JSONResponse:
    configuration = _service(request).detail(configuration_id)
    if configuration is None:
        problem = ProblemResponse(
            code="configuration_not_found",
            message="The requested configuration was not found.",
            correlation_id=_correlation_id(request),
        )
        return JSONResponse(status_code=404, content=problem.model_dump(mode="json"))
    return PlanningConfigurationResponse.model_validate(configuration)
