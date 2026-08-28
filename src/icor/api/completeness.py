"""Snapshot completeness reporting route."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from icor.api.schemas import CompletenessResponse, ProblemResponse

router = APIRouter(tags=["completeness"])


@router.get(
    "/api/completeness",
    response_model=CompletenessResponse,
    responses={503: {"model": ProblemResponse}},
)
def completeness(request: Request) -> CompletenessResponse | JSONResponse:
    service = request.app.state.completeness_service
    if service is None:
        problem = ProblemResponse(
            code="completeness_unavailable",
            message="No verified active completeness report is available.",
            correlation_id=request.state.correlation_id,
        )
        return JSONResponse(status_code=503, content=problem.model_dump(mode="json"))
    return CompletenessResponse.model_validate(service.report())
