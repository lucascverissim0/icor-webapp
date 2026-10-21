"""Eurocode status: what the running service can claim about its codes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from icor.api.schemas import EurocodeCoverageResponse, EurocodeStatusResponse, ProblemResponse

router = APIRouter(tags=["eurocodes"])


@router.get(
    "/api/v1/eurocodes/status",
    response_model=EurocodeStatusResponse,
    responses={500: {"model": ProblemResponse}},
)
def eurocode_status(request: Request) -> EurocodeStatusResponse:
    status = request.app.state.eurocode_status.status()
    return EurocodeStatusResponse(
        loaded=status.state.loaded,
        catalogue_sha256=status.state.catalogue_sha256,
        decisions_snapshot_id=status.state.decisions_snapshot_id,
        reviewed_identities=status.state.reviewed_identities,
        orphaned_identities=status.state.orphaned_identities,
        warnings=status.state.warnings,
        coverage=tuple(EurocodeCoverageResponse.model_validate(item) for item in status.coverage),
    )
