"""Authenticated local-only machine-learning export route."""

from datetime import date
from hmac import compare_digest
from ipaddress import ip_address
from typing import Annotated

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, Response

from icor.api.schemas import ProblemResponse

router = APIRouter(tags=["exports"])


def _problem(request: Request, code: str, message: str, status: int) -> JSONResponse:
    body = ProblemResponse(
        code=code,
        message=message,
        correlation_id=request.state.correlation_id,
    )
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


def _is_local_client(request: Request) -> bool:
    if request.client is None:
        return False
    host = request.client.host
    if host in {"localhost", "testclient"}:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


@router.get(
    "/api/exports/ml.csv",
    responses={
        200: {"content": {"text/csv": {}}},
        403: {"model": ProblemResponse},
        503: {"model": ProblemResponse},
    },
)
def ml_export(
    request: Request,
    cutoff: date,
    export_token: Annotated[str | None, Header(alias="X-ICOR-Export-Token")] = None,
) -> Response:
    service = request.app.state.ml_export_service
    configured_token = request.app.state.export_token
    if service is None or configured_token is None:
        return _problem(
            request,
            "ml_export_unavailable",
            "The protected snapshot export is not configured.",
            503,
        )
    if (
        not _is_local_client(request)
        or export_token is None
        or not compare_digest(export_token, configured_token)
    ):
        return _problem(request, "ml_export_forbidden", "Export access denied.", 403)
    filename = f"icor-ml-{service.snapshot_id}-{cutoff.isoformat()}.csv"
    return Response(
        service.render_csv(cutoff),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )
