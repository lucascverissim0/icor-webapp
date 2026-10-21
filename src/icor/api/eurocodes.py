"""Eurocode status and data coverage: what the running service can claim about its codes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from icor.api.schemas import (
    EurocodeCoverageResponse,
    EurocodeDataCoverageResponse,
    EurocodeDataReportResponse,
    EurocodeStatusResponse,
    ProblemResponse,
)
from icor.eurocodes.data_coverage import DOCUMENT_FORMAT, DOCUMENT_VERSION

router = APIRouter(tags=["eurocodes"])

_REPORT_FIELDS = (
    "snapshot_id",
    "year",
    "thresholds",
    "totals",
    "codes",
    "no_data",
    "markets",
    "country_gap",
)
_RUN_REPORT = "Run scripts/report_eurocode_data.py to measure the active snapshot."


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


def report_path(directory: Path, snapshot_id: str) -> Path:
    return directory / f"data-coverage-{snapshot_id}.json"


@router.get(
    "/api/v1/eurocodes/data-coverage",
    response_model=EurocodeDataCoverageResponse,
    responses={422: {"model": ProblemResponse}, 500: {"model": ProblemResponse}},
)
def eurocode_data_coverage(request: Request) -> EurocodeDataCoverageResponse | JSONResponse:
    """The private data report written for the active snapshot, never another's.

    The report is measured offline (it reads every cohort of the snapshot) and
    names real codes, so it lives beside the private map. A report written for
    another snapshot or catalogue is not served: its numbers would describe
    data this service does not show.
    """

    eurocodes = request.app.state.eurocodes
    if eurocodes is None:
        body = ProblemResponse(
            code="eurocodes_not_loaded",
            message="Eurocodes are not loaded in this service.",
            correlation_id=request.state.correlation_id,
        )
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))
    snapshot_id = request.app.state.eurocode_report_snapshot_id
    directory = request.app.state.eurocode_report_dir
    if snapshot_id is None or directory is None:
        return EurocodeDataCoverageResponse(
            available=False, message="No snapshot report can be read by this service."
        )
    path = report_path(directory, snapshot_id)
    if not path.is_file():
        return EurocodeDataCoverageResponse(available=False, message=_RUN_REPORT)
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("format") != DOCUMENT_FORMAT
        or document.get("version") != DOCUMENT_VERSION
        or document.get("snapshot_id") != snapshot_id
        or document.get("catalogue_sha256") != eurocodes.catalogue_sha256
    ):
        return EurocodeDataCoverageResponse(
            available=False,
            message="The report on disk was written for another snapshot or catalogue. "
            + _RUN_REPORT,
        )
    fields = {name: document[name] for name in _REPORT_FIELDS}
    fields["official_source"] = document.get("official_source") or None
    return EurocodeDataCoverageResponse(
        available=True, report=EurocodeDataReportResponse.model_validate(fields)
    )
