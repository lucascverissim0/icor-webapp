"""FastAPI application factory for the local planner service."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from icor.api.completeness import router as completeness_router
from icor.api.eurocodes import router as eurocode_router
from icor.api.evidence import router as evidence_router
from icor.api.exports import router as export_router
from icor.api.opportunities import router as opportunity_router
from icor.api.planner import router as planner_router
from icor.api.registrations import router as registration_router
from icor.api.schemas import FieldError, ProblemResponse
from icor.api.vehicle_forecasts import router as vehicle_forecast_router
from icor.application.client_snapshot import reviewed_snapshot_id
from icor.application.completeness import CompletenessQueryService
from icor.application.coverage import CoverageRepository, ProductionCoverageService
from icor.application.evidence_review import EvidenceReviewService
from icor.application.ml_export import MLExportService
from icor.application.opportunities import (
    EurocodeGroupingUnavailable,
    OpportunityGroupBy,
    OpportunityService,
)
from icor.application.planner import PlannerRepository, PlannerService
from icor.application.ranking import DemandScoreV2
from icor.application.registrations import RegistrationService
from icor.application.worked_models import IcorWorkedModelCatalog
from icor.eurocodes.runtime import NOT_LOADED, EurocodeMap, EurocodeState
from icor.infrastructure.snapshot_eurocodes import EurocodeStatus, SnapshotEurocodeStatus
from icor.infrastructure.snapshot_opportunity_repository import SnapshotOpportunityRepository
from icor.infrastructure.snapshot_planner_repository import SnapshotPlannerRepository
from icor.infrastructure.snapshot_store import (
    SnapshotStore,
    SnapshotUnavailableError,
    trusts_baked_snapshot,
)
from icor.infrastructure.snapshot_vehicle_forecast_repository import (
    SnapshotVehicleForecastRepository,
)
from icor.infrastructure.sqlite_coverage_repository import SQLiteCoverageRepository

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEB_ORIGIN = "http://127.0.0.1:5173"
DEFAULT_COVERAGE_DB = ROOT / ".local" / "production-coverage.sqlite3"
DEFAULT_EVIDENCE_ROOT = ROOT / ".local" / "evidence"
DEFAULT_EUROCODE_DIR = ROOT / ".local" / "private" / "eurocode"


class _FixedEurocodeStatus:
    """A status known without measuring: nothing loaded, or no snapshot to measure."""

    def __init__(self, status: EurocodeStatus) -> None:
        self._status = status

    def status(self) -> EurocodeStatus:
        return self._status


def _configured_eurocodes(
    *, local_active_snapshot: bool
) -> tuple[EurocodeMap | None, str | None, Path | None]:
    """The private Eurocode map this process serves, why none is, and its folder.

    ``ICOR_EUROCODE_DIR`` names the folder explicitly, and a folder named that
    way that cannot be loaded stops the process rather than serving rows
    without codes. Without it, only a local service reading the active snapshot
    falls back to the private folder in this checkout, when present; the client
    release and services composed with their own sources never do, so what
    they serve does not depend on which machine they run on.
    """

    configured = os.getenv("ICOR_EUROCODE_DIR")
    if configured:
        return EurocodeMap.from_directory(Path(configured)), None, Path(configured)
    if not local_active_snapshot or not (
        (DEFAULT_EUROCODE_DIR / "catalog.json").is_file()
        and (DEFAULT_EUROCODE_DIR / "decisions.json").is_file()
    ):
        return None, None, None
    try:
        return EurocodeMap.from_directory(DEFAULT_EUROCODE_DIR), None, DEFAULT_EUROCODE_DIR
    except (OSError, ValueError, KeyError) as error:
        LOGGER.warning("Eurocodes not loaded error_type=%s", type(error).__name__)
        return (
            None,
            f"The private Eurocode files could not be loaded ({type(error).__name__}).",
            None,
        )


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


def create_app(
    repository: PlannerRepository | None = None,
    coverage_repository: CoverageRepository | None = None,
    evidence_service: EvidenceReviewService | None = None,
    registration_service: RegistrationService | None = None,
    completeness_service=None,
    ml_export_service=None,
    export_token: str | None = None,
    snapshot_root: Path | None = None,
    vehicle_forecast_service=None,
    client_release: bool = False,
    eurocodes: EurocodeMap | None = None,
    eurocode_report_dir: Path | None = None,
    eurocode_report_snapshot_id: str | None = None,
) -> FastAPI:
    app = FastAPI(
        title="ICOR Planner API",
        version="1.0.0",
        description="Local evidence-led contract for vehicle registration analysis and planning.",
    )
    selected_repository = repository
    snapshot_manifest = None
    snapshot_ledger = None
    has_explicit_service_override = any(
        service is not None
        for service in (
            evidence_service,
            registration_service,
            completeness_service,
            ml_export_service,
            vehicle_forecast_service,
        )
    )
    if selected_repository is None and not has_explicit_service_override:
        root = snapshot_root or Path(os.getenv("ICOR_EVIDENCE_ACTIVE_ROOT", DEFAULT_EVIDENCE_ROOT))
        try:
            snapshot_manifest, snapshot_ledger = SnapshotStore(
                root, trust_verified_marker=trusts_baked_snapshot()
            ).open_active_snapshot()
            if snapshot_manifest.versions.generation_registry.endswith(
                "-v0"
            ) or snapshot_manifest.versions.generation_resolver.endswith("-v0"):
                raise SnapshotUnavailableError(
                    "active snapshot does not contain generation planning data"
                )
            selected_repository = SnapshotPlannerRepository(snapshot_ledger, snapshot_manifest)
        except SnapshotUnavailableError as error:
            snapshot_manifest = None
            snapshot_ledger = None
            LOGGER.warning("Planning snapshot unavailable error_type=%s", type(error).__name__)
    eurocode_warning = None
    if eurocodes is None:
        eurocodes, eurocode_warning, eurocode_dir = _configured_eurocodes(
            local_active_snapshot=(
                not client_release
                and repository is None
                and snapshot_root is None
                and snapshot_manifest is not None
            )
        )
        if eurocode_report_dir is None and eurocode_dir is not None:
            # Data reports name real codes, so they live beside the private map.
            eurocode_report_dir = eurocode_dir / "reports"
    selected_coverage_repository = coverage_repository or SQLiteCoverageRepository(
        Path(os.getenv("ICOR_COVERAGE_DB", DEFAULT_COVERAGE_DB))
    )
    app.state.planner_service = (
        PlannerService(selected_repository) if selected_repository is not None else None
    )
    app.state.coverage_service = (
        ProductionCoverageService(selected_repository, selected_coverage_repository)
        if selected_repository is not None
        else None
    )
    if isinstance(selected_repository, SnapshotPlannerRepository) and isinstance(
        selected_coverage_repository, SQLiteCoverageRepository
    ):
        opportunity_repository = SnapshotOpportunityRepository(
            selected_repository,
            selected_coverage_repository,
            DemandScoreV2(),
            worked_models=IcorWorkedModelCatalog.from_path(
                ROOT / "data" / "icor_supported_models.txt"
            ),
            model_year_catalog=client_release,
            eurocodes=eurocodes,
        )
        if client_release:
            # Grouping the whole snapshot is what makes a score comparable
            # across filters, and it takes seconds. The client release serves
            # Eurocodes first and model years beneath them, so build those
            # populations at boot rather than charging the first visitor for
            # them; the host's health check has a grace period, a visitor has none.
            for grouping in (
                (OpportunityGroupBy.EUROCODE, OpportunityGroupBy.MODEL_YEAR)
                if eurocodes is not None
                else (OpportunityGroupBy.MODEL_YEAR,)
            ):
                opportunity_repository.warm(grouping)
                LOGGER.info("Ranked market population ready group_by=%s", grouping)
        elif eurocodes is not None:
            # The Eurocode ranking is the interface's first view, and allocating
            # every forecast cell takes seconds; build it while the app starts.
            threading.Thread(
                target=opportunity_repository.warm,
                args=(OpportunityGroupBy.EUROCODE,),
                name="eurocode-ranking",
                daemon=True,
            ).start()
        app.state.opportunity_service = OpportunityService(
            repository=opportunity_repository
        )
    elif selected_repository is not None:
        app.state.opportunity_service = OpportunityService(
            selected_repository,
            selected_coverage_repository,
            DemandScoreV2(),
            eurocodes=eurocodes,
        )
    else:
        app.state.opportunity_service = None
    if evidence_service is None and snapshot_manifest is not None and snapshot_ledger is not None:
        evidence_service = EvidenceReviewService.from_snapshot(
            snapshot_ledger.path, snapshot_manifest
        )
    if (
        registration_service is None
        and snapshot_manifest is not None
        and snapshot_ledger is not None
    ):
        registration_service = RegistrationService(snapshot_ledger.path, snapshot_manifest)
    app.state.snapshot_manifest = snapshot_manifest
    app.state.snapshot_repository = snapshot_ledger
    app.state.eurocodes = eurocodes
    app.state.eurocode_report_dir = eurocode_report_dir
    # The snapshot a data report must have been written for. A service composed
    # without a snapshot (the browser fixture) names one explicitly.
    app.state.eurocode_report_snapshot_id = (
        snapshot_manifest.snapshot_id
        if snapshot_manifest is not None
        else eurocode_report_snapshot_id
    )
    if eurocodes is not None and snapshot_manifest is not None and snapshot_ledger is not None:
        snapshot_status = SnapshotEurocodeStatus(
            eurocodes,
            snapshot_ledger.path,
            # A client snapshot is its reviewed full snapshot, pruned.
            reviewed_snapshot_id(snapshot_manifest, snapshot_ledger.path),
        )
        # Measuring coverage reads every forecast cell; start it now so the
        # first visitor does not wait for it.
        snapshot_status.warm_in_background()
        app.state.eurocode_status = snapshot_status
    elif eurocodes is not None:
        app.state.eurocode_status = _FixedEurocodeStatus(
            EurocodeStatus(
                EurocodeState(
                    loaded=True,
                    catalogue_sha256=eurocodes.catalogue_sha256,
                    decisions_snapshot_id=eurocodes.decisions.metadata.get("snapshot_id"),
                    reviewed_identities=len(eurocodes.decisions.identities),
                    warnings=("No snapshot is open, so coverage is not measured.",),
                )
            )
        )
    else:
        app.state.eurocode_status = _FixedEurocodeStatus(
            EurocodeStatus(
                NOT_LOADED
                if eurocode_warning is None
                else EurocodeState(loaded=False, warnings=(eurocode_warning,))
            )
        )
    if (
        vehicle_forecast_service is None
        and snapshot_manifest is not None
        and snapshot_ledger is not None
    ):
        vehicle_forecast_service = SnapshotVehicleForecastRepository(
            snapshot_ledger.path,
            snapshot_manifest.snapshot_id,
            model_year_only=client_release,
            eurocodes=eurocodes,
        )
    app.state.vehicle_forecast_service = vehicle_forecast_service
    if completeness_service is None and snapshot_manifest is not None:
        completeness_service = CompletenessQueryService(snapshot_ledger, snapshot_manifest)
    if ml_export_service is None and snapshot_manifest is not None:
        ml_export_service = MLExportService(snapshot_ledger, snapshot_manifest.snapshot_id)
    configured_export_token = export_token or os.getenv("ICOR_EXPORT_TOKEN")
    if configured_export_token is not None and len(configured_export_token) < 32:
        raise ValueError("ICOR_EXPORT_TOKEN must contain at least 32 characters")
    app.state.completeness_service = completeness_service
    app.state.ml_export_service = ml_export_service
    app.state.export_token = configured_export_token
    app.state.evidence_service = evidence_service
    app.state.registration_service = registration_service
    app.state.client_release = client_release

    @app.middleware("http")
    async def correlation_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        request.state.correlation_id = uuid4().hex
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = request.state.correlation_id
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
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

    @app.exception_handler(EurocodeGroupingUnavailable)
    async def eurocodes_not_loaded(
        request: Request, error: EurocodeGroupingUnavailable
    ) -> JSONResponse:
        return _problem(
            request,
            code="eurocodes_not_loaded",
            message=(
                "Eurocodes are not loaded in this service, so the ranking cannot be "
                "read by Eurocode. Choose another grouping."
            ),
            status_code=422,
        )

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
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Accept", "Content-Type"],
    )
    app.include_router(planner_router)
    app.include_router(opportunity_router)
    app.include_router(eurocode_router)
    app.include_router(vehicle_forecast_router)
    app.include_router(evidence_router)
    app.include_router(registration_router)
    app.include_router(completeness_router)
    app.include_router(export_router)
    return app
