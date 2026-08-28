"""Explicit browser-test composition; never used by production or preview runners."""

from __future__ import annotations

import os
from pathlib import Path

from icor.api.app import create_app
from icor.application.completeness import CompletenessQueryService
from icor.application.evidence_review import EvidenceReviewService
from icor.application.registrations import RegistrationService
from icor.infrastructure.demo_planner_repository import DemoPlannerRepository
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

ROOT = Path(__file__).resolve().parents[1]


def create_e2e_app():  # type: ignore[no-untyped-def]
    """Compose sealed evidence and synthetic interaction fixtures only for Playwright."""
    evidence_path = _required_candidate("ICOR_E2E_EVIDENCE_CANDIDATE")
    generation_path = _required_candidate("ICOR_E2E_GENERATION_CANDIDATE")
    evidence_service = EvidenceReviewService.from_candidate(evidence_path)
    generation_service = EvidenceReviewService.from_candidate(generation_path)
    generation_repository = SQLiteEvidenceRepository(generation_service.database_path)
    return create_app(
        repository=DemoPlannerRepository.from_path(ROOT / "data" / "demo" / "planner-v1.json"),
        evidence_service=evidence_service,
        registration_service=RegistrationService.from_candidate(evidence_path),
        completeness_service=CompletenessQueryService(
            generation_repository, generation_service.manifest
        ),
    )


def _required_candidate(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError("E2E candidate configuration is required")
    return Path(value)
