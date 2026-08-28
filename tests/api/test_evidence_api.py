from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from icor.api.app import create_app
from icor.application.evidence_review import (
    EvidenceObservationPage,
    EvidenceObservationQuery,
    EvidenceObservationRow,
    EvidenceReleaseSummary,
    EvidenceSummary,
)
from icor.domain.snapshots import SnapshotVersions
from icor.infrastructure.sqlite_coverage_repository import SQLiteCoverageRepository

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1", "localhost"])


class StaticEvidenceService:
    def __init__(self) -> None:
        self.query: EvidenceObservationQuery | None = None

    def summary(self) -> EvidenceSummary:
        return EvidenceSummary(
            snapshot_id="snapshot-review",
            status="candidate",
            built_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
            database_sha256="a" * 64,
            observation_count=3,
            published_value_count=0,
            warning_count=0,
            versions=SnapshotVersions(*("review-v1",) * 8),
            releases=(
                EvidenceReleaseSummary(
                    release_id="eea-final",
                    source_id="eea",
                    publisher="EEA publisher",
                    source_url="https://example.test/eea",
                    terms_url="https://example.test/terms",
                    published_at=datetime(2026, 8, 7, tzinfo=UTC),
                    coverage_start=date(2024, 1, 1),
                    coverage_end=date(2024, 12, 31),
                    geography="EEA reporting countries",
                    measure="new_registrations",
                    dependency_group="eu-register",
                    raw_record_count=3,
                    accepted_record_count=2,
                    rejected_record_count=1,
                    quarantined_record_count=0,
                    observation_count=2,
                    total_value=Decimal("15"),
                ),
            ),
            mapping_status_counts={"rejected": 1, "unresolved": 2},
            geographies=("DE", "FR"),
            measures=("new_registrations",),
        )

    def list_observations(self, query: EvidenceObservationQuery) -> EvidenceObservationPage:
        self.query = query
        return EvidenceObservationPage(
            items=(
                EvidenceObservationRow(
                    observation_id="obs-eea-golf",
                    release_id="eea-final",
                    original_row_locator="group:rows-2-4:members-3",
                    geography="DE",
                    period_start=date(2024, 1, 1),
                    period_end=date(2024, 12, 31),
                    period_precision="year",
                    measure="new_registrations",
                    value=Decimal("12"),
                    unit="vehicles",
                    publication_status="final",
                    original_make="Volkswagen",
                    original_model="Golf",
                    original_model_year=None,
                    original_type="publisher type",
                    mapping_status="unresolved",
                    transformation_notes=("Preserved publisher label.",),
                    validation_flags=(),
                    confidence_total=70,
                    confidence_reasons=("Official source; identity unresolved.",),
                ),
            ),
            total=1,
            page=query.page,
            page_size=query.page_size,
            pages=1,
        )


def _client(tmp_path: Path, service: StaticEvidenceService | None) -> TestClient:
    return TestClient(
        create_app(
            coverage_repository=SQLiteCoverageRepository(tmp_path / "coverage.sqlite3"),
            evidence_service=service,
        )
    )


def test_evidence_summary_serializes_candidate_provenance(tmp_path: Path) -> None:
    with _client(tmp_path, StaticEvidenceService()) as client:
        response = client.get("/api/v1/evidence/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["snapshot_id"] == "snapshot-review"
    assert body["published_value_count"] == 0
    assert body["mapping_status_counts"] == {"rejected": 1, "unresolved": 2}
    assert body["releases"][0]["total_value"] == "15"
    assert body["releases"][0]["publisher"] == "EEA publisher"


def test_evidence_observations_pass_bounded_filters(tmp_path: Path) -> None:
    service = StaticEvidenceService()
    with _client(tmp_path, service) as client:
        response = client.get(
            "/api/v1/evidence/observations",
            params={
                "release_id": "eea-final",
                "geography": "DE",
                "measure": "new_registrations",
                "mapping_status": "unresolved",
                "search": "golf",
                "page": 2,
                "page_size": 10,
            },
        )

    assert response.status_code == 200
    assert service.query == EvidenceObservationQuery(
        release_id="eea-final",
        geography="DE",
        measure="new_registrations",
        mapping_status="unresolved",
        search="golf",
        page=2,
        page_size=10,
    )
    assert response.json()["items"][0]["original_model"] == "Golf"


def test_missing_candidate_returns_typed_unavailable_without_fixture_fallback(
    tmp_path: Path,
) -> None:
    with _client(tmp_path, None) as client:
        response = client.get("/api/v1/evidence/summary")

    assert response.status_code == 503
    assert response.json()["code"] == "evidence_unavailable"
    assert "path" not in response.text.casefold()
    assert "traceback" not in response.text.casefold()


def test_invalid_evidence_query_uses_existing_safe_problem_shape(tmp_path: Path) -> None:
    with _client(tmp_path, StaticEvidenceService()) as client:
        response = client.get("/api/v1/evidence/observations?page_size=101")

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_health_fails_closed_when_only_evidence_service_is_injected(tmp_path: Path) -> None:
    with _client(tmp_path, StaticEvidenceService()) as client:
        response = client.get("/api/health")

    assert response.status_code == 503
    assert response.json()["code"] == "planning_snapshot_unavailable"
