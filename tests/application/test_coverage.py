from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from icor.application.coverage import (
    CanonicalCoverageError,
    CoverageNotFoundError,
    CreateCoverageCommand,
    ProductionCoverageService,
)
from icor.domain.opportunities import CoverageMatchType, ProductionCoverage
from icor.infrastructure.demo_planner_repository import DemoPlannerRepository

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data" / "demo" / "planner-v1.json"


class MemoryCoverageRepository:
    def __init__(self) -> None:
        self.rows: dict[str, ProductionCoverage] = {}

    def list_all(self) -> tuple[ProductionCoverage, ...]:
        return tuple(self.rows[key] for key in sorted(self.rows))

    def get(self, coverage_id: str) -> ProductionCoverage | None:
        return self.rows.get(coverage_id)

    def create(self, coverage: ProductionCoverage) -> ProductionCoverage:
        self.rows[coverage.coverage_id] = coverage
        return coverage

    def update(self, coverage: ProductionCoverage) -> ProductionCoverage | None:
        if coverage.coverage_id not in self.rows:
            return None
        self.rows[coverage.coverage_id] = coverage
        return coverage

    def delete(self, coverage_id: str) -> bool:
        return self.rows.pop(coverage_id, None) is not None


@pytest.fixture
def repository() -> MemoryCoverageRepository:
    return MemoryCoverageRepository()


@pytest.fixture
def service(repository: MemoryCoverageRepository) -> ProductionCoverageService:
    now = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
    timestamps = iter((now, now + timedelta(minutes=1)))
    return ProductionCoverageService(
        DemoPlannerRepository.from_path(FIXTURE),
        repository,
        clock=lambda: next(timestamps),
        id_factory=lambda: "coverage-fixed",
    )


def test_exact_create_resolves_canonical_vehicle_and_sku(
    service: ProductionCoverageService,
) -> None:
    saved = service.create(
        CreateCoverageCommand(
            match_type=CoverageMatchType.EXACT_CONFIGURATION,
            configuration_id="demo-aurora-a1-camera-fr-2030",
            brand=None,
            model=None,
            model_year=2025,
            note="Confirmed locally.",
        )
    )

    assert saved.brand == "Aurora Mobility"
    assert saved.model == "A1 Horizon"
    assert saved.sku == "DEMO-AUR-A1-CAM"
    assert saved.configuration_id == "demo-aurora-a1-camera-fr-2030"


def test_fallback_create_requires_a_known_canonical_vehicle_year(
    service: ProductionCoverageService,
) -> None:
    with pytest.raises(CanonicalCoverageError, match="canonical"):
        service.create(
            CreateCoverageCommand(
                match_type=CoverageMatchType.VEHICLE_YEAR_FALLBACK,
                configuration_id=None,
                brand="Unknown",
                model="Vehicle",
                model_year=2025,
                note=None,
            )
        )


def test_update_preserves_identity_creation_time_and_changes_note(
    service: ProductionCoverageService,
) -> None:
    saved = service.create(
        CreateCoverageCommand(
            match_type=CoverageMatchType.EXACT_CONFIGURATION,
            configuration_id="demo-aurora-a1-camera-fr-2030",
            brand=None,
            model=None,
            model_year=2025,
            note=None,
        )
    )
    updated = service.update(
        saved.coverage_id,
        CreateCoverageCommand(
            match_type=CoverageMatchType.EXACT_CONFIGURATION,
            configuration_id=saved.configuration_id,
            brand=None,
            model=None,
            model_year=2025,
            note="Now documented.",
        ),
    )

    assert updated.coverage_id == saved.coverage_id
    assert updated.created_at == saved.created_at
    assert updated.updated_at > saved.updated_at
    assert updated.note == "Now documented."


def test_update_and_delete_report_missing_identity(
    service: ProductionCoverageService,
) -> None:
    command = CreateCoverageCommand(
        match_type=CoverageMatchType.VEHICLE_YEAR_FALLBACK,
        configuration_id=None,
        brand="Aurora Mobility",
        model="A1 Horizon",
        model_year=2025,
        note=None,
    )
    with pytest.raises(CoverageNotFoundError):
        service.update("missing", command)
    with pytest.raises(CoverageNotFoundError):
        service.delete("missing")
