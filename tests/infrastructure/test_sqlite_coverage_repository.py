import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from icor.application.coverage import CoverageSchemaError, DuplicateCoverageError
from icor.domain.opportunities import CoverageMatchType, ProductionCoverage
from icor.infrastructure.sqlite_coverage_repository import SQLiteCoverageRepository


def coverage(
    coverage_id: str = "coverage-1",
    *,
    match_type: CoverageMatchType = CoverageMatchType.EXACT_CONFIGURATION,
    configuration_id: str | None = "demo-config-1",
    brand: str = "Aurora Mobility",
    model: str = "A1 Horizon",
    model_year: int = 2025,
    sku: str | None = "DEMO-AUR-A1-CAM",
) -> ProductionCoverage:
    timestamp = datetime(2026, 8, 26, 10, 0, tzinfo=UTC)
    return ProductionCoverage(
        coverage_id=coverage_id,
        match_type=match_type,
        configuration_id=configuration_id,
        brand=brand,
        model=model,
        model_year=model_year,
        sku=sku,
        note="Confirmed locally.",
        created_at=timestamp,
        updated_at=timestamp,
    )


@pytest.fixture
def repository(tmp_path: Path) -> SQLiteCoverageRepository:
    return SQLiteCoverageRepository(tmp_path / "coverage.sqlite3")


def test_repository_migrates_an_empty_database(tmp_path: Path) -> None:
    path = tmp_path / "coverage.sqlite3"
    repository = SQLiteCoverageRepository(path)

    assert repository.schema_version == 1
    assert repository.list_all() == ()
    assert path.is_file()


def test_repository_round_trips_create_update_and_delete(
    repository: SQLiteCoverageRepository,
) -> None:
    original = coverage()
    assert repository.create(original) == original
    assert repository.get(original.coverage_id) == original

    updated = replace(
        original,
        note="Updated local evidence.",
        updated_at=original.updated_at + timedelta(minutes=1),
    )
    assert repository.update(updated) == updated
    assert repository.list_all() == (updated,)
    assert repository.delete(updated.coverage_id) is True
    assert repository.delete(updated.coverage_id) is False


def test_duplicate_exact_identity_rolls_back(
    repository: SQLiteCoverageRepository,
) -> None:
    original = coverage()
    repository.create(original)

    with pytest.raises(DuplicateCoverageError):
        repository.create(coverage("coverage-2"))

    assert repository.list_all() == (original,)


def test_duplicate_fallback_identity_rolls_back(
    repository: SQLiteCoverageRepository,
) -> None:
    original = coverage(
        match_type=CoverageMatchType.VEHICLE_YEAR_FALLBACK,
        configuration_id=None,
        sku=None,
    )
    repository.create(original)

    with pytest.raises(DuplicateCoverageError):
        repository.create(replace(original, coverage_id="coverage-2"))

    assert repository.list_all() == (original,)


def test_update_returns_none_for_a_missing_identity(
    repository: SQLiteCoverageRepository,
) -> None:
    assert repository.update(coverage()) is None


def test_repository_refuses_an_unsupported_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "coverage.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        connection.execute("INSERT INTO schema_version (version) VALUES (999)")

    with pytest.raises(CoverageSchemaError, match="unsupported"):
        SQLiteCoverageRepository(path)


def test_repository_reports_corrupt_database_without_rebuilding(tmp_path: Path) -> None:
    path = tmp_path / "coverage.sqlite3"
    original = b"not-a-sqlite-database"
    path.write_bytes(original)

    with pytest.raises(CoverageSchemaError, match="cannot be read"):
        SQLiteCoverageRepository(path)

    assert path.read_bytes() == original
