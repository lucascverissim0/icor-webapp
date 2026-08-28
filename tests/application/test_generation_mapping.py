from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from icor.application.generation_mapping import GenerationMappingService
from icor.domain.evidence import (
    CanonicalVehicle,
    EvidenceConfidence,
    MappingStatus,
    Measure,
    Observation,
    PeriodPrecision,
    PublicationStatus,
    ReleaseManifest,
)
from icor.domain.generations import AssignmentMethod, GenerationIdentityKind
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository


def confidence() -> EvidenceConfidence:
    return EvidenceConfidence(25, 10, 25, 10, 10, ("Official evidence.",))


def observation(
    identifier: str,
    *,
    measure: Measure,
    period_year: int,
    vehicle_id: str | None,
    status: MappingStatus,
    registration_cohort_year: int | None = None,
    manufacture_year: int | None = None,
) -> Observation:
    return Observation(
        observation_id=identifier,
        release_id="release-official-history",
        original_row_locator=f"row:{identifier}",
        geography="DE",
        geography_version="de-v1",
        period_start=date(period_year, 1, 1),
        period_end=date(period_year, 12, 31),
        period_precision=PeriodPrecision.YEAR,
        measure=measure,
        value=Decimal("10"),
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        original_make="Volkswagen",
        original_model="Golf",
        original_model_year=None,
        original_type=None,
        source_make_identifier="VOLKSWAGEN",
        source_model_identifier="VOLKSWAGEN GOLF",
        normalized_make="volkswagen",
        normalized_model="golf",
        normalized_model_year=None,
        canonical_vehicle_id=vehicle_id,
        mapping_status=status,
        transformation_notes=("Test observation.",),
        validation_flags=(),
        evidence_confidence=confidence(),
        registration_cohort_year=registration_cohort_year,
        manufacture_year=manufacture_year,
        model_year=None,
    )


def test_generation_mapping_assigns_every_usable_observation_once(tmp_path: Path) -> None:
    repository = SQLiteEvidenceRepository(tmp_path / "evidence.sqlite3", writable=True)
    release = ReleaseManifest(
        release_id="release-official-history",
        source_id="official-history",
        publisher="Official publisher",
        source_url="https://example.test/history",
        retrieved_at=datetime(2026, 8, 28, tzinfo=UTC),
        published_at=datetime(2026, 8, 1, tzinfo=UTC),
        coverage_start=date(2014, 1, 1),
        coverage_end=date(2025, 12, 31),
        geography="DE",
        geography_version="de-v1",
        measure=Measure.NEW_REGISTRATIONS,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="official-register",
        terms_url="https://example.test/terms",
        permitted_local_use="Reuse permitted.",
        artifact_path="artifact.csv",
        artifact_bytes=1,
        sha256="a" * 64,
        parser_name="test-parser",
        parser_version="v1",
        expected_schema="test-v1",
        raw_record_count=4,
        accepted_record_count=4,
        rejected_record_count=0,
        quarantined_record_count=0,
    )
    vehicle = CanonicalVehicle(
        "vehicle-volkswagen-golf-eu",
        "Volkswagen",
        "Golf",
        None,
        "Europe",
    )
    repository.add_release(release)
    repository.add_vehicle(vehicle)
    repository.add_observations(
        (
            observation(
                "observation-registration-2024",
                measure=Measure.NEW_REGISTRATIONS,
                period_year=2024,
                vehicle_id=vehicle.vehicle_id,
                status=MappingStatus.NORMALIZED_LABEL,
            ),
            observation(
                "observation-age-2020",
                measure=Measure.ACTIVE_FLEET,
                period_year=2025,
                vehicle_id=vehicle.vehicle_id,
                status=MappingStatus.NORMALIZED_LABEL,
                registration_cohort_year=2020,
            ),
            observation(
                "observation-stock-evidence-only",
                measure=Measure.ACTIVE_FLEET,
                period_year=2025,
                vehicle_id=vehicle.vehicle_id,
                status=MappingStatus.NORMALIZED_LABEL,
            ),
            observation(
                "observation-rejected",
                measure=Measure.NEW_REGISTRATIONS,
                period_year=2024,
                vehicle_id=None,
                status=MappingStatus.REJECTED,
            ),
        )
    )

    result = GenerationMappingService().apply(
        repository,
        reviewed_at=datetime(2026, 8, 28, 8, 0, tzinfo=UTC),
    )

    assert result.usable_count == 2
    assert result.assigned_count == 2
    assert result.evidence_only_count == 1
    assert result.unassigned_ids == ()
    generations = repository.list_generations()
    assert len(generations) == 1
    assert generations[0].identity_kind is GenerationIdentityKind.ESTIMATED
    assert generations[0].start_month.year == 2020
    assert generations[0].end_month is not None
    assert generations[0].end_month.year == 2024
    assignments = repository.list_generation_assignments()
    assert len(assignments) == 2
    assert {item.method for item in assignments} == {
        AssignmentMethod.ESTIMATED_GENERATION
    }
