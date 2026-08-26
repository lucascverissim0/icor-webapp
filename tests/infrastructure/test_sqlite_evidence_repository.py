from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from icor.domain.evidence import (
    CanonicalVehicle,
    EvidenceConfidence,
    IdentityMapping,
    MappingStatus,
    Measure,
    Observation,
    PeriodPrecision,
    PublicationStatus,
    PublishedValue,
    ReleaseManifest,
    ValueStatus,
)
from icor.domain.snapshots import SnapshotManifest, SnapshotStatus, SnapshotVersions
from icor.infrastructure.sqlite_evidence_repository import (
    DuplicateEvidenceError,
    EvidenceSchemaError,
    ImmutableEvidenceError,
    SQLiteEvidenceRepository,
)


def confidence() -> EvidenceConfidence:
    return EvidenceConfidence(25, 10, 25, 20, 20, ("authoritative source",))


@pytest.fixture
def release() -> ReleaseManifest:
    return ReleaseManifest(
        release_id="eea-2024",
        source_id="eea",
        publisher="European Environment Agency",
        source_url="https://example.test/eea/2024",
        retrieved_at=datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
        published_at=datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
        coverage_start=date(2024, 1, 1),
        coverage_end=date(2024, 12, 31),
        geography="DE",
        geography_version="de-2024",
        measure=Measure.NEW_REGISTRATIONS,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="eea-direct",
        terms_url="https://example.test/terms",
        permitted_local_use="Research use permitted.",
        artifact_path="artifact.csv",
        artifact_bytes=20,
        sha256="a" * 64,
        parser_name="eea_csv",
        parser_version="v1",
        expected_schema="eea-2024-v1",
        raw_record_count=1,
        accepted_record_count=1,
        rejected_record_count=0,
        quarantined_record_count=0,
    )


@pytest.fixture
def vehicle() -> CanonicalVehicle:
    return CanonicalVehicle("vehicle-vw-golf-2024", "Volkswagen", "Golf", 2024, "DE")


@pytest.fixture
def observation(vehicle: CanonicalVehicle) -> Observation:
    return Observation(
        observation_id="observation-eea-de-2024-1",
        release_id="eea-2024",
        original_row_locator="sheet1:2",
        geography="DE",
        geography_version="de-2024",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        period_precision=PeriodPrecision.YEAR,
        measure=Measure.NEW_REGISTRATIONS,
        value=Decimal("1"),
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        original_make="Volkswagen",
        original_model="Golf",
        original_model_year="2024",
        original_type=None,
        source_make_identifier="vw",
        source_model_identifier="golf",
        normalized_make="Volkswagen",
        normalized_model="Golf",
        normalized_model_year=2024,
        canonical_vehicle_id=vehicle.vehicle_id,
        mapping_status=MappingStatus.EXACT_IDENTIFIER,
        transformation_notes=("source row normalized",),
        validation_flags=(),
        evidence_confidence=confidence(),
    )


@pytest.fixture
def mapping(vehicle: CanonicalVehicle, observation: Observation) -> IdentityMapping:
    return IdentityMapping(
        mapping_id="mapping-eea-de-2024-1",
        observation_id=observation.observation_id,
        canonical_vehicle_id=vehicle.vehicle_id,
        status=MappingStatus.EXACT_IDENTIFIER,
        reason="source identifier matched registry",
        reviewed_at=datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
    )


@pytest.fixture
def published_value(vehicle: CanonicalVehicle, observation: Observation) -> PublishedValue:
    return PublishedValue(
        value_id="published-eea-de-2024-1",
        status=ValueStatus.OBSERVED,
        measure=Measure.NEW_REGISTRATIONS,
        unit="vehicles",
        geography="DE",
        geography_version="de-2024",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        canonical_vehicle_id=vehicle.vehicle_id,
        mapping_status=MappingStatus.EXACT_IDENTIFIER,
        value=Decimal("1"),
        p10=None,
        p50=None,
        p90=None,
        input_ids=(observation.observation_id,),
        method_version="observed-v1",
        evidence_confidence=confidence(),
        forecast_confidence=None,
        warnings=("direct observation",),
    )


@pytest.fixture
def snapshot() -> SnapshotManifest:
    return SnapshotManifest(
        snapshot_id="snapshot-2024",
        status=SnapshotStatus.CANDIDATE,
        built_at=datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
        deterministic_seed=17,
        release_ids=("eea-2024",),
        versions=SnapshotVersions(*("v1",) * 8),
        database_sha256="b" * 64,
        observation_count=1,
        published_value_count=1,
        warnings=("candidate snapshot",),
    )


@pytest.fixture
def repository(tmp_path: Path) -> SQLiteEvidenceRepository:
    return SQLiteEvidenceRepository(tmp_path / "evidence.sqlite3", writable=True)


def seed_dependencies(
    repository: SQLiteEvidenceRepository,
    release: ReleaseManifest,
    vehicle: CanonicalVehicle,
    observation: Observation,
) -> None:
    repository.add_release(release)
    repository.add_vehicle(vehicle)
    repository.add_observations((observation,))


def test_empty_database_migrates_to_schema_v1(tmp_path: Path) -> None:
    repository = SQLiteEvidenceRepository(tmp_path / "evidence.sqlite3", writable=True)

    assert repository.schema_version == 1


def test_future_schema_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        connection.execute("INSERT INTO schema_version (version) VALUES (2)")

    with pytest.raises(EvidenceSchemaError, match="newer"):
        SQLiteEvidenceRepository(path, writable=True)


@pytest.mark.parametrize("setup", ("missing", "corrupt"))
def test_existing_database_requires_valid_version_table(tmp_path: Path, setup: str) -> None:
    path = tmp_path / f"{setup}.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        if setup == "corrupt":
            connection.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")

    with pytest.raises(EvidenceSchemaError, match="schema version"):
        SQLiteEvidenceRepository(path, writable=True)


def test_read_only_repository_rejects_writes(
    tmp_path: Path, release: ReleaseManifest
) -> None:
    path = tmp_path / "evidence.sqlite3"
    SQLiteEvidenceRepository(path, writable=True)
    repository = SQLiteEvidenceRepository(path)

    with pytest.raises(ImmutableEvidenceError, match="read-only"):
        repository.add_release(release)


def test_observation_identity_is_immutable(
    repository: SQLiteEvidenceRepository,
    release: ReleaseManifest,
    vehicle: CanonicalVehicle,
    observation: Observation,
) -> None:
    repository.add_release(release)
    repository.add_vehicle(vehicle)
    repository.add_observations((observation,))

    with pytest.raises(DuplicateEvidenceError):
        repository.add_observations((replace(observation, value=Decimal("2")),))

    assert repository.get_observation(observation.observation_id) == observation


def test_observation_batch_rolls_back_when_one_identity_is_duplicate(
    repository: SQLiteEvidenceRepository,
    release: ReleaseManifest,
    vehicle: CanonicalVehicle,
    observation: Observation,
) -> None:
    repository.add_release(release)
    repository.add_vehicle(vehicle)
    repository.add_observations((observation,))
    later = replace(
        observation,
        observation_id="observation-eea-de-2024-2",
        original_row_locator="sheet1:3",
    )

    with pytest.raises(DuplicateEvidenceError):
        repository.add_observations((later, replace(observation, value=Decimal("2"))))

    assert repository.get_observation(later.observation_id) is None


def test_observation_requires_an_existing_release(
    repository: SQLiteEvidenceRepository, observation: Observation
) -> None:
    with pytest.raises(ImmutableEvidenceError, match="reference"):
        repository.add_observations((observation,))


def test_source_row_locator_is_unique_per_release(
    repository: SQLiteEvidenceRepository,
    release: ReleaseManifest,
    vehicle: CanonicalVehicle,
    observation: Observation,
) -> None:
    repository.add_release(release)
    repository.add_vehicle(vehicle)
    repository.add_observations((observation,))

    with pytest.raises(DuplicateEvidenceError, match="row locator"):
        repository.add_observations(
            (replace(observation, observation_id="observation-eea-de-2024-2"),)
        )


def test_mapping_requires_existing_vehicle_and_observation(
    repository: SQLiteEvidenceRepository, mapping: IdentityMapping
) -> None:
    with pytest.raises(ImmutableEvidenceError, match="reference"):
        repository.add_mapping(mapping)


def test_mapping_status_is_retained(
    repository: SQLiteEvidenceRepository,
    release: ReleaseManifest,
    vehicle: CanonicalVehicle,
    observation: Observation,
    mapping: IdentityMapping,
) -> None:
    seed_dependencies(repository, release, vehicle, observation)
    repository.add_mapping(mapping)

    assert repository.get_mapping(mapping.mapping_id) == mapping


def test_published_value_retains_every_input(
    repository: SQLiteEvidenceRepository,
    release: ReleaseManifest,
    vehicle: CanonicalVehicle,
    observation: Observation,
    published_value: PublishedValue,
) -> None:
    seed_dependencies(repository, release, vehicle, observation)
    repository.add_published_values((published_value,))

    assert repository.get_published_value(published_value.value_id).input_ids == (
        "observation-eea-de-2024-1",
    )


def test_published_value_requires_existing_vehicle_and_inputs(
    repository: SQLiteEvidenceRepository, published_value: PublishedValue
) -> None:
    with pytest.raises(ImmutableEvidenceError, match="reference"):
        repository.add_published_values((published_value,))


def test_published_value_rejects_ambiguous_input(
    repository: SQLiteEvidenceRepository,
    release: ReleaseManifest,
    vehicle: CanonicalVehicle,
    observation: Observation,
    published_value: PublishedValue,
) -> None:
    ambiguous = replace(
        observation,
        canonical_vehicle_id=None,
        mapping_status=MappingStatus.AMBIGUOUS,
    )
    seed_dependencies(repository, release, vehicle, ambiguous)

    with pytest.raises(ImmutableEvidenceError, match="unresolved"):
        repository.add_published_values((published_value,))


def test_list_observations_is_deterministically_ordered(
    repository: SQLiteEvidenceRepository,
    release: ReleaseManifest,
    vehicle: CanonicalVehicle,
    observation: Observation,
) -> None:
    repository.add_release(release)
    repository.add_vehicle(vehicle)
    repository.add_observations(
        (
            replace(
                observation,
                observation_id="observation-eea-de-2024-z",
                original_row_locator="z",
            ),
            replace(
                observation,
                observation_id="observation-eea-de-2024-a",
                original_row_locator="a",
            ),
        )
    )

    assert [item.observation_id for item in repository.list_observations()] == [
        "observation-eea-de-2024-a",
        "observation-eea-de-2024-z",
    ]


def test_snapshot_requires_existing_releases(
    repository: SQLiteEvidenceRepository, snapshot: SnapshotManifest
) -> None:
    with pytest.raises(ImmutableEvidenceError, match="reference"):
        repository.add_snapshot(snapshot)
