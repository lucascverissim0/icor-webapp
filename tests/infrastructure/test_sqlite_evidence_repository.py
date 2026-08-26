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

    with repository._connect() as connection, pytest.raises(sqlite3.OperationalError):
        connection.execute("INSERT INTO source_release (release_id) VALUES ('direct-write')")


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

    assert repository.get_published_value(published_value.value_id) == published_value


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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("measure", Measure.ACTIVE_FLEET),
        ("unit", "registrations"),
        ("geography", "FR"),
        ("geography_version", "fr-2024"),
        ("period_start", date(2023, 1, 1)),
        ("period_end", date(2025, 12, 31)),
    ),
)
def test_published_value_rejects_semantically_incompatible_input(
    repository: SQLiteEvidenceRepository,
    release: ReleaseManifest,
    vehicle: CanonicalVehicle,
    observation: Observation,
    published_value: PublishedValue,
    field: str,
    value: object,
) -> None:
    incompatible_observation = replace(observation, **{field: value})
    seed_dependencies(repository, release, vehicle, incompatible_observation)

    with pytest.raises(ImmutableEvidenceError, match="incompatible"):
        repository.add_published_values((published_value,))


def test_published_value_rejects_input_for_a_different_vehicle(
    repository: SQLiteEvidenceRepository,
    release: ReleaseManifest,
    vehicle: CanonicalVehicle,
    observation: Observation,
    published_value: PublishedValue,
) -> None:
    different_vehicle = replace(vehicle, vehicle_id="vehicle-vw-golf-2025", model_year=2025)
    repository.add_release(release)
    repository.add_vehicle(vehicle)
    repository.add_vehicle(different_vehicle)
    repository.add_observations(
        (replace(observation, canonical_vehicle_id=different_vehicle.vehicle_id),)
    )

    with pytest.raises(ImmutableEvidenceError, match="incompatible"):
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


def test_all_lists_are_deterministically_ordered(
    repository: SQLiteEvidenceRepository,
    release: ReleaseManifest,
    vehicle: CanonicalVehicle,
    observation: Observation,
    mapping: IdentityMapping,
    published_value: PublishedValue,
    snapshot: SnapshotManifest,
) -> None:
    later_release = replace(release, release_id="eea-2025")
    later_vehicle = replace(vehicle, vehicle_id="vehicle-vw-golf-2025", model_year=2025)
    later_observation = replace(
        observation,
        observation_id="observation-eea-de-2024-2",
        original_row_locator="sheet1:3",
    )
    later_mapping = replace(mapping, mapping_id="mapping-eea-de-2024-2")
    later_value = replace(published_value, value_id="published-eea-de-2024-2")
    later_snapshot = replace(snapshot, snapshot_id="snapshot-2025", release_ids=("eea-2025",))
    repository.add_release(later_release)
    repository.add_release(release)
    repository.add_vehicle(later_vehicle)
    repository.add_vehicle(vehicle)
    repository.add_observations((later_observation, observation))
    repository.add_mapping(later_mapping)
    repository.add_mapping(mapping)
    repository.add_published_values((later_value, published_value))
    repository.add_snapshot(later_snapshot)
    repository.add_snapshot(snapshot)

    assert [item.release_id for item in repository.list_releases()] == ["eea-2024", "eea-2025"]
    assert [item.vehicle_id for item in repository.list_vehicles()] == [
        "vehicle-vw-golf-2024",
        "vehicle-vw-golf-2025",
    ]
    assert [item.observation_id for item in repository.list_observations()] == [
        "observation-eea-de-2024-1",
        "observation-eea-de-2024-2",
    ]
    assert [item.mapping_id for item in repository.list_mappings()] == [
        "mapping-eea-de-2024-1",
        "mapping-eea-de-2024-2",
    ]
    assert [item.value_id for item in repository.list_published_values()] == [
        "published-eea-de-2024-1",
        "published-eea-de-2024-2",
    ]
    assert [item.snapshot_id for item in repository.list_snapshots()] == [
        "snapshot-2024",
        "snapshot-2025",
    ]


def test_canonical_vehicle_identity_is_unique(
    repository: SQLiteEvidenceRepository, vehicle: CanonicalVehicle
) -> None:
    repository.add_vehicle(vehicle)

    with pytest.raises(DuplicateEvidenceError):
        repository.add_vehicle(replace(vehicle, vehicle_id="vehicle-duplicate"))


def test_snapshot_requires_existing_releases(
    repository: SQLiteEvidenceRepository, snapshot: SnapshotManifest
) -> None:
    with pytest.raises(ImmutableEvidenceError, match="reference"):
        repository.add_snapshot(snapshot)


def test_snapshot_round_trips_existing_release_membership(
    repository: SQLiteEvidenceRepository, release: ReleaseManifest, snapshot: SnapshotManifest
) -> None:
    repository.add_release(release)
    repository.add_snapshot(snapshot)

    assert repository.get_snapshot(snapshot.snapshot_id) == snapshot


def test_snapshot_membership_has_database_foreign_keys(
    tmp_path: Path, release: ReleaseManifest, snapshot: SnapshotManifest
) -> None:
    path = tmp_path / "evidence.sqlite3"
    repository = SQLiteEvidenceRepository(path, writable=True)
    repository.add_release(release)
    repository.add_snapshot(snapshot)

    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO snapshot_release (snapshot_id, release_id, release_position)
                VALUES (?, ?, ?)""",
                (snapshot.snapshot_id, "missing-release", 1),
            )


def test_structurally_corrupt_v1_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "corrupt-v1.sqlite3"
    SQLiteEvidenceRepository(path, writable=True)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE published_value_input")

    with pytest.raises(EvidenceSchemaError, match="schema"):
        SQLiteEvidenceRepository(path)


@pytest.mark.parametrize(
    ("expected", "replacement"),
    (
        (
            "measure TEXT NOT NULL CHECK (measure IN ('new_registrations', 'active_fleet'))",
            "measure TEXT NOT NULL",
        ),
        ("artifact_bytes INTEGER NOT NULL", "artifact_bytes TEXT NOT NULL"),
        ("publisher TEXT NOT NULL", "publisher TEXT"),
    ),
)
def test_schema_contract_rejects_altered_column_constraints(
    tmp_path: Path, expected: str, replacement: str
) -> None:
    path = tmp_path / "altered-v1.sqlite3"
    SQLiteEvidenceRepository(path, writable=True)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA writable_schema = ON")
        connection.execute(
            "UPDATE sqlite_master SET sql = REPLACE(sql, ?, ?) WHERE name = 'source_release'",
            (expected, replacement),
        )
        connection.execute("PRAGMA writable_schema = OFF")

    with pytest.raises(EvidenceSchemaError, match="schema"):
        SQLiteEvidenceRepository(path)


def test_failed_migration_leaves_no_version_table(tmp_path: Path) -> None:
    class FailingMigrationRepository(SQLiteEvidenceRepository):
        def _migration_statements(self) -> tuple[str, ...]:
            return (*super()._migration_statements(), "CREATE TABLE (")

    path = tmp_path / "migration-failure.sqlite3"

    with pytest.raises(sqlite3.OperationalError):
        FailingMigrationRepository(path, writable=True)

    with sqlite3.connect(path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    assert tables == []
