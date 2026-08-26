from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from hashlib import sha256
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
from icor.evidence.serialization import sha256_file
from icor.evidence.validation import ReleaseValidator, Severity, SnapshotValidator
from icor.infrastructure.release_store import ReleaseStore, StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository


def _release_manifest(contents: bytes) -> ReleaseManifest:
    return ReleaseManifest(
        release_id="eea-2024-20260826",
        source_id="eea",
        publisher="European Environment Agency",
        source_url="https://example.test/eea/2024",
        retrieved_at=datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
        published_at=datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
        coverage_start=date(2024, 1, 1),
        coverage_end=date(2024, 12, 31),
        geography="EU",
        geography_version="eu-2024",
        measure=Measure.NEW_REGISTRATIONS,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="eea-direct",
        terms_url="https://example.test/eea/terms",
        permitted_local_use="Research and local validation are permitted.",
        artifact_path="artifact.csv",
        artifact_bytes=len(contents),
        sha256=sha256(contents).hexdigest(),
        parser_name="eea_csv",
        parser_version="v1",
        expected_schema="eea-2024-v1",
        raw_record_count=1,
        accepted_record_count=1,
        rejected_record_count=0,
        quarantined_record_count=0,
    )


@pytest.fixture
def stored_release(tmp_path: Path) -> StoredRelease:
    contents = b"make,model,count\nA,B,1\n"
    artifact = tmp_path / "incoming.csv"
    artifact.write_bytes(contents)
    return ReleaseStore(tmp_path / "raw").stage(artifact, _release_manifest(contents))


def _release_with(stored_release: StoredRelease, **changes: object) -> StoredRelease:
    manifest = stored_release.manifest
    for name, value in changes.items():
        object.__setattr__(manifest, name, value)
    return stored_release


def test_checksum_failure_is_mandatory_and_blocks_release(stored_release: StoredRelease) -> None:
    stored_release.artifact_path.write_bytes(b"corrupted")

    report = ReleaseValidator().validate(stored_release)

    assert report.can_promote is False
    assert report.findings[0].code == "release.checksum_mismatch"
    assert report.findings[0].severity is Severity.ERROR


def test_release_reports_missing_artifact_without_a_path(stored_release: StoredRelease) -> None:
    stored_release.artifact_path.unlink()

    report = ReleaseValidator().validate(stored_release)

    assert report.findings == (
        report.findings[0],
    )
    assert report.findings[0].code == "release.artifact_missing"
    assert str(stored_release.artifact_path) not in report.findings[0].message


def test_release_reports_byte_size_mismatch(stored_release: StoredRelease) -> None:
    _release_with(stored_release, artifact_bytes=1)

    report = ReleaseValidator().validate(stored_release)

    assert "release.byte_size_mismatch" in {finding.code for finding in report.findings}


def test_release_reports_missing_terms_metadata(stored_release: StoredRelease) -> None:
    _release_with(stored_release, terms_url="")

    report = ReleaseValidator().validate(stored_release)

    assert "release.terms_metadata_missing" in {finding.code for finding in report.findings}


def test_record_counts_must_reconcile(stored_release: StoredRelease) -> None:
    _release_with(
        stored_release,
        raw_record_count=10,
        accepted_record_count=8,
        rejected_record_count=1,
        quarantined_record_count=0,
    )

    report = ReleaseValidator().validate(stored_release)

    assert any(finding.code == "release.record_count_mismatch" for finding in report.findings)


def test_release_reports_reversed_coverage(stored_release: StoredRelease) -> None:
    _release_with(stored_release, coverage_start=date(2025, 1, 1), coverage_end=date(2024, 1, 1))

    report = ReleaseValidator().validate(stored_release)

    assert "release.coverage_reversed" in {finding.code for finding in report.findings}


def test_valid_release_can_promote(stored_release: StoredRelease) -> None:
    report = ReleaseValidator().validate(stored_release)

    assert report == report.__class__(())
    assert report.can_promote is True


def _confidence() -> EvidenceConfidence:
    return EvidenceConfidence(25, 10, 25, 20, 20, ("authoritative source",))


@pytest.fixture
def repository(tmp_path: Path) -> SQLiteEvidenceRepository:
    return SQLiteEvidenceRepository(tmp_path / "evidence.sqlite3", writable=True)


@pytest.fixture
def evidence_records() -> tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue]:
    release = _release_manifest(b"make,model,count\nA,B,1\n")
    vehicle = CanonicalVehicle("vehicle-example-alpha-2024", "Example", "Alpha", 2024, "EU")
    observation = Observation(
        observation_id="observation-eea-eu-2024-1",
        release_id=release.release_id,
        original_row_locator="sheet1:2",
        geography="EU",
        geography_version="eu-2024",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        period_precision=PeriodPrecision.YEAR,
        measure=Measure.NEW_REGISTRATIONS,
        value=Decimal("1"),
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        original_make="Example",
        original_model="Alpha",
        original_model_year="2024",
        original_type=None,
        source_make_identifier="example",
        source_model_identifier="alpha",
        normalized_make="Example",
        normalized_model="Alpha",
        normalized_model_year=2024,
        canonical_vehicle_id=vehicle.vehicle_id,
        mapping_status=MappingStatus.EXACT_IDENTIFIER,
        transformation_notes=("source row normalized",),
        validation_flags=(),
        evidence_confidence=_confidence(),
    )
    value = PublishedValue(
        value_id="published-eea-eu-2024-1",
        status=ValueStatus.OBSERVED,
        measure=Measure.NEW_REGISTRATIONS,
        unit="vehicles",
        geography="EU",
        geography_version="eu-2024",
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
        evidence_confidence=_confidence(),
        forecast_confidence=None,
        warnings=("direct observation",),
    )
    return release, vehicle, observation, value


def _seed(
    repository: SQLiteEvidenceRepository,
    records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    release, vehicle, observation, value = records
    repository.add_release(release)
    repository.add_vehicle(vehicle)
    repository.add_observations((observation,))
    repository.add_published_values((value,))


def _snapshot(repository: SQLiteEvidenceRepository, **changes: object) -> SnapshotManifest:
    manifest = SnapshotManifest(
        snapshot_id="snapshot-20260826",
        status=SnapshotStatus.CANDIDATE,
        built_at=datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
        deterministic_seed=17,
        release_ids=("eea-2024-20260826",),
        versions=SnapshotVersions(*("v1",) * 8),
        database_sha256=sha256_file(repository.path),
        observation_count=1,
        published_value_count=1,
        warnings=("candidate snapshot",),
    )
    return replace(manifest, **changes)


def _corrupt(
    repository: SQLiteEvidenceRepository, statement: str, parameters: tuple[object, ...]
) -> None:
    with sqlite3.connect(repository.path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(statement, parameters)


def test_snapshot_rejects_orphan_published_inputs(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    _seed(repository, evidence_records)
    manifest = _snapshot(repository)
    _corrupt(
        repository,
        "DELETE FROM observation WHERE observation_id = ?",
        ("observation-eea-eu-2024-1",),
    )

    report = SnapshotValidator().validate(repository, manifest)

    assert report.can_promote is False
    assert "snapshot.orphan_input" in {finding.code for finding in report.findings}


def test_snapshot_rejects_unordered_intervals(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    _seed(repository, evidence_records)
    manifest = _snapshot(repository)
    _corrupt(
        repository,
        "UPDATE published_value SET p10 = ?, p50 = ?, p90 = ? WHERE value_id = ?",
        ("20", "10", "30", "published-eea-eu-2024-1"),
    )

    report = SnapshotValidator().validate(repository, manifest)

    assert "snapshot.interval_order" in {finding.code for finding in report.findings}


def test_snapshot_rejects_negative_interval_bounds(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    _seed(repository, evidence_records)
    manifest = _snapshot(repository)
    _corrupt(
        repository,
        "UPDATE published_value SET p10 = ?, p50 = ?, p90 = ? WHERE value_id = ?",
        ("-20", "-10", "-5", "published-eea-eu-2024-1"),
    )

    report = SnapshotValidator().validate(repository, manifest)

    assert "snapshot.interval_negative" in {finding.code for finding in report.findings}


def test_snapshot_rejects_inputs_with_missing_published_value(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    _seed(repository, evidence_records)
    manifest = _snapshot(repository)
    _corrupt(
        repository,
        "DELETE FROM published_value WHERE value_id = ?",
        ("published-eea-eu-2024-1",),
    )

    report = SnapshotValidator().validate(repository, manifest)

    assert "snapshot.orphan_input" in {finding.code for finding in report.findings}


@pytest.mark.parametrize(
    ("table", "record_id"),
    [
        ("observation", "observation-eea-eu-2024-1"),
        ("published_value", "published-eea-eu-2024-1"),
    ],
)
def test_snapshot_rejects_negative_values(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
    table: str,
    record_id: str,
) -> None:
    _seed(repository, evidence_records)
    manifest = _snapshot(repository)
    key = "observation_id" if table == "observation" else "value_id"
    _corrupt(repository, f"UPDATE {table} SET value = ? WHERE {key} = ?", ("-1", record_id))

    report = SnapshotValidator().validate(repository, manifest)

    assert "snapshot.negative_value" in {finding.code for finding in report.findings}


def test_snapshot_rejects_manifest_count_mismatch(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    _seed(repository, evidence_records)

    report = SnapshotValidator().validate(repository, _snapshot(repository, observation_count=2))

    assert "snapshot.observation_count_mismatch" in {finding.code for finding in report.findings}


def test_snapshot_rejects_unresolved_model_publication(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    _seed(repository, evidence_records)
    manifest = _snapshot(repository)
    _corrupt(
        repository,
        "UPDATE published_value SET mapping_status = ? WHERE value_id = ?",
        ("unresolved", "published-eea-eu-2024-1"),
    )

    report = SnapshotValidator().validate(repository, manifest)

    assert "snapshot.unresolved_publication" in {finding.code for finding in report.findings}


def test_snapshot_rejects_absent_releases(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    _seed(repository, evidence_records)

    report = SnapshotValidator().validate(
        repository, _snapshot(repository, release_ids=("missing",))
    )

    assert "snapshot.release_missing" in {finding.code for finding in report.findings}


def test_snapshot_rejects_release_used_by_observations_but_absent_from_manifest(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    release, _, observation, _ = evidence_records
    _seed(repository, evidence_records)
    extra_release = replace(release, release_id="eea-2023-20260826")
    repository.add_release(extra_release)
    repository.add_observations(
        (
            replace(
                observation,
                observation_id="observation-eea-eu-2023-1",
                original_row_locator="sheet1:3",
                release_id=extra_release.release_id,
            ),
        )
    )

    report = SnapshotValidator().validate(repository, _snapshot(repository, observation_count=2))

    assert "snapshot.release_unmanifested" in {finding.code for finding in report.findings}


def test_snapshot_allows_unused_stored_release(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    release, _, _, _ = evidence_records
    _seed(repository, evidence_records)
    repository.add_release(replace(release, release_id="eea-2023-20260826"))

    report = SnapshotValidator().validate(repository, _snapshot(repository))

    assert "snapshot.release_unmanifested" not in {finding.code for finding in report.findings}


def test_snapshot_rejects_database_hash_mismatch(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    _seed(repository, evidence_records)

    report = SnapshotValidator().validate(
        repository, _snapshot(repository, database_sha256="0" * 64)
    )

    finding = next(
        finding for finding in report.findings if finding.code == "snapshot.database_hash_mismatch"
    )
    assert finding.record_id is None


def test_snapshot_rejects_unresolved_linked_identity_mapping(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    _, _, observation, _ = evidence_records
    _seed(repository, evidence_records)
    repository.add_mapping(
        IdentityMapping(
            mapping_id="mapping-eea-eu-2024-unresolved",
            observation_id=observation.observation_id,
            canonical_vehicle_id=None,
            status=MappingStatus.UNRESOLVED,
            reason="requires review",
            reviewed_at=datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
        )
    )

    report = SnapshotValidator().validate(repository, _snapshot(repository))

    assert "snapshot.unresolved_publication" in {finding.code for finding in report.findings}


def test_snapshot_rejects_unresolved_linked_observation(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    _seed(repository, evidence_records)
    manifest = _snapshot(repository)
    _corrupt(
        repository,
        "UPDATE observation SET mapping_status = ? WHERE observation_id = ?",
        ("unresolved", "observation-eea-eu-2024-1"),
    )

    report = SnapshotValidator().validate(repository, manifest)

    assert "snapshot.unresolved_publication" in {finding.code for finding in report.findings}


def test_snapshot_sanitizes_mixed_type_orphan_identifiers_deterministically(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    _seed(repository, evidence_records)
    manifest = _snapshot(repository)
    _corrupt(
        repository,
        """INSERT INTO published_value_input (value_id, observation_id, input_position)
        VALUES (?, ?, ?)""",
        (b"raw value", "missing-observation-a", 2),
    )
    _corrupt(
        repository,
        """INSERT INTO published_value_input (value_id, observation_id, input_position)
        VALUES (?, ?, ?)""",
        ("raw value", "missing-observation-b", 3),
    )

    report = SnapshotValidator().validate(repository, manifest)

    orphan_findings = [
        finding for finding in report.findings if finding.code == "snapshot.orphan_input"
    ]
    assert [finding.record_id for finding in orphan_findings] == [None, None]


def test_snapshot_clean_report_can_promote(
    repository: SQLiteEvidenceRepository,
    evidence_records: tuple[ReleaseManifest, CanonicalVehicle, Observation, PublishedValue],
) -> None:
    _seed(repository, evidence_records)

    report = SnapshotValidator().validate(repository, _snapshot(repository))

    assert report.findings == ()
    assert report.can_promote is True
