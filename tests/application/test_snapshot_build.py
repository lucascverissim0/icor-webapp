from __future__ import annotations

import csv
import os
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest

from icor.application.snapshot_build import (
    SnapshotBuilder,
    SnapshotBuildError,
    SnapshotBuildRequest,
)
from icor.domain.evidence import (
    CanonicalVehicle,
    EvidenceConfidence,
    IdentityMapping,
    MappingStatus,
    Measure,
    Observation,
    PeriodPrecision,
    PublicationStatus,
    ReleaseManifest,
)
from icor.domain.snapshots import SnapshotStatus, SnapshotVersions
from icor.evidence.release_manifests import (
    load_snapshot_manifest,
    write_release_manifest,
)
from icor.evidence.serialization import canonical_json_bytes
from icor.infrastructure.release_store import ReleaseStore, StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

ARTIFACT = b"country,year,make,model,count\nDE,2024,Example,Alpha,10\nFR,2024,Example,Alpha,5\n"
TRANSIENT_ARTIFACT = (
    b"country,year,make,model,count\n"
    b"DE,2024,Example,Alpha,999\n"
    b"FR,2024,Example,Alpha,888\n"
)
BUILD_AS_OF = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


def _release_manifest() -> ReleaseManifest:
    return ReleaseManifest(
        release_id="sample-2024-20260826",
        source_id="sample",
        publisher="Example publisher",
        source_url="https://example.test/sample/2024",
        retrieved_at=datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
        published_at=datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
        coverage_start=date(2024, 1, 1),
        coverage_end=date(2024, 12, 31),
        geography="EU",
        geography_version="eu-2024",
        measure=Measure.NEW_REGISTRATIONS,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="sample-direct",
        terms_url="https://example.test/terms",
        permitted_local_use="Local contract testing is permitted.",
        artifact_path="artifact.csv",
        artifact_bytes=len(ARTIFACT),
        sha256=sha256(ARTIFACT).hexdigest(),
        parser_name="sample_csv",
        parser_version="v1",
        expected_schema="sample-v1",
        raw_record_count=2,
        accepted_record_count=2,
        rejected_record_count=0,
        quarantined_record_count=0,
    )


def _confidence() -> EvidenceConfidence:
    return EvidenceConfidence(25, 10, 25, 20, 0, ("single approved source",))


class TwoObservationLoader:
    def __init__(self, *, reverse: bool) -> None:
        self.reverse = reverse

    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        release_id = releases[0].release_id
        vehicle = CanonicalVehicle(
            "vehicle-example-alpha-2024", "Example", "Alpha", 2024, "EU"
        )
        repository.add_vehicle(vehicle)
        observations = [
            Observation(
                observation_id=f"observation-{country.casefold()}-2024",
                release_id=release_id,
                original_row_locator=f"row:{position}",
                geography=country,
                geography_version="eu-2024",
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                period_precision=PeriodPrecision.YEAR,
                measure=Measure.NEW_REGISTRATIONS,
                value=Decimal(value),
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
            for position, (country, value) in enumerate(
                (("DE", "10"), ("FR", "5")), start=2
            )
        ]
        if self.reverse:
            observations.reverse()
        repository.add_observations(observations)


class ChangingReleaseLoader(TwoObservationLoader):
    def __init__(
        self,
        *,
        reverse: bool,
        stored_artifact_path: Path,
        stored_manifest_path: Path,
    ) -> None:
        super().__init__(reverse=reverse)
        self.stored_artifact_path = stored_artifact_path
        self.stored_manifest_path = stored_manifest_path

    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        super().load(releases, repository)
        changed_artifact = b"changed-during-build\n"
        self.stored_artifact_path.write_bytes(changed_artifact)
        write_release_manifest(
            self.stored_manifest_path,
            replace(
                releases[0].manifest,
                artifact_bytes=len(changed_artifact),
                sha256=sha256(changed_artifact).hexdigest(),
            ),
        )


class TransientMutationLoader:
    def __init__(self, stored_artifact_path: Path) -> None:
        self.stored_artifact_path = stored_artifact_path

    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        original_bytes = self.stored_artifact_path.read_bytes()
        self.stored_artifact_path.write_bytes(TRANSIENT_ARTIFACT)
        try:
            with releases[0].artifact_path.open(encoding="utf-8", newline="") as artifact:
                rows = tuple(csv.DictReader(artifact))
        finally:
            self.stored_artifact_path.write_bytes(original_bytes)

        vehicle = CanonicalVehicle(
            "vehicle-example-alpha-2024", "Example", "Alpha", 2024, "EU"
        )
        repository.add_vehicle(vehicle)
        for position, row in enumerate(rows, start=2):
            country = row["country"]
            observation = Observation(
                observation_id=f"observation-{country.casefold()}-2024",
                release_id=releases[0].release_id,
                original_row_locator=f"row:{position}",
                geography=country,
                geography_version="eu-2024",
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                period_precision=PeriodPrecision.YEAR,
                measure=Measure.NEW_REGISTRATIONS,
                value=Decimal(row["count"]),
                unit="vehicles",
                publication_status=PublicationStatus.FINAL,
                original_make=row["make"],
                original_model=row["model"],
                original_model_year=row["year"],
                original_type=None,
                source_make_identifier=None,
                source_model_identifier=None,
                normalized_make=row["make"],
                normalized_model=row["model"],
                normalized_model_year=int(row["year"]),
                canonical_vehicle_id=vehicle.vehicle_id,
                mapping_status=MappingStatus.NORMALIZED_LABEL,
                transformation_notes=("source row normalized",),
                validation_flags=(),
                evidence_confidence=_confidence(),
            )
            repository.add_observations((observation,))
            repository.add_mapping(
                IdentityMapping(
                    mapping_id=f"mapping-{country.casefold()}-2024",
                    observation_id=observation.observation_id,
                    canonical_vehicle_id=vehicle.vehicle_id,
                    status=MappingStatus.NORMALIZED_LABEL,
                    reason="normalized source labels matched the registry",
                    reviewed_at=BUILD_AS_OF,
                )
            )


@pytest.fixture
def release_store(tmp_path: Path) -> ReleaseStore:
    artifact = tmp_path / "incoming.csv"
    artifact.write_bytes(ARTIFACT)
    store = ReleaseStore(tmp_path / "raw")
    store.stage(artifact, _release_manifest())
    return store


@pytest.fixture
def build_request() -> SnapshotBuildRequest:
    return SnapshotBuildRequest(
        release_ids=("sample-2024-20260826",),
        versions=SnapshotVersions(*("v1",) * 8),
        deterministic_seed=17,
        build_as_of=BUILD_AS_OF,
    )


def _builder(root: Path, release_store: ReleaseStore, *, reverse: bool) -> SnapshotBuilder:
    return SnapshotBuilder(root, release_store, TwoObservationLoader(reverse=reverse))


def test_identical_inputs_produce_identical_snapshot(
    tmp_path: Path,
    release_store: ReleaseStore,
    build_request: SnapshotBuildRequest,
) -> None:
    first = _builder(tmp_path / "first", release_store, reverse=False).build(build_request)
    second = _builder(tmp_path / "second", release_store, reverse=True).build(build_request)

    assert first.manifest.snapshot_id == second.manifest.snapshot_id
    assert first.manifest.database_sha256 == second.manifest.database_sha256
    assert first.database_path.read_bytes() == second.database_path.read_bytes()
    assert first.manifest.built_at == BUILD_AS_OF
    assert first.manifest.status is SnapshotStatus.CANDIDATE


def test_candidate_contains_canonical_manifest_database_and_validation_report(
    tmp_path: Path,
    release_store: ReleaseStore,
    build_request: SnapshotBuildRequest,
) -> None:
    result = _builder(tmp_path / "evidence", release_store, reverse=True).build(build_request)

    assert result.candidate_path == (
        tmp_path / "evidence" / "candidates" / result.manifest.snapshot_id
    )
    assert {path.name for path in result.candidate_path.iterdir()} == {
        "evidence.sqlite3",
        "snapshot.json",
        "validation.json",
    }
    assert load_snapshot_manifest(result.manifest_path) == result.manifest
    assert result.validation_report.can_promote is True
    assert result.validation_path.read_bytes() == canonical_json_bytes(result.validation_report)

    repository = SQLiteEvidenceRepository(result.database_path)
    assert [row.observation_id for row in repository.list_observations()] == [
        "observation-de-2024",
        "observation-fr-2024",
    ]


def test_snapshot_id_changes_when_method_version_changes(
    tmp_path: Path,
    release_store: ReleaseStore,
    build_request: SnapshotBuildRequest,
) -> None:
    builder = _builder(tmp_path / "evidence", release_store, reverse=True)
    first = builder.build(build_request)
    changed = replace(
        build_request,
        versions=replace(build_request.versions, confidence_method="confidence-v2"),
    )

    assert builder.build(changed).manifest.snapshot_id != first.manifest.snapshot_id


def test_snapshot_id_changes_when_build_as_of_changes(
    tmp_path: Path,
    release_store: ReleaseStore,
    build_request: SnapshotBuildRequest,
) -> None:
    builder = _builder(tmp_path / "evidence", release_store, reverse=False)
    first = builder.build(build_request)
    changed = replace(
        build_request,
        build_as_of=datetime(2026, 8, 26, 12, 1, tzinfo=UTC),
    )

    assert builder.build(changed).manifest.snapshot_id != first.manifest.snapshot_id


@pytest.mark.parametrize(
    "changes",
    [
        {"release_ids": ("sample-z", "sample-a")},
        {"release_ids": ("sample-a", "sample-a")},
        {"deterministic_seed": -1},
        {"build_as_of": datetime(2026, 8, 26, 12, 0)},
    ],
)
def test_build_request_rejects_nondeterministic_inputs(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "release_ids": ("sample-a",),
        "versions": SnapshotVersions(*("v1",) * 8),
        "deterministic_seed": 17,
        "build_as_of": BUILD_AS_OF,
    }
    values.update(changes)

    with pytest.raises(ValueError):
        SnapshotBuildRequest(**values)  # type: ignore[arg-type]


def test_build_rejects_release_replaced_during_loading(
    tmp_path: Path,
    release_store: ReleaseStore,
    build_request: SnapshotBuildRequest,
) -> None:
    root = tmp_path / "evidence"
    stored = release_store.verify(build_request.release_ids[0])
    builder = SnapshotBuilder(
        root,
        release_store,
        ChangingReleaseLoader(
            reverse=False,
            stored_artifact_path=stored.artifact_path,
            stored_manifest_path=stored.manifest_path,
        ),
    )

    with pytest.raises(SnapshotBuildError, match="changed during build"):
        builder.build(build_request)

    assert not (root / "candidates").exists() or not any(
        (root / "candidates").iterdir()
    )


def test_loader_consumes_private_checksummed_release_bytes_during_transient_mutation(
    tmp_path: Path,
    release_store: ReleaseStore,
    build_request: SnapshotBuildRequest,
) -> None:
    stored = release_store.verify(build_request.release_ids[0])
    builder = SnapshotBuilder(
        tmp_path / "evidence",
        release_store,
        TransientMutationLoader(stored.artifact_path),
    )

    result = builder.build(build_request)

    repository = SQLiteEvidenceRepository(result.database_path)
    assert tuple(row.value for row in repository.list_observations()) == (
        Decimal("10"),
        Decimal("5"),
    )
    assert result.validation_report.can_promote is True
    assert release_store.verify(stored.release_id).artifact_path.read_bytes() == ARTIFACT


def test_build_rejects_symlinked_candidates_directory(
    tmp_path: Path,
    release_store: ReleaseStore,
    build_request: SnapshotBuildRequest,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    outside = tmp_path / "outside-candidates"
    outside.mkdir()
    try:
        (root / "candidates").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if os.name == "nt" and error.winerror == 1314:
            pytest.skip(f"symlinks require Windows developer privileges: {error}")
        raise

    with pytest.raises(SnapshotBuildError, match="unsafe|contain"):
        _builder(root, release_store, reverse=False).build(build_request)

    assert list(outside.iterdir()) == []
