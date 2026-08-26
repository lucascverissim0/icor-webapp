from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from icor.application.snapshot_build import SnapshotBuilder, SnapshotBuildRequest
from icor.domain.evidence import Measure, PublicationStatus, ReleaseManifest
from icor.domain.snapshots import SnapshotVersions
from icor.evidence.validation import SnapshotValidator
from icor.infrastructure.release_store import ReleaseStore, StoredRelease
from icor.infrastructure.snapshot_store import (
    SnapshotPromotionError,
    SnapshotStore,
    SnapshotUnavailableError,
)
from icor.infrastructure.sqlite_evidence_repository import (
    ImmutableEvidenceError,
    SQLiteEvidenceRepository,
)

ARTIFACT = b"country,year,make,model,count\n"


class EmptyLoader:
    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        assert releases


def _manifest() -> ReleaseManifest:
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
        raw_record_count=0,
        accepted_record_count=0,
        rejected_record_count=0,
        quarantined_record_count=0,
    )


@pytest.fixture
def evidence_root(tmp_path: Path) -> Path:
    return tmp_path / "evidence"


@pytest.fixture
def release_store(tmp_path: Path) -> ReleaseStore:
    artifact = tmp_path / "incoming.csv"
    artifact.write_bytes(ARTIFACT)
    store = ReleaseStore(tmp_path / "raw")
    store.stage(artifact, _manifest())
    return store


@pytest.fixture
def build_request() -> SnapshotBuildRequest:
    return SnapshotBuildRequest(
        release_ids=("sample-2024-20260826",),
        versions=SnapshotVersions(*("v1",) * 8),
        deterministic_seed=17,
        build_as_of=datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
    )


@pytest.fixture
def builder(evidence_root: Path, release_store: ReleaseStore) -> SnapshotBuilder:
    return SnapshotBuilder(evidence_root, release_store, EmptyLoader())


@pytest.fixture
def snapshot_store(evidence_root: Path) -> SnapshotStore:
    return SnapshotStore(
        evidence_root,
        clock=lambda: datetime(2026, 8, 26, 13, 0, tzinfo=UTC),
    )


def test_failed_candidate_leaves_active_snapshot_unchanged(
    snapshot_store: SnapshotStore,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
) -> None:
    valid_candidate = builder.build(build_request).manifest.snapshot_id
    invalid_candidate = "snapshot-invalid"
    snapshot_store.candidate_path(invalid_candidate).mkdir(parents=True)
    snapshot_store.promote(valid_candidate)
    active_before = snapshot_store.active_manifest()

    with pytest.raises(SnapshotPromotionError):
        snapshot_store.promote(invalid_candidate)

    assert snapshot_store.active_manifest() == active_before


def test_no_active_snapshot_is_typed_unavailable(snapshot_store: SnapshotStore) -> None:
    with pytest.raises(SnapshotUnavailableError):
        snapshot_store.open_active_repository()


def test_missing_candidate_is_typed_promotion_failure(snapshot_store: SnapshotStore) -> None:
    with pytest.raises(SnapshotPromotionError, match="candidate"):
        snapshot_store.promote("snapshot-missing")


@pytest.mark.parametrize("filename", ["evidence.sqlite3", "snapshot.json", "validation.json"])
def test_candidate_missing_required_file_cannot_promote(
    snapshot_store: SnapshotStore,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
    filename: str,
) -> None:
    candidate = builder.build(build_request)
    (candidate.candidate_path / filename).unlink()

    with pytest.raises(SnapshotPromotionError):
        snapshot_store.promote(candidate.manifest.snapshot_id)

    assert not snapshot_store.active_path.exists()


def test_hash_change_after_initial_validation_cannot_become_active(
    snapshot_store: SnapshotStore,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = builder.build(build_request)
    validate = SnapshotValidator.validate
    validation_calls = 0

    def corrupt_after_first_validation(
        validator: SnapshotValidator,
        repository: SQLiteEvidenceRepository,
        manifest: object,
    ) -> object:
        nonlocal validation_calls
        report = validate(validator, repository, manifest)  # type: ignore[arg-type]
        validation_calls += 1
        if validation_calls == 1:
            with repository.path.open("ab") as database:
                database.write(b"changed-after-validation")
        return report

    monkeypatch.setattr(SnapshotValidator, "validate", corrupt_after_first_validation)

    with pytest.raises(SnapshotPromotionError):
        snapshot_store.promote(candidate.manifest.snapshot_id)

    assert not snapshot_store.active_path.exists()


def test_interrupted_pointer_write_preserves_previous_active_snapshot(
    snapshot_store: SnapshotStore,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = builder.build(build_request)
    second = builder.build(
        replace(
            build_request,
            versions=replace(build_request.versions, forecast_method="forecast-v2"),
        )
    )
    snapshot_store.promote(first.manifest.snapshot_id)
    active_bytes = snapshot_store.active_path.read_bytes()
    replace_file = os.replace

    def interrupt_active_replace(source: Path | str, destination: Path | str) -> None:
        if Path(destination) == snapshot_store.active_path:
            raise OSError("simulated interruption")
        replace_file(source, destination)

    monkeypatch.setattr("icor.infrastructure.snapshot_store.os.replace", interrupt_active_replace)

    with pytest.raises(SnapshotPromotionError):
        snapshot_store.promote(second.manifest.snapshot_id)

    assert snapshot_store.active_path.read_bytes() == active_bytes
    assert snapshot_store.active_manifest() == first.manifest


def test_repeat_promotion_is_idempotent(
    snapshot_store: SnapshotStore,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
) -> None:
    candidate = builder.build(build_request)
    first = snapshot_store.promote(candidate.manifest.snapshot_id)
    pointer_bytes = snapshot_store.active_path.read_bytes()

    second = snapshot_store.promote(candidate.manifest.snapshot_id)

    assert second == first
    assert snapshot_store.active_path.read_bytes() == pointer_bytes
    assert [path.name for path in (snapshot_store.root / "snapshots").iterdir()] == [
        candidate.manifest.snapshot_id
    ]


def test_active_repository_is_read_only(
    snapshot_store: SnapshotStore,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
) -> None:
    candidate = builder.build(build_request)
    snapshot_store.promote(candidate.manifest.snapshot_id)

    repository = snapshot_store.open_active_repository()

    assert repository.writable is False
    with pytest.raises(ImmutableEvidenceError):
        repository.add_release(_manifest())
