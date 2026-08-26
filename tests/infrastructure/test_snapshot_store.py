from __future__ import annotations

import os
import sqlite3
import stat
import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from icor.application.snapshot_build import SnapshotBuilder, SnapshotBuildRequest
from icor.domain.evidence import Measure, PublicationStatus, ReleaseManifest
from icor.domain.snapshots import SnapshotManifest, SnapshotStatus, SnapshotVersions
from icor.evidence.serialization import canonical_json_bytes, sha256_file
from icor.evidence.validation import SnapshotValidator
from icor.infrastructure.release_store import ReleaseStore, StoredRelease
from icor.infrastructure.snapshot_filesystem import SnapshotFilesystem, SnapshotPathError
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

    monkeypatch.setattr(
        "icor.infrastructure.snapshot_filesystem.os.replace", interrupt_active_replace
    )

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


def _rewrite_candidate_manifest(
    candidate_path: Path, manifest: SnapshotManifest
) -> None:
    (candidate_path / "snapshot.json").write_bytes(canonical_json_bytes(manifest))
    report = SnapshotValidator().validate(
        SQLiteEvidenceRepository(candidate_path / "evidence.sqlite3"), manifest
    )
    assert report.can_promote
    (candidate_path / "validation.json").write_bytes(canonical_json_bytes(report))


@pytest.mark.parametrize(
    "identity_field",
    [
        "built_at",
        "deterministic_seed",
        "source_registry",
        "identity_registry",
        "reconciliation_method",
        "confidence_method",
        "estimation_method",
        "survival_method",
        "hazard_method",
        "forecast_method",
    ],
)
def test_promotion_rejects_manifest_identity_edits_that_retain_the_old_id(
    snapshot_store: SnapshotStore,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
    identity_field: str,
) -> None:
    candidate = builder.build(build_request)
    manifest = candidate.manifest
    if identity_field == "built_at":
        changed = replace(
            manifest, built_at=datetime(2026, 8, 26, 12, 1, tzinfo=UTC)
        )
    elif identity_field == "deterministic_seed":
        changed = replace(manifest, deterministic_seed=18)
    else:
        changed = replace(
            manifest,
            versions=replace(manifest.versions, **{identity_field: "forged-v2"}),
        )
    _rewrite_candidate_manifest(candidate.candidate_path, changed)

    with pytest.raises(SnapshotPromotionError, match="identity"):
        snapshot_store.promote(manifest.snapshot_id)

    assert not snapshot_store.active_path.exists()


def test_promotion_rejects_changed_persisted_release_hash_with_old_snapshot_id(
    snapshot_store: SnapshotStore,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
) -> None:
    candidate = builder.build(build_request)
    with closing(sqlite3.connect(candidate.database_path)) as connection:
        connection.execute(
            "UPDATE source_release SET sha256 = ? WHERE release_id = ?",
            ("0" * 64, "sample-2024-20260826"),
        )
        connection.commit()
    changed = replace(
        candidate.manifest,
        database_sha256=sha256_file(candidate.database_path),
    )
    _rewrite_candidate_manifest(candidate.candidate_path, changed)

    with pytest.raises(SnapshotPromotionError, match="identity"):
        snapshot_store.promote(candidate.manifest.snapshot_id)

    assert not snapshot_store.active_path.exists()


def test_promotion_rejects_extra_persisted_release_and_preserves_active_pointer(
    snapshot_store: SnapshotStore,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
) -> None:
    first = builder.build(build_request)
    second = builder.build(
        replace(
            build_request,
            versions=replace(build_request.versions, hazard_method="hazard-v2"),
        )
    )
    snapshot_store.promote(first.manifest.snapshot_id)
    active_bytes = snapshot_store.active_path.read_bytes()
    extra_release = replace(
        _manifest(),
        release_id="extra-2024-20260826",
        source_id="extra",
        dependency_group="extra-direct",
    )
    SQLiteEvidenceRepository(second.database_path, writable=True).add_release(
        extra_release
    )
    with closing(sqlite3.connect(second.database_path)) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("VACUUM")
    changed = replace(
        second.manifest,
        database_sha256=sha256_file(second.database_path),
    )
    _rewrite_candidate_manifest(second.candidate_path, changed)

    with pytest.raises(SnapshotPromotionError, match="release|identity"):
        snapshot_store.promote(second.manifest.snapshot_id)

    assert snapshot_store.active_path.read_bytes() == active_bytes
    assert snapshot_store.active_manifest() == first.manifest


@pytest.mark.parametrize(
    "manifest_change",
    [
        {"status": SnapshotStatus.ACTIVE},
        {"warnings": ("forged-warning",)},
    ],
)
def test_promotion_rejects_noncanonical_status_or_warnings(
    snapshot_store: SnapshotStore,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
    manifest_change: dict[str, object],
) -> None:
    candidate = builder.build(build_request)
    _rewrite_candidate_manifest(
        candidate.candidate_path,
        replace(candidate.manifest, **manifest_change),
    )

    with pytest.raises(SnapshotPromotionError, match="canonical"):
        snapshot_store.promote(candidate.manifest.snapshot_id)

    assert not snapshot_store.active_path.exists()


class RecordingFilesystem(SnapshotFilesystem):
    def __init__(self) -> None:
        self.events: list[tuple[str, Path]] = []

    def fsync_file(self, path: Path) -> None:
        self.events.append(("fsync_file", path))

    def fsync_directory(self, path: Path) -> None:
        self.events.append(("fsync_directory", path))

    def publish_directory(self, source: Path, destination: Path) -> None:
        self.events.append(("publish_directory", destination))
        super().publish_directory(source, destination)

    def replace_verified_file(
        self,
        source: Path,
        destination: Path,
        verify: Callable[[], None],
        *,
        stable_directory: Path,
        stable_files: tuple[Path, ...],
    ) -> None:
        self.events.append(("replace_file", destination))
        super().replace_verified_file(
            source,
            destination,
            verify,
            stable_directory=stable_directory,
            stable_files=stable_files,
        )


def test_promotion_flushes_files_and_directories_in_durable_order(
    evidence_root: Path,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
) -> None:
    candidate = builder.build(build_request)
    filesystem = RecordingFilesystem()
    store = SnapshotStore(
        evidence_root,
        clock=lambda: datetime(2026, 8, 26, 13, 0, tzinfo=UTC),
        filesystem=filesystem,
    )

    store.promote(candidate.manifest.snapshot_id)

    publication_index = next(
        index
        for index, event in enumerate(filesystem.events)
        if event[0] == "publish_directory"
    )
    flushed_snapshot_files = {
        path.name
        for event, path in filesystem.events[:publication_index]
        if event == "fsync_file"
    }
    assert flushed_snapshot_files == {
        "evidence.sqlite3",
        "snapshot.json",
        "validation.json",
    }
    snapshots_flush_index = filesystem.events.index(
        ("fsync_directory", evidence_root / "snapshots")
    )
    replace_index = filesystem.events.index(("replace_file", store.active_path))
    root_flush_index = len(filesystem.events) - 1
    assert publication_index < snapshots_flush_index < replace_index < root_flush_index
    assert filesystem.events[root_flush_index] == (
        "fsync_directory",
        evidence_root,
    )


def _create_directory_symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        if os.name == "nt" and error.winerror == 1314:
            pytest.skip(f"symlinks require Windows developer privileges: {error}")
        raise


@pytest.mark.parametrize("redirected_directory", ["root", "candidates", "snapshots"])
def test_promotion_rejects_symlinked_storage_components(
    tmp_path: Path,
    release_store: ReleaseStore,
    build_request: SnapshotBuildRequest,
    redirected_directory: str,
) -> None:
    real_root = tmp_path / "real-evidence"
    candidate = SnapshotBuilder(real_root, release_store, EmptyLoader()).build(build_request)
    store_root = real_root
    if redirected_directory == "root":
        store_root = tmp_path / "linked-evidence"
        _create_directory_symlink_or_skip(store_root, real_root)
    elif redirected_directory == "candidates":
        outside = tmp_path / "outside-candidates"
        (real_root / "candidates").rename(outside)
        _create_directory_symlink_or_skip(real_root / "candidates", outside)
    else:
        outside = tmp_path / "outside-snapshots"
        outside.mkdir()
        _create_directory_symlink_or_skip(real_root / "snapshots", outside)
    store = SnapshotStore(store_root)

    with pytest.raises(SnapshotPromotionError, match="unsafe|contain"):
        store.promote(candidate.manifest.snapshot_id)

    assert not (real_root / "active.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction contract")
def test_promotion_rejects_windows_junction_component(
    evidence_root: Path,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
    tmp_path: Path,
) -> None:
    candidate = builder.build(build_request)
    outside = tmp_path / "junction-target"
    (evidence_root / "candidates").rename(outside)
    junction = evidence_root / "candidates"
    creation = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert creation.returncode == 0, creation.stderr or creation.stdout
    try:
        with pytest.raises(SnapshotPromotionError, match="unsafe|contain"):
            SnapshotStore(evidence_root).promote(candidate.manifest.snapshot_id)
    finally:
        os.rmdir(junction)


class TamperingFilesystem(SnapshotFilesystem):
    def __init__(self, target_database: Path) -> None:
        self.target_database = target_database

    def replace_verified_file(
        self,
        source: Path,
        destination: Path,
        verify: Callable[[], None],
        *,
        stable_directory: Path,
        stable_files: tuple[Path, ...],
    ) -> None:
        self.target_database.chmod(stat.S_IREAD | stat.S_IWRITE)
        with self.target_database.open("ab") as database:
            database.write(b"tampered-before-pointer")
        super().replace_verified_file(
            source,
            destination,
            verify,
            stable_directory=stable_directory,
            stable_files=stable_files,
        )


def test_tamper_between_target_validation_and_pointer_replace_preserves_active(
    evidence_root: Path,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
) -> None:
    first = builder.build(build_request)
    second = builder.build(
        replace(
            build_request,
            versions=replace(build_request.versions, hazard_method="hazard-v2"),
        )
    )
    initial_store = SnapshotStore(evidence_root)
    initial_store.promote(first.manifest.snapshot_id)
    active_bytes = initial_store.active_path.read_bytes()
    target_database = (
        evidence_root / "snapshots" / second.manifest.snapshot_id / "evidence.sqlite3"
    )
    tampering_store = SnapshotStore(
        evidence_root,
        filesystem=TamperingFilesystem(target_database),
    )

    with pytest.raises(SnapshotPromotionError):
        tampering_store.promote(second.manifest.snapshot_id)

    assert initial_store.active_path.read_bytes() == active_bytes
    assert initial_store.active_manifest() == first.manifest


class SwappingAfterVerificationFilesystem(SnapshotFilesystem):
    def __init__(self, target: Path) -> None:
        self.target = target
        self.displaced = target.with_name(f"{target.name}.displaced")

    def replace_atomic_file(self, source: Path, destination: Path) -> None:
        try:
            self.target.rename(self.displaced)
        except PermissionError as error:
            raise SnapshotPathError("stable snapshot blocked target swap") from error
        super().replace_atomic_file(source, destination)


def test_target_swap_after_verification_preserves_last_known_good_pointer(
    evidence_root: Path,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
) -> None:
    first = builder.build(build_request)
    second = builder.build(
        replace(
            build_request,
            versions=replace(build_request.versions, forecast_method="forecast-v2"),
        )
    )
    initial_store = SnapshotStore(evidence_root)
    initial_store.promote(first.manifest.snapshot_id)
    active_bytes = initial_store.active_path.read_bytes()
    second_target = evidence_root / "snapshots" / second.manifest.snapshot_id
    store = SnapshotStore(
        evidence_root,
        filesystem=SwappingAfterVerificationFilesystem(second_target),
    )

    with pytest.raises(SnapshotPromotionError, match="stable|changed|failed|unsafe"):
        store.promote(second.manifest.snapshot_id)

    assert initial_store.active_path.read_bytes() == active_bytes
    assert initial_store.active_manifest() == first.manifest


class BlockingPublishFilesystem(SnapshotFilesystem):
    def __init__(self) -> None:
        self.publication_started = threading.Event()
        self.allow_publication = threading.Event()
        self._first_publication = True

    def publish_directory(self, source: Path, destination: Path) -> None:
        if self._first_publication:
            self._first_publication = False
            self.publication_started.set()
            assert self.allow_publication.wait(timeout=5)
        super().publish_directory(source, destination)


def test_concurrent_same_snapshot_promotion_reuses_identical_pointer_bytes(
    evidence_root: Path,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
) -> None:
    candidate = builder.build(build_request)
    filesystem = BlockingPublishFilesystem()
    clock_calls = 0
    clock_lock = threading.Lock()

    def clock() -> datetime:
        nonlocal clock_calls
        with clock_lock:
            clock_calls += 1
            return datetime(2026, 8, 26, 13, clock_calls, tzinfo=UTC)

    store = SnapshotStore(evidence_root, clock=clock, filesystem=filesystem)

    def promote_and_read_pointer() -> bytes:
        store.promote(candidate.manifest.snapshot_id)
        return store.active_path.read_bytes()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(promote_and_read_pointer)
        assert filesystem.publication_started.wait(timeout=5)
        second_started = threading.Event()

        def run_second() -> bytes:
            second_started.set()
            return promote_and_read_pointer()

        second = executor.submit(run_second)
        assert second_started.wait(timeout=5)
        filesystem.allow_publication.set()
        first_pointer = first.result(timeout=10)
        second_pointer = second.result(timeout=10)

    assert first_pointer == second_pointer == store.active_path.read_bytes()
    assert clock_calls == 1


def test_crashed_process_releases_promotion_lock_for_next_promoter(
    evidence_root: Path,
    builder: SnapshotBuilder,
    build_request: SnapshotBuildRequest,
) -> None:
    candidate = builder.build(build_request)
    script = """
import os
import sys
from pathlib import Path

from icor.infrastructure.snapshot_filesystem import SnapshotFilesystem

with SnapshotFilesystem().promotion_lock(Path(sys.argv[1])):
    print("locked", flush=True)
    os._exit(73)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(evidence_root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "locked"
    _, stderr = process.communicate(timeout=5)
    assert process.returncode == 73, stderr

    promoted = SnapshotStore(evidence_root).promote(candidate.manifest.snapshot_id)

    assert promoted == candidate.manifest
