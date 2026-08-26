import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from icor.domain.evidence import Measure, PublicationStatus, ReleaseManifest
from icor.infrastructure.release_store import (
    ReleaseAlreadyExistsError,
    ReleaseIntegrityError,
    ReleaseStore,
)
from icor.infrastructure.snapshot_filesystem import SnapshotFilesystem, SnapshotPathError


def release_manifest_for(
    artifact_bytes: bytes,
    release_id: str = "eea-2024-20260826",
    source_id: str = "eea",
) -> ReleaseManifest:
    return ReleaseManifest(
        release_id=release_id,
        source_id=source_id,
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
        artifact_bytes=len(artifact_bytes),
        sha256=sha256_file_bytes(artifact_bytes),
        parser_name="eea_csv",
        parser_version="v1",
        expected_schema="eea-2024-v1",
        raw_record_count=1,
        accepted_record_count=1,
        rejected_record_count=0,
        quarantined_record_count=0,
    )


def sha256_file_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def source_artifact(
    tmp_path: Path,
    contents: bytes = b"make,model,count\nA,B,1\n",
    name: str = "incoming.csv",
) -> Path:
    artifact = tmp_path / name
    artifact.write_bytes(contents)
    return artifact


def create_symlink_or_skip(link: Path, target: Path, *, is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=is_directory)
    except OSError as error:
        if os.name == "nt" and error.winerror == 1314:
            pytest.skip(f"symlinks require Windows developer privileges: {error}")
        raise


def create_directory_alias(link: Path, target: Path) -> None:
    if os.name != "nt":
        link.symlink_to(target, target_is_directory=True)
        return
    creation = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert creation.returncode == 0, creation.stderr or creation.stdout


def remove_directory_alias(link: Path) -> None:
    if os.name == "nt":
        os.rmdir(link)
    else:
        link.unlink()


@pytest.fixture
def release_manifest() -> ReleaseManifest:
    return release_manifest_for(b"make,model,count\nA,B,1\n")


@pytest.fixture
def store(tmp_path: Path) -> ReleaseStore:
    return ReleaseStore(tmp_path / "raw")


def test_stage_copies_artifact_and_manifest_under_release_id(
    tmp_path: Path, release_manifest: ReleaseManifest
) -> None:
    store = ReleaseStore(tmp_path / "raw")

    stored = store.stage(source_artifact(tmp_path), release_manifest)

    assert stored.artifact_path == (
        tmp_path / "raw" / "eea" / "eea-2024-20260826" / "artifact.csv"
    )
    assert stored.artifact_path.read_bytes() == b"make,model,count\nA,B,1\n"
    assert stored.manifest_path == stored.artifact_path.parent / "manifest.json"
    assert store.verify(release_manifest.release_id) == stored


def test_alias_root_stage_get_and_verify_preserve_the_pinned_operation_path(
    tmp_path: Path, release_manifest: ReleaseManifest
) -> None:
    lexical_root = tmp_path / "evidence"
    pinned_root = tmp_path / "pinned-evidence"
    outside = tmp_path / "outside"
    lexical_root.mkdir()
    outside.mkdir()
    artifact = source_artifact(tmp_path)
    filesystem = SnapshotFilesystem()

    def exercise(operation_root: Path) -> None:
        assert operation_root.resolve(strict=True) != operation_root
        store = ReleaseStore(operation_root / "releases", filesystem=filesystem)

        staged = store.stage(artifact, release_manifest)
        fetched = store.get(release_manifest.release_id)
        verified = store.verify(release_manifest.release_id)
        alias_release = (
            operation_root
            / "releases"
            / release_manifest.source_id
            / release_manifest.release_id
        )

        assert staged == fetched == verified
        for stored in (staged, fetched, verified):
            assert stored.artifact_path == alias_release / "artifact.csv"
            assert stored.manifest_path == alias_release / "manifest.json"
        assert (
            pinned_root
            / "releases"
            / release_manifest.source_id
            / release_manifest.release_id
            / "artifact.csv"
        ).read_bytes() == b"make,model,count\nA,B,1\n"
        assert list(outside.iterdir()) == []

    if os.name != "nt":
        pin_entered = False
        operation_exercised = False
        alias_created = False
        try:
            with filesystem.pin_root(lexical_root, create=False) as operation_root:
                pin_entered = True
                lexical_root.rename(pinned_root)
                create_directory_alias(lexical_root, outside)
                alias_created = True
                exercise(operation_root)
                operation_exercised = True
        except SnapshotPathError as error:
            if not pin_entered:
                assert str(error) == "platform cannot anchor the evidence root safely"
                assert lexical_root.is_dir()
                assert list(lexical_root.iterdir()) == []
                assert not pinned_root.exists()
                assert list(outside.iterdir()) == []
                return
            assert operation_exercised
            assert str(error) == "stable snapshot path cannot be opened"
        else:
            pytest.fail("substituted evidence root was not rejected")
        finally:
            if alias_created:
                remove_directory_alias(lexical_root)
        return

    lexical_root.rename(pinned_root)
    operation_root = tmp_path / "descriptor-alias"
    create_directory_alias(lexical_root, outside)
    create_directory_alias(operation_root, pinned_root)
    filesystem._pinned_operation_root = operation_root
    try:
        exercise(operation_root)
    finally:
        filesystem._pinned_operation_root = None
        remove_directory_alias(operation_root)
        remove_directory_alias(lexical_root)


def test_existing_release_cannot_be_replaced(store: ReleaseStore, tmp_path: Path) -> None:
    first = release_manifest_for(b"one")
    store.stage(source_artifact(tmp_path, b"one"), first)

    with pytest.raises(ReleaseAlreadyExistsError):
        store.stage(source_artifact(tmp_path, b"two"), release_manifest_for(b"two"))

    assert store.verify(first.release_id).artifact_path.read_bytes() == b"one"


def test_stage_preserves_destination_created_during_publish(
    store: ReleaseStore,
    release_manifest: ReleaseManifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = store.root / release_manifest.source_id / release_manifest.release_id
    copy_with_fsync = store._copy_with_fsync
    path_rename = Path.rename

    def create_colliding_destination(source_path: Path, staged_path: Path) -> None:
        copy_with_fsync(source_path, staged_path)
        destination.mkdir(parents=True)

    def replace_empty_destination(staging: Path, target: Path) -> Path:
        if staging.parent.name == ".staging" and target == destination and target.exists():
            shutil.rmtree(target)
        return path_rename(staging, target)

    monkeypatch.setattr(store, "_copy_with_fsync", create_colliding_destination)
    monkeypatch.setattr(Path, "rename", replace_empty_destination)

    with pytest.raises(ReleaseIntegrityError, match="incomplete"):
        store.stage(source_artifact(tmp_path), release_manifest)

    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_stage_rejects_source_artifact_that_does_not_match_manifest(
    store: ReleaseStore, release_manifest: ReleaseManifest, tmp_path: Path
) -> None:
    mismatched_manifest = replace(release_manifest, sha256="0" * 64)

    with pytest.raises(ReleaseIntegrityError, match="SHA-256"):
        store.stage(source_artifact(tmp_path), mismatched_manifest)

    assert not store.root.exists()


def test_verify_rejects_post_stage_artifact_corruption(
    store: ReleaseStore, release_manifest: ReleaseManifest, tmp_path: Path
) -> None:
    stored = store.stage(source_artifact(tmp_path), release_manifest)
    stored.artifact_path.write_bytes(b"corrupted")

    with pytest.raises(ReleaseIntegrityError):
        store.verify(release_manifest.release_id)


def test_get_rejects_a_symlinked_source_directory(
    store: ReleaseStore, release_manifest: ReleaseManifest, tmp_path: Path
) -> None:
    outside_store = ReleaseStore(tmp_path / "outside")
    outside_store.stage(source_artifact(tmp_path), release_manifest)
    store.root.mkdir()
    source_link = store.root / release_manifest.source_id
    create_symlink_or_skip(
        source_link,
        outside_store.root / release_manifest.source_id,
        is_directory=True,
    )

    with pytest.raises(ReleaseIntegrityError, match="symlink"):
        store.get(release_manifest.release_id)


def test_get_rejects_a_symlinked_release_directory(
    store: ReleaseStore, release_manifest: ReleaseManifest, tmp_path: Path
) -> None:
    outside_store = ReleaseStore(tmp_path / "outside")
    outside_store.stage(source_artifact(tmp_path), release_manifest)
    release_link = store.root / release_manifest.source_id / release_manifest.release_id
    release_link.parent.mkdir(parents=True)
    create_symlink_or_skip(
        release_link,
        outside_store.root / release_manifest.source_id / release_manifest.release_id,
        is_directory=True,
    )

    with pytest.raises(ReleaseIntegrityError, match="symlink"):
        store.get(release_manifest.release_id)


def test_get_rejects_a_symlinked_artifact(
    store: ReleaseStore, release_manifest: ReleaseManifest, tmp_path: Path
) -> None:
    stored = store.stage(source_artifact(tmp_path), release_manifest)
    outside_artifact = tmp_path / "outside.csv"
    outside_artifact.write_bytes(stored.artifact_path.read_bytes())
    stored.artifact_path.unlink()
    create_symlink_or_skip(stored.artifact_path, outside_artifact)

    with pytest.raises(ReleaseIntegrityError, match="symlink"):
        store.get(release_manifest.release_id)


def test_get_rejects_incomplete_release_directory(store: ReleaseStore) -> None:
    incomplete = store.root / "eea" / "eea-2024-20260826"
    incomplete.mkdir(parents=True)
    (incomplete / "artifact.csv").write_bytes(b"incomplete")

    with pytest.raises(ReleaseIntegrityError, match="incomplete"):
        store.get("eea-2024-20260826")


@pytest.mark.parametrize("release_id", ["../escape", "C:escape", "/absolute"])
def test_get_rejects_traversing_or_absolute_release_ids(
    store: ReleaseStore, release_id: str
) -> None:
    with pytest.raises(ValueError, match="release identifier"):
        store.get(release_id)


def test_stage_is_idempotent_for_identical_release_content(
    store: ReleaseStore, release_manifest: ReleaseManifest, tmp_path: Path
) -> None:
    first = store.stage(source_artifact(tmp_path), release_manifest)
    second_source = tmp_path / "another-incoming.csv"
    second_source.write_bytes(b"make,model,count\nA,B,1\n")

    second = store.stage(second_source, release_manifest)

    assert second == first


def test_stage_rejects_a_release_id_already_owned_by_another_source(
    store: ReleaseStore, tmp_path: Path
) -> None:
    first = release_manifest_for(b"one", source_id="eea")
    second = release_manifest_for(b"two", source_id="kba")
    store.stage(source_artifact(tmp_path, b"one"), first)

    with pytest.raises(ReleaseAlreadyExistsError):
        store.stage(source_artifact(tmp_path, b"two"), second)

    assert store.verify(first.release_id).source_id == "eea"


def test_stage_rejects_a_release_id_locked_by_another_stager(
    store: ReleaseStore, release_manifest: ReleaseManifest, tmp_path: Path
) -> None:
    lock = store.root / ".locks" / release_manifest.release_id
    lock.mkdir(parents=True)

    with pytest.raises(ReleaseAlreadyExistsError, match="in progress"):
        store.stage(source_artifact(tmp_path), release_manifest)

    assert not (store.root / release_manifest.source_id).exists()


def test_concurrent_distinct_releases_create_shared_directories_idempotently(
    store: ReleaseStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.root.mkdir()
    locks_root = store.root / ".locks"
    absent_locks_barrier = threading.Barrier(2)
    path_exists = Path.exists

    def synchronize_absent_locks(path: Path) -> bool:
        exists = path_exists(path)
        if path == locks_root and not exists:
            absent_locks_barrier.wait(timeout=5)
        return exists

    monkeypatch.setattr(Path, "exists", synchronize_absent_locks)
    manifests = (
        release_manifest_for(b"first", release_id="eea-2024-20260826-a"),
        release_manifest_for(b"second", release_id="eea-2024-20260826-b"),
    )
    artifacts = (
        source_artifact(tmp_path, b"first", "first.csv"),
        source_artifact(tmp_path, b"second", "second.csv"),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(store.stage, artifact, manifest)
            for artifact, manifest in zip(artifacts, manifests, strict=True)
        ]
        stored = [future.result(timeout=10) for future in futures]

    assert {release.release_id for release in stored} == {
        "eea-2024-20260826-a",
        "eea-2024-20260826-b",
    }


def test_stage_rejects_manifest_path_that_disagrees_with_source_extension(
    store: ReleaseStore, tmp_path: Path
) -> None:
    artifact = b'{"release":"2024"}\n'
    manifest = release_manifest_for(artifact)

    with pytest.raises(ReleaseIntegrityError, match="artifact path"):
        store.stage(source_artifact(tmp_path, artifact, "incoming.json"), manifest)

    assert not store.root.exists()


def test_list_releases_is_stably_sorted_by_release_id(store: ReleaseStore, tmp_path: Path) -> None:
    later = release_manifest_for(b"later", release_id="eea-2025-20260826")
    earlier = release_manifest_for(b"earlier", release_id="eea-2023-20260826")
    store.stage(source_artifact(tmp_path, b"later"), later)
    store.stage(source_artifact(tmp_path, b"earlier"), earlier)

    assert [stored.release_id for stored in store.list_releases()] == [
        "eea-2023-20260826",
        "eea-2025-20260826",
    ]
