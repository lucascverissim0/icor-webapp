"""Deterministic construction of isolated evidence snapshot candidates."""

from __future__ import annotations

import sqlite3
import stat
from contextlib import closing, suppress
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from re import fullmatch
from typing import Protocol, cast
from uuid import uuid4

from icor.domain.snapshots import SnapshotManifest, SnapshotStatus, SnapshotVersions
from icor.evidence.release_manifests import (
    load_snapshot_manifest,
    write_release_manifest,
)
from icor.evidence.serialization import canonical_json_bytes, sha256_file
from icor.evidence.validation import (
    ReleaseValidator,
    Severity,
    SnapshotValidator,
    ValidationReport,
)
from icor.infrastructure.release_store import ReleaseStore, StoredRelease
from icor.infrastructure.snapshot_filesystem import SnapshotFilesystem, SnapshotPathError
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

_IDENTIFIER_PATTERN = r"[a-z0-9][a-z0-9._-]{0,79}"
_SHA256_PATTERN = r"[0-9a-f]{64}"


class SnapshotBuildError(RuntimeError):
    """A requested candidate cannot be built from verified local evidence."""


class EvidenceLoader(Protocol):
    """Load verified release contents into a writable evidence repository."""

    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None: ...


class RepositoryTransformer(Protocol):
    """Wrap the scratch repository with one deterministic transformation boundary."""

    def __call__(
        self, repository: SQLiteEvidenceRepository, reviewed_at: datetime
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class SnapshotBuildRequest:
    """All inputs that determine one snapshot's identity and contents."""

    release_ids: tuple[str, ...]
    versions: SnapshotVersions
    deterministic_seed: int
    build_as_of: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.release_ids, tuple) or not self.release_ids:
            raise ValueError("snapshot release IDs are required")
        if any(
            type(release_id) is not str
            or fullmatch(_IDENTIFIER_PATTERN, release_id) is None
            for release_id in self.release_ids
        ):
            raise ValueError("snapshot release identifier is invalid")
        if tuple(sorted(self.release_ids)) != self.release_ids:
            raise ValueError("snapshot release IDs must be sorted")
        if len(set(self.release_ids)) != len(self.release_ids):
            raise ValueError("snapshot release IDs must be unique")
        if not isinstance(self.versions, SnapshotVersions):
            raise ValueError("snapshot versions are required")
        if type(self.deterministic_seed) is not int or self.deterministic_seed < 0:
            raise ValueError("deterministic seed must be a non-negative integer")
        if (
            not isinstance(self.build_as_of, datetime)
            or self.build_as_of.tzinfo is None
            or self.build_as_of.utcoffset() != timedelta(0)
        ):
            raise ValueError("build_as_of must be a UTC datetime")


@dataclass(frozen=True, slots=True)
class SnapshotBuildResult:
    """The paths, manifest, and quality decision for one isolated candidate."""

    candidate_path: Path
    database_path: Path
    manifest_path: Path
    validation_path: Path
    manifest: SnapshotManifest
    validation_report: ValidationReport


def snapshot_id_for(
    *,
    build_as_of: datetime,
    deterministic_seed: int,
    versions: SnapshotVersions,
    release_artifact_hashes: tuple[tuple[str, str], ...],
) -> str:
    """Derive the canonical identity shared by candidate build and promotion."""
    release_ids = tuple(release_id for release_id, _ in release_artifact_hashes)
    if not release_ids or release_ids != tuple(sorted(release_ids)):
        raise ValueError("snapshot identity release IDs must be nonempty and sorted")
    if len(set(release_ids)) != len(release_ids):
        raise ValueError("snapshot identity release IDs must be unique")
    if any(
        fullmatch(_IDENTIFIER_PATTERN, release_id) is None
        or fullmatch(_SHA256_PATTERN, artifact_hash) is None
        for release_id, artifact_hash in release_artifact_hashes
    ):
        raise ValueError("snapshot identity release artifact is invalid")
    request = SnapshotBuildRequest(
        release_ids=release_ids,
        versions=versions,
        deterministic_seed=deterministic_seed,
        build_as_of=build_as_of,
    )
    identity_payload = {
        "build_as_of": request.build_as_of,
        "deterministic_seed": request.deterministic_seed,
        "releases": tuple(
            {"release_id": release_id, "sha256": artifact_hash}
            for release_id, artifact_hash in release_artifact_hashes
        ),
        "versions": request.versions,
    }
    digest = sha256(canonical_json_bytes(identity_payload)).hexdigest()
    return f"snapshot-{digest[:20]}"


class SnapshotBuilder:
    """Build content-derived candidate snapshots beneath an explicit local root."""

    def __init__(
        self,
        root: Path,
        release_store: ReleaseStore,
        loader: EvidenceLoader,
        *,
        release_validator: ReleaseValidator | None = None,
        snapshot_validator: SnapshotValidator | None = None,
        filesystem: SnapshotFilesystem | None = None,
        repository_transformer: RepositoryTransformer | None = None,
    ) -> None:
        self.root = Path(root)
        self.release_store = release_store
        self.loader = loader
        self.release_validator = release_validator or ReleaseValidator()
        self.snapshot_validator = snapshot_validator or SnapshotValidator()
        self.filesystem = filesystem or SnapshotFilesystem()
        self.repository_transformer = repository_transformer

    def build(self, request: SnapshotBuildRequest) -> SnapshotBuildResult:
        """Build and validate one deterministic candidate without promoting it."""
        if not isinstance(request, SnapshotBuildRequest):
            raise TypeError("request must be a SnapshotBuildRequest")
        try:
            self.root = self.filesystem.prepare_root(self.root)
            candidates_root = self.filesystem.prepare_directory(
                self.root / "candidates", self.root
            )
            releases = self._verified_releases(request.release_ids)
            snapshot_id = snapshot_id_for(
                build_as_of=request.build_as_of,
                deterministic_seed=request.deterministic_seed,
                versions=request.versions,
                release_artifact_hashes=tuple(
                    (release.release_id, release.manifest.sha256) for release in releases
                ),
            )
            candidate_path = candidates_root / snapshot_id
            if candidate_path.exists():
                self.filesystem.require_directory(candidate_path, self.root)
                return self._load_existing(candidate_path, request, snapshot_id)

            staging_path = self.filesystem.prepare_directory(
                candidates_root / f".build-{uuid4().hex}", self.root
            )
        except SnapshotPathError as error:
            raise SnapshotBuildError("snapshot build path is unsafe or not contained") from error

        try:
            result = self._build_staging(staging_path, request, releases, snapshot_id)
            for path in (
                result.database_path,
                result.manifest_path,
                result.validation_path,
            ):
                self.filesystem.fsync_file(path)
            self.filesystem.fsync_directory(staging_path)
            try:
                self.filesystem.publish_directory(staging_path, candidate_path)
            except FileExistsError:
                self.filesystem.require_directory(candidate_path, self.root)
                return self._load_existing(candidate_path, request, snapshot_id)
            self.filesystem.fsync_directory(candidates_root)
            return replace(
                result,
                candidate_path=candidate_path,
                database_path=candidate_path / "evidence.sqlite3",
                manifest_path=candidate_path / "snapshot.json",
                validation_path=candidate_path / "validation.json",
            )
        except SnapshotPathError as error:
            raise SnapshotBuildError("snapshot build path is unsafe or not contained") from error
        finally:
            if staging_path.exists():
                with suppress(SnapshotPathError):
                    self.filesystem.cleanup_directory(staging_path, self.root)

    def _verified_releases(self, release_ids: tuple[str, ...]) -> tuple[StoredRelease, ...]:
        releases: list[StoredRelease] = []
        for release_id in release_ids:
            try:
                release = self.release_store.verify(release_id)
            except (OSError, RuntimeError, ValueError) as error:
                raise SnapshotBuildError(f"release cannot be verified: {release_id}") from error
            if not self.release_validator.validate(release).can_promote:
                raise SnapshotBuildError(f"release failed validation: {release_id}")
            releases.append(release)
        return tuple(releases)

    def _build_staging(
        self,
        staging_path: Path,
        request: SnapshotBuildRequest,
        releases: tuple[StoredRelease, ...],
        snapshot_id: str,
    ) -> SnapshotBuildResult:
        scratch_root = self.filesystem.prepare_directory(
            staging_path / ".scratch", self.root
        )
        loader_releases = self._seal_loader_releases(releases, scratch_root)
        scratch_repository = SQLiteEvidenceRepository(
            scratch_root / "evidence.sqlite3", writable=True
        )
        for release in releases:
            scratch_repository.add_release(release.manifest)
        loader_repository = (
            scratch_repository
            if self.repository_transformer is None
            else self.repository_transformer(scratch_repository, request.build_as_of)
        )
        self.loader.load(
            loader_releases,
            cast(SQLiteEvidenceRepository, loader_repository),
        )

        database_path = staging_path / "evidence.sqlite3"
        repository = SQLiteEvidenceRepository(database_path, writable=True)
        self._replay_canonically(scratch_repository, repository)
        self.filesystem.cleanup_directory(scratch_root, self.root)
        self._finalize_database(database_path)

        for release in releases:
            try:
                refreshed = self.release_store.verify(release.release_id)
            except (OSError, RuntimeError, ValueError) as error:
                raise SnapshotBuildError(
                    f"release changed during build: {release.release_id}"
                ) from error
            if refreshed != release:
                raise SnapshotBuildError(
                    f"release changed during build: {release.release_id}"
                )

        readonly_repository = SQLiteEvidenceRepository(database_path)
        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            status=SnapshotStatus.CANDIDATE,
            built_at=request.build_as_of,
            deterministic_seed=request.deterministic_seed,
            release_ids=request.release_ids,
            versions=request.versions,
            database_sha256=sha256_file(database_path),
            observation_count=len(readonly_repository.list_observations()),
            published_value_count=len(readonly_repository.list_published_values()),
            warnings=(),
        )
        report = self.snapshot_validator.validate(readonly_repository, manifest)
        warnings = tuple(
            finding.code for finding in report.findings if finding.severity is Severity.WARNING
        )
        if warnings:
            manifest = replace(manifest, warnings=warnings)

        manifest_path = staging_path / "snapshot.json"
        validation_path = staging_path / "validation.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        validation_path.write_bytes(canonical_json_bytes(report))
        return SnapshotBuildResult(
            candidate_path=staging_path,
            database_path=database_path,
            manifest_path=manifest_path,
            validation_path=validation_path,
            manifest=manifest,
            validation_report=report,
        )

    def _seal_loader_releases(
        self,
        releases: tuple[StoredRelease, ...],
        scratch_root: Path,
    ) -> tuple[StoredRelease, ...]:
        sealed_root = self.filesystem.prepare_directory(
            scratch_root / "releases", self.root
        )
        sealed: list[StoredRelease] = []
        for release in releases:
            release_root = self.filesystem.prepare_directory(
                sealed_root / release.release_id, self.root
            )
            artifact_path = release_root / release.manifest.artifact_path
            try:
                self.filesystem.copy_file(
                    release.artifact_path,
                    artifact_path,
                    source_root=self.release_store.root,
                    destination_root=self.root,
                )
                self.filesystem.fsync_file(artifact_path)
            except (OSError, SnapshotPathError) as error:
                raise SnapshotBuildError(
                    f"release cannot be sealed for loading: {release.release_id}"
                ) from error
            if (
                sha256_file(artifact_path) != release.manifest.sha256
                or artifact_path.stat().st_size != release.manifest.artifact_bytes
            ):
                raise SnapshotBuildError(
                    f"release changed before loading: {release.release_id}"
                )

            manifest_path = release_root / "manifest.json"
            write_release_manifest(manifest_path, release.manifest)
            artifact_path.chmod(stat.S_IREAD)
            manifest_path.chmod(stat.S_IREAD)
            release_root.chmod(stat.S_IREAD | stat.S_IEXEC)
            sealed.append(
                replace(
                    release,
                    artifact_path=artifact_path,
                    manifest_path=manifest_path,
                )
            )
        return tuple(sealed)

    @staticmethod
    def _replay_canonically(
        source: SQLiteEvidenceRepository, target: SQLiteEvidenceRepository
    ) -> None:
        for release in source.list_releases():
            target.add_release(release)
        for vehicle in source.list_vehicles():
            target.add_vehicle(vehicle)
        observations = source.list_observations()
        if observations:
            target.add_observations(observations)
        mappings = source.list_mappings()
        if mappings:
            target.add_identity_attributions((), (), mappings)
        published_values = source.list_published_values()
        if published_values:
            target.add_published_values(published_values)

    @staticmethod
    def _finalize_database(path: Path) -> None:
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("VACUUM")

    def _load_existing(
        self,
        candidate_path: Path,
        request: SnapshotBuildRequest,
        snapshot_id: str,
    ) -> SnapshotBuildResult:
        database_path = candidate_path / "evidence.sqlite3"
        manifest_path = candidate_path / "snapshot.json"
        validation_path = candidate_path / "validation.json"
        try:
            manifest = load_snapshot_manifest(manifest_path)
            if (
                manifest.snapshot_id != snapshot_id
                or manifest.built_at != request.build_as_of
                or manifest.deterministic_seed != request.deterministic_seed
                or manifest.release_ids != request.release_ids
                or manifest.versions != request.versions
            ):
                raise SnapshotBuildError("existing candidate identity does not match request")
            repository = SQLiteEvidenceRepository(database_path)
            report = self.snapshot_validator.validate(repository, manifest)
            if validation_path.read_bytes() != canonical_json_bytes(report):
                raise SnapshotBuildError("existing candidate validation report does not match")
        except SnapshotBuildError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise SnapshotBuildError("existing candidate cannot be verified") from error
        return SnapshotBuildResult(
            candidate_path=candidate_path,
            database_path=database_path,
            manifest_path=manifest_path,
            validation_path=validation_path,
            manifest=manifest,
            validation_report=report,
        )
