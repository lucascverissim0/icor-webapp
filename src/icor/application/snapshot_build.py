"""Deterministic construction of isolated evidence snapshot candidates."""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from re import fullmatch
from tempfile import TemporaryDirectory
from typing import Protocol

from icor.domain.snapshots import SnapshotManifest, SnapshotStatus, SnapshotVersions
from icor.evidence.release_manifests import load_snapshot_manifest
from icor.evidence.serialization import canonical_json_bytes, sha256_file
from icor.evidence.validation import (
    ReleaseValidator,
    Severity,
    SnapshotValidator,
    ValidationReport,
)
from icor.infrastructure.release_store import ReleaseStore, StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

_IDENTIFIER_PATTERN = r"[a-z0-9][a-z0-9._-]{0,79}"


class SnapshotBuildError(RuntimeError):
    """A requested candidate cannot be built from verified local evidence."""


class EvidenceLoader(Protocol):
    """Load verified release contents into a writable evidence repository."""

    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None: ...


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
    ) -> None:
        self.root = Path(root)
        self.release_store = release_store
        self.loader = loader
        self.release_validator = release_validator or ReleaseValidator()
        self.snapshot_validator = snapshot_validator or SnapshotValidator()

    def build(self, request: SnapshotBuildRequest) -> SnapshotBuildResult:
        """Build and validate one deterministic candidate without promoting it."""
        if not isinstance(request, SnapshotBuildRequest):
            raise TypeError("request must be a SnapshotBuildRequest")
        releases = self._verified_releases(request.release_ids)
        snapshot_id = self._snapshot_id(request, releases)
        candidate_path = self.root / "candidates" / snapshot_id
        if candidate_path.exists():
            return self._load_existing(candidate_path, request, snapshot_id)

        candidates_root = candidate_path.parent
        candidates_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".build-", dir=candidates_root) as temporary:
            staging_path = Path(temporary)
            result = self._build_staging(staging_path, request, releases, snapshot_id)
            try:
                staging_path.rename(candidate_path)
            except FileExistsError:
                return self._load_existing(candidate_path, request, snapshot_id)
            return replace(
                result,
                candidate_path=candidate_path,
                database_path=candidate_path / "evidence.sqlite3",
                manifest_path=candidate_path / "snapshot.json",
                validation_path=candidate_path / "validation.json",
            )

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

    @staticmethod
    def _snapshot_id(
        request: SnapshotBuildRequest, releases: tuple[StoredRelease, ...]
    ) -> str:
        identity_payload = {
            "build_as_of": request.build_as_of,
            "deterministic_seed": request.deterministic_seed,
            "releases": tuple(
                {
                    "release_id": release.release_id,
                    "sha256": release.manifest.sha256,
                }
                for release in releases
            ),
            "versions": request.versions,
        }
        digest = sha256(canonical_json_bytes(identity_payload)).hexdigest()
        return f"snapshot-{digest[:20]}"

    def _build_staging(
        self,
        staging_path: Path,
        request: SnapshotBuildRequest,
        releases: tuple[StoredRelease, ...],
        snapshot_id: str,
    ) -> SnapshotBuildResult:
        scratch_root = staging_path / ".scratch"
        scratch_repository = SQLiteEvidenceRepository(
            scratch_root / "evidence.sqlite3", writable=True
        )
        for release in releases:
            scratch_repository.add_release(release.manifest)
        self.loader.load(releases, scratch_repository)

        database_path = staging_path / "evidence.sqlite3"
        repository = SQLiteEvidenceRepository(database_path, writable=True)
        self._replay_canonically(scratch_repository, repository)
        shutil.rmtree(scratch_root)
        self._finalize_database(database_path)

        for release in releases:
            try:
                self.release_store.verify(release.release_id)
            except (OSError, RuntimeError, ValueError) as error:
                raise SnapshotBuildError(
                    f"release changed during build: {release.release_id}"
                ) from error

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
        for mapping in source.list_mappings():
            target.add_mapping(mapping)
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
