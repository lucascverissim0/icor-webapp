"""Atomic promotion and lookup of last-known-good evidence snapshots."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from re import fullmatch
from tempfile import NamedTemporaryFile
from uuid import uuid4

from icor.application.snapshot_build import snapshot_id_for
from icor.domain.snapshots import SnapshotManifest, SnapshotStatus
from icor.evidence.release_manifests import load_snapshot_manifest
from icor.evidence.serialization import canonical_json_bytes, sha256_file
from icor.evidence.validation import Severity, SnapshotValidator, ValidationReport
from icor.infrastructure.snapshot_filesystem import SnapshotFilesystem, SnapshotPathError
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

_IDENTIFIER_PATTERN = r"[a-z0-9][a-z0-9._-]{0,79}"
_SHA256_PATTERN = r"[0-9a-f]{64}"
_POINTER_FIELDS = frozenset({"snapshot_id", "manifest_sha256", "promoted_at"})
_CANDIDATE_FILES = frozenset({"evidence.sqlite3", "snapshot.json", "validation.json"})


class SnapshotPromotionError(RuntimeError):
    """A candidate could not safely replace the active pointer."""


class SnapshotUnavailableError(RuntimeError):
    """No complete, verified active snapshot is available."""


class SnapshotStore:
    """Promote immutable candidates while retaining every prior good snapshot."""

    def __init__(
        self,
        root: Path,
        *,
        clock: Callable[[], datetime] | None = None,
        validator: SnapshotValidator | None = None,
        filesystem: SnapshotFilesystem | None = None,
    ) -> None:
        self.root = Path(root)
        self.clock = clock or (lambda: datetime.now(UTC))
        self.validator = validator or SnapshotValidator()
        self.filesystem = filesystem or SnapshotFilesystem()

    @property
    def active_path(self) -> Path:
        return self.root / "active.json"

    def candidate_path(self, snapshot_id: str) -> Path:
        """Return the isolated candidate directory for a validated identifier."""
        self._require_identifier(snapshot_id)
        return self.root / "candidates" / snapshot_id

    def promote(self, snapshot_id: str) -> SnapshotManifest:
        """Verify, publish, re-verify, then atomically activate one candidate."""
        self._require_identifier(snapshot_id)
        try:
            self.root = self.filesystem.prepare_root(self.root)
            with self.filesystem.promotion_lock(self.root):
                if self.active_path.exists():
                    try:
                        active = self.active_manifest()
                    except SnapshotUnavailableError:
                        pass
                    else:
                        if active.snapshot_id == snapshot_id:
                            return active

                candidate = self.candidate_path(snapshot_id)
                self._verify_snapshot_directory(candidate, snapshot_id)
                target = self._publish_candidate(candidate, snapshot_id)
                manifest, manifest_digest, _ = self._verify_snapshot_directory(
                    target, snapshot_id
                )
                self._replace_active_pointer(
                    snapshot_id,
                    manifest_digest,
                    target=target,
                )
                return manifest
        except SnapshotPromotionError:
            raise
        except SnapshotPathError as error:
            raise SnapshotPromotionError(
                "candidate promotion path is unsafe or not contained"
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise SnapshotPromotionError("candidate promotion failed") from error

    def active_manifest(self) -> SnapshotManifest:
        """Return the manifest selected by a complete and verified active pointer."""
        try:
            manifest, _ = self._resolve_active_snapshot()
            return manifest
        except SnapshotUnavailableError:
            raise
        except SnapshotPathError as error:
            raise SnapshotUnavailableError(
                "active snapshot path is unsafe or not contained"
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise SnapshotUnavailableError("active snapshot cannot be verified") from error

    def open_active_repository(self) -> SQLiteEvidenceRepository:
        """Open the verified active ledger through its read-only repository boundary."""
        _, repository = self.open_active_snapshot()
        return repository

    def open_active_snapshot(
        self,
    ) -> tuple[SnapshotManifest, SQLiteEvidenceRepository]:
        """Resolve one active pointer to its matching manifest and repository."""
        try:
            manifest, target = self._resolve_active_snapshot()
            path = self.filesystem.require_file(target / "evidence.sqlite3", self.root)
            return manifest, SQLiteEvidenceRepository(path)
        except SnapshotUnavailableError:
            raise
        except SnapshotPathError as error:
            raise SnapshotUnavailableError(
                "active snapshot database path is unsafe or not contained"
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise SnapshotUnavailableError("active snapshot database is unavailable") from error

    def _resolve_active_snapshot(self) -> tuple[SnapshotManifest, Path]:
        self.root = self.filesystem.require_root(self.root)
        pointer = self._load_active_pointer()
        snapshot_id = pointer["snapshot_id"]
        target = self.root / "snapshots" / snapshot_id
        manifest, manifest_digest, _ = self._verify_snapshot_directory(target, snapshot_id)
        if manifest_digest != pointer["manifest_sha256"]:
            raise SnapshotUnavailableError("active snapshot manifest does not match pointer")
        return manifest, target

    def _publish_candidate(self, candidate: Path, snapshot_id: str) -> Path:
        snapshots_root = self.filesystem.prepare_directory(
            self.root / "snapshots", self.root
        )
        target = snapshots_root / snapshot_id
        if os.path.lexists(target):
            self.filesystem.require_directory(target, self.root)
            self._verify_snapshot_directory(target, snapshot_id)
            self.filesystem.make_immutable(target, self.root)
            return target

        staging = self.filesystem.prepare_directory(
            snapshots_root / f".promoting-{snapshot_id}-{uuid4().hex}", self.root
        )
        try:
            for filename in sorted(_CANDIDATE_FILES):
                source = candidate / filename
                destination = staging / filename
                self.filesystem.copy_file(
                    source,
                    destination,
                    source_root=self.root,
                    destination_root=self.root,
                )
                self.filesystem.fsync_file(destination)
            self.filesystem.fsync_directory(staging)
            self._verify_snapshot_directory(staging, snapshot_id)
            self.filesystem.make_immutable(staging, self.root)
            try:
                self.filesystem.publish_directory(staging, target)
            except FileExistsError:
                self.filesystem.require_directory(target, self.root)
                self._verify_snapshot_directory(target, snapshot_id)
                self.filesystem.make_immutable(target, self.root)
            self.filesystem.fsync_directory(snapshots_root)
            return target
        finally:
            if os.path.lexists(staging):
                self.filesystem.cleanup_directory(staging, self.root)

    def _verify_snapshot_directory(
        self, directory: Path, snapshot_id: str
    ) -> tuple[SnapshotManifest, str, ValidationReport]:
        try:
            directory = self.filesystem.require_directory(directory, self.root)
        except SnapshotPathError as error:
            raise SnapshotPromotionError(
                "candidate snapshot directory is unsafe or not contained"
            ) from error
        try:
            entries = tuple(directory.iterdir())
        except OSError as error:
            raise SnapshotPromotionError("candidate snapshot directory cannot be read") from error
        if {entry.name for entry in entries} != _CANDIDATE_FILES:
            raise SnapshotPromotionError("candidate snapshot files are incomplete or unsafe")
        try:
            for entry in entries:
                self.filesystem.require_file(entry, self.root)
        except SnapshotPathError as error:
            raise SnapshotPromotionError(
                "candidate snapshot files are incomplete or unsafe"
            ) from error

        database_path = directory / "evidence.sqlite3"
        manifest_path = directory / "snapshot.json"
        validation_path = directory / "validation.json"
        try:
            manifest = load_snapshot_manifest(manifest_path)
            if manifest.snapshot_id != snapshot_id:
                raise SnapshotPromotionError("candidate snapshot identity does not match")
            repository = SQLiteEvidenceRepository(database_path)
            releases_by_id = {
                release.release_id: release for release in repository.list_releases()
            }
            if tuple(sorted(releases_by_id)) != manifest.release_ids:
                raise SnapshotPromotionError(
                    "candidate snapshot release identity set does not match"
                )
            try:
                release_artifact_hashes = tuple(
                    (release_id, releases_by_id[release_id].sha256)
                    for release_id in manifest.release_ids
                )
            except KeyError as error:
                raise SnapshotPromotionError(
                    "candidate snapshot identity release is unavailable"
                ) from error
            expected_snapshot_id = snapshot_id_for(
                build_as_of=manifest.built_at,
                deterministic_seed=manifest.deterministic_seed,
                versions=manifest.versions,
                release_artifact_hashes=release_artifact_hashes,
            )
            accepted_snapshot_ids = {expected_snapshot_id}
            if (
                manifest.versions.generation_registry == "generation-registry-v0"
                and manifest.versions.generation_resolver == "generation-resolver-v0"
            ):
                accepted_snapshot_ids.add(
                    snapshot_id_for(
                        build_as_of=manifest.built_at,
                        deterministic_seed=manifest.deterministic_seed,
                        versions=manifest.versions,
                        release_artifact_hashes=release_artifact_hashes,
                        legacy_generation_versions=True,
                    )
                )
            if manifest.snapshot_id not in accepted_snapshot_ids:
                raise SnapshotPromotionError("candidate snapshot identity does not match")
            report = self.validator.validate(repository, manifest)
            if not report.can_promote:
                raise SnapshotPromotionError("candidate snapshot failed validation")
            expected_warnings = tuple(
                finding.code
                for finding in report.findings
                if finding.severity is Severity.WARNING
            )
            if (
                manifest.status is not SnapshotStatus.CANDIDATE
                or manifest.warnings != expected_warnings
            ):
                raise SnapshotPromotionError("candidate snapshot manifest is not canonical")
            if validation_path.read_bytes() != canonical_json_bytes(report):
                raise SnapshotPromotionError("candidate validation report does not match")
            manifest_digest = sha256_file(manifest_path)
        except SnapshotPromotionError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise SnapshotPromotionError("candidate snapshot cannot be verified") from error
        return manifest, manifest_digest, report

    def _replace_active_pointer(
        self,
        snapshot_id: str,
        manifest_digest: str,
        *,
        target: Path,
    ) -> None:
        promoted_at = self.clock()
        if (
            not isinstance(promoted_at, datetime)
            or promoted_at.tzinfo is None
            or promoted_at.utcoffset() != timedelta(0)
        ):
            raise SnapshotPromotionError("promotion clock must return a UTC datetime")
        pointer = {
            "manifest_sha256": manifest_digest,
            "promoted_at": promoted_at.isoformat(),
            "snapshot_id": snapshot_id,
        }
        self.filesystem.require_root(self.root)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                "wb", delete=False, dir=self.root, prefix=".active.json."
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(canonical_json_bytes(pointer))
                temporary.flush()
                os.fsync(temporary.fileno())
            self.filesystem.require_file(temporary_path, self.root)
            if os.path.lexists(self.active_path):
                self.filesystem.require_file(self.active_path, self.root)

            def verify_target() -> None:
                _, refreshed_digest, _ = self._verify_snapshot_directory(
                    target, snapshot_id
                )
                if refreshed_digest != manifest_digest:
                    raise SnapshotPromotionError(
                        "published snapshot changed before pointer replacement"
                    )

            self.filesystem.replace_verified_file(
                temporary_path,
                self.active_path,
                verify_target,
                stable_directory=target,
                stable_files=tuple(target / name for name in sorted(_CANDIDATE_FILES)),
            )
            self.filesystem.fsync_directory(self.root)
        except OSError as error:
            raise SnapshotPromotionError("active snapshot pointer could not be replaced") from error
        finally:
            if temporary_path is not None and os.path.lexists(temporary_path):
                self.filesystem.require_file(temporary_path, self.root).unlink()

    def _load_active_pointer(self) -> dict[str, str]:
        if not os.path.lexists(self.active_path):
            raise SnapshotUnavailableError("no active snapshot is available")
        try:
            active_path = self.filesystem.require_file(self.active_path, self.root)
            payload = json.loads(
                active_path.read_text(encoding="utf-8"),
                object_pairs_hook=self._reject_duplicate_fields,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SnapshotUnavailableError("active snapshot pointer is malformed") from error
        if not isinstance(payload, dict) or frozenset(payload) != _POINTER_FIELDS:
            raise SnapshotUnavailableError("active snapshot pointer is malformed")
        snapshot_id = payload["snapshot_id"]
        manifest_digest = payload["manifest_sha256"]
        promoted_at = payload["promoted_at"]
        try:
            self._require_identifier(snapshot_id)
        except ValueError as error:
            raise SnapshotUnavailableError("active snapshot pointer is malformed") from error
        if (
            type(manifest_digest) is not str
            or fullmatch(_SHA256_PATTERN, manifest_digest) is None
            or type(promoted_at) is not str
        ):
            raise SnapshotUnavailableError("active snapshot pointer is malformed")
        try:
            promoted = datetime.fromisoformat(promoted_at)
        except ValueError as error:
            raise SnapshotUnavailableError("active snapshot pointer is malformed") from error
        if promoted.tzinfo is None or promoted.utcoffset() != timedelta(0):
            raise SnapshotUnavailableError("active snapshot pointer is malformed")
        return {
            "snapshot_id": snapshot_id,
            "manifest_sha256": manifest_digest,
            "promoted_at": promoted_at,
        }

    @staticmethod
    def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotUnavailableError("active snapshot pointer contains duplicate fields")
            result[key] = value
        return result

    @staticmethod
    def _require_identifier(value: object) -> None:
        if type(value) is not str or fullmatch(_IDENTIFIER_PATTERN, value) is None:
            raise ValueError("snapshot identifier is invalid")
