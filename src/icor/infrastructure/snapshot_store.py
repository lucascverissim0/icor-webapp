"""Atomic promotion and lookup of last-known-good evidence snapshots."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from re import fullmatch
from tempfile import NamedTemporaryFile
from uuid import uuid4

from icor.domain.snapshots import SnapshotManifest
from icor.evidence.release_manifests import load_snapshot_manifest
from icor.evidence.serialization import canonical_json_bytes, sha256_file
from icor.evidence.validation import SnapshotValidator, ValidationReport
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
    ) -> None:
        self.root = Path(root)
        self.clock = clock or (lambda: datetime.now(UTC))
        self.validator = validator or SnapshotValidator()

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
        if self.active_path.is_file():
            try:
                active = self.active_manifest()
            except SnapshotUnavailableError:
                pass
            else:
                if active.snapshot_id == snapshot_id:
                    return active

        candidate = self.candidate_path(snapshot_id)
        try:
            self._verify_snapshot_directory(candidate, snapshot_id)
            target = self._publish_candidate(candidate, snapshot_id)
            manifest, manifest_digest, _ = self._verify_snapshot_directory(target, snapshot_id)
            self._replace_active_pointer(snapshot_id, manifest_digest)
            return manifest
        except SnapshotPromotionError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise SnapshotPromotionError("candidate promotion failed") from error

    def active_manifest(self) -> SnapshotManifest:
        """Return the manifest selected by a complete and verified active pointer."""
        try:
            pointer = self._load_active_pointer()
            snapshot_id = pointer["snapshot_id"]
            target = self.root / "snapshots" / snapshot_id
            manifest, manifest_digest, _ = self._verify_snapshot_directory(target, snapshot_id)
            if manifest_digest != pointer["manifest_sha256"]:
                raise SnapshotUnavailableError("active snapshot manifest does not match pointer")
            return manifest
        except SnapshotUnavailableError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise SnapshotUnavailableError("active snapshot cannot be verified") from error

    def open_active_repository(self) -> SQLiteEvidenceRepository:
        """Open the verified active ledger through its read-only repository boundary."""
        manifest = self.active_manifest()
        path = self.root / "snapshots" / manifest.snapshot_id / "evidence.sqlite3"
        try:
            return SQLiteEvidenceRepository(path)
        except (OSError, RuntimeError, ValueError) as error:
            raise SnapshotUnavailableError("active snapshot database is unavailable") from error

    def _publish_candidate(self, candidate: Path, snapshot_id: str) -> Path:
        snapshots_root = self.root / "snapshots"
        snapshots_root.mkdir(parents=True, exist_ok=True)
        target = snapshots_root / snapshot_id
        if target.exists():
            self._verify_snapshot_directory(target, snapshot_id)
            return target

        staging = snapshots_root / f".promoting-{snapshot_id}-{uuid4().hex}"
        try:
            shutil.copytree(candidate, staging)
            self._verify_snapshot_directory(staging, snapshot_id)
            try:
                staging.rename(target)
            except FileExistsError:
                self._verify_snapshot_directory(target, snapshot_id)
            return target
        finally:
            if staging.exists() and not staging.is_symlink():
                shutil.rmtree(staging)

    def _verify_snapshot_directory(
        self, directory: Path, snapshot_id: str
    ) -> tuple[SnapshotManifest, str, ValidationReport]:
        if directory.is_symlink() or not directory.is_dir():
            raise SnapshotPromotionError("candidate snapshot directory is unavailable")
        try:
            entries = tuple(directory.iterdir())
        except OSError as error:
            raise SnapshotPromotionError("candidate snapshot directory cannot be read") from error
        if {entry.name for entry in entries} != _CANDIDATE_FILES or any(
            entry.is_symlink() or not entry.is_file() for entry in entries
        ):
            raise SnapshotPromotionError("candidate snapshot files are incomplete or unsafe")

        database_path = directory / "evidence.sqlite3"
        manifest_path = directory / "snapshot.json"
        validation_path = directory / "validation.json"
        try:
            manifest = load_snapshot_manifest(manifest_path)
            if manifest.snapshot_id != snapshot_id:
                raise SnapshotPromotionError("candidate snapshot identity does not match")
            repository = SQLiteEvidenceRepository(database_path)
            report = self.validator.validate(repository, manifest)
            if not report.can_promote:
                raise SnapshotPromotionError("candidate snapshot failed validation")
            if validation_path.read_bytes() != canonical_json_bytes(report):
                raise SnapshotPromotionError("candidate validation report does not match")
            manifest_digest = sha256_file(manifest_path)
        except SnapshotPromotionError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise SnapshotPromotionError("candidate snapshot cannot be verified") from error
        return manifest, manifest_digest, report

    def _replace_active_pointer(self, snapshot_id: str, manifest_digest: str) -> None:
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
        self.root.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                "wb", delete=False, dir=self.root, prefix=".active.json."
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(canonical_json_bytes(pointer))
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.active_path)
        except OSError as error:
            raise SnapshotPromotionError("active snapshot pointer could not be replaced") from error
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _load_active_pointer(self) -> dict[str, str]:
        if self.active_path.is_symlink() or not self.active_path.is_file():
            raise SnapshotUnavailableError("no active snapshot is available")
        try:
            payload = json.loads(
                self.active_path.read_text(encoding="utf-8"),
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
