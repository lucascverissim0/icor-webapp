"""Filesystem storage for immutable, verified source releases."""

from __future__ import annotations

import ctypes
import errno
import os
import re
import shutil
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from icor.domain.evidence import ReleaseManifest
from icor.evidence.release_manifests import (
    ManifestError,
    load_release_manifest,
    write_release_manifest,
)
from icor.evidence.serialization import sha256_file

_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}\Z")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class ReleaseAlreadyExistsError(RuntimeError):
    """A release ID is already occupied by different immutable content."""


class ReleaseIntegrityError(RuntimeError):
    """A stored release is incomplete, malformed, or does not match its manifest."""


@dataclass(frozen=True, slots=True)
class StoredRelease:
    """The on-disk locations and validated manifest for one stored release."""

    source_id: str
    release_id: str
    artifact_path: Path
    manifest_path: Path
    manifest: ReleaseManifest


class ReleaseStore:
    """Store source releases once under ``<root>/<source_id>/<release_id>``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def stage(self, source_artifact: Path, manifest: ReleaseManifest) -> StoredRelease:
        """Copy one verified artifact and its manifest into immutable local storage."""
        if not isinstance(manifest, ReleaseManifest):
            raise TypeError("manifest must be a ReleaseManifest")
        self._require_identifier(manifest.source_id, "source")
        self._require_identifier(manifest.release_id, "release")

        source_path = Path(source_artifact)
        if source_path.is_symlink() or not source_path.is_file():
            raise ReleaseIntegrityError(
                "source artifact is missing, symlinked, or not a regular file"
            )
        if not source_path.suffix:
            raise ReleaseIntegrityError("source artifact must have an extension")
        artifact_name = f"artifact{source_path.suffix}"
        if manifest.artifact_path != artifact_name:
            raise ReleaseIntegrityError(
                "manifest artifact path does not match the source extension"
            )
        self._verify_file_matches_manifest(source_path, manifest, "source artifact")

        root = self._store_root(create=True)
        destination = self.root / manifest.source_id / manifest.release_id
        lock = self._acquire_release_lock(manifest.release_id, root)
        try:
            matches = self._find_release_paths(manifest.release_id, root)
            if matches:
                if len(matches) == 1 and matches[0] == destination:
                    return self._return_existing_or_raise(destination, manifest, root)
                raise ReleaseAlreadyExistsError(f"release already exists: {manifest.release_id}")

            staging_root = self.root / ".staging"
            self._prepare_directory(staging_root, root, "staging directory")
            staging = staging_root / uuid4().hex
            try:
                staging.mkdir()
                self._require_safe_directory(staging, root, "staging directory")
                artifact_path = staging / artifact_name
                self._copy_with_fsync(source_path, artifact_path)
                self._verify_file_matches_manifest(artifact_path, manifest, "staged artifact")
                write_release_manifest(staging / "manifest.json", manifest)

                self._prepare_directory(destination.parent, root, "source directory")
                try:
                    self._publish_no_replace(staging, destination)
                except FileExistsError:
                    return self._return_existing_or_raise(destination, manifest, root)
            finally:
                if staging.exists() and not staging.is_symlink():
                    shutil.rmtree(staging)

            stored = self._load_release_directory(destination, root)
            return self._verify_stored_release(stored, root)
        finally:
            lock.rmdir()

    def get(self, release_id: str) -> StoredRelease:
        """Return one complete stored release without hashing its artifact again."""
        self._require_identifier(release_id, "release")
        root = self._store_root(create=False)
        if root is None:
            raise FileNotFoundError(f"release does not exist: {release_id}")

        matches = self._find_release_paths(release_id, root)
        if not matches:
            raise FileNotFoundError(f"release does not exist: {release_id}")
        if len(matches) > 1:
            raise ReleaseIntegrityError(f"release ID appears under multiple sources: {release_id}")
        return self._load_release_directory(matches[0], root)

    def verify(self, release_id: str) -> StoredRelease:
        """Return a stored release only after exact byte and checksum verification."""
        stored = self.get(release_id)
        root = self._store_root(create=False)
        if root is None:  # pragma: no cover - get() already established the root exists.
            raise ReleaseIntegrityError("release store disappeared during verification")
        return self._verify_stored_release(stored, root)

    def list_releases(self) -> tuple[StoredRelease, ...]:
        """Return all complete releases in stable release-ID order."""
        root = self._store_root(create=False)
        if root is None:
            return ()

        releases: list[StoredRelease] = []
        seen_release_ids: set[str] = set()
        for source_path in self.root.iterdir():
            if source_path.name in {".locks", ".staging"}:
                if source_path.is_symlink():
                    raise ReleaseIntegrityError("internal store directory must not be a symlink")
                continue
            if source_path.is_symlink():
                raise ReleaseIntegrityError("source directory must not be a symlink")
            if not source_path.is_dir():
                continue
            self._require_identifier(source_path.name, "source")
            self._require_safe_directory(source_path, root, "source directory")
            for release_path in source_path.iterdir():
                stored = self._load_release_directory(release_path, root)
                if stored.release_id in seen_release_ids:
                    raise ReleaseIntegrityError(
                        f"release ID appears under multiple sources: {stored.release_id}"
                    )
                seen_release_ids.add(stored.release_id)
                releases.append(stored)
        return tuple(sorted(releases, key=lambda stored: stored.release_id))

    @staticmethod
    def _require_identifier(value: object, label: str) -> None:
        if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{label} identifier is invalid")

    def _store_root(self, *, create: bool) -> Path | None:
        if self.root.is_symlink():
            raise ReleaseIntegrityError("release store root must not be a symlink")
        if not self.root.exists():
            if not create:
                return None
            self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ReleaseIntegrityError("release store root is not a directory")
        return self.root.resolve(strict=True)

    def _find_release_paths(self, release_id: str, root: Path) -> list[Path]:
        matches: list[Path] = []
        for source_path in self.root.iterdir():
            if source_path.name in {".locks", ".staging"}:
                if source_path.is_symlink():
                    raise ReleaseIntegrityError("internal store directory must not be a symlink")
                continue
            if source_path.is_symlink():
                raise ReleaseIntegrityError("source directory must not be a symlink")
            if not source_path.is_dir():
                continue
            self._require_identifier(source_path.name, "source")
            self._require_safe_directory(source_path, root, "source directory")
            candidate = source_path / release_id
            if candidate.exists() or candidate.is_symlink():
                self._require_safe_directory(candidate, root, "release directory")
                matches.append(candidate)
        return matches

    def _acquire_release_lock(self, release_id: str, root: Path) -> Path:
        locks_root = self.root / ".locks"
        self._prepare_directory(locks_root, root, "release lock directory")
        lock = locks_root / release_id
        try:
            lock.mkdir()
        except FileExistsError as error:
            raise ReleaseAlreadyExistsError(
                f"release staging is already in progress: {release_id}"
            ) from error
        self._require_safe_directory(lock, root, "release lock directory")
        return lock

    def _return_existing_or_raise(
        self, destination: Path, manifest: ReleaseManifest, root: Path
    ) -> StoredRelease:
        stored = self._verify_stored_release(self._load_release_directory(destination, root), root)
        if stored.manifest != manifest:
            raise ReleaseAlreadyExistsError(f"release already exists: {manifest.release_id}")
        return stored

    def _load_release_directory(self, release_path: Path, root: Path) -> StoredRelease:
        self._require_safe_directory(release_path, root, "release directory")
        self._require_identifier(release_path.name, "release")
        self._require_safe_directory(release_path.parent, root, "source directory")
        self._require_identifier(release_path.parent.name, "source")

        manifest_path = release_path / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ReleaseIntegrityError(f"release directory is incomplete: {release_path}")
        self._require_contained(manifest_path, root, "manifest")

        artifact_candidates = tuple(
            path for path in release_path.iterdir() if path.name.startswith("artifact.")
        )
        if any(path.is_symlink() for path in artifact_candidates):
            raise ReleaseIntegrityError("artifact must not be a symlink")
        artifact_paths = tuple(path for path in artifact_candidates if path.is_file())
        if len(artifact_paths) != 1:
            raise ReleaseIntegrityError(f"release directory is incomplete: {release_path}")
        artifact_path = artifact_paths[0]
        self._require_contained(artifact_path, root, "artifact")
        try:
            manifest = load_release_manifest(manifest_path)
        except ManifestError as error:
            raise ReleaseIntegrityError(f"release manifest is invalid: {error}") from error
        if (
            manifest.release_id != release_path.name
            or manifest.source_id != release_path.parent.name
            or manifest.artifact_path != artifact_path.name
        ):
            raise ReleaseIntegrityError("release directory does not match its manifest")

        return StoredRelease(
            source_id=manifest.source_id,
            release_id=manifest.release_id,
            artifact_path=artifact_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    @staticmethod
    def _prepare_directory(path: Path, root: Path, label: str) -> None:
        with suppress(FileExistsError):
            path.mkdir(parents=True, exist_ok=True)
        ReleaseStore._require_safe_directory(path, root, label)

    @staticmethod
    def _require_safe_directory(path: Path, root: Path, label: str) -> None:
        if path.is_symlink():
            raise ReleaseIntegrityError(f"{label} must not be a symlink")
        if not path.is_dir():
            raise ReleaseIntegrityError(f"{label} is incomplete")
        ReleaseStore._require_contained(path, root, label)

    @staticmethod
    def _require_contained(path: Path, root: Path, label: str) -> None:
        try:
            path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as error:
            raise ReleaseIntegrityError(f"{label} escapes the release store root") from error

    @staticmethod
    def _publish_no_replace(staging: Path, destination: Path) -> None:
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if kernel32.MoveFileW(str(staging), str(destination)):
                return
            error = ctypes.get_last_error()
            if error in {80, 183}:
                raise FileExistsError(error, "release destination already exists", destination)
            raise ctypes.WinError(error)

        if sys.platform.startswith("linux"):
            libc = ctypes.CDLL(None, use_errno=True)
            try:
                renameat2 = libc.renameat2
            except AttributeError as error:
                raise ReleaseIntegrityError(
                    "platform lacks non-replacing release publication"
                ) from error
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            if renameat2(
                _AT_FDCWD,
                os.fsencode(staging),
                _AT_FDCWD,
                os.fsencode(destination),
                _RENAME_NOREPLACE,
            ) == 0:
                return
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(error, "release destination already exists", destination)
            raise OSError(error, os.strerror(error), destination)

        raise ReleaseIntegrityError("platform lacks non-replacing release publication")

    @staticmethod
    def _copy_with_fsync(source_path: Path, destination: Path) -> None:
        with source_path.open("rb") as source, destination.open("xb") as destination_file:
            while chunk := source.read(1024 * 1024):
                destination_file.write(chunk)
            destination_file.flush()
            os.fsync(destination_file.fileno())

    @staticmethod
    def _verify_file_matches_manifest(
        artifact_path: Path, manifest: ReleaseManifest, label: str
    ) -> None:
        if sha256_file(artifact_path) != manifest.sha256:
            raise ReleaseIntegrityError(f"{label} SHA-256 does not match its manifest")
        if artifact_path.stat().st_size != manifest.artifact_bytes:
            raise ReleaseIntegrityError(f"{label} byte count does not match its manifest")

    def _verify_stored_release(self, stored: StoredRelease, root: Path) -> StoredRelease:
        if stored.artifact_path.is_symlink() or not stored.artifact_path.is_file():
            raise ReleaseIntegrityError("artifact is missing or a symlink")
        self._require_contained(stored.artifact_path, root, "artifact")
        self._verify_file_matches_manifest(stored.artifact_path, stored.manifest, "stored artifact")
        return stored
