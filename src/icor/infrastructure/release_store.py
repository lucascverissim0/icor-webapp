"""Filesystem storage for immutable, verified source releases."""

from __future__ import annotations

import os
import re
import shutil
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
        if not source_path.is_file():
            raise ReleaseIntegrityError("source artifact is missing or not a regular file")
        if not source_path.suffix:
            raise ReleaseIntegrityError("source artifact must have an extension")
        self._verify_file_matches_manifest(source_path, manifest, "source artifact")

        destination = self.root / manifest.source_id / manifest.release_id
        if destination.exists():
            return self._return_existing_or_raise(destination, manifest)

        staging = self.root / ".staging" / uuid4().hex
        try:
            staging.mkdir(parents=True)
            artifact_path = staging / f"artifact{source_path.suffix}"
            self._copy_with_fsync(source_path, artifact_path)
            self._verify_file_matches_manifest(artifact_path, manifest, "staged artifact")
            write_release_manifest(staging / "manifest.json", manifest)

            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                staging.rename(destination)
            except FileExistsError:
                return self._return_existing_or_raise(destination, manifest)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        return self._verify_stored_release(self._load_release_directory(destination))

    def get(self, release_id: str) -> StoredRelease:
        """Return one complete stored release without hashing its artifact again."""
        self._require_identifier(release_id, "release")
        if not self.root.is_dir():
            raise FileNotFoundError(f"release does not exist: {release_id}")

        matches = [
            source_path / release_id
            for source_path in self.root.iterdir()
            if source_path.name != ".staging"
            and source_path.is_dir()
            and (source_path / release_id).exists()
        ]
        if not matches:
            raise FileNotFoundError(f"release does not exist: {release_id}")
        if len(matches) > 1:
            raise ReleaseIntegrityError(f"release ID appears under multiple sources: {release_id}")
        return self._load_release_directory(matches[0])

    def verify(self, release_id: str) -> StoredRelease:
        """Return a stored release only after exact byte and checksum verification."""
        return self._verify_stored_release(self.get(release_id))

    def list_releases(self) -> tuple[StoredRelease, ...]:
        """Return all complete releases in stable release-ID order."""
        if not self.root.is_dir():
            return ()

        releases: list[StoredRelease] = []
        seen_release_ids: set[str] = set()
        for source_path in self.root.iterdir():
            if source_path.name == ".staging" or not source_path.is_dir():
                continue
            self._require_identifier(source_path.name, "source")
            for release_path in source_path.iterdir():
                if not release_path.is_dir():
                    raise ReleaseIntegrityError(f"release directory is incomplete: {release_path}")
                stored = self._load_release_directory(release_path)
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

    def _return_existing_or_raise(
        self, destination: Path, manifest: ReleaseManifest
    ) -> StoredRelease:
        stored = self._verify_stored_release(self._load_release_directory(destination))
        if stored.manifest != manifest:
            raise ReleaseAlreadyExistsError(f"release already exists: {manifest.release_id}")
        return stored

    def _load_release_directory(self, release_path: Path) -> StoredRelease:
        if not release_path.is_dir():
            raise ReleaseIntegrityError(f"release directory is incomplete: {release_path}")
        self._require_identifier(release_path.name, "release")
        self._require_identifier(release_path.parent.name, "source")

        manifest_path = release_path / "manifest.json"
        artifact_paths = tuple(
            path
            for path in release_path.iterdir()
            if path.is_file() and path.name.startswith("artifact.")
        )
        if not manifest_path.is_file() or len(artifact_paths) != 1:
            raise ReleaseIntegrityError(f"release directory is incomplete: {release_path}")
        try:
            manifest = load_release_manifest(manifest_path)
        except ManifestError as error:
            raise ReleaseIntegrityError(f"release manifest is invalid: {error}") from error
        if (
            manifest.release_id != release_path.name
            or manifest.source_id != release_path.parent.name
        ):
            raise ReleaseIntegrityError("release directory identifiers do not match its manifest")

        return StoredRelease(
            source_id=manifest.source_id,
            release_id=manifest.release_id,
            artifact_path=artifact_paths[0],
            manifest_path=manifest_path,
            manifest=manifest,
        )

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

    def _verify_stored_release(self, stored: StoredRelease) -> StoredRelease:
        self._verify_file_matches_manifest(stored.artifact_path, stored.manifest, "stored artifact")
        return stored
