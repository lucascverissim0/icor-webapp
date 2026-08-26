"""Strict release and snapshot manifest decoding with atomic release writes."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import fields
from datetime import date, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import NamedTemporaryFile

from icor.domain.evidence import Measure, PublicationStatus, ReleaseManifest
from icor.domain.snapshots import SnapshotManifest, SnapshotStatus, SnapshotVersions

from .serialization import canonical_json_bytes


class ManifestError(ValueError):
    """A persisted manifest is malformed, incomplete, or unsafe."""


_RELEASE_FIELDS = frozenset(field.name for field in fields(ReleaseManifest))
_SNAPSHOT_FIELDS = frozenset(field.name for field in fields(SnapshotManifest))
_VERSION_FIELDS = frozenset(field.name for field in fields(SnapshotVersions))


def load_release_manifest(path: Path) -> ReleaseManifest:
    """Load one release manifest after exact-schema and domain validation."""
    payload = _load_json_object(path)
    _require_exact_keys(payload, _RELEASE_FIELDS, "release manifest")
    _validate_artifact_path(payload["artifact_path"])
    try:
        return ReleaseManifest(
            **{
                **payload,
                "retrieved_at": _parse_datetime(payload["retrieved_at"], "retrieved_at"),
                "published_at": _parse_datetime(payload["published_at"], "published_at"),
                "coverage_start": _parse_date(payload["coverage_start"], "coverage_start"),
                "coverage_end": _parse_date(payload["coverage_end"], "coverage_end"),
                "measure": _parse_enum(Measure, payload["measure"], "measure"),
                "publication_status": _parse_enum(
                    PublicationStatus, payload["publication_status"], "publication_status"
                ),
            }
        )
    except (TypeError, ValueError) as error:
        raise ManifestError(str(error)) from error


def load_snapshot_manifest(path: Path) -> SnapshotManifest:
    """Load a snapshot manifest so snapshot identity uses the same strict boundary."""
    payload = _load_json_object(path)
    _require_exact_keys(payload, _SNAPSHOT_FIELDS, "snapshot manifest")
    versions = payload["versions"]
    if not isinstance(versions, dict):
        raise ManifestError("snapshot versions must be an object")
    _require_exact_keys(versions, _VERSION_FIELDS, "snapshot versions")
    try:
        release_ids = _parse_string_list(payload["release_ids"], "release_ids")
        warnings = _parse_string_list(payload["warnings"], "warnings")
        return SnapshotManifest(
            **{
                **payload,
                "status": _parse_enum(SnapshotStatus, payload["status"], "status"),
                "built_at": _parse_datetime(payload["built_at"], "built_at"),
                "release_ids": tuple(release_ids),
                "versions": SnapshotVersions(**versions),
                "warnings": tuple(warnings),
            }
        )
    except (TypeError, ValueError) as error:
        raise ManifestError(str(error)) from error


def write_release_manifest(path: Path, manifest: ReleaseManifest) -> None:
    """Atomically persist a release manifest using canonical JSON."""
    if not isinstance(manifest, ReleaseManifest):
        raise TypeError("manifest must be a ReleaseManifest")
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb", delete=False, dir=path.parent, prefix=f".{path.name}."
        ) as temp:
            temporary_path = Path(temp.name)
            temp.write(canonical_json_bytes(manifest))
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        document = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ManifestError("manifest must be UTF-8") from error
    except OSError as error:
        raise ManifestError(f"manifest cannot be read: {error}") from error
    try:
        payload = json.loads(document, object_pairs_hook=_reject_duplicate_object_keys)
    except json.JSONDecodeError as error:
        raise ManifestError("manifest contains malformed JSON") from error
    if not isinstance(payload, dict):
        raise ManifestError("manifest JSON must be an object")
    return payload


def _reject_duplicate_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"manifest contains duplicate field: {key}")
        result[key] = value
    return result


def _require_exact_keys(
    payload: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(payload)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise ManifestError(f"{label} contains unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ManifestError(f"{label} is missing required fields: {', '.join(sorted(missing))}")


def _parse_datetime(value: object, label: str) -> datetime:
    if type(value) is not str:
        raise ManifestError(f"{label} must be an ISO datetime")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ManifestError(f"{label} must be an ISO datetime") from error


def _parse_date(value: object, label: str) -> date:
    if type(value) is not str:
        raise ManifestError(f"{label} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ManifestError(f"{label} must be an ISO date") from error


def _parse_enum[E: Measure | PublicationStatus | SnapshotStatus](
    enum_type: type[E], value: object, label: str
) -> E:
    if type(value) is not str:
        raise ManifestError(f"{label} is unsupported")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ManifestError(f"{label} is unsupported") from error


def _parse_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise ManifestError(f"{label} must be a list of strings")
    return value


def _validate_artifact_path(value: object) -> None:
    if type(value) is not str:
        raise ManifestError("artifact path must be a relative string")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    is_unsafe = (
        bool(posix_path.anchor)
        or bool(windows_path.anchor)
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    )
    if is_unsafe:
        raise ManifestError("artifact path must be relative and cannot traverse parents")
