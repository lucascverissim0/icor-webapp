"""Immutable snapshot identity and reproducibility contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from re import fullmatch


class SnapshotStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUPERSEDED = "superseded"


_IDENTIFIER_PATTERN = r"[a-z0-9][a-z0-9._-]{0,79}"
_SHA256_PATTERN = r"[0-9a-f]{64}"


def _require_identifier(value: str, label: str) -> None:
    if type(value) is not str or fullmatch(_IDENTIFIER_PATTERN, value) is None:
        raise ValueError(f"{label} identifier is invalid")


def _require_text(value: str, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} version is required")


@dataclass(frozen=True, slots=True)
class SnapshotVersions:
    source_registry: str
    identity_registry: str
    reconciliation_method: str
    confidence_method: str
    estimation_method: str
    survival_method: str
    hazard_method: str
    forecast_method: str
    generation_registry: str = "generation-registry-v0"
    generation_resolver: str = "generation-resolver-v0"

    def __post_init__(self) -> None:
        for value in (
            self.source_registry,
            self.identity_registry,
            self.reconciliation_method,
            self.confidence_method,
            self.estimation_method,
            self.survival_method,
            self.hazard_method,
            self.forecast_method,
            self.generation_registry,
            self.generation_resolver,
        ):
            _require_text(value, "snapshot")


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    snapshot_id: str
    status: SnapshotStatus
    built_at: datetime
    deterministic_seed: int
    release_ids: tuple[str, ...]
    versions: SnapshotVersions
    database_sha256: str
    observation_count: int
    published_value_count: int
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.snapshot_id, "snapshot")
        if not isinstance(self.status, SnapshotStatus):
            raise ValueError("snapshot status is unsupported")
        if (
            not isinstance(self.built_at, datetime)
            or self.built_at.tzinfo is None
            or self.built_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("built_at must be a UTC datetime")
        for value, label in (
            (self.deterministic_seed, "deterministic seed"),
            (self.observation_count, "observation count"),
            (self.published_value_count, "published value count"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if not isinstance(self.release_ids, tuple) or not self.release_ids:
            raise ValueError("snapshot release IDs are required")
        for release_id in self.release_ids:
            _require_identifier(release_id, "release")
        if tuple(sorted(self.release_ids)) != self.release_ids:
            raise ValueError("snapshot release IDs must be sorted")
        if len(set(self.release_ids)) != len(self.release_ids):
            raise ValueError("snapshot release IDs must be unique")
        if not isinstance(self.versions, SnapshotVersions):
            raise ValueError("snapshot versions are required")
        if (
            type(self.database_sha256) is not str
            or fullmatch(_SHA256_PATTERN, self.database_sha256) is None
        ):
            raise ValueError("database SHA-256 must be a lowercase 64-character hexadecimal digest")
        if not isinstance(self.warnings, tuple) or any(
            type(warning) is not str or not warning.strip() for warning in self.warnings
        ):
            raise ValueError("snapshot warning entries must be nonblank")
