"""Read-only release and candidate-snapshot quality gates."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path

from icor.domain.snapshots import SnapshotManifest
from icor.evidence.serialization import sha256_file
from icor.infrastructure.release_store import StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository


class Severity(StrEnum):
    """The promotion impact of a validation finding."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    """A stable, sanitized validation outcome."""

    code: str
    severity: Severity
    message: str
    record_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Sorted findings and the resulting snapshot-promotion decision."""

    findings: tuple[ValidationFinding, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(self.findings, key=_finding_key)) != self.findings:
            raise ValueError("validation findings must be sorted")

    @property
    def can_promote(self) -> bool:
        return not any(row.severity is Severity.ERROR for row in self.findings)


class ReleaseValidator:
    """Validate a stored source artifact without mutating release storage."""

    def validate(self, stored_release: StoredRelease) -> ValidationReport:
        manifest = stored_release.manifest
        findings: list[ValidationFinding] = []
        if not manifest.terms_url.strip() or not manifest.permitted_local_use.strip():
            findings.append(
                _error("release.terms_metadata_missing", "Release terms metadata is missing.")
            )
        if manifest.coverage_start > manifest.coverage_end:
            findings.append(
                _error("release.coverage_reversed", "Release coverage dates are not ordered.")
            )
        if (
            manifest.accepted_record_count
            + manifest.rejected_record_count
            + manifest.quarantined_record_count
            != manifest.raw_record_count
        ):
            findings.append(
                _error("release.record_count_mismatch", "Release record counts do not reconcile.")
            )

        artifact_path = stored_release.artifact_path
        if artifact_path.is_symlink() or not artifact_path.is_file():
            findings.append(_error("release.artifact_missing", "Release artifact is unavailable."))
        else:
            try:
                digest = sha256_file(artifact_path)
                if digest != manifest.sha256:
                    findings.append(
                        _error(
                            "release.checksum_mismatch", "Release artifact checksum does not match."
                        )
                    )
                elif artifact_path.stat().st_size != manifest.artifact_bytes:
                    findings.append(
                        _error(
                            "release.byte_size_mismatch",
                            "Release artifact byte size does not match.",
                        )
                    )
            except OSError:
                findings.append(
                    _error("release.artifact_unreadable", "Release artifact cannot be read.")
                )
        return _report(findings)


class SnapshotValidator:
    """Validate a candidate manifest and SQLite evidence ledger without writes."""

    def validate(
        self, repository: SQLiteEvidenceRepository, manifest: SnapshotManifest
    ) -> ValidationReport:
        findings: list[ValidationFinding] = []
        path = repository.path
        if not path.is_file():
            return _report(
                [_error("snapshot.database_missing", "Snapshot database is unavailable.")]
            )
        try:
            if sha256_file(path) != manifest.database_sha256:
                findings.append(
                    _error(
                        "snapshot.database_hash_mismatch",
                        "Snapshot database checksum does not match.",
                    )
                )
            with _open_read_only(path) as connection:
                release_ids = {
                    row["release_id"]
                    for row in connection.execute("SELECT release_id FROM source_release")
                }
                for release_id in manifest.release_ids:
                    if release_id not in release_ids:
                        findings.append(
                            _error(
                                "snapshot.release_missing",
                                "Snapshot references a release that is unavailable.",
                                release_id,
                            )
                        )
                observation_count = _count(connection, "observation")
                if observation_count != manifest.observation_count:
                    findings.append(
                        _error(
                            "snapshot.observation_count_mismatch",
                            "Snapshot observation count does not match.",
                        )
                    )
                published_value_count = _count(connection, "published_value")
                if published_value_count != manifest.published_value_count:
                    findings.append(
                        _error(
                            "snapshot.published_value_count_mismatch",
                            "Snapshot published value count does not match.",
                        )
                    )
                findings.extend(_database_findings(connection))
        except (OSError, sqlite3.Error):
            findings.append(
                _error("snapshot.database_unreadable", "Snapshot database cannot be validated.")
            )
        return _report(findings)


def _database_findings(connection: sqlite3.Connection) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for row in connection.execute(
        """SELECT published_value_input.value_id, published_value_input.observation_id
        FROM published_value_input
        LEFT JOIN observation ON observation.observation_id = published_value_input.observation_id
        WHERE observation.observation_id IS NULL"""
    ):
        findings.append(
            _error(
                "snapshot.orphan_input",
                "Published value has an unavailable input.",
                row["value_id"],
            )
        )
    for table, identifier in (("observation", "observation_id"), ("published_value", "value_id")):
        for row in connection.execute(f"SELECT {identifier}, value FROM {table}"):
            if _is_negative(row["value"]):
                findings.append(
                    _error(
                        "snapshot.negative_value",
                        "Evidence value must not be negative.",
                        row[identifier],
                    )
                )
    for row in connection.execute(
        """SELECT value_id, p10, p50, p90 FROM published_value
        WHERE p10 IS NOT NULL OR p50 IS NOT NULL OR p90 IS NOT NULL"""
    ):
        intervals = tuple(_decimal(row[name]) for name in ("p10", "p50", "p90"))
        if any(item is None for item in intervals) or any(item is _INVALID for item in intervals):
            findings.append(
                _error(
                    "snapshot.interval_invalid", "Published interval is invalid.", row["value_id"]
                )
            )
        elif not intervals[0] <= intervals[1] <= intervals[2]:
            findings.append(
                _error(
                    "snapshot.interval_order", "Published interval is not ordered.", row["value_id"]
                )
            )
    for row in connection.execute(
        """SELECT value_id FROM published_value
        WHERE mapping_status IN ('ambiguous', 'rejected', 'unresolved')"""
    ):
        findings.append(
            _error(
                "snapshot.unresolved_publication",
                "Published value has an unresolved identity mapping.",
                row["value_id"],
            )
        )
    return findings


_INVALID = object()


def _decimal(value: object) -> Decimal | None | object:
    if value is None:
        return None
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return _INVALID
    return decimal if decimal.is_finite() else _INVALID


def _is_negative(value: object) -> bool:
    decimal = _decimal(value)
    return decimal is _INVALID or (isinstance(decimal, Decimal) and decimal < 0)


def _open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _error(code: str, message: str, record_id: str | None = None) -> ValidationFinding:
    return ValidationFinding(code, Severity.ERROR, message, record_id)


def _finding_key(finding: ValidationFinding) -> tuple[Severity, str, str]:
    return finding.severity, finding.code, finding.record_id or ""


def _report(findings: list[ValidationFinding]) -> ValidationReport:
    return ValidationReport(tuple(sorted(findings, key=_finding_key)))
