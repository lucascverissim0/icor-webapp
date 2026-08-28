"""Read-only release and candidate-snapshot quality gates."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from re import compile

from icor.domain.snapshots import SnapshotManifest
from icor.evidence.serialization import sha256_file
from icor.infrastructure.release_store import StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

_SAFE_RECORD_ID = compile(r"[a-z0-9][a-z0-9._-]{0,79}\Z")


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
            with closing(_open_read_only(path)) as connection:
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
                for row in connection.execute(
                    "SELECT DISTINCT release_id FROM observation ORDER BY release_id"
                ):
                    if row["release_id"] not in manifest.release_ids:
                        findings.append(
                            _error(
                                "snapshot.release_unmanifested",
                                "Snapshot omits a release used by observations.",
                                row["release_id"],
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
                findings.extend(_database_findings(connection, manifest))
        except (OSError, sqlite3.Error):
            findings.append(
                _error("snapshot.database_unreadable", "Snapshot database cannot be validated.")
            )
        return _report(findings)


def _database_findings(
    connection: sqlite3.Connection,
    manifest: SnapshotManifest,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for row in connection.execute(
        """SELECT published_value.value_id
        FROM published_value
        LEFT JOIN published_value_input
        ON published_value_input.value_id = published_value.value_id
        GROUP BY published_value.value_id
        HAVING COUNT(published_value_input.observation_id) = 0"""
    ):
        findings.append(
            _error(
                "snapshot.missing_input",
                "Published value must have at least one input.",
                row["value_id"],
            )
        )
    for row in connection.execute(
        """SELECT published_value_input.value_id, published_value_input.observation_id
        FROM published_value_input
        LEFT JOIN published_value ON published_value.value_id = published_value_input.value_id
        LEFT JOIN observation ON observation.observation_id = published_value_input.observation_id
        WHERE published_value.value_id IS NULL OR observation.observation_id IS NULL"""
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
        elif any(interval < 0 for interval in intervals):
            findings.append(
                _error(
                    "snapshot.interval_negative",
                    "Published interval must not be negative.",
                    row["value_id"],
                )
            )
        elif not intervals[0] <= intervals[1] <= intervals[2]:
            findings.append(
                _error(
                    "snapshot.interval_order", "Published interval is not ordered.", row["value_id"]
                )
            )
    for row in connection.execute(
        """SELECT published_value_input.value_id,
        published_value_input.observation_id,
        COUNT(identity_mapping.mapping_id) AS mapping_count
        FROM published_value_input
        JOIN observation
        ON observation.observation_id = published_value_input.observation_id
        LEFT JOIN identity_mapping
        ON identity_mapping.observation_id = observation.observation_id
        GROUP BY published_value_input.value_id,
        published_value_input.observation_id
        HAVING COUNT(identity_mapping.mapping_id) != 1"""
    ):
        findings.append(
            _error(
                "snapshot.mapping_cardinality",
                "Published input must have exactly one selected identity mapping.",
                row["value_id"],
            )
        )
    for row in connection.execute(
        """SELECT DISTINCT published_value.value_id
        FROM published_value
        JOIN published_value_input
        ON published_value_input.value_id = published_value.value_id
        JOIN observation
        ON observation.observation_id = published_value_input.observation_id
        JOIN identity_mapping
        ON identity_mapping.observation_id = observation.observation_id
        WHERE identity_mapping.canonical_vehicle_id
            IS NOT observation.canonical_vehicle_id
        OR identity_mapping.status != observation.mapping_status
        OR identity_mapping.canonical_vehicle_id
            IS NOT published_value.canonical_vehicle_id
        OR identity_mapping.status != published_value.mapping_status
        OR observation.canonical_vehicle_id
            IS NOT published_value.canonical_vehicle_id
        OR observation.mapping_status != published_value.mapping_status"""
    ):
        findings.append(
            _error(
                "snapshot.mapping_attribution_mismatch",
                "Published value identity mapping attribution is inconsistent.",
                row["value_id"],
            )
        )
    for row in connection.execute(
        """SELECT DISTINCT published_value.value_id FROM published_value
        LEFT JOIN published_value_input ON published_value_input.value_id = published_value.value_id
        LEFT JOIN observation ON observation.observation_id = published_value_input.observation_id
        LEFT JOIN identity_mapping ON identity_mapping.observation_id = observation.observation_id
        WHERE published_value.mapping_status IN ('ambiguous', 'rejected', 'unresolved')
        OR observation.mapping_status IN ('ambiguous', 'rejected', 'unresolved')
        OR identity_mapping.status IN ('ambiguous', 'rejected', 'unresolved')"""
    ):
        findings.append(
            _error(
                "snapshot.unresolved_publication",
                "Published value has an unresolved identity mapping.",
                row["value_id"],
            )
        )
    if not manifest.versions.generation_registry.endswith("-v0"):
        findings.extend(_generation_planning_findings(connection, manifest))
    return findings


def _generation_planning_findings(
    connection: sqlite3.Connection,
    manifest: SnapshotManifest,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    required_tables = {
        "generation_entry",
        "generation_assignment",
        "generation_alternative",
        "cohort_estimate",
        "opportunity_estimate",
        "completeness_record",
    }
    available = {
        row["name"]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if not required_tables <= available:
        return [
            _error(
                "snapshot.generation_schema_missing",
                "Generation planning schema is unavailable.",
            )
        ]

    for row in connection.execute(
        """SELECT observation.observation_id
        FROM observation
        LEFT JOIN generation_assignment
        ON generation_assignment.observation_id = observation.observation_id
        WHERE observation.canonical_vehicle_id IS NOT NULL
        AND observation.mapping_status NOT IN ('ambiguous', 'rejected', 'unresolved')
        AND (
            observation.registration_cohort_year IS NOT NULL
            OR observation.manufacture_year IS NOT NULL
            OR (
                observation.measure = 'new_registrations'
                AND SUBSTR(observation.period_start, 1, 4) =
                    SUBSTR(observation.period_end, 1, 4)
            )
        )
        AND generation_assignment.assignment_id IS NULL
        ORDER BY observation.observation_id"""
    ):
        findings.append(
            _error(
                "snapshot.generation_assignment_missing",
                "Usable observation has no generation assignment.",
                row["observation_id"],
            )
        )

    for row in connection.execute(
        """SELECT generation_assignment.assignment_id,
        observation.canonical_vehicle_id AS observation_vehicle_id,
        generation_entry.canonical_vehicle_id AS generation_vehicle_id,
        generation_assignment.training_weight,
        generation_assignment.registry_version,
        generation_assignment.resolver_version
        FROM generation_assignment
        LEFT JOIN observation
        ON observation.observation_id = generation_assignment.observation_id
        LEFT JOIN generation_entry
        ON generation_entry.generation_id = generation_assignment.selected_generation_id
        ORDER BY generation_assignment.assignment_id"""
    ):
        assignment_id = row["assignment_id"]
        if (
            row["observation_vehicle_id"] is None
            or row["generation_vehicle_id"] is None
            or row["observation_vehicle_id"] != row["generation_vehicle_id"]
        ):
            findings.append(
                _error(
                    "snapshot.generation_assignment_orphan",
                    "Generation assignment has incompatible lineage.",
                    assignment_id,
                )
            )
        weight = _decimal(row["training_weight"])
        if weight is None or weight is _INVALID or not Decimal(0) <= weight <= Decimal(1):
            findings.append(
                _error(
                    "snapshot.generation_weight_invalid",
                    "Generation assignment training weight is invalid.",
                    assignment_id,
                )
            )
        if (
            row["registry_version"] != manifest.versions.generation_registry
            or row["resolver_version"] != manifest.versions.generation_resolver
        ):
            findings.append(
                _error(
                    "snapshot.generation_version_mismatch",
                    "Generation assignment versions do not match the snapshot.",
                    assignment_id,
                )
            )

    for row in connection.execute(
        """SELECT generation_id, start_month, end_month FROM generation_entry
        ORDER BY generation_id"""
    ):
        try:
            start = date.fromisoformat(row["start_month"])
            end = date.fromisoformat(row["end_month"]) if row["end_month"] else None
        except (TypeError, ValueError):
            start = end = None
        if start is None or start.day != 1 or (end is not None and end.day != 1):
            findings.append(
                _error(
                    "snapshot.generation_window_invalid",
                    "Generation window does not use valid month precision.",
                    row["generation_id"],
                )
            )
        elif end is not None and end < start:
            findings.append(
                _error(
                    "snapshot.generation_window_reversed",
                    "Generation window is not ordered.",
                    row["generation_id"],
                )
            )

    for table, identifier, names in (
        (
            "cohort_estimate",
            "cohort_id",
            ("active_fleet_p10", "active_fleet_p50", "active_fleet_p90"),
        ),
        ("opportunity_estimate", "opportunity_id", ("p10", "p50", "p90")),
    ):
        for row in connection.execute(
            f"SELECT {identifier}, {', '.join(names)} FROM {table} ORDER BY {identifier}"
        ):
            intervals = tuple(_decimal(row[name]) for name in names)
            if (
                any(item is None or item is _INVALID for item in intervals)
                or any(item < 0 for item in intervals)
                or not intervals[0] <= intervals[1] <= intervals[2]
            ):
                findings.append(
                    _error(
                        "snapshot.generation_interval_invalid",
                        "Generation planning interval is invalid.",
                        row[identifier],
                    )
                )

    if _count(connection, "completeness_record") == 0:
        findings.append(
            _error(
                "snapshot.completeness_missing",
                "Generation planning completeness records are unavailable.",
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


def _error(code: str, message: str, record_id: object = None) -> ValidationFinding:
    return ValidationFinding(code, Severity.ERROR, message, _safe_record_id(record_id))


def _safe_record_id(value: object) -> str | None:
    if type(value) is str and _SAFE_RECORD_ID.fullmatch(value) is not None:
        return value
    return None


def _finding_key(finding: ValidationFinding) -> tuple[Severity, str, str]:
    return finding.severity, finding.code, finding.record_id or ""


def _report(findings: list[ValidationFinding]) -> ValidationReport:
    return ValidationReport(tuple(sorted(findings, key=_finding_key)))
