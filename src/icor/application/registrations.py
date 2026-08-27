"""Read-only official registration ranking application service."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from icor.application.evidence_review import EvidenceReviewService
from icor.domain.snapshots import SnapshotManifest

_SUPPORTED_GEOGRAPHY = "EU27"
_SUPPORTED_YEAR = 2024
_EEA_SOURCE_ID = "eea-co2-monitoring"
_EEA_PARSER_NAME = "eea_co2_cars_zip_v1"
_IDENTITY_REGISTRY = "exact-normalized-model-family-v1"
_EU27_EEA_CODES = (
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR",
    "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO",
    "SE", "SI", "SK",
)


class RegistrationUnavailableError(RuntimeError):
    """A verified canonical registration snapshot is unavailable."""


@dataclass(frozen=True, slots=True)
class RegistrationQuery:
    geography: str = _SUPPORTED_GEOGRAPHY
    year: int = _SUPPORTED_YEAR
    search: str | None = None
    page: int = 1
    page_size: int = 25

    def validate(self) -> None:
        if self.geography != _SUPPORTED_GEOGRAPHY:
            raise ValueError("registration geography is unsupported")
        if self.year != _SUPPORTED_YEAR:
            raise ValueError("registration year is unsupported")
        if type(self.page) is not int or self.page < 1:
            raise ValueError("registration page must be positive")
        if type(self.page_size) is not int or not 1 <= self.page_size <= 100:
            raise ValueError("registration page size must be between 1 and 100")
        if self.search is not None and (
            type(self.search) is not str or len(self.search) > 100
        ):
            raise ValueError("registration search must be at most 100 characters")


@dataclass(frozen=True, slots=True)
class RegistrationRow:
    rank: int
    vehicle_id: str
    make: str
    model: str
    model_year: None
    registrations: Decimal
    status: str
    evidence_confidence: int
    input_observation_count: int
    release_ids: tuple[str, ...]
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RegistrationPage:
    items: tuple[RegistrationRow, ...]
    total: int
    total_registrations: Decimal
    page: int
    page_size: int
    pages: int
    snapshot_id: str


@dataclass(frozen=True, slots=True)
class RegistrationSummary:
    snapshot_id: str
    status: str
    built_at: datetime
    database_sha256: str
    identity_registry: str
    geographies: tuple[str, ...]
    years: tuple[int, ...]
    total_registrations: Decimal
    model_count: int
    model_year_available: bool
    release_ids: tuple[str, ...]


class RegistrationService:
    """Query canonical registration evidence without fixture fallback."""

    def __init__(self, database_path: Path, manifest: SnapshotManifest) -> None:
        self.database_path = database_path
        self.manifest = manifest

    @classmethod
    def from_candidate(cls, path: Path) -> RegistrationService:
        try:
            evidence = EvidenceReviewService.from_candidate(path)
            if evidence.manifest.versions.identity_registry != _IDENTITY_REGISTRY:
                raise ValueError("canonical identity registry is unavailable")
            return cls(evidence.database_path, evidence.manifest)
        except (OSError, RuntimeError, ValueError) as error:
            raise RegistrationUnavailableError(
                "canonical registration data is unavailable"
            ) from error

    def summary(self) -> RegistrationSummary:
        total, total_registrations = self._totals(search=None)
        with self._connect() as connection:
            release_ids = tuple(
                row["release_id"]
                for row in connection.execute(
                    """SELECT release_id FROM source_release
                    WHERE source_id = ? AND parser_name = ? AND publication_status = 'final'
                    ORDER BY release_id""",
                    (_EEA_SOURCE_ID, _EEA_PARSER_NAME),
                )
            )
        if not release_ids:
            raise RegistrationUnavailableError(
                "canonical registration data is unavailable"
            )
        return RegistrationSummary(
            snapshot_id=self.manifest.snapshot_id,
            status=self.manifest.status.value,
            built_at=self.manifest.built_at,
            database_sha256=self.manifest.database_sha256,
            identity_registry=self.manifest.versions.identity_registry,
            geographies=(_SUPPORTED_GEOGRAPHY,),
            years=(_SUPPORTED_YEAR,),
            total_registrations=total_registrations,
            model_count=total,
            model_year_available=False,
            release_ids=release_ids,
        )

    def ranking(self, query: RegistrationQuery) -> RegistrationPage:
        query.validate()
        search = query.search.strip() if query.search and query.search.strip() else None
        grouped_sql, parameters = self._grouped_query(search)
        total, total_registrations = self._totals(search)
        offset = (query.page - 1) * query.page_size
        with self._connect() as connection:
            rows = connection.execute(
                f"""WITH grouped AS ({grouped_sql}), ranked AS (
                    SELECT ROW_NUMBER() OVER (
                        ORDER BY registrations DESC, LOWER(make), LOWER(model), vehicle_id
                    ) AS rank, * FROM grouped
                )
                SELECT * FROM ranked ORDER BY rank LIMIT ? OFFSET ?""",
                (*parameters, query.page_size, offset),
            ).fetchall()
        items = tuple(_registration_row(row) for row in rows)
        pages = (total + query.page_size - 1) // query.page_size if total else 0
        return RegistrationPage(
            items=items,
            total=total,
            total_registrations=total_registrations,
            page=query.page,
            page_size=query.page_size,
            pages=pages,
            snapshot_id=self.manifest.snapshot_id,
        )

    def _totals(self, search: str | None) -> tuple[int, Decimal]:
        grouped_sql, parameters = self._grouped_query(search)
        with self._connect() as connection:
            row = connection.execute(
                f"""WITH grouped AS ({grouped_sql})
                SELECT COUNT(*) AS model_count,
                COALESCE(SUM(registrations), 0) AS total_registrations FROM grouped""",
                parameters,
            ).fetchone()
        return int(row["model_count"]), Decimal(str(row["total_registrations"]))

    def _grouped_query(self, search: str | None) -> tuple[str, tuple[object, ...]]:
        country_placeholders = ", ".join("?" for _ in _EU27_EEA_CODES)
        confidence = """CASE
            WHEN o.confidence_applied_cap IS NOT NULL AND o.confidence_applied_cap < (
                o.confidence_authority + o.confidence_publication_status +
                o.confidence_coverage + o.confidence_identity +
                o.confidence_independent_agreement
            ) THEN o.confidence_applied_cap
            ELSE o.confidence_authority + o.confidence_publication_status +
                o.confidence_coverage + o.confidence_identity +
                o.confidence_independent_agreement END"""
        clauses = [
            "r.source_id = ?",
            "r.parser_name = ?",
            "r.publication_status = 'final'",
            "o.publication_status = 'final'",
            "o.measure = 'new_registrations'",
            "o.unit = 'vehicles'",
            "o.period_start = '2024-01-01'",
            "o.period_end = '2024-12-31'",
            "o.mapping_status = 'normalized_label'",
            "m.status = o.mapping_status",
            "m.canonical_vehicle_id = o.canonical_vehicle_id",
            "v.model_year IS NULL",
            f"o.geography IN ({country_placeholders})",
        ]
        parameters: list[object] = [_EEA_SOURCE_ID, _EEA_PARSER_NAME, *_EU27_EEA_CODES]
        if search is not None:
            escaped = _escape_like(search.casefold())
            clauses.append(
                "(LOWER(v.make) LIKE ? ESCAPE '\\' OR LOWER(v.model) LIKE ? ESCAPE '\\')"
            )
            parameters.extend((f"%{escaped}%", f"%{escaped}%"))
        return (
            f"""SELECT v.vehicle_id, v.make, v.model,
            SUM(CAST(o.value AS NUMERIC)) AS registrations,
            MIN({confidence}) AS evidence_confidence,
            COUNT(o.observation_id) AS input_observation_count,
            GROUP_CONCAT(DISTINCT r.release_id) AS release_ids,
            GROUP_CONCAT(DISTINCT r.source_id) AS source_ids
            FROM observation o
            JOIN source_release r ON r.release_id = o.release_id
            JOIN canonical_vehicle v ON v.vehicle_id = o.canonical_vehicle_id
            JOIN identity_mapping m ON m.observation_id = o.observation_id
            WHERE {' AND '.join(clauses)}
            GROUP BY v.vehicle_id, v.make, v.model""",
            tuple(parameters),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"{self.database_path.resolve().as_uri()}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection


def _registration_row(row: sqlite3.Row) -> RegistrationRow:
    return RegistrationRow(
        rank=row["rank"],
        vehicle_id=row["vehicle_id"],
        make=row["make"],
        model=row["model"],
        model_year=None,
        registrations=Decimal(str(row["registrations"])),
        status="derived_observed",
        evidence_confidence=row["evidence_confidence"],
        input_observation_count=row["input_observation_count"],
        release_ids=_split_group(row["release_ids"]),
        source_ids=_split_group(row["source_ids"]),
    )


def _split_group(value: str) -> tuple[str, ...]:
    return tuple(sorted(value.split(",")))


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
