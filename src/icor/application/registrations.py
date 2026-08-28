"""Read-only official registration ranking application service."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from icor.application.evidence_review import EvidenceReviewService
from icor.domain.snapshots import SnapshotManifest, SnapshotVersions
from icor.infrastructure.snapshot_store import SnapshotStore

_EEA_SOURCE_ID = "eea-co2-monitoring"
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
    geography: str = "EU27"
    year: int = 2024
    search: str | None = None
    page: int = 1
    page_size: int = 25

    def validate(self) -> None:
        if type(self.geography) is not str or not self.geography.strip():
            raise ValueError("registration geography is required")
        if type(self.year) is not int or not 1900 <= self.year <= 2200:
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
    versions: SnapshotVersions | None = None


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

    @classmethod
    def from_active(cls, root: Path) -> RegistrationService:
        try:
            manifest, repository = SnapshotStore(root).open_active_snapshot()
            if manifest.versions.identity_registry != _IDENTITY_REGISTRY:
                raise ValueError("canonical identity registry is unavailable")
            return cls(repository.path, manifest)
        except (OSError, RuntimeError, ValueError) as error:
            raise RegistrationUnavailableError(
                "canonical registration data is unavailable"
            ) from error

    def summary(self) -> RegistrationSummary:
        with self._connect() as connection:
            years, geographies = self._scope(connection)
            if not years:
                raise RegistrationUnavailableError(
                    "canonical registration data is unavailable"
                )
            latest_year = max(years)
            total, total_registrations = self._totals(
                connection, "EU27", latest_year, search=None
            )
            release_ids = tuple(
                row["release_id"]
                for row in connection.execute(
                    """SELECT release_id FROM source_release
                    WHERE source_id = ? AND publication_status = 'final'
                    ORDER BY release_id""",
                    (_EEA_SOURCE_ID,),
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
            geographies=("EU27", *geographies),
            years=years,
            total_registrations=total_registrations,
            model_count=total,
            model_year_available=False,
            release_ids=release_ids,
            versions=self.manifest.versions,
        )

    def ranking(self, query: RegistrationQuery) -> RegistrationPage:
        query.validate()
        search = query.search.strip() if query.search and query.search.strip() else None
        grouped_sql, parameters = self._grouped_query(
            query.geography, query.year, search
        )
        offset = (query.page - 1) * query.page_size
        with self._connect() as connection:
            years, geographies = self._scope(connection)
            if query.year not in years or (
                query.geography != "EU27" and query.geography not in geographies
            ):
                raise RegistrationUnavailableError(
                    "requested registration scope is unavailable"
                )
            rows = connection.execute(
                f"""WITH grouped AS ({grouped_sql}), ranked AS (
                    SELECT ROW_NUMBER() OVER (
                        ORDER BY registrations DESC, LOWER(make), LOWER(model), vehicle_id
                    ) AS rank,
                    COUNT(*) OVER () AS total_count,
                    SUM(registrations) OVER () AS complete_total_registrations,
                    * FROM grouped
                )
                SELECT * FROM ranked ORDER BY rank LIMIT ? OFFSET ?""",
                (*parameters, query.page_size, offset),
            ).fetchall()
            if rows:
                total = int(rows[0]["total_count"])
                total_registrations = Decimal(
                    str(rows[0]["complete_total_registrations"])
                )
            else:
                total, total_registrations = self._totals(
                    connection, query.geography, query.year, search
                )
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

    def _totals(
        self,
        connection: sqlite3.Connection,
        geography: str,
        year: int,
        search: str | None,
    ) -> tuple[int, Decimal]:
        grouped_sql, parameters = self._grouped_query(geography, year, search)
        row = connection.execute(
            f"""WITH grouped AS ({grouped_sql})
            SELECT COUNT(*) AS model_count,
            COALESCE(SUM(registrations), 0) AS total_registrations FROM grouped""",
            parameters,
        ).fetchone()
        return int(row["model_count"]), Decimal(str(row["total_registrations"]))

    def _grouped_query(
        self, geography: str, year: int, search: str | None
    ) -> tuple[str, tuple[object, ...]]:
        country_codes = _EU27_EEA_CODES if geography == "EU27" else (geography,)
        country_placeholders = ", ".join("?" for _ in country_codes)
        confidence = """CASE
            WHEN o.confidence_applied_cap IS NOT NULL AND o.confidence_applied_cap < (
                o.confidence_authority + o.confidence_publication_status +
                o.confidence_coverage + o.confidence_identity +
                o.confidence_independent_agreement
            ) THEN o.confidence_applied_cap
            ELSE o.confidence_authority + o.confidence_publication_status +
                o.confidence_coverage + o.confidence_identity +
                o.confidence_independent_agreement END"""
        observation_clauses = [
            "o.release_id IN (SELECT release_id FROM source_release "
            "WHERE source_id = ? AND publication_status = 'final')",
            "o.publication_status = 'final'",
            "o.measure = 'new_registrations'",
            "o.unit = 'vehicles'",
            "o.period_start = ?",
            "o.period_end = ?",
            "o.mapping_status = 'normalized_label'",
            f"o.geography IN ({country_placeholders})",
        ]
        vehicle_clauses = ["v.model_year IS NULL"]
        parameters: list[object] = [
            _EEA_SOURCE_ID,
            _EEA_SOURCE_ID,
            f"{year:04d}-01-01",
            f"{year:04d}-12-31",
            *country_codes,
        ]
        if search is not None:
            escaped = _escape_like(search.casefold())
            vehicle_clauses.append(
                "(LOWER(v.make) LIKE ? ESCAPE '\\' OR LOWER(v.model) LIKE ? ESCAPE '\\')"
            )
            parameters.extend((f"%{escaped}%", f"%{escaped}%"))
        return (
            f"""SELECT v.vehicle_id, v.make, v.model, grouped.registrations,
            grouped.evidence_confidence, grouped.input_observation_count,
            grouped.release_ids, ? AS source_ids
            FROM (
                SELECT o.canonical_vehicle_id,
                SUM(CAST(o.value AS NUMERIC)) AS registrations,
                MIN({confidence}) AS evidence_confidence,
                COUNT(o.observation_id) AS input_observation_count,
                GROUP_CONCAT(DISTINCT o.release_id) AS release_ids
                FROM observation o
                WHERE {' AND '.join(observation_clauses)}
                GROUP BY o.canonical_vehicle_id
            ) AS grouped
            JOIN canonical_vehicle v
                ON v.vehicle_id = grouped.canonical_vehicle_id
            WHERE {' AND '.join(vehicle_clauses)}""",
            tuple(parameters),
        )

    def _scope(
        self, connection: sqlite3.Connection
    ) -> tuple[tuple[int, ...], tuple[str, ...]]:
        years = tuple(
            int(row["year"])
            for row in connection.execute(
                """SELECT DISTINCT CAST(SUBSTR(o.period_end, 1, 4) AS INTEGER) AS year
                FROM observation o JOIN source_release r ON r.release_id = o.release_id
                WHERE r.source_id = ? AND r.publication_status = 'final'
                AND o.measure = 'new_registrations' ORDER BY year""",
                (_EEA_SOURCE_ID,),
            )
        )
        geographies = tuple(
            row["geography"]
            for row in connection.execute(
                """SELECT DISTINCT o.geography FROM observation o
                JOIN source_release r ON r.release_id = o.release_id
                WHERE r.source_id = ? AND r.publication_status = 'final'
                AND o.measure = 'new_registrations' ORDER BY o.geography""",
                (_EEA_SOURCE_ID,),
            )
        )
        return years, geographies

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
