"""Read-only official registration ranking application service."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from icor.application.evidence_review import EvidenceReviewService
from icor.domain.evidence import CanonicalVehicle
from icor.domain.snapshots import SnapshotManifest, SnapshotVersions
from icor.generations.public_catalog import official_public_generation_catalog
from icor.infrastructure.snapshot_store import SnapshotStore

_EEA_SOURCE_ID = "eea-co2-monitoring"
_IDENTITY_REGISTRY = "exact-normalized-model-family-v1"
_GENERATION_REGISTRY = "public-generation-registry-v1"
_PUBLIC_GENERATION_CATALOG = official_public_generation_catalog()


class RegistrationUnavailableError(RuntimeError):
    """A verified canonical registration snapshot is unavailable."""

    code = 'registration_data_unavailable'

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


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
class RegistrationLabelBreakdown:
    source_make: str
    source_model: str
    registrations: Decimal
    input_observation_count: int
    release_ids: tuple[str, ...]
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RegistrationRow:
    rank: int
    vehicle_id: str
    make: str
    model: str
    model_year: int
    model_year_basis: str
    generation_name: str | None
    generation_basis: str
    generation_confidence: str
    generation_source_url: str | None
    registrations: Decimal
    status: str
    evidence_confidence: int
    input_observation_count: int
    release_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    publication_status: str = 'final'
    evidence_kind: str = 'observed'
    label_breakdown: tuple[RegistrationLabelBreakdown, ...] = ()


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
class RegistrationAvailability:
    geography: str
    year: int
    status: str
    evidence_kind: str


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
    availability: tuple[RegistrationAvailability, ...] = ()
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
            availability = self._availability(connection)
            years, geographies = self._projection_scope(connection)
            if not years:
                raise RegistrationUnavailableError(
                    "canonical registration data is unavailable"
                )
            eu27_years = [item.year for item in availability if item.geography == 'EU27']
            if not eu27_years:
                raise RegistrationUnavailableError(
                    'canonical registration data is unavailable'
                )
            latest_year = max(eu27_years)
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
            availability=availability,
            versions=self.manifest.versions,
        )

    def ranking(self, query: RegistrationQuery) -> RegistrationPage:
        query.validate()
        search = query.search.strip() if query.search and query.search.strip() else None
        grouped_sql, parameters = self._projected_query(
            query.geography, query.year, search
        )
        offset = (query.page - 1) * query.page_size
        breakdowns: dict[str, tuple[RegistrationLabelBreakdown, ...]] = {}
        with self._connect() as connection:
            years, geographies = self._projection_scope(connection)
            if not self._has_scope(connection, query.geography, query.year):
                raise RegistrationUnavailableError(
                    'requested registration scope is unavailable',
                    code='scope_unavailable',
                )
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
            breakdowns = self._label_breakdowns(connection, query, rows)
        items = tuple(
            _registration_row(row, query.year, breakdowns.get(row['vehicle_id'], ()))
            for row in rows
        )
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

    def _label_breakdowns(
        self,
        connection: sqlite3.Connection,
        query: RegistrationQuery,
        families: list[sqlite3.Row],
    ) -> dict[str, tuple[RegistrationLabelBreakdown, ...]]:
        if not families:
            return {}
        placeholders = ', '.join('?' for _ in families)
        first = families[0]
        rows = connection.execute(
            f'''SELECT * FROM registration_label_aggregate
            WHERE geography = ? AND year = ? AND publication_status = ?
            AND evidence_kind = ? AND family_vehicle_id IN ({placeholders})
            ORDER BY family_vehicle_id, LOWER(source_make), LOWER(source_model)''',
            (
                query.geography, query.year, first['publication_status'],
                first['evidence_kind'], *(row['vehicle_id'] for row in families),
            ),
        )
        return _group_label_rows(rows)

    def _totals(
        self,
        connection: sqlite3.Connection,
        geography: str,
        year: int,
        search: str | None,
    ) -> tuple[int, Decimal]:
        grouped_sql, parameters = self._projected_query(geography, year, search)
        row = connection.execute(
            f"""WITH grouped AS ({grouped_sql})
            SELECT COUNT(*) AS model_count,
            COALESCE(SUM(registrations), 0) AS total_registrations FROM grouped""",
            parameters,
        ).fetchone()
        return int(row["model_count"]), Decimal(str(row["total_registrations"]))

    def _projected_query(
        self, geography: str, year: int, search: str | None
    ) -> tuple[str, tuple[object, ...]]:
        clauses = ['f.geography = ?', 'f.year = ?']
        parameters: list[object] = [geography, year]
        for token in _search_tokens(search or '') if search else ():
            escaped = _escape_like(token)
            clauses.append(
                '(LOWER(f.make) LIKE ? ESCAPE \'\\\' OR LOWER(f.model) LIKE ? ESCAPE \'\\\')'
            )
            parameters.extend((f'%{escaped}%', f'%{escaped}%'))
        return self._projection_sql(' AND '.join(clauses)), tuple(parameters)

    @staticmethod
    def _projection_sql(clauses: str) -> str:
        return f'''SELECT f.family_vehicle_id AS vehicle_id, f.make, f.model,
            CAST(f.registrations AS NUMERIC) AS registrations,
            f.evidence_confidence, f.input_observation_count,
            f.release_ids, f.source_ids, f.publication_status, f.evidence_kind
        FROM registration_family_aggregate f
        WHERE {clauses}
        AND (f.publication_status, f.evidence_kind) = (
            SELECT x.publication_status, x.evidence_kind
            FROM registration_family_aggregate x
            WHERE x.geography = f.geography AND x.year = f.year
            ORDER BY CASE x.publication_status
                WHEN 'final' THEN 0 WHEN 'provisional' THEN 1 ELSE 2 END,
                CASE x.evidence_kind WHEN 'observed' THEN 0 ELSE 1 END
            LIMIT 1
        )'''

    @staticmethod
    def _has_scope(connection: sqlite3.Connection, geography: str, year: int) -> bool:
        return connection.execute(
            'SELECT 1 FROM registration_family_aggregate WHERE geography = ? AND year = ? LIMIT 1',
            (geography, year),
        ).fetchone() is not None

    def _availability(
        self, connection: sqlite3.Connection
    ) -> tuple[RegistrationAvailability, ...]:
        rows = connection.execute(
            '''SELECT DISTINCT geography, year, publication_status, evidence_kind
            FROM registration_family_aggregate
            ORDER BY geography, year, publication_status, evidence_kind'''
        )
        return tuple(
            RegistrationAvailability(
                row['geography'], row['year'], row['publication_status'], row['evidence_kind']
            )
            for row in rows
        )

    def _projection_scope(
        self, connection: sqlite3.Connection
    ) -> tuple[tuple[int, ...], tuple[str, ...]]:
        availability = self._availability(connection)
        if not availability:
            return (), ()
        latest = max(item.year for item in availability)
        years = tuple(range(2000, latest + 1))
        countries = sorted({item.geography for item in availability} - {'EU27'})
        return years, tuple(countries)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"{self.database_path.resolve().as_uri()}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection


def _group_label_rows(
    rows: Iterable[sqlite3.Row],
) -> dict[str, tuple[RegistrationLabelBreakdown, ...]]:
    grouped: dict[str, list[RegistrationLabelBreakdown]] = {}
    for row in rows:
        grouped.setdefault(row['family_vehicle_id'], []).append(
            RegistrationLabelBreakdown(
                source_make=row['source_make'],
                source_model=row['source_model'],
                registrations=Decimal(str(row['registrations'])),
                input_observation_count=row['input_observation_count'],
                release_ids=_split_group(row['release_ids']),
                source_ids=_split_group(row['source_ids']),
            )
        )
    return {key: tuple(value) for key, value in grouped.items()}


def _registration_row(
    row: sqlite3.Row,
    registration_year: int,
    labels: tuple[RegistrationLabelBreakdown, ...] = (),
) -> RegistrationRow:
    vehicle = CanonicalVehicle(
        vehicle_id=row["vehicle_id"],
        make=row["make"],
        model=row["model"],
        model_year=None,
        market="Europe",
    )
    generation = _PUBLIC_GENERATION_CATALOG.entry_for_year(
        vehicle,
        registration_year,
        registry_version=_GENERATION_REGISTRY,
    )
    return RegistrationRow(
        rank=row["rank"],
        vehicle_id=row["vehicle_id"],
        make=row["make"],
        model=row["model"],
        model_year=registration_year,
        model_year_basis="registration_year_proxy",
        generation_name=generation.display_name if generation is not None else None,
        generation_basis=(
            "manufacturer_generation_window"
            if generation is not None
            else "registration_year_proxy"
        ),
        generation_confidence="high" if generation is not None else "low",
        generation_source_url=(
            generation.evidence_ids[0] if generation is not None else None
        ),
        registrations=Decimal(str(row["registrations"])),
        status="derived_observed",
        evidence_confidence=row["evidence_confidence"],
        input_observation_count=row["input_observation_count"],
        release_ids=_split_group(row["release_ids"]),
        source_ids=_split_group(row["source_ids"]),
        publication_status=row['publication_status'],
        evidence_kind=row['evidence_kind'],
        label_breakdown=labels,
    )


def _split_group(value: str) -> tuple[str, ...]:
    parsed = json.loads(value)
    return tuple(sorted(parsed if isinstance(parsed, list) else value.split(",")))


def _search_tokens(value: str) -> tuple[str, ...]:
    normalized = value.casefold().strip()
    tokens = tuple(re.findall(r"[^\W_]+", normalized))
    return tokens or (normalized,)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
