"""Read-only queries for one verified source-evidence candidate snapshot."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from icor.domain.evidence import MappingStatus, Measure
from icor.domain.snapshots import SnapshotManifest, SnapshotStatus, SnapshotVersions
from icor.evidence.release_manifests import load_snapshot_manifest
from icor.evidence.serialization import sha256_file
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

_YEAR_SEMANTICS = {
    "observation_year",
    "registration_cohort_year",
    "manufacture_year",
    "model_year",
}
_OBSERVATION_RESPONSE_COLUMNS = """observation_id, release_id, original_row_locator,
    geography, period_start, period_end, period_precision, measure, value, unit,
    publication_status, original_make, original_model, original_model_year, original_type,
    mapping_status, transformation_notes, validation_flags, confidence_authority,
    confidence_publication_status, confidence_coverage, confidence_identity,
    confidence_independent_agreement, confidence_reasons, confidence_applied_cap,
    registration_cohort_year, manufacture_year, model_year"""


@dataclass(frozen=True, slots=True)
class EvidenceReleaseSummary:
    release_id: str
    source_id: str
    publisher: str
    source_url: str
    terms_url: str
    published_at: datetime
    coverage_start: date
    coverage_end: date
    geography: str
    measure: str
    dependency_group: str
    raw_record_count: int
    accepted_record_count: int
    rejected_record_count: int
    quarantined_record_count: int
    observation_count: int
    total_value: Decimal
    publication_status: str = "final"
    validation_warning_count: int = 0
    what_it_proves: str = ""
    limitations: str = ""


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    snapshot_id: str
    status: str
    built_at: datetime
    database_sha256: str
    observation_count: int
    published_value_count: int
    warning_count: int
    versions: SnapshotVersions
    releases: tuple[EvidenceReleaseSummary, ...]
    mapping_status_counts: dict[str, int]
    geographies: tuple[str, ...]
    measures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceObservationQuery:
    release_id: str | None = None
    geography: str | None = None
    measure: str | None = None
    mapping_status: str | None = None
    search: str | None = None
    observation_year: int | None = None
    year_semantics: str | None = None
    page: int = 1
    page_size: int = 25

    def __post_init__(self) -> None:
        if type(self.page) is not int or self.page < 1:
            raise ValueError("page must be positive")
        if type(self.page_size) is not int or not 1 <= self.page_size <= 100:
            raise ValueError("page size must be between 1 and 100")
        if self.search is not None and (type(self.search) is not str or len(self.search) > 100):
            raise ValueError("search must be at most 100 characters")
        if self.measure is not None and self.measure not in {item.value for item in Measure}:
            raise ValueError("measure is unsupported")
        if self.mapping_status is not None and self.mapping_status not in {
            item.value for item in MappingStatus
        }:
            raise ValueError("mapping status is unsupported")
        if self.observation_year is not None and (
            type(self.observation_year) is not int
            or not 1900 <= self.observation_year <= 2200
        ):
            raise ValueError("observation year is unsupported")
        if self.year_semantics is not None and self.year_semantics not in _YEAR_SEMANTICS:
            raise ValueError("year semantics is unsupported")


@dataclass(frozen=True, slots=True)
class EvidenceObservationRow:
    observation_id: str
    release_id: str
    original_row_locator: str
    geography: str
    period_start: date
    period_end: date
    period_precision: str
    measure: str
    value: Decimal
    unit: str
    publication_status: str
    original_make: str
    original_model: str
    original_model_year: str | None
    original_type: str | None
    mapping_status: str
    transformation_notes: tuple[str, ...]
    validation_flags: tuple[str, ...]
    confidence_total: int
    confidence_reasons: tuple[str, ...]
    observation_year: int
    registration_cohort_year: int | None
    manufacture_year: int | None
    model_year: int | None


@dataclass(frozen=True, slots=True)
class EvidenceObservationPage:
    items: tuple[EvidenceObservationRow, ...]
    total: int
    page: int
    page_size: int
    pages: int


class EvidenceReviewService:
    """Query a sealed candidate without making it an active product snapshot."""

    def __init__(self, candidate_path: Path) -> None:
        self.candidate_path = candidate_path
        self.database_path = candidate_path / "evidence.sqlite3"
        self.manifest = load_snapshot_manifest(candidate_path / "snapshot.json")
        self._observation_total_cache: dict[
            tuple[str, tuple[object, ...]], int
        ] = {}

    @classmethod
    def from_snapshot(
        cls, database_path: Path, manifest: SnapshotManifest
    ) -> EvidenceReviewService:
        service = object.__new__(cls)
        service.candidate_path = Path(database_path).parent
        service.database_path = Path(database_path)
        service.manifest = manifest
        service._observation_total_cache = {}
        return service

    @classmethod
    def from_candidate(cls, path: Path) -> EvidenceReviewService:
        candidate = Path(path).resolve(strict=True)
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError("evidence candidate is invalid")
        database = candidate / "evidence.sqlite3"
        manifest_path = candidate / "snapshot.json"
        if any(item.is_symlink() or not item.is_file() for item in (database, manifest_path)):
            raise ValueError("evidence candidate is incomplete")
        service = cls(candidate)
        if service.manifest.status is not SnapshotStatus.CANDIDATE:
            raise ValueError("evidence snapshot must be a candidate")
        if service.manifest.snapshot_id != candidate.name:
            raise ValueError("evidence candidate identity does not match its directory")
        if sha256_file(database) != service.manifest.database_sha256:
            raise ValueError("evidence candidate checksum does not match its manifest")

        repository = SQLiteEvidenceRepository(database)
        releases = repository.list_releases()
        with service._connect() as connection:
            observation_count = int(
                connection.execute("SELECT COUNT(*) FROM observation").fetchone()[0]
            )
            published_count = int(
                connection.execute("SELECT COUNT(*) FROM published_value").fetchone()[0]
            )
        if observation_count != service.manifest.observation_count:
            raise ValueError("evidence observation count does not match its manifest")
        if published_count != service.manifest.published_value_count:
            raise ValueError("evidence published-value count does not match its manifest")
        if (
            tuple(sorted(release.release_id for release in releases))
            != service.manifest.release_ids
        ):
            raise ValueError("evidence release membership does not match its manifest")
        return service

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.database_path.as_posix()}?mode=ro", uri=True, check_same_thread=False
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    def summary(self) -> EvidenceSummary:
        with self._connect() as connection:
            release_rows = connection.execute(
                """SELECT r.*, s.observation_count, s.total_value,
                s.mapping_status_counts, s.validation_flag_counts,
                s.geographies, s.measures, s.observation_years
                FROM source_release r
                JOIN evidence_release_summary s ON s.release_id = r.release_id
                ORDER BY r.release_id"""
            ).fetchall()
        mapping_status_counts: dict[str, int] = {}
        geographies: set[str] = set()
        measures: set[str] = set()
        releases = tuple(
            EvidenceReleaseSummary(
                release_id=row["release_id"],
                source_id=row["source_id"],
                publisher=row["publisher"],
                source_url=row["source_url"],
                terms_url=row["terms_url"],
                published_at=datetime.fromisoformat(row["published_at"]),
                coverage_start=date.fromisoformat(row["coverage_start"]),
                coverage_end=date.fromisoformat(row["coverage_end"]),
                geography=row["geography"],
                measure=row["measure"],
                dependency_group=row["dependency_group"],
                raw_record_count=row["raw_record_count"],
                accepted_record_count=row["accepted_record_count"],
                rejected_record_count=row["rejected_record_count"],
                quarantined_record_count=row["quarantined_record_count"],
                observation_count=row["observation_count"],
                total_value=Decimal(str(row["total_value"])),
                publication_status=row["publication_status"],
                validation_warning_count=sum(
                    _json_count_mapping(row["validation_flag_counts"]).values()
                ),
                what_it_proves=_source_explanation(row["measure"])[0],
                limitations=_source_explanation(row["measure"])[1],
            )
            for row in release_rows
        )
        for row in release_rows:
            for status, count in _json_count_mapping(row["mapping_status_counts"]).items():
                mapping_status_counts[status] = mapping_status_counts.get(status, 0) + count
            geographies.update(_json_tuple(row["geographies"]))
            measures.update(_json_tuple(row["measures"]))
        return EvidenceSummary(
            snapshot_id=self.manifest.snapshot_id,
            status=self.manifest.status.value,
            built_at=self.manifest.built_at,
            database_sha256=self.manifest.database_sha256,
            observation_count=self.manifest.observation_count,
            published_value_count=self.manifest.published_value_count,
            warning_count=len(self.manifest.warnings),
            versions=self.manifest.versions,
            releases=releases,
            mapping_status_counts=mapping_status_counts,
            geographies=tuple(sorted(geographies)),
            measures=tuple(sorted(measures)),
        )

    def list_observations(self, query: EvidenceObservationQuery) -> EvidenceObservationPage:
        clauses: list[str] = []
        parameters: list[object] = []
        for column, value in (
            ("release_id", query.release_id),
            ("geography", query.geography),
            ("measure", query.measure),
            ("mapping_status", query.mapping_status),
        ):
            if value:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if query.observation_year is not None:
            clauses.append("period_end BETWEEN ? AND ?")
            parameters.extend(
                (
                    f"{query.observation_year:04d}-01-01",
                    f"{query.observation_year:04d}-12-31",
                )
            )
        if query.year_semantics and query.year_semantics != "observation_year":
            clauses.append(f"{query.year_semantics} IS NOT NULL")
        if query.search and query.search.strip():
            escaped = _escape_like(query.search.strip().casefold())
            clauses.append(
                "(LOWER(original_make) LIKE ? ESCAPE '\\' OR "
                "LOWER(original_model) LIKE ? ESCAPE '\\')"
            )
            parameters.extend((f"%{escaped}%", f"%{escaped}%"))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        offset = (query.page - 1) * query.page_size
        query_parameters = tuple(parameters)
        order_by = (
            "observation_id"
            if query.search and query.search.strip()
            else "release_id, geography, original_make, original_model, period_end, observation_id"
        )
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT {_OBSERVATION_RESPONSE_COLUMNS} FROM observation{where}
                ORDER BY {order_by} LIMIT ? OFFSET ?""",
                (*query_parameters, query.page_size, offset),
            ).fetchall()
        total = self._observation_total(where, query_parameters)
        items = tuple(_observation_row(row) for row in rows)
        pages = (total + query.page_size - 1) // query.page_size if total else 0
        return EvidenceObservationPage(items, total, query.page, query.page_size, pages)

    def _observation_total(self, where: str, parameters: tuple[object, ...]) -> int:
        key = where, parameters
        cached = self._observation_total_cache.get(key)
        if cached is not None:
            return cached
        with self._connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM observation{where}", parameters
                ).fetchone()[0]
            )
        if len(self._observation_total_cache) >= 256:
            self._observation_total_cache.pop(next(iter(self._observation_total_cache)))
        self._observation_total_cache[key] = total
        return total


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _json_tuple(value: str) -> tuple[str, ...]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError("evidence JSON field is invalid")
    return tuple(parsed)


def _json_count_mapping(value: str) -> dict[str, int]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or any(
        not isinstance(key, str) or type(count) is not int
        for key, count in parsed.items()
    ):
        raise ValueError("evidence count JSON field is invalid")
    return parsed


def _source_explanation(measure: str) -> tuple[str, str]:
    if measure == Measure.NEW_REGISTRATIONS.value:
        return (
            "Counts newly registered vehicles in the release's stated place and period.",
            "Registration year does not prove model year, manufacture year, or windshield fitment.",
        )
    if measure == Measure.ACTIVE_FLEET.value:
        return (
            "Counts licensed or registered stock observed in the release period.",
            "Stock is not annual registrations; cohort years describe vehicle age, "
            "not observation time.",
        )
    return (
        "Reports the publisher's stated vehicle measure and coverage.",
        "The source does not prove facts outside its stated measure, geography, and period.",
    )


def _observation_row(row: sqlite3.Row) -> EvidenceObservationRow:
    confidence_total = sum(
        row[column]
        for column in (
            "confidence_authority",
            "confidence_publication_status",
            "confidence_coverage",
            "confidence_identity",
            "confidence_independent_agreement",
        )
    )
    if row["confidence_applied_cap"] is not None:
        confidence_total = min(confidence_total, row["confidence_applied_cap"])
    return EvidenceObservationRow(
        observation_id=row["observation_id"],
        release_id=row["release_id"],
        original_row_locator=row["original_row_locator"],
        geography=row["geography"],
        period_start=date.fromisoformat(row["period_start"]),
        period_end=date.fromisoformat(row["period_end"]),
        period_precision=row["period_precision"],
        measure=row["measure"],
        value=Decimal(row["value"]),
        unit=row["unit"],
        publication_status=row["publication_status"],
        original_make=row["original_make"],
        original_model=row["original_model"],
        original_model_year=row["original_model_year"],
        original_type=row["original_type"],
        mapping_status=row["mapping_status"],
        transformation_notes=_json_tuple(row["transformation_notes"]),
        validation_flags=_json_tuple(row["validation_flags"]),
        confidence_total=confidence_total,
        confidence_reasons=_json_tuple(row["confidence_reasons"]),
        observation_year=date.fromisoformat(row["period_end"]).year,
        registration_cohort_year=row["registration_cohort_year"],
        manufacture_year=row["manufacture_year"],
        model_year=row["model_year"],
    )
