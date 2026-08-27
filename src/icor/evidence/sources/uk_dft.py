"""UK DfT/DVLA generic-model registration and licensed-fleet adapters."""

from __future__ import annotations

import calendar
import csv
import re
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from icor.domain.evidence import (
    EvidenceConfidence,
    MappingStatus,
    Measure,
    Observation,
    PeriodPrecision,
    PublicationStatus,
)
from icor.evidence.normalization import normalize_vehicle_label, stable_evidence_id
from icor.infrastructure.release_store import StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

_QUARTER = re.compile(r"(20\d{2}) Q([1-4])\Z")
_FINAL_YEAR = 2025
_INSERT_BATCH_SIZE = 5_000
_OBSERVATION_BATCH_SIZE = 2_000


@dataclass(frozen=True, slots=True)
class _SourceContract:
    parser_name: str
    base_columns: tuple[str, ...]
    measure: Measure
    earliest_year: int
    licensed_only: bool


_REGISTRATIONS = _SourceContract(
    parser_name="uk_dft_veh0160_csv_v1",
    base_columns=("BodyType", "Make", "GenModel", "Model", "Fuel"),
    measure=Measure.NEW_REGISTRATIONS,
    earliest_year=2001,
    licensed_only=False,
)
_ACTIVE_FLEET = _SourceContract(
    parser_name="uk_dft_veh0120_csv_v1",
    base_columns=("BodyType", "Make", "GenModel", "Model", "Fuel", "LicenceStatus"),
    measure=Measure.ACTIVE_FLEET,
    earliest_year=1994,
    licensed_only=True,
)


class UKFirstRegistrationLoader:
    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        _UKWideCSVLoader(_REGISTRATIONS).load(releases, repository)


class UKActiveFleetLoader:
    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        _UKWideCSVLoader(_ACTIVE_FLEET).load(releases, repository)


class _UKWideCSVLoader:
    def __init__(self, contract: _SourceContract) -> None:
        self.contract = contract

    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        for release in releases:
            self._load_release(release, repository)

    def _load_release(self, release: StoredRelease, repository: SQLiteEvidenceRepository) -> None:
        manifest = release.manifest
        if manifest.parser_name != self.contract.parser_name:
            raise ValueError("UK DfT release parser name is unsupported")
        if manifest.publication_status is not PublicationStatus.FINAL:
            raise ValueError("UK DfT release must be final")
        if manifest.coverage_end != date(_FINAL_YEAR, 12, 31):
            raise ValueError("UK DfT release must end at finalized 2025 Q4")

        with tempfile.TemporaryDirectory(
            prefix="uk-dft-aggregate-", dir=release.artifact_path.parent
        ) as temporary:
            aggregate_path = Path(temporary) / "aggregate.sqlite3"
            raw, accepted, rejected = self._aggregate(release.artifact_path, aggregate_path)
            if (raw, accepted, rejected, 0) != (
                manifest.raw_record_count,
                manifest.accepted_record_count,
                manifest.rejected_record_count,
                manifest.quarantined_record_count,
            ):
                raise ValueError("UK DfT parser counts do not match manifest")
            self._write_observations(release, aggregate_path, repository)

    def _aggregate(self, artifact: Path, database_path: Path) -> tuple[int, int, int]:
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                """CREATE TABLE aggregate (
                make TEXT NOT NULL, generic_model TEXT NOT NULL,
                period TEXT NOT NULL, registrations INTEGER NOT NULL,
                blocked INTEGER NOT NULL, first_row INTEGER NOT NULL,
                last_row INTEGER NOT NULL, member_rows INTEGER NOT NULL,
                PRIMARY KEY (make, generic_model, period)
                ) WITHOUT ROWID"""
            )
            statement = """INSERT INTO aggregate VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT DO UPDATE SET
                registrations = registrations + excluded.registrations,
                blocked = MAX(blocked, excluded.blocked),
                last_row = excluded.last_row,
                member_rows = member_rows + 1"""
            raw_count = accepted_count = rejected_count = 0
            batch: list[tuple[object, ...]] = []
            with artifact.open("r", encoding="cp1252", newline="") as stream:
                reader = csv.DictReader(stream)
                fieldnames = tuple(reader.fieldnames or ())
                quarter_columns = self._validate_schema(fieldnames)
                for row_number, row in enumerate(reader, start=2):
                    raw_count += 1
                    if not self._accept_row(row):
                        rejected_count += 1
                        continue
                    accepted_count += 1
                    make = _required_label(row.get("Make"), "make")
                    model = _required_label(row.get("GenModel"), "generic model")
                    for period in quarter_columns:
                        registrations, blocked = _cell_value(row.get(period))
                        batch.append(
                            (
                                make,
                                model,
                                period,
                                registrations,
                                blocked,
                                row_number,
                                row_number,
                            )
                        )
                        if len(batch) >= _INSERT_BATCH_SIZE:
                            connection.executemany(statement, batch)
                            batch.clear()
                if batch:
                    connection.executemany(statement, batch)
            connection.commit()
            return raw_count, accepted_count, rejected_count

    def _validate_schema(self, fieldnames: tuple[str, ...]) -> tuple[str, ...]:
        base_length = len(self.contract.base_columns)
        if fieldnames[:base_length] != self.contract.base_columns:
            raise ValueError("UK DfT CSV schema is unsupported")
        periods: list[str] = []
        for field in fieldnames[base_length:]:
            match = _QUARTER.fullmatch(field)
            if match is None:
                raise ValueError("UK DfT quarter column is invalid")
            year = int(match.group(1))
            if self.contract.earliest_year <= year <= _FINAL_YEAR:
                periods.append(field)
        if not periods or f"{_FINAL_YEAR} Q4" not in periods:
            raise ValueError("UK DfT finalized quarter coverage is missing")
        return tuple(periods)

    def _accept_row(self, row: dict[str, str | None]) -> bool:
        if row.get("BodyType") != "Cars":
            return False
        return not self.contract.licensed_only or row.get("LicenceStatus") == "Licensed"

    def _write_observations(
        self,
        release: StoredRelease,
        database_path: Path,
        repository: SQLiteEvidenceRepository,
    ) -> None:
        batch: list[Observation] = []
        with closing(sqlite3.connect(database_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM aggregate ORDER BY make, generic_model, period"
            )
            for row in rows:
                if row["blocked"] or row["registrations"] == 0:
                    continue
                period_start, period_end = _quarter_dates(row["period"])
                make = str(row["make"])
                model = str(row["generic_model"])
                status_note = "; LicenceStatus=Licensed" if self.contract.licensed_only else ""
                key = (make, model, row["period"])
                group_id = stable_evidence_id("uk-dft-group", *key)
                batch.append(
                    Observation(
                        observation_id=stable_evidence_id("obs-uk", release.release_id, *key),
                        release_id=release.release_id,
                        original_row_locator=(
                            f"{group_id}:rows-{row['first_row']}-{row['last_row']}:"
                            f"members-{row['member_rows']}"
                        ),
                        geography="GB",
                        geography_version=release.manifest.geography_version,
                        period_start=(
                            period_end
                            if self.contract.measure is Measure.ACTIVE_FLEET
                            else period_start
                        ),
                        period_end=period_end,
                        period_precision=PeriodPrecision.QUARTER,
                        measure=self.contract.measure,
                        value=Decimal(int(row["registrations"])),
                        unit="vehicles",
                        publication_status=PublicationStatus.FINAL,
                        original_make=make,
                        original_model=model,
                        original_model_year=None,
                        original_type=(
                            f"DfT generic model; detailed models and fuels aggregated{status_note}"
                        ),
                        source_make_identifier=make,
                        source_model_identifier=stable_evidence_id("uk-model", make, model),
                        normalized_make=normalize_vehicle_label(make),
                        normalized_model=normalize_vehicle_label(model),
                        normalized_model_year=None,
                        canonical_vehicle_id=None,
                        mapping_status=MappingStatus.UNRESOLVED,
                        transformation_notes=(
                            "Aggregated DfT detailed model and fuel rows to Make, "
                            "GenModel, and quarter.",
                            "Source columns after 2025 Q4 were excluded as provisional.",
                        ),
                        validation_flags=(),
                        evidence_confidence=_confidence(),
                    )
                )
                if len(batch) >= _OBSERVATION_BATCH_SIZE:
                    repository.add_observations(batch)
                    batch.clear()
        if batch:
            repository.add_observations(batch)


def _required_label(value: str | None, label: str) -> str:
    if value is None or normalize_vehicle_label(value) is None:
        raise ValueError(f"UK DfT row is missing {label}")
    return " ".join(value.split())


def _cell_value(value: str | None) -> tuple[int, int]:
    if value == "[c]" or value in {"[x]", "[z]", "", None}:
        return 0, 1
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("UK DfT vehicle count is invalid") from error
    if parsed < 0:
        raise ValueError("UK DfT vehicle count is invalid")
    return parsed, 0


def _quarter_dates(period: str) -> tuple[date, date]:
    match = _QUARTER.fullmatch(period)
    if match is None:
        raise ValueError("UK DfT quarter is invalid")
    year = int(match.group(1))
    quarter = int(match.group(2))
    start_month = 1 + (quarter - 1) * 3
    end_month = start_month + 2
    return (
        date(year, start_month, 1),
        date(year, end_month, calendar.monthrange(year, end_month)[1]),
    )


def _confidence() -> EvidenceConfidence:
    return EvidenceConfidence(
        authority=25,
        publication_status=10,
        coverage=25,
        identity=5,
        independent_agreement=10,
        reasons=(
            "Official finalized DfT/DVLA administrative vehicle evidence.",
            "DfT generic-model identity is retained without an unreviewed alias.",
            "Agreement component is neutral until cross-source overlap is evaluated.",
        ),
    )
