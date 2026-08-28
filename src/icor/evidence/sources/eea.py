"""EEA finalized 2024 passenger-car CO2 monitoring adapter."""

from __future__ import annotations

import csv
import io
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import closing
from datetime import date
from decimal import Decimal
from pathlib import Path
from re import fullmatch
from zipfile import BadZipFile, ZipFile

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

PARSER_NAME = "eea_co2_cars_zip_v1"
ANNUAL_AGGREGATE_PARSER_NAME = "eea_co2_cars_annual_aggregate_csv_v1"
ANNUAL_AGGREGATE_SCHEMA = (
    "Year",
    "Status",
    "Version_file",
    "MS",
    "Mk",
    "Cn",
    "TAN",
    "T",
    "Va",
    "Ve",
    "Ft",
    "Registrations",
    "SourceRows",
)
EXPECTED_MEMBER = "co2cars_2024fv30.csv"
EXPECTED_SCHEMA = (
    "ID",
    "MS",
    "Mp",
    "VFN",
    "Mh",
    "Man",
    "MMS",
    "TAN",
    "T",
    "Va",
    "Ve",
    "Mk",
    "Cn",
    "Ct",
    "Cr",
    "M_kg_",
    "Mt",
    "Enedc_g_km_",
    "Ewltp_g_km_",
    "W_mm_",
    "At1_mm_",
    "At2_mm_",
    "Ft",
    "Fm",
    "Ec_cm3_",
    "Ep_KW_",
    "Z_Wh_km_",
    "IT",
    "Ernedc_g_km_",
    "Erwltp_g_km_",
    "De",
    "Vf",
    "R",
    "Year",
    "Status",
    "Version_file",
    "E_g_km_",
    "Er_g_km_",
    "Zr",
    "Dr",
    "Fc",
    "Ech",
    "RLFI",
)
_GROUP_COLUMNS = ("MS", "Mk", "Cn", "TAN", "T", "Va", "Ve", "Ft")
_INSERT_BATCH_SIZE = 5_000
_OBSERVATION_BATCH_SIZE = 2_000


class EEAAnnualAggregateLoader:
    """Load deterministic official SQL aggregates for one finalized EEA year."""

    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        for release in releases:
            self._load_release(release, repository)

    def _load_release(
        self,
        release: StoredRelease,
        repository: SQLiteEvidenceRepository,
    ) -> None:
        manifest = release.manifest
        match = fullmatch(r"eea-co2cars-(20\d{2})-final-(v\d+)", manifest.release_id)
        if match is None:
            raise ValueError("EEA annual aggregate release ID is unsupported")
        year, version = int(match.group(1)), match.group(2)
        if manifest.parser_name != ANNUAL_AGGREGATE_PARSER_NAME:
            raise ValueError("EEA annual aggregate parser name is unsupported")
        if manifest.publication_status is not PublicationStatus.FINAL:
            raise ValueError("EEA annual aggregate release must have final status")
        if manifest.coverage_start != date(year, 1, 1) or manifest.coverage_end != date(
            year, 12, 31
        ):
            raise ValueError("EEA annual aggregate must cover its release calendar year")

        raw_count = accepted_count = rejected_count = 0
        for _, source_rows, _, key in self._validated_rows(release.artifact_path, year, version):
            raw_count += source_rows
            if any(normalize_vehicle_label(value) is None for value in key[:3]):
                rejected_count += source_rows
            else:
                accepted_count += source_rows

        if (raw_count, accepted_count, rejected_count, 0) != (
            manifest.raw_record_count,
            manifest.accepted_record_count,
            manifest.rejected_record_count,
            manifest.quarantined_record_count,
        ):
            raise ValueError("EEA annual aggregate parser counts do not match manifest")

        with tempfile.TemporaryDirectory(prefix="eea-annual-model-") as temporary:
            aggregate_path = Path(temporary) / "aggregate.sqlite3"
            self._aggregate_models(release.artifact_path, year, version, aggregate_path)
            self._write_annual_observations(release, year, aggregate_path, repository)

    def _aggregate_models(
        self, artifact_path: Path, year: int, version: str, database_path: Path
    ) -> None:
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                """CREATE TABLE aggregate (
                country TEXT NOT NULL, make TEXT NOT NULL, model TEXT NOT NULL,
                first_group INTEGER NOT NULL, last_group INTEGER NOT NULL,
                technical_groups INTEGER NOT NULL, source_rows INTEGER NOT NULL,
                registrations INTEGER NOT NULL,
                PRIMARY KEY (country, make, model)
                ) WITHOUT ROWID"""
            )
            statement = """INSERT INTO aggregate VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT DO UPDATE SET last_group = excluded.last_group,
            technical_groups = technical_groups + 1,
            source_rows = source_rows + excluded.source_rows,
            registrations = registrations + excluded.registrations"""
            batch: list[tuple[object, ...]] = []
            for group_number, source_rows, registrations, key in self._validated_rows(
                artifact_path, year, version
            ):
                if any(normalize_vehicle_label(value) is None for value in key[:3]):
                    continue
                country, make, model = key[:3]
                batch.append(
                    (
                        country,
                        make,
                        model,
                        group_number,
                        group_number,
                        source_rows,
                        registrations,
                    )
                )
                if len(batch) >= _INSERT_BATCH_SIZE:
                    connection.executemany(statement, batch)
                    batch.clear()
            if batch:
                connection.executemany(statement, batch)
            connection.commit()

    def _write_annual_observations(
        self,
        release: StoredRelease,
        year: int,
        database_path: Path,
        repository: SQLiteEvidenceRepository,
    ) -> None:
        batch: list[Observation] = []
        with closing(sqlite3.connect(database_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM aggregate ORDER BY country, make, model"
            )
            for row in rows:
                country, make, model = row["country"], row["make"], row["model"]
                batch.append(
                    Observation(
                        observation_id=stable_evidence_id(
                            "obs-eea-annual", release.release_id, country, make, model
                        ),
                        release_id=release.release_id,
                        original_row_locator=(
                            f"official-sql-groups-{row['first_group']}-{row['last_group']}:"
                            f"technical-groups-{row['technical_groups']}:"
                            f"source-rows-{row['source_rows']}"
                        ),
                        geography=country,
                        geography_version=release.manifest.geography_version,
                        period_start=date(year, 1, 1),
                        period_end=date(year, 12, 31),
                        period_precision=PeriodPrecision.YEAR,
                        measure=Measure.NEW_REGISTRATIONS,
                        value=Decimal(row["registrations"]),
                        unit="vehicles",
                        publication_status=PublicationStatus.FINAL,
                        original_make=make,
                        original_model=model,
                        original_model_year=None,
                        original_type=None,
                        source_make_identifier=make,
                        source_model_identifier=stable_evidence_id(
                            "eea-model", make, model
                        ),
                        normalized_make=normalize_vehicle_label(make),
                        normalized_model=normalize_vehicle_label(model),
                        normalized_model_year=None,
                        canonical_vehicle_id=None,
                        mapping_status=MappingStatus.UNRESOLVED,
                        transformation_notes=(
                            "Canonical aggregate exported from the official EEA Discodata SQL API.",
                            f"Contributing source rows: {row['source_rows']}.",
                            f"Contributing technical-key groups: {row['technical_groups']}.",
                            "Snapshot observation aggregated on Year, MS, Mk, and Cn; "
                            "the immutable artifact retains TAN, T, Va, Ve, and Ft detail.",
                        ),
                        validation_flags=(),
                        evidence_confidence=_unresolved_confidence(),
                        registration_cohort_year=year,
                        manufacture_year=None,
                        model_year=None,
                    )
                )
                if len(batch) >= _OBSERVATION_BATCH_SIZE:
                    repository.add_observations(batch)
                    batch.clear()
        if batch:
            repository.add_observations(batch)

    def _validated_rows(
        self, artifact_path: Path, year: int, version: str
    ) -> Iterator[tuple[int, int, int, tuple[str, ...]]]:
        with artifact_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=";")
            if tuple(reader.fieldnames or ()) != ANNUAL_AGGREGATE_SCHEMA:
                raise ValueError("EEA annual aggregate CSV schema is unsupported")
            for group_number, row in enumerate(reader, start=1):
                self._validate_annual_row(row, year=year, version=version)
                source_rows = _parse_nonnegative_integer(row["SourceRows"], "source rows")
                if source_rows == 0:
                    raise ValueError("EEA annual aggregate source rows must be positive")
                registrations = _parse_nonnegative_integer(
                    row["Registrations"], "registration weight"
                )
                key = tuple((row[column] or "").strip() for column in _GROUP_COLUMNS)
                yield group_number, source_rows, registrations, key

    @staticmethod
    def _validate_annual_row(
        row: dict[str, str | None], *, year: int, version: str
    ) -> None:
        if row.get("Status") != "F":
            raise ValueError("EEA annual aggregate row does not have final status")
        if row.get("Year") != str(year):
            raise ValueError("EEA annual aggregate row is outside its release year")
        if row.get("Version_file") != version:
            raise ValueError("EEA annual aggregate row version is unexpected")


def _parse_nonnegative_integer(value: str | None, label: str) -> int:
    if value is None or fullmatch(r"0|[1-9]\d*", value) is None:
        raise ValueError(f"EEA annual aggregate {label} must be a non-negative integer")
    return int(value)


class EEAPassengerCarLoader:
    """Aggregate immutable EEA registration rows on documented technical keys."""

    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        for release in releases:
            self._load_release(release, repository)

    def _load_release(self, release: StoredRelease, repository: SQLiteEvidenceRepository) -> None:
        manifest = release.manifest
        if manifest.parser_name != PARSER_NAME:
            raise ValueError("EEA release parser name is unsupported")
        if manifest.publication_status is not PublicationStatus.FINAL:
            raise ValueError("EEA release must have final status")
        if manifest.coverage_start != date(2024, 1, 1) or manifest.coverage_end != date(
            2024, 12, 31
        ):
            raise ValueError("EEA release must cover calendar year 2024")

        try:
            with ZipFile(release.artifact_path) as archive:
                members = archive.infolist()
                if len(members) != 1 or members[0].filename.casefold() != EXPECTED_MEMBER:
                    raise ValueError("EEA archive member is unexpected")
                member = members[0]
                if member.is_dir() or member.flag_bits & 0x1:
                    raise ValueError("EEA archive member is unsupported")
                with tempfile.TemporaryDirectory(prefix="eea-aggregate-") as temporary:
                    aggregate_path = Path(temporary) / "aggregate.sqlite3"
                    counts = self._aggregate(archive, member.filename, aggregate_path)
                    if (*counts, 0) != (
                        manifest.raw_record_count,
                        manifest.accepted_record_count,
                        manifest.rejected_record_count,
                        manifest.quarantined_record_count,
                    ):
                        raise ValueError("EEA parser counts do not match manifest")
                    self._write_observations(release, aggregate_path, repository)
        except BadZipFile as error:
            raise ValueError("EEA artifact is not a valid ZIP archive") from error

    def _aggregate(
        self, archive: ZipFile, member: str, database_path: Path
    ) -> tuple[int, int, int]:
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                """CREATE TABLE aggregate (
                country TEXT NOT NULL, make TEXT NOT NULL, model TEXT NOT NULL,
                type_approval TEXT NOT NULL, vehicle_type TEXT NOT NULL,
                variant TEXT NOT NULL, version TEXT NOT NULL, fuel TEXT NOT NULL,
                first_row INTEGER NOT NULL, last_row INTEGER NOT NULL,
                registrations INTEGER NOT NULL,
                PRIMARY KEY (
                    country, make, model, type_approval, vehicle_type,
                    variant, version, fuel
                )
                ) WITHOUT ROWID"""
            )
            statement = """INSERT INTO aggregate VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT DO UPDATE SET last_row = excluded.last_row,
            registrations = registrations + 1"""
            batch: list[tuple[object, ...]] = []
            raw_count = accepted_count = rejected_count = 0
            with (
                archive.open(member) as binary,
                io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text,
            ):
                reader = csv.DictReader(text, delimiter=";")
                if tuple(reader.fieldnames or ()) != EXPECTED_SCHEMA:
                    raise ValueError("EEA CSV schema is unsupported")
                for row_number, row in enumerate(reader, start=2):
                    raw_count += 1
                    self._validate_row(row)
                    key = tuple((row[column] or "").strip() for column in _GROUP_COLUMNS)
                    if any(normalize_vehicle_label(value) is None for value in key[:3]):
                        rejected_count += 1
                        continue
                    accepted_count += 1
                    batch.append((*key, row_number, row_number))
                    if len(batch) >= _INSERT_BATCH_SIZE:
                        connection.executemany(statement, batch)
                        batch.clear()
                if batch:
                    connection.executemany(statement, batch)
            connection.commit()
            return raw_count, accepted_count, rejected_count

    @staticmethod
    def _validate_row(row: dict[str, str | None]) -> None:
        if row.get("Status") != "F":
            raise ValueError("EEA row does not have final status")
        if row.get("Year") != "2024":
            raise ValueError("EEA row is outside 2024")
        if row.get("Version_file") != "v30":
            raise ValueError("EEA row version is not v30")
        if row.get("R") != "1":
            raise ValueError("EEA registration weight must equal one")

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
                """SELECT * FROM aggregate ORDER BY country, make, model,
                type_approval, vehicle_type, variant, version, fuel"""
            )
            for row in rows:
                key = tuple(
                    str(row[column])
                    for column in (
                        "country",
                        "make",
                        "model",
                        "type_approval",
                        "vehicle_type",
                        "variant",
                        "version",
                        "fuel",
                    )
                )
                country, make, model, type_approval, vehicle_type, variant, version, fuel = key
                group_id = stable_evidence_id("eea-group", *key)
                original_type = "|".join((type_approval, vehicle_type, variant, version, fuel))
                batch.append(
                    Observation(
                        observation_id=stable_evidence_id("obs-eea", release.release_id, *key),
                        release_id=release.release_id,
                        original_row_locator=(
                            f"{group_id}:rows-{row['first_row']}-{row['last_row']}:members-{row['registrations']}"
                        ),
                        geography=country,
                        geography_version=release.manifest.geography_version,
                        period_start=date(2024, 1, 1),
                        period_end=date(2024, 12, 31),
                        period_precision=PeriodPrecision.YEAR,
                        measure=Measure.NEW_REGISTRATIONS,
                        value=Decimal(int(row["registrations"])),
                        unit="vehicles",
                        publication_status=PublicationStatus.FINAL,
                        original_make=make,
                        original_model=model,
                        original_model_year=None,
                        original_type=original_type,
                        source_make_identifier=make,
                        source_model_identifier=stable_evidence_id(
                            "eea-model", make, model, type_approval, vehicle_type, variant, version
                        ),
                        normalized_make=normalize_vehicle_label(make),
                        normalized_model=normalize_vehicle_label(model),
                        normalized_model_year=None,
                        canonical_vehicle_id=None,
                        mapping_status=MappingStatus.UNRESOLVED,
                        transformation_notes=(
                            "Aggregated finalized registration rows on MS, Mk, Cn, "
                            "TAN, T, Va, Ve, and Ft.",
                            f"Contributing raw rows: {row['registrations']}.",
                        ),
                        validation_flags=(),
                        evidence_confidence=_unresolved_confidence(),
                        registration_cohort_year=2024,
                        manufacture_year=None,
                        model_year=None,
                    )
                )
                if len(batch) >= _OBSERVATION_BATCH_SIZE:
                    repository.add_observations(batch)
                    batch.clear()
        if batch:
            repository.add_observations(batch)


def _unresolved_confidence() -> EvidenceConfidence:
    return EvidenceConfidence(
        authority=25,
        publication_status=10,
        coverage=25,
        identity=0,
        independent_agreement=10,
        reasons=(
            "Official finalized EEA/DG CLIMA administrative registration evidence.",
            "Canonical model-year identity is unresolved; excluded from model-level publication.",
            "Agreement component is neutral until dependency-aware overlap is evaluated.",
        ),
    )
