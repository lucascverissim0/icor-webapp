"""EEA finalized 2024 passenger-car CO2 monitoring adapter."""

from __future__ import annotations

import csv
import io
import sqlite3
import tempfile
from contextlib import closing
from datetime import date
from decimal import Decimal
from pathlib import Path
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
