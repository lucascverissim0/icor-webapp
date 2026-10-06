"""Finland Traficom open vehicle register: passenger cars in traffic, by cohort."""

from __future__ import annotations

import csv
import io
import re
import sqlite3
import tempfile
import unicodedata
import zipfile
from contextlib import closing
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

PARSER_NAME = "traficom_vehicles_zip_v1"
PASSENGER_CAR_CLASSES = frozenset({"M1", "M1G"})
EARLIEST_COHORT_YEAR = 1900
REQUIRED_COLUMNS = (
    "ajoneuvoluokka",
    "kayttoonottopvm",
    "merkkiSelvakielinen",
    "kaupallinenNimi",
    "mallimerkinta",
)
_FIRST_USE = re.compile(r"(\d{4})\d{4}\Z")
_INSERT_BATCH_SIZE = 20_000
_OBSERVATION_BATCH_SIZE = 2_000


class TraficomVehicleLoader:
    """Load the dated in-traffic passenger-car stock by first-use cohort year."""

    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        for release in releases:
            self._load_release(release, repository)

    def _load_release(
        self, release: StoredRelease, repository: SQLiteEvidenceRepository
    ) -> None:
        manifest = release.manifest
        if manifest.parser_name != PARSER_NAME:
            raise ValueError("Traficom vehicle parser name is unsupported")
        if manifest.publication_status is not PublicationStatus.FINAL:
            raise ValueError("Traficom vehicle release must be final")
        if manifest.geography != "FI" or manifest.measure is not Measure.ACTIVE_FLEET:
            raise ValueError("Traficom vehicle release geography or measure is unsupported")
        stock_date = manifest.coverage_end
        with tempfile.TemporaryDirectory(prefix="traficom-") as temporary:
            database = Path(temporary) / "aggregate.sqlite3"
            counts = self._aggregate(release.artifact_path, database, stock_date.year)
            if (*counts, 0) != (
                manifest.raw_record_count,
                manifest.accepted_record_count,
                manifest.rejected_record_count,
                manifest.quarantined_record_count,
            ):
                raise ValueError("Traficom vehicle parser counts do not match manifest")
            self._write_observations(release, stock_date, database, repository)

    @staticmethod
    def _aggregate(
        artifact: Path, database: Path, latest_cohort_year: int
    ) -> tuple[int, int, int]:
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                """CREATE TABLE aggregate (
                make TEXT NOT NULL, model TEXT NOT NULL, cohort INTEGER NOT NULL,
                vehicles INTEGER NOT NULL, first_row INTEGER NOT NULL,
                last_row INTEGER NOT NULL,
                PRIMARY KEY (make, model, cohort)) WITHOUT ROWID"""
            )
            statement = """INSERT INTO aggregate VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT DO UPDATE SET vehicles = vehicles + 1, last_row = excluded.last_row"""
            raw_count = accepted_count = rejected_count = 0
            batch: list[tuple[object, ...]] = []
            with zipfile.ZipFile(artifact) as archive:
                members = [
                    info
                    for info in archive.infolist()
                    if not info.is_dir() and info.filename.lower().endswith(".csv")
                ]
                if len(members) != 1:
                    raise ValueError("Traficom archive must contain exactly one CSV")
                with archive.open(members[0]) as binary:
                    stream = io.TextIOWrapper(binary, encoding="iso-8859-1", newline="")
                    reader = csv.reader(stream, delimiter=";")
                    header = next(reader, None)
                    if header is None or any(name not in header for name in REQUIRED_COLUMNS):
                        raise ValueError("Traficom vehicle CSV schema is unsupported")
                    position = {name: header.index(name) for name in REQUIRED_COLUMNS}
                    width = len(header)
                    for row_number, row in enumerate(reader, start=2):
                        if len(row) != width:
                            raise ValueError("Traficom vehicle CSV row is malformed")
                        raw_count += 1
                        key = _classify(row, position, latest_cohort_year)
                        if key is None:
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
    def _write_observations(
        release: StoredRelease,
        stock_date: date,
        database: Path,
        repository: SQLiteEvidenceRepository,
    ) -> None:
        batch: list[Observation] = []
        with closing(sqlite3.connect(database)) as connection:
            rows = connection.execute(
                """SELECT make, model, cohort, vehicles, first_row, last_row
                FROM aggregate ORDER BY make, model, cohort"""
            )
            for make, model, cohort, vehicles, first_row, last_row in rows:
                key = (make, model, str(cohort))
                group_id = stable_evidence_id("traficom-group", *key)
                batch.append(
                    Observation(
                        observation_id=stable_evidence_id(
                            "obs-traficom", release.release_id, *key
                        ),
                        release_id=release.release_id,
                        original_row_locator=f"{group_id}:rows-{first_row}-{last_row}",
                        geography="FI",
                        geography_version=release.manifest.geography_version,
                        period_start=stock_date,
                        period_end=stock_date,
                        period_precision=PeriodPrecision.DAY,
                        measure=Measure.ACTIVE_FLEET,
                        value=Decimal(vehicles),
                        unit="vehicles",
                        publication_status=PublicationStatus.FINAL,
                        original_make=make,
                        original_model=model,
                        original_model_year=None,
                        original_type=(
                            "Traficom commercial name (type designation when blank); "
                            "ajoneuvoluokka=M1 or M1G"
                        ),
                        source_make_identifier=make,
                        source_model_identifier=stable_evidence_id(
                            "traficom-model", make, model
                        ),
                        normalized_make=normalize_vehicle_label(make),
                        normalized_model=normalize_vehicle_label(model),
                        normalized_model_year=None,
                        canonical_vehicle_id=None,
                        mapping_status=MappingStatus.UNRESOLVED,
                        transformation_notes=(
                            "Aggregated Traficom passenger cars (classes M1 and M1G) in "
                            "traffic use to make, commercial name, and first-use year.",
                            "Cohort is the year of kayttoonottopvm, the first-use date; "
                            "rows without kaupallinenNimi are rejected.",
                            "Stock is the register on the data date, not a year end.",
                        ),
                        validation_flags=(),
                        evidence_confidence=_confidence(),
                        registration_cohort_year=cohort,
                    )
                )
                if len(batch) >= _OBSERVATION_BATCH_SIZE:
                    repository.add_observations(batch)
                    batch.clear()
        if batch:
            repository.add_observations(batch)


def _classify(
    row: list[str], position: dict[str, int], latest_cohort_year: int
) -> tuple[str, str, int] | None:
    if row[position["ajoneuvoluokka"]] not in PASSENGER_CAR_CLASSES:
        return None
    make = _collapse(row[position["merkkiSelvakielinen"]])
    # Only the commercial name identifies a model. When it is blank the type
    # designation (`mallimerkinta`) holds body-type strings such as
    # "2DSEDAN 1300-11/2400", which would mint junk vehicles, so the row is
    # rejected and counted rather than guessed.
    model = _collapse(row[position["kaupallinenNimi"]])
    if normalize_vehicle_label(make) is None or normalize_vehicle_label(model) is None:
        return None
    first_use = _FIRST_USE.match(row[position["kayttoonottopvm"]].strip())
    if first_use is None:
        return None
    cohort = int(first_use.group(1))
    if not EARLIEST_COHORT_YEAR <= cohort <= latest_cohort_year:
        return None
    return make, model, cohort


def _collapse(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _confidence() -> EvidenceConfidence:
    return EvidenceConfidence(
        authority=25,
        publication_status=10,
        coverage=25,
        identity=5,
        independent_agreement=10,
        reasons=(
            "Official Traficom administrative register of vehicles in traffic use.",
            "Traficom commercial-name identity is retained without an unreviewed alias.",
            "Agreement component is neutral until cross-source overlap is evaluated.",
        ),
    )
