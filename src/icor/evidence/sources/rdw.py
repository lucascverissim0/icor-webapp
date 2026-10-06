"""Netherlands RDW register of currently registered passenger cars, by cohort."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

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

PARSER_NAME = "rdw_registered_cars_csv_v1"
SCHEMA = ("merk", "handelsbenaming", "cohort", "vehicles")
EARLIEST_COHORT_YEAR = 1900
_OBSERVATION_BATCH_SIZE = 2_000


@dataclass(frozen=True, slots=True)
class RDWGroup:
    make: str
    model: str
    cohort_year: int
    vehicles: int
    row_number: int


def classify_group(row: dict[str, str], *, latest_cohort_year: int) -> RDWGroup | None:
    """Return the accepted group, or None when the row cannot be used as evidence."""

    make = row["merk"]
    model = row["handelsbenaming"]
    cohort = row["cohort"]
    vehicles = row["vehicles"]
    if normalize_vehicle_label(make) is None or normalize_vehicle_label(model) is None:
        return None
    if not _positive_integer(cohort) or not _positive_integer(vehicles):
        return None
    cohort_year = int(cohort)
    if not EARLIEST_COHORT_YEAR <= cohort_year <= latest_cohort_year:
        return None
    return RDWGroup(make, model, cohort_year, int(vehicles), 0)


def _positive_integer(value: str) -> bool:
    return value.isascii() and value.isdigit() and int(value) > 0


class RDWRegisteredCarsLoader:
    """Load the dated active passenger-car stock by first-admission cohort year."""

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
            raise ValueError("RDW registered-cars parser name is unsupported")
        if manifest.publication_status is not PublicationStatus.FINAL:
            raise ValueError("RDW registered-cars release must be final")
        if manifest.geography != "NL" or manifest.measure is not Measure.ACTIVE_FLEET:
            raise ValueError("RDW registered-cars release geography or measure is unsupported")
        stock_date = manifest.coverage_end
        raw_count, rejected_count, groups = self._read(release, stock_date.year)
        if (raw_count, len(groups), rejected_count, 0) != (
            manifest.raw_record_count,
            manifest.accepted_record_count,
            manifest.rejected_record_count,
            manifest.quarantined_record_count,
        ):
            raise ValueError("RDW registered-cars parser counts do not match manifest")
        self._write_observations(release, stock_date, groups, repository)

    @staticmethod
    def _read(
        release: StoredRelease, latest_cohort_year: int
    ) -> tuple[int, int, list[RDWGroup]]:
        raw_count = rejected_count = 0
        groups: list[RDWGroup] = []
        seen: set[tuple[str, str, int]] = set()
        with release.artifact_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != SCHEMA:
                raise ValueError("RDW registered-cars CSV schema is unsupported")
            for row_number, row in enumerate(reader, start=2):
                raw_count += 1
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("RDW registered-cars CSV row is malformed")
                group = classify_group(row, latest_cohort_year=latest_cohort_year)
                if group is None:
                    rejected_count += 1
                    continue
                key = (group.make, group.model, group.cohort_year)
                if key in seen:
                    raise ValueError("RDW registered-cars CSV contains a duplicate group")
                seen.add(key)
                groups.append(
                    RDWGroup(
                        group.make,
                        group.model,
                        group.cohort_year,
                        group.vehicles,
                        row_number,
                    )
                )
        return raw_count, rejected_count, groups

    @staticmethod
    def _write_observations(
        release: StoredRelease,
        stock_date: date,
        groups: list[RDWGroup],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        batch: list[Observation] = []
        for group in groups:
            key = (group.make, group.model, str(group.cohort_year))
            group_id = stable_evidence_id("rdw-group", *key)
            batch.append(
                Observation(
                    observation_id=stable_evidence_id("obs-rdw", release.release_id, *key),
                    release_id=release.release_id,
                    original_row_locator=f"{group_id}:row-{group.row_number}",
                    geography="NL",
                    geography_version=release.manifest.geography_version,
                    period_start=stock_date,
                    period_end=stock_date,
                    period_precision=PeriodPrecision.DAY,
                    measure=Measure.ACTIVE_FLEET,
                    value=Decimal(group.vehicles),
                    unit="vehicles",
                    publication_status=PublicationStatus.FINAL,
                    original_make=group.make,
                    original_model=group.model,
                    original_model_year=None,
                    original_type=(
                        "RDW commercial name; voertuigsoort=Personenauto; "
                        "tenaamstellen_mogelijk=Ja"
                    ),
                    source_make_identifier=group.make,
                    source_model_identifier=stable_evidence_id(
                        "rdw-model", group.make, group.model
                    ),
                    normalized_make=normalize_vehicle_label(group.make),
                    normalized_model=normalize_vehicle_label(group.model),
                    normalized_model_year=None,
                    canonical_vehicle_id=None,
                    mapping_status=MappingStatus.UNRESOLVED,
                    transformation_notes=(
                        "Aggregated RDW registered passenger cars to merk, "
                        "handelsbenaming, and first-admission year.",
                        "Cohort is the year of datum_eerste_toelating_dt, the first "
                        "admission anywhere, not necessarily first Dutch registration.",
                        "Stock is the register on the retrieval date, not a year end.",
                    ),
                    validation_flags=(),
                    evidence_confidence=_confidence(),
                    registration_cohort_year=group.cohort_year,
                )
            )
            if len(batch) >= _OBSERVATION_BATCH_SIZE:
                repository.add_observations(batch)
                batch.clear()
        if batch:
            repository.add_observations(batch)


def _confidence() -> EvidenceConfidence:
    return EvidenceConfidence(
        authority=25,
        publication_status=10,
        coverage=25,
        identity=5,
        independent_agreement=10,
        reasons=(
            "Official RDW administrative register of currently registered vehicles.",
            "RDW commercial-name identity is retained without an unreviewed alias.",
            "Agreement component is neutral until cross-source overlap is evaluated.",
        ),
    )
