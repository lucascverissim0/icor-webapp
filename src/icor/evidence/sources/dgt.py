"""Spain DGT annual register of vehicles on the road, passenger cars by cohort."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from icor.domain.evidence import (
    STOCK_LOWER_BOUND_FLAG,
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

PARSER_NAME = "dgt_parque_cohorts_csv_v1"
SCHEMA = ("marca", "modelo", "cohort", "vehicles")
#: DGT replaces a rare make or model name with this character.
MASK = "¡"
EARLIEST_COHORT_YEAR = 1900
# DGT masks whole models at a time (27% of passenger cars), so a cohort's count
# is only a lower bound. From 2010 EEA registrations name every model sold in
# Spain, and a lower bound would replace them; before 2010 nothing else exists.
LATEST_COHORT_YEAR = 2009
_OBSERVATION_BATCH_SIZE = 2_000


@dataclass(frozen=True, slots=True)
class DGTGroup:
    make: str
    model: str
    cohort_year: int
    vehicles: int
    row_number: int


def classify_group(row: dict[str, str]) -> DGTGroup | None:
    """Return the accepted group, or None when the row cannot be used as evidence."""

    make = row["marca"]
    model = row["modelo"]
    cohort = row["cohort"]
    vehicles = row["vehicles"]
    if MASK in make or MASK in model:
        return None
    if normalize_vehicle_label(make) is None or normalize_vehicle_label(model) is None:
        return None
    if not _positive_integer(cohort) or not _positive_integer(vehicles):
        return None
    cohort_year = int(cohort)
    if not EARLIEST_COHORT_YEAR <= cohort_year <= LATEST_COHORT_YEAR:
        return None
    return DGTGroup(make, model, cohort_year, int(vehicles), 0)


def _positive_integer(value: str) -> bool:
    return value.isascii() and value.isdigit() and int(value) > 0


class DGTParqueLoader:
    """Load the year-end passenger-car stock by first-registration cohort year."""

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
            raise ValueError("DGT parque parser name is unsupported")
        if manifest.publication_status is not PublicationStatus.FINAL:
            raise ValueError("DGT parque release must be final")
        if manifest.geography != "ES" or manifest.measure is not Measure.ACTIVE_FLEET:
            raise ValueError("DGT parque release geography or measure is unsupported")
        stock_date = manifest.coverage_end
        if (stock_date.month, stock_date.day) != (12, 31):
            raise ValueError("DGT parque release must end on a year end")
        raw_count, rejected_count, groups = self._read(release)
        if (raw_count, len(groups), rejected_count, 0) != (
            manifest.raw_record_count,
            manifest.accepted_record_count,
            manifest.rejected_record_count,
            manifest.quarantined_record_count,
        ):
            raise ValueError("DGT parque parser counts do not match manifest")
        self._write_observations(release, stock_date, groups, repository)

    @staticmethod
    def _read(release: StoredRelease) -> tuple[int, int, list[DGTGroup]]:
        raw_count = rejected_count = 0
        groups: list[DGTGroup] = []
        seen: set[tuple[str, str, int]] = set()
        with release.artifact_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != SCHEMA:
                raise ValueError("DGT parque CSV schema is unsupported")
            for row_number, row in enumerate(reader, start=2):
                raw_count += 1
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("DGT parque CSV row is malformed")
                group = classify_group(row)
                if group is None:
                    rejected_count += 1
                    continue
                key = (group.make, group.model, group.cohort_year)
                if key in seen:
                    raise ValueError("DGT parque CSV contains a duplicate group")
                seen.add(key)
                groups.append(
                    DGTGroup(
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
        groups: list[DGTGroup],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        batch: list[Observation] = []
        for group in groups:
            key = (group.make, group.model, str(group.cohort_year))
            group_id = stable_evidence_id("dgt-group", *key)
            batch.append(
                Observation(
                    observation_id=stable_evidence_id("obs-dgt", release.release_id, *key),
                    release_id=release.release_id,
                    original_row_locator=f"{group_id}:row-{group.row_number}",
                    geography="ES",
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
                        "DGT MODELO (commercial name); CAT_EURO M1 or M1G, or "
                        "TIPO_DGT TURISMOS without a category"
                    ),
                    source_make_identifier=group.make,
                    source_model_identifier=stable_evidence_id(
                        "dgt-model", group.make, group.model
                    ),
                    normalized_make=normalize_vehicle_label(group.make),
                    normalized_model=normalize_vehicle_label(group.model),
                    normalized_model_year=None,
                    canonical_vehicle_id=None,
                    mapping_status=MappingStatus.UNRESOLVED,
                    transformation_notes=(
                        "Aggregated DGT parque passenger cars to MARCA, MODELO, and the "
                        "year of FEC_PRIM_MATR, the first registration anywhere.",
                        "DGT masks rare make and model names; masked cars are excluded, "
                        "so the count is a lower bound.",
                        f"Only cohorts up to {LATEST_COHORT_YEAR} are used; EEA "
                        "registrations cover Spain from 2010.",
                    ),
                    validation_flags=(STOCK_LOWER_BOUND_FLAG,),
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
        coverage=15,
        identity=5,
        independent_agreement=10,
        reasons=(
            "Official DGT administrative register of vehicles not deregistered at year end.",
            "DGT masks rare model names, so a cohort's count is a lower bound.",
            "DGT commercial-name identity is retained without an unreviewed alias.",
            "Agreement component is neutral until cross-source overlap is evaluated.",
        ),
    )
