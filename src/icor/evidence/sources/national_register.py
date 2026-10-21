"""National registers of cars on the road, as counts by make, model and cohort.

Every register is acquired into the same counts-only aggregate
(`icor.evidence.register_aggregate`): make, model, the year of first
registration anywhere, and vehicles. What differs between registers is kept in
one reviewed profile each: the market, the labels it uses for a masked car, and
whether its counts are complete. A register that hides some cars (masked rare
models, or many cars without a first-registration year) is only a lower bound:
its cohorts are flagged, and planning keeps it per vehicle only where it counts
more than the registrations surviving to its date. Every register's labels are
flagged too, so that they never teach the identity vocabulary the reviewed
vehicles fold by.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from icor.domain.evidence import (
    STOCK_LOWER_BOUND_FLAG,
    VOCABULARY_EXCLUDED_FLAG,
    EvidenceConfidence,
    MappingStatus,
    Measure,
    Observation,
    PeriodPrecision,
    PublicationStatus,
)
from icor.evidence.normalization import normalize_vehicle_label, stable_evidence_id
from icor.evidence.register_aggregate import SCHEMA
from icor.infrastructure.release_store import StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

PARSER_NAME = "national_register_cohorts_csv_v1"
#: Register cohorts the model uses: twenty years before the newest registrations.
EARLIEST_COHORT_YEAR = 2006
_OBSERVATION_BATCH_SIZE = 2_000


@dataclass(frozen=True, slots=True)
class RegisterProfile:
    source_id: str
    geography: str
    publisher: str
    #: What the register's first-registration year means, and the filters applied.
    definition: str
    #: Normalized labels the register writes in place of a hidden make or model.
    masks: frozenset[str] = frozenset()
    lower_bound: bool = False


#: Reviewed registers. A profile is added only after its register passed the
#: reconciliation, identity and completeness gates recorded in the handoff
#: (scripts/measure_register_gates.py).
PROFILES: dict[str, RegisterProfile] = {
    profile.source_id: profile
    for profile in (
        RegisterProfile(
            source_id="snca-parc-automobile",
            geography="LU",
            publisher="SNCA Luxembourg",
            definition=(
                "the year of DATCIRPRM, the first registration anywhere (CATEU M1 or M1G; "
                "every vehicle in circulation)"
            ),
            # Placeholders the register writes for a missing name.
            masks=frozenset({"*", "--", "§"}),
        ),
        RegisterProfile(
            source_id="regitra-tp-parko-duomenys",
            geography="LT",
            publisher="Regitra Lithuania",
            definition=(
                "the year of PIRM_REG_DATA, the first registration anywhere "
                "(KATEGORIJA_KLASE M1, in public traffic or not, as Eurostat counts "
                "Lithuania's fleet)"
            ),
            # A private car unique by make, model and municipality is depersonalised.
            # Masked cars are 1.5% of 2010+ cohorts and rare by construction; counting
            # the register only as a lower bound would instead keep new cars registered
            # in Lithuania for export (45,288 Fiat 500s registered, 562 on the road).
            masks=frozenset({"nuasmeninta"}),
        ),
        RegisterProfile(
            source_id="transpordiamet-soidukite-staatused",
            geography="EE",
            publisher="Transpordiamet Estonia",
            definition=(
                "the year of ESMANE_REG, the first registration anywhere (KATEGOORIA M1 or "
                "M1G, registered or suspended, as Eurostat counts Estonia's fleet)"
            ),
        ),
    )
}


@dataclass(frozen=True, slots=True)
class RegisterGroup:
    make: str
    model: str
    cohort_year: int
    vehicles: int
    row_number: int


def classify_group(
    row: dict[str, str], profile: RegisterProfile, row_number: int = 0
) -> RegisterGroup | None:
    """The accepted group, or None when the row is not evidence the model can use."""

    make, model = row["make"], row["model"]
    normalized_make = normalize_vehicle_label(make)
    normalized_model = normalize_vehicle_label(model)
    if normalized_make is None or normalized_model is None:
        return None
    if normalized_make in profile.masks or normalized_model in profile.masks:
        return None
    cohort, vehicles = row["cohort"], row["vehicles"]
    if not (_positive_integer(cohort) and _positive_integer(vehicles)):
        return None
    cohort_year = int(cohort)
    if cohort_year < EARLIEST_COHORT_YEAR:
        return None
    return RegisterGroup(make, model, cohort_year, int(vehicles), row_number)


def _positive_integer(value: str) -> bool:
    return value.isascii() and value.isdigit() and int(value) > 0


class NationalRegisterLoader:
    """Load a reviewed register's cars on the road at its date, by cohort."""

    def __init__(self, profiles: dict[str, RegisterProfile] | None = None) -> None:
        self._profiles = PROFILES if profiles is None else profiles

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
            raise ValueError("national register parser name is unsupported")
        profile = self._profiles.get(manifest.source_id)
        if profile is None:
            raise ValueError(f"no reviewed register profile for {manifest.source_id}")
        if manifest.publication_status is not PublicationStatus.FINAL:
            raise ValueError("national register release must be final")
        if manifest.geography != profile.geography or manifest.measure is not Measure.ACTIVE_FLEET:
            raise ValueError("national register release geography or measure is unsupported")
        raw_count, rejected_count, groups = self._read(release, profile)
        if (raw_count, len(groups), rejected_count, 0) != (
            manifest.raw_record_count,
            manifest.accepted_record_count,
            manifest.rejected_record_count,
            manifest.quarantined_record_count,
        ):
            raise ValueError("national register parser counts do not match manifest")
        self._write_observations(release, profile, manifest.coverage_end, groups, repository)

    @staticmethod
    def _read(
        release: StoredRelease, profile: RegisterProfile
    ) -> tuple[int, int, list[RegisterGroup]]:
        raw_count = rejected_count = 0
        groups: list[RegisterGroup] = []
        seen: set[tuple[str, str, int]] = set()
        with release.artifact_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != SCHEMA:
                raise ValueError("national register CSV schema is unsupported")
            for row_number, row in enumerate(reader, start=2):
                raw_count += 1
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("national register CSV row is malformed")
                group = classify_group(row, profile, row_number)
                if group is None:
                    rejected_count += 1
                    continue
                key = (group.make, group.model, group.cohort_year)
                if key in seen:
                    raise ValueError("national register CSV contains a duplicate group")
                seen.add(key)
                groups.append(group)
        return raw_count, rejected_count, groups

    @staticmethod
    def _write_observations(
        release: StoredRelease,
        profile: RegisterProfile,
        stock_date: date,
        groups: list[RegisterGroup],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        notes = (
            f"Aggregated {profile.publisher} passenger cars to make, model and "
            f"{profile.definition}.",
            f"Only cohorts from {EARLIEST_COHORT_YEAR} are used.",
        )
        if profile.lower_bound:
            notes += (
                "The register hides some cars, so a cohort's count is a lower bound; "
                "planning keeps it only where it counts more than registrations.",
            )
        batch: list[Observation] = []
        for group in groups:
            key = (group.make, group.model, str(group.cohort_year))
            group_id = stable_evidence_id("register-group", profile.geography, *key)
            batch.append(
                Observation(
                    observation_id=stable_evidence_id("obs-register", release.release_id, *key),
                    release_id=release.release_id,
                    original_row_locator=f"{group_id}:row-{group.row_number}",
                    geography=profile.geography,
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
                    original_type=f"{profile.publisher} make and commercial name",
                    source_make_identifier=group.make,
                    source_model_identifier=stable_evidence_id(
                        "register-model", profile.geography, group.make, group.model
                    ),
                    normalized_make=normalize_vehicle_label(group.make),
                    normalized_model=normalize_vehicle_label(group.model),
                    normalized_model_year=None,
                    canonical_vehicle_id=None,
                    mapping_status=MappingStatus.UNRESOLVED,
                    transformation_notes=notes,
                    validation_flags=(
                        (STOCK_LOWER_BOUND_FLAG,) if profile.lower_bound else ()
                    )
                    + (VOCABULARY_EXCLUDED_FLAG,),
                    evidence_confidence=_confidence(profile),
                    registration_cohort_year=group.cohort_year,
                )
            )
            if len(batch) >= _OBSERVATION_BATCH_SIZE:
                repository.add_observations(batch)
                batch.clear()
        if batch:
            repository.add_observations(batch)


def _confidence(profile: RegisterProfile) -> EvidenceConfidence:
    return EvidenceConfidence(
        authority=25,
        publication_status=10,
        coverage=15 if profile.lower_bound else 25,
        identity=5,
        independent_agreement=10,
        reasons=(
            f"Official {profile.publisher} administrative register of cars on the road.",
            (
                "The register hides some cars, so a cohort's count is a lower bound."
                if profile.lower_bound
                else "The register counts every car on the road."
            ),
            "Register commercial-name identity is retained without an unreviewed alias.",
            "Agreement component is neutral until cross-source overlap is evaluated.",
        ),
    )
