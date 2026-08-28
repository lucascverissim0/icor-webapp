from datetime import date
from types import SimpleNamespace

from icor.application.completeness import CompletenessService
from icor.domain.evidence import MappingStatus, Measure
from icor.domain.generations import GenerationIdentityKind


class Repository:
    def __init__(self) -> None:
        self.release = SimpleNamespace(
            release_id="release-eea-2024",
            geography="EEA reporting countries",
            coverage_end=date(2024, 12, 31),
            rejected_record_count=7,
        )
        self.observations = (
            SimpleNamespace(
                observation_id="observation-registration",
                release_id=self.release.release_id,
                geography="DE",
                period_end=date(2024, 12, 31),
                registration_cohort_year=2024,
                canonical_vehicle_id="vehicle-golf",
                mapping_status=MappingStatus.NORMALIZED_LABEL,
                measure=Measure.NEW_REGISTRATIONS,
            ),
            SimpleNamespace(
                observation_id="observation-fleet-evidence-only",
                release_id=self.release.release_id,
                geography="DE",
                period_end=date(2024, 12, 31),
                registration_cohort_year=None,
                canonical_vehicle_id="vehicle-golf",
                mapping_status=MappingStatus.NORMALIZED_LABEL,
                measure=Measure.ACTIVE_FLEET,
            ),
        )
        self.assignment = SimpleNamespace(
            observation_id="observation-registration",
            selected_generation_id="generation-golf",
        )
        self.generation = SimpleNamespace(
            generation_id="generation-golf",
            identity_kind=GenerationIdentityKind.ESTIMATED,
        )
        self.records = ()

    def list_releases(self):
        return (self.release,)

    def list_observations(self):
        return self.observations

    def list_generation_assignments(self):
        return (self.assignment,)

    def list_generations(self):
        return (self.generation,)

    def list_cohort_estimates(self):
        return (
            SimpleNamespace(
                input_observation_ids=("observation-registration",),
            ),
        )

    def add_completeness_records(self, records):
        self.records += tuple(records)


def test_completeness_separates_forecastable_and_evidence_only_rows() -> None:
    repository = Repository()

    count = CompletenessService().materialize(repository)

    assert count == 2
    de = next(item for item in repository.records if item.geography == "DE")
    assert de.observation_count == 2
    assert de.usable_observation_count == 2
    assert de.assigned_observation_count == 1
    assert de.forecastable_count == 1
    assert de.evidence_only_count == 1
    assert de.estimated_generation_count == 1
    source_scope = next(
        item for item in repository.records if item.geography == "EEA reporting countries"
    )
    assert source_scope.rejected_record_count == 7
