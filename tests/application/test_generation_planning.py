from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace

from icor.application.generation_planning import GenerationPlanningService
from icor.domain.evidence import ConfidenceBand, Measure


class Repository:
    def __init__(self) -> None:
        self.releases = (
            SimpleNamespace(
                release_id="release-eea-history",
                dependency_group="european-register",
                source_id="eea-co2-monitoring",
            ),
        )
        self.generation = SimpleNamespace(
            generation_id="generation-volkswagen-golf-eu",
            canonical_vehicle_id="vehicle-volkswagen-golf-eu",
        )
        self.observations = tuple(
            SimpleNamespace(
                observation_id=f"observation-golf-{year}",
                release_id="release-eea-history",
                canonical_vehicle_id="vehicle-volkswagen-golf-eu",
                geography="DE",
                registration_cohort_year=year,
                period_start=date(year, 1, 1),
                period_end=date(year, 12, 31),
                measure=Measure.NEW_REGISTRATIONS,
                value=Decimal(value),
            )
            for year, value in ((2020, "100"), (2021, "110"), (2022, "120"))
        )
        self.assignments = tuple(
            SimpleNamespace(
                observation_id=item.observation_id,
                selected_generation_id=self.generation.generation_id,
                confidence=ConfidenceBand.LOW,
            )
            for item in self.observations
        )
        self.cohorts = ()
        self.opportunities = ()

    def list_releases(self):
        return self.releases

    def list_generations(self):
        return (self.generation,)

    def list_generation_assignments(self):
        return self.assignments

    def list_observations(self):
        return self.observations

    def add_cohort_estimates(self, values):
        self.cohorts += tuple(values)

    def add_opportunity_estimates(self, values):
        self.opportunities += tuple(values)


def test_generation_planning_materializes_forecast_cohorts_and_interval() -> None:
    repository = Repository()

    result = GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    assert result.cohort_count == 9
    assert result.opportunity_count == 1
    assert {item.registration_cohort_year for item in repository.cohorts} == set(
        range(2020, 2029)
    )
    opportunity = repository.opportunities[0]
    assert opportunity.p10 <= opportunity.p50 <= opportunity.p90
    assert opportunity.active_fleet_p50 > 0
    assert "uncalibrated-fitment-and-hazard" in opportunity.reason_codes


def test_internal_registration_gap_is_explicitly_estimated() -> None:
    repository = Repository()
    repository.observations[1].registration_cohort_year = 2022
    repository.observations[1].period_start = date(2022, 1, 1)
    repository.observations[1].period_end = date(2022, 12, 31)
    repository.observations[2].registration_cohort_year = 2023
    repository.observations[2].period_start = date(2023, 1, 1)
    repository.observations[2].period_end = date(2023, 12, 31)

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    estimated = next(
        item for item in repository.cohorts if item.registration_cohort_year == 2021
    )
    assert estimated.registrations == Decimal("105")
    assert "estimated-registration-cohort" in estimated.reason_codes


def test_sparse_series_without_backtest_history_remains_evidence_only() -> None:
    repository = Repository()
    repository.observations = repository.observations[:2]
    repository.assignments = repository.assignments[:2]

    result = GenerationPlanningService().apply(
        repository, horizons=(2028,), seed=20260827
    )

    assert result.cohort_count == 0
    assert result.opportunity_count == 0
    assert result.evidence_only_series_count == 1
