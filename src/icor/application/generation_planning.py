"""Materialize generation cohorts and assumption-led replacement opportunity."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from icor.domain.cohorts import CohortEstimate, OpportunityEstimate
from icor.domain.evidence import ConfidenceBand, Measure
from icor.evidence.normalization import stable_evidence_id
from icor.forecasting.reconciliation import RegistrationInput, RegistrationReconciler
from icor.forecasting.registration_forecast import RegistrationForecaster
from icor.forecasting.replacement_hazard import ReplacementHazardModel
from icor.forecasting.survival import CohortSurvivalModel
from icor.forecasting.uncertainty import OpportunityUncertaintyModel

_BATCH_SIZE = 2_000
_SUPPORTED_HORIZONS = frozenset({2028, 2031})


class _Repository(Protocol):
    def list_releases(self): ...

    def list_generations(self): ...

    def list_generation_assignments(self): ...

    def list_observations(self): ...

    def add_cohort_estimates(self, values): ...

    def add_opportunity_estimates(self, values): ...


@dataclass(frozen=True, slots=True)
class GenerationPlanningResult:
    cohort_count: int
    opportunity_count: int
    reconciled_input_count: int
    excluded_correlated_input_count: int


@dataclass(frozen=True, slots=True)
class _AnnualValue:
    registrations: Decimal
    observation_ids: tuple[str, ...]
    status: str


class GenerationPlanningService:
    """Create one reproducible cohort/opportunity baseline from assigned evidence."""

    def __init__(self) -> None:
        self.reconciler = RegistrationReconciler()
        self.forecaster = RegistrationForecaster()
        self.survival = CohortSurvivalModel()
        self.hazard = ReplacementHazardModel(geography_multipliers={"GB": "1.10"})
        self.uncertainty = OpportunityUncertaintyModel(draw_count=256)

    def apply(
        self,
        repository: _Repository,
        *,
        horizons: tuple[int, ...] = (2028, 2031),
        seed: int = 20260827,
    ) -> GenerationPlanningResult:
        if (
            not horizons
            or len(horizons) != len(set(horizons))
            or any(year not in _SUPPORTED_HORIZONS for year in horizons)
        ):
            raise ValueError("planning horizons must be unique supported years")
        if type(seed) is not int:
            raise ValueError("planning seed must be an integer")

        releases = {item.release_id: item for item in repository.list_releases()}
        generations = {item.generation_id: item for item in repository.list_generations()}
        assignments = {
            item.observation_id: item for item in repository.list_generation_assignments()
        }
        release_buckets: dict[
            tuple[str, str, str, int, str], tuple[Decimal, list[str]]
        ] = {}
        for observation in repository.list_observations():
            assignment = assignments.get(observation.observation_id)
            if (
                assignment is None
                or observation.measure is not Measure.NEW_REGISTRATIONS
                or observation.canonical_vehicle_id is None
            ):
                continue
            year = observation.registration_cohort_year
            if year is None and observation.period_start.year == observation.period_end.year:
                year = observation.period_end.year
            if year is None:
                continue
            key = (
                assignment.selected_generation_id,
                observation.canonical_vehicle_id,
                observation.geography,
                year,
                observation.release_id,
            )
            value, identifiers = release_buckets.get(key, (Decimal(0), []))
            release_buckets[key] = (
                value + observation.value,
                [*identifiers, observation.observation_id],
            )

        annual_candidates: dict[
            tuple[str, str, str, int], list[RegistrationInput]
        ] = defaultdict(list)
        input_observations: dict[tuple[str, str, str, int, str], tuple[str, ...]] = {}
        for key, (value, observation_ids) in release_buckets.items():
            generation_id, vehicle_id, geography, year, release_id = key
            release = releases.get(release_id)
            if release is None:
                raise ValueError("planning observation release is unavailable")
            annual_key = generation_id, vehicle_id, geography, year
            annual_candidates[annual_key].append(
                RegistrationInput(
                    release_id,
                    release.dependency_group,
                    value,
                    _release_priority(release.source_id),
                )
            )
            input_observations[key] = tuple(sorted(observation_ids))

        series: dict[tuple[str, str, str], dict[int, _AnnualValue]] = defaultdict(dict)
        selected_count = excluded_count = 0
        for annual_key in sorted(annual_candidates):
            generation_id, vehicle_id, geography, year = annual_key
            result = self.reconciler.reconcile(tuple(annual_candidates[annual_key]))
            selected_count += len(result.selected_input_ids)
            excluded_count += len(result.excluded_input_ids)
            observation_ids = tuple(
                sorted(
                    identifier
                    for release_id in result.selected_input_ids
                    for identifier in input_observations[
                        generation_id, vehicle_id, geography, year, release_id
                    ]
                )
            )
            series[generation_id, vehicle_id, geography][year] = _AnnualValue(
                result.value,
                observation_ids,
                result.status,
            )

        cohorts: list[CohortEstimate] = []
        opportunities: list[OpportunityEstimate] = []
        for series_key in sorted(series):
            generation_id, vehicle_id, geography = series_key
            generation = generations.get(generation_id)
            if generation is None or generation.canonical_vehicle_id != vehicle_id:
                raise ValueError("planning generation vehicle is unavailable")
            values = _fill_internal_gaps(series[series_key])
            all_input_ids = tuple(
                sorted(
                    {
                        identifier
                        for annual in values.values()
                        for identifier in annual.observation_ids
                    }
                )
            )
            forecast_method, forecasts = self._forecast(values, max(horizons))
            for year, value in forecasts.items():
                values[year] = _AnnualValue(value, all_input_ids, "forecast")
            for horizon in sorted(horizons):
                horizon_cohorts: list[CohortEstimate] = []
                opportunity_components = []
                for cohort_year in sorted(year for year in values if year <= horizon):
                    annual = values[cohort_year]
                    fleet = self.survival.interval(
                        annual.registrations,
                        age_years=horizon - cohort_year,
                    )
                    cohort = CohortEstimate(
                        cohort_id=stable_evidence_id(
                            "cohort",
                            generation_id,
                            geography,
                            str(cohort_year),
                            str(horizon),
                        ),
                        generation_id=generation_id,
                        canonical_vehicle_id=vehicle_id,
                        geography=geography,
                        registration_cohort_year=cohort_year,
                        as_of_year=horizon,
                        registrations=annual.registrations,
                        active_fleet_p10=fleet.p10,
                        active_fleet_p50=fleet.p50,
                        active_fleet_p90=fleet.p90,
                        input_observation_ids=annual.observation_ids,
                        survival_method=self.survival.method,
                        confidence=ConfidenceBand.LOW,
                        reason_codes=(
                            f"{annual.status}-registration-cohort",
                            "assumption-led-survival-not-calibrated",
                        ),
                    )
                    horizon_cohorts.append(cohort)
                    hazard = self.hazard.interval(
                        age_years=horizon - cohort_year,
                        geography=geography,
                    )
                    opportunity_components.append((fleet, hazard))
                cohorts.extend(horizon_cohorts)
                fleet_p10 = sum(
                    (item.active_fleet_p10 for item in horizon_cohorts), Decimal(0)
                )
                fleet_p50 = sum(
                    (item.active_fleet_p50 for item in horizon_cohorts), Decimal(0)
                )
                fleet_p90 = sum(
                    (item.active_fleet_p90 for item in horizon_cohorts), Decimal(0)
                )
                effective_hazard = tuple(
                    _effective_hazard(opportunity_components, position)
                    for position in range(3)
                )
                opportunity_interval = self.uncertainty.estimate(
                    active_fleet_p10=fleet_p10,
                    active_fleet_p50=fleet_p50,
                    active_fleet_p90=fleet_p90,
                    hazard_p10=effective_hazard[0],
                    hazard_p50=effective_hazard[1],
                    hazard_p90=effective_hazard[2],
                    seed=_stable_seed(
                        seed,
                        stable_evidence_id(
                            "opportunity-seed", generation_id, geography, str(horizon)
                        ),
                    ),
                )
                opportunities.append(
                    OpportunityEstimate(
                        opportunity_id=stable_evidence_id(
                            "opportunity", generation_id, geography, str(horizon)
                        ),
                        generation_id=generation_id,
                        canonical_vehicle_id=vehicle_id,
                        geography=geography,
                        horizon_year=horizon,
                        p10=opportunity_interval.p10,
                        p50=opportunity_interval.p50,
                        p90=opportunity_interval.p90,
                        active_fleet_p50=fleet_p50,
                        input_cohort_ids=tuple(item.cohort_id for item in horizon_cohorts),
                        hazard_method=self.hazard.method,
                        forecast_method=forecast_method,
                        confidence=ConfidenceBand.LOW,
                        assumption_ids=(
                            *self.survival.assumption_ids,
                            *self.hazard.assumption_ids,
                            "no-proprietary-fitment-calibration-v1",
                        ),
                        reason_codes=(
                            "assumption-led-opportunity-baseline",
                            "uncalibrated-fitment-and-hazard",
                            "no-exact-windshield-fitment-inference",
                        ),
                    )
                )

        _write_batches(repository.add_cohort_estimates, cohorts)
        _write_batches(repository.add_opportunity_estimates, opportunities)
        return GenerationPlanningResult(
            len(cohorts),
            len(opportunities),
            selected_count,
            excluded_count,
        )

    def _forecast(
        self, values: dict[int, _AnnualValue], horizon: int
    ) -> tuple[str, dict[int, Decimal]]:
        observed = {year: value.registrations for year, value in values.items()}
        latest = max(observed)
        if latest >= horizon:
            return "observed-history-only-v1", {}
        if len(observed) == 1:
            registration = next(iter(observed.values()))
            return "single-observation-constant-v1", {
                year: registration for year in range(latest + 1, horizon + 1)
            }
        forecast = self.forecaster.forecast(observed, horizon_year=horizon)
        return forecast.method, dict(forecast.values)


def _fill_internal_gaps(values: dict[int, _AnnualValue]) -> dict[int, _AnnualValue]:
    completed = dict(values)
    years = sorted(values)
    all_inputs = tuple(
        sorted(
            {
                identifier
                for annual in values.values()
                for identifier in annual.observation_ids
            }
        )
    )
    for year in range(years[0], years[-1] + 1):
        if year in completed:
            continue
        before = max(candidate for candidate in years if candidate < year)
        after = min(candidate for candidate in years if candidate > year)
        fraction = Decimal(year - before) / Decimal(after - before)
        estimate = values[before].registrations + fraction * (
            values[after].registrations - values[before].registrations
        )
        completed[year] = _AnnualValue(estimate, all_inputs, "estimated")
    return completed


def _release_priority(source_id: str) -> int:
    if source_id.startswith(("kba-", "uk-dft-")):
        return 30
    if source_id == "eea-co2-monitoring":
        return 20
    return 10


def _stable_seed(seed: int, identifier: str) -> int:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _effective_hazard(components: list, position: int) -> Decimal:
    selected = [
        ((fleet.p10, fleet.p50, fleet.p90)[position], hazard[position])
        for fleet, hazard in components
    ]
    fleet_total = sum((fleet for fleet, _ in selected), Decimal(0))
    if fleet_total == 0:
        return Decimal(0)
    opportunity_total = sum(
        (fleet * hazard for fleet, hazard in selected),
        Decimal(0),
    )
    return opportunity_total / fleet_total


def _write_batches(write, records: list) -> None:
    for start in range(0, len(records), _BATCH_SIZE):
        write(records[start : start + _BATCH_SIZE])
