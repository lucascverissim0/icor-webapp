"""Materialize generation cohorts and assumption-led replacement opportunity."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Protocol

from icor.domain.cohorts import CohortEstimate, OpportunityEstimate
from icor.domain.evidence import ConfidenceBand, Measure
from icor.domain.generations import GenerationIdentityKind
from icor.evidence.normalization import stable_evidence_id
from icor.forecasting.reconciliation import (
    CoverageCandidate,
    RegistrationCoverageSelector,
    RegistrationInput,
    RegistrationReconciler,
    release_precedence,
)
from icor.forecasting.registration_forecast import RegistrationForecaster
from icor.forecasting.replacement_hazard import ReplacementHazardModel
from icor.forecasting.survival import CohortSurvivalModel, SurvivalInterval
from icor.forecasting.uncertainty import OpportunityUncertaintyModel

_BATCH_SIZE = 20_000
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
    evidence_only_series_count: int
    corroborating_input_count: int = 0
    superseded_coverage_input_count: int = 0
    #: Series with one or two annual values. They used to be dropped; their
    #: observed cohorts are real vehicles, so they are now planned and labelled.
    short_history_series_count: int = 0
    #: Series whose complete cohorts come from a register's observed stock.
    stock_anchored_series_count: int = 0
    #: Observed stock cohorts too old for the survival curve to project.
    beyond_support_cohort_count: int = 0


@dataclass(frozen=True, slots=True)
class _AnnualValue:
    registrations: Decimal
    observation_ids: tuple[str, ...]
    status: str
    extra_reason_codes: tuple[str, ...] = ()


class GenerationPlanningService:
    """Create one reproducible cohort/opportunity baseline from assigned evidence."""

    def __init__(
        self,
        *,
        batch_size: int = _BATCH_SIZE,
        survival: CohortSurvivalModel | None = None,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("planning batch size must be a positive integer")
        self.batch_size = batch_size
        self.reconciler = RegistrationReconciler()
        self.forecaster = RegistrationForecaster()
        # Injectable so a calibrated retention curve can replace the assumed one
        # without editing this service. Its .method flows into every cohort record.
        self.survival = survival or CohortSurvivalModel()
        self.hazard = ReplacementHazardModel()
        self.uncertainty = OpportunityUncertaintyModel()

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
        # Observed active fleet by cohort, kept only at each market's newest
        # stock date so memory holds one register snapshot per market.
        stock_dates: dict[str, date] = {}
        stock_series: dict[tuple[str, str, str], dict[int, _AnnualValue]] = {}
        for observation in repository.list_observations():
            assignment = assignments.get(observation.observation_id)
            if assignment is None or observation.canonical_vehicle_id is None:
                continue
            if observation.measure is Measure.ACTIVE_FLEET:
                _collect_stock(observation, assignment, stock_dates, stock_series)
                continue
            if observation.measure is not Measure.NEW_REGISTRATIONS:
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

        coverage_candidates: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
        for key in release_buckets:
            _generation, _vehicle, geography, year, release_id = key
            release = releases.get(release_id)
            if release is None:
                raise ValueError("planning observation release is unavailable")
            coverage_candidates[geography, year].add(
                (release.source_id, str(release.publication_status))
            )
        coverage_selector = RegistrationCoverageSelector()
        coverage: dict[tuple[str, int], str] = {}
        coverage_corroborators: dict[tuple[str, int], tuple[str, ...]] = {}
        for scope, candidates in coverage_candidates.items():
            selection = coverage_selector.select(
                tuple(
                    CoverageCandidate(source_id=source, publication_status=status)
                    for source, status in sorted(candidates)
                )
            )
            coverage[scope] = selection.source_id
            coverage_corroborators[scope] = selection.corroborating_source_ids

        annual_candidates: dict[
            tuple[str, str, str, int], list[RegistrationInput]
        ] = defaultdict(list)
        input_observations: dict[tuple[str, str, str, int, str], tuple[str, ...]] = {}
        superseded_coverage_count = 0
        for key, (value, observation_ids) in release_buckets.items():
            generation_id, vehicle_id, geography, year, release_id = key
            release = releases.get(release_id)
            if release is None:
                raise ValueError("planning observation release is unavailable")
            if release.source_id != coverage[geography, year]:
                # Another publisher decomposes this country-year. Keeping this
                # release would add the vehicles only it happens to name on top
                # of a total that already covers the whole market.
                superseded_coverage_count += 1
                continue
            annual_key = generation_id, vehicle_id, geography, year
            annual_candidates[annual_key].append(
                RegistrationInput(
                    release_id,
                    release.dependency_group,
                    value,
                    release_precedence(release.source_id),
                )
            )
            input_observations[key] = tuple(sorted(observation_ids))

        series: dict[tuple[str, str, str], dict[int, _AnnualValue]] = defaultdict(dict)
        selected_count = excluded_count = corroborating_count = 0
        for annual_key in sorted(annual_candidates):
            generation_id, vehicle_id, geography, year = annual_key
            result = self.reconciler.reconcile(tuple(annual_candidates[annual_key]))
            selected_count += len(result.selected_input_ids)
            excluded_count += len(result.excluded_input_ids)
            corroborating_count += len(result.corroborating_input_ids)
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
                _coverage_reason_codes(
                    result.agreement, coverage_corroborators[geography, year]
                ),
            )

        cohort_buffer: list[CohortEstimate] = []
        opportunity_buffer: list[OpportunityEstimate] = []
        cohort_count = opportunity_count = 0

        def flush_planning_records() -> None:
            nonlocal cohort_count, opportunity_count
            _write_batches(
                repository.add_cohort_estimates,
                cohort_buffer,
                batch_size=self.batch_size,
            )
            _write_batches(
                repository.add_opportunity_estimates,
                opportunity_buffer,
                batch_size=self.batch_size,
            )
            cohort_count += len(cohort_buffer)
            opportunity_count += len(opportunity_buffer)
            cohort_buffer.clear()
            opportunity_buffer.clear()

        # Every year each market's coverage source reported. A vehicle absent
        # from a reported year is not a gap to fill.
        covered_years: dict[str, set[int]] = defaultdict(set)
        for geography, year in coverage:
            covered_years[geography].add(year)

        # The newest year any vehicle was observed in each market. A series that
        # reaches it is still on sale there; one that stops short has ended.
        latest_year: dict[str, int] = {}
        for _generation, _vehicle, geography in series:
            latest_year[geography] = max(
                latest_year.get(geography, 0), max(series[_generation, _vehicle, geography])
            )

        # A register's count of the cars still on the road replaces modelled
        # survival for the cohorts it covers completely, but only in a market
        # whose registrations are planned too: a stock reported for a different
        # territory (DfT's UK-wide VEH0124 beside GB registrations) would be
        # added on top of a fleet already counted.
        anchor_year = {
            geography: _last_complete_cohort(stock_date)
            for geography, stock_date in stock_dates.items()
            if geography in latest_year
        }
        anchored_stock = {
            key: {year: value for year, value in cohorts.items() if year <= anchor_year[key[2]]}
            for key, cohorts in stock_series.items()
            if key[2] in anchor_year
        }
        anchored_stock = {key: cohorts for key, cohorts in anchored_stock.items() if cohorts}

        evidence_only_series_count = short_history_series_count = 0
        stock_anchored_series_count = beyond_support_cohort_count = 0
        for series_key in sorted(set(series) | set(anchored_stock)):
            generation_id, vehicle_id, geography = series_key
            generation = generations.get(generation_id)
            if generation is None or generation.canonical_vehicle_id != vehicle_id:
                raise ValueError("planning generation vehicle is unavailable")
            values: dict[int, _AnnualValue] = {}
            forecast_method = _STOCK_ONLY_METHOD
            history_codes: tuple[str, ...] = ()
            observed_years = series.get(series_key)
            if observed_years:
                short_history = len(observed_years) < 3
                end_year = _end_year(
                    generation, max(observed_years), latest_year[geography]
                )
                values = _fill_internal_gaps(observed_years, covered_years[geography])
                if end_year is not None:
                    values = {
                        year: annual for year, annual in values.items() if year <= end_year
                    }
                if values:
                    short_history_series_count += short_history
                    history_codes = (
                        ("short-registration-history",) if short_history else ()
                    )
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
                    if end_year is not None:
                        forecasts = {
                            year: value
                            for year, value in forecasts.items()
                            if year <= end_year
                        }
                    for year, value in forecasts.items():
                        # A year the market reported is evidence, not a gap:
                        # a vehicle absent from it sold none under this name.
                        if year not in covered_years[geography]:
                            values[year] = _AnnualValue(value, all_input_ids, "forecast")
            stock = anchored_stock.get(series_key, {})
            if geography in anchor_year:
                values = {
                    year: annual
                    for year, annual in values.items()
                    if year > anchor_year[geography]
                }
            if not values and not stock:
                evidence_only_series_count += 1
                continue
            stock_anchored_series_count += bool(stock)
            stock_codes = ("observed-stock-anchored",) if stock else ()
            for horizon in sorted(horizons):
                horizon_cohorts: list[CohortEstimate] = []
                opportunity_components = []
                for cohort_year in sorted(
                    year for year in {*values, *stock} if year <= horizon
                ):
                    age = horizon - cohort_year
                    if cohort_year in stock:
                        annual = stock[cohort_year]
                        fleet = self._conditional_fleet(
                            annual.registrations,
                            age_years=age,
                            stock_age_years=stock_dates[geography].year - cohort_year,
                        )
                        if fleet is None:
                            beyond_support_cohort_count += 1
                            continue
                        cohort_codes = (
                            "observed-stock-anchored-cohort",
                            self.survival.reason_code(geography),
                        )
                    else:
                        annual = values[cohort_year]
                        fleet = self.survival.interval(
                            annual.registrations, age_years=age
                        )
                        cohort_codes = (
                            f"{annual.status}-registration-cohort",
                            self.survival.reason_code(geography),
                            *annual.extra_reason_codes,
                            *history_codes,
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
                        reason_codes=cohort_codes,
                    )
                    horizon_cohorts.append(cohort)
                    hazard = self.hazard.interval(
                        age_years=age,
                        geography=geography,
                    )
                    opportunity_components.append((fleet, hazard))
                if not horizon_cohorts:
                    continue
                cohort_buffer.extend(horizon_cohorts)
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
                opportunity_buffer.append(
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
                        uncertainty_method=self.uncertainty.method,
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
                            *history_codes,
                            *stock_codes,
                        ),
                    )
                )
            if (
                len(cohort_buffer) >= self.batch_size
                or len(opportunity_buffer) >= self.batch_size
            ):
                flush_planning_records()
        flush_planning_records()
        return GenerationPlanningResult(
            cohort_count,
            opportunity_count,
            selected_count,
            excluded_count,
            evidence_only_series_count,
            corroborating_count,
            superseded_coverage_count,
            short_history_series_count,
            stock_anchored_series_count,
            beyond_support_cohort_count,
        )

    def _conditional_fleet(
        self, stock: Decimal, *, age_years: int, stock_age_years: int
    ) -> SurvivalInterval | None:
        """Project cars observed on the road at one age to an older age.

        Each quantile's curve is used for its own ratio, so the observed count
        is the anchor and only the remaining ageing is modelled. A cohort the
        curve cannot reach, or one it already counts as gone, returns None.
        """

        scale = Decimal(1_000_000)
        try:
            at_stock = self.survival.interval(scale, age_years=max(stock_age_years, 0))
            at_horizon = self.survival.interval(scale, age_years=age_years)
        except ValueError:
            return None
        if min(at_stock.p10, at_stock.p50, at_stock.p90) <= 0:
            return None
        return SurvivalInterval(
            *(
                (stock * horizon_share / stock_share).quantize(Decimal("0.0001"))
                for horizon_share, stock_share in (
                    (at_horizon.p10, at_stock.p10),
                    (at_horizon.p50, at_stock.p50),
                    (at_horizon.p90, at_stock.p90),
                )
            )
        )

    def _forecast(
        self, values: dict[int, _AnnualValue], horizon: int
    ) -> tuple[str, dict[int, Decimal]]:
        observed = {year: value.registrations for year, value in values.items()}
        # Forecast from the latest unbroken run: a reported year the vehicle is
        # absent from is a break in its history, not a value to invent.
        run_start = max(observed)
        while run_start - 1 in observed:
            run_start -= 1
        observed = {year: value for year, value in observed.items() if year >= run_start}
        latest = max(observed)
        if latest >= horizon:
            return "observed-history-only-v1", {}
        if len(observed) == 1:
            registration = next(iter(observed.values()))
            return "single-observation-constant-v1", {
                year: registration for year in range(latest + 1, horizon + 1)
            }
        forecast = self.forecaster.forecast(
            observed,
            horizon_year=horizon,
            evaluate_backtest=False,
        )
        return forecast.method, dict(forecast.values)


_STOCK_ONLY_METHOD = "observed-stock-only-v1"


def _last_complete_cohort(stock_date: date) -> int:
    """The newest cohort a register snapshot has seen in full."""

    return stock_date.year if (stock_date.month, stock_date.day) == (12, 31) else (
        stock_date.year - 1
    )


def _collect_stock(observation, assignment, stock_dates, stock_series) -> None:
    year = observation.registration_cohort_year
    if year is None:
        return
    geography = observation.geography
    stock_date = observation.period_end
    current = stock_dates.get(geography)
    if current is not None and stock_date < current:
        return
    if current is None or stock_date > current:
        stock_dates[geography] = stock_date
        for key in [key for key in stock_series if key[2] == geography]:
            del stock_series[key]
    key = (assignment.selected_generation_id, observation.canonical_vehicle_id, geography)
    cohorts = stock_series.setdefault(key, {})
    existing = cohorts.get(year)
    cohorts[year] = _AnnualValue(
        (existing.registrations if existing else Decimal(0)) + observation.value,
        tuple(
            sorted(
                {*(existing.observation_ids if existing else ()), observation.observation_id}
            )
        ),
        "observed-stock",
    )


def _end_year(generation, last_observed: int, market_latest: int) -> int | None:
    """The last cohort year a series may hold, or None while it is on sale.

    A reviewed generation's end is a documented manufacturer fact. An estimated
    generation's end is only the last year any data mentions it, so it says
    nothing about discontinuation by itself: the series has ended only if the
    market reports later years without it.
    """

    if getattr(generation, "identity_kind", None) is GenerationIdentityKind.ESTIMATED:
        return None if last_observed >= market_latest else last_observed
    if generation.end_month is not None:
        return generation.end_month.year
    return None


def _fill_internal_gaps(
    values: dict[int, _AnnualValue], reported_years: set[int] | frozenset[int] = frozenset()
) -> dict[int, _AnnualValue]:
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
        if year in completed or year in reported_years:
            continue
        before = max(candidate for candidate in years if candidate < year)
        after = min(candidate for candidate in years if candidate > year)
        fraction = Decimal(year - before) / Decimal(after - before)
        estimate = values[before].registrations + fraction * (
            values[after].registrations - values[before].registrations
        )
        completed[year] = _AnnualValue(estimate, all_inputs, "estimated")
    return completed


def _coverage_reason_codes(
    agreement: str, corroborating_sources: tuple[str, ...]
) -> tuple[str, ...]:
    """Name the corroboration on the cohort, so provenance travels with the value."""

    if agreement == "disputed":
        return ("independent-publisher-disagreement-above-tolerance",)
    if agreement == "corroborated" or corroborating_sources:
        return ("corroborated-by-independent-publisher",)
    return ()


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


def _write_batches(write, records: list, *, batch_size: int) -> None:
    for start in range(0, len(records), batch_size):
        write(records[start : start + batch_size])
