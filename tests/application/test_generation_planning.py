from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace

from icor.application.generation_planning import GenerationPlanningService
from icor.domain.evidence import ConfidenceBand, Measure
from icor.domain.generations import GenerationIdentityKind


class Repository:
    def __init__(self) -> None:
        self.releases = (
            SimpleNamespace(
                release_id="release-eea-history",
                dependency_group="european-register",
                source_id="eea-co2-monitoring",
                publication_status="final",
            ),
        )
        self.generation = SimpleNamespace(
            generation_id="generation-volkswagen-golf-eu",
            canonical_vehicle_id="vehicle-volkswagen-golf-eu",
            end_month=None,
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
        return (self.generation, *getattr(self, "extra_generations", ()))

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


def _add_vehicle(repository, name: str, years_values, *, kind=None, end_month=None):
    """Add a second vehicle in DE so the market's latest data year is known."""

    generation = SimpleNamespace(
        generation_id=f"generation-{name}",
        canonical_vehicle_id=f"vehicle-{name}",
        end_month=end_month,
        identity_kind=kind,
    )
    observations = tuple(
        SimpleNamespace(
            observation_id=f"observation-{name}-{year}",
            release_id="release-eea-history",
            canonical_vehicle_id=generation.canonical_vehicle_id,
            geography="DE",
            registration_cohort_year=year,
            period_start=date(year, 1, 1),
            period_end=date(year, 12, 31),
            measure=Measure.NEW_REGISTRATIONS,
            value=Decimal(value),
        )
        for year, value in years_values
    )
    repository.extra_generations = (*getattr(repository, "extra_generations", ()), generation)
    repository.observations = (*repository.observations, *observations)
    repository.assignments = (
        *repository.assignments,
        *(
            SimpleNamespace(
                observation_id=item.observation_id,
                selected_generation_id=generation.generation_id,
                confidence=ConfidenceBand.LOW,
            )
            for item in observations
        ),
    )
    return generation


def _cohort_years(repository, vehicle_id: str, horizon: int = 2028) -> set[int]:
    return {
        item.registration_cohort_year
        for item in repository.cohorts
        if item.canonical_vehicle_id == vehicle_id and item.as_of_year == horizon
    }


def test_a_short_history_still_counts_the_cars_already_on_the_road() -> None:
    """Two registration years are real vehicles needing windshields.

    The old rule dropped any series with fewer than three annual values, so a
    model launched in 2021 or sold in small numbers produced no demand at all.
    """

    repository = Repository()
    repository.observations = repository.observations[1:]
    repository.assignments = repository.assignments[1:]

    result = GenerationPlanningService().apply(
        repository, horizons=(2028,), seed=20260827
    )

    assert result.evidence_only_series_count == 0
    assert result.short_history_series_count == 1
    assert _cohort_years(repository, "vehicle-volkswagen-golf-eu") == set(range(2021, 2029))
    opportunity = repository.opportunities[0]
    assert opportunity.p50 > 0
    assert "short-registration-history" in opportunity.reason_codes


def test_a_single_registration_year_on_sale_now_is_carried_forward() -> None:
    repository = Repository()
    _add_vehicle(
        repository,
        "launch",
        ((2022, "40"),),
        kind=GenerationIdentityKind.ESTIMATED,
        end_month=date(2022, 12, 1),
    )

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    assert _cohort_years(repository, "vehicle-launch") == set(range(2022, 2029))
    launch = next(
        item for item in repository.opportunities if item.canonical_vehicle_id == "vehicle-launch"
    )
    assert launch.forecast_method == "single-observation-constant-v1"


def test_a_short_history_that_stopped_selling_keeps_only_its_real_cohorts() -> None:
    repository = Repository()
    _add_vehicle(
        repository,
        "ended",
        ((2020, "30"),),
        kind=GenerationIdentityKind.ESTIMATED,
        end_month=date(2020, 12, 1),
    )

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    assert _cohort_years(repository, "vehicle-ended") == {2020}
    ended = next(
        item for item in repository.opportunities if item.canonical_vehicle_id == "vehicle-ended"
    )
    assert ended.p50 > 0


def test_an_estimated_window_ending_at_the_latest_data_year_is_still_on_sale() -> None:
    """An estimated generation's end is just the last year we have data for.

    Reading it as a discontinuation made every model outside the reviewed
    catalog stop selling in its last data year, so none of them gained a single
    future registration.
    """

    repository = Repository()
    repository.generation.end_month = date(2022, 12, 1)
    repository.generation.identity_kind = GenerationIdentityKind.ESTIMATED

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    assert _cohort_years(repository, "vehicle-volkswagen-golf-eu") == set(range(2020, 2029))


def test_an_estimated_series_that_vanished_before_the_market_latest_year_has_ended() -> None:
    repository = Repository()
    repository.generation.end_month = date(2022, 12, 1)
    repository.generation.identity_kind = GenerationIdentityKind.ESTIMATED
    _add_vehicle(repository, "current", ((2022, "50"), (2023, "60"), (2024, "70")))

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    assert _cohort_years(repository, "vehicle-volkswagen-golf-eu") == {2020, 2021, 2022}


def test_discontinued_generation_stops_new_cohorts_but_keeps_replacement_demand() -> None:
    repository = Repository()
    repository.generation.end_month = date(2022, 12, 1)

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    assert {item.registration_cohort_year for item in repository.cohorts} == {
        2020,
        2021,
        2022,
    }
    opportunity = repository.opportunities[0]
    assert opportunity.active_fleet_p50 > 0
    assert opportunity.p50 > 0


def test_planning_flushes_bounded_dependency_complete_series_batches() -> None:
    class StreamingRepository(Repository):
        def __init__(self) -> None:
            super().__init__()
            france = tuple(
                SimpleNamespace(
                    **{
                        **vars(item),
                        "observation_id": item.observation_id.replace("golf", "golf-fr"),
                        "geography": "FR",
                    }
                )
                for item in self.observations
            )
            self.observations += france
            self.assignments += tuple(
                SimpleNamespace(
                    observation_id=item.observation_id,
                    selected_generation_id=self.generation.generation_id,
                    confidence=ConfidenceBand.LOW,
                )
                for item in france
            )
            self.write_events: list[tuple[str, int]] = []

        def add_cohort_estimates(self, values):
            batch = tuple(values)
            self.write_events.append(("cohort", len(batch)))
            super().add_cohort_estimates(batch)

        def add_opportunity_estimates(self, values):
            batch = tuple(values)
            cohort_ids = {item.cohort_id for item in self.cohorts}
            assert all(
                set(item.input_cohort_ids) <= cohort_ids for item in batch
            ), "opportunities must be written only after their cohorts"
            self.write_events.append(("opportunity", len(batch)))
            super().add_opportunity_estimates(batch)

    repository = StreamingRepository()

    result = GenerationPlanningService(batch_size=5).apply(
        repository, horizons=(2028,), seed=20260827
    )

    assert result.cohort_count == 18
    assert result.opportunity_count == 2
    assert max(size for _, size in repository.write_events) <= 5
    first_opportunity = next(
        index
        for index, event in enumerate(repository.write_events)
        if event[0] == "opportunity"
    )
    assert any(
        event[0] == "cohort"
        for event in repository.write_events[first_opportunity + 1 :]
    )


class TwoPublisherRepository:
    """GB 2020-2022 reported by both the EEA compilation and the UK register.

    Both publishers cover the whole market. The EEA names a vehicle the register
    does not, and the register names one the EEA does not, which is exactly the
    shape that used to inflate the country by adding the two decompositions.
    """

    def __init__(self) -> None:
        self.releases = (
            *(
                SimpleNamespace(
                    release_id=f"release-eea-{year}",
                    dependency_group=f"european-passenger-car-registrations-{year}",
                    source_id="eea-co2-monitoring",
                    publication_status="final",
                )
                for year in (2020, 2021, 2022)
            ),
            SimpleNamespace(
                release_id="release-uk-dft",
                dependency_group="uk-dvla-vehicle-register",
                source_id="uk-dft-veh0160",
                publication_status="final",
            ),
        )
        self.generation = SimpleNamespace(
            generation_id="generation-volkswagen-golf-eu",
            canonical_vehicle_id="vehicle-volkswagen-golf-eu",
            end_month=None,
        )
        self.register_only_generation = SimpleNamespace(
            generation_id="generation-register-only",
            canonical_vehicle_id="vehicle-register-only",
            end_month=None,
        )
        observations = []
        for year in (2020, 2021, 2022):
            observations.append(
                SimpleNamespace(
                    observation_id=f"observation-eea-golf-{year}",
                    release_id=f"release-eea-{year}",
                    canonical_vehicle_id="vehicle-volkswagen-golf-eu",
                    geography="GB",
                    registration_cohort_year=year,
                    period_start=date(year, 1, 1),
                    period_end=date(year, 12, 31),
                    measure=Measure.NEW_REGISTRATIONS,
                    value=Decimal("100"),
                )
            )
            for quarter in range(4):
                observations.append(
                    SimpleNamespace(
                        observation_id=f"observation-dft-golf-{year}-q{quarter + 1}",
                        release_id="release-uk-dft",
                        canonical_vehicle_id="vehicle-volkswagen-golf-eu",
                        geography="GB",
                        registration_cohort_year=None,
                        period_start=date(year, 1 + 3 * quarter, 1),
                        period_end=date(year, 3 + 3 * quarter, 28),
                        measure=Measure.NEW_REGISTRATIONS,
                        value=Decimal("24"),
                    )
                )
            observations.append(
                SimpleNamespace(
                    observation_id=f"observation-dft-register-only-{year}",
                    release_id="release-uk-dft",
                    canonical_vehicle_id="vehicle-register-only",
                    geography="GB",
                    registration_cohort_year=year,
                    period_start=date(year, 1, 1),
                    period_end=date(year, 12, 31),
                    measure=Measure.NEW_REGISTRATIONS,
                    value=Decimal("40"),
                )
            )
        self.observations = tuple(observations)
        by_vehicle = {
            "vehicle-volkswagen-golf-eu": self.generation.generation_id,
            "vehicle-register-only": self.register_only_generation.generation_id,
        }
        self.assignments = tuple(
            SimpleNamespace(
                observation_id=item.observation_id,
                selected_generation_id=by_vehicle[item.canonical_vehicle_id],
                confidence=ConfidenceBand.LOW,
            )
            for item in self.observations
        )
        self.cohorts = ()
        self.opportunities = ()

    def drop_the_eea_releases(self) -> None:
        """Leave the register as the only publisher, as GB 2021+ actually is."""

        self.releases = tuple(
            item for item in self.releases if item.source_id != "eea-co2-monitoring"
        )
        keep = {item.release_id for item in self.releases}
        self.observations = tuple(
            item for item in self.observations if item.release_id in keep
        )
        assigned = {item.observation_id for item in self.observations}
        self.assignments = tuple(
            item for item in self.assignments if item.observation_id in assigned
        )

    def list_releases(self):
        return self.releases

    def list_generations(self):
        return (self.generation, self.register_only_generation)

    def list_generation_assignments(self):
        return self.assignments

    def list_observations(self):
        return self.observations

    def add_cohort_estimates(self, values):
        self.cohorts += tuple(values)

    def add_opportunity_estimates(self, values):
        self.opportunities += tuple(values)


def _observed(repository, vehicle_id: str, year: int):
    return next(
        (
            cohort
            for cohort in repository.cohorts
            if cohort.canonical_vehicle_id == vehicle_id
            and cohort.registration_cohort_year == year
        ),
        None,
    )


def test_two_publishers_of_one_country_year_are_not_summed() -> None:
    repository = TwoPublisherRepository()

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    cohort = _observed(repository, "vehicle-volkswagen-golf-eu", 2020)
    assert cohort is not None
    assert cohort.registrations == Decimal("100")
    assert cohort.registrations != Decimal("196")


def test_the_pan_european_compilation_decomposes_the_country_year() -> None:
    repository = TwoPublisherRepository()

    result = GenerationPlanningService().apply(
        repository, horizons=(2028,), seed=20260827
    )

    # The register names a vehicle the compilation does not. Adding it would add
    # cars already counted inside the national total the compilation reports.
    assert _observed(repository, "vehicle-register-only", 2020) is None
    assert result.superseded_coverage_input_count > 0


def test_corroborating_observations_are_not_cited_as_cohort_inputs() -> None:
    repository = TwoPublisherRepository()

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    cohort = _observed(repository, "vehicle-volkswagen-golf-eu", 2021)
    assert cohort is not None
    assert cohort.input_observation_ids == ("observation-eea-golf-2021",)


def test_a_reconciled_cohort_keeps_its_reason_code() -> None:
    """The forecast repository matches on this literal; it must not drift."""

    repository = TwoPublisherRepository()

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    cohort = _observed(repository, "vehicle-volkswagen-golf-eu", 2020)
    assert cohort is not None
    assert "reconciled-registration-cohort" in cohort.reason_codes


def test_quarterly_observations_of_one_release_are_summed_within_the_year() -> None:
    """The register publishes quarters; annualising them is not double counting."""

    repository = TwoPublisherRepository()
    repository.drop_the_eea_releases()

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    cohort = _observed(repository, "vehicle-volkswagen-golf-eu", 2022)
    assert cohort is not None
    assert cohort.registrations == Decimal("96")
    assert cohort.input_observation_ids == tuple(
        sorted(f"observation-dft-golf-2022-q{quarter}" for quarter in range(1, 5))
    )


def test_the_only_publisher_present_decomposes_the_country_year() -> None:
    repository = TwoPublisherRepository()
    repository.drop_the_eea_releases()

    result = GenerationPlanningService().apply(
        repository, horizons=(2028,), seed=20260827
    )

    assert _observed(repository, "vehicle-register-only", 2020) is not None
    assert result.superseded_coverage_input_count == 0


def _add_stock(repository, generation, geography: str, stock_date: date, cohorts):
    """Observed active fleet by cohort, as the RDW register or DfT VEH0124 report it."""

    observations = tuple(
        SimpleNamespace(
            observation_id=f"stock-{generation.generation_id}-{geography.lower()}-{year}",
            release_id="release-stock",
            canonical_vehicle_id=generation.canonical_vehicle_id,
            geography=geography,
            registration_cohort_year=year,
            period_start=stock_date,
            period_end=stock_date,
            measure=Measure.ACTIVE_FLEET,
            value=Decimal(value),
        )
        for year, value in cohorts
    )
    repository.releases = (
        *repository.releases,
        SimpleNamespace(
            release_id="release-stock",
            dependency_group="national-register",
            source_id="national-register",
            publication_status="final",
        ),
    )
    repository.observations = (*repository.observations, *observations)
    repository.assignments = (
        *repository.assignments,
        *(
            SimpleNamespace(
                observation_id=item.observation_id,
                selected_generation_id=generation.generation_id,
                confidence=ConfidenceBand.LOW,
            )
            for item in observations
        ),
    )


def test_observed_stock_replaces_modelled_survival_for_complete_cohorts() -> None:
    """A register's count of cars still on the road beats a modelled survival.

    It also holds cohorts the registration source never saw: 2018-2019 here,
    standing for pre-2010 cohorts and used imports.
    """

    repository = Repository()
    survival = GenerationPlanningService().survival
    _add_stock(
        repository,
        repository.generation,
        "DE",
        date(2023, 10, 6),
        ((2018, "70"), (2019, "80"), (2020, "95"), (2021, "105"), (2022, "118"), (2023, "60")),
    )

    result = GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    assert result.stock_anchored_series_count == 1
    cohorts = {
        item.registration_cohort_year: item
        for item in repository.cohorts
        if item.as_of_year == 2028
    }
    # Complete cohorts come from the register, the partial 2023 cohort and later
    # ones from the registration forecast, so no year is counted twice.
    assert set(cohorts) == set(range(2018, 2029))
    for year in range(2018, 2023):
        assert "observed-stock-anchored-cohort" in cohorts[year].reason_codes
    for year in range(2023, 2029):
        assert "observed-stock-anchored-cohort" not in cohorts[year].reason_codes
    expected = (
        Decimal(95)
        * survival.interval(Decimal(1_000_000), age_years=8).p50
        / survival.interval(Decimal(1_000_000), age_years=3).p50
    )
    assert abs(cohorts[2020].active_fleet_p50 - expected) < Decimal("0.01")
    assert cohorts[2020].registrations == Decimal(95)
    assert len(repository.opportunities) == 1


def test_stock_alone_still_produces_demand_in_a_registration_market() -> None:
    repository = Repository()
    imported = _add_vehicle(repository, "import-only", ())
    _add_stock(repository, imported, "DE", date(2023, 10, 6), ((2015, "12"), (2016, "9")))

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    imported_opportunity = next(
        item
        for item in repository.opportunities
        if item.canonical_vehicle_id == "vehicle-import-only"
    )
    assert imported_opportunity.p50 > 0
    assert _cohort_years(repository, "vehicle-import-only") == {2015, 2016}


def test_stock_in_a_geography_without_registrations_is_never_added_on_top() -> None:
    """DfT's VEH0124 is UK-wide while registrations are GB; adding it would double count."""

    repository = Repository()
    _add_stock(repository, repository.generation, "UK", date(2025, 12, 31), ((2020, "500"),))

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    assert {item.geography for item in repository.cohorts} == {"DE"}
    assert {item.geography for item in repository.opportunities} == {"DE"}


def test_a_gap_in_a_year_the_market_reported_is_not_invented() -> None:
    """A complete country-year that omits a vehicle says it sold none under that name.

    Interpolating across it added registrations the publisher already filed
    under another label: 1.36M extra German 2012 registrations on the baseline.
    """

    repository = Repository()
    # Golf observed 2020 and 2022 only; another vehicle shows DE reported 2021.
    repository.observations[1].registration_cohort_year = 2022
    repository.observations[1].period_start = date(2022, 1, 1)
    repository.observations[1].period_end = date(2022, 12, 31)
    repository.observations = repository.observations[:2]
    repository.assignments = repository.assignments[:2]
    _add_vehicle(repository, "witness", ((2020, "5"), (2021, "5"), (2022, "5")))

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    golf_years = _cohort_years(repository, "vehicle-volkswagen-golf-eu")
    assert 2021 not in golf_years
    assert {2020, 2022} <= golf_years


def test_a_gap_in_a_year_the_market_never_reported_is_still_estimated() -> None:
    repository = Repository()
    repository.observations[1].registration_cohort_year = 2022
    repository.observations[1].period_start = date(2022, 1, 1)
    repository.observations[1].period_end = date(2022, 12, 31)
    repository.observations = repository.observations[:2]
    repository.assignments = repository.assignments[:2]

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    estimated = next(
        item
        for item in repository.cohorts
        if item.registration_cohort_year == 2021 and item.as_of_year == 2028
    )
    assert estimated.registrations == Decimal("105")
    assert "estimated-registration-cohort" in estimated.reason_codes


def test_a_forecast_never_fills_a_year_the_market_reported() -> None:
    """A reviewed generation that ended in 2022 but whose 2022 cars were filed
    under its successor must not get a forecast 2022 cohort on top."""

    repository = Repository()
    repository.generation.end_month = date(2023, 12, 1)
    repository.observations = repository.observations[:2]
    repository.assignments = repository.assignments[:2]
    _add_vehicle(repository, "successor", ((2022, "50"), (2023, "60")))

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    assert _cohort_years(repository, "vehicle-volkswagen-golf-eu") == {2020, 2021}


def test_a_reviewed_series_absent_from_a_later_reported_year_has_ended() -> None:
    """A reviewed window says when a generation may sell, not that it still does.

    Seven reviewed series absent from German 2025 still forecast 210k cars for
    2026 while their 2025 registrations sat under a new label.
    """

    repository = Repository()  # reviewed (identity kind unset), open window
    _add_vehicle(repository, "relabelled", ((2022, "50"), (2023, "60")))

    GenerationPlanningService().apply(repository, horizons=(2028,), seed=20260827)

    assert _cohort_years(repository, "vehicle-volkswagen-golf-eu") == {2020, 2021, 2022}
