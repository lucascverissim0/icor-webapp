"""Reading fleet cells at any year from a snapshot that stores only its horizons.

The snapshot holds 2028; other years are derived with the same survival curve,
and only after every stored 2028 value has been reproduced exactly.
"""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal

import pytest

from icor.application.generation_planning import GenerationPlanningService
from icor.forecasting.survival import (
    AgeAnchoredSurvivalModel,
    load_net_retained_survival_model,
    load_promoted_survival_model,
)
from icor.infrastructure.snapshot_data_coverage import read_fleet

SURVIVAL = load_promoted_survival_model()

_SCHEMA = """
CREATE TABLE canonical_vehicle (vehicle_id TEXT, make TEXT, model TEXT);
CREATE TABLE opportunity_estimate (canonical_vehicle_id TEXT);
CREATE TABLE cohort_estimate (
    cohort_id TEXT, generation_id TEXT, canonical_vehicle_id TEXT, geography TEXT,
    registration_cohort_year INTEGER, as_of_year INTEGER, registrations TEXT,
    active_fleet_p10 TEXT, active_fleet_p50 TEXT, active_fleet_p90 TEXT,
    survival_method TEXT, confidence TEXT, reason_codes TEXT);
CREATE TABLE cohort_input (cohort_id TEXT, observation_id TEXT, input_position INTEGER);
CREATE TABLE observation (
    observation_id TEXT, release_id TEXT, geography TEXT, measure TEXT, period_end TEXT,
    registration_cohort_year INTEGER);
CREATE TABLE source_release (release_id TEXT, source_id TEXT);
"""


def _registration_fleet(registrations: str, age: int, survival=SURVIVAL, geography=None):
    interval = survival.interval(Decimal(registrations), age_years=age, geography=geography)
    return str(interval.p10), str(interval.p50), str(interval.p90)


def _register_fleet(stock: str, *, age: int, stock_age: int, survival=SURVIVAL):
    planner = GenerationPlanningService(survival=survival)
    interval = planner._conditional_fleet(
        Decimal(stock), age_years=age, stock_age_years=stock_age, geography="NL"
    )
    return str(interval.p10), str(interval.p50), str(interval.p90)


def _snapshot(*, tamper: bool = False, survival=SURVIVAL) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(_SCHEMA)
    connection.executemany(
        "INSERT INTO canonical_vehicle VALUES (?, ?, ?)",
        [("v1", "AURORA MOBILITY", "A1 HORIZON"), ("v2", "MERIDIAN MOTORS", "M5 VECTOR")],
    )
    connection.executemany("INSERT INTO opportunity_estimate VALUES (?)", [("v1",), ("v2",)])
    registration = _registration_fleet("1000", 2028 - 2020, survival, "DE")
    register = _register_fleet("400", age=2028 - 2005, stock_age=2026 - 2005, survival=survival)
    forecast = _registration_fleet("300", 2028 - 2027, survival, "DE")
    if tamper:
        registration = (registration[0], str(Decimal(registration[1]) + 1), registration[2])
    rows = [
        ("c1", "v1", "DE", 2020, "1000", *registration, "reconciled-registration-cohort"),
        ("c2", "v2", "NL", 2005, "400", *register, "observed-stock-anchored-cohort"),
        ("c3", "v1", "DE", 2027, "300", *forecast, "forecast-registration-cohort"),
    ]
    connection.executemany(
        "INSERT INTO cohort_estimate VALUES (?, 'g', ?, ?, ?, 2028, ?, ?, ?, ?, 'm', 'x', ?)",
        [(*row[:-1], json.dumps([row[-1], "survival"])) for row in rows],
    )
    connection.executemany(
        "INSERT INTO observation VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("o1", "r1", "DE", "new_registrations", "2020-12-31", 2020),
            ("o2", "r2", "NL", "active_fleet", "2026-10-06", 2005),
        ],
    )
    connection.executemany(
        "INSERT INTO cohort_input VALUES (?, ?, 0)",
        [("c1", "o1"), ("c2", "o2"), ("c3", "o1")],
    )
    connection.executemany(
        "INSERT INTO source_release VALUES (?, ?)", [("r1", "pub-new"), ("r2", "pub-register")]
    )
    return connection


def test_fleet_at_another_year_follows_the_curve_the_snapshot_was_built_with():
    read = read_fleet(_snapshot(), SURVIVAL, year=2026, comparison_years={"NL": [2024, 2028]})

    assert read.verified_cohorts == 3
    cells = {(cell.geography, cell.cohort_year): cell for cell in read.cells}
    registered = cells["DE", 2020]
    assert registered.evidence == "registrations"
    assert registered.fleet.p50 == SURVIVAL.interval(Decimal(1000), age_years=6).p50
    assert registered.sources == frozenset({"pub-new"})
    # A register's cohort, seen at the register's own year, is the register's count.
    counted = cells["NL", 2005]
    assert counted.evidence == "stock"
    assert counted.fleet.p50 == Decimal(400)
    assert counted.sources == frozenset({"pub-register"})
    # A forecast cohort registered after the report year has no car on the road yet,
    # and its inputs are history, not evidence for the cohort.
    assert cells["DE", 2027].fleet is None
    assert cells["DE", 2027].sources == frozenset()
    assert set(read.comparison) == {"NL"}
    # Before its own date a register's cohort is its count, never aged back.
    assert read.comparison["NL"][2024][2005] == Decimal(400)
    assert read.comparison["NL"][2028][2005] == Decimal(
        _register_fleet("400", age=2028 - 2005, stock_age=2026 - 2005)[1]
    )
    assert read.cohort_evidence == {
        "DE": {2020: frozenset({"registrations"})},
        "NL": {2005: frozenset({"stock"})},
    }


def test_an_age_anchored_snapshot_is_read_as_planning_built_it():
    # Changing between the register's age (21) and the horizon's (23), so ageing with
    # the anchor would differ.
    anchors = tuple(Decimal("1.5") if age < 22 else Decimal("1.1") for age in range(41))
    survival = AgeAnchoredSurvivalModel(
        load_net_retained_survival_model(), {"DE": anchors, "NL": anchors}, method="test-v1"
    )

    read = read_fleet(
        _snapshot(survival=survival), survival, year=2026, comparison_years={"NL": [2028]}
    )

    assert read.verified_cohorts == 3
    cells = {(cell.geography, cell.cohort_year): cell for cell in read.cells}
    assert (
        cells["DE", 2020].fleet.p50
        == survival.interval(Decimal(1000), age_years=6, geography="DE").p50
    )
    # A register's cars are aged without the anchor, as planning ages them.
    assert read.comparison["NL"][2028][2005] == Decimal(
        _register_fleet("400", age=2028 - 2005, stock_age=2026 - 2005, survival=survival)[1]
    )


def test_a_snapshot_built_with_another_curve_is_refused():
    with pytest.raises(ValueError, match="another curve"):
        read_fleet(_snapshot(tamper=True), SURVIVAL, year=2026, comparison_years={})


def test_years_past_the_stored_horizon_are_refused():
    with pytest.raises(ValueError, match="2028"):
        read_fleet(_snapshot(), SURVIVAL, year=2030, comparison_years={})


def test_a_lower_bound_register_cohort_is_marked_at_least():
    connection = _snapshot()
    connection.execute(
        "UPDATE cohort_estimate SET reason_codes = ? WHERE cohort_id = 'c2'",
        (json.dumps(["observed-stock-anchored-cohort", "survival", "observed-stock-lower-bound"]),),
    )

    read = read_fleet(connection, SURVIVAL, year=2026, comparison_years={})

    cells = {(cell.geography, cell.cohort_year): cell for cell in read.cells}
    assert cells["NL", 2005].lower_bound is True
    assert cells["NL", 2005].evidence == "stock"
    assert cells["DE", 2020].lower_bound is False
    assert read.cohort_evidence["NL"] == {2005: frozenset({"lower-bound"})}


def test_official_registrations_the_model_lacks_are_estimated_for_the_country_only():
    from icor.eurocodes.data_coverage import OfficialStock
    from icor.evidence.eurostat_new_registrations import NewRegistrations
    from icor.infrastructure.snapshot_data_coverage import FleetRead, country_comparisons

    fleet = FleetRead(
        cells=(),
        comparison={"PL": {2025: {2012: Decimal(80), 2015: Decimal(90), 2008: Decimal(10)}}},
        verified_cohorts=0,
        cohort_evidence={
            "PL": {
                2012: frozenset({"registrations"}),
                2015: frozenset({"registrations"}),
                2008: frozenset({"lower-bound"}),
                2018: frozenset({"stock"}),
            }
        },
    )
    stock = OfficialStock(geography="PL", year=2025, total=1_000)
    registrations = {
        "PL": {
            # Before 2013 Poland counted used imports among new registrations.
            2012: NewRegistrations(200),
            2015: NewRegistrations(150),
            2016: NewRegistrations(100, "d"),
            2008: NewRegistrations(100),
            2018: NewRegistrations(500),
            1970: NewRegistrations(900),
        }
    }

    (comparison,) = country_comparisons(
        fleet, {"PL": stock}, registrations, {"PL": "3y"}, lambda _geo, _age: Decimal("0.5")
    ).values()

    assert comparison.convention == "3y"
    # Years without model-level registrations: survivors less the register's lower bound.
    assert comparison.before_sources[2025] == {2008: Decimal(40), 2016: Decimal(50)}
    # Years with registrations: only where the official count is comparable.
    assert comparison.unnamed[2025] == {}
    # 2012 has registrations, so its import-laden official count is not used at all.
    assert comparison.imports_in_registrations == frozenset({2008})
    # A full register count leaves nothing to estimate; cohorts past the curve's support neither.
    assert (
        2018 not in comparison.before_sources[2025] and 1970 not in comparison.before_sources[2025]
    )
