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
from icor.forecasting.survival import load_promoted_survival_model
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


def _registration_fleet(registrations: str, age: int) -> tuple[str, str, str]:
    interval = SURVIVAL.interval(Decimal(registrations), age_years=age)
    return str(interval.p10), str(interval.p50), str(interval.p90)


def _register_fleet(stock: str, *, age: int, stock_age: int) -> tuple[str, str, str]:
    planner = GenerationPlanningService(survival=SURVIVAL)
    interval = planner._conditional_fleet(Decimal(stock), age_years=age, stock_age_years=stock_age)
    return str(interval.p10), str(interval.p50), str(interval.p90)


def _snapshot(*, tamper: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(_SCHEMA)
    connection.executemany(
        "INSERT INTO canonical_vehicle VALUES (?, ?, ?)",
        [("v1", "AURORA MOBILITY", "A1 HORIZON"), ("v2", "MERIDIAN MOTORS", "M5 VECTOR")],
    )
    connection.executemany("INSERT INTO opportunity_estimate VALUES (?)", [("v1",), ("v2",)])
    registration = _registration_fleet("1000", 2028 - 2020)
    register = _register_fleet("400", age=2028 - 2005, stock_age=2026 - 2005)
    forecast = _registration_fleet("300", 2028 - 2027)
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
    read = read_fleet(_snapshot(), SURVIVAL, year=2026, comparison_years={"NL": 2024})

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
    assert read.comparison["NL"][2005] == Decimal(
        _register_fleet("400", age=2024 - 2005, stock_age=2026 - 2005)[1]
    )


def test_a_snapshot_built_with_another_curve_is_refused():
    with pytest.raises(ValueError, match="another curve"):
        read_fleet(_snapshot(tamper=True), SURVIVAL, year=2026, comparison_years={})


def test_years_past_the_stored_horizon_are_refused():
    with pytest.raises(ValueError, match="2028"):
        read_fleet(_snapshot(), SURVIVAL, year=2030, comparison_years={})
