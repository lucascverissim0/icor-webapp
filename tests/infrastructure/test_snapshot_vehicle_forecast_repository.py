from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from icor.forecasting.survival import CohortSurvivalModel
from icor.infrastructure.snapshot_vehicle_forecast_repository import (
    SnapshotVehicleForecastRepository,
    VehicleForecastSelectionError,
)


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE canonical_vehicle (
                vehicle_id TEXT PRIMARY KEY, make TEXT, model TEXT
            );
            CREATE TABLE generation_entry (
                generation_id TEXT PRIMARY KEY, canonical_vehicle_id TEXT,
                display_name TEXT, start_month TEXT, end_month TEXT,
                identity_kind TEXT, confidence_reasons TEXT
            );
            CREATE TABLE cohort_estimate (
                cohort_id TEXT PRIMARY KEY, generation_id TEXT,
                canonical_vehicle_id TEXT, geography TEXT,
                registration_cohort_year INTEGER, as_of_year INTEGER,
                registrations TEXT, active_fleet_p10 TEXT,
                active_fleet_p50 TEXT, active_fleet_p90 TEXT,
                survival_method TEXT, reason_codes TEXT
            );
            CREATE TABLE opportunity_estimate (
                opportunity_id TEXT PRIMARY KEY, generation_id TEXT,
                canonical_vehicle_id TEXT, geography TEXT, horizon_year INTEGER
            );
            CREATE TABLE opportunity_input (
                opportunity_id TEXT, cohort_id TEXT, input_position INTEGER
            );
            """
        )
        vehicles = (
            ("golf", "Volkswagen", "Golf"),
            ("gte", "Volkswagen VW", "Golf GTE"),
            ("plus", "Volkswagen", "Golf Plus"),
        )
        connection.executemany("INSERT INTO canonical_vehicle VALUES (?, ?, ?)", vehicles)
        for vehicle_id, _make, model in vehicles:
            generation_id = f"estimated-{vehicle_id}"
            connection.execute(
                "INSERT INTO generation_entry VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    generation_id,
                    vehicle_id,
                    f"estimated {model}",
                    "2010-01-01",
                    "2024-12-01",
                    "estimated",
                    '["estimated"]',
                ),
            )
        rows = (
            ("golf-be-2020", "golf", "BE", 2020, "100", "80", "90", "95", "reconciled"),
            ("golf-be-2020-estimate", "golf", "BE", 2020, "10", "8", "9", "10", "forecast"),
            ("gte-be-2021", "gte", "BE", 2021, "50", "40", "45", "48", "observed"),
            ("golf-fr-2020", "golf", "FR", 2020, "200", "160", "180", "190", "observed"),
            ("golf-be-2019", "golf", "BE", 2019, "25", "18", "20", "22", "reconciled"),
            ("golf-be-2018", "golf", "BE", 2018, "500", "300", "350", "400", "observed"),
            ("plus-be-2020", "plus", "BE", 2020, "900", "700", "800", "850", "observed"),
            ("golf-be-2026", "golf", "BE", 2026, "1000", "900", "950", "980", "forecast"),
        )
        for cohort_id, vehicle_id, geography, year, registrations, p10, p50, p90, status in rows:
            generation_id = f"estimated-{vehicle_id}"
            connection.execute(
                "INSERT INTO cohort_estimate "
                "VALUES (?, ?, ?, ?, ?, 2028, ?, ?, ?, ?, 'fixture-survival-v1', ?)",
                (
                    cohort_id,
                    generation_id,
                    vehicle_id,
                    geography,
                    year,
                    registrations,
                    p10,
                    p50,
                    p90,
                    f'["{status}-registration-cohort"]',
                ),
            )
            opportunity_id = f"opportunity-{cohort_id}"
            connection.execute(
                "INSERT INTO opportunity_estimate VALUES (?, ?, ?, ?, 2028)",
                (opportunity_id, generation_id, vehicle_id, geography),
            )
            connection.execute(
                "INSERT INTO opportunity_input VALUES (?, ?, 0)",
                (opportunity_id, cohort_id),
            )


@pytest.fixture
def repository(tmp_path: Path) -> SnapshotVehicleForecastRepository:
    path = tmp_path / "snapshot.sqlite3"
    _database(path)
    return SnapshotVehicleForecastRepository(path, "snapshot-test")


def test_options_support_text_search_then_year_or_generation_selection(
    repository: SnapshotVehicleForecastRepository,
) -> None:
    matches = repository.options(search="golf")
    brand_matches = repository.options(brand="Volkswagen")
    selected = repository.options(brand="Volkswagen", model="Golf")

    assert matches.brands == ("Volkswagen",)
    assert [(item.brand, item.model) for item in brand_matches.vehicles] == [
        ("Volkswagen", "Golf"),
        ("Volkswagen", "Golf Plus"),
    ]
    assert [(item.brand, item.model) for item in matches.vehicles] == [
        ("Volkswagen", "Golf"),
        ("Volkswagen", "Golf Plus"),
    ]
    assert selected.years == (2018, 2019, 2020, 2021)
    assert [item.name for item in selected.generations] == ["Golf Mk6", "Golf Mk7", "Golf Mk8"]
    assert selected.horizons == (2028,)


def test_verified_only_options_and_forecasts_reject_unreviewed_models(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snapshot.sqlite3"
    _database(path)
    repository = SnapshotVehicleForecastRepository(
        path, "snapshot-test", verified_only=True
    )

    matches = repository.options(search="golf")
    unavailable = repository.options(brand="Volkswagen", model="Golf Plus")

    assert [(item.brand, item.model) for item in matches.vehicles] == [
        ("Volkswagen", "Golf")
    ]
    assert unavailable.horizons == ()
    with pytest.raises(VehicleForecastSelectionError, match="reviewed generation"):
        repository.forecast(
            brand="Volkswagen",
            model="Golf Plus",
            year=2020,
            generation=None,
            horizon=2028,
        )


def test_client_model_year_catalog_includes_unreviewed_source_names(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snapshot.sqlite3"
    _database(path)
    repository = SnapshotVehicleForecastRepository(
        path, "snapshot-test", model_year_only=True
    )

    matches = repository.options(search="golf")
    selected = repository.options(brand="Volkswagen", model="Golf Plus")
    result = repository.forecast(
        brand="Volkswagen",
        model="Golf Plus",
        year=2020,
        generation=None,
        horizon=2028,
    )

    assert [(item.brand, item.model) for item in matches.vehicles] == [
        ("Volkswagen", "Golf"),
        ("Volkswagen", "Golf Plus"),
    ]
    assert selected.years == (2020,)
    assert selected.generations == ()
    assert result.generation_name == (
        "Volkswagen Golf Plus — 2020 registration cohort"
    )
    assert result.generation_basis == "official_source_registration_cohort"
    assert result.generation_confidence == "source-reported"
    assert result.included_cohort_years == (2020,)


def test_reviewed_generation_forecast_combines_aliases_and_surviving_cohorts(
    repository: SnapshotVehicleForecastRepository,
) -> None:
    """Selecting a generation aggregates every cohort and every make spelling."""

    result = repository.forecast(
        brand="Volkswagen", model="Golf", year=None, generation="Golf Mk8", horizon=2028
    )

    assert result.generation_name == "Golf Mk8"
    assert result.generation_basis == "manufacturer_generation_window"
    assert result.included_cohort_years == (2020, 2021)
    assert result.excluded_forecast_cohort_years == (2026,)
    assert result.excluded_ambiguous_years == (2019,)
    belgium = next(row for row in result.markets if row.code == "BE")
    europe = next(row for row in result.markets if row.code == "EU27")
    # 2021 is the `Volkswagen VW` Golf GTE cohort: a different spelling of the
    # same vehicle, which only counts because the identities are merged.
    assert belgium.registration_cohort_units == 160
    assert belgium.active_fleet.base_units == 144
    assert europe.registration_cohort_units == 360
    assert europe.active_fleet.base_units == 324
    assert 0 < belgium.replacements.downside_units <= belgium.replacements.base_units
    assert belgium.replacements.base_units <= belgium.replacements.upside_units
    assert all(
        row.code in {"EU27", "BE", "FR", "ES", "NL", "GB", "DE", "PL"} for row in result.markets
    )
    assert next(row for row in result.markets if row.code == "GB").availability == "unavailable"


def test_a_selected_registration_year_forecasts_that_year_and_no_other(
    repository: SnapshotVehicleForecastRepository,
) -> None:
    """Asking for 2020 must not quietly return the whole generation.

    A year used to resolve to a generation and then pull every cohort in it, so
    a vehicle whose generation window is one estimated block reported sixteen
    model years under the single year the user had picked.
    """

    result = repository.forecast(
        brand="Volkswagen", model="Golf", year=2020, generation=None, horizon=2028
    )

    assert result.selected_year == 2020
    assert result.included_cohort_years == (2020,)
    belgium = next(row for row in result.markets if row.code == "BE")
    assert belgium.registration_cohort_units == 110
    assert "2020 registration cohort" in result.generation_name


def test_a_transition_year_cohort_is_served_without_guessing_its_generation(
    repository: SnapshotVehicleForecastRepository,
) -> None:
    """2019 straddles Mk7 and Mk8, but the 2019 cohort itself is not ambiguous.

    Refusing the request protected a generation label the user had not asked
    for. The registration year is observed evidence, so it is served, and the
    generation is reported as source-derived rather than claimed as reviewed.
    """

    result = repository.forecast(
        brand="Volkswagen", model="Golf", year=2019, generation=None, horizon=2028
    )

    assert result.included_cohort_years == (2019,)
    assert result.generation_confidence == "source-reported"
    assert result.generation_basis == "official_source_registration_cohort"


def test_survival_method_is_read_from_the_model_not_a_literal(
    repository: SnapshotVehicleForecastRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This channel must report the curve that produced the rows it serves.

    The fleet quantiles come from `cohort_estimate`, so the only honest source of
    the method is the `survival_method` recorded on those same rows. Reporting a
    freshly constructed model instead made the channel claim
    `constant-annual-retention-v1` while serving cohorts built by the calibrated
    curve, which is how the ranking page and the forecast page came to disagree.
    """
    del monkeypatch

    result = repository.forecast(
        brand="Volkswagen", model="Golf", year=2020, generation=None, horizon=2028
    )

    assert result.survival_method == "fixture-survival-v1"
    assert result.survival_method != CohortSurvivalModel().method
