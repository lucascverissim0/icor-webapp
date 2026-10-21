from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from icor.eurocodes.catalog import parse_rows
from icor.eurocodes.runtime import EurocodeMap
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
                canonical_vehicle_id TEXT, geography TEXT, horizon_year INTEGER,
                p10 TEXT, p50 TEXT, p90 TEXT, active_fleet_p50 TEXT,
                hazard_method TEXT, forecast_method TEXT, confidence TEXT,
                assumption_ids TEXT, reason_codes TEXT
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
                "INSERT INTO opportunity_estimate VALUES "
                "(?, ?, ?, ?, 2028, ?, ?, ?, ?, 'fixture-hazard', "
                "'fixture-forecast', 'medium', '[]', '[]')",
                (
                    opportunity_id,
                    generation_id,
                    vehicle_id,
                    geography,
                    p10,
                    p50,
                    p90,
                    p50,
                ),
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
    codes = [row.code for row in result.markets]
    # Every market the snapshot covers, not eight hand-picked ones: Europe first,
    # then the countries with data by size, then the rest as unavailable.
    assert codes[0] == "EU27"
    assert len(codes) == len(set(codes)) == 31
    assert {"IT", "SE", "AT", "NO", "IS", "GB"} <= set(codes)
    available = [row for row in result.markets[1:] if row.availability != "unavailable"]
    assert [row.code for row in result.markets[1 : 1 + len(available)]] == [
        row.code for row in available
    ]
    units = [row.registration_cohort_units for row in available]
    assert units == sorted(units, reverse=True)
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


def test_the_european_figure_names_the_members_that_are_not_in_it(
    repository: SnapshotVehicleForecastRepository,
) -> None:
    """An EU27 row is a partial sum; presenting it as Europe overstates it."""

    result = repository.forecast(
        brand="Volkswagen", model="Golf", year=2020, generation=None, horizon=2028
    )

    coverage = result.european_coverage
    assert coverage is not None
    assert coverage.contributing_markets == ("BE", "FR")
    assert "DE" in coverage.missing_markets
    assert "GB" not in coverage.missing_markets  # GB is not an EU27 member
    assert not set(coverage.contributing_markets) & set(coverage.missing_markets)


def test_the_demand_rank_places_the_selection_in_its_population(
    repository: SnapshotVehicleForecastRepository,
) -> None:
    """A percentile needs a population, so a single vehicle is ranked against one.

    It is on the same 0-100 scale as `DemandScoreV2`.
    """

    result = repository.forecast(
        brand="Volkswagen", model="Golf", year=2020, generation=None, horizon=2028
    )

    rank = result.demand_rank
    assert rank is not None
    assert 0.0 <= rank.percentile <= 1.0
    assert rank.demand_points == pytest.approx(rank.percentile * 100.0)
    assert 1 <= rank.rank <= rank.population
    assert rank.basis == "european_opportunity_p50_at_horizon"


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


def test_a_register_anchored_cohort_is_observed_evidence() -> None:
    """A vehicle known in one market only from a register must still be searchable."""

    from icor.infrastructure.snapshot_vehicle_forecast_repository import (
        _latest_evidence_year,
    )

    rows = [
        {"registration_cohort_year": 2019, "reason_codes": '["observed-stock-anchored-cohort"]'},
        {"registration_cohort_year": 2027, "reason_codes": '["forecast-registration-cohort"]'},
    ]

    assert _latest_evidence_year(rows) == 2019


def _golf_eurocodes() -> EurocodeMap:
    """Fictional codes (`Q` prefix, unused by the real catalogue); 2020 is a launch year."""

    catalogue = parse_rows(
        (
            ("Q701", "VOLKSWAGEN", "GOLF 5D HBK 2015-", None, 2015),
            ("Q702", "VOLKSWAGEN", "GOLF 5D HBK 2020-", None, 2020),
        ),
        make_aliases={},
        source_sha256="d" * 64,
    )
    return EurocodeMap.from_documents(
        catalogue,
        {
            "format": "icor-eurocode-decisions",
            "version": 1,
            "metadata": {"catalogue_sha256": "d" * 64},
            "families": {
                "volkswagen/golf": {
                    "label": "Volkswagen Golf",
                    "make": "volkswagen",
                    "codes": [
                        {"code": "Q701", "label": "Golf A", "first_year": 2015, "rows": [2]},
                        {
                            "code": "Q702",
                            "label": "Golf B",
                            "first_year": 2020,
                            "first_year_latest": 2021,
                            "replaces": ["Q701"],
                            "rows": [3],
                        },
                    ],
                }
            },
            "identities": {"volkswagen|golf": {"family": "volkswagen/golf"}},
        },
    )


def test_each_market_breaks_its_registrations_down_by_eurocode(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.sqlite3"
    _database(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE cohort_estimate SET registrations = '10.5' "
            "WHERE cohort_id = 'golf-be-2020-estimate'"
        )
    repository = SnapshotVehicleForecastRepository(
        path, "snapshot-test", eurocodes=_golf_eurocodes()
    )

    result = repository.forecast(
        brand="Volkswagen", model="Golf", year=2020, generation=None, horizon=2028
    )

    markets = {market.code: market for market in result.markets}
    for code, units in (("BE", 111), ("FR", 200), ("EU27", 311)):
        breakdown = markets[code].eurocodes
        assert markets[code].registration_cohort_units == units
        assert breakdown is not None and breakdown.total_units == units, code
        assert [(s.code, s.shared_units, s.shared_with) for s in breakdown.codes] == [
            ("Q701", units, ("Q702",)),
            ("Q702", units, ("Q701",)),
        ], code
    assert markets["DE"].availability == "unavailable"
    assert markets["DE"].eurocodes is None


def test_without_eurocodes_markets_carry_none(
    repository: SnapshotVehicleForecastRepository,
) -> None:
    result = repository.forecast(
        brand="Volkswagen", model="Golf", year=2020, generation=None, horizon=2028
    )

    assert all(market.eurocodes is None for market in result.markets)
