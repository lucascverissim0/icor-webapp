"""Cars on the road per Eurocode, the evidence behind them, and what the model cannot see.

All codes and vehicles are fictional (`Q` prefix, which the real catalogue does
not use).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from icor.eurocodes.catalog import parse_rows
from icor.eurocodes.data_coverage import (
    CountryComparison,
    FleetCell,
    FleetRange,
    OfficialStock,
    build_report,
    depth_tier,
)
from icor.eurocodes.demo import demo_eurocodes

AURORA = "aurora mobility|a1 horizon"
MERIDIAN = "meridian motors|m5 vector"


def _cell(
    identity, year, geography, evidence, cars, *, raw="A1 HORIZON", sources=(), lower_bound=False
):
    units = Decimal(cars)
    return FleetCell(
        identity=identity,
        raw_model=raw,
        geography=geography,
        cohort_year=year,
        evidence=evidence,
        fleet=FleetRange(units * Decimal("0.9"), units, units * Decimal("1.1")),
        fleet_2028=units * Decimal("0.8"),
        sources=frozenset(sources),
        lower_bound=lower_bound,
    )


def _report(cells, **options):
    return build_report(demo_eurocodes(), cells, year=2026, snapshot_id="demo", **options)


def _code(report, code):
    return next(item for item in report["codes"] if item["code"] == code)


def test_a_codes_cars_are_certain_or_shared_and_shared_cars_are_never_split():
    report = _report(
        [
            _cell(AURORA, 2024, "DE", "registrations", 100, sources={"pub-a"}),
            _cell(AURORA, 2025, "DE", "registrations", 40, sources={"pub-a"}),
            _cell(AURORA, 2026, "DE", "forecast", 30),
            _cell(AURORA, 2027, "DE", "forecast", 0),
        ]
    )

    old, new = _code(report, "Q201"), _code(report, "Q202")
    assert old["fleet"]["certain_p50"] == 100
    assert old["fleet"]["shared_p50"] == 70
    assert old["shared_with"] == ["Q202"]
    assert new["fleet"]["certain_p50"] == 0
    assert new["fleet"]["shared_p50"] == 70
    assert "shared-only" in new["flags"]
    assert "new" not in old["flags"]
    # The shared launch years count once in the market and in the totals.
    assert report["totals"]["fleet"] == 170
    assert report["totals"]["by_outcome"]["shared"] == 70


def test_every_car_is_accounted_for_by_outcome_and_by_evidence():
    report = _report(
        [
            _cell(AURORA, 2024, "DE", "registrations", 100),
            _cell(AURORA, 2019, "DE", "registrations", 9),
            _cell(MERIDIAN, 2024, "NL", "stock", 50, raw="M5 VECTOR"),
            _cell("nobody|x", 2024, "NL", "interpolated", 7, raw="X"),
        ]
    )

    totals = report["totals"]
    assert totals["fleet"] == 166
    assert totals["by_outcome"] == {
        "single": 150,
        "shared": 0,
        "outside": 9,
        "unassignable": 0,
        "unreviewed": 7,
    }
    assert totals["by_evidence"] == {
        "stock": 50,
        "registrations": 109,
        "interpolated": 7,
        "forecast": 0,
    }
    markets = {item["geography"]: item for item in report["markets"]}
    assert markets["NL"]["fleet"] == 57


def test_a_register_that_counts_only_part_of_its_cars_is_labelled_at_least():
    report = _report(
        [
            _cell(MERIDIAN, 2024, "NL", "stock", 50, raw="M5 VECTOR", lower_bound=True),
            _cell(MERIDIAN, 2024, "DE", "registrations", 30, raw="M5 VECTOR"),
            _cell(AURORA, 2024, "DE", "registrations", 10),
        ]
    )

    meridian, aurora = _code(report, "Q301"), _code(report, "Q201")
    assert meridian["lower_bound_p50"] == 50
    assert "lower-bound" in meridian["flags"]
    assert aurora["lower_bound_p50"] == 0
    assert "lower-bound" not in aurora["flags"]
    markets = {item["geography"]: item for item in report["markets"]}
    assert markets["NL"]["lower_bound"] == 50
    assert markets["DE"]["lower_bound"] == 0


def test_cars_registered_after_the_report_year_are_not_on_the_road_yet():
    late = FleetCell(AURORA, "A1 HORIZON", "DE", 2027, "forecast", None, Decimal(5))

    report = _report([_cell(AURORA, 2024, "DE", "registrations", 10), late])

    assert report["totals"]["fleet"] == 10


def test_an_unknown_kind_of_evidence_is_refused():
    with pytest.raises(ValueError, match="evidence kind"):
        _report([_cell(AURORA, 2024, "DE", "guess", 10)])


def test_one_markets_register_is_not_enough_for_tier_a():
    report = _report([_cell(MERIDIAN, 2024, "NL", "stock", 50, raw="M5 VECTOR")])

    meridian = _code(report, "Q301")
    assert meridian["observed_share"] == 1.0
    assert meridian["observed_markets"] == ["NL"]
    assert meridian["tier"] == "B"
    assert "one-market" in meridian["flags"]
    # Launched two years before the report year: mostly forecast cars, by nature.
    assert "new" in meridian["flags"]


def test_observed_cars_in_five_markets_reach_tier_a():
    markets = ("DE", "FR", "IT", "ES", "PL")
    report = _report(
        [_cell(MERIDIAN, 2024, market, "registrations", 10, raw="M5 VECTOR") for market in markets]
    )

    assert _code(report, "Q301")["tier"] == "A"


@pytest.mark.parametrize(
    ("share", "markets", "pre_2010", "tier"),
    [
        (None, 0, "0", "D"),
        ("0.95", 12, "0", "A"),
        ("0.95", 4, "0", "B"),
        ("0.6", 12, "0", "B"),
        ("0.3", 12, "0", "C"),
        ("0.95", 12, "0.1", "B"),
        ("0.95", 12, "0.5", "C"),
        ("0.3", 12, "0.1", "C"),
    ],
)
def test_depth_tier_is_capped_by_the_years_most_markets_cannot_see(share, markets, pre_2010, tier):
    observed = None if share is None else Decimal(share)

    assert depth_tier(observed, markets, Decimal(pre_2010)) == tier


def test_codes_without_any_car_are_listed_with_why():
    catalogue = parse_rows(
        (
            ("Q901", "AURORA MOBILITY", "A0 DART 3D HBK 1998-2004", 2004, 1998),
            ("Q902", "AURORA MOBILITY", "A9 HAULER VRACHTWAGEN 2015-", None, 2015),
        ),
        make_aliases={},
        source_sha256="c" * 64,
    )

    report = _report(
        [_cell(AURORA, 2024, "DE", "registrations", 10)], catalogue_rows=catalogue.rows
    )

    no_data = {item["code"]: item for item in report["no_data"]}
    # Reviewed codes no car reaches, and catalogue codes nobody reviewed.
    assert no_data["Q401"]["reviewed"] is True
    assert no_data["Q901"]["reviewed"] is False
    assert (no_data["Q901"]["first_year"], no_data["Q901"]["last_year"]) == (1998, 2004)
    # Trucks are out of scope.
    assert "Q902" not in no_data
    assert "Q201" not in no_data
    assert report["totals"]["no_data"]["not_reviewed"] == 1


def test_the_country_gap_compares_official_stock_band_by_band():
    stock = OfficialStock(
        geography="GB",
        year=2024,
        total=1_000,
        bands={"Y_LT2": 100, "Y2-5": 150, "Y5-10": 150, "Y10-20": 400, "Y_GT20": 200},
        publisher_geography="UK",
        status={"TOTAL": "i"},
        band_year=2024,
        band_total=1_000,
    )
    modelled = {2024: {2023: Decimal(90), 2010: Decimal(300), 2004: Decimal(50)}}

    report = _report(
        [_cell(AURORA, 2024, "GB", "registrations", 1)],
        countries={"GB": CountryComparison(stock, modelled, convention="2y")},
    )

    (gap,) = report["country_gap"]
    assert gap["publisher_geography"] == "UK"
    assert gap["official_status"] == "i"
    assert (gap["official"], gap["modelled"], gap["modelled_share"]) == (1_000, 440, 0.44)
    bands = {item["band"]: item for item in gap["bands"]}
    assert (bands["Y_LT2"]["official"], bands["Y_LT2"]["modelled"]) == (100, 90)
    assert bands["Y10-20"]["modelled"] == 300
    assert bands["Y_GT20"]["modelled"] == 50
    assert bands["Y2-5"]["modelled"] == 0
    assert bands["Y_LT2"]["modelled_alternative"] is None
    # Under twenty years: 800 official, 390 modelled.
    assert gap["under_20"]["official"] == 800
    assert gap["under_20"]["modelled"] == 390
    assert gap["under_20"]["estimated"]["missing"] == 410


def test_a_three_year_convention_moves_every_band_edge_up_a_year():
    stock = OfficialStock(
        geography="AT",
        year=2025,
        total=600,
        bands={"Y_LT2": 300, "Y2-5": 100, "Y5-10": 100, "Y10-20": 60, "Y_GT20": 40},
        band_year=2025,
        band_total=600,
    )
    # Ages 2, 5, 10 and 20: under three years they are the last year of each band.
    modelled = {2025: {2023: Decimal(10), 2020: Decimal(20), 2015: Decimal(30), 2005: Decimal(40)}}

    two = _report([], countries={"AT": CountryComparison(stock, modelled, convention="2y")})
    three = _report([], countries={"AT": CountryComparison(stock, modelled, convention="3y")})

    def bands(report):
        return {item["band"]: item["modelled"] for item in report["country_gap"][0]["bands"]}

    assert bands(two) == {"Y_LT2": 0, "Y2-5": 10, "Y5-10": 20, "Y10-20": 30, "Y_GT20": 40}
    assert bands(three) == {"Y_LT2": 10, "Y2-5": 20, "Y5-10": 30, "Y10-20": 40, "Y_GT20": 0}


def test_italys_convention_counts_four_registration_years_in_two_to_five():
    stock = OfficialStock(
        geography="IT",
        year=2025,
        total=400,
        bands={"Y_LT2": 100, "Y2-5": 100, "Y5-10": 100, "Y10-20": 100},
        band_year=2025,
        band_total=400,
        open_ended=True,
    )
    # Ages 1, 2, 5, 6, 9 and 10.
    modelled = {
        2025: {
            2024: Decimal(1),
            2023: Decimal(2),
            2020: Decimal(4),
            2019: Decimal(8),
            2016: Decimal(16),
            2015: Decimal(32),
        }
    }

    report = _report([], countries={"IT": CountryComparison(stock, modelled, convention="2y-4y")})

    bands = {item["band"]: item["modelled"] for item in report["country_gap"][0]["bands"]}
    assert bands == {"Y_LT2": 1, "Y2-5": 6, "Y5-10": 24, "Y10-20": 32}


def test_an_unclear_convention_is_reported_under_both_readings():
    stock = OfficialStock(
        geography="SE",
        year=2025,
        total=100,
        bands={"Y_LT2": 50, "Y2-5": 20, "Y5-10": 10, "Y10-20": 10, "Y_GT20": 10},
        band_year=2025,
        band_total=100,
    )
    modelled = {2025: {2023: Decimal(30)}}

    report = _report([], countries={"SE": CountryComparison(stock, modelled, convention="unclear")})

    band = report["country_gap"][0]["bands"][0]
    assert (band["band"], band["modelled"], band["modelled_alternative"]) == ("Y_LT2", 0, 30)


def test_an_office_without_a_band_over_twenty_years_counts_them_in_ten_to_twenty():
    stock = OfficialStock(
        geography="IT",
        year=2025,
        total=500,
        bands={"Y_LT2": 100, "Y2-5": 100, "Y5-10": 100, "Y10-20": 200},
        band_year=2025,
        band_total=500,
        open_ended=True,
    )
    modelled = {2025: {2012: Decimal(70), 1999: Decimal(30)}}

    report = _report([], countries={"IT": CountryComparison(stock, modelled, convention="2y")})

    (gap,) = report["country_gap"]
    assert [band["band"] for band in gap["bands"]] == ["Y_LT2", "Y2-5", "Y5-10", "Y10-20"]
    assert gap["bands"][-1]["modelled"] == 100
    assert gap["open_ended"] is True
    assert gap["under_20"] is None


def test_the_missing_cars_are_split_by_likely_reason_and_never_reach_a_code():
    stock = OfficialStock(
        geography="PL",
        year=2025,
        total=1_000,
        bands={"Y_LT2": 100, "Y2-5": 100, "Y5-10": 100, "Y10-20": 500, "Y_GT20": 200},
        band_year=2025,
        band_total=1_000,
    )
    modelled = {2025: {2012: Decimal(100), 2024: Decimal(120)}}
    before = {2025: {2008: Decimal(150)}}
    unnamed = {2025: {2012: Decimal(400)}}
    cells = [_cell(AURORA, 2012, "PL", "registrations", 100)]

    plain = _report(cells)
    report = _report(
        cells,
        countries={
            "PL": CountryComparison(
                stock, modelled, convention="2y", before_sources=before, unnamed=unnamed
            )
        },
    )

    bands = {item["band"]: item for item in report["country_gap"][0]["bands"]}
    # 400 missing at 10-19 years: 150 from years without model-level sources, the
    # rest of the unnamed estimate capped at what is left, nothing unexplained.
    assert bands["Y10-20"]["estimated"] == {
        "missing": 400,
        "before_sources": 150,
        "unnamed": 250,
        "other": 0,
        "above_official": 0,
    }
    # The model holds 20 more cars under two years than the office counts.
    assert bands["Y_LT2"]["estimated"]["above_official"] == 20
    assert bands["Y_LT2"]["estimated"]["missing"] == 0
    assert report["country_gap"][0]["estimated"]["missing"] == 780
    # Estimates never touch the codes or the market totals.
    assert report["codes"] == plain["codes"]
    assert report["markets"] == plain["markets"]
    assert report["totals"] == plain["totals"]
