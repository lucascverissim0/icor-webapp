"""Cars on the road per Eurocode, the evidence behind them, and what the model cannot see.

All codes and vehicles are fictional (`Q` prefix, which the real catalogue does
not use).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from icor.eurocodes.catalog import parse_rows
from icor.eurocodes.data_coverage import (
    FleetCell,
    FleetRange,
    OfficialStock,
    build_report,
    depth_tier,
)
from icor.eurocodes.demo import demo_eurocodes

AURORA = "aurora mobility|a1 horizon"
MERIDIAN = "meridian motors|m5 vector"


def _cell(identity, year, geography, evidence, cars, *, raw="A1 HORIZON", sources=()):
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
        bands={"Y_LT2": 100, "Y10-20": 400, "Y_GT20": 200},
        publisher_geography="UK",
    )
    modelled = {"GB": {2023: Decimal(90), 2010: Decimal(300), 2004: Decimal(50)}}

    report = _report(
        [_cell(AURORA, 2024, "GB", "registrations", 1)],
        official_stock={"GB": stock},
        official_fleet=modelled,
    )

    (gap,) = report["country_gap"]
    assert gap["publisher_geography"] == "UK"
    assert (gap["official"], gap["modelled"], gap["modelled_share"]) == (1_000, 440, 0.44)
    bands = {item["band"]: item for item in gap["bands"]}
    assert bands["Y_LT2"] == {"band": "Y_LT2", "official": 100, "modelled": 90}
    assert bands["Y10-20"]["modelled"] == 300
    assert bands["Y_GT20"]["modelled"] == 50
    assert bands["Y2-5"]["official"] is None
