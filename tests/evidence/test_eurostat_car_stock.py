"""Eurostat's passenger cars by age, read from JSON-stat into the app's market codes."""

from __future__ import annotations

import pytest

from icor.evidence.eurostat_car_stock import band_by_year, parse

AGES = ["TOTAL", "Y_LT2", "Y2-5", "Y5-10", "Y10-20", "Y_GT20"]


def _document(
    values: dict[tuple[str, str, str], int], status: dict[tuple[str, str, str], str] | None = None
) -> dict:
    geos, years, ages = ["EL", "UK", "DE", "IT"], ["2023", "2024", "2025"], AGES

    def index(items):
        return {"category": {"index": {item: position for position, item in enumerate(items)}}}

    def position(geo, year, age):
        return str(
            ((geos.index(geo) * len(years)) + years.index(year)) * len(ages) + ages.index(age)
        )

    flat = {position(*key): value for key, value in values.items()}
    flags = {position(*key): flag for key, flag in (status or {}).items()}
    return {
        "id": ["freq", "unit", "geo", "time", "age"],
        "size": [1, 1, len(geos), len(years), len(ages)],
        "dimension": {
            "freq": index(["A"]),
            "unit": index(["NR"]),
            "geo": index(geos),
            "time": index(years),
            "age": index(ages),
        },
        "value": flat,
        "status": flags,
    }


VALUES = {
    ("EL", "2025", "TOTAL"): 600,
    ("UK", "2024", "TOTAL"): 3_000,
    ("UK", "2024", "Y_LT2"): 300,
    ("UK", "2024", "Y_GT20"): 150,
    ("DE", "2024", "TOTAL"): 1_000,
    **{("DE", "2024", age): 200 for age in AGES[1:]},
    ("DE", "2025", "TOTAL"): 1_100,
    ("DE", "2025", "Y_LT2"): 250,
    # Italy publishes no band over twenty years; "Y10-20" holds every older car.
    ("IT", "2025", "TOTAL"): 400,
    ("IT", "2025", "Y_LT2"): 100,
    ("IT", "2025", "Y2-5"): 100,
    ("IT", "2025", "Y5-10"): 100,
    ("IT", "2025", "Y10-20"): 100,
}


def test_each_market_gets_its_newest_total_in_app_codes():
    stocks = parse(_document(VALUES), {"GR", "GB", "DE", "FR"})

    assert set(stocks) == {"GR", "GB", "DE"}
    assert (stocks["GR"].year, stocks["GR"].total, stocks["GR"].publisher_geography) == (
        2025,
        600,
        "EL",
    )
    assert stocks["GB"].year == 2024
    assert stocks["GB"].publisher_geography == "UK"
    assert (stocks["DE"].year, stocks["DE"].total, stocks["DE"].publisher_geography) == (
        2025,
        1_100,
        None,
    )


def test_bands_come_from_the_newest_year_that_reports_them_all():
    stocks = parse(_document(VALUES), {"GR", "GB", "DE"})

    # 2025 reports only the youngest band: the bands are 2024's, with 2024's total.
    assert (stocks["DE"].band_year, stocks["DE"].band_total) == (2024, 1_000)
    assert dict(stocks["DE"].bands) == dict.fromkeys(AGES[1:], 200)
    # A year with only some bands never provides bands.
    assert (stocks["GB"].band_year, dict(stocks["GB"].bands)) == (None, {})
    assert stocks["GR"].band_year is None


def test_a_missing_over_20_band_that_the_others_sum_to_makes_10_20_open_ended():
    stocks = parse(_document(VALUES), {"IT"})

    assert stocks["IT"].band_year == 2025
    assert stocks["IT"].open_ended is True
    assert "Y_GT20" not in stocks["IT"].bands


def test_four_bands_that_do_not_sum_to_the_total_are_not_complete():
    values = {**VALUES, ("IT", "2025", "TOTAL"): 500}

    assert parse(_document(values), {"IT"})["IT"].band_year is None


def test_status_flags_are_kept_per_cell():
    status = {("DE", "2025", "TOTAL"): "p", ("DE", "2024", "Y_GT20"): "i"}

    stock = parse(_document(VALUES, status), {"DE"})["DE"]

    assert dict(stock.status) == {"TOTAL": "p"}
    assert dict(stock.band_status) == {"Y_GT20": "i"}


def test_a_requested_year_is_kept_only_where_reported():
    stocks = parse(_document(VALUES), {"GR", "GB", "DE"}, year=2024)

    assert set(stocks) == {"GB", "DE"}
    assert stocks["DE"].total == 1_000


def test_one_band_is_read_for_every_year():
    assert band_by_year(_document(VALUES), {"DE", "GB"}, "Y_LT2") == {
        "DE": {2024: 200, 2025: 250},
        "GB": {2024: 300},
    }


def test_only_car_counts_are_accepted():
    document = _document(VALUES)
    document["dimension"]["unit"] = {"category": {"index": {"PC": 0}}}

    with pytest.raises(ValueError, match="unit NR"):
        parse(document, {"DE"})
