"""Eurostat's passenger cars by age, read from JSON-stat into the app's market codes."""

from __future__ import annotations

import pytest

from icor.evidence.eurostat_car_stock import parse


def _document(values: dict[tuple[str, str, str], int]) -> dict:
    geos, years, ages = ["EL", "UK", "DE"], ["2024", "2025"], ["TOTAL", "Y_LT2", "Y_GT20"]

    def index(items):
        return {"category": {"index": {item: position for position, item in enumerate(items)}}}

    flat = {}
    for (geo, year, age), value in values.items():
        position = ((geos.index(geo) * len(years)) + years.index(year)) * len(ages) + ages.index(
            age
        )
        flat[str(position)] = value
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
    }


VALUES = {
    ("EL", "2025", "TOTAL"): 600,
    ("UK", "2024", "TOTAL"): 3_000,
    ("UK", "2024", "Y_LT2"): 300,
    ("UK", "2024", "Y_GT20"): 150,
    ("DE", "2024", "TOTAL"): 4_800,
    ("DE", "2025", "TOTAL"): 4_900,
    ("DE", "2025", "Y_LT2"): 700,
}


def test_each_market_gets_its_newest_reported_year_in_app_codes():
    stocks = parse(_document(VALUES), {"GR", "GB", "DE", "FR"})

    assert set(stocks) == {"GR", "GB", "DE"}
    assert (stocks["GR"].year, stocks["GR"].total, stocks["GR"].publisher_geography) == (
        2025,
        600,
        "EL",
    )
    assert stocks["GB"].year == 2024
    assert stocks["GB"].publisher_geography == "UK"
    assert dict(stocks["GB"].bands) == {"Y_LT2": 300, "Y_GT20": 150}
    assert (stocks["DE"].year, stocks["DE"].total, stocks["DE"].publisher_geography) == (
        2025,
        4_900,
        None,
    )


def test_a_requested_year_is_kept_only_where_reported():
    stocks = parse(_document(VALUES), {"GR", "GB", "DE"}, year=2024)

    assert set(stocks) == {"GB", "DE"}
    assert stocks["DE"].total == 4_800


def test_only_car_counts_are_accepted():
    document = _document(VALUES)
    document["dimension"]["unit"] = {"category": {"index": {"PC": 0}}}

    with pytest.raises(ValueError, match="unit NR"):
        parse(document, {"DE"})
