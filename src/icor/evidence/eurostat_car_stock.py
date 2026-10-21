"""Eurostat's passenger cars by age (`road_eqs_carage`), read from its JSON-stat file.

It is the official count of cars on the road per country, with five age bands
and no make or model. It cannot anchor a model's cohorts, but it measures how
many cars the model's fleet does not hold, and how old they are: cars
registered before a market's first model-level source year, and used imports,
which new-car registrations never see.

Eurostat's country codes differ from the app's in two places: Greece is `EL`
(the app's `GR`), and the United Kingdom is `UK`, which includes Northern
Ireland, while the app's registrations are for Great Britain (`GB`). The second
is a comparison of two slightly different territories and is labelled so.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from icor.eurocodes.data_coverage import AGE_BANDS, OfficialStock

DATASET = "road_eqs_carage"
URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    f"{DATASET}?format=JSON&lang=EN"
)
LICENCE = "Eurostat reuse policy (Commission Decision 2011/833/EU), attribution required"

_GEOGRAPHY = {"EL": "GR", "UK": "GB"}


def _cells(document: Mapping[str, Any]) -> dict[tuple[str, ...], int]:
    dimensions = document["id"]
    sizes = document["size"]
    if len(dimensions) != len(sizes):
        raise ValueError("JSON-stat dimensions and sizes disagree")
    labels = {
        name: {
            position: key
            for key, position in document["dimension"][name]["category"]["index"].items()
        }
        for name in dimensions
    }
    cells: dict[tuple[str, ...], int] = {}
    for flat, value in document["value"].items():
        if value is None:
            continue
        remainder = int(flat)
        coordinates: list[str] = []
        for name, size in zip(reversed(dimensions), reversed(sizes), strict=True):
            coordinates.append(labels[name][remainder % size])
            remainder //= size
        coordinates.reverse()
        cells[tuple(coordinates)] = int(value)
    return cells


def parse(
    document: Mapping[str, Any], geographies: set[str], *, year: int | None = None
) -> dict[str, OfficialStock]:
    """Each requested market's total in ``year``, else its newest reported year, in app codes."""

    dimensions = document["id"]
    if document["dimension"].get("unit", {}).get("category", {}).get("index", {}).keys() - {"NR"}:
        raise ValueError("expected car counts (unit NR) only")
    position = {name: dimensions.index(name) for name in ("geo", "time", "age")}
    by_market: dict[str, dict[int, dict[str, int]]] = {}
    for coordinates, value in _cells(document).items():
        publisher = coordinates[position["geo"]]
        geography = _GEOGRAPHY.get(publisher, publisher)
        if geography not in geographies:
            continue
        period = int(coordinates[position["time"]])
        ages = by_market.setdefault(geography, {}).setdefault(period, {})
        ages[coordinates[position["age"]]] = value
    stocks = {}
    band_names = {name for name, _low, _high in AGE_BANDS}
    for geography, years in by_market.items():
        reported = [item for item, values in years.items() if "TOTAL" in values]
        if year is not None:
            reported = [item for item in reported if item == year]
        if not reported:
            continue
        stock_year = max(reported)
        values = years[stock_year]
        publisher = next((code for code, app in _GEOGRAPHY.items() if app == geography), geography)
        stocks[geography] = OfficialStock(
            geography=geography,
            year=stock_year,
            total=values["TOTAL"],
            bands={name: count for name, count in values.items() if name in band_names},
            publisher_geography=publisher if publisher != geography else None,
        )
    return stocks
