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


def json_stat_cells(document: Mapping[str, Any]) -> dict[tuple[str, ...], tuple[int, str | None]]:
    """Every reported cell as (value, status flag), keyed by its coordinates."""

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
    status = document.get("status") or {}
    cells: dict[tuple[str, ...], tuple[int, str | None]] = {}
    for flat, value in document["value"].items():
        if value is None:
            continue
        remainder = int(flat)
        coordinates: list[str] = []
        for name, size in zip(reversed(dimensions), reversed(sizes), strict=True):
            coordinates.append(labels[name][remainder % size])
            remainder //= size
        coordinates.reverse()
        cells[tuple(coordinates)] = (int(value), status.get(flat))
    return cells


#: The four bands below twenty years sum to the total within this share when an
#: office publishes no "Y_GT20" band and folds those cars into "Y10-20" (Italy, Czechia).
OPEN_ENDED_TOLERANCE = 0.005


def _complete_bands(values: Mapping[str, int]) -> bool | None:
    """True when every band is reported, False when the last one is open-ended, else None."""

    names = [name for name, _low, _high in AGE_BANDS]
    if "TOTAL" not in values or any(name not in values for name in names[:-1]):
        return None
    if names[-1] in values:
        return True
    below_twenty = sum(values[name] for name in names[:-1])
    if abs(below_twenty - values["TOTAL"]) <= OPEN_ENDED_TOLERANCE * values["TOTAL"]:
        return False
    return None


def parse(
    document: Mapping[str, Any], geographies: set[str], *, year: int | None = None
) -> dict[str, OfficialStock]:
    """Each requested market's official stock, in app codes.

    The total is the one of ``year``, else of the newest year that reports one.
    The age bands are those of the newest year, up to the total's, that reports
    them all; offices often publish a newer total before its age split. A year
    reporting only some bands is not used for bands, since its missing bands
    cannot be told apart from cars the office did not age. Each value keeps its
    Eurostat status flag (``i`` imputed, ``p`` provisional, ``b`` break in series).
    """

    dimensions = document["id"]
    if document["dimension"].get("unit", {}).get("category", {}).get("index", {}).keys() - {"NR"}:
        raise ValueError("expected car counts (unit NR) only")
    position = {name: dimensions.index(name) for name in ("geo", "time", "age")}
    by_market: dict[str, dict[int, dict[str, tuple[int, str | None]]]] = {}
    for coordinates, cell in json_stat_cells(document).items():
        publisher = coordinates[position["geo"]]
        geography = _GEOGRAPHY.get(publisher, publisher)
        if geography not in geographies:
            continue
        period = int(coordinates[position["time"]])
        ages = by_market.setdefault(geography, {}).setdefault(period, {})
        ages[coordinates[position["age"]]] = cell
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
        band_year = None
        open_ended = False
        for candidate in sorted(years, reverse=True):
            if candidate > stock_year:
                continue
            complete = _complete_bands({name: cell[0] for name, cell in years[candidate].items()})
            if complete is not None:
                band_year, open_ended = candidate, not complete
                break
        band_values = years[band_year] if band_year is not None else {}
        publisher = next((code for code, app in _GEOGRAPHY.items() if app == geography), geography)
        stocks[geography] = OfficialStock(
            geography=geography,
            year=stock_year,
            total=values["TOTAL"][0],
            bands={name: cell[0] for name, cell in band_values.items() if name in band_names},
            publisher_geography=publisher if publisher != geography else None,
            status={name: flag for name, (_value, flag) in values.items() if flag},
            band_year=band_year,
            band_total=band_values["TOTAL"][0] if band_year is not None else None,
            band_status={name: flag for name, (_value, flag) in band_values.items() if flag},
            open_ended=open_ended,
        )
    return stocks


def band_by_year(
    document: Mapping[str, Any], geographies: set[str], band: str
) -> dict[str, dict[int, int]]:
    """One age band's count for every year each requested market reports it."""

    dimensions = document["id"]
    position = {name: dimensions.index(name) for name in ("geo", "time", "age")}
    result: dict[str, dict[int, int]] = {}
    for coordinates, (value, _status) in json_stat_cells(document).items():
        if coordinates[position["age"]] != band:
            continue
        publisher = coordinates[position["geo"]]
        geography = _GEOGRAPHY.get(publisher, publisher)
        if geography in geographies:
            result.setdefault(geography, {})[int(coordinates[position["time"]])] = value
    return result
