"""Eurostat's new passenger-car registrations (`road_eqr_carmot`), read from JSON-stat.

National totals per year from 1970, with no make or model. They size the cars
registered before a market's first model-level source year (EEA monitoring names
models from 2010), and they measure the registrations a model-level source
leaves unnamed. Before 2013 some offices (BG, CY, LT, LV, PL, RO, SK) counted
used imports among new registrations; values flagged ``d`` differ in definition.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icor.evidence.eurostat_car_stock import json_stat_cells

DATASET = "road_eqr_carmot"
URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    f"{DATASET}?format=JSON&lang=EN"
)
LICENCE = "Eurostat reuse policy (Commission Decision 2011/833/EU), attribution required"

#: Offices whose new registrations included used imports before 2013, when the
#: series breaks (Eurostat metadata; visible as a level shift in the data).
USED_IMPORTS_BEFORE_2013 = frozenset({"BG", "CY", "LT", "LV", "PL", "RO", "SK"})

_GEOGRAPHY = {"EL": "GR", "UK": "GB"}
#: National totals from 2010, written by scripts/pin_eurostat_registration_totals.py.
TOTALS_PATH = Path(__file__).with_name("eurostat_registration_totals.json")


@dataclass(frozen=True, slots=True)
class NewRegistrations:
    count: int
    #: Eurostat status flag (``d`` definition differs, ``e``, ``b``, ``i``, ``p``).
    status: str | None = None


def parse(
    document: Mapping[str, Any], geographies: set[str]
) -> dict[str, dict[int, NewRegistrations]]:
    """Each requested market's new cars by year, all fuels and engines, in app codes."""

    dimensions = document["id"]
    for name, expected in (("unit", "NR"), ("mot_nrg", "TOTAL"), ("engine", "TOTAL")):
        if expected not in document["dimension"].get(name, {}).get("category", {}).get("index", {}):
            raise ValueError(f"expected {name} {expected} in {DATASET}")
    position = {name: dimensions.index(name) for name in ("unit", "mot_nrg", "engine", "geo")}
    time_position = dimensions.index("time")
    result: dict[str, dict[int, NewRegistrations]] = {}
    for coordinates, (value, status) in json_stat_cells(document).items():
        if (
            coordinates[position["unit"]] != "NR"
            or coordinates[position["mot_nrg"]] != "TOTAL"
            or coordinates[position["engine"]] != "TOTAL"
        ):
            continue
        publisher = coordinates[position["geo"]]
        geography = _GEOGRAPHY.get(publisher, publisher)
        if geography not in geographies:
            continue
        year = int(coordinates[time_position])
        result.setdefault(geography, {})[year] = NewRegistrations(value, status)
    return result


def load_committed_totals(path: Path | None = None) -> dict[tuple[str, int], int]:
    """The committed national new-car totals, by (market, year)."""

    document = json.loads((path or TOTALS_PATH).read_text(encoding="utf-8"))
    if document.get("dataset") != DATASET:
        raise ValueError("committed registration totals are not road_eqr_carmot")
    return {
        (geography, int(year)): int(count)
        for geography, years in document["totals"].items()
        for year, count in years.items()
    }


def includes_used_imports(geography: str, year: int) -> bool:
    return geography in USED_IMPORTS_BEFORE_2013 and year < 2013
