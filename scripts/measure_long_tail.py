#!/usr/bin/env python3
"""Measure how well a snapshot serves less popular vehicles.

Vehicles are the canonical make/model identities users search for (publisher
spellings folded by ``VehicleIdentityIndex``). They are split into ten volume
bands of officially observed EEA new registrations, 2010-2024, and for each
band the report states:

* ``identities`` -- how many vehicles fall in it;
* ``forecast_identity_share`` -- share of them with any replacement forecast;
* ``forecast_registration_share`` -- share of their observed registrations that
  belongs to a raw vehicle id with a forecast, which is what spelling
  fragmentation and the short-series rule lose;
* ``ranked_identity_share`` -- share whose 2031 base demand is above zero, the
  population the opportunity score ranks.

A fixed panel of real, lower-volume models is looked up the way the search page
does it, reporting whether each is found and whether it has a forecast.

Reads one snapshot database read-only, so it is safe beside a running app:

    uv run python scripts/measure_long_tail.py --database <snapshot>/evidence.sqlite3
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from icor.evidence.vehicle_identity import VehicleIdentityIndex, parse_vehicle_query

#: Real models that sell in the low tens of thousands or fewer across Europe,
#: chosen across makes, segments and launch years, including discontinued ones.
TAIL_PANEL: tuple[str, ...] = (
    "Alfa Romeo Tonale", "Alfa Romeo Giulietta", "Alfa Romeo MiTo", "DS 4", "DS 7",
    "Lancia Ypsilon", "Lancia Delta", "Subaru Outback", "Subaru Forester", "Subaru XV",
    "Jeep Avenger", "Jeep Renegade", "Jeep Compass", "Smart Forfour", "Cupra Born",
    "Cupra Formentor", "Mazda MX-30", "Mazda CX-30", "Mitsubishi ASX",
    "Mitsubishi Space Star", "Suzuki Ignis", "Suzuki Jimny", "SsangYong Tivoli",
    "SsangYong Korando", "Honda HR-V", "Honda Civic", "Lexus NX", "Lexus UX",
    "Infiniti Q30", "Alpine A110", "Polestar 2", "MG ZS", "MG 4", "Lynk & Co 01",
    "BYD Atto 3", "Dacia Lodgy", "Opel Cascada", "Fiat Tipo", "Seat Mii", "Kia Soul",
    "Hyundai Ioniq", "Nissan Leaf", "Renault Zoe", "Peugeot Rifter",
    "Citroen C4 Cactus", "Volvo V90", "Saab 9-3", "Chevrolet Cruze",
)

_FIRST_YEAR, _LAST_YEAR = 2010, 2024
_HORIZON = 2031
#: EEA registrations 2010-2024 summed over all markets.
_BANDS: tuple[tuple[float, float], ...] = (
    (1, 100), (100, 1_000), (1_000, 10_000), (10_000, 100_000), (100_000, float("inf")),
)


def _observed_volume(connection: sqlite3.Connection) -> dict[str, float]:
    """EEA new registrations per raw vehicle id, one publisher so none is doubled."""

    rows = connection.execute(
        """SELECT o.canonical_vehicle_id, SUM(CAST(o.value AS REAL))
        FROM observation o
        JOIN source_release r ON r.release_id = o.release_id
        WHERE o.measure = 'new_registrations'
            AND o.canonical_vehicle_id IS NOT NULL
            AND r.source_id LIKE 'eea%'
            AND CAST(substr(o.period_end, 1, 4) AS INTEGER) BETWEEN ? AND ?
        GROUP BY o.canonical_vehicle_id""",
        (_FIRST_YEAR, _LAST_YEAR),
    )
    return {vehicle_id: float(total or 0.0) for vehicle_id, total in rows}


def _forecast_demand(connection: sqlite3.Connection) -> dict[str, float]:
    rows = connection.execute(
        """SELECT canonical_vehicle_id, SUM(CAST(p50 AS REAL))
        FROM opportunity_estimate WHERE horizon_year = ?
        GROUP BY canonical_vehicle_id""",
        (_HORIZON,),
    )
    return {vehicle_id: float(total or 0.0) for vehicle_id, total in rows}


def summarize(
    vehicles: list[tuple[str, str, str]],
    volume: dict[str, float],
    demand: dict[str, float],
) -> dict[str, object]:
    """Pure aggregation, separated from SQLite so it can be tested directly."""

    index = VehicleIdentityIndex.from_rows(
        (vehicle_id, make, model, volume.get(vehicle_id, 0.0))
        for vehicle_id, make, model in vehicles
    )
    per_identity: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"volume": 0.0, "forecast_volume": 0.0, "demand": 0.0, "forecast": 0.0}
    )
    for vehicle_id, _make, _model in vehicles:
        key = index.identity_for(vehicle_id)
        if key is None:
            continue
        entry = per_identity[key]
        amount = volume.get(vehicle_id, 0.0)
        entry["volume"] += amount
        if vehicle_id in demand:
            entry["forecast"] = 1.0
            entry["forecast_volume"] += amount
            entry["demand"] += demand[vehicle_id]

    observed = [entry for entry in per_identity.values() if entry["volume"] > 0]
    # Fixed volume bands rather than deciles: most identities are publisher typos
    # with a handful of registrations, so deciles would describe typos, not cars.
    bands = []
    for low, high in _BANDS:
        members = [entry for entry in observed if low <= entry["volume"] < high]
        if not members:
            continue
        total = sum(entry["volume"] for entry in members)
        bands.append(
            {
                "registrations_from": low,
                "registrations_below": high if high != float("inf") else None,
                "identities": len(members),
                "forecast_identity_share": round(
                    sum(entry["forecast"] for entry in members) / len(members), 4
                ),
                "forecast_registration_share": round(
                    sum(entry["forecast_volume"] for entry in members) / total, 4
                )
                if total
                else 0.0,
                "ranked_identity_share": round(
                    sum(entry["demand"] > 0 for entry in members) / len(members), 4
                ),
            }
        )
    count = len(observed)

    forecast_index = VehicleIdentityIndex.from_rows(
        (vehicle_id, make, model, volume.get(vehicle_id, 0.0))
        for vehicle_id, make, model in vehicles
        if vehicle_id in demand
    )
    panel = []
    for name in TAIL_PANEL:
        observed_match = index.match(parse_vehicle_query(name), limit=1)
        searchable = forecast_index.match(parse_vehicle_query(name), limit=1)
        panel.append(
            {
                "vehicle": name,
                "observed": bool(observed_match),
                "searchable_with_forecast": bool(searchable),
                "observed_as": (
                    f"{observed_match[0].display_make} {observed_match[0].display_model}"
                    if observed_match
                    else None
                ),
            }
        )
    total_volume = sum(entry["volume"] for entry in observed)
    return {
        "identities_observed": count,
        "registrations_observed": round(total_volume),
        "forecast_registration_share": round(
            sum(entry["forecast_volume"] for entry in observed) / total_volume, 4
        )
        if total_volume
        else 0.0,
        "bands": bands,
        "tail_panel_found": sum(item["observed"] for item in panel),
        "tail_panel_with_forecast": sum(item["searchable_with_forecast"] for item in panel),
        "tail_panel_size": len(panel),
        "tail_panel": panel,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.database.is_file():
        print(f"no snapshot database at {args.database}", file=sys.stderr)
        return 2
    connection = sqlite3.connect(f"{args.database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        vehicles = [
            (vehicle_id, make, model)
            for vehicle_id, make, model in connection.execute(
                "SELECT vehicle_id, make, model FROM canonical_vehicle"
            )
        ]
        row = connection.execute("SELECT snapshot_id FROM snapshot").fetchone()
        result = summarize(vehicles, _observed_volume(connection), _forecast_demand(connection))
    finally:
        connection.close()
    manifest = args.database.parent / "snapshot.json"
    snapshot_id = (
        row[0]
        if row
        else json.loads(manifest.read_text())["snapshot_id"]
        if manifest.is_file()
        else None
    )
    result = {"snapshot_id": snapshot_id, **result}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
