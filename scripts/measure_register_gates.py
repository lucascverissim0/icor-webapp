#!/usr/bin/env python3
"""Measure a national register aggregate against a snapshot's registrations, before integrating it.

Counts only. The gates a register must pass (docs/CODEX_HANDOFF.md, Mission 3):

- name match: of the register's 2010+ cars, the share whose real vehicle (publisher
  spellings folded with the snapshot's own vocabulary) the market's registrations
  already hold, and of the market's 2010+ registrations, the share whose vehicle the
  register names;
- loss: real vehicles with at least 2,000 registrations 2010+ that the register would
  leave with less than half of the cars their registrations leave on the road at its
  date, as a share of registrations (a complete register replaces them). Where a
  market's cars leave it young (Luxembourg's lease cars), every vehicle falls short
  alike, so the gate compares each vehicle with its market's own cohort ratio
  (`relative`), which isolates names and masking from real outflow;
- masked: cars the register hides or cannot place in a cohort;
- refold: snapshot vehicles whose identity changes once the register's labels exist
  (must be zero; the labels never teach the vocabulary).

usage: measure_register_gates.py --register CSV --geography LU --stock-date 2026-10-06
       [--masks LABEL ...] [--candidate SNAPSHOT_ID]
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

from icor.evidence.normalization import normalize_vehicle_label
from icor.evidence.vehicle_identity import VehicleIdentityIndex
from icor.forecasting.survival import load_promoted_survival_model
from icor.infrastructure.snapshot_identity import _vocabulary_volume, load_identity_index
from icor.infrastructure.snapshot_store import SnapshotStore

ROOT = Path(__file__).resolve().parents[1]
FIRST_MODEL_YEAR = 2010
LOSS_SUPPORT = 2_000
LOSS_RATIO = Decimal("0.5")


def _ledger(root: Path, candidate: str | None) -> tuple[str, Path]:
    if candidate:
        return candidate, root / "candidates" / candidate / "evidence.sqlite3"
    manifest, ledger = SnapshotStore(root, trust_verified_marker=True).open_active_snapshot()
    return manifest.snapshot_id, ledger.path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", required=True, type=Path)
    parser.add_argument("--geography", required=True)
    parser.add_argument("--stock-date", required=True, type=date.fromisoformat)
    parser.add_argument("--masks", nargs="*", default=())
    parser.add_argument("--root", type=Path, default=ROOT / ".local" / "evidence")
    parser.add_argument("--candidate")
    parser.add_argument("--horizon", type=int, default=2028)
    args = parser.parse_args()
    masks = {normalize_vehicle_label(item) for item in args.masks}
    last_cohort = (
        args.stock_date.year
        if (args.stock_date.month, args.stock_date.day) == (12, 31)
        else args.stock_date.year - 1
    )

    register: dict[tuple[str, str], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    totals = {"cars": 0, "masked": 0, "blank": 0, "no_cohort": 0, "cars_2010": 0, "masked_2010": 0}
    with args.register.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            vehicles = int(row["vehicles"])
            totals["cars"] += vehicles
            cohort = int(row["cohort"]) if row["cohort"].isdigit() else None
            in_scope = cohort is not None and FIRST_MODEL_YEAR <= cohort <= last_cohort
            make = normalize_vehicle_label(row["make"])
            model = normalize_vehicle_label(row["model"])
            if cohort is None:
                totals["no_cohort"] += vehicles
            if in_scope:
                totals["cars_2010"] += vehicles
            if make is None or model is None:
                totals["blank"] += vehicles
                totals["masked_2010"] += vehicles if in_scope else 0
                continue
            if make in masks or model in masks:
                totals["masked"] += vehicles
                totals["masked_2010"] += vehicles if in_scope else 0
                continue
            if cohort is not None:
                register[row["make"], row["model"]][cohort] += vehicles

    snapshot_id, ledger = _ledger(args.root, args.candidate)
    connection = sqlite3.connect(f"{ledger.resolve().as_uri()}?mode=ro", uri=True)
    try:
        baseline = load_identity_index(connection)
        rows = connection.execute(
            "SELECT v.vehicle_id, v.make, v.model, COALESCE(c.registrations, 0) "
            "FROM canonical_vehicle v JOIN (SELECT DISTINCT canonical_vehicle_id AS vehicle_id "
            "FROM opportunity_estimate) o ON o.vehicle_id = v.vehicle_id LEFT JOIN ("
            "SELECT canonical_vehicle_id AS vehicle_id, SUM(CAST(registrations AS REAL)) "
            "AS registrations FROM cohort_estimate GROUP BY canonical_vehicle_id) c "
            "ON c.vehicle_id = v.vehicle_id"
        ).fetchall()
        teaching = _vocabulary_volume(connection, rows)
        if teaching is None:
            teaching = {vehicle_id: float(volume or 0) for vehicle_id, _m, _n, volume in rows}
        registered: dict[str, dict[int, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
        for vehicle_id, year, value, codes in connection.execute(
            "SELECT canonical_vehicle_id, registration_cohort_year, registrations, reason_codes "
            "FROM cohort_estimate WHERE geography = ? AND as_of_year = ? "
            "AND registration_cohort_year BETWEEN ? AND ?",
            (args.geography, args.horizon, FIRST_MODEL_YEAR, last_cohort),
        ):
            first = json.loads(codes)[0]
            if first in {"observed-registration-cohort", "reconciled-registration-cohort"}:
                registered[vehicle_id][year] += Decimal(value)
    finally:
        connection.close()

    register_ids = {label: f"register:{label[0]}|{label[1]}" for label in register}
    index = VehicleIdentityIndex.from_rows(
        [*rows, *((vehicle_id, *label, 0.0) for label, vehicle_id in register_ids.items())],
        vocabulary_volume=teaching,
    )
    refold = sum(
        1
        for vehicle_id, *_ in rows
        if index.identity_for(vehicle_id) != baseline.identity_for(vehicle_id)
    )

    def real(vehicle_id: str) -> tuple[str, str]:
        return index.identity_for(vehicle_id) or ("", vehicle_id)

    survival = load_promoted_survival_model()
    surviving: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    registrations: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    surviving_by_year: dict[int, Decimal] = defaultdict(Decimal)
    surviving_cells: dict[tuple[str, str], dict[int, Decimal]] = defaultdict(
        lambda: defaultdict(Decimal)
    )
    for vehicle_id, years in registered.items():
        for year, value in years.items():
            alive = survival.interval(value, age_years=args.stock_date.year - year).p50
            surviving[real(vehicle_id)] += alive
            surviving_cells[real(vehicle_id)][year] += alive
            registrations[real(vehicle_id)] += value
            surviving_by_year[year] += alive
    stock: dict[tuple[str, str], int] = defaultdict(int)
    stock_by_year: dict[int, int] = defaultdict(int)
    for label, years in register.items():
        for year, vehicles in years.items():
            if FIRST_MODEL_YEAR <= year <= last_cohort:
                stock[real(register_ids[label])] += vehicles
                stock_by_year[year] += vehicles

    named = sum(stock.values())
    named_known = sum(count for key, count in stock.items() if key in registrations)
    volume = sum(registrations.values())
    volume_named = sum(value for key, value in registrations.items() if key in stock)
    losses = sorted(
        (
            (key, registrations[key], surviving[key], stock.get(key, 0))
            for key in registrations
            if registrations[key] >= LOSS_SUPPORT
            and Decimal(stock.get(key, 0)) < LOSS_RATIO * surviving[key]
        ),
        key=lambda item: -item[1],
    )
    loss_volume = sum(item[1] for item in losses)
    market_ratio = {
        year: Decimal(stock_by_year.get(year, 0)) / value
        for year, value in surviving_by_year.items()
        if value > 0
    }
    expected = {
        key: sum(value * market_ratio.get(year, Decimal(0)) for year, value in cells.items())
        for key, cells in surviving_cells.items()
    }
    relative = sorted(
        (
            (key, registrations[key], expected[key], stock.get(key, 0))
            for key in registrations
            if registrations[key] >= LOSS_SUPPORT
            and Decimal(stock.get(key, 0)) < LOSS_RATIO * expected[key]
        ),
        key=lambda item: -item[1],
    )
    relative_volume = sum(item[1] for item in relative)
    result = {
        "snapshot_id": snapshot_id,
        "geography": args.geography,
        "register": args.register.name,
        "cohorts": [FIRST_MODEL_YEAR, last_cohort],
        "register_cars": totals,
        "masked_share_2010": round(totals["masked_2010"] / max(totals["cars_2010"], 1), 4),
        "name_match_register": round(named_known / max(named, 1), 4),
        "name_match_registrations": round(float(volume_named / max(volume, 1)), 4),
        "loss_identities": len(losses),
        "loss_share": round(float(loss_volume / max(volume, 1)), 4),
        "largest_losses": [
            {
                "vehicle": " ".join(key),
                "registrations": int(reg),
                "surviving": int(alive),
                "register": count,
            }
            for key, reg, alive, count in losses[:15]
        ],
        "relative_loss_identities": len(relative),
        "relative_loss_share": round(float(relative_volume / max(volume, 1)), 4),
        "largest_relative_losses": [
            {
                "vehicle": " ".join(key),
                "registrations": int(reg),
                "expected": int(value),
                "register": count,
            }
            for key, reg, value, count in relative[:15]
        ],
        "refold": refold,
        "by_cohort": {
            str(year): {
                "register": stock_by_year.get(year, 0),
                "surviving_registrations": int(surviving_by_year.get(year, 0)),
            }
            for year in range(FIRST_MODEL_YEAR, last_cohort + 1)
        },
    }
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
