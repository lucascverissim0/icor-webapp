#!/usr/bin/env python3
"""Benchmark a UK licensed-stock survival curve against ICOR's constant baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from icor.forecasting.survival import CohortSurvivalModel
from icor.forecasting.survival_calibration import (
    CohortStock,
    RegistrationCohort,
    calibrate_licensed_stock_curve,
)
from icor.infrastructure.snapshot_store import SnapshotStore

_FIRST_COMPLETE_YEAR = 2015
_LAST_FINAL_YEAR = 2025
_QUANTUM = Decimal("0.000001")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run leave-one-registration-cohort-out validation using official UK "
            "DfT registrations and licensed stock."
        )
    )
    parser.add_argument("--root", type=Path, default=Path(".local/evidence"))
    parser.add_argument("--uk-registrations", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_registrations(path: Path) -> tuple[RegistrationCohort, ...]:
    totals = {
        year: Decimal(0) for year in range(_FIRST_COMPLETE_YEAR, _LAST_FINAL_YEAR + 1)
    }
    with path.open(encoding="cp1252", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"BodyType"} | {
            f"{year} Q{quarter}"
            for year in totals
            for quarter in range(1, 5)
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"UK registrations CSV is missing columns: {sorted(missing)}")
        car_rows = 0
        for row_number, row in enumerate(reader, start=2):
            if row["BodyType"] != "Cars":
                continue
            car_rows += 1
            for year in totals:
                for quarter in range(1, 5):
                    raw = row[f"{year} Q{quarter}"].strip()
                    try:
                        value = Decimal(raw)
                    except Exception as error:
                        raise ValueError(
                            f"non-numeric UK registration value at row {row_number}"
                        ) from error
                    if not value.is_finite() or value < 0 or value != value.to_integral():
                        raise ValueError(
                            f"invalid UK registration value at row {row_number}"
                        )
                    totals[year] += value
    if car_rows == 0 or any(value <= 0 for value in totals.values()):
        raise ValueError("UK registrations CSV has incomplete Cars coverage")
    return tuple(RegistrationCohort(year, totals[year]) for year in sorted(totals))


def _load_stocks(database: Path) -> tuple[CohortStock, ...]:
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            """SELECT o.registration_cohort_year,
                CAST(SUBSTR(o.period_end, 1, 4) AS INTEGER) AS stock_year,
                SUM(CAST(o.value AS INTEGER)) AS licensed_vehicles
            FROM observation o
            JOIN source_release r ON r.release_id = o.release_id
            WHERE o.measure = 'active_fleet'
              AND o.registration_cohort_year IS NOT NULL
              AND r.source_id LIKE 'uk-dft-veh0124-%'
            GROUP BY o.registration_cohort_year, stock_year
            ORDER BY o.registration_cohort_year, stock_year"""
        ).fetchall()
    stocks = tuple(
        CohortStock(int(cohort_year), int(stock_year), Decimal(vehicles))
        for cohort_year, stock_year, vehicles in rows
        if vehicles is not None
    )
    if not stocks:
        raise RuntimeError("active snapshot has no UK DfT licensed-stock cohorts")
    return stocks


def _metric(value: Decimal) -> str:
    return str(value.quantize(_QUANTUM))


def main() -> int:
    args = _parser().parse_args()
    registration_path = args.uk_registrations.resolve(strict=True)
    manifest, repository = SnapshotStore(args.root).open_active_snapshot()
    registrations = _load_registrations(registration_path)
    stocks = _load_stocks(repository.path)
    registration_by_year = {item.cohort_year: item.registrations for item in registrations}
    stock_by_cohort = defaultdict(list)
    for stock in stocks:
        if stock.cohort_year in registration_by_year and stock.age_years >= 1:
            stock_by_cohort[stock.cohort_year].append(stock)

    baseline = CohortSurvivalModel()
    candidate_error = Decimal(0)
    baseline_error = Decimal(0)
    candidate_signed_error = Decimal(0)
    baseline_signed_error = Decimal(0)
    denominator = Decimal(0)
    points = 0
    by_age: dict[int, list[Decimal]] = defaultdict(
        lambda: [Decimal(0), Decimal(0), Decimal(0)]
    )
    evaluated_cohorts: list[int] = []
    for cohort_year in sorted(stock_by_cohort):
        curve = calibrate_licensed_stock_curve(
            registrations,
            stocks,
            excluded_cohort_years=frozenset({cohort_year}),
        )
        used = False
        for stock in stock_by_cohort[cohort_year]:
            if stock.age_years >= len(curve.shares) or stock.licensed_vehicles <= 0:
                continue
            registrations_value = registration_by_year[cohort_year]
            candidate = curve.remaining(registrations_value, age_years=stock.age_years)
            current = baseline.remaining(registrations_value, age_years=stock.age_years)
            actual = stock.licensed_vehicles
            candidate_error += abs(candidate - actual)
            baseline_error += abs(current - actual)
            candidate_signed_error += candidate - actual
            baseline_signed_error += current - actual
            denominator += actual
            points += 1
            used = True
            age_totals = by_age[stock.age_years]
            age_totals[0] += abs(candidate - actual)
            age_totals[1] += abs(current - actual)
            age_totals[2] += actual
        if used:
            evaluated_cohorts.append(cohort_year)

    if denominator <= 0 or baseline_error <= 0:
        raise RuntimeError("no eligible UK cohort-stock holdout points were found")
    candidate_wape = candidate_error / denominator
    baseline_wape = baseline_error / denominator
    report = {
        "snapshot_id": manifest.snapshot_id,
        "method": "uk-dft-licensed-stock-longitudinal-v1",
        "validation_design": "leave-one-registration-cohort-out",
        "evaluated_cohorts": evaluated_cohorts,
        "points_evaluated": points,
        "actual_vehicle_years": str(denominator.quantize(Decimal("1"))),
        "candidate_wape": _metric(candidate_wape),
        "constant_retention_wape": _metric(baseline_wape),
        "relative_error_reduction_percent": str(
            ((Decimal(1) - candidate_wape / baseline_wape) * Decimal(100)).quantize(
                Decimal("0.01")
            )
        ),
        "candidate_weighted_bias": _metric(candidate_signed_error / denominator),
        "constant_retention_weighted_bias": _metric(
            baseline_signed_error / denominator
        ),
        "by_age": {
            str(age): {
                "candidate_wape": _metric(values[0] / values[2]),
                "constant_retention_wape": _metric(values[1] / values[2]),
                "actual_vehicle_years": str(values[2].quantize(Decimal("1"))),
            }
            for age, values in sorted(by_age.items())
        },
        "registration_source": {
            "path": str(registration_path),
            "bytes": registration_path.stat().st_size,
            "sha256": _sha256(registration_path),
            "url": (
                "https://assets.publishing.service.gov.uk/media/"
                "6a54d2eca6586e258d371d71/df_VEH0160_UK.csv"
            ),
        },
        "target_policy": (
            "Cars only; complete calendar years 2015-2025; age one and older; "
            "licensed observations only"
        ),
        "validation_limit": (
            "aggregate UK Cars across makes and models; single current publication, "
            "not an as-of-origin backtest; UK licensed stock is an administrative proxy "
            "affected by imports, exports, relicensing, scrappage and record revisions; "
            "generation-level accuracy and transfer to other countries are unproven"
        ),
        "promotion_decision": "research_challenger_only",
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
