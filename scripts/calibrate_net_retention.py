#!/usr/bin/env python3
"""Calibrate per-market net retention on Eurostat and write the reviewed curve artifact.

Reads pinned copies of `road_eqs_carage` and `road_eqr_carmot` (fetched by
`scripts/report_eurocode_data.py --fetch-eurostat`), fits every market except
the United Kingdom (the survival curve's own) with
`icor.forecasting.net_retention`, and keeps a market only when its curve fitted
to 2022 predicts 2023-2025 better than the uncalibrated curve. Counts only.

usage: calibrate_net_retention.py [--stock FILE] [--registrations FILE] [--output FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

from icor.evidence import eurostat_age_conventions, eurostat_car_stock, eurostat_new_registrations
from icor.forecasting import net_retention
from icor.forecasting.survival import PROMOTED_CURVE_PATH, load_promoted_survival_model

ROOT = Path(__file__).resolve().parents[1]
EUROSTAT = ROOT / ".local" / "external-data" / "eurostat"
OUTPUT = PROMOTED_CURVE_PATH.parent / "eurostat_net_retention.json"
METHOD = "eurostat-net-retention-v1"
EXCLUDED = {"GB"}
FIT_UNTIL = 2025
BACKTEST_UNTIL = 2022


def _newest(dataset: str) -> Path:
    files = sorted(EUROSTAT.glob(f"{dataset}-*.json"))
    if not files:
        raise SystemExit(f"no pinned {dataset}; run report_eurocode_data.py --fetch-eurostat")
    return files[-1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def market_years(
    stock_document: dict, registration_document: dict
) -> dict[str, list[net_retention.MarketYear]]:
    """Each market's clean stock years: known convention, unflagged bands, usable registrations."""

    publishers = stock_document["dimension"]["geo"]["category"]["index"]
    geographies = {
        {"EL": "GR", "UK": "GB"}.get(code, code) for code in publishers if len(code) == 2
    } - EXCLUDED
    years = sorted(int(year) for year in stock_document["dimension"]["time"]["category"]["index"])
    registrations = eurostat_new_registrations.parse(registration_document, geographies)
    youngest = eurostat_car_stock.band_by_year(stock_document, geographies, "Y_LT2")
    second = eurostat_car_stock.band_by_year(stock_document, geographies, "Y2-5")
    result: dict[str, list[net_retention.MarketYear]] = {}
    for geography in sorted(geographies):
        market_registrations = registrations.get(geography, {})
        usable = {
            year: item.count
            for year, item in market_registrations.items()
            if item.status != "d"
            and not eurostat_new_registrations.includes_used_imports(geography, year)
        }
        conventions = eurostat_age_conventions.measure(
            youngest.get(geography, {}), market_registrations, geography, second.get(geography)
        )
        for year in years:
            convention = conventions.get(year)
            if convention is None or convention.convention == eurostat_age_conventions.UNCLEAR:
                continue
            stock = eurostat_car_stock.parse(stock_document, {geography}, year=year).get(geography)
            if stock is None or stock.band_year != year:
                continue
            official = {
                name: value
                for name, value in stock.bands.items()
                if name not in stock.band_status and "TOTAL" not in stock.band_status
            }
            result.setdefault(geography, []).append(
                net_retention.MarketYear(
                    year=year,
                    convention=convention.convention,
                    official=official,
                    registrations=usable,
                    open_ended=stock.open_ended,
                )
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=Path)
    parser.add_argument("--registrations", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    stock_path = args.stock or _newest(eurostat_car_stock.DATASET)
    registrations_path = args.registrations or _newest(eurostat_new_registrations.DATASET)
    stock_document = json.loads(stock_path.read_text(encoding="utf-8"))
    registration_document = json.loads(registrations_path.read_text(encoding="utf-8"))

    survival = load_promoted_survival_model()
    scale = Decimal(1_000_000)
    support = len(survival.band.p50.shares)
    shares = [float(survival.interval(scale, age_years=age).p50 / scale) for age in range(support)]

    def share(age: int) -> float:
        return shares[age]

    markets: dict[str, dict] = {}
    rejected: dict[str, dict] = {}
    for geography, items in market_years(stock_document, registration_document).items():
        goal = net_retention.targets(items, share, until=FIT_UNTIL)
        fitted = [item for item in items if item.year <= FIT_UNTIL]
        curve = net_retention.fit_curve(goal, fitted[-1], share, support=support)
        if curve is None:
            continue
        _backtest_curve, test = net_retention.backtest(
            items, share, fit_until=BACKTEST_UNTIL, support=support
        )
        entry = {
            "targets": {name: round(value, 4) for name, value in goal.items()},
            "years": [item.year for item in fitted if net_retention.band_ratios(item, share)],
            "backtest": {
                "fit_until": BACKTEST_UNTIL,
                "cells": len(test.cells),
                "uncalibrated_error": round(test.error(3), 4),
                "calibrated_error": round(test.error(4), 4),
            },
        }
        if test.passes:
            markets[geography] = {"multipliers": [f"{value:.4f}" for value in curve], **entry}
        else:
            rejected[geography] = entry
    document = {
        "method": METHOD,
        "base_method": survival.method,
        "excluded_markets": sorted(EXCLUDED),
        "tolerance": net_retention.TOLERANCE,
        "recent_years": net_retention.RECENT_YEARS,
        "fit_until": FIT_UNTIL,
        "inputs": {
            "stock": {"file": stock_path.name, "sha256": _sha256(stock_path)},
            "registrations": {
                "file": registrations_path.name,
                "sha256": _sha256(registrations_path),
            },
            "base_curve": {
                "file": PROMOTED_CURVE_PATH.name,
                "sha256": _sha256(PROMOTED_CURVE_PATH),
            },
        },
        "markets": markets,
        "rejected_markets": rejected,
    }
    args.output.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for geography, entry in sorted({**markets, **rejected}.items()):
        test = entry["backtest"]
        state = "calibrated" if geography in markets else "rejected"
        print(
            f"{geography} {state:<10} targets {entry['targets']} backtest "
            f"{test['uncalibrated_error']:.3f} -> {test['calibrated_error']:.3f} "
            f"({test['cells']} cells)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
