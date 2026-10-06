#!/usr/bin/env python3
"""Check that summing per-vehicle registration forecasts does not bias markets.

The ranking adds up thousands of per-vehicle forecasts, so a small per-series
bias becomes a large market one. From an origin year, every reconciled series
observed at the origin is forecast on its latest unbroken run with the
production forecaster, the forecasts are summed per market, and the sums are
compared with the reconciled actual totals. Forecasting each market's total
directly is reported beside it as the alternative.

    uv run python scripts/backtest_market_totals.py --database <snapshot>/evidence.sqlite3
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from icor.forecasting.registration_forecast import RegistrationForecaster


def _forecast(
    forecaster: RegistrationForecaster, history: dict[int, Decimal], origin: int, last: int
) -> dict[int, Decimal] | None:
    known = {year: value for year, value in history.items() if year <= origin}
    if origin not in known:
        return None
    start = origin
    while start - 1 in known:
        start -= 1
    run = {year: value for year, value in known.items() if year >= start}
    if len(run) == 1:
        return {year: run[origin] for year in range(origin + 1, last + 1)}
    return dict(forecaster.forecast(run, horizon_year=last, evaluate_backtest=False).values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--origin", type=int, default=2023)
    parser.add_argument("--last", type=int, default=2025)
    args = parser.parse_args(argv)
    targets = range(args.origin + 1, args.last + 1)
    connection = sqlite3.connect(f"{args.database.resolve().as_uri()}?mode=ro", uri=True)
    series: dict[tuple[str, str, str], dict[int, Decimal]] = defaultdict(dict)
    try:
        rows = connection.execute(
            """SELECT generation_id, canonical_vehicle_id, geography,
                registration_cohort_year, registrations
            FROM cohort_estimate WHERE as_of_year = 2031
            AND reason_codes LIKE '%"reconciled-registration-cohort"%'"""
        )
        for generation, vehicle, geography, year, registrations in rows:
            series[generation, vehicle, geography][int(year)] = Decimal(registrations)
    finally:
        connection.close()
    forecaster = RegistrationForecaster()
    actual: dict[str, dict[int, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    summed: dict[str, dict[int, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    market: dict[str, dict[int, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    for (_generation, _vehicle, geography), history in series.items():
        for year, value in history.items():
            market[geography][year] += value
            if year in targets:
                actual[geography][year] += value
        forecast = _forecast(forecaster, history, args.origin, args.last)
        if forecast:
            for year in targets:
                summed[geography][year] += forecast[year]
    totals = defaultdict(Decimal)
    markets = {}
    for geography in sorted(actual):
        known = {year: value for year, value in market[geography].items() if year <= args.origin}
        if len(known) < 3 or args.origin not in known:
            continue
        direct = dict(
            forecaster.forecast(known, horizon_year=args.last, evaluate_backtest=False).values
        )
        for year in targets:
            totals["actual"] += actual[geography][year]
            totals["summed"] += summed[geography][year]
            totals["direct"] += direct[year]
            totals["summed_error"] += abs(summed[geography][year] - actual[geography][year])
            totals["direct_error"] += abs(direct[year] - actual[geography][year])
        markets[geography] = {
            str(year): {
                "actual": int(actual[geography][year]),
                "summed": int(summed[geography][year]),
                "direct": int(direct[year]),
            }
            for year in targets
        }
    print(
        json.dumps(
            {
                "origin": args.origin,
                "targets": list(targets),
                "summed_bias": round(float(totals["summed"] / totals["actual"] - 1), 4),
                "direct_bias": round(float(totals["direct"] / totals["actual"] - 1), 4),
                "summed_market_wape": round(float(totals["summed_error"] / totals["actual"]), 4),
                "direct_market_wape": round(float(totals["direct_error"] / totals["actual"]), 4),
                "markets": markets,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
