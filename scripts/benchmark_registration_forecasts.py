#!/usr/bin/env python3
"""Compare the production registration forecast with its replaced baseline."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from icor.forecasting.registration_forecast import RegistrationForecaster
from icor.infrastructure.snapshot_store import SnapshotStore

_HOLDOUT_YEARS = 2
_MINIMUM_OBSERVED_YEARS = 5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run newest-period outer validation on the active ICOR snapshot."
    )
    parser.add_argument("--root", type=Path, default=Path(".local/evidence"))
    return parser


def _series(root: Path) -> tuple[str, tuple[dict[int, Decimal], ...]]:
    manifest, repository = SnapshotStore(root).open_active_snapshot()
    grouped: dict[tuple[str, str, str], dict[int, Decimal]] = defaultdict(dict)
    with sqlite3.connect(
        f"{repository.path.resolve().as_uri()}?mode=ro", uri=True
    ) as connection:
        rows = connection.execute(
            """SELECT generation_id, canonical_vehicle_id, geography,
                registration_cohort_year, registrations, reason_codes
            FROM cohort_estimate
            WHERE as_of_year = 2028
            AND registration_cohort_year <= 2025"""
        )
        for generation_id, vehicle_id, geography, year, value, reasons in rows:
            if "reconciled-registration-cohort" not in reasons:
                continue
            grouped[generation_id, vehicle_id, geography][int(year)] = Decimal(value)
    eligible = tuple(
        run
        for history in grouped.values()
        for run in _contiguous_runs(history)
        if len(run) >= _MINIMUM_OBSERVED_YEARS
    )
    return manifest.snapshot_id, eligible


def _contiguous_runs(history: dict[int, Decimal]) -> tuple[dict[int, Decimal], ...]:
    runs: list[dict[int, Decimal]] = []
    current: dict[int, Decimal] = {}
    previous: int | None = None
    for year in sorted(history):
        if previous is not None and year != previous + 1:
            runs.append(current)
            current = {}
        current[year] = history[year]
        previous = year
    if current:
        runs.append(current)
    return tuple(runs)


def _legacy_forecast(history: dict[int, Decimal], horizon_year: int) -> dict[int, Decimal]:
    candidates = (_recent_mean, _linear)
    years = sorted(history)
    scores = []
    for order, estimator in enumerate(candidates):
        errors = []
        for position in range(2, len(years)):
            training = {year: history[year] for year in years[:position]}
            actual = history[years[position]]
            predicted = max(Decimal(0), estimator(training, years[position]))
            errors.append(abs(predicted - actual) / max(actual, Decimal(1)))
        score = sum(errors, start=Decimal(0)) / max(len(errors), 1)
        scores.append((score, order, estimator))
    estimator = min(scores, key=lambda item: (item[0], item[1]))[2]
    return {
        year: max(Decimal(0), estimator(history, year))
        for year in range(years[-1] + 1, horizon_year + 1)
    }


def _recent_mean(history: dict[int, Decimal], _: int) -> Decimal:
    values = [history[year] for year in sorted(history)[-3:]]
    return sum(values, start=Decimal(0)) / len(values)


def _linear(history: dict[int, Decimal], target_year: int) -> Decimal:
    years = sorted(history)[-5:]
    origin = years[0]
    xs = [Decimal(year - origin) for year in years]
    ys = [history[year] for year in years]
    mean_x = sum(xs, start=Decimal(0)) / len(xs)
    mean_y = sum(ys, start=Decimal(0)) / len(ys)
    denominator = sum(((value - mean_x) ** 2 for value in xs), start=Decimal(0))
    slope = (
        Decimal(0)
        if denominator == 0
        else sum(
            (
                (x_value - mean_x) * (y_value - mean_y)
                for x_value, y_value in zip(xs, ys, strict=True)
            ),
            start=Decimal(0),
        )
        / denominator
    )
    return mean_y + slope * (Decimal(target_year - origin) - mean_x)


def main() -> int:
    args = _parser().parse_args()
    snapshot_id, histories = _series(args.root)
    production = RegistrationForecaster()
    production_error = Decimal(0)
    baseline_error = Decimal(0)
    denominator = Decimal(0)
    for history in histories:
        years = sorted(history)
        training_years = years[:-_HOLDOUT_YEARS]
        holdout_years = years[-_HOLDOUT_YEARS:]
        training = {year: history[year] for year in training_years}
        production_values = dict(
            production.forecast(
                training,
                horizon_year=holdout_years[-1],
                evaluate_backtest=False,
            ).values
        )
        baseline_values = _legacy_forecast(training, holdout_years[-1])
        for year in holdout_years:
            actual = history[year]
            production_error += abs(production_values[year] - actual)
            baseline_error += abs(baseline_values[year] - actual)
            denominator += max(actual, Decimal(1))
    if denominator == 0:
        raise RuntimeError("no eligible observed-only contiguous series were found")
    production_wape = production_error / denominator
    baseline_wape = baseline_error / denominator
    print(
        json.dumps(
            {
                "snapshot_id": snapshot_id,
                "method": production.method,
                "holdout_years": _HOLDOUT_YEARS,
                "minimum_observed_years": _MINIMUM_OBSERVED_YEARS,
                "series_evaluated": len(histories),
                "production_wape": str(production_wape.quantize(Decimal("0.000001"))),
                "replaced_baseline_wape": str(
                    baseline_wape.quantize(Decimal("0.000001"))
                ),
                "relative_error_reduction_percent": str(
                    (
                        (Decimal(1) - production_wape / baseline_wape) * Decimal(100)
                    ).quantize(Decimal("0.01"))
                ),
                "target_policy": (
                    "reconciled observed-source cohort rows only; no interpolated targets"
                ),
                "validation_limit": (
                    "current snapshot lacks sufficient as-of publication vintages"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
