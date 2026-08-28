"""Simple registration forecasts selected by deterministic rolling-origin error."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

_QUANTUM = Decimal("0.0001")


@dataclass(frozen=True, slots=True)
class RegistrationForecast:
    method: str
    values: tuple[tuple[int, Decimal], ...]
    backtest_error: Decimal


class RegistrationForecaster:
    """Choose constant-mean or linear-trend baselines with expanding backtests."""

    def forecast(
        self, history: dict[int, Decimal], *, horizon_year: int
    ) -> RegistrationForecast:
        if len(history) < 2:
            raise ValueError("registration forecasting requires at least two annual values")
        years = sorted(history)
        if years != list(range(years[0], years[-1] + 1)):
            raise ValueError("registration history must be annual and contiguous")
        if horizon_year <= years[-1]:
            raise ValueError("registration forecast horizon must follow observed history")
        if any(not value.is_finite() or value < 0 for value in history.values()):
            raise ValueError("registration history must be finite and non-negative")

        candidates = (
            ("rolling-origin-constant-v1", self._constant),
            ("rolling-origin-linear-v1", self._linear),
        )
        scored = []
        for order, (method, estimator) in enumerate(candidates):
            errors: list[Decimal] = []
            for position in range(2, len(years)):
                training = {year: history[year] for year in years[:position]}
                actual = history[years[position]]
                predicted = max(Decimal(0), estimator(training, years[position]))
                errors.append(abs(predicted - actual) / max(actual, Decimal(1)))
            score = sum(errors, start=Decimal(0)) / max(len(errors), 1)
            scored.append((score, order, method, estimator))
        score, _, method, estimator = min(scored, key=lambda item: (item[0], item[1]))
        values = tuple(
            (
                year,
                max(Decimal(0), estimator(history, year)).quantize(_QUANTUM),
            )
            for year in range(years[-1] + 1, horizon_year + 1)
        )
        return RegistrationForecast(method, values, score.quantize(_QUANTUM))

    @staticmethod
    def _constant(history: dict[int, Decimal], _: int) -> Decimal:
        recent = [history[year] for year in sorted(history)[-3:]]
        return sum(recent, start=Decimal(0)) / len(recent)

    @staticmethod
    def _linear(history: dict[int, Decimal], target_year: int) -> Decimal:
        years = sorted(history)[-5:]
        origin = years[0]
        xs = [Decimal(year - origin) for year in years]
        ys = [history[year] for year in years]
        mean_x = sum(xs, start=Decimal(0)) / len(xs)
        mean_y = sum(ys, start=Decimal(0)) / len(ys)
        denominator = sum(((x - mean_x) ** 2 for x in xs), start=Decimal(0))
        slope = (
            Decimal(0)
            if denominator == 0
            else sum(
                ((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)),
                start=Decimal(0),
            )
            / denominator
        )
        return mean_y + slope * (Decimal(target_year - origin) - mean_x)
