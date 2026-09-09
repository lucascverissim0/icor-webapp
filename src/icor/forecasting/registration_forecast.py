"""Backtested robust forecasts for sparse annual vehicle registrations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

_QUANTUM = Decimal("0.0001")
_RECENCY_WEIGHT = Decimal("0.5")
_DAMPING = Decimal("0.8")
_MAX_BACKTEST_HORIZON = 4


@dataclass(frozen=True, slots=True)
class RegistrationForecast:
    method: str
    values: tuple[tuple[int, Decimal], ...]
    backtest_error: Decimal


class RegistrationForecaster:
    """Use the fixed ensemble that wins the latest held-out snapshot benchmark.

    The method combines a highly robust last-observation forecast with a damped
    five-year trend. A fixed, globally validated weight avoids unstable per-series
    model selection across short and noisy vehicle histories.
    """

    method = "validated-recency-damped-ensemble-v2"

    def forecast(
        self,
        history: dict[int, Decimal],
        *,
        horizon_year: int,
        evaluate_backtest: bool = True,
    ) -> RegistrationForecast:
        if type(evaluate_backtest) is not bool:
            raise ValueError("evaluate_backtest must be a boolean")
        self._validate(history, horizon_year)
        years = sorted(history)
        values = tuple(
            (
                year,
                max(Decimal(0), self._estimate(history, year)).quantize(_QUANTUM),
            )
            for year in range(years[-1] + 1, horizon_year + 1)
        )
        return RegistrationForecast(
            self.method,
            values,
            (
                self._rolling_origin_wape(history).quantize(_QUANTUM)
                if evaluate_backtest
                else Decimal(0)
            ),
        )

    @staticmethod
    def _validate(history: dict[int, Decimal], horizon_year: int) -> None:
        if len(history) < 2:
            raise ValueError("registration forecasting requires at least two annual values")
        years = sorted(history)
        if years != list(range(years[0], years[-1] + 1)):
            raise ValueError("registration history must be annual and contiguous")
        if horizon_year <= years[-1]:
            raise ValueError("registration forecast horizon must follow observed history")
        if any(not value.is_finite() or value < 0 for value in history.values()):
            raise ValueError("registration history must be finite and non-negative")

    def _rolling_origin_wape(self, history: dict[int, Decimal]) -> Decimal:
        years = sorted(history)
        absolute_error = Decimal(0)
        denominator = Decimal(0)
        for position in range(3, len(years)):
            training = {year: history[year] for year in years[:position]}
            for step in range(
                1,
                min(_MAX_BACKTEST_HORIZON, len(years) - position) + 1,
            ):
                year = years[position + step - 1]
                actual = history[year]
                predicted = max(Decimal(0), self._estimate(training, year))
                absolute_error += abs(predicted - actual)
                denominator += max(actual, Decimal(1))
        return absolute_error / denominator if denominator else Decimal(0)

    def _estimate(self, history: dict[int, Decimal], target_year: int) -> Decimal:
        years = sorted(history)
        last = history[years[-1]]
        steps = target_year - years[-1]
        damped = last + self._slope(history) * sum(
            (_DAMPING**step for step in range(1, steps + 1)),
            start=Decimal(0),
        )
        return _RECENCY_WEIGHT * last + (Decimal(1) - _RECENCY_WEIGHT) * damped

    @staticmethod
    def _slope(history: dict[int, Decimal]) -> Decimal:
        years = sorted(history)[-5:]
        origin = years[0]
        xs = [Decimal(year - origin) for year in years]
        ys = [history[year] for year in years]
        mean_x = sum(xs, start=Decimal(0)) / len(xs)
        mean_y = sum(ys, start=Decimal(0)) / len(ys)
        denominator = sum(((x - mean_x) ** 2 for x in xs), start=Decimal(0))
        if denominator == 0:
            return Decimal(0)
        return sum(
            ((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)),
            start=Decimal(0),
        ) / denominator
