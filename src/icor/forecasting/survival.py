"""Versioned passenger-car cohort survival assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

_QUANTUM = Decimal("0.0001")


@dataclass(frozen=True, slots=True)
class SurvivalInterval:
    p10: Decimal
    p50: Decimal
    p90: Decimal


class CohortSurvivalModel:
    method = "constant-annual-retention-v1"
    assumption_ids = (
        "survival-retention-p10-0.92",
        "survival-retention-p50-0.9444",
        "survival-retention-p90-0.965",
    )

    def __init__(
        self,
        *,
        retention_p10: Decimal = Decimal("0.92"),
        retention_p50: Decimal = Decimal("0.9444"),
        retention_p90: Decimal = Decimal("0.965"),
    ) -> None:
        if not Decimal(0) < retention_p10 <= retention_p50 <= retention_p90 <= Decimal(1):
            raise ValueError("survival retention assumptions must be ordered probabilities")
        self.retention_p10 = retention_p10
        self.retention_p50 = retention_p50
        self.retention_p90 = retention_p90

    def remaining(self, registrations: Decimal, *, age_years: int) -> Decimal:
        return self._remaining(registrations, age_years, self.retention_p50)

    def interval(self, registrations: Decimal, *, age_years: int) -> SurvivalInterval:
        return SurvivalInterval(
            self._remaining(registrations, age_years, self.retention_p10),
            self._remaining(registrations, age_years, self.retention_p50),
            self._remaining(registrations, age_years, self.retention_p90),
        )

    @staticmethod
    def _remaining(registrations: Decimal, age_years: int, retention: Decimal) -> Decimal:
        if not registrations.is_finite() or registrations < 0:
            raise ValueError("cohort registrations must be finite and non-negative")
        if type(age_years) is not int or age_years < 0:
            raise ValueError("cohort age must be a non-negative integer")
        return (registrations * retention**age_years).quantize(_QUANTUM)
