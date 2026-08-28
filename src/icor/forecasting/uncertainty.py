"""Seeded propagation of survival and hazard assumption intervals."""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal

_QUANTUM = Decimal("0.0001")


@dataclass(frozen=True, slots=True)
class OpportunityInterval:
    p10: Decimal
    p50: Decimal
    p90: Decimal


class OpportunityUncertaintyModel:
    method = "seeded-triangular-propagation-v1"

    def __init__(self, *, draw_count: int = 2000) -> None:
        if type(draw_count) is not int or draw_count < 100:
            raise ValueError("uncertainty draw count must be at least 100")
        self.draw_count = draw_count

    def estimate(
        self,
        *,
        active_fleet_p10: Decimal,
        active_fleet_p50: Decimal,
        active_fleet_p90: Decimal,
        hazard_p10: Decimal,
        hazard_p50: Decimal,
        hazard_p90: Decimal,
        seed: int,
    ) -> OpportunityInterval:
        self._ordered(active_fleet_p10, active_fleet_p50, active_fleet_p90, "fleet")
        self._ordered(hazard_p10, hazard_p50, hazard_p90, "hazard")
        if type(seed) is not int:
            raise ValueError("uncertainty seed must be an integer")
        generator = random.Random(seed)
        samples = sorted(
            Decimal(
                str(
                    generator.triangular(
                        float(active_fleet_p10),
                        float(active_fleet_p90),
                        float(active_fleet_p50),
                    )
                    * generator.triangular(
                        float(hazard_p10),
                        float(hazard_p90),
                        float(hazard_p50),
                    )
                )
            )
            for _ in range(self.draw_count)
        )
        return OpportunityInterval(
            self._quantile(samples, Decimal("0.10")),
            self._quantile(samples, Decimal("0.50")),
            self._quantile(samples, Decimal("0.90")),
        )

    @staticmethod
    def _ordered(low: Decimal, middle: Decimal, high: Decimal, label: str) -> None:
        if not low.is_finite() or low < 0 or not low <= middle <= high:
            raise ValueError(f"{label} uncertainty interval must be finite and ordered")

    @staticmethod
    def _quantile(values: list[Decimal], probability: Decimal) -> Decimal:
        index = int((len(values) - 1) * probability)
        return values[index].quantize(_QUANTUM)
