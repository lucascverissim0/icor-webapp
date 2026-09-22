"""Seeded propagation of survival and hazard assumption intervals."""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal
from statistics import NormalDist

_QUANTUM = Decimal("0.0001")
_STANDARD_NORMAL = NormalDist()
_DECILE_Z = _STANDARD_NORMAL.inv_cdf(0.9)
_UNIT_GUARD = 1e-12

# Both the offline build path and the live query path must draw the same number of
# samples, or the ranking page and the forecast page disagree about the same vehicle.
DEFAULT_DRAW_COUNT = 2000


class _SplitNormalQuantiles:
    """Draw from a distribution whose P10, P50 and P90 are the given values.

    Two half-normals share a median, one spread below it and one above, so an
    asymmetric input interval is reproduced exactly rather than approximated.
    Support is clamped at zero because neither a fleet count nor a hazard rate
    can be negative.
    """

    __slots__ = ("_median", "_lower_spread", "_upper_spread")

    def __init__(self, p10: Decimal, p50: Decimal, p90: Decimal) -> None:
        self._median = float(p50)
        self._lower_spread = float(p50 - p10) / _DECILE_Z
        self._upper_spread = float(p90 - p50) / _DECILE_Z

    def draw(self, generator: random.Random) -> float:
        probability = min(max(generator.random(), _UNIT_GUARD), 1.0 - _UNIT_GUARD)
        deviate = _STANDARD_NORMAL.inv_cdf(probability)
        spread = self._lower_spread if probability <= 0.5 else self._upper_spread
        value = self._median + spread * deviate
        return value if value > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class OpportunityInterval:
    p10: Decimal
    p50: Decimal
    p90: Decimal


class OpportunityUncertaintyModel:
    """`method` is per-instance: a variant must never report the default.

    A class attribute here would be reported by every instance regardless of
    what it actually did, which is the defect already fixed once in the
    survival model.
    """

    def __init__(
        self,
        *,
        draw_count: int = DEFAULT_DRAW_COUNT,
        method: str = "quantile-matched-split-normal-propagation-v2",
    ) -> None:
        if type(draw_count) is not int or draw_count < 100:
            raise ValueError("uncertainty draw count must be at least 100")
        if not method.strip():
            raise ValueError("the uncertainty method is required")
        self.draw_count = draw_count
        self.method = method

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
        fleet = _SplitNormalQuantiles(active_fleet_p10, active_fleet_p50, active_fleet_p90)
        hazard = _SplitNormalQuantiles(hazard_p10, hazard_p50, hazard_p90)
        samples = sorted(
            Decimal(str(fleet.draw(generator) * hazard.draw(generator)))
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
