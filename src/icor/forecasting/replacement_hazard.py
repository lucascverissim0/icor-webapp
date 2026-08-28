"""Explicit age and geography assumptions for annual windshield replacement hazard."""

from __future__ import annotations

from decimal import Decimal


class ReplacementHazardModel:
    method = "age-band-geography-hazard-v1"
    assumption_ids = (
        "hazard-age-0-3-0.020",
        "hazard-age-4-7-0.035",
        "hazard-age-8-12-0.050",
        "hazard-age-13-plus-0.060",
        "hazard-uncertainty-plus-minus-20pct",
    )

    def __init__(self, *, geography_multipliers: dict[str, str] | None = None) -> None:
        self.geography_multipliers = {
            key: Decimal(value) for key, value in (geography_multipliers or {}).items()
        }
        if any(value <= 0 for value in self.geography_multipliers.values()):
            raise ValueError("hazard geography multipliers must be positive")

    def annual_probability(self, *, age_years: int, geography: str) -> Decimal:
        if type(age_years) is not int or age_years < 0:
            raise ValueError("hazard age must be a non-negative integer")
        if not geography.strip():
            raise ValueError("hazard geography is required")
        if age_years <= 3:
            base = Decimal("0.020")
        elif age_years <= 7:
            base = Decimal("0.035")
        elif age_years <= 12:
            base = Decimal("0.050")
        else:
            base = Decimal("0.060")
        return min(Decimal(1), base * self.geography_multipliers.get(geography, Decimal(1)))

    def interval(self, *, age_years: int, geography: str) -> tuple[Decimal, Decimal, Decimal]:
        p50 = self.annual_probability(age_years=age_years, geography=geography)
        return p50 * Decimal("0.8"), p50, min(Decimal(1), p50 * Decimal("1.2"))
