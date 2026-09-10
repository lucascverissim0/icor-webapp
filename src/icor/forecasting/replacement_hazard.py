"""Evidence-anchored annual windshield replacement hazard."""

from __future__ import annotations

from decimal import Decimal

_GLASS_CLAIM_FREQUENCY = Decimal("0.0606")
_WINDSHIELD_REPLACEMENT_SHARE = Decimal("0.71")
_BASE_REPLACEMENT_RATE = _GLASS_CLAIM_FREQUENCY * _WINDSHIELD_REPLACEMENT_SHARE
_SCENARIO_SPREAD = Decimal("0.20")


class ReplacementHazardModel:
    method = "france-insurance-windshield-hazard-v2"
    assumption_ids = (
        "france-assureurs-2025-covered-glass-claim-frequency-0.0606",
        "pacifica-2022-windshield-replacement-share-0.71",
        "derived-windshield-replacement-rate-0.043026",
        "hazard-scenario-band-plus-minus-20pct",
        "no-free-age-or-cross-market-calibration",
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
        return min(
            Decimal(1),
            _BASE_REPLACEMENT_RATE
            * self.geography_multipliers.get(geography, Decimal(1)),
        )

    def interval(self, *, age_years: int, geography: str) -> tuple[Decimal, Decimal, Decimal]:
        p50 = self.annual_probability(age_years=age_years, geography=geography)
        return (
            p50 * (Decimal(1) - _SCENARIO_SPREAD),
            p50,
            min(Decimal(1), p50 * (Decimal(1) + _SCENARIO_SPREAD)),
        )
