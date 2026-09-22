"""Versioned passenger-car cohort survival assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from icor.forecasting.survival_calibration import LicensedStockSurvivalBand

_QUANTUM = Decimal("0.0001")


@dataclass(frozen=True, slots=True)
class SurvivalInterval:
    p10: Decimal
    p50: Decimal
    p90: Decimal


def _assumption_id(quantile: str, retention: Decimal) -> str:
    """Name the retention actually in use, so provenance cannot drift from the value."""

    return f"survival-retention-{quantile}-{format(retention.normalize(), 'f')}"


class CohortSurvivalModel:
    """Constant annual retention compounded by cohort age.

    `method` and `assumption_ids` are per-instance: a calibrated model must never
    inherit the default model's provenance, because these strings are written into
    the immutable evidence record and shown to the client.
    """

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
        self.method = "constant-annual-retention-v1"
        self.assumption_ids = (
            _assumption_id("p10", retention_p10),
            _assumption_id("p50", retention_p50),
            _assumption_id("p90", retention_p90),
        )

    def remaining(self, registrations: Decimal, *, age_years: int) -> Decimal:
        return self._remaining(registrations, age_years, self.retention_p50)

    def interval(self, registrations: Decimal, *, age_years: int) -> SurvivalInterval:
        return SurvivalInterval(
            self._remaining(registrations, age_years, self.retention_p10),
            self._remaining(registrations, age_years, self.retention_p50),
            self._remaining(registrations, age_years, self.retention_p90),
        )

    def reason_code(self, geography: str) -> str:
        """Name the survival evidence backing a cohort in this geography."""

        if not geography.strip():
            raise ValueError("cohort geography is required")
        return "assumption-led-survival-not-calibrated"

    @staticmethod
    def _remaining(registrations: Decimal, age_years: int, retention: Decimal) -> Decimal:
        if not registrations.is_finite() or registrations < 0:
            raise ValueError("cohort registrations must be finite and non-negative")
        if type(age_years) is not int or age_years < 0:
            raise ValueError("cohort age must be a non-negative integer")
        return (registrations * retention**age_years).quantize(_QUANTUM)


class CalibratedCohortSurvivalModel:
    """Serve a measured licensed-stock retention band instead of an assumption.

    The median is the pooled UK curve validated by
    `scripts/benchmark_survival_calibration.py`; the envelope is the observed
    spread of the same transition across registration cohorts.

    The curve is measured on one country. Applying it elsewhere is a transfer,
    not a measurement, so `reason_code` says which of the two a cohort got and
    the geography is never silently absorbed.
    """

    def __init__(
        self,
        band: LicensedStockSurvivalBand,
        *,
        calibrated_geography: str = "GB",
        method: str = "uk-dft-licensed-stock-band-v1",
    ) -> None:
        if not calibrated_geography.strip():
            raise ValueError("the calibrated geography is required")
        if not method.strip():
            raise ValueError("the survival method is required")
        self.band = band
        self.calibrated_geography = calibrated_geography
        self.method = method
        support = band.p50.calibrated_support_age or 0
        self.assumption_ids = (
            f"survival-licensed-stock-anchor-cohorts-{band.p50.anchor_cohort_count}",
            f"survival-licensed-stock-calibrated-age-{support}",
            f"survival-licensed-stock-band-cohorts-{len(band.cohort_counts)}",
        )

    def remaining(self, registrations: Decimal, *, age_years: int) -> Decimal:
        return self.band.p50.remaining(registrations, age_years=age_years)

    def interval(self, registrations: Decimal, *, age_years: int) -> SurvivalInterval:
        return SurvivalInterval(
            self.band.p10.remaining(registrations, age_years=age_years),
            self.band.p50.remaining(registrations, age_years=age_years),
            self.band.p90.remaining(registrations, age_years=age_years),
        )

    def reason_code(self, geography: str) -> str:
        if not geography.strip():
            raise ValueError("cohort geography is required")
        if geography.casefold() == self.calibrated_geography.casefold():
            return "licensed-stock-calibrated-survival"
        return "licensed-stock-calibrated-survival-transferred"
