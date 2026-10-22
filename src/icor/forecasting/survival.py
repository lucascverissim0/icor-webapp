"""Versioned passenger-car cohort survival assumptions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from icor.forecasting.survival_calibration import (
    LicensedStockSurvivalBand,
    LicensedStockSurvivalCurve,
)

_QUANTUM = Decimal("0.0001")
_ONE = Decimal(1)
#: A cohort whose cars the market's net retention lowered below the curve.
NET_RETENTION_REASON = "eurostat-net-retention-calibrated"


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

    def interval(
        self, registrations: Decimal, *, age_years: int, geography: str | None = None
    ) -> SurvivalInterval:
        return SurvivalInterval(
            self._remaining(registrations, age_years, self.retention_p10),
            self._remaining(registrations, age_years, self.retention_p50),
            self._remaining(registrations, age_years, self.retention_p90),
        )

    def retention(self, geography: str, *, age_years: int) -> Decimal:
        """The market's net retention at an age: one, since this model has none."""

        return _ONE

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
        calibrated_geography: str = "UK",
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

    def interval(
        self, registrations: Decimal, *, age_years: int, geography: str | None = None
    ) -> SurvivalInterval:
        return SurvivalInterval(
            self.band.p10.remaining(registrations, age_years=age_years),
            self.band.p50.remaining(registrations, age_years=age_years),
            self.band.p90.remaining(registrations, age_years=age_years),
        )

    def retention(self, geography: str, *, age_years: int) -> Decimal:
        """The market's net retention at an age: one, since this model has none."""

        return _ONE

    def reason_code(self, geography: str) -> str:
        if not geography.strip():
            raise ValueError("cohort geography is required")
        source = self.calibrated_geography.casefold()
        if geography.casefold() == source:
            return f"licensed-stock-calibrated-survival-{source}"
        return f"licensed-stock-calibrated-survival-transferred-from-{source}"


PROMOTED_CURVE_PATH = Path(__file__).parent / "survival_curves" / "uk_licensed_stock.json"


def load_promoted_survival_model(
    path: Path | None = None,
) -> CalibratedCohortSurvivalModel:
    """Build the production model from the committed, reviewed curve artifact.

    A snapshot build must not recalibrate. The curve it serves is whichever
    artifact was reviewed and committed, so the model a snapshot used can be read
    off the repository rather than reconstructed from whatever inputs happened to
    be on disk at build time.
    """

    resolved = path or PROMOTED_CURVE_PATH
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    support = payload["calibrated_support_age"]
    anchors = payload["anchor_cohort_count"]

    def _curve(key: str) -> LicensedStockSurvivalCurve:
        return LicensedStockSurvivalCurve(
            shares=tuple(Decimal(value) for value in payload["shares"][key]),
            anchor_cohort_count=anchors,
            transition_counts=tuple(
                (int(age), int(count)) for age, count in payload["band_cohort_counts"]
            ),
            calibrated_support_age=support,
        )

    band = LicensedStockSurvivalBand(
        p10=_curve("p10"),
        p50=_curve("p50"),
        p90=_curve("p90"),
        cohort_counts=tuple(
            (int(age), int(count)) for age, count in payload["band_cohort_counts"]
        ),
    )
    return CalibratedCohortSurvivalModel(
        band,
        calibrated_geography=payload["calibrated_geography"],
        method=payload["method"],
    )


class NetRetainedSurvivalModel:
    """The promoted curve, lowered per market by its Eurostat-calibrated net retention.

    Markets that export or re-export many young cars keep fewer than the curve,
    measured where few do, leaves on the road (`icor.forecasting.net_retention`).
    A calibrated market's quantiles are all multiplied by its retention at the
    cohort's age, which is never above one; every other market gets the curve
    unchanged. The method names both the curve and the calibration, so a snapshot
    records which it was built with; a recalibration bumps the calibration's method.
    """

    def __init__(
        self,
        base: CalibratedCohortSurvivalModel,
        multipliers: Mapping[str, tuple[Decimal, ...]],
        *,
        method: str,
    ) -> None:
        if not method.strip():
            raise ValueError("the net retention method is required")
        for geography, curve in multipliers.items():
            if not curve or any(not Decimal(0) < value <= _ONE for value in curve):
                raise ValueError(f"{geography}: net retention must lie in (0, 1]")
            if any(later > earlier for earlier, later in zip(curve, curve[1:], strict=False)):
                raise ValueError(f"{geography}: net retention must not increase with age")
        self.base = base
        self.band = base.band
        self.calibrated_geography = base.calibrated_geography
        self.multipliers = dict(multipliers)
        self.method = f"{base.method}+{method}"
        self.assumption_ids = (
            *base.assumption_ids,
            f"survival-{method}-markets-{'-'.join(sorted(multipliers)) or 'none'}",
        )

    def retention(self, geography: str, *, age_years: int) -> Decimal:
        curve = self.multipliers.get(geography)
        if curve is None:
            return _ONE
        return curve[min(age_years, len(curve) - 1)]

    def remaining(self, registrations: Decimal, *, age_years: int) -> Decimal:
        return self.base.remaining(registrations, age_years=age_years)

    def interval(
        self, registrations: Decimal, *, age_years: int, geography: str | None = None
    ) -> SurvivalInterval:
        interval = self.base.interval(registrations, age_years=age_years)
        factor = _ONE if geography is None else self.retention(geography, age_years=age_years)
        if factor == _ONE:
            return interval
        return SurvivalInterval(
            (interval.p10 * factor).quantize(_QUANTUM),
            (interval.p50 * factor).quantize(_QUANTUM),
            (interval.p90 * factor).quantize(_QUANTUM),
        )

    def reason_code(self, geography: str) -> str:
        return self.base.reason_code(geography)


NET_RETENTION_PATH = PROMOTED_CURVE_PATH.parent / "eurostat_net_retention.json"


def load_net_retained_survival_model(
    path: Path | None = None, *, base: CalibratedCohortSurvivalModel | None = None
) -> NetRetainedSurvivalModel:
    """The promoted curve with the committed, reviewed net retention artifact."""

    payload = json.loads((path or NET_RETENTION_PATH).read_text(encoding="utf-8"))
    curve = base or load_promoted_survival_model()
    if payload["base_method"] != curve.method:
        raise ValueError("net retention was calibrated on another survival curve")
    return NetRetainedSurvivalModel(
        curve,
        {
            geography: tuple(Decimal(value) for value in market["multipliers"])
            for geography, market in payload["markets"].items()
        },
        method=payload["method"],
    )


def load_survival_model_for(
    method: str,
) -> CalibratedCohortSurvivalModel | NetRetainedSurvivalModel:
    """The committed model a snapshot recorded as its survival method, so it can be verified."""

    base = load_promoted_survival_model()
    if method == base.method:
        return base
    retained = load_net_retained_survival_model(base=base)
    if method == retained.method:
        return retained
    raise ValueError(f"no committed survival model has method {method!r}")
