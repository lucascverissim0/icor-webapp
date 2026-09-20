"""Calibrate a licensed-stock retention curve from registration cohorts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

_QUANTUM = Decimal("0.0001")
_ZERO = Decimal(0)
_ONE = Decimal(1)


@dataclass(frozen=True, slots=True)
class RegistrationCohort:
    cohort_year: int
    registrations: Decimal

    def __post_init__(self) -> None:
        if type(self.cohort_year) is not int:
            raise ValueError("cohort year must be an integer")
        if not self.registrations.is_finite() or self.registrations < _ZERO:
            raise ValueError("cohort registrations must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CohortStock:
    cohort_year: int
    stock_year: int
    licensed_vehicles: Decimal

    def __post_init__(self) -> None:
        if type(self.cohort_year) is not int or type(self.stock_year) is not int:
            raise ValueError("cohort and stock years must be integers")
        if self.stock_year < self.cohort_year:
            raise ValueError("stock year cannot precede cohort year")
        if not self.licensed_vehicles.is_finite() or self.licensed_vehicles < _ZERO:
            raise ValueError("licensed stock must be finite and non-negative")

    @property
    def age_years(self) -> int:
        return self.stock_year - self.cohort_year


@dataclass(frozen=True, slots=True)
class LicensedStockSurvivalCurve:
    """A monotone empirical curve, with age zero fixed at one."""

    shares: tuple[Decimal, ...]
    anchor_cohort_count: int
    transition_counts: tuple[tuple[int, int], ...]

    def remaining_share(self, age_years: int) -> Decimal:
        if type(age_years) is not int or age_years < 0:
            raise ValueError("cohort age must be a non-negative integer")
        if age_years >= len(self.shares):
            raise ValueError(f"cohort age {age_years} exceeds calibrated support")
        return self.shares[age_years]

    def remaining(self, registrations: Decimal, *, age_years: int) -> Decimal:
        if not registrations.is_finite() or registrations < _ZERO:
            raise ValueError("cohort registrations must be finite and non-negative")
        return (registrations * self.remaining_share(age_years)).quantize(_QUANTUM)


def calibrate_licensed_stock_curve(
    registrations: Iterable[RegistrationCohort],
    stocks: Iterable[CohortStock],
    *,
    excluded_cohort_years: frozenset[int] = frozenset(),
    maximum_age_years: int = 40,
) -> LicensedStockSurvivalCurve:
    """Estimate age-one retention and later same-cohort stock transitions.

    Age one is anchored against registrations. Later ages use pooled longitudinal
    transitions for the same cohorts, which avoids comparing different cohort mixes.
    Administrative stock can grow through imports or record corrections, so each
    annual transition is capped at one to keep the resulting curve monotone.
    """

    if type(maximum_age_years) is not int or maximum_age_years < 1:
        raise ValueError('maximum age must be a positive integer')

    registration_by_year: dict[int, Decimal] = {}
    for item in registrations:
        if item.cohort_year in registration_by_year:
            raise ValueError(f"duplicate registration cohort: {item.cohort_year}")
        if item.cohort_year not in excluded_cohort_years:
            registration_by_year[item.cohort_year] = item.registrations
    if not registration_by_year:
        raise ValueError("calibration requires registration cohorts")

    stock_by_cohort_age: dict[tuple[int, int], Decimal] = {}
    for item in stocks:
        if item.cohort_year in excluded_cohort_years:
            continue
        key = (item.cohort_year, item.age_years)
        stock_by_cohort_age[key] = stock_by_cohort_age.get(key, _ZERO) + item.licensed_vehicles

    age_one_years = tuple(
        year for year in registration_by_year if (year, 1) in stock_by_cohort_age
    )
    if not age_one_years:
        raise ValueError("calibration requires an age-one stock anchor")
    anchor_years = tuple(
        year
        for year in age_one_years
        if registration_by_year[year] > _ZERO
    )
    registration_exposure = sum(
        (registration_by_year[year] for year in anchor_years),
        start=_ZERO,
    )
    if registration_exposure <= _ZERO:
        raise ValueError("calibration requires positive registration exposure")
    age_one_stock = sum(
        (stock_by_cohort_age[(year, 1)] for year in anchor_years),
        start=_ZERO,
    )
    if age_one_stock <= _ZERO:
        raise ValueError("calibration requires an age-one stock anchor")

    anchor = min(_ONE, age_one_stock / registration_exposure)
    shares = [_ONE, anchor]
    transition_counts: list[tuple[int, int]] = []
    age = 1
    while age < maximum_age_years:
        eligible = tuple(sorted({
            cohort_year
            for cohort_year, _ in stock_by_cohort_age
            if (
                cohort_year not in registration_by_year
                or registration_by_year[cohort_year] > _ZERO
            )
            and stock_by_cohort_age.get((cohort_year, age), _ZERO) > _ZERO
            and (cohort_year, age + 1) in stock_by_cohort_age
        }))
        anchored = tuple(
            year for year in eligible if registration_by_year.get(year, _ZERO) > _ZERO
        )
        if anchored:
            eligible = anchored
        if not eligible:
            break
        current_stock = sum(
            (stock_by_cohort_age[(cohort_year, age)] for cohort_year in eligible),
            start=_ZERO,
        )
        next_stock = sum(
            (stock_by_cohort_age[(cohort_year, age + 1)] for cohort_year in eligible),
            start=_ZERO,
        )
        transition = min(_ONE, next_stock / current_stock)
        shares.append(shares[-1] * transition)
        transition_counts.append((age, len(eligible)))
        age += 1

    return LicensedStockSurvivalCurve(
        shares=tuple(shares),
        anchor_cohort_count=len(anchor_years),
        transition_counts=tuple(transition_counts),
    )
