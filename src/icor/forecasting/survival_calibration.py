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
    """A monotone empirical curve, with age zero fixed at one.

    `calibrated_support_age` is the oldest age backed by observed stock
    transitions. Ages beyond it come from `extend_with_monotone_tail` and are
    extrapolation, which callers must be able to tell apart from evidence.
    """

    shares: tuple[Decimal, ...]
    anchor_cohort_count: int
    transition_counts: tuple[tuple[int, int], ...]
    calibrated_support_age: int | None = None

    def __post_init__(self) -> None:
        if self.calibrated_support_age is None:
            object.__setattr__(self, "calibrated_support_age", len(self.shares) - 1)

    def is_extrapolated(self, age_years: int) -> bool:
        return age_years > (self.calibrated_support_age or 0)

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


_TAIL_TRANSITION_WINDOW = 5
_MINIMUM_BAND_COHORTS = 3


def extend_with_monotone_tail(
    curve: LicensedStockSurvivalCurve,
    *,
    maximum_age_years: int,
) -> LicensedStockSurvivalCurve:
    """Continue a calibrated curve past its support at its recent decay rate.

    The observed UK curve is not a single parametric shape: retention is near
    1.0 through the first decade, falls to about 0.79 a year between ages 11 and
    20, then rises again as durable survivors remain. A Weibull or Gompertz fit
    reproduces none of that, so the tail does not invent a shape. It continues
    the geometric mean of the last few observed annual transitions, which is the
    weakest assumption that still guarantees a decaying, monotone curve.

    The rate must land strictly inside (0, 1). A curve whose recent transitions
    were all capped at 1.0 carries no evidence of a decay rate, and is rejected
    here rather than silently extended into an immortal fleet.
    """

    if type(maximum_age_years) is not int:
        raise ValueError("maximum age must be an integer")
    support = len(curve.shares) - 1
    if maximum_age_years < support:
        raise ValueError("maximum age cannot be shorter than the calibrated support")
    if maximum_age_years == support:
        return curve

    transitions = [
        curve.shares[age] / curve.shares[age - 1]
        for age in range(1, len(curve.shares))
        if curve.shares[age - 1] > _ZERO
    ]
    window = transitions[-_TAIL_TRANSITION_WINDOW:]
    if not window:
        raise ValueError("tail retention requires at least one observed transition")

    product = _ONE
    for transition in window:
        product *= transition
    rate = _decimal_root(product, len(window))
    if not _ZERO < rate < _ONE:
        raise ValueError(
            "tail retention must fall strictly inside (0, 1); the calibrated "
            "curve shows no decay to continue"
        )

    shares = list(curve.shares)
    for _ in range(support, maximum_age_years):
        shares.append((shares[-1] * rate).quantize(_QUANTUM))

    return LicensedStockSurvivalCurve(
        shares=tuple(shares),
        anchor_cohort_count=curve.anchor_cohort_count,
        transition_counts=curve.transition_counts,
        calibrated_support_age=support,
    )


def _decimal_root(value: Decimal, degree: int) -> Decimal:
    """Take an exact-enough n-th root without leaving Decimal for floats."""

    if degree == 1:
        return value
    return Decimal(float(value) ** (1.0 / degree))


@dataclass(frozen=True, slots=True)
class LicensedStockSurvivalBand:
    """A calibrated median curve with an evidence-based P10/P90 envelope."""

    p10: LicensedStockSurvivalCurve
    p50: LicensedStockSurvivalCurve
    p90: LicensedStockSurvivalCurve
    cohort_counts: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        lengths = {len(self.p10.shares), len(self.p50.shares), len(self.p90.shares)}
        if len(lengths) != 1:
            raise ValueError("survival band curves must share one support")
        for low, mid, high in zip(
            self.p10.shares, self.p50.shares, self.p90.shares, strict=True
        ):
            if not low <= mid <= high:
                raise ValueError("survival band quantiles must be ordered at every age")


def calibrate_licensed_stock_band(
    registrations: Iterable[RegistrationCohort],
    stocks: Iterable[CohortStock],
    *,
    excluded_cohort_years: frozenset[int] = frozenset(),
    maximum_age_years: int = 40,
) -> LicensedStockSurvivalBand:
    """Wrap the validated pooled curve in a band measured across cohorts.

    The median is the pooled curve exactly as `calibrate_licensed_stock_curve`
    produces it, so the benchmarked accuracy is unchanged. The envelope comes
    from the spread of the *same* annual transition measured separately on each
    registration cohort: at every age the UK data supplies eleven cohorts, so
    this is observed dispersion rather than an assumed percentage.

    Quantile transitions are compounded, which treats a cohort that decays
    faster than the median as continuing to do so. That is the conservative
    reading - independence would give a narrower band - and it matches the data,
    where cohort deviations persist rather than cancel year to year.
    """

    stock_items = tuple(stocks)
    registration_items = tuple(registrations)
    median = calibrate_licensed_stock_curve(
        registration_items,
        stock_items,
        excluded_cohort_years=excluded_cohort_years,
        maximum_age_years=maximum_age_years,
    )

    totals: dict[tuple[int, int], Decimal] = {}
    for item in stock_items:
        if item.cohort_year in excluded_cohort_years:
            continue
        key = (item.cohort_year, item.age_years)
        totals[key] = totals.get(key, _ZERO) + item.licensed_vehicles

    per_age: dict[int, list[Decimal]] = {}
    for (cohort_year, age), value in totals.items():
        following = totals.get((cohort_year, age + 1))
        if following is None or value <= _ZERO:
            continue
        per_age.setdefault(age, []).append(min(_ONE, following / value))

    low_factor = _ONE
    high_factor = _ONE
    low_shares = [_ONE]
    high_shares = [_ONE]
    cohort_counts: list[tuple[int, int]] = []
    for age in range(1, len(median.shares)):
        observations = sorted(per_age.get(age - 1, ()))
        if len(observations) >= _MINIMUM_BAND_COHORTS:
            centre = _quantile(observations, Decimal("0.5"))
            if centre > _ZERO:
                low_factor *= _quantile(observations, Decimal("0.1")) / centre
                high_factor *= _quantile(observations, Decimal("0.9")) / centre
            cohort_counts.append((age, len(observations)))
        share = median.shares[age]
        low_shares.append(share * low_factor)
        high_shares.append(min(_ONE, share * high_factor))

    return LicensedStockSurvivalBand(
        p10=_bounded(low_shares, median, upper=True),
        p50=median,
        p90=_bounded(high_shares, median, upper=False),
        cohort_counts=tuple(cohort_counts),
    )


def _quantile(values: list[Decimal], probability: Decimal) -> Decimal:
    index = int(round(float(probability) * (len(values) - 1)))
    return values[min(max(index, 0), len(values) - 1)]


def _bounded(
    shares: list[Decimal],
    median: LicensedStockSurvivalCurve,
    *,
    upper: bool,
) -> LicensedStockSurvivalCurve:
    """Make a quantile curve non-increasing and keep it on its side of the median.

    Compounding a drifting factor onto a decaying share can break monotonicity,
    and clamping that alone could push P90 under the median. Both sequences are
    non-increasing, so taking an elementwise min (for P10) or max (for P90)
    against the median preserves monotonicity while restoring the ordering the
    band contract requires.
    """

    guarded = [min(_ONE, shares[0])]
    for share in shares[1:]:
        guarded.append(min(share, guarded[-1]))
    combine = min if upper else max
    ordered = tuple(
        combine(value, centre)
        for value, centre in zip(guarded, median.shares, strict=True)
    )
    return LicensedStockSurvivalCurve(
        shares=ordered,
        anchor_cohort_count=median.anchor_cohort_count,
        transition_counts=median.transition_counts,
        calibrated_support_age=median.calibrated_support_age,
    )
