from decimal import Decimal

import pytest

from icor.forecasting.survival_calibration import (
    CohortStock,
    RegistrationCohort,
    calibrate_licensed_stock_curve,
    extend_with_monotone_tail,
)


def test_calibration_anchors_age_one_and_uses_longitudinal_transitions() -> None:
    registrations = (
        RegistrationCohort(2018, Decimal("100")),
        RegistrationCohort(2019, Decimal("200")),
    )
    stocks = (
        CohortStock(2018, 2019, Decimal("90")),
        CohortStock(2018, 2020, Decimal("81")),
        CohortStock(2019, 2020, Decimal("180")),
        CohortStock(2019, 2021, Decimal("171")),
    )

    curve = calibrate_licensed_stock_curve(registrations, stocks)

    assert curve.remaining_share(0) == Decimal(1)
    assert curve.remaining_share(1) == Decimal("0.9")
    assert curve.remaining_share(2) == Decimal("0.84")
    assert curve.remaining(Decimal("1000"), age_years=2) == Decimal("840.0000")
    assert curve.anchor_cohort_count == 2
    assert curve.transition_counts == ((1, 2),)


def test_calibration_caps_net_stock_growth_and_stays_monotone() -> None:
    registrations = (RegistrationCohort(2020, Decimal("100")),)
    stocks = (
        CohortStock(2020, 2021, Decimal("90")),
        CohortStock(2020, 2022, Decimal("99")),
        CohortStock(2020, 2023, Decimal("80")),
    )

    curve = calibrate_licensed_stock_curve(registrations, stocks)

    assert curve.remaining_share(1) == Decimal("0.9")
    assert curve.remaining_share(2) == Decimal("0.9")
    assert curve.remaining_share(3) < curve.remaining_share(2)


def test_calibration_excludes_a_requested_cohort_for_cross_validation() -> None:
    registrations = (
        RegistrationCohort(2019, Decimal("100")),
        RegistrationCohort(2020, Decimal("100")),
    )
    stocks = (
        CohortStock(2019, 2020, Decimal("80")),
        CohortStock(2019, 2021, Decimal("72")),
        CohortStock(2020, 2021, Decimal("100")),
        CohortStock(2020, 2022, Decimal("100")),
    )

    curve = calibrate_licensed_stock_curve(
        registrations,
        stocks,
        excluded_cohort_years=frozenset({2020}),
    )

    assert curve.remaining_share(1) == Decimal("0.8")
    assert curve.remaining_share(2) == Decimal("0.72")
    assert curve.anchor_cohort_count == 1


def test_zero_registration_cohort_cannot_influence_transitions() -> None:
    registrations = (
        RegistrationCohort(2019, Decimal("100")),
        RegistrationCohort(2020, Decimal("0")),
    )
    stocks = (
        CohortStock(2019, 2020, Decimal("80")),
        CohortStock(2019, 2021, Decimal("72")),
        CohortStock(2020, 2021, Decimal("1000")),
        CohortStock(2020, 2022, Decimal("1000")),
    )

    curve = calibrate_licensed_stock_curve(registrations, stocks)

    assert curve.remaining_share(2) == Decimal("0.72")
    assert curve.transition_counts == ((1, 1),)


def test_stock_only_older_cohorts_extend_curve_without_changing_anchor() -> None:
    registrations = (RegistrationCohort(2020, Decimal(100)),)
    stocks = (
        CohortStock(2020, 2021, Decimal(90)),
        CohortStock(2010, 2011, Decimal(50)),
        CohortStock(2010, 2012, Decimal(40)),
    )

    curve = calibrate_licensed_stock_curve(registrations, stocks)

    assert curve.remaining_share(1) == Decimal('0.9')
    assert curve.remaining_share(2) == Decimal('0.72')
    assert curve.transition_counts == ((1, 1),)


def test_calibration_stops_at_configured_maximum_age() -> None:
    registrations = (RegistrationCohort(2020, Decimal(100)),)
    stocks = tuple(CohortStock(2020, year, Decimal(90)) for year in range(2021, 2027))

    curve = calibrate_licensed_stock_curve(registrations, stocks, maximum_age_years=3)

    assert len(curve.shares) == 4
    with pytest.raises(ValueError, match='exceeds calibrated support'):
        curve.remaining_share(4)


@pytest.mark.parametrize(
    ("registrations", "stocks", "message"),
    [
        ((), (), "registration cohorts"),
        (
            (RegistrationCohort(2020, Decimal("0")),),
            (CohortStock(2020, 2021, Decimal("1")),),
            "positive registration exposure",
        ),
        (
            (RegistrationCohort(2020, Decimal("1")),),
            (CohortStock(2020, 2020, Decimal("1")),),
            "age-one stock anchor",
        ),
    ],
)
def test_calibration_fails_closed_without_usable_evidence(
    registrations: tuple[RegistrationCohort, ...],
    stocks: tuple[CohortStock, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        calibrate_licensed_stock_curve(registrations, stocks)


def test_the_tail_extends_the_curve_beyond_its_calibrated_support() -> None:
    registrations = (RegistrationCohort(2020, Decimal(100)),)
    stocks = tuple(CohortStock(2020, year, Decimal(90) * Decimal("0.9") ** (year - 2021))
                   for year in range(2021, 2025))
    curve = calibrate_licensed_stock_curve(registrations, stocks, maximum_age_years=4)

    extended = extend_with_monotone_tail(curve, maximum_age_years=10)

    assert len(extended.shares) == 11
    assert extended.shares[: len(curve.shares)] == curve.shares
    assert extended.calibrated_support_age == len(curve.shares) - 1


def test_the_tail_never_increases_and_stays_below_the_last_observed_share() -> None:
    registrations = (RegistrationCohort(2020, Decimal(100)),)
    stocks = tuple(CohortStock(2020, year, Decimal(90) * Decimal("0.9") ** (year - 2021))
                   for year in range(2021, 2025))
    curve = calibrate_licensed_stock_curve(registrations, stocks, maximum_age_years=4)

    extended = extend_with_monotone_tail(curve, maximum_age_years=30)

    shares = extended.shares
    assert all(later <= earlier for earlier, later in zip(shares[:-1], shares[1:], strict=True))
    last_calibrated = curve.shares[-1]
    assert all(share < last_calibrated for share in shares[len(curve.shares):])


def test_the_tail_decays_at_the_recent_calibrated_rate() -> None:
    """A steady 0.9 transition must continue as 0.9, not as some invented shape."""
    registrations = (RegistrationCohort(2020, Decimal(100)),)
    stocks = tuple(CohortStock(2020, year, Decimal(90) * Decimal("0.9") ** (year - 2021))
                   for year in range(2021, 2028))
    curve = calibrate_licensed_stock_curve(registrations, stocks, maximum_age_years=7)

    extended = extend_with_monotone_tail(curve, maximum_age_years=12)

    step = extended.remaining_share(9) / extended.remaining_share(8)
    assert abs(step - Decimal("0.9")) < Decimal("0.001")


def test_a_curve_that_cannot_decay_is_rejected_at_calibration_time() -> None:
    """A flat curve gives no evidence of a decay rate, so it must fail loudly."""
    registrations = (RegistrationCohort(2020, Decimal(100)),)
    stocks = tuple(CohortStock(2020, year, Decimal(100)) for year in range(2021, 2028))
    curve = calibrate_licensed_stock_curve(registrations, stocks, maximum_age_years=7)

    with pytest.raises(ValueError, match="tail retention"):
        extend_with_monotone_tail(curve, maximum_age_years=12)


def test_extending_to_the_existing_support_changes_nothing() -> None:
    registrations = (RegistrationCohort(2020, Decimal(100)),)
    stocks = tuple(CohortStock(2020, year, Decimal(90) * Decimal("0.9") ** (year - 2021))
                   for year in range(2021, 2025))
    curve = calibrate_licensed_stock_curve(registrations, stocks, maximum_age_years=4)

    extended = extend_with_monotone_tail(curve, maximum_age_years=len(curve.shares) - 1)

    assert extended.shares == curve.shares


def test_the_tail_cannot_shorten_the_curve() -> None:
    registrations = (RegistrationCohort(2020, Decimal(100)),)
    stocks = tuple(CohortStock(2020, year, Decimal(90) * Decimal("0.9") ** (year - 2021))
                   for year in range(2021, 2025))
    curve = calibrate_licensed_stock_curve(registrations, stocks, maximum_age_years=4)

    with pytest.raises(ValueError, match="maximum age"):
        extend_with_monotone_tail(curve, maximum_age_years=2)
