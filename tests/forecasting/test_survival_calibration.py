from decimal import Decimal

import pytest

from icor.forecasting.survival_calibration import (
    CohortStock,
    RegistrationCohort,
    calibrate_licensed_stock_curve,
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
