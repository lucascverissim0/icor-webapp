from decimal import Decimal

from icor.forecasting.survival import CohortSurvivalModel


def test_one_year_old_cohort_receives_one_full_year_of_attrition() -> None:
    model = CohortSurvivalModel()

    assert model.remaining(Decimal("1000"), age_years=1) == Decimal("944.4000")


def test_survival_interval_is_ordered() -> None:
    interval = CohortSurvivalModel().interval(Decimal("1000"), age_years=8)

    assert interval.p10 <= interval.p50 <= interval.p90
