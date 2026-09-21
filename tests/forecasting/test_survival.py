from decimal import Decimal

import pytest

from icor.forecasting.survival import CohortSurvivalModel


def test_one_year_old_cohort_receives_one_full_year_of_attrition() -> None:
    model = CohortSurvivalModel()

    assert model.remaining(Decimal("1000"), age_years=1) == Decimal("944.4000")


def test_survival_interval_is_ordered() -> None:
    interval = CohortSurvivalModel().interval(Decimal("1000"), age_years=8)

    assert interval.p10 <= interval.p50 <= interval.p90


def test_an_unaged_cohort_loses_nothing() -> None:
    assert CohortSurvivalModel().remaining(Decimal("1000"), age_years=0) == Decimal("1000.0000")


def test_retention_compounds_across_years() -> None:
    model = CohortSurvivalModel()

    assert model.remaining(Decimal("1000"), age_years=2) == Decimal("891.8914")
    assert model.remaining(Decimal("1000"), age_years=8) == Decimal("632.7728")


def test_remaining_never_increases_with_age() -> None:
    model = CohortSurvivalModel()

    remaining = [model.remaining(Decimal("1000"), age_years=age) for age in range(0, 21)]

    pairs = zip(remaining[:-1], remaining[1:], strict=True)
    assert all(later <= earlier for earlier, later in pairs)


def test_the_interval_brackets_the_central_estimate() -> None:
    model = CohortSurvivalModel()

    interval = model.interval(Decimal("1000"), age_years=8)

    assert interval.p10 == Decimal("513.2189")
    assert interval.p50 == model.remaining(Decimal("1000"), age_years=8)
    assert interval.p90 == Decimal("752.0012")


def test_a_calibrated_retention_curve_can_replace_the_assumed_one() -> None:
    """Part D swaps a calibrated model in, so custom retentions must be honoured."""
    model = CohortSurvivalModel(
        retention_p10=Decimal("0.80"),
        retention_p50=Decimal("0.90"),
        retention_p90=Decimal("0.95"),
    )

    assert model.remaining(Decimal("1000"), age_years=1) == Decimal("900.0000")


@pytest.mark.parametrize(
    "retentions",
    [
        {"retention_p50": Decimal("0.90"), "retention_p10": Decimal("0.95")},
        {"retention_p90": Decimal("1.01")},
        {"retention_p10": Decimal("0")},
        {"retention_p10": Decimal("-0.1")},
    ],
)
def test_unordered_or_out_of_range_retentions_are_rejected(retentions: dict) -> None:
    with pytest.raises(ValueError, match="ordered probabilities"):
        CohortSurvivalModel(**retentions)


@pytest.mark.parametrize(
    ("registrations", "age_years"),
    [
        (Decimal("-1"), 1),
        (Decimal("NaN"), 1),
        (Decimal("1000"), -1),
        (Decimal("1000"), True),
    ],
)
def test_invalid_cohort_inputs_are_rejected(registrations: Decimal, age_years: int) -> None:
    with pytest.raises(ValueError):
        CohortSurvivalModel().remaining(registrations, age_years=age_years)


def test_the_planning_service_accepts_an_injected_survival_model() -> None:
    """Calibration must be swappable without editing GenerationPlanningService."""
    from icor.application.generation_planning import GenerationPlanningService

    calibrated = CohortSurvivalModel(retention_p50=Decimal("0.93"))

    assert GenerationPlanningService(survival=calibrated).survival is calibrated
    assert isinstance(GenerationPlanningService().survival, CohortSurvivalModel)


def test_assumption_ids_report_the_retentions_actually_used() -> None:
    """A calibrated model must not inherit the default model's provenance strings."""
    model = CohortSurvivalModel(
        retention_p10=Decimal("0.80"),
        retention_p50=Decimal("0.90"),
        retention_p90=Decimal("0.95"),
    )

    assert model.assumption_ids == (
        "survival-retention-p10-0.8",
        "survival-retention-p50-0.9",
        "survival-retention-p90-0.95",
    )


def test_the_default_model_keeps_its_published_assumption_ids() -> None:
    assert CohortSurvivalModel().assumption_ids == (
        "survival-retention-p10-0.92",
        "survival-retention-p50-0.9444",
        "survival-retention-p90-0.965",
    )


def test_provenance_is_per_instance_not_shared_by_the_class() -> None:
    """Two models must never report each other's assumptions."""
    default = CohortSurvivalModel()
    calibrated = CohortSurvivalModel(retention_p50=Decimal("0.93"))

    assert calibrated.assumption_ids != default.assumption_ids
    assert default.assumption_ids == CohortSurvivalModel().assumption_ids


def test_assumption_ids_are_unique_when_retentions_coincide() -> None:
    """Equal retentions are legal, and the evidence record rejects duplicate IDs."""
    model = CohortSurvivalModel(
        retention_p10=Decimal("0.9"),
        retention_p50=Decimal("0.9"),
        retention_p90=Decimal("0.9"),
    )

    assert len(set(model.assumption_ids)) == 3


def test_assumption_ids_are_valid_evidence_identifiers() -> None:
    from re import fullmatch

    from icor.domain.cohorts import _IDENTIFIER

    for identifier in CohortSurvivalModel(retention_p50=Decimal("0.9375")).assumption_ids:
        assert fullmatch(_IDENTIFIER, identifier) is not None


def test_the_method_is_readable_from_the_instance() -> None:
    assert CohortSurvivalModel().method == "constant-annual-retention-v1"
