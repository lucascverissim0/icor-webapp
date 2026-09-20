from dataclasses import replace
from decimal import Decimal

import pytest

from icor.forecasting.promotion_gate import (
    PromotionEvidence,
    PromotionPolicy,
    ValidationSlice,
    evaluate_promotion,
)


def metric(
    name: str,
    *,
    baseline: str = "20",
    challenger: str = "15",
    actual: str = "100",
    interval: tuple[str, str] | None = ("80", "100"),
) -> ValidationSlice:
    return ValidationSlice(
        name=name,
        point_count=10,
        actual_units=Decimal(actual),
        baseline_absolute_error=Decimal(baseline),
        challenger_absolute_error=Decimal(challenger),
        interval_covered_units=(Decimal(interval[0]) if interval else None),
        interval_total_units=(Decimal(interval[1]) if interval else None),
    )


def evidence(**changes) -> PromotionEvidence:
    partition = (
        metric("first", actual="450", baseline="90", challenger="67.5"),
        metric("second", actual="450", baseline="90", challenger="67.5"),
    )
    values = {
        "outcome": "annual_vehicle_registrations",
        "holdout_snapshot_id": "snapshot-new",
        "development_snapshot_ids": ("snapshot-old",),
        "observed_targets_only": True,
        "publication_vintages_as_of_origin": True,
        "unseen_holdout": True,
        "eligible_target_units": Decimal("1000"),
        "evaluated_target_units": Decimal("900"),
        "overall": metric(
            "overall",
            actual="900",
            baseline="180",
            challenger="135",
            interval=("720", "900"),
        ),
        "horizons": tuple(
            replace(item, name=name)
            for item, name in zip(partition, ("h1", "h2"), strict=True)
        ),
        "countries": tuple(
            replace(item, name=name)
            for item, name in zip(partition, ("BE", "FR"), strict=True)
        ),
        "identity_confidence": tuple(
            replace(item, name=name)
            for item, name in zip(partition, ("high", "low"), strict=True)
        ),
        "history_lengths": tuple(
            replace(item, name=name)
            for item, name in zip(partition, ("5-7", "8+"), strict=True)
        ),
        "volume_deciles": tuple(
            replace(item, name=name)
            for item, name in zip(partition, ("d1", "d10"), strict=True)
        ),
    }
    values.update(changes)
    return PromotionEvidence(**values)


def test_complete_unseen_observed_benchmark_can_pass() -> None:
    decision = evaluate_promotion(evidence())

    assert decision.eligible
    assert decision.blockers == ()


@pytest.mark.parametrize(
    ("change", "blocker"),
    (
        ({"observed_targets_only": False}, "targets_include_estimated_or_forecast_values"),
        (
            {"publication_vintages_as_of_origin": False},
            "publication_vintages_do_not_reconstruct_forecast_origin",
        ),
        ({"unseen_holdout": False}, "holdout_not_frozen_and_unseen"),
        (
            {"eligible_target_units": Decimal("1200")},
            "evaluated_volume_coverage_below_policy",
        ),
    ),
)
def test_trust_prerequisites_fail_closed(change, blocker: str) -> None:
    assert blocker in evaluate_promotion(evidence(**change)).blockers


def test_reused_holdout_and_missing_intervals_are_blocking() -> None:
    result = evidence(
        holdout_snapshot_id="snapshot-old",
        overall=metric(
            "overall", actual="900", baseline="180", challenger="135", interval=None
        ),
    )

    assert evaluate_promotion(result).blockers == (
        "holdout_snapshot_used_during_development",
        "prediction_intervals_not_evaluated",
    )


def test_each_horizon_and_material_segment_must_avoid_regression() -> None:
    regression = metric("FR", baseline="90", challenger="110", actual="450")
    result = evidence(
        horizons=(
            metric("h1", actual="450", baseline="90", challenger="67.5"),
            metric("h2", actual="450", baseline="90", challenger="95"),
        ),
        countries=(
            metric("BE", actual="450", baseline="90", challenger="67.5"),
            regression,
        ),
    )

    decision = evaluate_promotion(result)

    assert "challenger_does_not_improve_every_horizon" in decision.blockers
    assert "material_country_regression" in decision.blockers


def test_interval_coverage_must_be_empirically_calibrated() -> None:
    result = evidence(
        overall=metric(
            "overall",
            actual="900",
            baseline="180",
            challenger="135",
            interval=("585", "900"),
        )
    )

    assert "prediction_interval_coverage_outside_policy" in evaluate_promotion(result).blockers


def test_invalid_slice_rejects_partial_or_impossible_interval_counts() -> None:
    with pytest.raises(ValueError, match="required together"):
        replace(metric("bad"), interval_total_units=None)
    with pytest.raises(ValueError, match="ordered and positive"):
        replace(metric("bad"), interval_covered_units=Decimal("101"))


def test_policy_rejects_thresholds_that_cannot_be_interpreted() -> None:
    with pytest.raises(ValueError, match="supported policy range"):
        PromotionPolicy(minimum_evaluated_volume_coverage=Decimal("1.1"))
