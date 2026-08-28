from decimal import Decimal

import pytest

from icor.domain.cohorts import CohortEstimate, CompletenessRecord, OpportunityEstimate
from icor.domain.evidence import ConfidenceBand


def cohort(**overrides: object) -> CohortEstimate:
    values: dict[str, object] = {
        "cohort_id": "cohort-golf-8-eu-2020-2028",
        "generation_id": "generation-volkswagen-golf-8-eu",
        "canonical_vehicle_id": "vehicle-volkswagen-golf-eu",
        "geography": "EU-27",
        "registration_cohort_year": 2020,
        "as_of_year": 2028,
        "registrations": Decimal("1000"),
        "active_fleet_p10": Decimal("650"),
        "active_fleet_p50": Decimal("700"),
        "active_fleet_p90": Decimal("750"),
        "input_observation_ids": ("observation-eea-golf-2020",),
        "survival_method": "eu-passenger-car-survival-v1",
        "confidence": ConfidenceBand.MEDIUM,
        "reason_codes": ("cohort-survival-reconstruction",),
    }
    values.update(overrides)
    return CohortEstimate(**values)  # type: ignore[arg-type]


def opportunity(**overrides: object) -> OpportunityEstimate:
    values: dict[str, object] = {
        "opportunity_id": "opportunity-golf-8-eu-2028",
        "generation_id": "generation-volkswagen-golf-8-eu",
        "canonical_vehicle_id": "vehicle-volkswagen-golf-eu",
        "geography": "EU-27",
        "horizon_year": 2028,
        "p10": Decimal("8"),
        "p50": Decimal("10"),
        "p90": Decimal("14"),
        "active_fleet_p50": Decimal("700"),
        "input_cohort_ids": ("cohort-golf-8-eu-2020-2028",),
        "hazard_method": "assumption-led-windshield-hazard-v1",
        "forecast_method": "generation-opportunity-v1",
        "confidence": ConfidenceBand.LOW,
        "assumption_ids": ("assumption-windshield-hazard-eu-v1",),
        "reason_codes": ("uncalibrated-proprietary-fitment",),
    }
    values.update(overrides)
    return OpportunityEstimate(**values)  # type: ignore[arg-type]


def test_cohort_interval_is_ordered_and_preserves_registration_semantics() -> None:
    estimate = cohort()

    assert estimate.registration_cohort_year == 2020
    assert estimate.active_fleet_p10 <= estimate.active_fleet_p50 <= estimate.active_fleet_p90


def test_opportunity_interval_is_ordered_and_non_negative() -> None:
    estimate = opportunity()

    assert Decimal(0) <= estimate.p10 <= estimate.p50 <= estimate.p90


@pytest.mark.parametrize(
    ("factory", "overrides"),
    (
        (cohort, {"active_fleet_p10": Decimal("701")}),
        (cohort, {"registrations": Decimal("-1")}),
        (opportunity, {"p90": Decimal("9")}),
        (opportunity, {"p10": Decimal("NaN")}),
    ),
)
def test_estimates_reject_invalid_or_unordered_values(factory, overrides) -> None:
    with pytest.raises(ValueError, match="ordered|non-negative"):
        factory(**overrides)


def test_completeness_cannot_claim_more_assignments_than_usable_observations() -> None:
    with pytest.raises(ValueError, match="assigned observations"):
        CompletenessRecord(
            completeness_id="completeness-eu-2020",
            geography="EU-27",
            year=2020,
            release_count=2,
            observation_count=100,
            usable_observation_count=80,
            assigned_observation_count=81,
            canonical_family_count=12,
            sourced_generation_count=7,
            estimated_generation_count=5,
            forecastable_count=10,
            evidence_only_count=2,
            rejected_record_count=20,
            reason_codes=("annual-completeness",),
        )
