from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from icor.domain.evidence import ConfidenceBand
from icor.domain.generations import (
    AssignmentMethod,
    GenerationAlternative,
    GenerationAssignment,
    GenerationEntry,
    GenerationIdentityKind,
)


def generation(**overrides: object) -> GenerationEntry:
    values: dict[str, object] = {
        "generation_id": "generation-volkswagen-golf-8-eu",
        "canonical_vehicle_id": "vehicle-volkswagen-golf-eu",
        "display_name": "Golf VIII",
        "market": "EU",
        "start_month": date(2019, 10, 1),
        "end_month": None,
        "identity_kind": GenerationIdentityKind.MANUFACTURER_CONFIRMED,
        "body_style": None,
        "facelift": None,
        "platform": "MQB Evo",
        "evidence_ids": ("evidence-vw-golf-history",),
        "dependency_groups": ("volkswagen-archive",),
        "confidence_reasons": ("Manufacturer history gives the European launch month.",),
        "registry_version": "eu-generation-registry-v1",
    }
    values.update(overrides)
    return GenerationEntry(**values)  # type: ignore[arg-type]


def test_generation_window_uses_month_precision_and_can_be_open_ended() -> None:
    entry = generation()

    assert entry.start_month == date(2019, 10, 1)
    assert entry.end_month is None


def test_estimated_generation_cannot_claim_an_official_designation() -> None:
    with pytest.raises(ValueError, match="estimated generation label"):
        generation(
            identity_kind=GenerationIdentityKind.ESTIMATED,
            display_name="Golf Mk8",
        )


def test_generation_rejects_reversed_or_non_month_windows() -> None:
    with pytest.raises(ValueError, match="month"):
        generation(start_month=date(2019, 10, 2))
    with pytest.raises(ValueError, match="ordered"):
        generation(start_month=date(2020, 1, 1), end_month=date(2019, 12, 1))


def test_assignment_keeps_one_selection_and_ranked_alternatives() -> None:
    assignment = GenerationAssignment(
        assignment_id="assignment-eea-golf-2020",
        observation_id="observation-eea-golf-2020",
        selected_generation_id="generation-volkswagen-golf-8-eu",
        alternatives=(
            GenerationAlternative(
                generation_id="generation-volkswagen-golf-7-eu",
                rank=2,
                loss_reason="fewer-active-months",
            ),
        ),
        method=AssignmentMethod.ACTIVE_MONTH_COVERAGE,
        evidence_ids=("evidence-vw-golf-history",),
        confidence=ConfidenceBand.LOW,
        reason_codes=("transition-year-tiebreak",),
        training_weight=Decimal("0.55"),
        resolver_version="generation-resolver-v1",
        registry_version="eu-generation-registry-v1",
        reviewed_at=datetime(2026, 8, 28, 8, 0, tzinfo=UTC),
    )

    assert assignment.selected_generation_id not in {
        alternative.generation_id for alternative in assignment.alternatives
    }
    assert assignment.training_weight == Decimal("0.55")


@pytest.mark.parametrize("weight", (Decimal("-0.01"), Decimal("1.01"), Decimal("NaN")))
def test_assignment_rejects_invalid_training_weight(weight: Decimal) -> None:
    with pytest.raises(ValueError, match="training weight"):
        GenerationAssignment(
            assignment_id="assignment-eea-golf-2020",
            observation_id="observation-eea-golf-2020",
            selected_generation_id="generation-volkswagen-golf-8-eu",
            alternatives=(),
            method=AssignmentMethod.UNIQUE_WINDOW,
            evidence_ids=("evidence-vw-golf-history",),
            confidence=ConfidenceBand.HIGH,
            reason_codes=("unique-window",),
            training_weight=weight,
            resolver_version="generation-resolver-v1",
            registry_version="eu-generation-registry-v1",
            reviewed_at=datetime(2026, 8, 28, 8, 0, tzinfo=UTC),
        )
