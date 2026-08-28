from datetime import UTC, date, datetime

from icor.domain.generations import AssignmentMethod, GenerationIdentityKind
from icor.generations.registry import GenerationRegistry
from icor.generations.resolver import GenerationResolver, ResolutionRequest


def entry(identifier: str, start: str, end: str | None):
    from icor.domain.generations import GenerationEntry

    start_year, start_month = (int(part) for part in start.split("-"))
    end_date = None
    if end is not None:
        end_year, end_month = (int(part) for part in end.split("-"))
        end_date = date(end_year, end_month, 1)
    return GenerationEntry(
        generation_id=identifier,
        canonical_vehicle_id="vehicle-volkswagen-golf-eu",
        display_name=identifier,
        market="EU",
        start_month=date(start_year, start_month, 1),
        end_month=end_date,
        identity_kind=GenerationIdentityKind.REGISTRY_CORROBORATED,
        body_style=None,
        facelift=None,
        platform=None,
        evidence_ids=("evidence-generation-window",),
        dependency_groups=("reviewed-registry",),
        confidence_reasons=("Reviewed registry window.",),
        registry_version="eu-generation-registry-v1",
    )


def request(**overrides: object) -> ResolutionRequest:
    values: dict[str, object] = {
        "observation_id": "observation-golf-2020",
        "canonical_vehicle_id": "vehicle-volkswagen-golf-eu",
        "market": "EU",
        "registration_cohort_year": 2020,
        "exact_generation_id": None,
        "descriptor_generation_ids": (),
        "launched_generation_ids": ("generation-golf-8",),
        "reviewed_at": datetime(2026, 8, 28, 8, 0, tzinfo=UTC),
    }
    values.update(overrides)
    return ResolutionRequest(**values)  # type: ignore[arg-type]


def test_exact_identifier_precedes_calendar_window() -> None:
    resolver = GenerationResolver(
        GenerationRegistry(
            (
                entry("generation-golf-7", "2012-09", "2020-12"),
                entry("generation-golf-8", "2019-10", None),
            )
        )
    )

    result = resolver.resolve(request(exact_generation_id="generation-golf-8"))

    assert result.selected_generation_id == "generation-golf-8"
    assert result.method is AssignmentMethod.EXACT_IDENTIFIER


def test_transition_year_prefers_greater_active_month_coverage() -> None:
    resolver = GenerationResolver(
        GenerationRegistry(
            (
                entry("generation-golf-7", "2012-09", "2020-09"),
                entry("generation-golf-8", "2020-10", None),
            )
        )
    )

    result = resolver.resolve(request())

    assert result.selected_generation_id == "generation-golf-7"
    assert result.method is AssignmentMethod.ACTIVE_MONTH_COVERAGE
    assert [alternative.generation_id for alternative in result.alternatives] == [
        "generation-golf-8"
    ]


def test_equal_transition_coverage_uses_newer_launched_generation() -> None:
    resolver = GenerationResolver(
        GenerationRegistry(
            (
                entry("generation-golf-7", "2012-09", "2020-06"),
                entry("generation-golf-8", "2020-07", None),
            )
        )
    )

    result = resolver.resolve(request())

    assert result.selected_generation_id == "generation-golf-8"
    assert result.method is AssignmentMethod.NEWER_LAUNCH_TIEBREAK
