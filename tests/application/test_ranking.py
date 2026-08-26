from icor.application.ranking import DemandReadinessV1
from icor.domain.opportunities import OpportunityCandidate
from icor.domain.planner import DemandRange


def candidate(
    identity: str,
    base: int,
    *,
    exact: int = 0,
    fallback: int = 0,
) -> OpportunityCandidate:
    return OpportunityCandidate(
        group_id=identity,
        demand=DemandRange(
            downside_units=max(0, base - 10),
            base_units=base,
            upside_units=base + 10,
        ),
        exact_covered_base_units=exact,
        fallback_covered_base_units=fallback,
        uncovered_base_units=base - exact - fallback,
    )


def test_single_non_zero_exact_candidate_receives_full_score() -> None:
    score = DemandReadinessV1().score((candidate("a", 100, exact=100),))[0]

    assert score.demand_percentile == 1
    assert score.demand_points == 80
    assert score.readiness_ratio == 1
    assert score.readiness_points == 20
    assert score.total_points == 100


def test_fallback_units_receive_half_readiness_weight() -> None:
    score = DemandReadinessV1().score((candidate("a", 100, fallback=100),))[0]

    assert score.readiness_ratio == 0.5
    assert score.readiness_points == 10


def test_equal_demand_uses_the_same_average_rank_percentile() -> None:
    scores = DemandReadinessV1().score(
        (candidate("low", 100), candidate("tie-b", 200), candidate("tie-a", 200))
    )

    by_id = {score.group_id: score for score in scores}
    assert by_id["low"].demand_percentile == 0
    assert by_id["tie-a"].demand_percentile == 0.75
    assert by_id["tie-b"].demand_percentile == 0.75


def test_all_zero_candidates_receive_zero_points_without_division_error() -> None:
    scores = DemandReadinessV1().score((candidate("b", 0), candidate("a", 0)))

    assert [(score.group_id, score.total_points) for score in scores] == [
        ("b", 0),
        ("a", 0),
    ]


def test_mixed_coverage_does_not_double_count_readiness() -> None:
    score = DemandReadinessV1().score(
        (candidate("mixed", 100, exact=40, fallback=30),)
    )[0]

    assert score.readiness_ratio == 0.55
    assert score.readiness_points == 11

