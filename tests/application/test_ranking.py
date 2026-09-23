from icor.application.ranking import (
    DemandPopulation,
    DemandReadinessV1,
    demand_percentile_rank,
)
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



def test_population_argument_sets_the_percentile_denominator() -> None:
    """A narrowed ranking is still scored against the market it came from."""

    market = tuple(candidate(f"m{index}", (index + 1) * 10) for index in range(10))
    shown = (market[0], market[9])

    scores = DemandReadinessV1().score(shown, population=market)

    by_id = {score.group_id: score for score in scores}
    assert by_id["m0"].demand_percentile == 0
    assert by_id["m9"].demand_percentile == 1
    assert by_id["m9"].demand_rank == 1
    assert by_id["m0"].demand_rank == 10
    assert {score.demand_population for score in scores} == {10}


def test_a_candidate_is_scored_from_its_population_entry() -> None:
    """The filtered units are not the units the score is computed from.

    A market filter leaves a vehicle carrying a fraction of its demand and none
    of its coverage. Scoring that fraction is what made a score mean something
    different on every screen.
    """

    whole = candidate("golf", 1_000, exact=1_000)
    slice_of_it = candidate("golf", 10)
    others = tuple(candidate(f"other{index}", 100) for index in range(4))

    filtered = DemandReadinessV1().score(
        (slice_of_it,), population=(whole, *others)
    )[0]
    unfiltered = DemandReadinessV1().score((whole,), population=(whole, *others))[0]

    assert filtered == unfiltered
    assert filtered.demand_percentile == 1
    assert filtered.readiness_ratio == 1
    assert filtered.total_points == 100


def test_a_vehicle_absent_from_the_population_is_scored_from_its_own_units() -> None:
    """A group the population does not know must not vanish or throw."""

    score = DemandReadinessV1().score(
        (candidate("unknown", 50),), population=(candidate("known", 100),)
    )[0]

    assert score.demand_population == 1
    assert score.demand_rank is None


def test_zero_demand_is_unranked_and_leaves_the_denominator_alone() -> None:
    """Counting vehicles that forecast nothing would flatten the scale.

    Three of these five forecast nothing. Were they members, the smaller of the
    two real vehicles would sit at the halfway mark rather than at the bottom.
    """

    scores = DemandReadinessV1().score(
        (
            candidate("empty-a", 0),
            candidate("empty-b", 0),
            candidate("empty-c", 0),
            candidate("small", 100),
            candidate("large", 200),
        )
    )

    by_id = {score.group_id: score for score in scores}
    assert by_id["small"].demand_percentile == 0
    assert by_id["large"].demand_percentile == 1
    assert by_id["small"].demand_rank == 2
    assert by_id["empty-a"].demand_rank is None
    assert by_id["empty-a"].demand_percentile == 0
    assert {score.demand_population for score in scores} == {2}


def test_the_population_agrees_with_the_single_value_percentile() -> None:
    """One definition of "percentile", asserted rather than hoped for.

    ``DemandPopulation`` exists because asking ``demand_percentile_rank`` once
    per member is quadratic on fifty thousand groups. The batch answer has to be
    the same answer, or the ranking and the vehicle forecast drift apart.
    """

    values = [0, 0, 5, 5, 5, 11, 11, 40, 91, 91, 91, 300]
    population = DemandPopulation.of(values)
    ranked = sorted(value for value in values if value > 0)

    for value in sorted(set(ranked)):
        assert population.percentile(value) == demand_percentile_rank(ranked, value)
        assert population.rank(value) == sum(
            1 for item in ranked if item > value
        ) + 1
    assert population.size == len(ranked)
    assert population.percentile(0) == 0.0
    assert population.rank(0) is None
