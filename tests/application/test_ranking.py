from icor.application.ranking import (
    DemandPopulation,
    DemandScoreV2,
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


def test_the_largest_demand_scores_one_hundred_on_demand_alone() -> None:
    score = DemandScoreV2().score((candidate("a", 100),))[0]

    assert score.demand_percentile == 1
    assert score.demand_points == 100
    assert score.total_points == 100
    assert (score.strategy_name, score.strategy_version) == ("demand", "2")


def test_icor_coverage_never_changes_a_score() -> None:
    """ICOR having worked on a car is shown beside the score, never inside it."""

    market = (candidate("low", 100), candidate("high", 300))
    uncovered = DemandScoreV2().score(market)
    covered = DemandScoreV2().score(
        (candidate("low", 100, exact=60, fallback=40), candidate("high", 300, fallback=300))
    )

    assert [score.total_points for score in covered] == [
        score.total_points for score in uncovered
    ]
    assert [score.total_points for score in covered] == [0, 100]


def test_equal_demand_uses_the_same_average_rank_percentile() -> None:
    scores = DemandScoreV2().score(
        (candidate("low", 100), candidate("tie-b", 200), candidate("tie-a", 200))
    )

    by_id = {score.group_id: score for score in scores}
    assert by_id["low"].demand_percentile == 0
    assert by_id["tie-a"].demand_percentile == 0.75
    assert by_id["tie-b"].demand_percentile == 0.75


def test_all_zero_candidates_receive_zero_points_without_division_error() -> None:
    scores = DemandScoreV2().score((candidate("b", 0), candidate("a", 0)))

    assert [(score.group_id, score.total_points) for score in scores] == [
        ("b", 0),
        ("a", 0),
    ]


def test_population_argument_sets_the_percentile_denominator() -> None:
    """A narrowed ranking is still scored against the market it came from."""

    market = tuple(candidate(f"m{index}", (index + 1) * 10) for index in range(10))
    shown = (market[0], market[9])

    scores = DemandScoreV2().score(shown, population=market)

    by_id = {score.group_id: score for score in scores}
    assert by_id["m0"].demand_percentile == 0
    assert by_id["m9"].demand_percentile == 1
    assert by_id["m9"].demand_rank == 1
    assert by_id["m0"].demand_rank == 10
    assert {score.demand_population for score in scores} == {10}


def test_a_candidate_is_scored_from_its_population_entry() -> None:
    """The filtered units are not the units the score is computed from.

    A market filter leaves a vehicle carrying a fraction of its demand.
    Scoring that fraction is what made a score mean something different on
    every screen.
    """

    whole = candidate("golf", 1_000)
    slice_of_it = candidate("golf", 10)
    others = tuple(candidate(f"other{index}", 100) for index in range(4))

    filtered = DemandScoreV2().score(
        (slice_of_it,), population=(whole, *others)
    )[0]
    unfiltered = DemandScoreV2().score((whole,), population=(whole, *others))[0]

    assert filtered == unfiltered
    assert filtered.demand_percentile == 1
    assert filtered.total_points == 100


def test_a_vehicle_absent_from_the_population_is_scored_from_its_own_units() -> None:
    """A group the population does not know must not vanish or throw."""

    score = DemandScoreV2().score(
        (candidate("unknown", 50),), population=(candidate("known", 100),)
    )[0]

    assert score.demand_population == 1
    assert score.demand_rank is None


def test_zero_demand_is_unranked_and_leaves_the_denominator_alone() -> None:
    """Counting vehicles that forecast nothing would flatten the scale.

    Three of these five forecast nothing. Were they members, the smaller of the
    two real vehicles would sit at the halfway mark rather than at the bottom.
    """

    scores = DemandScoreV2().score(
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
