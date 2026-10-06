"""Replaceable, auditable opportunity-ranking policies."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from icor.domain.opportunities import OpportunityCandidate, OpportunityScore

#: The score is demand alone, on a 0-100 scale. ICOR production readiness was
#: removed from it on 2026-10-06 at ICOR's request: whether ICOR has already
#: worked on a vehicle is reported beside the score, never inside it.
DEMAND_POINTS_MAX = 100.0

#: What a demand percentile is measured against, reported beside every score so
#: a consumer can tell which population produced it. The vehicle forecast
#: reports its own, narrower basis in the same field on its own response.
WHOLE_MARKET_BASIS = "whole_market_base_units_all_markets_all_horizons"


class RankingStrategy(Protocol):
    name: str
    version: str

    def score(
        self,
        candidates: tuple[OpportunityCandidate, ...],
        *,
        population: tuple[OpportunityCandidate, ...] | None = None,
    ) -> tuple[OpportunityScore, ...]: ...


@dataclass(frozen=True, slots=True)
class DemandPopulation:
    """Where each demand figure sits among the figures worth ranking.

    Built once for a whole population and then asked about individual values,
    because the alternative — calling ``demand_percentile_rank`` once per
    member — rescans the population every time and is quadratic on the fifty
    thousand groups a model-year ranking holds.

    Vehicles with no forecast demand are **not** members. They still score
    zero, but counting them would put the smallest vehicle anyone can actually
    see halfway up the scale: on the promoted snapshot 26,695 of 50,811 model
    years forecast nothing, so including them would give every visible vehicle
    at least half of the demand points and flatten the ranking.
    """

    size: int
    percentiles: dict[float, float] = field(repr=False)
    ranks: dict[float, int] = field(repr=False)

    @classmethod
    def of(cls, values: Iterable[float]) -> DemandPopulation:
        ranked = sorted(value for value in values if value > 0)
        total = len(ranked)
        percentiles: dict[float, float] = {}
        ranks: dict[float, int] = {}
        denominator = total - 1
        start = 0
        while start < total:
            value = ranked[start]
            end = start
            while end + 1 < total and ranked[end + 1] == value:
                end += 1
            count = end - start + 1
            # Ties share the mean of the positions they occupy, so two vehicles
            # carrying the same demand always receive the same percentile.
            percentiles[value] = (
                1.0 if denominator == 0 else (start + (count - 1) / 2) / denominator
            )
            # One is the largest, matching the direction the vehicle forecast
            # counts in, so the two channels cannot disagree about a rank.
            ranks[value] = total - start - count + 1
            start = end + 1
        return cls(size=total, percentiles=percentiles, ranks=ranks)

    def percentile(self, value: float) -> float:
        """Where ``value`` sits, from 0.0 to 1.0; zero demand ranks nowhere."""

        return self.percentiles.get(value, 0.0) if value > 0 else 0.0

    def rank(self, value: float) -> int | None:
        """The 1-is-largest position of ``value``, or None when unranked."""

        return self.ranks.get(value) if value > 0 else None


class DemandScoreV2:
    name = "demand"
    version = "2"

    def score(
        self,
        candidates: tuple[OpportunityCandidate, ...],
        *,
        population: tuple[OpportunityCandidate, ...] | None = None,
    ) -> tuple[OpportunityScore, ...]:
        """Score candidates against ``population``, or against themselves.

        The population is a separate argument so that narrowing a ranking
        cannot change what a score means. A vehicle is scored from its entry in
        the population rather than from the units that survived the filter, so
        the score is a property of the vehicle. Coverage is deliberately not
        read: it is displayed, not scored.
        """

        members = population if population is not None else candidates
        by_group = {member.group_id: member for member in members}
        demand = DemandPopulation.of(member.demand.base_units for member in members)
        scores: list[OpportunityScore] = []
        for candidate in candidates:
            member = by_group.get(candidate.group_id, candidate)
            units = member.demand.base_units
            percentile = demand.percentile(units)
            points = percentile * DEMAND_POINTS_MAX
            scores.append(
                OpportunityScore(
                    group_id=candidate.group_id,
                    demand_percentile=percentile,
                    demand_points=points,
                    demand_rank=demand.rank(units),
                    demand_population=demand.size,
                    demand_basis=WHOLE_MARKET_BASIS,
                    total_points=points,
                    strategy_name=self.name,
                    strategy_version=self.version,
                    explanation=demand_explanation(points),
                )
            )
        return tuple(scores)


def demand_explanation(points: float) -> str:
    """One wording for every channel that reports a score."""

    return f"{points:g} of 100 points from whole-market replacement demand."


def demand_percentile_rank(sorted_values: Sequence[float], value: float) -> float:
    """Where one demand figure sits in its population, from 0.0 to 1.0.

    Ties share the mean of the positions they occupy, so two vehicles carrying
    the same demand always receive the same percentile. Extracted from the
    ranking strategy because the vehicle forecast now reports a rank for a
    single selection, and a second definition of "percentile" would let the two
    channels disagree about the same vehicle. ``DemandPopulation`` answers the
    same question for a whole population at once and is tested against this.
    """

    if not sorted_values or not any(sorted_values):
        return 0.0
    if len(sorted_values) == 1:
        return 1.0
    denominator = len(sorted_values) - 1
    positions = [index for index, item in enumerate(sorted_values) if item == value]
    if not positions:
        # Not a member of the population: report where it would land instead.
        # Clamped, because a value above every entry counts len(values) items
        # below it, which would otherwise exceed 1.0.
        return min(sum(1 for item in sorted_values if item < value) / denominator, 1.0)
    return (sum(positions) / len(positions)) / denominator
