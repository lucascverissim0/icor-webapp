"""Replaceable, auditable opportunity-ranking policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from icor.domain.opportunities import OpportunityCandidate, OpportunityScore

#: Demand is worth 80 of the 100 points; production readiness carries the rest.
DEMAND_POINTS_MAX = 80.0


class RankingStrategy(Protocol):
    name: str
    version: str

    def score(
        self, candidates: tuple[OpportunityCandidate, ...]
    ) -> tuple[OpportunityScore, ...]: ...


class DemandReadinessV1:
    name = "demand_readiness"
    version = "1"

    def score(
        self, candidates: tuple[OpportunityCandidate, ...]
    ) -> tuple[OpportunityScore, ...]:
        percentiles = _demand_percentiles(candidates)
        scores: list[OpportunityScore] = []
        for candidate in candidates:
            percentile = percentiles[candidate.group_id]
            if candidate.demand.base_units == 0:
                readiness_ratio = 0.0
            else:
                readiness_ratio = (
                    candidate.exact_covered_base_units
                    + candidate.fallback_covered_base_units * 0.5
                ) / candidate.demand.base_units
            demand_points = percentile * DEMAND_POINTS_MAX
            readiness_points = readiness_ratio * 20
            scores.append(
                OpportunityScore(
                    group_id=candidate.group_id,
                    demand_percentile=percentile,
                    demand_points=demand_points,
                    readiness_ratio=readiness_ratio,
                    readiness_points=readiness_points,
                    total_points=demand_points + readiness_points,
                    strategy_name=self.name,
                    strategy_version=self.version,
                    explanation=(
                        f"{demand_points:g} demand points and "
                        f"{readiness_points:g} production-readiness points."
                    ),
                )
            )
        return tuple(scores)


def demand_percentile_rank(sorted_values: Sequence[float], value: float) -> float:
    """Where one demand figure sits in its population, from 0.0 to 1.0.

    Ties share the mean of the positions they occupy, so two vehicles carrying
    the same demand always receive the same percentile. Extracted from the
    ranking strategy because the vehicle forecast now reports a rank for a
    single selection, and a second definition of "percentile" would let the two
    channels disagree about the same vehicle.
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


def _demand_percentiles(
    candidates: tuple[OpportunityCandidate, ...],
) -> dict[str, float]:
    if not candidates:
        return {}
    values = sorted(candidate.demand.base_units for candidate in candidates)
    return {
        candidate.group_id: demand_percentile_rank(values, candidate.demand.base_units)
        for candidate in candidates
    }
