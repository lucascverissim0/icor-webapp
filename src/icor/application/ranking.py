"""Replaceable, auditable opportunity-ranking policies."""

from __future__ import annotations

from typing import Protocol

from icor.domain.opportunities import OpportunityCandidate, OpportunityScore


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
            demand_points = percentile * 80
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


def _demand_percentiles(
    candidates: tuple[OpportunityCandidate, ...],
) -> dict[str, float]:
    if not candidates:
        return {}
    values = sorted(candidate.demand.base_units for candidate in candidates)
    if not any(values):
        return {candidate.group_id: 0.0 for candidate in candidates}
    if len(values) == 1:
        return {candidates[0].group_id: 1.0}
    percentiles: dict[str, float] = {}
    denominator = len(values) - 1
    for candidate in candidates:
        positions = [
            index for index, value in enumerate(values) if value == candidate.demand.base_units
        ]
        percentiles[candidate.group_id] = (sum(positions) / len(positions)) / denominator
    return percentiles
