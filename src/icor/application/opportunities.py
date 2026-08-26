"""Opportunity aggregation, coverage resolution, scoring, and drill-down."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

from icor.application.coverage import CoverageRepository
from icor.application.planner import PlannerRepository
from icor.application.ranking import RankingStrategy
from icor.domain.opportunities import (
    CoverageMatchType,
    CoverageStatus,
    OpportunityCandidate,
    OpportunityScore,
    ProductionCoverage,
)
from icor.domain.planner import (
    DemandRange,
    EvidenceStatus,
    ModelYearDemand,
    PlanningConfiguration,
)


class OpportunityGroupBy(StrEnum):
    BRAND = "brand"
    MODEL = "model"
    MODEL_YEAR = "model_year"


@dataclass(frozen=True, slots=True)
class OpportunityQuery:
    group_by: OpportunityGroupBy
    markets: tuple[str, ...] = ()
    horizons: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class OpportunitySummary:
    base_units: int
    exact_covered_base_units: int
    high_demand_uncovered_base_units: int


@dataclass(frozen=True, slots=True)
class OpportunityRow:
    group_id: str
    group_by: OpportunityGroupBy
    brand: str
    model: str | None
    model_year: int | None
    demand: DemandRange
    contributing_configuration_count: int
    exact_covered_base_units: int
    fallback_covered_base_units: int
    uncovered_base_units: int
    coverage_status: CoverageStatus
    score: OpportunityScore
    evidence_status: EvidenceStatus
    data_version: str


@dataclass(frozen=True, slots=True)
class OpportunityPage:
    items: tuple[OpportunityRow, ...]
    summary: OpportunitySummary
    strategy_name: str
    strategy_version: str
    integrity_warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OpportunityDrillDownRow:
    configuration: PlanningConfiguration
    model_year_demand: ModelYearDemand
    coverage_status: CoverageStatus


@dataclass(frozen=True, slots=True)
class _DemandAtom:
    configuration: PlanningConfiguration
    model_year_demand: ModelYearDemand
    coverage_status: CoverageStatus


class OpportunityService:
    def __init__(
        self,
        planner_repository: PlannerRepository,
        coverage_repository: CoverageRepository,
        ranking_strategy: RankingStrategy,
    ) -> None:
        self._planner_repository = planner_repository
        self._coverage_repository = coverage_repository
        self._ranking_strategy = ranking_strategy

    def list(self, query: OpportunityQuery) -> OpportunityPage:
        atoms, warnings = self._resolved_atoms(query)
        grouped = self._group(atoms, query.group_by)
        candidates = tuple(
            OpportunityCandidate(
                group_id=group_id,
                demand=_sum_demand(group_atoms),
                exact_covered_base_units=_sum_coverage(
                    group_atoms, CoverageStatus.EXACT_COVERED
                ),
                fallback_covered_base_units=_sum_coverage(
                    group_atoms, CoverageStatus.FALLBACK_ONLY
                ),
                uncovered_base_units=_sum_coverage(
                    group_atoms, CoverageStatus.UNCOVERED
                ),
            )
            for group_id, group_atoms in grouped.items()
        )
        scores = {
            score.group_id: score for score in self._ranking_strategy.score(candidates)
        }
        rows = tuple(
            self._row(group_id, group_atoms, query.group_by, scores[group_id])
            for group_id, group_atoms in grouped.items()
        )
        rows = tuple(
            sorted(
                rows,
                key=lambda row: (-row.score.total_points, -row.demand.base_units, row.group_id),
            )
        )
        return OpportunityPage(
            items=rows,
            summary=OpportunitySummary(
                base_units=sum(row.demand.base_units for row in rows),
                exact_covered_base_units=sum(
                    row.exact_covered_base_units for row in rows
                ),
                high_demand_uncovered_base_units=sum(
                    row.uncovered_base_units
                    for row in rows
                    if row.score.demand_percentile >= 0.75
                ),
            ),
            strategy_name=self._ranking_strategy.name,
            strategy_version=self._ranking_strategy.version,
            integrity_warnings=warnings,
        )

    def drill_down(
        self, group_id: str, query: OpportunityQuery
    ) -> tuple[OpportunityDrillDownRow, ...]:
        atoms, _warnings = self._resolved_atoms(query)
        grouped = self._group(atoms, query.group_by)
        return tuple(
            OpportunityDrillDownRow(
                configuration=atom.configuration,
                model_year_demand=atom.model_year_demand,
                coverage_status=atom.coverage_status,
            )
            for atom in grouped.get(group_id, ())
        )

    def _resolved_atoms(
        self, query: OpportunityQuery
    ) -> tuple[tuple[_DemandAtom, ...], tuple[str, ...]]:
        configurations = self._planner_repository.list_all()
        by_id = {row.configuration_id: row for row in configurations}
        canonical_atoms = tuple(
            (configuration, demand)
            for configuration in configurations
            for demand in configuration.model_year_demand
        )
        canonical_fallbacks = {
            (configuration.brand, configuration.model, demand.model_year)
            for configuration, demand in canonical_atoms
        }
        exact: dict[tuple[str, int], ProductionCoverage] = {}
        fallback: dict[tuple[str, str, int], ProductionCoverage] = {}
        warnings: list[str] = []
        for coverage in self._coverage_repository.list_all():
            if coverage.match_type is CoverageMatchType.EXACT_CONFIGURATION:
                configuration = by_id.get(coverage.configuration_id or "")
                valid = (
                    configuration is not None
                    and configuration.brand == coverage.brand
                    and configuration.model == coverage.model
                    and configuration.sku == coverage.sku
                    and any(
                        demand.model_year == coverage.model_year
                        for demand in configuration.model_year_demand
                    )
                )
                if valid:
                    exact[(configuration.configuration_id, coverage.model_year)] = coverage
                else:
                    warnings.append(
                        f"Coverage {coverage.coverage_id} has an unavailable canonical identity."
                    )
            else:
                key = (coverage.brand, coverage.model, coverage.model_year)
                if key in canonical_fallbacks:
                    fallback[key] = coverage
                else:
                    warnings.append(
                        f"Coverage {coverage.coverage_id} has an unavailable canonical identity."
                    )
        atoms: list[_DemandAtom] = []
        for configuration, demand in canonical_atoms:
            if query.markets and configuration.market not in query.markets:
                continue
            if query.horizons and configuration.forecast_horizon not in query.horizons:
                continue
            if (configuration.configuration_id, demand.model_year) in exact:
                status = CoverageStatus.EXACT_COVERED
            elif (configuration.brand, configuration.model, demand.model_year) in fallback:
                status = CoverageStatus.FALLBACK_ONLY
            else:
                status = CoverageStatus.UNCOVERED
            atoms.append(_DemandAtom(configuration, demand, status))
        return tuple(atoms), tuple(warnings)

    @staticmethod
    def _group(
        atoms: tuple[_DemandAtom, ...], group_by: OpportunityGroupBy
    ) -> dict[str, tuple[_DemandAtom, ...]]:
        groups: dict[str, list[_DemandAtom]] = {}
        for atom in atoms:
            identity = _group_identity(atom, group_by)
            group_id = _group_id(group_by, identity)
            groups.setdefault(group_id, []).append(atom)
        return {key: tuple(value) for key, value in groups.items()}

    @staticmethod
    def _row(
        group_id: str,
        atoms: tuple[_DemandAtom, ...],
        group_by: OpportunityGroupBy,
        score: OpportunityScore,
    ) -> OpportunityRow:
        first = atoms[0]
        identity = _group_identity(first, group_by)
        demand = _sum_demand(atoms)
        exact = _sum_coverage(atoms, CoverageStatus.EXACT_COVERED)
        fallback = _sum_coverage(atoms, CoverageStatus.FALLBACK_ONLY)
        uncovered = _sum_coverage(atoms, CoverageStatus.UNCOVERED)
        return OpportunityRow(
            group_id=group_id,
            group_by=group_by,
            brand=identity[0],
            model=identity[1] if len(identity) > 1 else None,
            model_year=identity[2] if len(identity) > 2 else None,
            demand=demand,
            contributing_configuration_count=len(
                {atom.configuration.configuration_id for atom in atoms}
            ),
            exact_covered_base_units=exact,
            fallback_covered_base_units=fallback,
            uncovered_base_units=uncovered,
            coverage_status=_coverage_status(demand.base_units, exact, fallback, uncovered),
            score=score,
            evidence_status=min(
                (atom.model_year_demand.evidence_status for atom in atoms),
                key=lambda status: list(EvidenceStatus).index(status),
            ),
            data_version=first.model_year_demand.data_version,
        )


def _group_identity(
    atom: _DemandAtom, group_by: OpportunityGroupBy
) -> tuple[str, ...] | tuple[str, str, int]:
    configuration = atom.configuration
    if group_by is OpportunityGroupBy.BRAND:
        return (configuration.brand,)
    if group_by is OpportunityGroupBy.MODEL:
        return configuration.brand, configuration.model
    return configuration.brand, configuration.model, atom.model_year_demand.model_year


def _group_id(group_by: OpportunityGroupBy, identity: tuple[object, ...]) -> str:
    payload = json.dumps((group_by.value, *identity), ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{group_by.value}-{digest}"


def _sum_demand(atoms: tuple[_DemandAtom, ...]) -> DemandRange:
    return DemandRange(
        downside_units=sum(atom.model_year_demand.demand.downside_units for atom in atoms),
        base_units=sum(atom.model_year_demand.demand.base_units for atom in atoms),
        upside_units=sum(atom.model_year_demand.demand.upside_units for atom in atoms),
    )


def _sum_coverage(atoms: tuple[_DemandAtom, ...], status: CoverageStatus) -> int:
    return sum(
        atom.model_year_demand.demand.base_units
        for atom in atoms
        if atom.coverage_status is status
    )


def _coverage_status(
    base: int, exact: int, fallback: int, uncovered: int
) -> CoverageStatus:
    if base == 0 or (exact == 0 and fallback == 0):
        return CoverageStatus.UNCOVERED
    if exact == base:
        return CoverageStatus.EXACT_COVERED
    if fallback == base:
        return CoverageStatus.FALLBACK_ONLY
    if sum(value > 0 for value in (exact, fallback, uncovered)) > 1:
        return CoverageStatus.MIXED
    return CoverageStatus.UNCOVERED
