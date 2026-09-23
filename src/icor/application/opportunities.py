"""Opportunity aggregation, coverage resolution, scoring, and drill-down."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from math import ceil
from typing import Protocol

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
from icor.domain.snapshots import SnapshotVersions

_CURRENT_EUROPE_MARKETS = frozenset(
    {
        "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "EU27",
        "FI", "FR", "GB", "GR", "HR", "HU", "IE", "IS", "IT", "LI",
        "LT", "LU", "LV", "MT", "NL", "NO", "PL", "PT", "RO", "SE",
        "SI", "SK", "UK", "Europe",
    }
)
_CANONICAL_WORLD_REGIONS = frozenset(
    {
        "Africa",
        "Asia-Pacific",
        "Europe",
        "Latin America & Caribbean",
        "Middle East",
        "North America",
    }
)


def world_region_for_market(market: str) -> str:
    if market in _CURRENT_EUROPE_MARKETS:
        return "Europe"
    if market in _CANONICAL_WORLD_REGIONS:
        return market
    return "Other / unclassified"


@dataclass(frozen=True, slots=True)
class OpportunityContribution:
    configuration_id: str
    market: str
    forecast_horizon: int
    generation: str
    body_style: str
    demand: DemandRange


class OpportunityRepository(Protocol):
    def search(self, query: OpportunityQuery) -> OpportunityPage: ...

    def get(self, group_id: str, query: OpportunityQuery) -> OpportunityRow | None: ...

    def fleet_estimates(
        self, group_id: str, query: OpportunityQuery
    ) -> tuple[OpportunityFleetEstimate, ...]: ...

    def contributions(
        self, group_id: str, query: OpportunityQuery
    ) -> tuple[OpportunityContribution, ...]: ...

    def drill_down(
        self,
        group_id: str,
        query: OpportunityQuery,
        page: int,
        page_size: int,
    ) -> tuple[OpportunityDrillDownRow, ...]: ...


# A search box, not a query language: long input buys nothing and is a way to
# make the database do unbounded work on an unauthenticated route.
_MAX_TEXT_LENGTH = 64


class OpportunityGroupBy(StrEnum):
    BRAND = "brand"
    MODEL = "model"
    MODEL_YEAR = "model_year"


class OpportunitySort(StrEnum):
    """The orders a ranking may be read in.

    Score is the product's own ranking and stays the default. Demand answers
    "which is the biggest", which the score deliberately does not, because it
    mixes demand with readiness. Vehicle is for finding a known car rather than
    discovering one.
    """

    SCORE = "score"
    DEMAND = "demand"
    VEHICLE = "vehicle"


@dataclass(frozen=True, slots=True)
class OpportunityQuery:
    group_by: OpportunityGroupBy
    markets: tuple[str, ...] = ()
    horizons: tuple[int, ...] = ()
    text: str = ""
    sort: OpportunitySort = OpportunitySort.SCORE
    page: int = 1
    page_size: int = 25

    def __post_init__(self) -> None:
        if self.page < 1 or not 1 <= self.page_size <= 100:
            raise ValueError("opportunity pagination is invalid")
        if len(self.text) > _MAX_TEXT_LENGTH:
            raise ValueError("opportunity search text is too long")


@dataclass(frozen=True, slots=True)
class OpportunitySummary:
    base_units: int
    exact_covered_base_units: int
    high_demand_uncovered_base_units: int


@dataclass(frozen=True, slots=True)
class OpportunityFleetEstimate:
    world_region: str
    forecast_horizon: int
    estimated_fleet_units: int

    def __post_init__(self) -> None:
        if not self.world_region.strip():
            raise ValueError("world region is required")
        if type(self.forecast_horizon) is not int:
            raise ValueError("forecast horizon must be an integer")
        if (
            type(self.estimated_fleet_units) is not int
            or self.estimated_fleet_units < 0
        ):
            raise ValueError("estimated fleet must use non-negative integer units")


@dataclass(frozen=True, slots=True)
class OpportunityRow:
    group_id: str
    group_by: OpportunityGroupBy
    brand: str
    model: str | None
    model_year: int | None
    generation_name: str | None
    generation_basis: str | None
    generation_source_url: str | None
    icor_worked_base_units: int
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
    snapshot_id: str | None = None
    versions: SnapshotVersions | None = None
    total: int = 0
    page: int = 1
    page_size: int = 25
    pages: int = 0
    # Every market and horizon this snapshot holds, so a filter can offer what
    # exists instead of a list written by hand that drifts from the data.
    available_markets: tuple[str, ...] = ()
    available_horizons: tuple[int, ...] = ()


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


def _matching_text(
    rows: tuple[OpportunityRow, ...], text: str
) -> tuple[OpportunityRow, ...]:
    """Narrow a ranking to the vehicles whose visible labels contain the text."""

    needle = text.strip().casefold()
    if not needle:
        return rows
    return tuple(
        row
        for row in rows
        if needle in row.brand.casefold()
        or (row.model is not None and needle in row.model.casefold())
    )


def _sorted_rows(
    rows: tuple[OpportunityRow, ...], sort: OpportunitySort
) -> tuple[OpportunityRow, ...]:
    """Order a ranking, ending every order on the same total tie-break.

    A partial order would let one row appear on two pages while another appeared
    on none, because the page boundary would fall in an arbitrary place.
    """

    if sort is OpportunitySort.DEMAND:
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    -row.demand.base_units,
                    -row.score.total_points,
                    row.group_id,
                ),
            )
        )
    if sort is OpportunitySort.VEHICLE:
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.brand.casefold(),
                    (row.model or "").casefold(),
                    row.model_year if row.model_year is not None else 0,
                    row.group_id,
                ),
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                -row.score.total_points,
                -row.demand.base_units,
                row.group_id,
            ),
        )
    )


class OpportunityService:
    def __init__(
        self,
        planner_repository: PlannerRepository | None = None,
        coverage_repository: CoverageRepository | None = None,
        ranking_strategy: RankingStrategy | None = None,
        *,
        repository: OpportunityRepository | None = None,
    ) -> None:
        self._planner_repository = planner_repository
        self._coverage_repository = coverage_repository
        self._ranking_strategy = ranking_strategy
        self._repository = repository

    def list(self, query: OpportunityQuery) -> OpportunityPage:
        if self._repository is not None:
            return self._repository.search(query)
        assert self._ranking_strategy is not None
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
        rows = _matching_text(rows, query.text)
        rows = _sorted_rows(rows, query.sort)
        total = len(rows)
        markets, horizons = self._available_facets()
        start = (query.page - 1) * query.page_size
        return OpportunityPage(
            items=rows[start : start + query.page_size],
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
            snapshot_id=getattr(self._planner_repository, "snapshot_id", None),
            versions=getattr(self._planner_repository, "versions", None),
            total=total,
            page=query.page,
            page_size=query.page_size,
            pages=ceil(total / query.page_size),
            available_markets=markets,
            available_horizons=horizons,
        )

    def _available_facets(self) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Every market and horizon the source holds, not just this page's."""

        assert self._planner_repository is not None
        configurations = self._planner_repository.list_all()
        return (
            tuple(sorted({row.market for row in configurations})),
            tuple(sorted({row.forecast_horizon for row in configurations})),
        )

    def get(self, group_id: str, query: OpportunityQuery) -> OpportunityRow | None:
        if self._repository is not None:
            return self._repository.get(group_id, query)
        assert self._ranking_strategy is not None
        atoms, _warnings = self._resolved_atoms(query)
        grouped = self._group(atoms, query.group_by)
        selected = grouped.get(group_id)
        if selected is None:
            return None
        candidates = tuple(
            OpportunityCandidate(
                group_id=identity,
                demand=_sum_demand(group_atoms),
                exact_covered_base_units=_sum_coverage(group_atoms, CoverageStatus.EXACT_COVERED),
                fallback_covered_base_units=_sum_coverage(
                    group_atoms, CoverageStatus.FALLBACK_ONLY
                ),
                uncovered_base_units=_sum_coverage(
                    group_atoms, CoverageStatus.UNCOVERED
                ),
            )
            for identity, group_atoms in grouped.items()
        )
        scores = {score.group_id: score for score in self._ranking_strategy.score(candidates)}
        return self._row(group_id, selected, query.group_by, scores[group_id])

    def drill_down(
        self,
        group_id: str,
        query: OpportunityQuery,
        page: int = 1,
        page_size: int = 100,
    ) -> tuple[OpportunityDrillDownRow, ...]:
        if self._repository is not None:
            return self._repository.drill_down(
                group_id, query, page, page_size
            )
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

    def fleet_estimates(
        self, group_id: str, query: OpportunityQuery
    ) -> tuple[OpportunityFleetEstimate, ...]:
        if self._repository is not None:
            return self._repository.fleet_estimates(group_id, query)
        atoms, _warnings = self._resolved_atoms(query)
        selected = self._group(atoms, query.group_by).get(group_id, ())
        by_configuration: dict[str, list[_DemandAtom]] = {}
        for atom in selected:
            by_configuration.setdefault(atom.configuration.configuration_id, []).append(
                atom
            )
        totals: dict[tuple[str, int], int] = {}
        for configuration_atoms in by_configuration.values():
            configuration = configuration_atoms[0].configuration
            selected_demand = sum(
                atom.model_year_demand.demand.base_units
                for atom in configuration_atoms
            )
            fleet_units = (
                round(
                    configuration.vehicle_exposure_units
                    * selected_demand
                    / configuration.demand.base_units
                )
                if configuration.demand.base_units
                else 0
            )
            key = (
                world_region_for_market(configuration.market),
                configuration.forecast_horizon,
            )
            totals[key] = totals.get(key, 0) + fleet_units
        return tuple(
            OpportunityFleetEstimate(region, horizon, units)
            for (region, horizon), units in sorted(
                totals.items(), key=lambda item: (item[0][1], item[0][0])
            )
        )

    def contributions(
        self, group_id: str, query: OpportunityQuery
    ) -> tuple[OpportunityContribution, ...]:
        if self._repository is not None:
            return self._repository.contributions(group_id, query)
        atoms, _warnings = self._resolved_atoms(query)
        selected = self._group(atoms, query.group_by).get(group_id, ())
        by_configuration: dict[str, list[_DemandAtom]] = {}
        for atom in selected:
            by_configuration.setdefault(atom.configuration.configuration_id, []).append(
                atom
            )
        return tuple(
            OpportunityContribution(
                configuration_id=configuration_id,
                market=configuration_atoms[0].configuration.market,
                forecast_horizon=configuration_atoms[0].configuration.forecast_horizon,
                generation=configuration_atoms[0].configuration.generation,
                body_style=configuration_atoms[0].configuration.body_style,
                demand=_sum_demand(tuple(configuration_atoms)),
            )
            for configuration_id, configuration_atoms in sorted(
                by_configuration.items()
            )
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
            generation_name=(
                first.configuration.generation
                if group_by is OpportunityGroupBy.MODEL_YEAR
                else None
            ),
            generation_basis=(
                "planning_configuration"
                if group_by is OpportunityGroupBy.MODEL_YEAR
                else None
            ),
            generation_source_url=None,
            icor_worked_base_units=0,
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
