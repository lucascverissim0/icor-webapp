"""Opportunity aggregation, coverage resolution, scoring, and drill-down."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from math import ceil
from typing import Protocol

from icor.application.coverage import CoverageRepository
from icor.application.planner import PlannerRepository
from icor.application.ranking import (
    WHOLE_MARKET_BASIS,
    DemandPopulation,
    RankingStrategy,
)
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
from icor.eurocodes.annotation import (
    EurocodeBreakdown,
    EurocodeCell,
    EurocodeSearch,
    eurocode_breakdown,
    eurocode_search,
)
from icor.eurocodes.grouping import (
    EurocodeGroup,
    EurocodeRanking,
    EurocodeTotals,
    RankedCell,
    rank_by_eurocode,
)
from icor.eurocodes.runtime import EurocodeMap

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
    #: One row per Eurocode, ranked on the units it certainly carries.
    EUROCODE = "eurocode"


class EurocodeGroupingUnavailable(ValueError):
    """The ranking cannot be read by Eurocode: no reviewed map is loaded."""


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
    #: None shows every vehicle; True or False keeps only the vehicles ICOR has,
    #: or has not, worked on. It narrows the rows and never moves a score.
    icor_worked: bool | None = None

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
    # Which Eurocodes the row's registrations may carry. None when Eurocodes
    # are not loaded, and for brand rows, which span too many codes to list.
    eurocodes: EurocodeBreakdown | None = None
    # On a Eurocode row: the code, its vehicles and the units it may share. The
    # row's demand is the code's certain units alone.
    eurocode: EurocodeGroup | None = None

    @property
    def icor_worked(self) -> bool:
        return (
            self.exact_covered_base_units > 0
            or self.fallback_covered_base_units > 0
            or self.icor_worked_base_units > 0
        )


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
    # The population every score on this page was measured against. Reported on
    # the page and not only on a row because an empty result — the moment the
    # claim matters most — has no row to read it from.
    demand_population: int = 0
    demand_basis: str = WHOLE_MARKET_BASIS
    # Whether rows carry Eurocodes at all, so "no codes" is never mistaken for
    # "none assigned".
    eurocodes_loaded: bool = False
    # The code reading of the search text, when the text is exactly a Eurocode:
    # rows that may carry it are listed beside rows whose name contains it.
    eurocode_search: EurocodeSearch | None = None
    # On a Eurocode ranking: the view's units by outcome, shared units counted
    # once, so the rows' certain units and these account for the whole view.
    eurocode_totals: EurocodeBreakdown | None = None


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


def _candidates(
    grouped: dict[str, tuple[_DemandAtom, ...]],
) -> tuple[OpportunityCandidate, ...]:
    return tuple(
        OpportunityCandidate(
            group_id=group_id,
            demand=_sum_demand(group_atoms),
            exact_covered_base_units=_sum_coverage(
                group_atoms, CoverageStatus.EXACT_COVERED
            ),
            fallback_covered_base_units=_sum_coverage(
                group_atoms, CoverageStatus.FALLBACK_ONLY
            ),
            uncovered_base_units=_sum_coverage(group_atoms, CoverageStatus.UNCOVERED),
        )
        for group_id, group_atoms in grouped.items()
    )


def _demand_population(population: tuple[OpportunityCandidate, ...]) -> int:
    """How many vehicles a percentile is measured against, for an empty page."""

    return DemandPopulation.of(
        candidate.demand.base_units for candidate in population
    ).size


def _matching_text(
    rows: tuple[OpportunityRow, ...],
    text: str,
    carrying: frozenset[str] = frozenset(),
) -> tuple[OpportunityRow, ...]:
    """Narrow a ranking to the vehicles whose visible labels contain the text.

    ``carrying`` names the groups that may carry the Eurocode the text names;
    they are kept beside the name matches.
    """

    needle = text.strip().casefold()
    if not needle:
        return rows
    return tuple(
        row
        for row in rows
        if needle in row.brand.casefold()
        or (row.model is not None and needle in row.model.casefold())
        or row.group_id in carrying
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


def eurocode_rows(
    ranking: EurocodeRanking,
    population: EurocodeRanking,
    strategy: RankingStrategy,
    *,
    group_id: Callable[[str], str],
    evidence_status: EvidenceStatus,
    data_version: str,
) -> tuple[OpportunityRow, ...]:
    """Score each code's certain units against the whole market's codes.

    ``population`` is the same ranking with no market or horizon filter, so a
    filter changes which units a row shows, never what its score means. The
    score is the ranking strategy's own, as for every other grouping.
    """

    def candidate(code: str, totals: EurocodeTotals) -> OpportunityCandidate:
        return OpportunityCandidate(
            group_id=group_id(code),
            demand=totals.demand,
            exact_covered_base_units=totals.exact_units,
            fallback_covered_base_units=totals.fallback_units,
            uncovered_base_units=totals.uncovered_units,
        )

    scores = {
        score.group_id: score
        for score in strategy.score(
            tuple(candidate(code, totals) for code, totals in ranking.groups.items()),
            population=tuple(
                candidate(code, totals) for code, totals in population.groups.items()
            ),
        )
    }
    return tuple(
        OpportunityRow(
            group_id=group_id(code),
            group_by=OpportunityGroupBy.EUROCODE,
            brand=totals.group.make,
            model=totals.group.label,
            model_year=None,
            generation_name=None,
            generation_basis=None,
            generation_source_url=None,
            icor_worked_base_units=totals.icor_worked_units,
            demand=totals.demand,
            contributing_configuration_count=totals.configuration_count,
            exact_covered_base_units=totals.exact_units,
            fallback_covered_base_units=totals.fallback_units,
            uncovered_base_units=totals.uncovered_units,
            coverage_status=_coverage_status(
                totals.demand.base_units,
                totals.exact_units,
                totals.fallback_units,
                totals.uncovered_units,
            ),
            score=scores[group_id(code)],
            evidence_status=evidence_status,
            data_version=data_version,
            eurocode=totals.group,
        )
        for code, totals in ranking.groups.items()
    )


def matching_eurocode_text(
    rows: tuple[OpportunityRow, ...], text: str
) -> tuple[OpportunityRow, ...]:
    """Narrow Eurocode rows to those whose code, name or vehicles contain the text."""

    needle = text.strip().casefold()
    if not needle:
        return rows
    return tuple(
        row
        for row in rows
        if row.eurocode is not None
        and (
            needle in row.eurocode.code.casefold()
            or needle in row.eurocode.label.casefold()
            or needle in row.eurocode.make.casefold()
            or any(
                needle in f"{vehicle.brand} {vehicle.model}".casefold()
                for vehicle in row.eurocode.vehicles
            )
        )
    )


def eurocode_contributions(totals: EurocodeTotals) -> tuple[OpportunityContribution, ...]:
    """A code's certain units by market and horizon."""

    return tuple(
        OpportunityContribution(
            configuration_id=f"{totals.group.code}|{market}|{horizon}",
            market=market,
            forecast_horizon=horizon,
            generation=totals.group.label,
            body_style="Every body the code fits",
            demand=demand,
        )
        for market, horizon, demand in totals.contributions
    )


def eurocode_page(
    rows: tuple[OpportunityRow, ...],
    query: OpportunityQuery,
    ranking: EurocodeRanking,
    population: EurocodeRanking,
    **page: object,
) -> OpportunityPage:
    """Filter, order and paginate Eurocode rows into a page, as the other groupings are."""

    rows = matching_eurocode_text(rows, query.text)
    if query.icor_worked is not None:
        rows = tuple(row for row in rows if row.icor_worked is query.icor_worked)
    rows = _sorted_rows(rows, query.sort)
    start = (query.page - 1) * query.page_size
    return OpportunityPage(
        items=rows[start : start + query.page_size],
        summary=OpportunitySummary(
            base_units=sum(row.demand.base_units for row in rows),
            exact_covered_base_units=sum(row.exact_covered_base_units for row in rows),
            high_demand_uncovered_base_units=sum(
                row.uncovered_base_units
                for row in rows
                if row.score.demand_percentile >= 0.75
            ),
        ),
        total=len(rows),
        page=query.page,
        page_size=query.page_size,
        pages=ceil(len(rows) / query.page_size),
        demand_population=DemandPopulation.of(
            totals.demand.base_units for totals in population.groups.values()
        ).size,
        demand_basis=WHOLE_MARKET_BASIS,
        eurocodes_loaded=True,
        eurocode_totals=ranking.totals,
        **page,  # type: ignore[arg-type]
    )


class OpportunityService:
    def __init__(
        self,
        planner_repository: PlannerRepository | None = None,
        coverage_repository: CoverageRepository | None = None,
        ranking_strategy: RankingStrategy | None = None,
        *,
        repository: OpportunityRepository | None = None,
        eurocodes: EurocodeMap | None = None,
    ) -> None:
        self._planner_repository = planner_repository
        self._coverage_repository = coverage_repository
        self._ranking_strategy = ranking_strategy
        self._repository = repository
        self._eurocodes = eurocodes

    def list(self, query: OpportunityQuery) -> OpportunityPage:
        if self._repository is not None:
            return self._repository.search(query)
        assert self._ranking_strategy is not None
        if query.group_by is OpportunityGroupBy.EUROCODE:
            ranking, population, warnings, atoms = self._eurocode_rankings(query)
            markets, horizons = self._available_facets()
            return eurocode_page(
                self._eurocode_rows(ranking, population, atoms),
                query,
                ranking,
                population,
                strategy_name=self._ranking_strategy.name,
                strategy_version=self._ranking_strategy.version,
                integrity_warnings=warnings,
                snapshot_id=getattr(self._planner_repository, "snapshot_id", None),
                versions=getattr(self._planner_repository, "versions", None),
                available_markets=markets,
                available_horizons=horizons,
            )
        atoms, warnings = self._resolved_atoms(query)
        grouped = self._group(atoms, query.group_by)
        candidates = _candidates(grouped)
        population = self._population(query)
        scores = {
            score.group_id: score
            for score in self._ranking_strategy.score(
                candidates, population=population
            )
        }
        rows = tuple(
            self._row(group_id, group_atoms, query.group_by, scores[group_id])
            for group_id, group_atoms in grouped.items()
        )
        reading = eurocode_search(self._eurocodes, query.text)
        carrying = frozenset(
            group_id
            for group_id, group_atoms in grouped.items()
            if reading is not None
            and reading.label is not None
            and any(share.code == reading.code for share in self._breakdown(group_atoms).codes)
        )
        rows = _matching_text(rows, query.text, carrying)
        if query.icor_worked is not None:
            rows = tuple(row for row in rows if row.icor_worked is query.icor_worked)
        rows = _sorted_rows(rows, query.sort)
        total = len(rows)
        markets, horizons = self._available_facets()
        start = (query.page - 1) * query.page_size
        return OpportunityPage(
            items=tuple(
                self._with_eurocodes(row, grouped[row.group_id], query.group_by)
                for row in rows[start : start + query.page_size]
            ),
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
            demand_population=(
                scores[rows[0].group_id].demand_population
                if rows
                else _demand_population(population)
            ),
            demand_basis=WHOLE_MARKET_BASIS,
            eurocodes_loaded=self._eurocodes is not None,
            eurocode_search=reading,
        )

    def _with_eurocodes(
        self,
        row: OpportunityRow,
        atoms: tuple[_DemandAtom, ...],
        group_by: OpportunityGroupBy,
    ) -> OpportunityRow:
        """The row with the Eurocodes its registrations may carry.

        This source has no identity index: its labels are already canonical,
        so a vehicle is keyed by its case-folded make and model.
        """

        if self._eurocodes is None or group_by is OpportunityGroupBy.BRAND:
            return row
        return replace(row, eurocodes=self._breakdown(atoms))

    def _breakdown(self, atoms: tuple[_DemandAtom, ...]) -> EurocodeBreakdown:
        assert self._eurocodes is not None, "only a loaded map breaks units down"
        cells = (
            EurocodeCell(
                identity=" ".join(
                    f"{atom.configuration.brand}|{atom.configuration.model}".split()
                ).casefold(),
                year=atom.model_year_demand.model_year,
                geography=atom.configuration.market,
                raw_model=atom.configuration.model,
                units=atom.model_year_demand.demand.base_units,
            )
            for atom in atoms
        )
        return eurocode_breakdown(self._eurocodes, cells)

    def _eurocode_rankings(
        self, query: OpportunityQuery
    ) -> tuple[
        EurocodeRanking, EurocodeRanking, tuple[str, ...], tuple[_DemandAtom, ...]
    ]:
        """This view's Eurocode ranking, the whole market's, the warnings and the atoms."""

        if self._eurocodes is None:
            raise EurocodeGroupingUnavailable("Eurocodes are not loaded")
        atoms, warnings = self._resolved_atoms(query)
        whole_market, _warnings = self._resolved_atoms(replace(query, markets=(), horizons=()))
        return (
            rank_by_eurocode(self._eurocodes, self._ranked_cells(atoms)),
            rank_by_eurocode(self._eurocodes, self._ranked_cells(whole_market)),
            warnings,
            atoms,
        )

    @staticmethod
    def _ranked_cells(atoms: tuple[_DemandAtom, ...]) -> tuple[RankedCell, ...]:
        return tuple(
            RankedCell(
                identity=" ".join(
                    f"{atom.configuration.brand}|{atom.configuration.model}".split()
                ).casefold(),
                year=atom.model_year_demand.model_year,
                geography=atom.configuration.market,
                raw_model=atom.configuration.model,
                brand=atom.configuration.brand,
                model=atom.configuration.model,
                horizon=atom.configuration.forecast_horizon,
                configuration=atom.configuration.configuration_id,
                demand=atom.model_year_demand.demand,
                coverage_status=atom.coverage_status,
                icor_worked=False,
            )
            for atom in atoms
        )

    def _eurocode_rows(
        self,
        ranking: EurocodeRanking,
        population: EurocodeRanking,
        atoms: tuple[_DemandAtom, ...],
    ) -> tuple[OpportunityRow, ...]:
        """Rows carrying the weakest evidence status in the view, as no row is stronger."""

        assert self._ranking_strategy is not None
        statuses = list(EvidenceStatus)
        return eurocode_rows(
            ranking,
            population,
            self._ranking_strategy,
            group_id=lambda code: _group_id(OpportunityGroupBy.EUROCODE, (code,)),
            evidence_status=min(
                (atom.model_year_demand.evidence_status for atom in atoms),
                key=statuses.index,
                default=EvidenceStatus.DEMONSTRATION,
            ),
            data_version=atoms[0].model_year_demand.data_version if atoms else "",
        )

    def _population(
        self, query: OpportunityQuery
    ) -> tuple[OpportunityCandidate, ...]:
        """Every vehicle in the source at this grouping level, unfiltered.

        A market or horizon chosen in the interface says which figures to show,
        not which vehicles a score may be compared with, so the population is
        read with those filters removed.
        """

        whole_market = replace(query, markets=(), horizons=())
        atoms, _warnings = self._resolved_atoms(whole_market)
        return _candidates(self._group(atoms, query.group_by))

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
        if query.group_by is OpportunityGroupBy.EUROCODE:
            ranking, population, _warnings, atoms = self._eurocode_rankings(query)
            return next(
                (
                    row
                    for row in self._eurocode_rows(ranking, population, atoms)
                    if row.group_id == group_id
                ),
                None,
            )
        atoms, _warnings = self._resolved_atoms(query)
        grouped = self._group(atoms, query.group_by)
        selected = grouped.get(group_id)
        if selected is None:
            return None
        candidates = _candidates(grouped)
        scores = {
            score.group_id: score
            for score in self._ranking_strategy.score(
                candidates, population=self._population(query)
            )
        }
        return self._with_eurocodes(
            self._row(group_id, selected, query.group_by, scores[group_id]),
            selected,
            query.group_by,
        )

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
        if query.group_by is OpportunityGroupBy.EUROCODE:
            # A code's units are allocated per registration, not per configuration.
            return ()
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
        if query.group_by is OpportunityGroupBy.EUROCODE:
            # A surviving fleet is not allocated to codes yet.
            return ()
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
        if query.group_by is OpportunityGroupBy.EUROCODE:
            ranking, _population, _warnings, _atoms = self._eurocode_rankings(query)
            return next(
                (
                    eurocode_contributions(totals)
                    for code, totals in ranking.groups.items()
                    if _group_id(OpportunityGroupBy.EUROCODE, (code,)) == group_id
                ),
                (),
            )
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
