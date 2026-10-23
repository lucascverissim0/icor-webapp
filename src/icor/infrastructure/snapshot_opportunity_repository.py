"""Bounded SQLite queries for opportunity ranking and drill-down."""

from __future__ import annotations

import base64
import contextlib
import json
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_UP, Decimal
from math import ceil
from pathlib import Path

from icor.application.opportunities import (
    EurocodeGroupingUnavailable,
    OpportunityContribution,
    OpportunityDrillDownRow,
    OpportunityFleetEstimate,
    OpportunityGroupBy,
    OpportunityPage,
    OpportunityQuery,
    OpportunityRow,
    OpportunitySort,
    OpportunitySummary,
    _coverage_status,
    eurocode_contributions,
    eurocode_page,
    eurocode_rows,
    world_region_for_market,
)
from icor.application.ranking import (
    DEMAND_POINTS_MAX,
    RankingStrategy,
    demand_explanation,
)
from icor.application.worked_models import IcorWorkedModelCatalog
from icor.domain.evidence import COHORT_EVIDENCE_KINDS, CanonicalVehicle
from icor.domain.opportunities import CoverageStatus, OpportunityScore
from icor.domain.planner import DemandRange, EvidenceStatus
from icor.eurocodes.annotation import (
    EurocodeBreakdown,
    EurocodeCell,
    eurocode_breakdown,
    eurocode_search,
)
from icor.eurocodes.grouping import EurocodeRanking, RankedCell, rank_by_eurocode
from icor.eurocodes.runtime import EurocodeMap, identity_key
from icor.evidence.normalization import source_vehicle_display_label
from icor.evidence.vehicle_identity import VehicleIdentityIndex
from icor.generations.public_catalog import ranking_public_generation_catalog
from icor.infrastructure.snapshot_identity import (
    load_identity_index,
    materialize_identity_table,
)
from icor.infrastructure.snapshot_opportunity_universe import (
    TABLE as _UNIVERSE_TABLE,
)
from icor.infrastructure.snapshot_opportunity_universe import (
    OpportunityUniverse,
)
from icor.infrastructure.snapshot_planner_repository import (
    SnapshotPlannerRepository,
    _cohort_attribution_ctes,
)
from icor.infrastructure.sqlite_coverage_repository import SQLiteCoverageRepository

_GENERATION_REGISTRY = "public-generation-registry-v2"
#: Row breakdowns kept per grouping and market/horizon filter; a page holds 100 at most.
_EUROCODE_CACHE_SIZE = 4096
#: Eurocode rankings kept, one per market/horizon view; each takes seconds to build.
_EUROCODE_VIEW_CACHE_SIZE = 16
#: The evidence kinds a Eurocode row reports, by the index ``temp.cohort_kind`` stores.
_EVIDENCE_KINDS = ("stock", "registrations", "interpolated", "forecast")


@dataclass(frozen=True, slots=True)
class _Filter:
    """A query's predicates, split by what each part of a statement may name."""

    #: Predicates naming only the opportunity estimate, safe anywhere.
    estimates: str
    estimate_parameters: tuple[object, ...]
    #: Every predicate, for the one place the canonical identity is in scope.
    atoms: str
    atom_parameters: tuple[object, ...]


#: No filter at all: the population a whole-market score is measured against.
_WHOLE_MARKET = _Filter("1 = 1", (), "1 = 1", ())


_SORT_ORDERS: dict[OpportunitySort, str] = {
    # Every order ends in the same tie-break so a page boundary cannot show one
    # row twice and hide another.
    OpportunitySort.SCORE: (
        "total_points DESC, base_units DESC, brand, model, model_year"
    ),
    OpportunitySort.DEMAND: (
        "base_units DESC, total_points DESC, brand, model, model_year"
    ),
    OpportunitySort.VEHICLE: (
        "brand COLLATE NOCASE, model COLLATE NOCASE, model_year, total_points DESC"
    ),
}


class SnapshotOpportunityRepository:
    """Rank snapshot opportunities without materializing planner configurations."""

    def __init__(
        self,
        planner: SnapshotPlannerRepository,
        coverage: SQLiteCoverageRepository,
        strategy: RankingStrategy,
        *,
        worked_models: IcorWorkedModelCatalog | None = None,
        verified_only: bool = False,
        model_year_catalog: bool = False,
        eurocodes: EurocodeMap | None = None,
    ) -> None:
        path = getattr(planner._ledger, "path", None)
        if not isinstance(path, Path):
            raise ValueError("snapshot opportunity repository requires SQLite")
        self._snapshot_path = path
        self._planner = planner
        self._coverage_path = coverage.path
        self._strategy = strategy
        self._worked_models = worked_models or IcorWorkedModelCatalog.empty()
        self._generation_catalog = ranking_public_generation_catalog()
        self._verified_only = verified_only
        self._model_year_catalog = model_year_catalog
        self._eurocodes = eurocodes
        self._eurocode_cache: dict[tuple[object, ...], EurocodeBreakdown] = {}
        self._identity_keys: dict[str, str] | None = None
        self._eurocode_views: dict[
            tuple[object, ...],
            tuple[EurocodeRanking, tuple[str, ...]],
        ] = {}
        self._eurocode_lock = threading.Lock()
        self.snapshot_id = planner.snapshot_id
        self.versions = planner.versions
        self._uncovered_cache: dict[OpportunityQuery, OpportunityPage] = {}
        self._identity_index: VehicleIdentityIndex | None = None
        self._facets: tuple[tuple[str, ...], tuple[int, ...]] | None = None
        # The snapshot is immutable and opened read-only, so a population built
        # from it never goes stale. Production coverage is the one mutable
        # input, and it is not part of any population; the fingerprint below
        # notices it moving for the cached pages that do carry it.
        self._universes: dict[OpportunityGroupBy, OpportunityUniverse] = {}
        self._coverage_fingerprint: tuple[object, ...] | None = None
        with sqlite3.connect(
            f"{self._snapshot_path.resolve().as_uri()}?mode=ro", uri=True
        ) as connection:
            self._materialized_attribution = (
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'opportunity_cohort_attribution'"
                ).fetchone()
                is not None
            )

    def search(self, query: OpportunityQuery) -> OpportunityPage:
        if query.group_by is OpportunityGroupBy.EUROCODE:
            return self._search_by_eurocode(query)
        offset = (query.page - 1) * query.page_size
        has_manual_coverage = self._refresh_for_coverage()
        if not has_manual_coverage:
            cached = self._uncovered_cache.get(query)
            if cached is not None:
                return cached
        include_coverage = has_manual_coverage or bool(self._worked_models.records)
        universe = self._universe(query.group_by)
        with self._connect(universe) as connection:
            reading = (
                None
                if self._verified_only
                else eurocode_search(self._eurocodes, query.text)
            )
            code_groups = (
                self._groups_carrying(connection, query, reading.code)
                if reading is not None and reading.label is not None
                else ()
            )
            cte, parameters = self._scored_cte(
                query, include_coverage=include_coverage, code_groups=code_groups
            )
            rows = connection.execute(
                f"""{cte}
                SELECT *, COUNT(*) OVER () summary_total,
                    SUM(base_units) OVER () summary_base_units,
                    SUM(exact_units) OVER () summary_exact_units,
                    SUM(CASE WHEN demand_percentile >= 0.75
                        THEN uncovered_units ELSE 0 END) OVER () summary_high_uncovered,
                    SUM(universe_missing) OVER () summary_universe_missing
                FROM scored
                ORDER BY {_SORT_ORDERS[query.sort]}
                LIMIT ? OFFSET ?""",
                (*parameters, query.page_size, offset),
            ).fetchall()
            if rows:
                summary = rows[0]
            else:
                summary = connection.execute(
                    f"""{cte}
                    SELECT COUNT(*) summary_total,
                        COALESCE(SUM(base_units), 0) summary_base_units,
                        COALESCE(SUM(exact_units), 0) summary_exact_units,
                        COALESCE(SUM(CASE WHEN demand_percentile >= 0.75
                            THEN uncovered_units ELSE 0 END), 0)
                            summary_high_uncovered,
                        COALESCE(SUM(universe_missing), 0)
                            summary_universe_missing
                    FROM scored""",
                    parameters,
                ).fetchone()
            warnings = self._integrity_warnings(connection)
            unranked = int(summary["summary_universe_missing"])
            if unranked:
                # Filtering can only remove atoms, so every group on screen
                # should also be in the unfiltered population. If one is not,
                # the population is stale rather than the row being wrong, and
                # saying so beats serving a silent zero.
                warnings = (
                    *warnings,
                    f"{unranked} groups are missing from the ranked market "
                    "population and are reported unscored.",
                )
            markets, horizons = self._available_facets(connection)
            breakdowns = self._eurocode_breakdowns(connection, query, rows)
        items = tuple(
            self._row(row, query.group_by, universe, breakdowns.get(_row_key(row)))
            for row in rows
        )
        total = int(summary["summary_total"])
        result = OpportunityPage(
            items=items,
            summary=OpportunitySummary(
                base_units=int(summary["summary_base_units"]),
                exact_covered_base_units=int(summary["summary_exact_units"]),
                high_demand_uncovered_base_units=int(
                    summary["summary_high_uncovered"]
                ),
            ),
            strategy_name=self._strategy.name,
            strategy_version=self._strategy.version,
            integrity_warnings=warnings,
            snapshot_id=self.snapshot_id,
            versions=self.versions,
            total=total,
            page=query.page,
            page_size=query.page_size,
            pages=ceil(total / query.page_size),
            available_markets=markets,
            available_horizons=horizons,
            demand_population=universe.population,
            demand_basis=universe.basis,
            eurocodes_loaded=self._eurocodes is not None,
            eurocode_search=reading,
        )
        if not has_manual_coverage:
            if len(self._uncovered_cache) >= 128:
                self._uncovered_cache.pop(next(iter(self._uncovered_cache)))
            self._uncovered_cache[query] = result
        return result

    def get(self, group_id: str, query: OpportunityQuery) -> OpportunityRow | None:
        identity = _decode_group_id(group_id, query.group_by)
        if identity is None:
            return None
        if query.group_by is OpportunityGroupBy.EUROCODE:
            ranking, population, _warnings = self._eurocode_view(query)
            return next(
                (
                    row
                    for row in self._eurocode_rows(ranking, population)
                    if row.group_id == group_id
                ),
                None,
            )
        has_coverage = self._refresh_for_coverage()
        if not has_coverage:
            for cached_query, cached_page in reversed(
                tuple(self._uncovered_cache.items())
            ):
                if (
                    cached_query.group_by is query.group_by
                    and cached_query.markets == query.markets
                    and cached_query.horizons == query.horizons
                ):
                    cached_row = next(
                        (
                            item
                            for item in cached_page.items
                            if item.group_id == group_id
                        ),
                        None,
                    )
                    if cached_row is not None:
                        return cached_row
        include_coverage = has_coverage or bool(self._worked_models.records)
        universe = self._universe(query.group_by)
        with self._connect(universe) as connection:
            cte, parameters = self._scored_cte(
                query, include_coverage=include_coverage
            )
            clauses = ["brand = ?"]
            identity_parameters: list[object] = [identity[0]]
            if query.group_by is not OpportunityGroupBy.BRAND:
                clauses.append("model = ?")
                identity_parameters.append(identity[1])
            if query.group_by is OpportunityGroupBy.MODEL_YEAR:
                clauses.append("model_year = ?")
                identity_parameters.append(identity[2])
            row = connection.execute(
                f"{cte} SELECT * FROM scored WHERE {' AND '.join(clauses)}",
                (*parameters, *identity_parameters),
            ).fetchone()
            if row is None:
                return None
            breakdowns = self._eurocode_breakdowns(connection, query, (row,))
        return self._row(row, query.group_by, universe, breakdowns.get(_row_key(row)))

    def contributions(
        self, group_id: str, query: OpportunityQuery
    ) -> tuple[OpportunityContribution, ...]:
        identity = _decode_group_id(group_id, query.group_by)
        if identity is None:
            return ()
        if query.group_by is OpportunityGroupBy.EUROCODE:
            ranking, _population, _warnings = self._eurocode_view(query)
            totals = ranking.groups.get(identity[0])
            return () if totals is None else eurocode_contributions(totals)
        clauses = ["ident.brand = ?"]
        parameters: list[object] = [identity[0]]
        if query.group_by is not OpportunityGroupBy.BRAND:
            clauses.append("ident.model = ?")
            parameters.append(identity[1])
        if query.group_by is OpportunityGroupBy.MODEL_YEAR:
            clauses.append("a.registration_cohort_year = ?")
            parameters.append(identity[2])
        if query.markets:
            clauses.append(
                f"o.geography IN ({', '.join('?' for _ in query.markets)})"
            )
            parameters.extend(query.markets)
        if query.horizons:
            clauses.append(
                f"o.horizon_year IN ({', '.join('?' for _ in query.horizons)})"
            )
            parameters.extend(query.horizons)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT o.opportunity_id configuration_id,
                    o.geography market, o.horizon_year forecast_horizon,
                    g.display_name generation,
                    COALESCE(g.body_style, 'Not evidenced') body_style,
                    SUM(a.downside_units) downside_units,
                    SUM(a.base_units) base_units,
                    SUM(a.upside_units) upside_units
                FROM opportunity_estimate o
                JOIN canonical_vehicle v
                    ON v.vehicle_id = o.canonical_vehicle_id
                JOIN temp.canonical_identity ident
                    ON ident.vehicle_id = o.canonical_vehicle_id
                JOIN generation_entry g
                    ON g.generation_id = o.generation_id
                JOIN opportunity_cohort_attribution a
                    ON a.opportunity_id = o.opportunity_id
                WHERE {' AND '.join(clauses)}
                GROUP BY o.opportunity_id, o.geography, o.horizon_year,
                    g.display_name, g.body_style
                ORDER BY o.horizon_year, o.geography, g.display_name,
                    o.opportunity_id""",
                parameters,
            ).fetchall()
        return tuple(
            OpportunityContribution(
                configuration_id=row["configuration_id"],
                market=row["market"],
                forecast_horizon=int(row["forecast_horizon"]),
                generation=row["generation"],
                body_style=row["body_style"],
                demand=DemandRange(
                    int(row["downside_units"]),
                    int(row["base_units"]),
                    int(row["upside_units"]),
                ),
            )
            for row in rows
        )

    def drill_down(
        self,
        group_id: str,
        query: OpportunityQuery,
        page: int,
        page_size: int,
    ) -> tuple[OpportunityDrillDownRow, ...]:
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("opportunity drill-down pagination is invalid")
        identity = _decode_group_id(group_id, query.group_by)
        if identity is None or query.group_by is OpportunityGroupBy.EUROCODE:
            # A code's units are allocated per registration, not per configuration.
            return ()
        clauses: list[str] = []
        parameters: list[object] = []
        if self._verified_only:
            if identity[1] is None:
                return ()
            profile = self._generation_catalog.profile_for(
                CanonicalVehicle(
                    "opportunity-drill-down",
                    identity[0],
                    identity[1],
                    None,
                    "Europe",
                )
            )
            if profile is None:
                return ()
            clauses.append(
                "("
                + " OR ".join(
                    "(LOWER(TRIM(vehicle.make)) = ? AND "
                    "LOWER(TRIM(vehicle.model)) = ?)"
                    for _ in profile._normalized_aliases
                )
                + ")"
            )
            parameters.extend(
                value for alias in profile._normalized_aliases for value in alias
            )
        else:
            clauses.append("vehicle.make = ?")
            parameters.append(identity[0])
            if query.group_by in {
                OpportunityGroupBy.MODEL,
                OpportunityGroupBy.MODEL_YEAR,
            }:
                clauses.append("vehicle.model = ?")
                parameters.append(identity[1])
        if query.markets:
            clauses.append(
                f"opportunity.geography IN ({', '.join('?' for _ in query.markets)})"
            )
            parameters.extend(query.markets)
        if query.horizons:
            clauses.append(
                f"opportunity.horizon_year IN ({', '.join('?' for _ in query.horizons)})"
            )
            parameters.extend(query.horizons)
        if query.group_by is OpportunityGroupBy.MODEL_YEAR:
            clauses.append(
                "EXISTS (SELECT 1 "
                "FROM opportunity_input i2 JOIN cohort_estimate c2 "
                "ON c2.cohort_id = i2.cohort_id "
                "WHERE i2.opportunity_id = opportunity.opportunity_id "
                "AND c2.registration_cohort_year = ?)"
            )
            parameters.append(identity[2])
        records = self._planner._query_sqlite(
            self._snapshot_path,
            " AND ".join(clauses),
            tuple(parameters),
            "opportunity_id",
            page_size,
            (page - 1) * page_size,
        )
        coverage = self._coverage_for(records)
        return tuple(
            OpportunityDrillDownRow(
                configuration=record,
                model_year_demand=demand,
                coverage_status=coverage.get(
                    (record.configuration_id, demand.model_year),
                    CoverageStatus.UNCOVERED,
                ),
            )
            for record in records
            for demand in record.model_year_demand
            if query.group_by is not OpportunityGroupBy.MODEL_YEAR
            or demand.model_year == identity[2]
        )

    def fleet_estimates(
        self, group_id: str, query: OpportunityQuery
    ) -> tuple[OpportunityFleetEstimate, ...]:
        identity = _decode_group_id(group_id, query.group_by)
        if identity is None or query.group_by is OpportunityGroupBy.EUROCODE:
            # A surviving fleet is not allocated to codes yet.
            return ()
        clauses: list[str] = []
        parameters: list[object] = []
        brand_expression = "ident.brand"
        model_expression = "ident.model"
        identity_join = ""
        if self._verified_only:
            identity_join = """JOIN temp.reviewed_vehicle_year reviewed
                ON reviewed.brand = LOWER(TRIM(v.make))
                AND reviewed.model = LOWER(TRIM(v.model))
                AND reviewed.registration_year = c.registration_cohort_year"""
            brand_expression = "reviewed.canonical_brand"
            model_expression = "reviewed.canonical_model"
        clauses.append(f"{brand_expression} = ?")
        parameters.append(identity[0])
        if query.group_by is not OpportunityGroupBy.BRAND:
            clauses.append(f"{model_expression} = ?")
            parameters.append(identity[1])
        if query.group_by is OpportunityGroupBy.MODEL_YEAR:
            clauses.append("c.registration_cohort_year = ?")
            parameters.append(identity[2])
        if query.markets:
            clauses.append(
                f"o.geography IN ({', '.join('?' for _ in query.markets)})"
            )
            parameters.extend(query.markets)
        if query.horizons:
            clauses.append(
                f"o.horizon_year IN ({', '.join('?' for _ in query.horizons)})"
            )
            parameters.extend(query.horizons)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT o.geography market,
                    o.horizon_year forecast_horizon,
                    SUM(CAST(c.active_fleet_p50 AS NUMERIC)) estimated_fleet_units
                FROM opportunity_estimate o
                JOIN canonical_vehicle v ON v.vehicle_id = o.canonical_vehicle_id
                JOIN temp.canonical_identity ident
                    ON ident.vehicle_id = o.canonical_vehicle_id
                JOIN opportunity_input i ON i.opportunity_id = o.opportunity_id
                JOIN cohort_estimate c ON c.cohort_id = i.cohort_id
                {identity_join}
                WHERE {' AND '.join(clauses)}
                GROUP BY o.geography, o.horizon_year
                ORDER BY o.horizon_year, o.geography""",
                parameters,
            ).fetchall()
        totals: dict[tuple[str, int], Decimal] = {}
        for row in rows:
            key = (
                world_region_for_market(row["market"]),
                int(row["forecast_horizon"]),
            )
            totals[key] = totals.get(key, Decimal(0)) + Decimal(
                str(row["estimated_fleet_units"])
            )
        return tuple(
            OpportunityFleetEstimate(
                region,
                horizon,
                int(units.quantize(Decimal("1"), rounding=ROUND_HALF_UP)),
            )
            for (region, horizon), units in sorted(
                totals.items(), key=lambda item: (item[0][1], item[0][0])
            )
        )

    def _coverage_for(self, records):  # type: ignore[no-untyped-def]
        if not records:
            return {}
        configuration_ids = tuple(row.configuration_id for row in records)
        placeholders = ", ".join("?" for _ in configuration_ids)
        pairs = {
            (row.brand, row.model, demand.model_year)
            for row in records
            for demand in row.model_year_demand
        }
        pair_clauses = " OR ".join(
            "(brand = ? AND model = ? AND model_year = ?)" for _ in pairs
        )
        pair_parameters = tuple(value for pair in sorted(pairs) for value in pair)
        with sqlite3.connect(self._coverage_path) as connection:
            connection.row_factory = sqlite3.Row
            exact_rows = connection.execute(
                f"""SELECT configuration_id, model_year FROM production_coverage
                WHERE match_type = 'exact_configuration'
                AND configuration_id IN ({placeholders})""",
                configuration_ids,
            ).fetchall()
            fallback_rows = connection.execute(
                f"""SELECT brand, model, model_year FROM production_coverage
                WHERE match_type = 'vehicle_year_fallback'
                AND ({pair_clauses})""",
                pair_parameters,
            ).fetchall()
        exact = {(row["configuration_id"], row["model_year"]) for row in exact_rows}
        fallback = {
            (row["brand"], row["model"], row["model_year"])
            for row in fallback_rows
        }
        return {
            (record.configuration_id, demand.model_year): (
                CoverageStatus.EXACT_COVERED
                if (record.configuration_id, demand.model_year) in exact
                else CoverageStatus.FALLBACK_ONLY
                if (record.brand, record.model, demand.model_year) in fallback
                else CoverageStatus.FALLBACK_ONLY
                if self._worked_models.matches(
                    record.brand, record.model, demand.model_year
                )
                else CoverageStatus.UNCOVERED
            )
            for record in records
            for demand in record.model_year_demand
        }

    def warm(self, group_by: OpportunityGroupBy) -> None:
        """Build the market population before the first visitor waits for it.

        Grouping the whole snapshot takes seconds, so a lazy first request pays
        for it while somebody watches. A host that keeps one machine running and
        gives its health check a grace period can pay it at boot instead.
        """

        if group_by is OpportunityGroupBy.EUROCODE:
            self._eurocode_view(OpportunityQuery(group_by=group_by))
            return
        self._universe(group_by)

    def _refresh_for_coverage(self) -> bool:
        """Report manual coverage, and drop what a change to it invalidates.

        Coverage is not part of the score and never decides where a group's
        units come from, so the whole-market populations stay valid. Cached
        pages and Eurocode views carry coverage units and statuses, so they
        have to go.

        Read through its own connection rather than the snapshot's, so a cached
        page can be served without opening the snapshot at all.
        """

        with contextlib.closing(
            sqlite3.connect(
                f"{self._coverage_path.resolve().as_uri()}?mode=ro", uri=True
            )
        ) as connection:
            count, updated, newest, oldest = connection.execute(
                """SELECT COUNT(*), COALESCE(MAX(updated_at), ''),
                    COALESCE(MAX(coverage_id), ''),
                    COALESCE(MIN(coverage_id), '')
                FROM production_coverage"""
            ).fetchone()
        fingerprint = (int(count), str(updated), str(newest), str(oldest))
        if fingerprint != self._coverage_fingerprint:
            self._uncovered_cache.clear()
            self._eurocode_views.clear()
            self._coverage_fingerprint = fingerprint
        return int(count) > 0

    def _universe(self, group_by: OpportunityGroupBy) -> OpportunityUniverse:
        """The market this grouping is ranked against.

        Keyed on the grouping level alone. ICOR's coverage is never part of a
        population, so it is never asked for here.

        Built on a connection of its own the first time, because the population
        has to exist before the connection that will join against it can be set
        up. That is one extra connection on the cold path, where seconds are
        already being spent, and none afterwards.
        """

        universe = self._universes.get(group_by)
        if universe is not None:
            return universe
        grouped, parameters = self._grouped_cte(
            group_by=group_by,
            include_coverage=False,
            predicates=_WHOLE_MARKET,
            atoms_name="universe_atoms",
        )
        assert not parameters, "the whole market binds no filter"
        with self._connect() as connection:
            universe = OpportunityUniverse.read(connection, grouped, group_by=group_by)
        self._universes[group_by] = universe
        return universe

    @staticmethod
    def _filter(
        query: OpportunityQuery, code_groups: Sequence[tuple[object, ...]] = ()
    ) -> _Filter:
        """Split a query into what each part of the statement may mention.

        The cohort-attribution CTEs join only the estimate, its inputs and their
        cohorts, so a predicate naming the canonical identity cannot be repeated
        inside them — it used to be, which made a searched model-year ranking a
        malformed statement on any snapshot without a materialized attribution
        table. They take the estimate-scoped predicates only; the identity match
        is applied where the identity is in scope.

        ``code_groups`` are the groups that may carry the Eurocode the search
        text names: they are listed beside the groups whose name contains it.
        """

        estimate: list[str] = []
        estimate_parameters: list[object] = []
        if query.markets:
            estimate.append(
                f"o.geography IN ({', '.join('?' for _ in query.markets)})"
            )
            estimate_parameters.extend(query.markets)
        if query.horizons:
            estimate.append(
                f"o.horizon_year IN ({', '.join('?' for _ in query.horizons)})"
            )
            estimate_parameters.extend(query.horizons)
        clauses = list(estimate)
        parameters = list(estimate_parameters)
        text = query.text.strip()
        if text:
            # Matched against the canonical labels rather than the raw ones, so
            # searching "volkswagen" finds the rows filed under "vw" too.
            matches = "ident.brand LIKE ? ESCAPE '@' OR ident.model LIKE ? ESCAPE '@'"
            pattern = f"%{_escape_like(text)}%"
            parameters.extend((pattern, pattern))
            if code_groups:
                # A group key's columns, in the order _group_key builds them.
                columns = (
                    "ident.brand",
                    "ident.model",
                    "attribution.registration_cohort_year",
                )[: len(code_groups[0])]
                placeholders = ", ".join("?" for _ in columns)
                matches += (
                    f" OR ({', '.join(columns)}) IN (VALUES "
                    + ", ".join(f"({placeholders})" for _ in code_groups)
                    + ")"
                )
                parameters.extend(value for group in code_groups for value in group)
            clauses.append(f"({matches})")
        return _Filter(
            estimates=" AND ".join(estimate) if estimate else "1 = 1",
            estimate_parameters=tuple(estimate_parameters),
            atoms=" AND ".join(clauses) if clauses else "1 = 1",
            atom_parameters=tuple(parameters),
        )

    def _grouped_cte(
        self,
        *,
        group_by: OpportunityGroupBy,
        include_coverage: bool,
        predicates: _Filter,
        atoms_name: str,
        cells: bool = False,
    ) -> tuple[str, tuple[object, ...]]:
        """The atoms-to-groups chain, and the values its placeholders need.

        Shared by the ranking and by the whole-market universe, which is this
        same chain with no filter. One definition of a group key means the two
        cannot disagree about which rows are one vehicle — including the
        canonical identity folding that makes ten spellings of Volkswagen one
        brand.

        Every grouping reads each estimate's cohort attribution, whatever ICOR's
        coverage is: coverage only fills the columns that report it, so it can
        never choose a group's units. A snapshot without a materialized
        attribution table computes the attribution inline, which binds the
        estimate-scoped predicates a second time; the parameters are returned
        beside the text so that no caller can get that count wrong.

        ``cells`` adds what a Eurocode allocation reads to each atom (the raw
        vehicle and its label, the market, the horizon) and the evidence kind of
        its cohort, from ``temp.cohort_kind``; the default statement is left
        exactly as it was.
        """

        model = "model" if group_by is not OpportunityGroupBy.BRAND else "NULL"
        year = "model_year" if group_by is OpportunityGroupBy.MODEL_YEAR else "NULL"
        group_columns = ["brand"]
        if group_by is not OpportunityGroupBy.BRAND:
            group_columns.append("model")
        if group_by is OpportunityGroupBy.MODEL_YEAR:
            group_columns.append("model_year")
        groups = ", ".join(group_columns)
        parameters = predicates.atom_parameters
        if self._materialized_attribution:
            cte_prefix = "WITH "
        else:
            cte_prefix = f"WITH {_cohort_attribution_ctes(predicates.estimates)}, "
            parameters = (
                *predicates.estimate_parameters,
                *predicates.atom_parameters,
            )
        model_year = "attribution.registration_cohort_year"
        lineage_joins = (
            "JOIN "
            + (
                "opportunity_cohort_attribution"
                if self._materialized_attribution
                else "cohort_attribution"
            )
            + " attribution "
            "ON attribution.opportunity_id = o.opportunity_id"
        )
        downside_units = "attribution.downside_units"
        base_units = "attribution.base_units"
        upside_units = "attribution.upside_units"
        coverage_status = (
            f"""CASE
                    WHEN EXISTS (
                        SELECT 1 FROM coverage_db.production_coverage pc
                        WHERE pc.match_type = 'exact_configuration'
                        AND pc.configuration_id = {atoms_name}.opportunity_id
                        AND pc.model_year = {atoms_name}.model_year
                    ) THEN 'exact_covered'
                    WHEN EXISTS (
                        SELECT 1 FROM coverage_db.production_coverage pc
                        WHERE pc.match_type = 'vehicle_year_fallback'
                        AND pc.brand = {atoms_name}.brand
                        AND pc.model = {atoms_name}.model
                        AND pc.model_year = {atoms_name}.model_year
                    ) THEN 'fallback_only'
                    WHEN EXISTS (
                        SELECT 1 FROM temp.icor_worked_model wm
                        WHERE wm.brand = LOWER({atoms_name}.brand)
                        AND wm.model = LOWER({atoms_name}.model)
                        AND wm.model_year = {atoms_name}.model_year
                    ) THEN 'fallback_only'
                    ELSE 'uncovered'
                END"""
            if include_coverage
            else "'uncovered'"
        )
        icor_worked = (
            f"""CASE WHEN EXISTS (
                    SELECT 1 FROM temp.icor_worked_model wm
                    WHERE wm.brand = LOWER({atoms_name}.brand)
                    AND wm.model = LOWER({atoms_name}.model)
                    AND wm.model_year = {atoms_name}.model_year
                ) THEN 1 ELSE 0 END"""
            if include_coverage
            else "0"
        )
        identity_join = ""
        brand_expression = "ident.brand"
        model_expression = "ident.model"
        if self._verified_only:
            identity_join = """JOIN temp.reviewed_vehicle_year reviewed
                ON reviewed.brand = LOWER(TRIM(v.make))
                AND reviewed.model = LOWER(TRIM(v.model))
                AND reviewed.registration_year =
                    attribution.registration_cohort_year"""
            brand_expression = "reviewed.canonical_brand"
            model_expression = "reviewed.canonical_model"
        cell_columns = (
            ", o.canonical_vehicle_id vehicle_id, v.model raw_model, "
            "o.geography geography, o.horizon_year horizon, o.rowid estimate_key, "
            "evidence.kind evidence_kind"
            if cells
            else ""
        )
        if cells:
            lineage_joins += (
                " LEFT JOIN temp.cohort_kind evidence"
                " ON evidence.cohort_id = attribution.cohort_id"
            )
        cte = f"""{cte_prefix}{atoms_name} AS MATERIALIZED (
            SELECT o.opportunity_id, {brand_expression} brand,
                {model_expression} model,
                {model_year} model_year,
                {downside_units} downside_units,
                {base_units} base_units,
                {upside_units} upside_units{cell_columns}
            FROM opportunity_estimate o
            JOIN canonical_vehicle v ON v.vehicle_id = o.canonical_vehicle_id
            JOIN temp.canonical_identity ident
                ON ident.vehicle_id = o.canonical_vehicle_id
            {identity_join}
            {lineage_joins}
            WHERE {predicates.atoms}
        ), atoms AS (
            SELECT {atoms_name}.*, {coverage_status} coverage_status,
                {icor_worked} icor_worked
            FROM {atoms_name}
        ), grouped AS (
            SELECT brand, {model} model, {year} model_year,
                SUM(downside_units) downside_units,
                SUM(base_units) base_units, SUM(upside_units) upside_units,
                COUNT(DISTINCT opportunity_id) configuration_count,
                SUM(CASE WHEN coverage_status = 'exact_covered'
                    THEN base_units ELSE 0 END) exact_units,
                SUM(CASE WHEN coverage_status = 'fallback_only'
                    THEN base_units ELSE 0 END) fallback_units,
                SUM(CASE WHEN coverage_status = 'uncovered'
                    THEN base_units ELSE 0 END) uncovered_units,
                SUM(CASE WHEN icor_worked = 1 THEN base_units ELSE 0 END)
                    icor_worked_units
            FROM atoms GROUP BY {groups}
        )"""
        return cte, parameters

    def _scored_cte(
        self,
        query: OpportunityQuery,
        *,
        include_coverage: bool = True,
        code_groups: Sequence[tuple[object, ...]] = (),
    ) -> tuple[str, tuple[object, ...]]:
        """Score the filtered rows against the unfiltered market.

        The percentile is read from
        ``temp.opportunity_universe`` rather than computed here, so a filter
        changes which rows appear and what units they carry but never what a
        score means. The join is NULL-safe with ``IS`` because ``model`` and
        ``model_year`` are NULL at the coarser grouping levels, where ``=``
        would never match and every score would silently be zero.
        """

        grouped, parameters = self._grouped_cte(
            group_by=query.group_by,
            include_coverage=include_coverage,
            predicates=self._filter(query, code_groups),
            atoms_name="base_atoms",
        )
        # A whole-group predicate, so it can only run after grouping: a vehicle
        # ICOR worked on for one model year is still "worked" as a model.
        worked = {
            None: "1 = 1",
            True: "exact_units + fallback_units + icor_worked_units > 0",
            False: "exact_units + fallback_units + icor_worked_units = 0",
        }[query.icor_worked]
        cte = f"""{grouped}, positioned AS (
            SELECT grouped.*,
                COALESCE(universe.universe_percentile, 0.0) demand_percentile,
                universe.universe_rank demand_rank,
                universe.brand IS NULL universe_missing
            FROM grouped
            LEFT JOIN temp.{_UNIVERSE_TABLE} universe
                ON universe.brand = grouped.brand
                AND universe.model IS grouped.model
                AND universe.model_year IS grouped.model_year
        ), scored AS (
            SELECT *, demand_percentile * {DEMAND_POINTS_MAX} demand_points,
                demand_percentile * {DEMAND_POINTS_MAX} total_points
            FROM positioned
            WHERE {worked}
        )"""
        return cte, parameters

    def _available_facets(
        self, connection: sqlite3.Connection
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Every market and horizon the snapshot can forecast.

        Deliberately not narrowed by the caller's own filters: a filter control
        that removed its other options once you used it could not be undone.
        """

        if self._facets is None:
            markets = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT geography FROM opportunity_estimate "
                    "ORDER BY geography"
                )
            )
            horizons = tuple(
                int(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT horizon_year FROM opportunity_estimate "
                    "ORDER BY horizon_year"
                )
            )
            self._facets = (markets, horizons)
        return self._facets

    def _connect(
        self, universe: OpportunityUniverse | None = None, *, cohort_kinds: bool = False
    ) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"{self._snapshot_path.resolve().as_uri()}?mode=ro", uri=True
        )
        connection.row_factory = sqlite3.Row
        coverage_uri = f"{self._coverage_path.resolve().as_uri()}?mode=ro"
        connection.execute("ATTACH DATABASE ? AS coverage_db", (coverage_uri,))
        connection.execute(
            "CREATE TEMP TABLE icor_worked_model "
            "(brand TEXT, model TEXT, model_year INTEGER, PRIMARY KEY (brand, model, model_year))"
        )
        connection.executemany(
            "INSERT INTO temp.icor_worked_model VALUES (?, ?, ?)",
            (
                (record.brand, record.model, record.model_year)
                for record in self._worked_models.records
            ),
        )
        if self._identity_index is None:
            self._identity_index = load_identity_index(connection)
        materialize_identity_table(connection, self._identity_index)
        if self._verified_only:
            connection.execute(
                "CREATE TEMP TABLE reviewed_vehicle_year "
                "(brand TEXT, model TEXT, registration_year INTEGER, "
                "canonical_brand TEXT, canonical_model TEXT, "
                "PRIMARY KEY (brand, model, registration_year))"
            )
            connection.executemany(
                "INSERT INTO temp.reviewed_vehicle_year VALUES (?, ?, ?, ?, ?)",
                (
                    (
                        alias[0],
                        alias[1],
                        year,
                        profile.aliases[0][0],
                        profile.aliases[0][1],
                    )
                    for profile in self._generation_catalog.profiles
                    for alias in profile._normalized_aliases
                    for year in range(
                        min(window.start_month.year for window in profile.windows),
                        2101,
                    )
                    if sum(
                        window.start_month.year <= year
                        and (
                            window.end_month is None
                            or window.end_month.year >= year
                        )
                        for window in profile.windows
                    )
                    == 1
                ),
            )
        if universe is not None:
            # Before the pragma: a temp table is a write, even though it never
            # touches the snapshot. Only the two methods that score pay for it.
            universe.materialize(connection)
        if cohort_kinds:
            _materialize_cohort_kinds(connection)
        connection.execute("PRAGMA query_only = ON")
        return connection

    @staticmethod
    def _integrity_warnings(connection: sqlite3.Connection) -> tuple[str, ...]:
        rows = connection.execute(
            """SELECT pc.coverage_id FROM coverage_db.production_coverage pc
            WHERE NOT EXISTS (
                SELECT 1 FROM opportunity_estimate o
                JOIN canonical_vehicle v ON v.vehicle_id = o.canonical_vehicle_id
                JOIN opportunity_input i ON i.opportunity_id = o.opportunity_id
                JOIN cohort_estimate c ON c.cohort_id = i.cohort_id
                WHERE (pc.match_type = 'exact_configuration'
                    AND pc.configuration_id = o.opportunity_id
                    AND pc.model_year = c.registration_cohort_year)
                OR (pc.match_type = 'vehicle_year_fallback'
                    AND pc.brand = v.make AND pc.model = v.model
                    AND pc.model_year = c.registration_cohort_year)
            ) ORDER BY pc.coverage_id LIMIT 100"""
        ).fetchall()
        return tuple(
            f"Coverage {row['coverage_id']} has an unavailable canonical identity."
            for row in rows
        )

    def _eurocode_breakdowns(
        self,
        connection: sqlite3.Connection,
        query: OpportunityQuery,
        rows: Sequence[sqlite3.Row],
    ) -> dict[tuple[str, str | None, int | None], EurocodeBreakdown]:
        """The Eurocodes of each row shown, from the same estimates its units sum.

        Every registration cell is allocated on its own: a reviewed variant is
        keyed on the publisher's spelling, a drive side on the market, a code's
        window on the registration year. Cohort attribution supplies the year
        and reconciles each estimate to its rounded total, so a breakdown sums
        to its row's units; an estimate without cohorts has no registration
        year to allocate and is the one thing it can leave out.

        Brand rows span too many codes to list, and the reviewed-only ranking
        groups under other labels, so neither is annotated. Search text names
        whole groups (by name or by Eurocode), so it never changes a shown
        group's estimates: the page's groups are selected here directly, and the
        text is neither a predicate nor part of the cache key.
        """

        if (
            self._eurocodes is None
            or self._verified_only
            or query.group_by in {OpportunityGroupBy.BRAND, OpportunityGroupBy.EUROCODE}
            or not rows
        ):
            return {}
        scope = (query.group_by, query.markets, query.horizons)
        found = {
            key: self._eurocode_cache[(scope, key)]
            for key in {_row_key(row) for row in rows}
            if (scope, key) in self._eurocode_cache
        }
        missing = sorted(
            {_row_key(row) for row in rows} - found.keys(),
            key=lambda key: (key[0], key[1] or "", key[2] or 0),
        )
        if missing:
            by_year = query.group_by is OpportunityGroupBy.MODEL_YEAR
            predicates = self._filter(query)
            if self._materialized_attribution:
                prefix = "WITH "
                attribution = "opportunity_cohort_attribution"
                parameters: tuple[object, ...] = ()
            else:
                prefix = f"WITH {_cohort_attribution_ctes(predicates.estimates)}, "
                attribution = "cohort_attribution"
                parameters = predicates.estimate_parameters
            pairs = sorted({(brand, model) for brand, model, _ in missing})
            year_filter = ""
            year_parameters: tuple[object, ...] = ()
            if by_year:
                year_filter = (
                    "AND (page_vehicle.brand, page_vehicle.model, "
                    "attribution.registration_cohort_year) IN (VALUES "
                    + ", ".join("(?, ?, ?)" for _ in missing)
                    + ")"
                )
                year_parameters = tuple(value for key in missing for value in key)
            # The page's vehicles first: the estimate table has no index on its
            # vehicle, and testing membership before joining the labels keeps a
            # page to a fraction of a second.
            cells = connection.execute(
                f"""{prefix}page_vehicle AS MATERIALIZED (
                    SELECT vehicle_id, brand, model FROM temp.canonical_identity ident
                    WHERE (brand, model) IN (VALUES {", ".join("(?, ?)" for _ in pairs)})
                )
                SELECT page_vehicle.brand, page_vehicle.model,
                    attribution.registration_cohort_year registration_year,
                    o.canonical_vehicle_id, v.make, v.model raw_model, o.geography,
                    SUM(attribution.base_units) base_units
                FROM opportunity_estimate o
                JOIN page_vehicle ON page_vehicle.vehicle_id = o.canonical_vehicle_id
                JOIN canonical_vehicle v ON v.vehicle_id = o.canonical_vehicle_id
                JOIN temp.canonical_identity ident
                    ON ident.vehicle_id = o.canonical_vehicle_id
                JOIN {attribution} attribution
                    ON attribution.opportunity_id = o.opportunity_id
                WHERE o.canonical_vehicle_id IN (SELECT vehicle_id FROM page_vehicle)
                    AND {predicates.estimates} {year_filter}
                GROUP BY 1, 2, 3, 4, 5, 6, 7""",
                (
                    *parameters,
                    *(value for pair in pairs for value in pair),
                    *predicates.estimate_parameters,
                    *year_parameters,
                ),
            ).fetchall()
            assert self._identity_index is not None, "set when the connection opened"
            grouped: dict[tuple[str, str | None, int | None], list[EurocodeCell]] = {
                key: [] for key in missing
            }
            for brand, model, year, vehicle_id, make, raw_model, geography, units in cells:
                key = (brand, model, int(year) if by_year else None)
                grouped[key].append(
                    EurocodeCell(
                        identity=identity_key(self._identity_index, vehicle_id, make, raw_model),
                        year=int(year),
                        geography=geography,
                        raw_model=raw_model,
                        units=int(units),
                    )
                )
            for key, key_cells in grouped.items():
                found[key] = eurocode_breakdown(self._eurocodes, key_cells)
                if len(self._eurocode_cache) >= _EUROCODE_CACHE_SIZE:
                    self._eurocode_cache.pop(next(iter(self._eurocode_cache)), None)
                self._eurocode_cache[(scope, key)] = found[key]
        return found

    def _search_by_eurocode(self, query: OpportunityQuery) -> OpportunityPage:
        ranking, population, warnings = self._eurocode_view(query)
        assert self._facets is not None, "read when the view was built"
        markets, horizons = self._facets
        return eurocode_page(
            self._eurocode_rows(ranking, population),
            query,
            ranking,
            population,
            strategy_name=self._strategy.name,
            strategy_version=self._strategy.version,
            integrity_warnings=warnings,
            snapshot_id=self.snapshot_id,
            versions=self.versions,
            available_markets=markets,
            available_horizons=horizons,
        )

    def _eurocode_rows(
        self, ranking: EurocodeRanking, population: EurocodeRanking
    ) -> tuple[OpportunityRow, ...]:
        return eurocode_rows(
            ranking,
            population,
            self._strategy,
            group_id=lambda code: _encode_group_id(
                OpportunityGroupBy.EUROCODE, (code, None, None)
            ),
            evidence_status=EvidenceStatus.VALIDATED,
            data_version=self.snapshot_id,
        )

    def _eurocode_view(
        self, query: OpportunityQuery
    ) -> tuple[EurocodeRanking, EurocodeRanking, tuple[str, ...]]:
        """This view's Eurocode ranking, the whole market's, and the integrity warnings."""

        if self._eurocodes is None or self._verified_only:
            raise EurocodeGroupingUnavailable("Eurocodes are not loaded")
        include_coverage = self._refresh_for_coverage() or bool(self._worked_models.records)
        ranking, warnings = self._eurocode_ranking(
            query.markets, query.horizons, include_coverage
        )
        population, _warnings = self._eurocode_ranking((), (), include_coverage)
        return ranking, population, warnings

    def _eurocode_ranking(
        self,
        markets: tuple[str, ...],
        horizons: tuple[int, ...],
        include_coverage: bool,
    ) -> tuple[EurocodeRanking, tuple[str, ...]]:
        """Allocate every forecast cell of a market/horizon view and fold it by code.

        Reads each estimate's cohort attribution once, through the same atoms the
        other groupings sum, so the codes' certain units and the view's outcome
        totals add up to the units those groupings show. Seconds on a full
        snapshot, so each view is kept until production coverage changes.
        """

        assert self._eurocodes is not None, "checked by the caller"
        key = (markets, horizons, include_coverage)
        cached = self._eurocode_views.get(key)
        if cached is not None:
            return cached
        # One build at a time: a request arriving during the boot-time build
        # waits for it rather than reading every cell a second time.
        with self._eurocode_lock:
            cached = self._eurocode_views.get(key)
            if cached is None:
                cached = self._build_eurocode_ranking(markets, horizons, include_coverage)
                if len(self._eurocode_views) >= _EUROCODE_VIEW_CACHE_SIZE:
                    self._eurocode_views.pop(next(iter(self._eurocode_views)), None)
                self._eurocode_views[key] = cached
        return cached

    def _build_eurocode_ranking(
        self,
        markets: tuple[str, ...],
        horizons: tuple[int, ...],
        include_coverage: bool,
    ) -> tuple[EurocodeRanking, tuple[str, ...]]:
        assert self._eurocodes is not None, "checked by the caller"
        query = OpportunityQuery(
            group_by=OpportunityGroupBy.EUROCODE, markets=markets, horizons=horizons
        )
        cte, parameters = self._grouped_cte(
            group_by=OpportunityGroupBy.EUROCODE,
            include_coverage=include_coverage,
            predicates=self._filter(query),
            atoms_name="eurocode_atoms",
            cells=True,
        )
        with self._connect(cohort_kinds=True) as connection:
            identity_of = self._vehicle_identities(connection)
            cursor = connection.execute(
                f"""{cte}
                SELECT vehicle_id, model_year, geography, raw_model, brand, model,
                    horizon, estimate_key, downside_units, base_units, upside_units,
                    coverage_status, icor_worked, evidence_kind
                FROM atoms""",
                parameters,
            )
            ranking = rank_by_eurocode(
                self._eurocodes,
                (
                    RankedCell(
                        identity=identity_of[row[0]],
                        year=int(row[1]),
                        geography=row[2],
                        raw_model=row[3],
                        brand=row[4],
                        model=row[5],
                        horizon=int(row[6]),
                        configuration=row[7],
                        demand=DemandRange(int(row[8]), int(row[9]), int(row[10])),
                        coverage_status=row[11],
                        icor_worked=bool(row[12]),
                        evidence=None if row[13] is None else _EVIDENCE_KINDS[row[13]],
                    )
                    for row in cursor
                ),
            )
            warnings = self._integrity_warnings(connection)
            self._available_facets(connection)
        return ranking, warnings

    def _groups_carrying(
        self, connection: sqlite3.Connection, query: OpportunityQuery, code: str
    ) -> tuple[tuple[object, ...], ...]:
        """The groups, under this view's markets and horizons, that may carry the code.

        The map names the reviewed identities that can reach the code; each of
        their registration cells is then allocated, exactly as a breakdown does,
        so a group is listed only when its own breakdown would show the code.
        """

        assert self._eurocodes is not None, "only a loaded map reads a code"
        identities = self._eurocodes.identities_carrying(code)
        vehicles = sorted(
            vehicle_id
            for vehicle_id, identity in self._vehicle_identities(connection).items()
            if identity in identities
        )
        if not vehicles:
            return ()
        predicates = self._filter(replace(query, text=""))
        if self._materialized_attribution:
            prefix = "WITH "
            attribution = "opportunity_cohort_attribution"
            parameters: tuple[object, ...] = ()
        else:
            prefix = f"WITH {_cohort_attribution_ctes(predicates.estimates)}, "
            attribution = "cohort_attribution"
            parameters = predicates.estimate_parameters
        cells = connection.execute(
            f"""{prefix}code_vehicle(vehicle_id) AS MATERIALIZED (
                VALUES {", ".join("(?)" for _ in vehicles)}
            )
            SELECT ident.brand, ident.model,
                attribution.registration_cohort_year registration_year,
                o.canonical_vehicle_id, v.model raw_model, o.geography,
                SUM(attribution.base_units) base_units
            FROM opportunity_estimate o
            JOIN canonical_vehicle v ON v.vehicle_id = o.canonical_vehicle_id
            JOIN temp.canonical_identity ident
                ON ident.vehicle_id = o.canonical_vehicle_id
            JOIN {attribution} attribution
                ON attribution.opportunity_id = o.opportunity_id
            WHERE o.canonical_vehicle_id IN (SELECT vehicle_id FROM code_vehicle)
                AND {predicates.estimates}
            GROUP BY 1, 2, 3, 4, 5, 6""",
            (*parameters, *vehicles, *predicates.estimate_parameters),
        ).fetchall()
        identity_of = self._vehicle_identities(connection)
        groups: set[tuple[object, ...]] = set()
        for brand, model, year, vehicle_id, raw_model, geography, units in cells:
            if not units:
                continue
            allocation = self._eurocodes.allocate(
                identity_of[vehicle_id], int(year), geography, raw_model=raw_model
            )
            if code in allocation.codes:
                groups.add(
                    (brand,)
                    if query.group_by is OpportunityGroupBy.BRAND
                    else (brand, model)
                    if query.group_by is OpportunityGroupBy.MODEL
                    else (brand, model, int(year))
                )
        return tuple(sorted(groups, key=repr))

    def _vehicle_identities(self, connection: sqlite3.Connection) -> dict[str, str]:
        """Each forecast vehicle's Eurocode identity key, computed once."""

        if self._identity_keys is None:
            assert self._identity_index is not None, "set when the connection opened"
            self._identity_keys = {
                vehicle_id: identity_key(self._identity_index, vehicle_id, make, model)
                for vehicle_id, make, model in connection.execute(
                    "SELECT vehicle_id, make, model FROM canonical_vehicle "
                    "WHERE vehicle_id IN "
                    "(SELECT canonical_vehicle_id FROM opportunity_estimate)"
                )
            }
        return self._identity_keys

    def _row(
        self,
        row: sqlite3.Row,
        group_by: OpportunityGroupBy,
        universe: OpportunityUniverse,
        eurocodes: EurocodeBreakdown | None = None,
    ) -> OpportunityRow:
        identity = (row["brand"], row["model"], row["model_year"])
        group_id = _encode_group_id(group_by, identity)
        demand = DemandRange(
            int(row["downside_units"]),
            int(row["base_units"]),
            int(row["upside_units"]),
        )
        exact = int(row["exact_units"])
        fallback = int(row["fallback_units"])
        uncovered = int(row["uncovered_units"])
        demand_points = float(row["demand_points"])
        rank = row["demand_rank"]
        score = OpportunityScore(
            group_id=group_id,
            demand_percentile=float(row["demand_percentile"]),
            demand_points=demand_points,
            demand_rank=None if rank is None else int(rank),
            demand_population=universe.population,
            demand_basis=universe.basis,
            total_points=float(row["total_points"]),
            strategy_name=self._strategy.name,
            strategy_version=self._strategy.version,
            explanation=demand_explanation(demand_points),
        )
        generation = None
        if row["model"] is not None and row["model_year"] is not None:
            vehicle = CanonicalVehicle(
                vehicle_id="opportunity-row",
                make=row["brand"],
                model=row["model"],
                model_year=None,
                market="Europe",
            )
            generation = self._generation_catalog.entry_for_year(
                vehicle,
                int(row["model_year"]),
                registry_version=_GENERATION_REGISTRY,
            )
        brand = (
            source_vehicle_display_label(row["brand"])
            if self._model_year_catalog
            else row["brand"]
        )
        model = (
            source_vehicle_display_label(row["model"])
            if self._model_year_catalog and row["model"] is not None
            else row["model"]
        )
        return OpportunityRow(
            group_id=group_id,
            group_by=group_by,
            brand=brand,
            model=model,
            model_year=row["model_year"],
            generation_name=generation.display_name if generation else None,
            generation_basis=(
                "manufacturer_generation_window" if generation else None
            ),
            generation_source_url=(generation.evidence_ids[0] if generation else None),
            icor_worked_base_units=int(row["icor_worked_units"]),
            demand=demand,
            contributing_configuration_count=int(row["configuration_count"]),
            exact_covered_base_units=exact,
            fallback_covered_base_units=fallback,
            uncovered_base_units=uncovered,
            coverage_status=_coverage_status(
                demand.base_units, exact, fallback, uncovered
            ),
            score=score,
            evidence_status=EvidenceStatus.VALIDATED,
            data_version=self.snapshot_id,
            eurocodes=eurocodes,
        )


def _row_key(row: sqlite3.Row) -> tuple[str, str | None, int | None]:
    return row["brand"], row["model"], row["model_year"]


def _materialize_cohort_kinds(connection: sqlite3.Connection) -> None:
    """Each cohort's evidence kind, as an index into ``_EVIDENCE_KINDS``.

    One pass over the cohorts. A first reason code no build writes is stored as
    -1 and then refused by name, never counted as some kind.
    """

    cases = " ".join("WHEN ? THEN ?" for _code in COHORT_EVIDENCE_KINDS)
    connection.execute(
        "CREATE TEMP TABLE cohort_kind "
        "(cohort_id TEXT PRIMARY KEY, kind INTEGER NOT NULL) WITHOUT ROWID"
    )
    connection.execute(
        f"""INSERT INTO temp.cohort_kind
        SELECT cohort_id, CASE json_extract(reason_codes, '$[0]') {cases} ELSE -1 END
        FROM cohort_estimate""",
        tuple(
            value
            for code, kind in COHORT_EVIDENCE_KINDS.items()
            for value in (code, _EVIDENCE_KINDS.index(kind))
        ),
    )
    unknown = connection.execute(
        """SELECT json_extract(c.reason_codes, '$[0]')
        FROM temp.cohort_kind k JOIN cohort_estimate c ON c.cohort_id = k.cohort_id
        WHERE k.kind = -1 LIMIT 1"""
    ).fetchone()
    if unknown is not None:
        raise ValueError(f"unknown cohort evidence {unknown[0]!r}")


def _escape_like(value: str) -> str:
    """Make a user's literal text mean itself inside a LIKE pattern."""

    for character in ("@", "%", "_"):
        value = value.replace(character, "@" + character)
    return value


def _encode_group_id(
    group_by: OpportunityGroupBy, identity: tuple[object, object, object]
) -> str:
    payload = json.dumps(
        (group_by.value, *identity), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    token = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f"{group_by.value}-{token}"


def _decode_group_id(
    group_id: str, group_by: OpportunityGroupBy
) -> tuple[str, str | None, int | None] | None:
    prefix = f"{group_by.value}-"
    if not group_id.startswith(prefix):
        return None
    token = group_id[len(prefix) :]
    try:
        payload = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        values = json.loads(payload)
    except (ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(values, list)
        or len(values) != 4
        or values[0] != group_by.value
        or not isinstance(values[1], str)
        or (values[2] is not None and not isinstance(values[2], str))
        or (values[3] is not None and type(values[3]) is not int)
    ):
        return None
    return values[1], values[2], values[3]
