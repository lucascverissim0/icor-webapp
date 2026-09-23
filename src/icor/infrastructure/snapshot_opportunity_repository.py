"""Bounded SQLite queries for opportunity ranking and drill-down."""

from __future__ import annotations

import base64
import contextlib
import json
import sqlite3
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from math import ceil
from pathlib import Path

from icor.application.opportunities import (
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
    world_region_for_market,
)
from icor.application.ranking import (
    DEMAND_POINTS_MAX,
    READINESS_POINTS_MAX,
    RankingStrategy,
)
from icor.application.worked_models import IcorWorkedModelCatalog
from icor.domain.evidence import CanonicalVehicle
from icor.domain.opportunities import CoverageStatus, OpportunityScore
from icor.domain.planner import DemandRange, EvidenceStatus
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
        self.snapshot_id = planner.snapshot_id
        self.versions = planner.versions
        self._uncovered_cache: dict[OpportunityQuery, OpportunityPage] = {}
        self._identity_index: VehicleIdentityIndex | None = None
        self._facets: tuple[tuple[str, ...], tuple[int, ...]] | None = None
        # The snapshot is immutable and opened read-only, so a population built
        # from it never goes stale on its own. Production coverage is the one
        # mutable input, and the fingerprint below is what notices it moving.
        self._universes: dict[
            tuple[OpportunityGroupBy, bool], OpportunityUniverse
        ] = {}
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
        offset = (query.page - 1) * query.page_size
        has_manual_coverage = self._refresh_for_coverage()
        if not has_manual_coverage:
            cached = self._uncovered_cache.get(query)
            if cached is not None:
                return cached
        include_coverage = has_manual_coverage or bool(self._worked_models.records)
        universe = self._universe(query.group_by, include_coverage)
        with self._connect(universe) as connection:
            cte, parameters = self._scored_cte(
                query, include_coverage=include_coverage
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
        items = tuple(self._row(row, query.group_by, universe) for row in rows)
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
        universe = self._universe(query.group_by, include_coverage)
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
        return self._row(row, query.group_by, universe) if row is not None else None

    def contributions(
        self, group_id: str, query: OpportunityQuery
    ) -> tuple[OpportunityContribution, ...]:
        identity = _decode_group_id(group_id, query.group_by)
        if identity is None:
            return ()
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
        if identity is None:
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
        if identity is None:
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

        has_manual_coverage = self._refresh_for_coverage()
        self._universe(
            group_by, has_manual_coverage or bool(self._worked_models.records)
        )

    def _refresh_for_coverage(self) -> bool:
        """Report manual coverage, and drop what a change to it invalidates.

        Readiness is part of the whole-market population now, so recording
        coverage for one vehicle changes the score of vehicles nobody mentioned.
        Both the population and the cached pages have to go.

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
            self._universes.clear()
            self._uncovered_cache.clear()
            self._coverage_fingerprint = fingerprint
        return int(count) > 0

    def _universe(
        self, group_by: OpportunityGroupBy, include_coverage: bool
    ) -> OpportunityUniverse:
        """The market this grouping is ranked against.

        Keyed on ``include_coverage`` as well as the grouping level because it
        decides where a group's units come from — summed cohort attribution or a
        rounded opportunity quantile — and the two are not interchangeable.

        Built on a connection of its own the first time, because the population
        has to exist before the connection that will join against it can be set
        up. That is one extra connection on the cold path, where seconds are
        already being spent, and none afterwards.
        """

        key = (group_by, include_coverage)
        universe = self._universes.get(key)
        if universe is not None:
            return universe
        grouped, parameters = self._grouped_cte(
            group_by=group_by,
            include_coverage=include_coverage,
            predicates=_WHOLE_MARKET,
            atoms_name="universe_atoms",
        )
        assert not parameters, "the whole market binds no filter"
        with self._connect() as connection:
            universe = OpportunityUniverse.read(
                connection,
                grouped,
                group_by=group_by,
                needs_model_year=self._needs_model_year(group_by, include_coverage),
            )
        self._universes[key] = universe
        return universe

    @staticmethod
    def _filter(query: OpportunityQuery) -> _Filter:
        """Split a query into what each part of the statement may mention.

        The cohort-attribution CTEs join only the estimate, its inputs and their
        cohorts, so a predicate naming the canonical identity cannot be repeated
        inside them — it used to be, which made a searched model-year ranking a
        malformed statement on any snapshot without a materialized attribution
        table. They take the estimate-scoped predicates only; the identity match
        is applied where the identity is in scope.
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
            clauses.append(
                "(ident.brand LIKE ? ESCAPE '@' "
                "OR ident.model LIKE ? ESCAPE '@')"
            )
            pattern = f"%{_escape_like(text)}%"
            parameters.extend((pattern, pattern))
        return _Filter(
            estimates=" AND ".join(estimate) if estimate else "1 = 1",
            estimate_parameters=tuple(estimate_parameters),
            atoms=" AND ".join(clauses) if clauses else "1 = 1",
            atom_parameters=tuple(parameters),
        )

    def _needs_model_year(
        self, group_by: OpportunityGroupBy, include_coverage: bool
    ) -> bool:
        return (
            include_coverage
            or group_by is OpportunityGroupBy.MODEL_YEAR
            or self._verified_only
            or self._model_year_catalog
        )

    def _grouped_cte(
        self,
        *,
        group_by: OpportunityGroupBy,
        include_coverage: bool,
        predicates: _Filter,
        atoms_name: str,
    ) -> tuple[str, tuple[object, ...]]:
        """The atoms-to-groups chain, and the values its placeholders need.

        Shared by the ranking and by the whole-market universe, which is this
        same chain with no filter. One definition of a group key means the two
        cannot disagree about which rows are one vehicle — including the
        canonical identity folding that makes ten spellings of Volkswagen one
        brand.

        A snapshot without a materialized attribution table computes the
        attribution inline, which binds the estimate-scoped predicates a second
        time; the parameters are returned beside the text so that no caller can
        get that count wrong.
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
        if self._needs_model_year(group_by, include_coverage):
            if self._materialized_attribution:
                cte_prefix = "WITH "
            else:
                cte_prefix = (
                    f"WITH {_cohort_attribution_ctes(predicates.estimates)}, "
                )
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
        else:
            cte_prefix = "WITH "
            model_year = "NULL"
            lineage_joins = ""
            downside_units = (
                "CAST(ROUND(CAST(o.p10 AS NUMERIC), 0) AS INTEGER)"
            )
            base_units = "CAST(ROUND(CAST(o.p50 AS NUMERIC), 0) AS INTEGER)"
            upside_units = (
                "CAST(ROUND(CAST(o.p90 AS NUMERIC), 0) AS INTEGER)"
            )
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
        cte = f"""{cte_prefix}{atoms_name} AS MATERIALIZED (
            SELECT o.opportunity_id, {brand_expression} brand,
                {model_expression} model,
                {model_year} model_year,
                {downside_units} downside_units,
                {base_units} base_units,
                {upside_units} upside_units
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
        self, query: OpportunityQuery, *, include_coverage: bool = True
    ) -> tuple[str, tuple[object, ...]]:
        """Score the filtered rows against the unfiltered market.

        The percentile and the readiness ratio are read from
        ``temp.opportunity_universe`` rather than computed here, so a filter
        changes which rows appear and what units they carry but never what a
        score means. The join is NULL-safe with ``IS`` because ``model`` and
        ``model_year`` are NULL at the coarser grouping levels, where ``=``
        would never match and every score would silently be zero.
        """

        grouped, parameters = self._grouped_cte(
            group_by=query.group_by,
            include_coverage=include_coverage,
            predicates=self._filter(query),
            atoms_name="base_atoms",
        )
        cte = f"""{grouped}, positioned AS (
            SELECT grouped.*,
                COALESCE(universe.universe_percentile, 0.0) demand_percentile,
                universe.universe_rank demand_rank,
                COALESCE(universe.universe_readiness_ratio, 0.0) readiness_ratio,
                universe.brand IS NULL universe_missing
            FROM grouped
            LEFT JOIN temp.{_UNIVERSE_TABLE} universe
                ON universe.brand = grouped.brand
                AND universe.model IS grouped.model
                AND universe.model_year IS grouped.model_year
        ), scored AS (
            SELECT *, demand_percentile * {DEMAND_POINTS_MAX} demand_points,
                readiness_ratio * {READINESS_POINTS_MAX} readiness_points,
                demand_percentile * {DEMAND_POINTS_MAX}
                    + readiness_ratio * {READINESS_POINTS_MAX} total_points
            FROM positioned
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
        self, universe: OpportunityUniverse | None = None
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

    def _row(
        self,
        row: sqlite3.Row,
        group_by: OpportunityGroupBy,
        universe: OpportunityUniverse,
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
        readiness_points = float(row["readiness_points"])
        rank = row["demand_rank"]
        score = OpportunityScore(
            group_id=group_id,
            demand_percentile=float(row["demand_percentile"]),
            demand_points=demand_points,
            demand_rank=None if rank is None else int(rank),
            demand_population=universe.population,
            demand_basis=universe.basis,
            readiness_ratio=float(row["readiness_ratio"]),
            readiness_points=readiness_points,
            total_points=float(row["total_points"]),
            strategy_name=self._strategy.name,
            strategy_version=self._strategy.version,
            explanation=(
                f"{demand_points:g} demand points and "
                f"{readiness_points:g} production-readiness points."
            ),
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
        )


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
