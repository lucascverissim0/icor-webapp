"""Bounded SQLite queries for opportunity ranking and drill-down."""

from __future__ import annotations

import base64
import json
import sqlite3
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
    OpportunitySummary,
    _coverage_status,
    world_region_for_market,
)
from icor.application.ranking import RankingStrategy
from icor.application.worked_models import IcorWorkedModelCatalog
from icor.domain.evidence import CanonicalVehicle
from icor.domain.opportunities import CoverageStatus, OpportunityScore
from icor.domain.planner import DemandRange, EvidenceStatus
from icor.evidence.normalization import source_vehicle_display_label
from icor.generations.public_catalog import ranking_public_generation_catalog
from icor.infrastructure.snapshot_planner_repository import (
    SnapshotPlannerRepository,
    _cohort_attribution_ctes,
)
from icor.infrastructure.sqlite_coverage_repository import SQLiteCoverageRepository

_GENERATION_REGISTRY = "public-generation-registry-v2"


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
        with self._connect() as connection:
            has_manual_coverage = bool(
                connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM coverage_db.production_coverage)"
                ).fetchone()[0]
            )
            if not has_manual_coverage:
                cached = self._uncovered_cache.get(query)
                if cached is not None:
                    return cached
            cte, parameters = self._scored_cte(
                query,
                include_coverage=has_manual_coverage or bool(self._worked_models.records),
            )
            rows = connection.execute(
                f"""{cte}
                SELECT *, COUNT(*) OVER () summary_total,
                    SUM(base_units) OVER () summary_base_units,
                    SUM(exact_units) OVER () summary_exact_units,
                    SUM(CASE WHEN demand_percentile >= 0.75
                        THEN uncovered_units ELSE 0 END) OVER () summary_high_uncovered
                FROM scored
                ORDER BY total_points DESC, base_units DESC, brand, model, model_year
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
                            summary_high_uncovered
                    FROM scored""",
                    parameters,
                ).fetchone()
            warnings = self._integrity_warnings(connection)
        items = tuple(self._row(row, query.group_by) for row in rows)
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
        with self._connect() as connection:
            has_coverage = bool(
                connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM coverage_db.production_coverage)"
                ).fetchone()[0]
            )
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
            cte, parameters = self._scored_cte(
                query,
                include_coverage=has_coverage or bool(self._worked_models.records),
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
        return self._row(row, query.group_by) if row is not None else None

    def contributions(
        self, group_id: str, query: OpportunityQuery
    ) -> tuple[OpportunityContribution, ...]:
        identity = _decode_group_id(group_id, query.group_by)
        if identity is None:
            return ()
        clauses = ["LOWER(TRIM(v.make)) = LOWER(TRIM(?))"]
        parameters: list[object] = [identity[0]]
        if query.group_by is not OpportunityGroupBy.BRAND:
            clauses.append("LOWER(TRIM(v.model)) = LOWER(TRIM(?))")
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
        brand_expression = "v.make"
        model_expression = "v.model"
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

    def _scored_cte(
        self, query: OpportunityQuery, *, include_coverage: bool = True
    ) -> tuple[str, tuple[object, ...]]:
        clauses: list[str] = []
        parameters: list[object] = []
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
        where = " AND ".join(clauses) if clauses else "1 = 1"
        model = "model" if query.group_by is not OpportunityGroupBy.BRAND else "NULL"
        year = "model_year" if query.group_by is OpportunityGroupBy.MODEL_YEAR else "NULL"
        group_columns = ["brand"]
        if query.group_by is not OpportunityGroupBy.BRAND:
            group_columns.append("model")
        if query.group_by is OpportunityGroupBy.MODEL_YEAR:
            group_columns.append("model_year")
        groups = ", ".join(group_columns)
        needs_model_year = (
            include_coverage
            or query.group_by is OpportunityGroupBy.MODEL_YEAR
            or self._verified_only
            or self._model_year_catalog
        )
        if needs_model_year:
            cte_prefix = (
                "WITH "
                if self._materialized_attribution
                else f"WITH {_cohort_attribution_ctes(where)}, "
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
            """CASE
                    WHEN EXISTS (
                        SELECT 1 FROM coverage_db.production_coverage pc
                        WHERE pc.match_type = 'exact_configuration'
                        AND pc.configuration_id = base_atoms.opportunity_id
                        AND pc.model_year = base_atoms.model_year
                    ) THEN 'exact_covered'
                    WHEN EXISTS (
                        SELECT 1 FROM coverage_db.production_coverage pc
                        WHERE pc.match_type = 'vehicle_year_fallback'
                        AND pc.brand = base_atoms.brand
                        AND pc.model = base_atoms.model
                        AND pc.model_year = base_atoms.model_year
                    ) THEN 'fallback_only'
                    WHEN EXISTS (
                        SELECT 1 FROM temp.icor_worked_model wm
                        WHERE wm.brand = LOWER(base_atoms.brand)
                        AND wm.model = LOWER(base_atoms.model)
                        AND wm.model_year = base_atoms.model_year
                    ) THEN 'fallback_only'
                    ELSE 'uncovered'
                END"""
            if include_coverage
            else "'uncovered'"
        )
        icor_worked = (
            """CASE WHEN EXISTS (
                    SELECT 1 FROM temp.icor_worked_model wm
                    WHERE wm.brand = LOWER(base_atoms.brand)
                    AND wm.model = LOWER(base_atoms.model)
                    AND wm.model_year = base_atoms.model_year
                ) THEN 1 ELSE 0 END"""
            if include_coverage
            else "0"
        )
        identity_join = ""
        brand_expression = "v.make"
        model_expression = "v.model"
        if self._verified_only:
            identity_join = """JOIN temp.reviewed_vehicle_year reviewed
                ON reviewed.brand = LOWER(TRIM(v.make))
                AND reviewed.model = LOWER(TRIM(v.model))
                AND reviewed.registration_year =
                    attribution.registration_cohort_year"""
            brand_expression = "reviewed.canonical_brand"
            model_expression = "reviewed.canonical_model"
        cte = f"""{cte_prefix}base_atoms AS MATERIALIZED (
            SELECT o.opportunity_id, {brand_expression} brand,
                {model_expression} model,
                {model_year} model_year,
                {downside_units} downside_units,
                {base_units} base_units,
                {upside_units} upside_units
            FROM opportunity_estimate o
            JOIN canonical_vehicle v ON v.vehicle_id = o.canonical_vehicle_id
            {identity_join}
            {lineage_joins}
            WHERE {where}
        ), atoms AS (
            SELECT base_atoms.*, {coverage_status} coverage_status,
                {icor_worked} icor_worked
            FROM base_atoms
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
        ), ranked AS (
            SELECT *, RANK() OVER (ORDER BY base_units) demand_rank,
                COUNT(*) OVER (PARTITION BY base_units) tie_count,
                COUNT(*) OVER () group_count
            FROM grouped
        ), percentile AS (
            SELECT *, CASE
                WHEN base_units = 0 THEN 0.0
                WHEN group_count = 1 THEN 1.0
                ELSE (demand_rank - 1 + (tie_count - 1) / 2.0) / (group_count - 1)
                END demand_percentile
            FROM ranked
        ), scored AS (
            SELECT *, demand_percentile * 80.0 demand_points,
                CASE WHEN base_units = 0 THEN 0.0
                    ELSE (exact_units + fallback_units * 0.5) * 1.0 / base_units
                    END readiness_ratio,
                CASE WHEN base_units = 0 THEN 0.0
                    ELSE (exact_units + fallback_units * 0.5) * 20.0 / base_units
                    END readiness_points,
                demand_percentile * 80.0 + CASE WHEN base_units = 0 THEN 0.0
                    ELSE (exact_units + fallback_units * 0.5) * 20.0 / base_units
                    END total_points
            FROM percentile
        )"""
        return cte, tuple(parameters)

    def _connect(self) -> sqlite3.Connection:
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

    def _row(self, row: sqlite3.Row, group_by: OpportunityGroupBy) -> OpportunityRow:
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
        score = OpportunityScore(
            group_id=group_id,
            demand_percentile=float(row["demand_percentile"]),
            demand_points=demand_points,
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
