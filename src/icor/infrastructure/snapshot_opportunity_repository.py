"""Bounded SQLite queries for opportunity ranking and drill-down."""

from __future__ import annotations

import base64
import json
import sqlite3
from math import ceil
from pathlib import Path

from icor.application.opportunities import (
    OpportunityDrillDownRow,
    OpportunityGroupBy,
    OpportunityPage,
    OpportunityQuery,
    OpportunityRow,
    OpportunitySummary,
    _coverage_status,
)
from icor.application.ranking import RankingStrategy
from icor.domain.opportunities import CoverageStatus, OpportunityScore
from icor.domain.planner import DemandRange, EvidenceStatus
from icor.infrastructure.snapshot_planner_repository import SnapshotPlannerRepository
from icor.infrastructure.sqlite_coverage_repository import SQLiteCoverageRepository


class SnapshotOpportunityRepository:
    """Rank snapshot opportunities without materializing planner configurations."""

    def __init__(
        self,
        planner: SnapshotPlannerRepository,
        coverage: SQLiteCoverageRepository,
        strategy: RankingStrategy,
    ) -> None:
        path = getattr(planner._ledger, "path", None)
        if not isinstance(path, Path):
            raise ValueError("snapshot opportunity repository requires SQLite")
        self._snapshot_path = path
        self._planner = planner
        self._coverage_path = coverage.path
        self._strategy = strategy
        self.snapshot_id = planner.snapshot_id
        self.versions = planner.versions

    def search(self, query: OpportunityQuery) -> OpportunityPage:
        cte, parameters = self._scored_cte(query)
        offset = (query.page - 1) * query.page_size
        with self._connect() as connection:
            summary = connection.execute(
                f"""{cte}
                SELECT COUNT(*) total, COALESCE(SUM(base_units), 0) base_units,
                    COALESCE(SUM(exact_units), 0) exact_units,
                    COALESCE(SUM(CASE WHEN demand_percentile >= 0.75
                        THEN uncovered_units ELSE 0 END), 0) high_uncovered
                FROM scored""",
                parameters,
            ).fetchone()
            rows = connection.execute(
                f"""{cte}
                SELECT * FROM scored
                ORDER BY total_points DESC, base_units DESC, brand, model, model_year
                LIMIT ? OFFSET ?""",
                (*parameters, query.page_size, offset),
            ).fetchall()
            warnings = self._integrity_warnings(connection)
        items = tuple(self._row(row, query.group_by) for row in rows)
        total = int(summary["total"])
        return OpportunityPage(
            items=items,
            summary=OpportunitySummary(
                base_units=int(summary["base_units"]),
                exact_covered_base_units=int(summary["exact_units"]),
                high_demand_uncovered_base_units=int(summary["high_uncovered"]),
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
        clauses = ["vehicle.make = ?"]
        parameters: list[object] = [identity[0]]
        if query.group_by in {OpportunityGroupBy.MODEL, OpportunityGroupBy.MODEL_YEAR}:
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
                "? = (SELECT MIN(c2.registration_cohort_year) "
                "FROM opportunity_input i2 JOIN cohort_estimate c2 "
                "ON c2.cohort_id = i2.cohort_id "
                "WHERE i2.opportunity_id = opportunity.opportunity_id)"
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
                else CoverageStatus.UNCOVERED
            )
            for record in records
            for demand in record.model_year_demand
        }

    def _scored_cte(
        self, query: OpportunityQuery
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
        cte = f"""WITH base_atoms AS (
            SELECT o.opportunity_id, v.make brand, v.model,
                MIN(c.registration_cohort_year) model_year,
                CAST(ROUND(CAST(o.p10 AS NUMERIC), 0) AS INTEGER) downside_units,
                CAST(ROUND(CAST(o.p50 AS NUMERIC), 0) AS INTEGER) base_units,
                CAST(ROUND(CAST(o.p90 AS NUMERIC), 0) AS INTEGER) upside_units
            FROM opportunity_estimate o
            JOIN canonical_vehicle v ON v.vehicle_id = o.canonical_vehicle_id
            JOIN opportunity_input i ON i.opportunity_id = o.opportunity_id
            JOIN cohort_estimate c ON c.cohort_id = i.cohort_id
            WHERE {where}
            GROUP BY o.opportunity_id
        ), atoms AS (
            SELECT base_atoms.*,
                CASE
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
                    ELSE 'uncovered'
                END coverage_status
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
                    THEN base_units ELSE 0 END) uncovered_units
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
        return OpportunityRow(
            group_id=group_id,
            group_by=group_by,
            brand=row["brand"],
            model=row["model"],
            model_year=row["model_year"],
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
