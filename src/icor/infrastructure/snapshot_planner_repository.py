"""Read-only planner projection over one verified evidence snapshot."""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from math import ceil
from pathlib import Path

from icor.domain.evidence import ConfidenceBand
from icor.domain.generations import GenerationIdentityKind
from icor.domain.planner import (
    Confidence,
    ConfidenceLevel,
    DemandRange,
    Equipment,
    EvidenceStatus,
    ModelYearDemand,
    PlannerPage,
    PlannerQuery,
    PlannerSummary,
    PlanningConfiguration,
    SortDirection,
    SortField,
    SourceSummary,
    filter_sort_paginate,
)
from icor.domain.snapshots import SnapshotManifest, SnapshotVersions


class SnapshotPlannerRepository:
    """Project generation-level opportunity rows without claiming exact fitment."""

    def __init__(self, ledger, manifest: SnapshotManifest) -> None:
        self._ledger = ledger
        self.manifest = manifest
        self.snapshot_id = manifest.snapshot_id
        self.versions: SnapshotVersions = manifest.versions
        self._records: tuple[PlanningConfiguration, ...] | None = None
        self._source_cache: tuple[SourceSummary, ...] | None = None
        self._search_cache: dict[PlannerQuery, PlannerPage] = {}

    def list_all(self) -> tuple[PlanningConfiguration, ...]:
        if self._records is None:
            self._records = self._project()
        return self._records

    def get(self, configuration_id: str) -> PlanningConfiguration | None:
        path = getattr(self._ledger, "path", None)
        if isinstance(path, Path):
            rows = self._query_sqlite(
                path,
                "opportunity.opportunity_id = ?",
                (configuration_id,),
                "opportunity.opportunity_id",
                1,
                0,
            )
            return rows[0] if rows else None
        return next(
            (row for row in self.list_all() if row.configuration_id == configuration_id),
            None,
        )

    def list_model_year_demand(
        self, configuration_id: str, page: int = 1, page_size: int = 100
    ) -> tuple[ModelYearDemand, ...]:
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("model-year pagination is invalid")
        record = self.get(configuration_id)
        if record is None:
            return ()
        start = (page - 1) * page_size
        return record.model_year_demand[start : start + page_size]

    def options(self):  # type: ignore[no-untyped-def]
        from icor.application.planner import (
            PlannerOptions,
            ScenarioMetadata,
            options_from_records,
        )

        path = getattr(self._ledger, "path", None)
        if not isinstance(path, Path):
            return options_from_records(self._project_objects(), self.versions)
        with self._connect(path) as connection:
            rows = connection.execute(
                """SELECT option_kind, option_value FROM planner_option
                ORDER BY option_kind, sort_key, option_value"""
            ).fetchall()
        if not rows:
            raise ValueError("planner repository contains no configurations")
        grouped: dict[str, list[str]] = {}
        for row in rows:
            grouped.setdefault(row["option_kind"], []).append(row["option_value"])
        return PlannerOptions(
            markets=tuple(grouped.get("market", ())),
            horizons=tuple(int(value) for value in grouped.get("horizon", ())),
            brands=tuple(grouped.get("brand", ())),
            models=tuple(grouped.get("model", ())),
            evidence_statuses=(EvidenceStatus.VALIDATED,),
            scenario=ScenarioMetadata(
                name="Generation replacement opportunity baseline",
                description=(
                    "Official registration history projected to generation-level "
                    "replacement opportunity ranges. This is not exact fitment demand."
                ),
                evidence_status=EvidenceStatus.VALIDATED,
                data_version=self.snapshot_id,
                updated_at=self.manifest.built_at,
                versions=self.versions,
            ),
        )

    def search(self, query: PlannerQuery) -> PlannerPage:
        path = getattr(self._ledger, "path", None)
        if not isinstance(path, Path):
            return filter_sort_paginate(self._project_objects(), query)
        cached = self._search_cache.get(query)
        if cached is not None:
            return cached
        result = self._search_sqlite(path, query)
        if len(self._search_cache) >= 128:
            self._search_cache.pop(next(iter(self._search_cache)))
        self._search_cache[query] = result
        return result

    def _project(self) -> tuple[PlanningConfiguration, ...]:
        path = getattr(self._ledger, "path", None)
        if isinstance(path, Path):
            return self._project_sqlite(path)
        return self._project_objects()

    def _sources(self) -> tuple[SourceSummary, ...]:
        if self._source_cache is None:
            self._source_cache = tuple(
            SourceSummary(
                name=item.publisher,
                description=f"Official release {item.release_id}: {item.source_url}",
            )
            for item in self._ledger.list_releases()
            )
        return self._source_cache

    def _project_sqlite(self, path: Path) -> tuple[PlanningConfiguration, ...]:
        return self._query_sqlite(
            path, "1 = 1", (), "opportunity_id", None, 0
        )

    def _query_sqlite(
        self,
        path: Path,
        where: str,
        parameters: tuple[object, ...],
        order_by: str,
        limit: int | None,
        offset: int,
    ) -> tuple[PlanningConfiguration, ...]:
        sources = self._sources()
        with self._connect(path) as connection:
            suffix = "" if limit is None else " LIMIT ? OFFSET ?"
            query_parameters = (
                parameters if limit is None else (*parameters, limit, offset)
            )
            outer_order = (
                order_by.replace("opportunity.", "")
                .replace("generation.", "")
                .replace("vehicle.", "")
            )
            rows = connection.execute(
                f"""WITH page AS MATERIALIZED (
                    {self._base_sql(where)}
                    ORDER BY {order_by}{suffix}
                )
                SELECT page.*, (
                    SELECT MIN(cohort.registration_cohort_year)
                    FROM opportunity_input input
                    JOIN cohort_estimate cohort ON cohort.cohort_id = input.cohort_id
                    WHERE input.opportunity_id = page.opportunity_id
                ) AS first_cohort_year
                FROM page ORDER BY {outer_order}""",
                query_parameters,
            ).fetchall()
        records = []
        for row in rows:
            downside, base, upside = (
                _units(Decimal(row[name])) for name in ("p10", "p50", "p90")
            )
            exposure = _units(Decimal(row["active_fleet_p50"]))
            confidence_band = ConfidenceBand(row["confidence"])
            reason_codes = tuple(json.loads(row["reason_codes"]))
            identity_kind = GenerationIdentityKind(row["identity_kind"])
            identity_reasons = tuple(json.loads(row["confidence_reasons"]))
            model_year_demand = (
                ModelYearDemand(
                    configuration_id=row["opportunity_id"],
                    model_year=row["first_cohort_year"],
                    forecast_horizon=row["horizon_year"],
                    demand=DemandRange(downside, base, upside),
                    evidence_status=EvidenceStatus.VALIDATED,
                    data_version=self.snapshot_id,
                    sources=sources,
                ),
            )
            records.append(
                PlanningConfiguration(
                    configuration_id=row["opportunity_id"],
                    sku=None,
                    part_family=None,
                    market=row["geography"],
                    brand=row["make"],
                    model=row["model"],
                    model_year_start=date.fromisoformat(row["start_month"]).year,
                    model_year_end=(
                        date.fromisoformat(row["end_month"]).year
                        if row["end_month"] is not None
                        else row["first_cohort_year"]
                    ),
                    generation=row["display_name"],
                    facelift=row["facelift"],
                    body_style=row["body_style"] or "Not evidenced",
                    drive_side=None,
                    equipment=Equipment(None, None, None, None, None),
                    forecast_horizon=row["horizon_year"],
                    demand=DemandRange(downside, base, upside),
                    vehicle_exposure_units=exposure,
                    replacement_rate=(min(1.0, base / exposure) if exposure else 0.0),
                    identity_confidence=_identity_confidence_values(
                        identity_kind, identity_reasons
                    ),
                    data_quality_confidence=_confidence(confidence_band, reason_codes),
                    evidence_status=EvidenceStatus.VALIDATED,
                    sources=sources,
                    updated_at=self.manifest.built_at,
                    data_version=self.snapshot_id,
                    model_year_demand=model_year_demand,
                    generation_id=row["generation_id"],
                    generation_identity_kind=identity_kind.value,
                    year_semantics="registration_cohort_year_range",
                    assumption_ids=tuple(json.loads(row["assumption_ids"])),
                    reason_codes=reason_codes,
                    evidence_ids=tuple(json.loads(row["evidence_ids"])),
                    method_versions=self.versions,
                )
            )
        return tuple(records)

    def _search_sqlite(self, path: Path, query: PlannerQuery) -> PlannerPage:
        clauses: list[str] = []
        parameters: list[object] = []
        for column, values in (
            ("opportunity.geography", query.markets),
            ("opportunity.horizon_year", query.horizons),
            ("vehicle.make", query.brands),
            ("vehicle.model", query.models),
        ):
            if values:
                clauses.append(f"{column} IN ({', '.join('?' for _ in values)})")
                parameters.extend(values)
        if query.evidence and EvidenceStatus.VALIDATED not in query.evidence:
            return PlannerPage(
                items=(),
                total=0,
                page=query.page,
                page_size=query.page_size,
                pages=0,
                summary=PlannerSummary(0, 0, 0, 0),
            )
        where = " AND ".join(clauses) if clauses else "1 = 1"
        with self._connect(path) as connection:
            summary = connection.execute(
                f"""WITH filtered AS ({self._base_sql(where)})
                SELECT COUNT(*) candidate_count,
                    COALESCE(SUM(CAST(ROUND(CAST(p10 AS NUMERIC), 0) AS INTEGER)), 0)
                        downside_units,
                    COALESCE(SUM(CAST(ROUND(CAST(p50 AS NUMERIC), 0) AS INTEGER)), 0)
                        base_units,
                    COALESCE(SUM(CAST(ROUND(CAST(p90 AS NUMERIC), 0) AS INTEGER)), 0)
                        upside_units
                FROM filtered""",
                tuple(parameters),
            ).fetchone()
        sort_columns = {
            SortField.BASE_DEMAND: "CAST(p50 AS NUMERIC)",
            SortField.DOWNSIDE_DEMAND: "CAST(p10 AS NUMERIC)",
            SortField.UPSIDE_DEMAND: "CAST(p90 AS NUMERIC)",
            SortField.BRAND: "LOWER(make)",
            SortField.MODEL: "LOWER(model)",
            SortField.IDENTITY_CONFIDENCE: (
                "CASE identity_kind WHEN 'estimated' THEN 1 ELSE 3 END"
            ),
            SortField.DATA_QUALITY_CONFIDENCE: (
                "CASE confidence WHEN 'very_low' THEN 1 WHEN 'low' THEN 1 "
                "WHEN 'medium' THEN 2 ELSE 3 END"
            ),
        }
        direction = "DESC" if query.direction is SortDirection.DESC else "ASC"
        order_by = f"{sort_columns[query.sort]} {direction}, opportunity_id ASC"
        offset = (query.page - 1) * query.page_size
        items = self._query_sqlite(
            path, where, tuple(parameters), order_by, query.page_size, offset
        )
        total = int(summary["candidate_count"])
        return PlannerPage(
            items=items,
            total=total,
            page=query.page,
            page_size=query.page_size,
            pages=ceil(total / query.page_size),
            summary=PlannerSummary(
                candidate_count=total,
                downside_units=int(summary["downside_units"]),
                base_units=int(summary["base_units"]),
                upside_units=int(summary["upside_units"]),
            ),
        )

    @staticmethod
    def _base_sql(where: str) -> str:
        return f"""SELECT opportunity.*, generation.display_name,
            generation.start_month, generation.end_month,
            generation.identity_kind, generation.body_style,
            generation.facelift, generation.confidence_reasons,
            generation.evidence_ids, vehicle.make, vehicle.model
            FROM opportunity_estimate opportunity
            JOIN generation_entry generation
                ON generation.generation_id = opportunity.generation_id
            JOIN canonical_vehicle vehicle
                ON vehicle.vehicle_id = opportunity.canonical_vehicle_id
            WHERE {where}"""

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro", uri=True
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    def _project_objects(self) -> tuple[PlanningConfiguration, ...]:
        vehicles = {item.vehicle_id: item for item in self._ledger.list_vehicles()}
        generations = {
            item.generation_id: item for item in self._ledger.list_generations()
        }
        cohorts = {item.cohort_id: item for item in self._ledger.list_cohort_estimates()}
        sources = self._sources()
        records = []
        for opportunity in self._ledger.list_opportunity_estimates():
            generation = generations[opportunity.generation_id]
            vehicle = vehicles[generation.canonical_vehicle_id]
            inputs = tuple(cohorts[item] for item in opportunity.input_cohort_ids)
            confidence = _confidence(opportunity.confidence, opportunity.reason_codes)
            downside = _units(opportunity.p10)
            base = _units(opportunity.p50)
            upside = _units(opportunity.p90)
            first_cohort_year = min(
                item.registration_cohort_year for item in inputs
            )
            model_year_demand = (
                ModelYearDemand(
                    configuration_id=opportunity.opportunity_id,
                    model_year=first_cohort_year,
                    forecast_horizon=opportunity.horizon_year,
                    demand=DemandRange(downside, base, upside),
                    evidence_status=EvidenceStatus.VALIDATED,
                    data_version=self.snapshot_id,
                    sources=sources,
                ),
            )
            exposure = _units(opportunity.active_fleet_p50)
            evidence_ids = tuple(
                dict.fromkeys(
                    evidence_id
                    for cohort in inputs
                    for evidence_id in cohort.input_observation_ids
                )
            )
            end_year = (
                generation.end_month.year
                if generation.end_month is not None
                else max(item.registration_cohort_year for item in inputs)
            )
            records.append(
                PlanningConfiguration(
                    configuration_id=opportunity.opportunity_id,
                    sku=None,
                    part_family=None,
                    market=opportunity.geography,
                    brand=vehicle.make,
                    model=vehicle.model,
                    model_year_start=generation.start_month.year,
                    model_year_end=end_year,
                    generation=generation.display_name,
                    facelift=generation.facelift,
                    body_style=generation.body_style or "Not evidenced",
                    drive_side=None,
                    equipment=Equipment(None, None, None, None, None),
                    forecast_horizon=opportunity.horizon_year,
                    demand=DemandRange(downside, base, upside),
                    vehicle_exposure_units=exposure,
                    replacement_rate=(min(1.0, base / exposure) if exposure else 0.0),
                    identity_confidence=_identity_confidence(generation),
                    data_quality_confidence=confidence,
                    evidence_status=EvidenceStatus.VALIDATED,
                    sources=sources,
                    updated_at=self.manifest.built_at,
                    data_version=self.snapshot_id,
                    model_year_demand=model_year_demand,
                    generation_id=generation.generation_id,
                    generation_identity_kind=generation.identity_kind.value,
                    year_semantics="registration_cohort_year_range",
                    assumption_ids=opportunity.assumption_ids,
                    reason_codes=opportunity.reason_codes,
                    evidence_ids=evidence_ids,
                    method_versions=self.versions,
                )
            )
        return tuple(sorted(records, key=lambda item: item.configuration_id))


def _units(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _confidence(band: ConfidenceBand, reasons: tuple[str, ...]) -> Confidence:
    level = {
        ConfidenceBand.VERY_LOW: ConfidenceLevel.LOW,
        ConfidenceBand.LOW: ConfidenceLevel.LOW,
        ConfidenceBand.MEDIUM: ConfidenceLevel.MEDIUM,
        ConfidenceBand.HIGH: ConfidenceLevel.HIGH,
    }[band]
    return Confidence(level, "; ".join(reasons))


def _identity_confidence(generation) -> Confidence:
    return _identity_confidence_values(
        generation.identity_kind, generation.confidence_reasons
    )


def _identity_confidence_values(
    identity_kind: GenerationIdentityKind, reasons: tuple[str, ...]
) -> Confidence:
    level = (
        ConfidenceLevel.LOW
        if identity_kind is GenerationIdentityKind.ESTIMATED
        else ConfidenceLevel.HIGH
    )
    return Confidence(level, "; ".join(reasons))
