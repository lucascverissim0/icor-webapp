"""Read-only planner projection over one verified evidence snapshot."""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
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
    PlanningConfiguration,
    SourceSummary,
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

    def list_all(self) -> tuple[PlanningConfiguration, ...]:
        if self._records is None:
            self._records = self._project()
        return self._records

    def get(self, configuration_id: str) -> PlanningConfiguration | None:
        return next(
            (row for row in self.list_all() if row.configuration_id == configuration_id),
            None,
        )

    def list_model_year_demand(self) -> tuple[ModelYearDemand, ...]:
        return tuple(
            demand for row in self.list_all() for demand in row.model_year_demand
        )

    def _project(self) -> tuple[PlanningConfiguration, ...]:
        path = getattr(self._ledger, "path", None)
        if isinstance(path, Path):
            return self._project_sqlite(path)
        return self._project_objects()

    def _sources(self) -> tuple[SourceSummary, ...]:
        return tuple(
            SourceSummary(
                name=item.publisher,
                description=f"Official release {item.release_id}: {item.source_url}",
            )
            for item in self._ledger.list_releases()
        )

    def _project_sqlite(self, path: Path) -> tuple[PlanningConfiguration, ...]:
        sources = self._sources()
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro", uri=True
        )
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """SELECT opportunity.*, generation.display_name,
                generation.start_month, generation.end_month,
                generation.identity_kind, generation.body_style,
                generation.facelift, generation.confidence_reasons,
                generation.evidence_ids, vehicle.make, vehicle.model,
                MIN(cohort.registration_cohort_year) AS first_cohort_year
                FROM opportunity_estimate opportunity
                JOIN generation_entry generation
                    ON generation.generation_id = opportunity.generation_id
                JOIN canonical_vehicle vehicle
                    ON vehicle.vehicle_id = opportunity.canonical_vehicle_id
                JOIN opportunity_input input
                    ON input.opportunity_id = opportunity.opportunity_id
                JOIN cohort_estimate cohort ON cohort.cohort_id = input.cohort_id
                GROUP BY opportunity.opportunity_id
                ORDER BY opportunity.opportunity_id"""
            ).fetchall()
        finally:
            connection.close()
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
