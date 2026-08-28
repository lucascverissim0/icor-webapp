"""Canonical planner entities and pure query behavior."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import ceil, isfinite

from icor.domain.snapshots import SnapshotVersions


class EvidenceStatus(StrEnum):
    DEMONSTRATION = "demonstration"
    PROTOTYPE = "prototype"
    VALIDATED = "validated"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SortField(StrEnum):
    BASE_DEMAND = "base_demand"
    DOWNSIDE_DEMAND = "downside_demand"
    UPSIDE_DEMAND = "upside_demand"
    BRAND = "brand"
    MODEL = "model"
    IDENTITY_CONFIDENCE = "identity_confidence"
    DATA_QUALITY_CONFIDENCE = "data_quality_confidence"


class SortDirection(StrEnum):
    ASC = "asc"
    DESC = "desc"


@dataclass(frozen=True, slots=True)
class Equipment:
    camera_adas: bool | None
    hud: bool | None
    heated: bool | None
    acoustic: bool | None
    rain_light_sensor: bool | None

    def __post_init__(self) -> None:
        values = (
            self.camera_adas,
            self.hud,
            self.heated,
            self.acoustic,
            self.rain_light_sensor,
        )
        if any(value is not None and type(value) is not bool for value in values):
            raise ValueError("equipment values must be true, false, or unknown")


@dataclass(frozen=True, slots=True)
class DemandRange:
    downside_units: int
    base_units: int
    upside_units: int

    def __post_init__(self) -> None:
        values = (self.downside_units, self.base_units, self.upside_units)
        if not all(type(value) is int for value in values) or not (
            0 <= self.downside_units <= self.base_units <= self.upside_units
        ):
            raise ValueError(
                "demand must use non-negative integer units with downside <= base <= upside"
            )


@dataclass(frozen=True, slots=True)
class Confidence:
    level: ConfidenceLevel
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("confidence reason is required")


@dataclass(frozen=True, slots=True)
class SourceSummary:
    name: str
    description: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.description.strip():
            raise ValueError("source name and description are required")


@dataclass(frozen=True, slots=True)
class ModelYearDemand:
    configuration_id: str
    model_year: int
    forecast_horizon: int
    demand: DemandRange
    evidence_status: EvidenceStatus
    data_version: str
    sources: tuple[SourceSummary, ...]

    def __post_init__(self) -> None:
        if not self.configuration_id.strip() or not self.data_version.strip():
            raise ValueError("model-year demand identity is required")
        if type(self.model_year) is not int or type(self.forecast_horizon) is not int:
            raise ValueError("model-year demand years must be integers")
        if not self.sources:
            raise ValueError("model-year demand requires source metadata")


@dataclass(frozen=True, slots=True)
class PlanningConfiguration:
    configuration_id: str
    sku: str | None
    part_family: str | None
    market: str
    brand: str
    model: str
    model_year_start: int
    model_year_end: int
    generation: str
    facelift: str | None
    body_style: str
    drive_side: str | None
    equipment: Equipment
    forecast_horizon: int
    demand: DemandRange
    vehicle_exposure_units: int
    replacement_rate: float
    identity_confidence: Confidence
    data_quality_confidence: Confidence
    evidence_status: EvidenceStatus
    sources: tuple[SourceSummary, ...]
    updated_at: datetime
    data_version: str
    model_year_demand: tuple[ModelYearDemand, ...]
    generation_id: str | None = None
    generation_identity_kind: str | None = None
    year_semantics: str = "registration_cohort_year"
    assumption_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    method_versions: SnapshotVersions | None = None

    def __post_init__(self) -> None:
        required_text = (
            self.configuration_id,
            self.market,
            self.brand,
            self.model,
            self.generation,
            self.body_style,
            self.data_version,
        )
        if any(not value.strip() for value in required_text):
            raise ValueError("canonical identity fields are required")
        if self.model_year_start > self.model_year_end:
            raise ValueError("model year range must ascend")
        if type(self.forecast_horizon) is not int:
            raise ValueError("forecast horizon must be an integer year")
        if type(self.vehicle_exposure_units) is not int or self.vehicle_exposure_units < 0:
            raise ValueError("vehicle exposure must use non-negative integer units")
        if not isfinite(self.replacement_rate) or not 0 <= self.replacement_rate <= 1:
            raise ValueError("replacement rate must be between zero and one")
        if not self.sources:
            raise ValueError("at least one source summary is required")
        if self.generation_id is not None and not self.generation_id.strip():
            raise ValueError("generation identity cannot be blank")
        if self.generation_identity_kind is not None and not self.generation_identity_kind.strip():
            raise ValueError("generation identity kind cannot be blank")
        if not self.year_semantics.strip():
            raise ValueError("year semantics are required")
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("updated_at must include a timezone")
        identities = {
            (row.configuration_id, row.model_year, row.forecast_horizon)
            for row in self.model_year_demand
        }
        if len(identities) != len(self.model_year_demand):
            raise ValueError("model-year demand identities must be unique")
        if not self.model_year_demand or any(
            row.configuration_id != self.configuration_id
            or not self.model_year_start <= row.model_year <= self.model_year_end
            or row.forecast_horizon != self.forecast_horizon
            or row.evidence_status != self.evidence_status
            or row.data_version != self.data_version
            for row in self.model_year_demand
        ):
            raise ValueError("model-year demand must match canonical identity and applicability")
        totals = DemandRange(
            downside_units=sum(row.demand.downside_units for row in self.model_year_demand),
            base_units=sum(row.demand.base_units for row in self.model_year_demand),
            upside_units=sum(row.demand.upside_units for row in self.model_year_demand),
        )
        if totals != self.demand:
            raise ValueError("model-year demand must reconcile with configuration demand")


@dataclass(frozen=True, slots=True)
class PlannerQuery:
    markets: tuple[str, ...] = ()
    horizons: tuple[int, ...] = ()
    brands: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    evidence: tuple[EvidenceStatus, ...] = ()
    sort: SortField = SortField.BASE_DEMAND
    direction: SortDirection = SortDirection.DESC
    page: int = 1
    page_size: int = 25

    def __post_init__(self) -> None:
        if type(self.page) is not int or type(self.page_size) is not int:
            raise ValueError("pagination requires integer page values")
        if self.page < 1 or not 1 <= self.page_size <= 100:
            raise ValueError("pagination requires page >= 1 and page_size between 1 and 100")


@dataclass(frozen=True, slots=True)
class PlannerSummary:
    candidate_count: int
    downside_units: int
    base_units: int
    upside_units: int


@dataclass(frozen=True, slots=True)
class PlannerPage:
    items: tuple[PlanningConfiguration, ...]
    total: int
    page: int
    page_size: int
    pages: int
    summary: PlannerSummary


def _matches(record: PlanningConfiguration, query: PlannerQuery) -> bool:
    return (
        (not query.markets or record.market in query.markets)
        and (not query.horizons or record.forecast_horizon in query.horizons)
        and (not query.brands or record.brand in query.brands)
        and (not query.models or record.model in query.models)
        and (not query.evidence or record.evidence_status in query.evidence)
    )


_CONFIDENCE_RANK = {
    ConfidenceLevel.LOW: 1,
    ConfidenceLevel.MEDIUM: 2,
    ConfidenceLevel.HIGH: 3,
}

_SORT_KEYS: dict[SortField, Callable[[PlanningConfiguration], int | str]] = {
    SortField.BASE_DEMAND: lambda row: row.demand.base_units,
    SortField.DOWNSIDE_DEMAND: lambda row: row.demand.downside_units,
    SortField.UPSIDE_DEMAND: lambda row: row.demand.upside_units,
    SortField.BRAND: lambda row: row.brand.casefold(),
    SortField.MODEL: lambda row: row.model.casefold(),
    SortField.IDENTITY_CONFIDENCE: lambda row: _CONFIDENCE_RANK[
        row.identity_confidence.level
    ],
    SortField.DATA_QUALITY_CONFIDENCE: lambda row: _CONFIDENCE_RANK[
        row.data_quality_confidence.level
    ],
}


def filter_sort_paginate(
    records: tuple[PlanningConfiguration, ...], query: PlannerQuery
) -> PlannerPage:
    """Apply exact canonical filters and return a stable immutable page."""
    matches = [record for record in records if _matches(record, query)]
    summary = PlannerSummary(
        candidate_count=len(matches),
        downside_units=sum(record.demand.downside_units for record in matches),
        base_units=sum(record.demand.base_units for record in matches),
        upside_units=sum(record.demand.upside_units for record in matches),
    )

    matches.sort(key=lambda row: row.configuration_id)
    matches.sort(
        key=_SORT_KEYS[query.sort],
        reverse=query.direction is SortDirection.DESC,
    )
    start = (query.page - 1) * query.page_size
    end = start + query.page_size
    return PlannerPage(
        items=tuple(matches[start:end]),
        total=len(matches),
        page=query.page,
        page_size=query.page_size,
        pages=ceil(len(matches) / query.page_size),
        summary=summary,
    )
