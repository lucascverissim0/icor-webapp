"""Versioned HTTP schemas for the planner API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from icor.domain.planner import ConfidenceLevel, EvidenceStatus


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class FieldError(ApiModel):
    field: str
    message: str


class ProblemResponse(ApiModel):
    code: str
    message: str
    correlation_id: str
    field_errors: list[FieldError] = Field(default_factory=list)


class HealthResponse(ApiModel):
    status: str
    fixture_ready: bool
    data_version: str


class ScenarioResponse(ApiModel):
    name: str
    description: str
    evidence_status: EvidenceStatus
    data_version: str
    updated_at: datetime


class PlannerOptionsResponse(ApiModel):
    markets: tuple[str, ...]
    horizons: tuple[int, ...]
    brands: tuple[str, ...]
    models: tuple[str, ...]
    evidence_statuses: tuple[EvidenceStatus, ...]
    scenario: ScenarioResponse


class EquipmentResponse(ApiModel):
    camera_adas: bool | None
    hud: bool | None
    heated: bool | None
    acoustic: bool | None
    rain_light_sensor: bool | None


class DemandRangeResponse(ApiModel):
    downside_units: int
    base_units: int
    upside_units: int


class ConfidenceResponse(ApiModel):
    level: ConfidenceLevel
    reason: str


class SourceSummaryResponse(ApiModel):
    name: str
    description: str


class PlanningConfigurationResponse(ApiModel):
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
    equipment: EquipmentResponse
    forecast_horizon: int
    demand: DemandRangeResponse
    vehicle_exposure_units: int
    replacement_rate: float
    identity_confidence: ConfidenceResponse
    data_quality_confidence: ConfidenceResponse
    evidence_status: EvidenceStatus
    sources: tuple[SourceSummaryResponse, ...]
    updated_at: datetime
    data_version: str


class PlannerSummaryResponse(ApiModel):
    candidate_count: int
    downside_units: int
    base_units: int
    upside_units: int


class PlannerPageResponse(ApiModel):
    items: tuple[PlanningConfigurationResponse, ...]
    total: int
    page: int
    page_size: int
    pages: int
    summary: PlannerSummaryResponse
