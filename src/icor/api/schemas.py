"""Versioned HTTP schemas for the planner API."""

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from icor.domain.opportunities import CoverageMatchType, CoverageStatus
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


class ModelYearDemandResponse(ApiModel):
    configuration_id: str
    model_year: int
    forecast_horizon: int
    demand: DemandRangeResponse
    evidence_status: EvidenceStatus
    data_version: str
    sources: tuple[SourceSummaryResponse, ...]


class ProductionCoverageRequest(ApiModel):
    match_type: CoverageMatchType
    configuration_id: str | None
    brand: str | None
    model: str | None
    model_year: int
    note: str | None = Field(default=None, max_length=500)


class ProductionCoverageResponse(ApiModel):
    coverage_id: str
    match_type: CoverageMatchType
    configuration_id: str | None
    brand: str
    model: str
    model_year: int
    sku: str | None
    note: str | None
    created_at: datetime
    updated_at: datetime


class OpportunityScoreResponse(ApiModel):
    demand_percentile: float
    demand_points: float
    readiness_ratio: float
    readiness_points: float
    total_points: float
    strategy_name: str
    strategy_version: str
    explanation: str

    @field_serializer("total_points")
    def serialize_total_points(self, value: float) -> float:
        return round(value, 1)


class OpportunityRowResponse(ApiModel):
    group_id: str
    group_by: str
    brand: str
    model: str | None
    model_year: int | None
    demand: DemandRangeResponse
    contributing_configuration_count: int
    exact_covered_base_units: int
    fallback_covered_base_units: int
    uncovered_base_units: int
    coverage_status: CoverageStatus
    score: OpportunityScoreResponse
    evidence_status: EvidenceStatus
    data_version: str


class OpportunitySummaryResponse(ApiModel):
    base_units: int
    exact_covered_base_units: int
    high_demand_uncovered_base_units: int


class OpportunityPageResponse(ApiModel):
    items: tuple[OpportunityRowResponse, ...]
    summary: OpportunitySummaryResponse
    strategy_name: str
    strategy_version: str
    integrity_warnings: tuple[str, ...]


class OpportunityDrillDownResponse(ApiModel):
    configuration: PlanningConfigurationResponse
    model_year_demand: ModelYearDemandResponse
    coverage_status: CoverageStatus


class DeleteCoverageResponse(ApiModel):
    coverage_id: str
    deleted: bool


class SnapshotVersionsResponse(ApiModel):
    source_registry: str
    identity_registry: str
    reconciliation_method: str
    confidence_method: str
    estimation_method: str
    survival_method: str
    hazard_method: str
    forecast_method: str


class EvidenceReleaseSummaryResponse(ApiModel):
    release_id: str
    source_id: str
    publisher: str
    source_url: str
    terms_url: str
    published_at: datetime
    coverage_start: date
    coverage_end: date
    geography: str
    measure: str
    dependency_group: str
    raw_record_count: int
    accepted_record_count: int
    rejected_record_count: int
    quarantined_record_count: int
    observation_count: int
    total_value: Decimal


class EvidenceSummaryResponse(ApiModel):
    snapshot_id: str
    status: str
    built_at: datetime
    database_sha256: str
    observation_count: int
    published_value_count: int
    warning_count: int
    versions: SnapshotVersionsResponse
    releases: tuple[EvidenceReleaseSummaryResponse, ...]
    mapping_status_counts: dict[str, int]
    geographies: tuple[str, ...]
    measures: tuple[str, ...]


class EvidenceObservationResponse(ApiModel):
    observation_id: str
    release_id: str
    original_row_locator: str
    geography: str
    period_start: date
    period_end: date
    period_precision: str
    measure: str
    value: Decimal
    unit: str
    publication_status: str
    original_make: str
    original_model: str
    original_model_year: str | None
    original_type: str | None
    mapping_status: str
    transformation_notes: tuple[str, ...]
    validation_flags: tuple[str, ...]
    confidence_total: int
    confidence_reasons: tuple[str, ...]


class EvidenceObservationPageResponse(ApiModel):
    items: tuple[EvidenceObservationResponse, ...]
    total: int
    page: int
    page_size: int
    pages: int


class RegistrationSummaryResponse(ApiModel):
    snapshot_id: str
    status: str
    built_at: datetime
    database_sha256: str
    identity_registry: str
    geographies: tuple[str, ...]
    years: tuple[int, ...]
    total_registrations: Decimal
    model_count: int
    model_year_available: bool
    release_ids: tuple[str, ...]


class RegistrationRowResponse(ApiModel):
    rank: int
    vehicle_id: str
    make: str
    model: str
    model_year: None
    registrations: Decimal
    status: str
    evidence_confidence: int
    input_observation_count: int
    release_ids: tuple[str, ...]
    source_ids: tuple[str, ...]


class RegistrationPageResponse(ApiModel):
    items: tuple[RegistrationRowResponse, ...]
    total: int
    total_registrations: Decimal
    page: int
    page_size: int
    pages: int
    snapshot_id: str
