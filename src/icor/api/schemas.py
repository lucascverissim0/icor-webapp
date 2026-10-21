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
    snapshot_ready: bool
    data_version: str


class RowProvenanceResponse(ApiModel):
    """Row-scope provenance: what produced this row, not what built the snapshot."""

    survival_method: str
    hazard_method: str
    forecast_method: str
    uncertainty_method: str


class SnapshotVersionsResponse(ApiModel):
    source_registry: str
    identity_registry: str
    reconciliation_method: str
    confidence_method: str
    estimation_method: str
    survival_method: str
    hazard_method: str
    forecast_method: str
    uncertainty_method: str
    generation_registry: str
    generation_resolver: str


class ScenarioResponse(ApiModel):
    name: str
    description: str
    evidence_status: EvidenceStatus
    data_version: str
    updated_at: datetime
    versions: SnapshotVersionsResponse | None = None


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
    generation_id: str | None
    generation_identity_kind: str | None
    year_semantics: str
    assumption_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    method_versions: SnapshotVersionsResponse | None = None
    row_methods: RowProvenanceResponse | None = None


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


class VehicleOptionResponse(ApiModel):
    brand: str
    model: str


class GenerationOptionResponse(ApiModel):
    key: str
    name: str
    start_year: int
    end_year: int | None
    basis: str
    confidence: str


class EurocodeSearchResponse(ApiModel):
    """A search text that is exactly a Eurocode: the code reading beside the name one."""

    code: str
    #: The code as reviewed vehicles carry it; null when no reviewed vehicle does.
    label: str | None = None
    first_year: int | None = None
    last_year: int | None = None


class VehicleForecastOptionsResponse(ApiModel):
    vehicles: tuple[VehicleOptionResponse, ...]
    years: tuple[int, ...]
    generations: tuple[GenerationOptionResponse, ...]
    horizons: tuple[int, ...]
    brands: tuple[str, ...] = ()
    #: A registration year read out of a free-text search, so `VW Golf 2020`
    #: can preselect 2020 rather than matching it as text and finding nothing.
    search_year: int | None = None
    #: When the search is exactly a Eurocode: that reading, and the vehicles that
    #: may carry the code, listed apart from the name matches in `vehicles`.
    eurocode_search: EurocodeSearchResponse | None = None
    eurocode_vehicles: tuple[VehicleOptionResponse, ...] = ()


class EurocodeShareResponse(ApiModel):
    """One Eurocode's part of a row: units certainly its own, and units it may share."""

    code: str
    label: str
    # Registration years the code can explain; last_year None means still current.
    first_year: int
    last_year: int | None
    drive: str
    certain_units: int
    # Never split: these units belong to this code or to one in shared_with.
    shared_units: int
    shared_with: tuple[str, ...]
    shared_with_uncoded: bool


class EurocodeBreakdownResponse(ApiModel):
    """A row's base units by Eurocode outcome; each unit is counted once."""

    codes: tuple[EurocodeShareResponse, ...]
    single_units: int
    shared_units: int
    outside_units: int
    unassignable_units: int
    unreviewed_units: int


class EurocodeVehicleResponse(ApiModel):
    """One vehicle label whose registrations may carry a code."""

    brand: str
    model: str
    certain_units: int
    shared_units: int


class EurocodeGroupResponse(ApiModel):
    """A Eurocode row's code, vehicles, and the units it may share beyond its certain ones."""

    code: str
    label: str
    make: str
    first_year: int
    last_year: int | None
    drive: str
    shared_units: int
    shared_with: tuple[str, ...]
    shared_with_uncoded: bool
    vehicles: tuple[EurocodeVehicleResponse, ...]


class EurocodeCoverageResponse(ApiModel):
    horizon: int
    total_units: int
    single_units: int
    shared_units: int
    outside_units: int
    unassignable_units: int
    unreviewed_units: int
    shared_with_uncoded_units: int


class EurocodeStatusResponse(ApiModel):
    """Whether Eurocodes are served, against which catalogue, and how much they cover."""

    loaded: bool
    catalogue_sha256: str | None
    decisions_snapshot_id: str | None
    reviewed_identities: int
    orphaned_identities: int
    warnings: tuple[str, ...]
    coverage: tuple[EurocodeCoverageResponse, ...]


class EurocodeDataFleetResponse(ApiModel):
    """A code's cars on the road at the report year (P50 unless named) and at 2028."""

    certain_p10: int
    certain_p50: int
    certain_p90: int
    # Cars the code shares with other codes, never split between them.
    shared_p50: int
    certain_2028: int
    shared_2028: int


class EurocodeDataEvidenceResponse(ApiModel):
    """Cars by the evidence that produced them."""

    # Cars a national register counted on the road.
    stock: int
    registrations: int
    interpolated: int
    forecast: int


class EurocodeDataCodeResponse(ApiModel):
    code: str
    label: str
    families: tuple[str, ...]
    first_year: int
    last_year: int | None
    drive: str
    fleet: EurocodeDataFleetResponse
    shared_with: tuple[str, ...]
    shared_with_uncoded: bool
    evidence: EurocodeDataEvidenceResponse
    # Share of the code's cars (certain and shared) from registrations or a register.
    observed_share: float | None
    # Cars from a register that masks rare model names: they are at least this many.
    lower_bound_p50: int
    markets: int
    observed_markets: tuple[str, ...]
    sources: tuple[str, ...]
    life_years: int
    observed_life_years: int
    years_before_2010: int
    tier: str
    flags: tuple[str, ...]


class EurocodeNoDataResponse(ApiModel):
    """A passenger-car code no car in the model reaches."""

    code: str
    label: str | None
    makes: tuple[str, ...]
    descriptions: tuple[str, ...]
    first_year: int | None
    last_year: int | None
    reviewed: bool
    tier: str


class EurocodeDataOutcomeResponse(ApiModel):
    single: int
    shared: int
    outside: int
    unassignable: int
    unreviewed: int


class EurocodeDataMarketResponse(ApiModel):
    geography: str
    fleet: int
    by_outcome: EurocodeDataOutcomeResponse
    by_evidence: EurocodeDataEvidenceResponse
    has_pre_2010_cohorts: bool
    # Of ``fleet``, the cars from a register that counts only part of them.
    lower_bound: int


class EurocodeDataBandResponse(ApiModel):
    band: str
    official: int | None
    modelled: int


class EurocodeDataCountryGapResponse(ApiModel):
    """A market's official count of cars on the road against the model's, by age band."""

    geography: str
    publisher_geography: str
    year: int
    official: int
    modelled: int
    modelled_share: float | None
    bands: tuple[EurocodeDataBandResponse, ...]


class EurocodeDataTierResponse(ApiModel):
    codes: int
    certain_p50: int


class EurocodeDataNoDataTotalsResponse(ApiModel):
    reviewed: int
    not_reviewed: int
    first_year_from_2000: int


class EurocodeDataTotalsResponse(ApiModel):
    fleet: int
    by_outcome: EurocodeDataOutcomeResponse
    by_evidence: EurocodeDataEvidenceResponse
    tiers: dict[str, EurocodeDataTierResponse]
    no_data: EurocodeDataNoDataTotalsResponse


class EurocodeDataThresholdsResponse(ApiModel):
    tier_a_observed_share: float
    tier_a_markets: int
    tier_b_observed_share: float
    pre_2010_share_cap_c: float


class EurocodeDataOfficialSourceResponse(ApiModel):
    dataset: str
    url: str
    licence: str
    file: str
    sha256: str
    updated: str | None
    missing_markets: tuple[str, ...]


class EurocodeDataReportResponse(ApiModel):
    snapshot_id: str
    year: int
    thresholds: EurocodeDataThresholdsResponse
    totals: EurocodeDataTotalsResponse
    codes: tuple[EurocodeDataCodeResponse, ...]
    no_data: tuple[EurocodeNoDataResponse, ...]
    markets: tuple[EurocodeDataMarketResponse, ...]
    country_gap: tuple[EurocodeDataCountryGapResponse, ...]
    official_source: EurocodeDataOfficialSourceResponse | None


class EurocodeDataCoverageResponse(ApiModel):
    """The data report for the active snapshot, or why there is none to show."""

    available: bool
    message: str | None = None
    report: EurocodeDataReportResponse | None = None


class MarketVehicleForecastResponse(ApiModel):
    code: str
    name: str
    availability: str
    registration_cohort_units: int | None
    cohort_count: int
    active_fleet: DemandRangeResponse | None
    replacements: DemandRangeResponse | None
    # The market's registrations of the selection by Eurocode; None when not loaded.
    eurocodes: EurocodeBreakdownResponse | None = None


class DemandRankResponse(ApiModel):
    percentile: float
    demand_points: float
    rank: int
    population: int
    basis: str


class EuropeanCoverageResponse(ApiModel):
    contributing_markets: tuple[str, ...]
    missing_markets: tuple[str, ...]


class VehicleForecastResponse(ApiModel):
    brand: str
    model: str
    selected_year: int | None
    generation_key: str
    generation_name: str
    generation_basis: str
    generation_confidence: str
    generation_source_url: str | None
    generation_start_year: int
    generation_end_year: int | None
    horizon: int
    included_cohort_years: tuple[int, ...]
    excluded_ambiguous_years: tuple[int, ...]
    excluded_forecast_cohort_years: tuple[int, ...]
    markets: tuple[MarketVehicleForecastResponse, ...]
    survival_method: str
    hazard_method: str
    uncertainty_method: str
    calibration_status: str
    data_version: str
    european_coverage: EuropeanCoverageResponse | None = None
    demand_rank: DemandRankResponse | None = None


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
    # Where the vehicle stands in the whole market and how large that market is.
    # A percentile on its own cannot say whether it was measured against fifty
    # vehicles or fifty thousand, and the score is only meaningful because the
    # population is the market rather than whatever the caller filtered to. A
    # vehicle with no forecast demand is unranked.
    demand_rank: int | None
    demand_population: int
    demand_basis: str
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
    generation_name: str | None
    generation_basis: str | None
    generation_source_url: str | None
    icor_worked_base_units: int
    demand: DemandRangeResponse
    contributing_configuration_count: int
    exact_covered_base_units: int
    fallback_covered_base_units: int
    uncovered_base_units: int
    coverage_status: CoverageStatus
    # True when ICOR has worked on any part of this vehicle: exact or fallback
    # production coverage, or the legacy worked-model catalog. Displayed beside
    # the score and never part of it.
    icor_worked: bool
    score: OpportunityScoreResponse
    evidence_status: EvidenceStatus
    data_version: str
    # None when Eurocodes are not loaded, and for brand rows.
    eurocodes: EurocodeBreakdownResponse | None = None
    # On a Eurocode row (group_by=eurocode): the code it ranks. Its demand is the
    # code's certain units; `shared_units` may belong to it or to `shared_with`.
    eurocode: EurocodeGroupResponse | None = None


class OpportunitySummaryResponse(ApiModel):
    base_units: int
    exact_covered_base_units: int
    high_demand_uncovered_base_units: int


class OpportunityFleetEstimateResponse(ApiModel):
    world_region: str
    forecast_horizon: int
    estimated_fleet_units: int


class OpportunityContributionResponse(ApiModel):
    configuration_id: str
    market: str
    forecast_horizon: int
    generation: str
    body_style: str
    demand: DemandRangeResponse


class OpportunityPageResponse(ApiModel):
    items: tuple[OpportunityRowResponse, ...]
    summary: OpportunitySummaryResponse
    strategy_name: str
    strategy_version: str
    integrity_warnings: tuple[str, ...]
    snapshot_id: str | None = None
    versions: SnapshotVersionsResponse | None = None
    total: int
    page: int
    page_size: int
    pages: int
    available_markets: tuple[str, ...] = ()
    available_horizons: tuple[int, ...] = ()
    # Repeated from the rows so an empty result — a search that matches nothing,
    # which is exactly when the claim matters most — can still say what the
    # scores would have been measured against.
    demand_population: int = 0
    demand_basis: str = ""
    eurocodes_loaded: bool = False
    #: When the search text is exactly a Eurocode, rows that may carry it are
    #: listed beside rows whose name contains the text; this names the code.
    eurocode_search: EurocodeSearchResponse | None = None
    #: On a Eurocode ranking: the view's units by outcome (codes empty), shared
    #: units counted once, so the rows' certain units and these cover the view.
    eurocode_totals: EurocodeBreakdownResponse | None = None


class OpportunityDrillDownResponse(ApiModel):
    configuration: PlanningConfigurationResponse
    model_year_demand: ModelYearDemandResponse
    coverage_status: CoverageStatus


class DeleteCoverageResponse(ApiModel):
    coverage_id: str
    deleted: bool


class CompletenessRecordResponse(ApiModel):
    completeness_id: str
    geography: str
    year: int
    release_count: int
    observation_count: int
    usable_observation_count: int
    assigned_observation_count: int
    canonical_family_count: int
    sourced_generation_count: int
    estimated_generation_count: int
    forecastable_count: int
    evidence_only_count: int
    rejected_record_count: int
    reason_codes: tuple[str, ...]


class CompletenessResponse(ApiModel):
    snapshot_id: str
    built_at: datetime
    versions: SnapshotVersionsResponse
    items: tuple[CompletenessRecordResponse, ...]


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
    publication_status: str = "final"
    validation_warning_count: int = 0
    what_it_proves: str = ""
    limitations: str = ""


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
    observation_year: int
    registration_cohort_year: int | None
    manufacture_year: int | None
    model_year: int | None


class EvidenceObservationPageResponse(ApiModel):
    items: tuple[EvidenceObservationResponse, ...]
    total: int
    page: int
    page_size: int
    pages: int


class RegistrationAvailabilityResponse(ApiModel):
    geography: str
    year: int
    status: str
    evidence_kind: str


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
    availability: tuple[RegistrationAvailabilityResponse, ...] = ()
    versions: SnapshotVersionsResponse | None = None


class RegistrationLabelBreakdownResponse(ApiModel):
    source_make: str
    source_model: str
    registrations: Decimal
    input_observation_count: int
    release_ids: tuple[str, ...]
    source_ids: tuple[str, ...]


class RegistrationRowResponse(ApiModel):
    rank: int
    vehicle_id: str
    make: str
    model: str
    model_year: int
    model_year_basis: str
    generation_name: str | None
    generation_basis: str
    generation_confidence: str
    generation_source_url: str | None
    registrations: Decimal
    status: str
    evidence_confidence: int
    input_observation_count: int
    release_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    publication_status: str = "final"
    evidence_kind: str = "observed"
    label_breakdown: tuple[RegistrationLabelBreakdownResponse, ...] = ()


class RegistrationPageResponse(ApiModel):
    items: tuple[RegistrationRowResponse, ...]
    total: int
    total_registrations: Decimal
    page: int
    page_size: int
    pages: int
    snapshot_id: str
