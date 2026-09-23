import type { components, operations } from './schema'


type Problem = components['schemas']['ProblemResponse']
type FieldError = components['schemas']['FieldError']
type PlannerOptions = components['schemas']['PlannerOptionsResponse']
type PlannerPage = components['schemas']['PlannerPageResponse']
type PlanningConfiguration = components['schemas']['PlanningConfigurationResponse']
type OpportunityPage = components['schemas']['OpportunityPageResponse']
type OpportunityDrillDown = components['schemas']['OpportunityDrillDownResponse']
type OpportunityContribution = components['schemas']['OpportunityContributionResponse']
type OpportunityFleetEstimate = components['schemas']['OpportunityFleetEstimateResponse']
type OpportunityGroupBy = components['schemas']['OpportunityGroupBy']
type ProductionCoverage = components['schemas']['ProductionCoverageResponse']
type ProductionCoverageRequest = components['schemas']['ProductionCoverageRequest']
type DeleteCoverageResponse = components['schemas']['DeleteCoverageResponse']
type EvidenceSummary = components['schemas']['EvidenceSummaryResponse']
type EvidenceObservationPage = components['schemas']['EvidenceObservationPageResponse']
type EvidenceMeasure = components['schemas']['Measure']
type EvidenceMappingStatus = components['schemas']['MappingStatus']
type RegistrationSummary = components['schemas']['RegistrationSummaryResponse']
type RegistrationPage = components['schemas']['RegistrationPageResponse']
type CompletenessReport = components['schemas']['CompletenessResponse']
type VehicleForecastOptions = components['schemas']['VehicleForecastOptionsResponse']
type VehicleForecast = components['schemas']['VehicleForecastResponse']
type ApiQuery = NonNullable<
  operations['configurations_api_v1_planner_configurations_get']['parameters']['query']
>
type RegistrationApiQuery = NonNullable<
  operations['ranking_api_v1_registrations_ranking_get']['parameters']['query']
>

export interface PlannerConfigurationsQuery {
  markets?: ApiQuery['market']
  horizons?: ApiQuery['horizon']
  brands?: ApiQuery['brand']
  models?: ApiQuery['model']
  evidence?: ApiQuery['evidence']
  sort?: ApiQuery['sort']
  direction?: ApiQuery['direction']
  page?: ApiQuery['page']
  pageSize?: ApiQuery['page_size']
}

export type OpportunitySort = 'score' | 'demand' | 'vehicle'

export interface OpportunitiesQuery {
  groupBy: OpportunityGroupBy
  markets?: string[]
  horizons?: number[]
  text?: string
  sort?: OpportunitySort
  page?: number
  pageSize?: number
}

export interface EvidenceObservationsQuery {
  releaseId?: string
  geography?: string
  measure?: EvidenceMeasure
  mappingStatus?: EvidenceMappingStatus
  search?: string
  observationYear?: number
  yearSemantics?: 'observation_year' | 'registration_cohort_year' | 'manufacture_year' | 'model_year'
  page?: number
  pageSize?: number
}

export interface RegistrationRankingQuery {
  geography?: RegistrationApiQuery['geography']
  year?: RegistrationApiQuery['year']
  search?: RegistrationApiQuery['search']
  page?: RegistrationApiQuery['page']
  pageSize?: RegistrationApiQuery['page_size']
}

export interface VehicleForecastOptionsQuery {
  search?: string
  brand?: string
  model?: string
  includeAllBrands?: boolean
}

export interface VehicleForecastQuery {
  brand: string
  model: string
  horizon: number
  year?: number
  generation?: string
}

export class ApiProblem extends Error {
  readonly code: string
  readonly correlationId: string | null
  readonly fieldErrors: FieldError[]
  readonly status: number

  constructor(
    code: string,
    message: string,
    correlationId: string | null,
    fieldErrors: FieldError[],
    status: number,
  ) {
    super(message)
    this.name = 'ApiProblem'
    this.code = code
    this.correlationId = correlationId
    this.fieldErrors = fieldErrors
    this.status = status
  }
}

function isFieldError(value: unknown): value is FieldError {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return typeof candidate.field === 'string' && typeof candidate.message === 'string'
}

function isProblem(value: unknown): value is Problem {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  const fieldErrors = candidate.field_errors
  return (
    typeof candidate.code === 'string' &&
    typeof candidate.message === 'string' &&
    typeof candidate.correlation_id === 'string' &&
    (fieldErrors === undefined ||
      (Array.isArray(fieldErrors) && fieldErrors.every(isFieldError)))
  )
}

function appendMany(
  parameters: URLSearchParams,
  name: string,
  values: readonly (string | number)[] | null | undefined,
): void {
  for (const value of values ?? []) parameters.append(name, String(value))
}

export class PlannerApiClient {
  private readonly fetcher: typeof fetch
  private readonly baseUrl: string

  constructor(
    fetcher?: typeof fetch,
    baseUrl = '',
  ) {
    this.fetcher = fetcher ?? ((input, init) => globalThis.fetch(input, init))
    this.baseUrl = baseUrl
  }

  async options(): Promise<PlannerOptions> {
    return this.request<PlannerOptions>('/api/v1/planner/options')
  }

  async vehicleForecastOptions(query: VehicleForecastOptionsQuery): Promise<VehicleForecastOptions> {
    const parameters = new URLSearchParams()
    if (query.search) parameters.set('search', query.search)
    if (query.brand) parameters.set('brand', query.brand)
    if (query.model) parameters.set('model', query.model)
    if (query.includeAllBrands) parameters.set('include_all_brands', 'true')
    const suffix = parameters.size > 0 ? `?${parameters.toString()}` : ''
    return this.request<VehicleForecastOptions>(`/api/v1/vehicle-forecasts/options${suffix}`)
  }

  async vehicleForecast(query: VehicleForecastQuery): Promise<VehicleForecast> {
    const parameters = new URLSearchParams({
      brand: query.brand,
      model: query.model,
      horizon: String(query.horizon),
    })
    if (query.year !== undefined) parameters.set('year', String(query.year))
    if (query.generation) parameters.set('generation', query.generation)
    return this.request<VehicleForecast>(`/api/v1/vehicle-forecasts?${parameters.toString()}`)
  }

  async configurations(
    query: PlannerConfigurationsQuery,
    signal?: AbortSignal,
  ): Promise<PlannerPage> {
    const parameters = new URLSearchParams()
    appendMany(parameters, 'market', query.markets)
    appendMany(parameters, 'horizon', query.horizons)
    appendMany(parameters, 'brand', query.brands)
    appendMany(parameters, 'model', query.models)
    appendMany(parameters, 'evidence', query.evidence)
    if (query.sort !== undefined) parameters.set('sort', query.sort)
    if (query.direction !== undefined) parameters.set('direction', query.direction)
    if (query.page !== undefined) parameters.set('page', String(query.page))
    if (query.pageSize !== undefined) parameters.set('page_size', String(query.pageSize))
    const suffix = parameters.size > 0 ? `?${parameters.toString()}` : ''
    return this.request<PlannerPage>(
      `/api/v1/planner/configurations${suffix}`,
      { signal },
    )
  }

  async configuration(configurationId: string): Promise<PlanningConfiguration> {
    const encoded = encodeURIComponent(configurationId)
    return this.request<PlanningConfiguration>(
      `/api/v1/planner/configurations/${encoded}`,
    )
  }

  async opportunities(query: OpportunitiesQuery, signal?: AbortSignal): Promise<OpportunityPage> {
    const parameters = opportunityParameters(query)
    return this.request<OpportunityPage>(`/api/v1/opportunities?${parameters}`, { signal })
  }

  async opportunity(
    groupId: string,
    query: OpportunitiesQuery,
    signal?: AbortSignal,
  ): Promise<OpportunityPage['items'][number]> {
    const parameters = opportunityParameters(query)
    return this.request<OpportunityPage['items'][number]>(
      `/api/v1/opportunities/${encodeURIComponent(groupId)}?${parameters}`,
      { signal },
    )
  }

  async opportunityContributions(
    groupId: string,
    query: OpportunitiesQuery,
    signal?: AbortSignal,
  ): Promise<OpportunityContribution[]> {
    const parameters = opportunityParameters(query)
    return this.request<OpportunityContribution[]>(
      '/api/v1/opportunities/' + encodeURIComponent(groupId) + '/contributions?' + parameters.toString(),
      { signal },
    )
  }

  async opportunityConfigurations(
    groupId: string,
    query: OpportunitiesQuery,
  ): Promise<OpportunityDrillDown[]> {
    const parameters = opportunityParameters(query)
    return this.request<OpportunityDrillDown[]>(
      `/api/v1/opportunities/${encodeURIComponent(groupId)}/configurations?${parameters}`,
    )
  }

  async opportunityFleet(
    groupId: string,
    query: OpportunitiesQuery,
    signal?: AbortSignal,
  ): Promise<OpportunityFleetEstimate[]> {
    const parameters = opportunityParameters(query)
    return this.request<OpportunityFleetEstimate[]>(
      `/api/v1/opportunities/${encodeURIComponent(groupId)}/fleet?${parameters}`,
      { signal },
    )
  }

  async coverage(): Promise<ProductionCoverage[]> {
    return this.request<ProductionCoverage[]>('/api/v1/production-coverage')
  }

  async createCoverage(payload: ProductionCoverageRequest): Promise<ProductionCoverage> {
    return this.request<ProductionCoverage>('/api/v1/production-coverage', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  }

  async updateCoverage(
    coverageId: string,
    payload: ProductionCoverageRequest,
  ): Promise<ProductionCoverage> {
    return this.request<ProductionCoverage>(
      `/api/v1/production-coverage/${encodeURIComponent(coverageId)}`,
      { method: 'PUT', body: JSON.stringify(payload) },
    )
  }

  async deleteCoverage(coverageId: string): Promise<DeleteCoverageResponse> {
    return this.request<DeleteCoverageResponse>(
      `/api/v1/production-coverage/${encodeURIComponent(coverageId)}`,
      { method: 'DELETE' },
    )
  }

  async evidenceSummary(): Promise<EvidenceSummary> {
    return this.request<EvidenceSummary>('/api/v1/evidence/summary')
  }

  async evidenceObservations(query: EvidenceObservationsQuery): Promise<EvidenceObservationPage> {
    const parameters = new URLSearchParams()
    if (query.releaseId) parameters.set('release_id', query.releaseId)
    if (query.geography) parameters.set('geography', query.geography)
    if (query.measure) parameters.set('measure', query.measure)
    if (query.mappingStatus) parameters.set('mapping_status', query.mappingStatus)
    if (query.search) parameters.set('search', query.search)
    if (query.observationYear !== undefined) {
      parameters.set('observation_year', String(query.observationYear))
    }
    if (query.yearSemantics) parameters.set('year_semantics', query.yearSemantics)
    if (query.page !== undefined) parameters.set('page', String(query.page))
    if (query.pageSize !== undefined) parameters.set('page_size', String(query.pageSize))
    return this.request<EvidenceObservationPage>(
      `/api/v1/evidence/observations?${parameters.toString()}`,
    )
  }

  async registrationSummary(): Promise<RegistrationSummary> {
    return this.request<RegistrationSummary>('/api/v1/registrations/summary')
  }

  async registrationRanking(query: RegistrationRankingQuery): Promise<RegistrationPage> {
    const parameters = new URLSearchParams()
    if (query.geography !== undefined) parameters.set('geography', query.geography)
    if (query.year !== undefined) parameters.set('year', String(query.year))
    if (query.search) parameters.set('search', query.search)
    if (query.page !== undefined) parameters.set('page', String(query.page))
    if (query.pageSize !== undefined) parameters.set('page_size', String(query.pageSize))
    const suffix = parameters.size > 0 ? `?${parameters.toString()}` : ''
    return this.request<RegistrationPage>(`/api/v1/registrations/ranking${suffix}`)
  }

  async completeness(): Promise<CompletenessReport> {
    return this.request<CompletenessReport>('/api/completeness')
  }

  async mlExport(cutoff: string, token: string): Promise<Blob> {
    const response = await this.fetcher(
      `${this.baseUrl}/api/exports/ml.csv?cutoff=${encodeURIComponent(cutoff)}`,
      {
        headers: {
          Accept: 'text/csv',
          'X-ICOR-Export-Token': token,
        },
      },
    )
    if (response.ok) return response.blob()
    let body: unknown
    try { body = await response.json() } catch { body = null }
    if (isProblem(body)) {
      throw new ApiProblem(
        body.code,
        body.message,
        body.correlation_id,
        body.field_errors ?? [],
        response.status,
      )
    }
    throw new ApiProblem('invalid_response', 'The export service returned an invalid response.', null, [], response.status)
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const headers = {
      Accept: 'application/json',
      ...(init?.body !== undefined && { 'Content-Type': 'application/json' }),
    }
    const response = await this.fetcher(`${this.baseUrl}${path}`, {
      ...init,
      headers,
    })
    if (response.ok) return (await response.json()) as T

    let body: unknown
    try {
      body = await response.json()
    } catch {
      body = null
    }
    if (isProblem(body)) {
      throw new ApiProblem(
        body.code,
        body.message,
        body.correlation_id,
        body.field_errors ?? [],
        response.status,
      )
    }
    throw new ApiProblem(
      'invalid_response',
      'The planner service returned an invalid error response.',
      response.headers.get('X-Correlation-ID'),
      [],
      response.status,
    )
  }
}

function opportunityParameters(query: OpportunitiesQuery): URLSearchParams {
  const parameters = new URLSearchParams({ group_by: query.groupBy })
  appendMany(parameters, 'market', query.markets)
  appendMany(parameters, 'horizon', query.horizons)
  if (query.text) parameters.set('q', query.text)
  if (query.sort && query.sort !== 'score') parameters.set('sort', query.sort)
  if (query.page !== undefined) parameters.set('page', String(query.page))
  if (query.pageSize !== undefined) parameters.set('page_size', String(query.pageSize))
  return parameters
}

export const plannerApi = new PlannerApiClient()
