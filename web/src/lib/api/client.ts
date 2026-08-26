import type { components, operations } from './schema'


type Problem = components['schemas']['ProblemResponse']
type FieldError = components['schemas']['FieldError']
type PlannerOptions = components['schemas']['PlannerOptionsResponse']
type PlannerPage = components['schemas']['PlannerPageResponse']
type PlanningConfiguration = components['schemas']['PlanningConfigurationResponse']
type ApiQuery = NonNullable<
  operations['configurations_api_v1_planner_configurations_get']['parameters']['query']
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

function isProblem(value: unknown): value is Problem {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Partial<Problem>
  return (
    typeof candidate.code === 'string' &&
    typeof candidate.message === 'string' &&
    typeof candidate.correlation_id === 'string' &&
    Array.isArray(candidate.field_errors)
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
  constructor(
    private readonly fetcher: typeof fetch = fetch,
    private readonly baseUrl = '',
  ) {}

  async options(): Promise<PlannerOptions> {
    return this.request<PlannerOptions>('/api/v1/planner/options')
  }

  async configurations(query: PlannerConfigurationsQuery): Promise<PlannerPage> {
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
    return this.request<PlannerPage>(`/api/v1/planner/configurations${suffix}`)
  }

  async configuration(configurationId: string): Promise<PlanningConfiguration> {
    const encoded = encodeURIComponent(configurationId)
    return this.request<PlanningConfiguration>(
      `/api/v1/planner/configurations/${encoded}`,
    )
  }

  private async request<T>(path: string): Promise<T> {
    const response = await this.fetcher(`${this.baseUrl}${path}`, {
      headers: { Accept: 'application/json' },
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

export const plannerApi = new PlannerApiClient()
