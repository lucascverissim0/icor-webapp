// @vitest-environment node

import { describe, expect, it, vi } from 'vitest'

import { ApiProblem, PlannerApiClient } from '../src/lib/api/client'


const successPage = {
  items: [],
  total: 0,
  page: 1,
  page_size: 25,
  pages: 0,
  summary: {
    candidate_count: 0,
    downside_units: 0,
    base_units: 0,
    upside_units: 0,
  },
}

describe('PlannerApiClient', () => {
  it('does not bind the browser fetch transport to the client instance', async () => {
    const browserFetch = vi.fn(function (this: unknown) {
      if (this instanceof PlannerApiClient) throw new TypeError('Illegal invocation')
      return Promise.resolve(new Response(JSON.stringify({
        markets: [], horizons: [], brands: [], models: [], evidence_statuses: [],
        scenario: {
          name: 'Demo', description: 'Synthetic demo', evidence_status: 'demonstration',
          data_version: 'demo', updated_at: '2026-08-25T12:00:00Z',
        },
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    })
    vi.stubGlobal('fetch', browserFetch)

    try {
      await expect(new PlannerApiClient().options()).resolves.toMatchObject({ markets: [] })
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it('serializes repeatable filters without inventing values', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify(successPage), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )

    await new PlannerApiClient(fetcher).configurations({
      markets: ['FR', 'DE'],
      brands: ['Aurora Mobility'],
      page: 1,
    })

    const input = fetcher.mock.calls[0]?.[0]
    expect(typeof input).toBe('string')
    if (typeof input !== 'string') throw new TypeError('Expected a string request URL')
    const url = input
    expect(url).toContain('market=FR')
    expect(url).toContain('market=DE')
    expect(url).toContain('brand=Aurora+Mobility')
    expect(url).toContain('page=1')
    expect(url).not.toContain('undefined')
  })

  it('throws the safe typed API problem for non-success responses', async () => {
    const problem = {
      code: 'invalid_request',
      message: 'One or more request values are invalid.',
      correlation_id: 'test-correlation',
      field_errors: [{ field: 'page', message: 'Input should be greater than or equal to 1' }],
    }
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify(problem), {
        status: 422,
        headers: { 'Content-Type': 'application/json' },
      }),
    )

    const request = new PlannerApiClient(fetcher).configurations({ page: 0 })

    await expect(request).rejects.toEqual(
      new ApiProblem(
        'invalid_request',
        'One or more request values are invalid.',
        'test-correlation',
        [{ field: 'page', message: 'Input should be greater than or equal to 1' }],
        422,
      ),
    )
  })

  it('uses a generic problem when an error response is not valid JSON', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response('<html>proxy failure</html>', { status: 502 }),
    )

    const request = new PlannerApiClient(fetcher).options()

    await expect(request).rejects.toMatchObject({
      code: 'invalid_response',
      status: 502,
      correlationId: null,
    })
  })

  it('accepts a contract-valid problem without field errors', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        JSON.stringify({
          code: 'not_found',
          message: 'The requested configuration was not found.',
          correlation_id: 'missing-configuration',
        }),
        { status: 404, headers: { 'Content-Type': 'application/json' } },
      ),
    )

    const request = new PlannerApiClient(fetcher).configuration('missing')

    await expect(request).rejects.toMatchObject({
      code: 'not_found',
      correlationId: 'missing-configuration',
      fieldErrors: [],
      status: 404,
    })
  })

  it('rejects malformed field error entries instead of trusting them', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        JSON.stringify({
          code: 'invalid_request',
          message: 'The request was invalid.',
          correlation_id: 'malformed-problem',
          field_errors: [{ field: 42, message: null }],
        }),
        { status: 422, headers: { 'Content-Type': 'application/json' } },
      ),
    )

    const request = new PlannerApiClient(fetcher).options()

    await expect(request).rejects.toMatchObject({
      code: 'invalid_response',
      correlationId: null,
      fieldErrors: [],
      status: 422,
    })
  })

  it('serializes opportunity filters using the API contract names', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ items: [], summary: { base_units: 0, exact_covered_base_units: 0, high_demand_uncovered_base_units: 0 }, strategy_name: 'demand_readiness', strategy_version: '1', integrity_warnings: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )

    await new PlannerApiClient(fetcher).opportunities({
      groupBy: 'model_year',
      markets: ['FR'],
      horizons: [2030],
    })

    expect(fetcher.mock.calls[0]?.[0]).toBe(
      '/api/v1/opportunities?group_by=model_year&market=FR&horizon=2030',
    )
  })

  it('sends explicit fallback coverage as JSON', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ coverage_id: 'coverage-1' }), {
        status: 201,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    const payload = {
      match_type: 'vehicle_year_fallback' as const,
      configuration_id: null,
      brand: 'Aurora Mobility',
      model: 'A1 Horizon',
      model_year: 2025,
      note: null,
    }

    await new PlannerApiClient(fetcher).createCoverage(payload)

    expect(fetcher.mock.calls[0]?.[0]).toBe('/api/v1/production-coverage')
    expect(fetcher.mock.calls[0]?.[1]).toMatchObject({
      method: 'POST',
      body: JSON.stringify(payload),
      headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    })
  })

  it('deletes coverage using an encoded canonical id', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ coverage_id: 'coverage/one', deleted: true }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )

    await new PlannerApiClient(fetcher).deleteCoverage('coverage/one')

    expect(fetcher.mock.calls[0]?.[0]).toBe('/api/v1/production-coverage/coverage%2Fone')
    expect(fetcher.mock.calls[0]?.[1]).toMatchObject({ method: 'DELETE' })
  })

  it('loads the official registration summary without a prototype route', async () => {
    const payload = {
      snapshot_id: 'snapshot-real-2024', status: 'candidate',
      built_at: '2026-08-27T12:00:00Z', database_sha256: 'a'.repeat(64),
      identity_registry: 'exact-normalized-model-family-v1',
      geographies: ['EU27'], years: [2024], total_registrations: '15',
      model_count: 1, model_year_available: false,
      release_ids: ['eea-co2cars-2024-final-v30-r1'],
    }
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )

    await expect(new PlannerApiClient(fetcher).registrationSummary()).resolves.toEqual(payload)
    expect(fetcher.mock.calls[0]?.[0]).toBe('/api/v1/registrations/summary')
  })

  it('serializes bounded official registration ranking filters', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({
        items: [], total: 0, total_registrations: '0', page: 2, page_size: 10,
        pages: 0, snapshot_id: 'snapshot-real-2024',
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )

    await new PlannerApiClient(fetcher).registrationRanking({
      geography: 'EU27', year: 2024, search: 'Alpha & Beta', page: 2, pageSize: 10,
    })

    expect(fetcher.mock.calls[0]?.[0]).toBe(
      '/api/v1/registrations/ranking?geography=EU27&year=2024&search=Alpha+%26+Beta&page=2&page_size=10',
    )
  })
})
