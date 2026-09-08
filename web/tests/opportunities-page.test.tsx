import { QueryClient } from '@tanstack/react-query'
import axe from 'axe-core'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { AppProviders } from '../src/app/providers'
import { OpportunitiesWorkbench } from '../src/features/opportunities/OpportunitiesPage'
import { PlannerApiClient } from '../src/lib/api/client'


export const configuration = {
  configuration_id: 'demo-aurora-a1-camera-fr-2030',
  sku: 'DEMO-AUR-A1-CAM',
  part_family: 'Demo camera acoustic family',
  market: 'FR',
  brand: 'Aurora Mobility',
  model: 'A1 Horizon',
  model_year_start: 2025,
  model_year_end: 2028,
  generation: 'Demo generation A',
  facelift: null,
  body_style: 'Hatchback',
  drive_side: 'left',
  equipment: { camera_adas: true, hud: null, heated: false, acoustic: true, rain_light_sensor: true },
  forecast_horizon: 2030,
  demand: { downside_units: 980, base_units: 1240, upside_units: 1510 },
  vehicle_exposure_units: 62000,
  replacement_rate: 0.02,
  identity_confidence: { level: 'medium', reason: 'Synthetic identity.' },
  data_quality_confidence: { level: 'low', reason: 'Demonstration only.' },
  evidence_status: 'demonstration',
  sources: [{ name: 'Synthetic vehicle scenario', description: 'Fictional scenario.' }],
  updated_at: '2026-08-25T12:00:00Z',
  data_version: 'demo-planner-v1',
} as const

export const opportunities = {
  items: [{
    group_id: 'brand-aurora',
    group_by: 'model_year',
    brand: 'Aurora Mobility',
    model: 'A1 Horizon',
    model_year: 2025,
    generation_name: 'Generation A',
    generation_basis: 'manufacturer_generation_window',
    icor_worked_base_units: 250,
    demand: { downside_units: 1700, base_units: 2150, upside_units: 2620 },
    contributing_configuration_count: 2,
    exact_covered_base_units: 250,
    fallback_covered_base_units: 0,
    uncovered_base_units: 1900,
    coverage_status: 'mixed',
    score: {
      demand_percentile: 1,
      demand_points: 80,
      readiness_ratio: 0.1163,
      readiness_points: 2.326,
      total_points: 82.3,
      strategy_name: 'demand_readiness',
      strategy_version: '1',
      explanation: '80 demand points and 2.326 production-readiness points.',
    },
    evidence_status: 'demonstration',
    data_version: 'demo-planner-v1',
  }],
  summary: {
    base_units: 2150,
    exact_covered_base_units: 250,
    high_demand_uncovered_base_units: 1900,
  },
  strategy_name: 'demand_readiness',
  strategy_version: '1',
  integrity_warnings: [],
  total: 2,
  page: 1,
  page_size: 1,
  pages: 2,
} as const

const registrationSummary = {
  snapshot_id: 'snapshot-real-2025', status: 'verified',
  built_at: '2026-08-27T12:00:00Z', database_sha256: 'a'.repeat(64),
  identity_registry: 'identity-v1', geographies: ['EU27'], years: [2024, 2025],
  total_registrations: '10800000', model_count: 100,
  model_year_available: false, release_ids: ['eea-2025-provisional'],
  availability: [{ geography: 'EU27', year: 2025, status: 'provisional', evidence_kind: 'observed' }],
}

export const drillDown = [{
  configuration,
  model_year_demand: {
    configuration_id: configuration.configuration_id,
    model_year: 2025,
    forecast_horizon: 2030,
    demand: { downside_units: 200, base_units: 250, upside_units: 300 },
    evidence_status: 'demonstration',
    data_version: 'demo-planner-v1',
    sources: configuration.sources,
  },
  coverage_status: 'exact_covered',
}] as const

export const configurationPage = {
  items: [configuration],
  total: 1,
  page: 1,
  page_size: 100,
  pages: 1,
  summary: { candidate_count: 1, downside_units: 980, base_units: 1240, upside_units: 1510 },
} as const

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

export function renderOpportunities(fetcher: typeof fetch) {
  const client = new PlannerApiClient(fetcher)
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const onSearchChange = vi.fn()
  const rendered = render(
    <AppProviders queryClient={queryClient}>
      <OpportunitiesWorkbench
        apiClient={client}
        onSearchChange={onSearchChange}
        search={{ groupBy: 'model_year', page: 1 }}
      />
    </AppProviders>,
  )
  return { ...rendered, onSearchChange, queryClient }
}

export function successFetcher() {
  return vi.fn<typeof fetch>().mockImplementation((input) => {
    const url = typeof input === 'string'
      ? input
      : input instanceof URL ? input.href : input.url
    if (url.includes('/planner/configurations')) return Promise.resolve(json(configurationPage))
    if (url.includes('/registrations/summary')) return Promise.resolve(json(registrationSummary))
    if (url.includes('/opportunities/') && url.includes('/configurations')) return Promise.resolve(json(drillDown))
    if (url.includes('/opportunities')) return Promise.resolve(json(opportunities))
    if (url.includes('/production-coverage')) return Promise.resolve(json([]))
    throw new Error(`Unhandled URL: ${url}`)
  })
}

describe('OpportunitiesWorkbench', () => {
  it('removes grouping and coverage mutations from the verified client release', async () => {
    const client = new PlannerApiClient(successFetcher())
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <AppProviders queryClient={queryClient}>
        <OpportunitiesWorkbench
          apiClient={client}
          clientRelease
          onSearchChange={vi.fn()}
          search={{ groupBy: 'model_year', page: 1 }}
        />
      </AppProviders>,
    )

    expect(await screen.findByText('2,150 replacements')).toBeVisible()
    expect(screen.getByText('Verified identity catalog')).toBeVisible()
    expect(screen.queryByText('Summarize ranking by')).not.toBeInTheDocument()
    expect(screen.queryByText('Manage ICOR worked-model coverage')).not.toBeInTheDocument()
  })

  it('leads with model, year, generation, forecast demand, and transparent scoring', async () => {
    renderOpportunities(successFetcher())

    expect(await screen.findByText('2,150 replacements')).toHaveClass('opportunity-demand__base')
    expect(screen.getByText('Score 82.3')).toHaveAccessibleDescription(
      /80 points from relative demand and 2.3 points from production readiness/i,
    )
    expect(screen.getByRole('heading', { name: /Aurora Mobility.*A1 Horizon.*2025/i })).toBeVisible()
    expect(screen.getByText('Generation A')).toBeVisible()
    expect(screen.getByText(/ICOR has worked on this vehicle-year/i)).toBeVisible()
    expect(screen.getByText(/Demand contribution: 80.0 of 80 points/i)).toBeVisible()
    expect(screen.getByText(/Readiness contribution: 2.3 of 20 points/i)).toBeVisible()
    expect(screen.getByRole('heading', { name: 'How the opportunity score is calculated' })).toBeVisible()
    expect(screen.getByText(/Demand points = demand percentile × 80/i)).toBeVisible()
    expect(screen.getByText(/Readiness points = \(exact units \+ 0.5 × fallback units\)/i)).toBeVisible()
    expect(screen.getByText(/Registration evidence through 2025.*provisional/i)).toBeVisible()
    expect(screen.getByText('Validated snapshot')).toBeVisible()
  })

  it('switches grouping without discarding market intent', async () => {
    const user = userEvent.setup()
    const fetcher = successFetcher()
    const client = new PlannerApiClient(fetcher)
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const onSearchChange = vi.fn()
    render(
      <AppProviders queryClient={queryClient}>
        <OpportunitiesWorkbench apiClient={client} onSearchChange={onSearchChange} search={{ groupBy: 'brand', market: ['FR'], page: 1 }} />
      </AppProviders>,
    )

    await user.click(await screen.findByRole('button', { name: 'Model years' }))

    expect(onSearchChange).toHaveBeenCalledWith({ groupBy: 'model_year', market: ['FR'], page: 1 })
  })

  it('navigates ranking pages and keeps filters', async () => {
    const user = userEvent.setup()
    const fetcher = successFetcher()
    const client = new PlannerApiClient(fetcher)
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const onSearchChange = vi.fn()
    render(<AppProviders queryClient={queryClient}><OpportunitiesWorkbench apiClient={client} onSearchChange={onSearchChange} search={{ groupBy: 'model_year', market: ['FR'], page: 1 }} /></AppProviders>)

    await user.click(await screen.findByRole('button', { name: 'Next page' }))
    expect(onSearchChange).toHaveBeenCalledWith({ groupBy: 'model_year', market: ['FR'], page: 2 })
  })

  it('drills into contributing configuration and model-year demand', async () => {
    const user = userEvent.setup()
    renderOpportunities(successFetcher())

    await user.click(await screen.findByRole('button', { name: /View Aurora Mobility.*details/i }))

    expect(await screen.findByText(/DEMO-AUR-A1-CAM/)).toBeVisible()
    expect(screen.getByText('2025')).toBeVisible()
    expect(screen.getByText('250 replacements')).toBeVisible()
  })

  it('has no automated accessibility violations in the ranked state', async () => {
    const { container } = renderOpportunities(successFetcher())
    await screen.findByText('2,150 replacements')

    expect((await axe.run(container)).violations).toEqual([])
  })
})
