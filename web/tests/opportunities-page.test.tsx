import { QueryClient } from '@tanstack/react-query'
import axe from 'axe-core'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { AppProviders } from '../src/app/providers'
import { OpportunitiesWorkbench } from '../src/features/opportunities/OpportunitiesPage'
import { OpportunityDetailView } from '../src/features/opportunities/OpportunityDetailPage'
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
    generation_source_url: 'https://manufacturer.example/generation-a',
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
  available_markets: ['DE', 'FR', 'GB'],
  available_horizons: [2028, 2031],
} as const

const registrationSummary = {
  snapshot_id: 'snapshot-real-2025', status: 'verified',
  built_at: '2026-08-27T12:00:00Z', database_sha256: 'a'.repeat(64),
  identity_registry: 'identity-v1', geographies: ['EU27'], years: [2024, 2025],
  total_registrations: '10800000', model_count: 100,
  model_year_available: false, release_ids: ['eea-2025-provisional'],
  availability: [{ geography: 'EU27', year: 2025, status: 'provisional', evidence_kind: 'observed' }],
}

const fleetEstimates = [
  { world_region: 'Europe', forecast_horizon: 2028, estimated_fleet_units: 50_000 },
  { world_region: 'North America', forecast_horizon: 2028, estimated_fleet_units: 12_000 },
  { world_region: 'Europe', forecast_horizon: 2031, estimated_fleet_units: 43_000 },
  { world_region: 'North America', forecast_horizon: 2031, estimated_fleet_units: 10_000 },
]

export const contributions = [{
  configuration_id: configuration.configuration_id,
  market: configuration.market,
  forecast_horizon: configuration.forecast_horizon,
  generation: configuration.generation,
  body_style: configuration.body_style,
  demand: { downside_units: 170, base_units: 250, upside_units: 320 },
}]

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
        onOpenDetails={vi.fn()}
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
    if (url.includes('/opportunities/') && url.includes('/contributions')) return Promise.resolve(json(contributions))
    if (url.includes('/opportunities/') && url.includes('/configurations')) return Promise.resolve(json(drillDown))
    if (url.includes('/opportunities/') && url.includes('/fleet')) return Promise.resolve(json(fleetEstimates))
    if (url.includes('/opportunities/brand-aurora')) return Promise.resolve(json(opportunities.items[0]))
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
          onOpenDetails={vi.fn()}
          onSearchChange={vi.fn()}
          search={{ groupBy: 'model_year', page: 1 }}
        />
      </AppProviders>,
    )

    expect(await screen.findByText('2,150 replacements')).toBeVisible()
    expect(screen.getByText('Official vehicle-year evidence')).toBeVisible()
    expect(screen.getByText('Generation A')).toBeVisible()
    expect(screen.queryByText('Summarize ranking by')).not.toBeInTheDocument()
    expect(screen.queryByText('Manage ICOR worked-model coverage')).not.toBeInTheDocument()
  })

  it('leads with model, year, generation, forecast demand, and transparent scoring', async () => {
    renderOpportunities(successFetcher())

    expect(await screen.findByText('2,150 replacements')).toHaveClass('opportunity-demand__base')
    expect(screen.getByText('Score 82.3')).toHaveAccessibleDescription(
      /80.0 demand \+ 2.3 readiness/i,
    )
    expect(screen.getByRole('heading', { name: /Aurora Mobility.*A1 Horizon.*2025/i })).toBeVisible()
    expect(screen.getByText('Generation A')).toBeVisible()
    expect(screen.getByText(/ICOR has worked on this vehicle-year/i)).toBeVisible()
    expect(screen.getByText(/100th demand percentile/i)).toBeVisible()
    expect(screen.getByRole('heading', { name: 'How the opportunity score is calculated' })).toBeVisible()
    expect(screen.getByText(/Demand points = demand percentile × 80/i)).toBeVisible()
    expect(screen.getByText(/Readiness points = \(exact units \+ 0.5 × fallback units\)/i)).toBeVisible()
    expect(screen.getByText(/Registration evidence through 2025.*provisional/i)).toBeVisible()
    expect(screen.getByText('Validated snapshot')).toBeVisible()
    const availableData = screen.getByLabelText('Available opportunity data')
    expect(availableData).toHaveTextContent('2 ranked records available')
    expect(availableData).toHaveTextContent('100 official model labels')
    expect(availableData).toHaveTextContent('10,800,000 registrations represented')
  })

  it('switches grouping without discarding market intent', async () => {
    const user = userEvent.setup()
    const fetcher = successFetcher()
    const client = new PlannerApiClient(fetcher)
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const onSearchChange = vi.fn()
    render(
      <AppProviders queryClient={queryClient}>
        <OpportunitiesWorkbench apiClient={client} onOpenDetails={vi.fn()} onSearchChange={onSearchChange} search={{ groupBy: 'brand', market: ['FR'], page: 1 }} />
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
    render(<AppProviders queryClient={queryClient}><OpportunitiesWorkbench apiClient={client} onOpenDetails={vi.fn()} onSearchChange={onSearchChange} search={{ groupBy: 'model_year', market: ['FR'], page: 1 }} /></AppProviders>)

    await screen.findByRole('button', { name: 'Next page' })
    expect(fetcher.mock.calls.some(([input]) => {
      const url = typeof input === 'string'
        ? input
        : input instanceof URL ? input.href : input.url
      return url.includes('page_size=100')
    })).toBe(true)

    await user.click(screen.getByRole('button', { name: 'Next page' }))
    expect(onSearchChange).toHaveBeenCalledWith({ groupBy: 'model_year', market: ['FR'], page: 2 })

    await user.click(screen.getByRole('button', { name: 'Last page' }))
    expect(onSearchChange).toHaveBeenCalledWith({ groupBy: 'model_year', market: ['FR'], page: 2 })
  })

  it('opens a dedicated detail page for a ranking', async () => {
    const user = userEvent.setup()
    const client = new PlannerApiClient(successFetcher())
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const onOpenDetails = vi.fn()
    render(<AppProviders queryClient={queryClient}><OpportunitiesWorkbench apiClient={client} onOpenDetails={onOpenDetails} onSearchChange={vi.fn()} search={{ groupBy: 'model_year', page: 1 }} /></AppProviders>)

    await user.click(await screen.findByRole('button', { name: /View Aurora Mobility.*details/i }))
    expect(onOpenDetails).toHaveBeenCalledWith('brand-aurora')
    expect(queryClient.getQueryData([
      'opportunities',
      'brand-aurora',
      { groupBy: 'model_year', markets: undefined, horizons: undefined },
    ])).toMatchObject({ group_id: 'brand-aurora', brand: 'Aurora Mobility' })
  })

  it('has no automated accessibility violations in the ranked state', async () => {
    const { container } = renderOpportunities(successFetcher())
    await screen.findByText('2,150 replacements')

    expect((await axe.run(container)).violations).toEqual([])
  })
})

describe('OpportunityDetailView', () => {
  it('explains the selected rank and its contributing forecasts on a dedicated page', async () => {
    const client = new PlannerApiClient(successFetcher())
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <AppProviders queryClient={queryClient}>
        <OpportunityDetailView
          apiClient={client}
          groupId="brand-aurora"
          search={{ groupBy: 'model_year', market: ['FR'], horizon: [2030], page: 1 }}
        />
      </AppProviders>,
    )

    expect(await screen.findByRole('heading', { name: /Aurora Mobility.*A1 Horizon.*2025/i })).toBeVisible()
    expect(screen.getByText('Generation A')).toBeVisible()
    expect(screen.getByText('82.3')).toBeVisible()
    expect(screen.getByText(/2,150/)).toBeVisible()
    expect(await screen.findByRole('heading', { name: 'Estimated active fleet' })).toBeVisible()
    expect(screen.getByText('62,000 vehicles')).toBeVisible()
    expect(screen.getByText('53,000 vehicles')).toBeVisible()
    expect(screen.getAllByText('North America')).toHaveLength(2)
    expect(screen.getByRole('progressbar', { name: 'Demand score' })).toHaveAttribute('value', '80')
    expect(await screen.findByRole('heading', { name: 'FR · 2030' })).toBeVisible()
    expect(screen.getByRole('link', { name: /Back to opportunity ranking/i })).toHaveAttribute(
      'href',
      expect.stringContaining('market=FR'),
    )
  })
})

describe('OpportunitiesWorkbench filters', () => {
  it('offers only the markets the snapshot actually holds', async () => {
    const { findByRole } = renderOpportunities(successFetcher())

    const markets = await findByRole('group', { name: /markets/i })
    for (const market of ['DE', 'FR', 'GB']) {
      expect(within(markets).getByRole('checkbox', { name: market })).toBeTruthy()
    }
    expect(within(markets).queryByRole('checkbox', { name: 'ZZ' })).toBeNull()
  })

  it('offers only the forecast years the snapshot actually holds', async () => {
    const { findByLabelText } = renderOpportunities(successFetcher())

    // The filters render before the ranking resolves, so the option list is
    // empty until the snapshot's facets arrive.
    await screen.findByRole('option', { name: '2028' })

    const select = await findByLabelText('Forecast year')
    const years = within(select as HTMLSelectElement)
      .getAllByRole('option')
      .map((option) => option.textContent)
    expect(years).toEqual(['All forecast years', '2028', '2031'])
  })

  it('selecting a market returns to the first page', async () => {
    const { findByRole, onSearchChange } = renderOpportunities(successFetcher())
    const markets = await findByRole('group', { name: /markets/i })

    await userEvent.click(within(markets).getByRole('checkbox', { name: 'FR' }))

    expect(onSearchChange).toHaveBeenCalledWith(
      expect.objectContaining({ market: ['FR'], page: 1 }),
    )
  })

  it('reordering the ranking returns to the first page', async () => {
    const { findByLabelText, onSearchChange } = renderOpportunities(successFetcher())
    const select = await findByLabelText('Order by')

    await userEvent.selectOptions(select, 'demand')

    expect(onSearchChange).toHaveBeenCalledWith(
      expect.objectContaining({ order: 'demand', page: 1 }),
    )
  })

  it('does not request a page per keystroke', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const { findByLabelText, onSearchChange } = renderOpportunities(successFetcher())
      const box = await findByLabelText('Search vehicle')

      await userEvent.type(box, 'Golf')
      expect(onSearchChange).not.toHaveBeenCalled()

      await vi.advanceTimersByTimeAsync(400)
      expect(onSearchChange).toHaveBeenCalledTimes(1)
      expect(onSearchChange).toHaveBeenCalledWith(
        expect.objectContaining({ q: 'Golf', page: 1 }),
      )
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('score scope disclosure', () => {
  it('says nothing about scope when the ranking is unfiltered', async () => {
    renderOpportunities(successFetcher())
    await screen.findByText(/How the opportunity score is calculated/i)

    expect(screen.queryByRole('note')).toBeNull()
  })

  it('warns that scores are relative once the ranking is narrowed', async () => {
    const client = new PlannerApiClient(successFetcher())
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <AppProviders queryClient={queryClient}>
        <OpportunitiesWorkbench
          apiClient={client}
          onOpenDetails={vi.fn()}
          onSearchChange={vi.fn()}
          search={{ groupBy: 'model_year', page: 1, q: 'Golf' }}
        />
      </AppProviders>,
    )

    expect(await screen.findByRole('note')).toHaveTextContent(/relative to the rows currently shown/i)
  })
})
