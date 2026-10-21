// Shared fixtures for the opportunities page tests. Kept out of any test file:
// importing from a test file makes Vitest run that file's tests a second time.
import { QueryClient } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import { vi } from 'vitest'

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
    generation_source_url: 'https://manufacturer.example/generation-a',
    icor_worked_base_units: 250,
    demand: { downside_units: 1700, base_units: 2150, upside_units: 2620 },
    contributing_configuration_count: 2,
    exact_covered_base_units: 250,
    fallback_covered_base_units: 0,
    uncovered_base_units: 1900,
    coverage_status: 'mixed',
    icor_worked: true,
    score: {
      demand_percentile: 1,
      demand_points: 100,
      demand_rank: 1,
      demand_population: 4872,
      demand_basis: 'whole_market_base_units_all_markets_all_horizons',
      total_points: 100,
      strategy_name: 'demand',
      strategy_version: '2',
      explanation: '100 of 100 points from whole-market replacement demand.',
    },
    evidence_status: 'demonstration',
    data_version: 'demo-planner-v1',
    eurocodes: {
      codes: [{
        code: 'Q201', label: 'Aurora Mobility A1 Horizon I', first_year: 2020, last_year: 2025,
        drive: 'LHD', certain_units: 1500, shared_units: 400, shared_with: ['Q202'],
        shared_with_uncoded: false,
      }],
      single_units: 1500, shared_units: 400, outside_units: 100, unassignable_units: 0,
      unreviewed_units: 150,
    },
  }],
  summary: {
    base_units: 2150,
    exact_covered_base_units: 250,
    high_demand_uncovered_base_units: 1900,
  },
  strategy_name: 'demand',
  strategy_version: '2',
  integrity_warnings: [],
  total: 2,
  page: 1,
  page_size: 1,
  pages: 2,
  available_markets: ['DE', 'FR', 'GB'],
  available_horizons: [2028, 2031],
  demand_population: 4872,
  demand_basis: 'whole_market_base_units_all_markets_all_horizons',
  eurocodes_loaded: true,
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

export const eurocodeStatus = {
  loaded: true, catalogue_sha256: null, decisions_snapshot_id: 'demo',
  reviewed_identities: 3, orphaned_identities: 0, warnings: [],
  coverage: [
    { horizon: 2028, total_units: 1000, single_units: 500, shared_units: 100, outside_units: 0, unassignable_units: 0, unreviewed_units: 400, shared_with_uncoded_units: 0 },
    { horizon: 2031, total_units: 2000, single_units: 1500, shared_units: 400, outside_units: 50, unassignable_units: 0, unreviewed_units: 50, shared_with_uncoded_units: 0 },
  ],
}

export function successFetcher() {
  return vi.fn<typeof fetch>().mockImplementation((input) => {
    const url = typeof input === 'string'
      ? input
      : input instanceof URL ? input.href : input.url
    if (url.includes('/eurocodes/status')) return Promise.resolve(json(eurocodeStatus))
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
