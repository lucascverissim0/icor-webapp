import { QueryClient } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { AppProviders } from '../src/app/providers'
import { CoverageManager } from '../src/features/opportunities/CoverageManager'
import { PlannerApiClient } from '../src/lib/api/client'


const configuration = {
  configuration_id: 'demo-aurora-a1-camera-fr-2030',
  sku: 'DEMO-AUR-A1-CAM',
  part_family: 'Demo camera acoustic family',
  market: 'FR', brand: 'Aurora Mobility', model: 'A1 Horizon',
  model_year_start: 2025, model_year_end: 2028, generation: 'Demo generation A',
  facelift: null, body_style: 'Hatchback', drive_side: 'left',
  equipment: { camera_adas: true, hud: null, heated: false, acoustic: true, rain_light_sensor: true },
  forecast_horizon: 2030,
  demand: { downside_units: 980, base_units: 1240, upside_units: 1510 },
  vehicle_exposure_units: 62000, replacement_rate: 0.02,
  identity_confidence: { level: 'medium', reason: 'Synthetic identity.' },
  data_quality_confidence: { level: 'low', reason: 'Demonstration only.' },
  evidence_status: 'demonstration',
  sources: [{ name: 'Synthetic vehicle scenario', description: 'Fictional scenario.' }],
  updated_at: '2026-08-25T12:00:00Z', data_version: 'demo-planner-v1',
} as const
const configurationPage = {
  items: [configuration], total: 1, page: 1, page_size: 100, pages: 1,
  summary: { candidate_count: 1, downside_units: 980, base_units: 1240, upside_units: 1510 },
} as const

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}


function renderManager(fetcher: typeof fetch) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const rendered = render(
    <AppProviders queryClient={queryClient}>
      <CoverageManager apiClient={new PlannerApiClient(fetcher)} opportunityQuery={{ groupBy: 'brand' }} />
    </AppProviders>,
  )
  return { ...rendered, queryClient }
}

function managerFetcher({ mutationStatus = 201, savedRows = [] as unknown[] } = {}) {
  return vi.fn<typeof fetch>().mockImplementation((input, init) => {
    const url = typeof input === 'string'
      ? input
      : input instanceof URL ? input.href : input.url
    if (url.includes('/planner/configurations')) return Promise.resolve(json(configurationPage))
    if (url.endsWith('/production-coverage') && init?.method === 'POST') {
      if (mutationStatus >= 400) {
        return Promise.resolve(json({ code: 'internal_error', message: 'Could not save coverage.', correlation_id: 'corr-123', field_errors: [] }, mutationStatus))
      }
      return Promise.resolve(json({ coverage_id: 'coverage-1', match_type: 'exact_configuration', configuration_id: configurationPage.items[0].configuration_id, brand: 'Aurora Mobility', model: 'A1 Horizon', model_year: 2025, sku: 'DEMO-AUR-A1-CAM', note: null, created_at: '2026-08-26T12:00:00Z', updated_at: '2026-08-26T12:00:00Z' }, 201))
    }
    if (url.endsWith('/production-coverage')) return Promise.resolve(json(savedRows))
    if (url.includes('/opportunities')) return Promise.resolve(json({ items: [], summary: { base_units: 0, exact_covered_base_units: 0, high_demand_uncovered_base_units: 0 }, strategy_name: 'demand_readiness', strategy_version: '1', integrity_warnings: [] }))
    if (init?.method === 'DELETE') return Promise.resolve(json({ coverage_id: 'coverage-1', deleted: true }))
    throw new Error(`Unhandled URL: ${url}`)
  })
}

async function chooseCanonicalVehicle(user: ReturnType<typeof userEvent.setup>) {
  await screen.findByRole('option', { name: 'Aurora Mobility' })
  await user.selectOptions(screen.getByLabelText('Brand'), 'Aurora Mobility')
  await user.selectOptions(screen.getByLabelText('Model'), 'A1 Horizon')
  await user.selectOptions(screen.getByLabelText('Model year'), '2025')
}

describe('CoverageManager', () => {
  it('requires deliberate confirmation before fallback creation', async () => {
    const user = userEvent.setup()
    renderManager(managerFetcher())
    await chooseCanonicalVehicle(user)

    await user.click(screen.getByLabelText('Exact configuration unknown'))

    expect(screen.getByRole('button', { name: 'Save fallback coverage' })).toBeDisabled()
    await user.click(screen.getByLabelText(/I understand this is lower precision/i))
    expect(screen.getByRole('button', { name: 'Save fallback coverage' })).toBeEnabled()
  })

  it('retains canonical selections and reports correlation after a failed save', async () => {
    const user = userEvent.setup()
    renderManager(managerFetcher({ mutationStatus: 500 }))
    await chooseCanonicalVehicle(user)
    await user.selectOptions(screen.getByLabelText('Exact configuration / SKU'), configurationPage.items[0].configuration_id)

    await user.click(screen.getByRole('button', { name: 'Save exact coverage' }))

    expect(await screen.findByText(/corr-123/)).toBeVisible()
    expect(screen.getByLabelText('Brand')).toHaveValue('Aurora Mobility')
    expect(screen.getByLabelText('Model year')).toHaveValue('2025')
  })

  it('requires confirmation before deleting saved coverage', async () => {
    const user = userEvent.setup()
    const saved = [{ coverage_id: 'coverage-1', match_type: 'exact_configuration', configuration_id: configurationPage.items[0].configuration_id, brand: 'Aurora Mobility', model: 'A1 Horizon', model_year: 2025, sku: 'DEMO-AUR-A1-CAM', note: null, created_at: '2026-08-26T12:00:00Z', updated_at: '2026-08-26T12:00:00Z' }]
    const fetcher = managerFetcher({ savedRows: saved })
    renderManager(fetcher)

    await user.click(await screen.findByRole('button', { name: /Delete Aurora Mobility/i }))
    expect(screen.getByRole('button', { name: /Confirm delete Aurora Mobility/i })).toBeVisible()
    expect(fetcher.mock.calls.some(([, init]) => init?.method === 'DELETE')).toBe(false)
  })
})
