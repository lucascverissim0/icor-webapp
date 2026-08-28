import { QueryClient } from '@tanstack/react-query'
import axe from 'axe-core'
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { AppProviders } from '../src/app/providers'
import {
  ConfigurationDetail,
  ResponsiveDetailLayout,
} from '../src/features/planner/ConfigurationDetail'
import { PlannerApiClient } from '../src/lib/api/client'


const configuration = {
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
  equipment: {
    camera_adas: true,
    hud: null,
    heated: false,
    acoustic: true,
    rain_light_sensor: true,
  },
  forecast_horizon: 2030,
  demand: { downside_units: 980, base_units: 1240, upside_units: 1510 },
  vehicle_exposure_units: 62000,
  replacement_rate: 0.02,
  identity_confidence: {
    level: 'medium',
    reason: 'Synthetic equipment distinctions are intentionally incomplete.',
  },
  data_quality_confidence: {
    level: 'low',
    reason: 'Demand values are demonstration-only and are not calibrated.',
  },
  evidence_status: 'demonstration',
  sources: [{
    name: 'Synthetic vehicle scenario',
    description: 'Fictional non-proprietary identity and exposure assumptions.',
  }],
  updated_at: '2026-08-25T12:00:00Z',
  data_version: 'demo-planner-v1',
} as const

const notFound = {
  code: 'configuration_not_found',
  message: 'The requested planning configuration was not found.',
  correlation_id: 'test-not-found',
  field_errors: [],
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function renderDetail(response: unknown = configuration, status = 200) {
  const fetcher = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(response, status))
  const apiClient = new PlannerApiClient(fetcher)
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const rendered = render(
    <AppProviders queryClient={queryClient}>
      <ConfigurationDetail
        apiClient={apiClient}
        backHref="/planner?market=FR&horizon=2030&page=1&sort=base_demand&direction=desc"
        configurationId="demo-aurora-a1-camera-fr-2030"
      />
    </AppProviders>,
  )
  return { ...rendered, fetcher }
}

describe('ConfigurationDetail', () => {
  it('keeps planner context beside the shared detail surface', () => {
    render(
      <ResponsiveDetailLayout
        detail={<p>Selected configuration</p>}
        planner={<p>Planner comparison</p>}
      />,
    )

    expect(screen.getByText('Planner comparison')).toBeVisible()
    expect(screen.getByRole('complementary', { name: 'Selected configuration' })).toBeVisible()
  })

  it('renders unknown equipment explicitly without inferring compatibility', async () => {
    renderDetail()

    expect(await screen.findByRole('heading', { name: 'Aurora Mobility A1 Horizon' })).toBeVisible()
    expect(screen.getByText('Head-up display')).toBeVisible()
    expect(screen.getAllByText('Unknown')).not.toHaveLength(0)
    expect(screen.getByText('Not fitted')).toBeVisible()
  })

  it('shows the demand assumptions, confidence reasons, and synthetic sources', async () => {
    const { container } = renderDetail()

    expect(await screen.findByText('1,240 units')).toBeVisible()
    expect(screen.getByText('62,000 vehicles')).toBeVisible()
    expect(screen.getByText('2%')).toBeVisible()
    expect(screen.getByText(/Synthetic equipment distinctions/)).toBeVisible()
    expect(screen.getByText('Synthetic vehicle scenario')).toBeVisible()
    expect(screen.getByText('demo-planner-v1')).toBeVisible()
    expect((await axe.run(container)).violations).toEqual([])
  })

  it('returns through a link that preserves planner search', async () => {
    renderDetail()

    expect(await screen.findByRole('link', { name: 'Back to planner' })).toHaveAttribute(
      'href',
      '/planner?market=FR&horizon=2030&page=1&sort=base_demand&direction=desc',
    )
  })

  it('offers a safe return for a missing configuration', async () => {
    renderDetail(notFound, 404)

    expect(await screen.findByRole('heading', { name: 'Opportunity not found' })).toBeVisible()
    expect(screen.getByRole('link', { name: 'Return to planner' })).toHaveAttribute(
      'href',
      '/planner?market=FR&horizon=2030&page=1&sort=base_demand&direction=desc',
    )
    expect(screen.queryByText('test-not-found')).not.toBeInTheDocument()
  })
})
