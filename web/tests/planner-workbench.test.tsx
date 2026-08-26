import { QueryClient } from '@tanstack/react-query'
import axe from 'axe-core'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { AppProviders } from '../src/app/providers'
import { PlannerWorkbench } from '../src/features/planner/PlannerPage'
import { PlannerApiClient } from '../src/lib/api/client'
import type { PlannerSearch } from '../src/lib/planner-search'


const options = {
  markets: ['DE', 'FR'],
  horizons: [2028, 2030],
  brands: ['Aurora Mobility', 'Renault'],
  models: ['A1 Horizon', 'Megane Vision'],
  evidence_statuses: ['demonstration'],
  scenario: {
    name: 'Windshield demand planning demonstration',
    description: 'Synthetic configuration-level demand for product workflow review.',
    evidence_status: 'demonstration',
    data_version: 'demo-planner-v1',
    updated_at: '2026-08-25T12:00:00Z',
  },
} as const

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
  sources: [
    {
      name: 'Synthetic vehicle scenario',
      description: 'Fictional non-proprietary identity and exposure assumptions.',
    },
  ],
  updated_at: '2026-08-25T12:00:00Z',
  data_version: 'demo-planner-v1',
} as const

const successPage = {
  items: [configuration],
  total: 1,
  page: 1,
  page_size: 25,
  pages: 1,
  summary: {
    candidate_count: 1,
    downside_units: 980,
    base_units: 1240,
    upside_units: 1510,
  },
} as const

const emptyPage = {
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
} as const

const serverProblem = {
  code: 'internal_error',
  message: 'The planner service could not complete the request.',
  correlation_id: 'test-server-error',
  field_errors: [],
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function renderWorkbench({
  pages = [successPage],
  search = { page: 1, sort: 'base_demand', direction: 'desc' },
  invalidKeys = [],
}: {
  pages?: readonly unknown[]
  search?: PlannerSearch
  invalidKeys?: string[]
} = {}) {
  const queuedPages = [...pages]
  const fetcher = vi.fn<typeof fetch>().mockImplementation((input) => {
    const url =
      typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.href
          : input.url
    if (url.includes('/options')) return Promise.resolve(jsonResponse(options))
    const response = queuedPages.shift() ?? successPage
    return Promise.resolve(
      response === serverProblem ? jsonResponse(response, 500) : jsonResponse(response),
    )
  })
  const client = new PlannerApiClient(fetcher)
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const onSearchChange = vi.fn()
  const onSelect = vi.fn()

  const rendered = render(
    <AppProviders queryClient={queryClient}>
      <PlannerWorkbench
        apiClient={client}
        invalidKeys={invalidKeys}
        onSearchChange={onSearchChange}
        onSelect={onSelect}
        search={search}
      />
    </AppProviders>,
  )

  return { ...rendered, fetcher, onSearchChange, onSelect }
}

describe('PlannerWorkbench', () => {
  it('shows demand range beside evidence and confidence', async () => {
    renderWorkbench()

    expect(await screen.findAllByText('1,240 units')).not.toHaveLength(0)
    expect(screen.getAllByText('980–1,510 units')).not.toHaveLength(0)
    expect(screen.getAllByText('Demonstration')).not.toHaveLength(0)
    expect(screen.getAllByText('Identity confidence: Medium')).not.toHaveLength(0)
  })

  it('keeps filters while retrying a server error', async () => {
    const user = userEvent.setup()
    renderWorkbench({
      pages: [serverProblem, successPage],
      search: { market: ['FR'], page: 1, sort: 'base_demand', direction: 'desc' },
    })

    expect(await screen.findByRole('checkbox', { name: 'France' })).toBeChecked()
    await user.click(await screen.findByRole('button', { name: 'Retry' }))

    expect(await screen.findAllByText('1,240 units')).not.toHaveLength(0)
    expect(screen.getByRole('checkbox', { name: 'France' })).toBeChecked()
  })

  it('names active constraints and offers reset for empty results', async () => {
    const user = userEvent.setup()
    const { onSearchChange } = renderWorkbench({
      pages: [emptyPage],
      search: {
        brand: ['Renault'],
        page: 1,
        sort: 'base_demand',
        direction: 'desc',
      },
    })

    expect(await screen.findByText(/Renault/)).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Reset filters' }))

    expect(onSearchChange).toHaveBeenCalledWith({
      page: 1,
      sort: 'base_demand',
      direction: 'desc',
    })
  })

  it('applies canonical filter choices and reports normalized URL keys', async () => {
    const user = userEvent.setup()
    const { onSearchChange } = renderWorkbench({ invalidKeys: ['market'] })

    expect(await screen.findByText('Adjusted URL filters: market')).toBeVisible()
    await user.click(screen.getByRole('checkbox', { name: 'Germany' }))
    await user.click(screen.getByRole('button', { name: 'Apply filters' }))

    expect(onSearchChange).toHaveBeenCalledWith({
      market: ['DE'],
      page: 1,
      sort: 'base_demand',
      direction: 'desc',
    })
  })

  it('selects a canonical configuration from an explicit action', async () => {
    const user = userEvent.setup()
    const { onSelect } = renderWorkbench()

    await user.click(await screen.findByRole('button', { name: /View details/ }))

    expect(onSelect).toHaveBeenCalledWith('demo-aurora-a1-camera-fr-2030')
  })

  it('replaces option values that the API does not recognize before querying', async () => {
    const { fetcher, onSearchChange } = renderWorkbench({
      search: {
        market: ['XX'],
        page: 3,
        sort: 'base_demand',
        direction: 'desc',
      },
    })

    await screen.findAllByText('1,240 units')

    expect(onSearchChange).toHaveBeenCalledWith({
      page: 1,
      sort: 'base_demand',
      direction: 'desc',
    })
    expect(fetcher.mock.calls.some(([input]) => {
      const url =
        typeof input === 'string'
          ? input
          : input instanceof URL
            ? input.href
            : input.url
      return url.includes('market=XX')
    })).toBe(false)
    expect(screen.getByText('Adjusted URL filters: market')).toBeVisible()
  })

  it('writes sorting and pagination changes through canonical search state', async () => {
    const user = userEvent.setup()
    const secondPage = { ...successPage, page: 2, pages: 3 }
    const { onSearchChange } = renderWorkbench({ pages: [secondPage] })

    await screen.findAllByText('1,240 units')
    await user.selectOptions(screen.getByRole('combobox', { name: 'Sort results' }), 'brand')
    expect(onSearchChange).toHaveBeenCalledWith({
      page: 1,
      sort: 'brand',
      direction: 'desc',
    })

    await user.click(screen.getByRole('button', { name: 'Next page' }))
    expect(onSearchChange).toHaveBeenCalledWith({
      page: 2,
      sort: 'base_demand',
      direction: 'desc',
    })
  })

  it('has no automated accessibility violations in the successful state', async () => {
    const { container } = renderWorkbench()
    await screen.findAllByText('1,240 units')

    const results = await axe.run(container)

    expect(results.violations).toEqual([])
  })
})
