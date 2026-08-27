import { QueryClient } from '@tanstack/react-query'
import axe from 'axe-core'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { AppProviders } from '../src/app/providers'
import { RegistrationsWorkbench } from '../src/features/registrations/RegistrationsPage'
import { PlannerApiClient } from '../src/lib/api/client'


const summary = {
  snapshot_id: 'snapshot-real-2024', status: 'candidate',
  built_at: '2026-08-27T12:00:00Z', database_sha256: 'a'.repeat(64),
  identity_registry: 'exact-normalized-model-family-v1',
  geographies: ['EU27'], years: [2024], total_registrations: '15000000',
  model_count: 2, model_year_available: false,
  release_ids: ['eea-co2cars-2024-final-v30-r1'],
}

const ranking = {
  items: [{
    rank: 1, vehicle_id: 'vehicle-example-alpha', make: 'Example Motors', model: 'Alpha',
    model_year: null, registrations: '1500000', status: 'derived_observed',
    evidence_confidence: 79, input_observation_count: 27,
    release_ids: ['eea-co2cars-2024-final-v30-r1'], source_ids: ['eea-co2-monitoring'],
  }],
  total: 2, total_registrations: '15000000', page: 1, page_size: 25, pages: 2,
  snapshot_id: 'snapshot-real-2024',
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', 'X-Correlation-ID': 'corr-real' },
  })
}

function successFetcher() {
  return vi.fn<typeof fetch>().mockImplementation((input) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    if (url.endsWith('/api/v1/registrations/summary')) return Promise.resolve(json(summary))
    if (url.includes('/api/v1/registrations/ranking')) return Promise.resolve(json(ranking))
    throw new Error(`Unhandled URL: ${url}`)
  })
}

function renderRegistrations(fetcher: typeof fetch) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <AppProviders queryClient={queryClient}>
      <RegistrationsWorkbench apiClient={new PlannerApiClient(fetcher)} />
    </AppProviders>,
  )
}

describe('RegistrationsWorkbench', () => {
  it('presents official registration evidence without inventing model year or demand', async () => {
    renderRegistrations(successFetcher())

    expect(await screen.findByText('Example Motors')).toBeVisible()
    expect(screen.getByRole('heading', { name: 'Official 2024 registrations' })).toBeVisible()
    expect(screen.getByText('15,000,000')).toBeVisible()
    expect(screen.getByText('1,500,000')).toBeVisible()
    expect(screen.getByText('Model year unavailable')).toBeVisible()
    expect(screen.getByText(/registration year is not model year/i)).toBeVisible()
    expect(screen.getByText(/windshield fitment and replacement forecasts are not inferred/i)).toBeVisible()
    expect(screen.getByRole('link', { name: 'Inspect source evidence' })).toHaveAttribute('href', '/evidence')
    expect(screen.queryByText(/demonstration forecast/i)).not.toBeInTheDocument()
  })

  it('sends URL-ready search and page changes through its state boundary', async () => {
    const user = userEvent.setup()
    const fetcher = successFetcher()
    renderRegistrations(fetcher)

    await screen.findByText('Example Motors')
    await user.type(screen.getByRole('searchbox', { name: 'Search make or model' }), 'Alpha')
    await user.click(screen.getByRole('button', { name: 'Search registrations' }))
    await user.click(screen.getByRole('button', { name: 'Next' }))

    expect(fetcher).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/v1\/registrations\/ranking\?.*search=Alpha/),
      expect.any(Object),
    )
    expect(fetcher).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/v1\/registrations\/ranking\?.*page=2/),
      expect.any(Object),
    )
  })

  it('shows a safe unavailable state with no prototype fallback', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(json({
      code: 'registration_data_unavailable',
      message: 'Verified official registration data is not available.',
      correlation_id: 'corr-real', field_errors: [],
    }, 503))
    renderRegistrations(fetcher)

    expect(await screen.findByText('Official registration data is unavailable')).toBeVisible()
    expect(screen.getByText('Reference: corr-real')).toBeVisible()
    expect(screen.queryByText('Example Motors')).not.toBeInTheDocument()
  })

  it('has no automated accessibility violations when populated', async () => {
    const { container } = renderRegistrations(successFetcher())
    await screen.findByText('Example Motors')

    expect((await axe.run(container)).violations).toEqual([])
  })
})
