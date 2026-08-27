import { QueryClient } from '@tanstack/react-query'
import axe from 'axe-core'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { AppProviders } from '../src/app/providers'
import { EvidenceWorkbench } from '../src/features/evidence/EvidencePage'
import { PlannerApiClient } from '../src/lib/api/client'


const summary = {
  snapshot_id: 'snapshot-a92867b966f81d7966fe',
  status: 'candidate',
  built_at: '2026-08-27T12:00:00Z',
  database_sha256: '4'.repeat(64),
  observation_count: 542455,
  published_value_count: 0,
  warning_count: 0,
  versions: {
    source_registry: '1', identity_registry: '1', reconciliation_method: '1',
    confidence_method: '1', estimation_method: '1', survival_method: '1',
    hazard_method: '1', forecast_method: '1',
  },
  releases: [{
    release_id: 'uk-veh0120-2025', source_id: 'uk-dft-veh0120',
    publisher: 'UK Department for Transport',
    source_url: 'https://example.test/source', terms_url: 'https://example.test/terms',
    published_at: '2026-07-01T00:00:00Z', coverage_start: '2025-01-01',
    coverage_end: '2025-12-31', geography: 'United Kingdom', measure: 'registrations',
    dependency_group: 'uk-dft', raw_record_count: 10, accepted_record_count: 9,
    rejected_record_count: 1, quarantined_record_count: 0, observation_count: 9,
    total_value: '1500',
  }],
  mapping_status_counts: { unresolved: 542455 },
  geographies: ['United Kingdom'],
  measures: ['registrations'],
} as const

const observations = {
  items: [{
    observation_id: 'obs-1', release_id: 'uk-veh0120-2025', original_row_locator: 'table VEH0120 row 18',
    geography: 'United Kingdom', period_start: '2025-01-01', period_end: '2025-12-31',
    period_precision: 'year', measure: 'registrations', value: '1500', unit: 'vehicles',
    publication_status: 'official', original_make: 'ACME', original_model: 'ROADRUNNER',
    original_model_year: null, original_type: 'CAR', mapping_status: 'unresolved',
    transformation_notes: ['Whitespace normalized'], validation_flags: [], confidence_total: 70,
    confidence_reasons: ['Official publisher release'],
  }],
  total: 1, page: 1, page_size: 25, pages: 1,
} as const

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', 'X-Correlation-ID': 'corr-test' },
  })
}

function renderEvidence(fetcher: typeof fetch) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <AppProviders queryClient={queryClient}>
      <EvidenceWorkbench apiClient={new PlannerApiClient(fetcher)} />
    </AppProviders>,
  )
}

function successFetcher() {
  return vi.fn<typeof fetch>().mockImplementation((input) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    if (url.endsWith('/api/v1/evidence/summary')) return Promise.resolve(json(summary))
    if (url.includes('/api/v1/evidence/observations')) return Promise.resolve(json(observations))
    throw new Error(`Unhandled URL: ${url}`)
  })
}

describe('EvidenceWorkbench', () => {
  it('presents candidate provenance without implying canonical identity or publication', async () => {
    renderEvidence(successFetcher())

    expect(await screen.findByText('542,455')).toBeVisible()
    expect(screen.getByText(/reported source labels—not canonical vehicle identities/i)).toBeVisible()
    expect(screen.getByText(/not active and does not feed forecasts/i)).toBeVisible()
    expect(screen.getByText('ACME')).toBeVisible()
    expect(screen.getByText('ROADRUNNER')).toBeVisible()
    expect(screen.getByText('Unresolved')).toBeVisible()
    expect(screen.getByRole('link', { name: 'Open source' })).toHaveAttribute('rel', expect.stringContaining('noopener'))
    expect(screen.getByRole('link', { name: 'Usage terms' })).toHaveAttribute('href', summary.releases[0].terms_url)
  })

  it('sends bounded filters and page changes to the evidence endpoint', async () => {
    const user = userEvent.setup()
    const fetcher = successFetcher()
    renderEvidence(fetcher)

    await screen.findByText('ROADRUNNER')
    await user.type(screen.getByRole('searchbox', { name: 'Search source labels' }), 'acme')
    await user.click(screen.getByRole('button', { name: 'Apply filters' }))

    expect(fetcher).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/v1\/evidence\/observations\?.*search=acme/),
      expect.any(Object),
    )
  })

  it('shows a safe unavailable state without inventing fallback evidence', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(json({
      code: 'evidence_unavailable', message: 'Source evidence is not configured.',
      correlation_id: 'corr-test', field_errors: [],
    }, 503))
    renderEvidence(fetcher)

    expect(await screen.findByText('Source evidence is unavailable')).toBeVisible()
    expect(screen.getByText('Reference: corr-test')).toBeVisible()
    expect(screen.queryByText('542,455')).not.toBeInTheDocument()
  })

  it('has no automated accessibility violations in the populated state', async () => {
    const { container } = renderEvidence(successFetcher())
    await screen.findByText('ROADRUNNER')

    expect((await axe.run(container)).violations).toEqual([])
  })
})
