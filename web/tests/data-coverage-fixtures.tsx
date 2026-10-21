// Shared fixtures for the data coverage tests. Every code is fictional (Q prefix) and every
// vehicle one of the invented demo brands.
import { QueryClient } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import { vi } from 'vitest'

import { AppProviders } from '../src/app/providers'
import { DataCoverageWorkbench } from '../src/features/data-coverage/DataCoveragePage'
import { PlannerApiClient } from '../src/lib/api/client'
import type { DataCoverageSearch } from '../src/lib/data-coverage-search'
import type { components } from '../src/lib/api/schema'


type Report = components['schemas']['EurocodeDataReportResponse']
type Code = components['schemas']['EurocodeDataCodeResponse']

export const evidence = { stock: 0, registrations: 0, interpolated: 0, forecast: 0 }

export function code(overrides: Partial<Code> & Pick<Code, 'code'>): Code {
  return {
    drive: 'LHD',
    evidence: { ...evidence, registrations: 100 },
    families: ['aurora/a1-horizon'],
    first_year: 2020,
    flags: [],
    fleet: {
      certain_2028: 80, certain_p10: 90, certain_p50: 100, certain_p90: 110,
      shared_2028: 0, shared_p50: 0,
    },
    label: 'Aurora Mobility A1 Horizon I',
    last_year: 2025,
    life_years: 6,
    markets: 2,
    observed_life_years: 6,
    observed_markets: ['DE', 'FR'],
    observed_share: 1,
    shared_with: [],
    shared_with_uncoded: false,
    sources: ['pub-a'],
    tier: 'B',
    years_before_2010: 0,
    ...overrides,
  }
}

export const report: Report = {
  codes: [
    code({ code: 'Q201', tier: 'A', observed_markets: ['DE', 'FR', 'NL', 'BE', 'ES'] }),
    code({
      code: 'Q202',
      label: 'Aurora Mobility A1 Horizon II',
      first_year: 2025,
      last_year: null,
      tier: 'C',
      flags: ['new', 'shared-only', 'one-market', 'pre-2010'],
      fleet: {
        certain_2028: 0, certain_p10: 0, certain_p50: 0, certain_p90: 0,
        shared_2028: 70, shared_p50: 70,
      },
      observed_share: 0.25,
      observed_markets: ['DE'],
      sources: [],
    }),
    code({
      code: 'Q301',
      label: 'Meridian Motors M5 Vector',
      drive: 'both',
      tier: 'B',
      fleet: {
        certain_2028: 1000, certain_p10: 1100, certain_p50: 1500, certain_p90: 1800,
        shared_2028: 0, shared_p50: 0,
      },
      observed_share: 0.6,
    }),
  ],
  country_gap: [
    {
      geography: 'FR', publisher_geography: 'FR', year: 2026, official: 1000, modelled: 900,
      modelled_share: 0.9,
      bands: [
        { band: 'Y_LT2', official: 100, modelled: 90 },
        { band: 'Y2-5', official: 200, modelled: 180 },
        { band: 'Y5-10', official: 300, modelled: 270 },
        { band: 'Y10-20', official: 400, modelled: 360 },
        { band: 'Y_GT20', official: null, modelled: 0 },
      ],
    },
    {
      geography: 'GB', publisher_geography: 'UK', year: 2026, official: 5000, modelled: 2000,
      modelled_share: 0.4,
      bands: [
        { band: 'Y_LT2', official: 1000, modelled: 900 },
        { band: 'Y2-5', official: null, modelled: 500 },
        { band: 'Y5-10', official: 1500, modelled: 400 },
        { band: 'Y10-20', official: 1500, modelled: 200 },
        { band: 'Y_GT20', official: 1000, modelled: 0 },
      ],
    },
  ],
  markets: [],
  no_data: [
    {
      code: 'Q401', label: 'Velora Works V3 Current', makes: [], descriptions: [],
      first_year: 2024, last_year: null, reviewed: true, tier: 'D',
    },
    {
      code: 'Q501', label: null, makes: ['VELORA WORKS'], descriptions: ['V3 CURRENT WAGON'],
      first_year: null, last_year: null, reviewed: false, tier: 'D',
    },
  ],
  official_source: null,
  snapshot_id: 'demo-snapshot',
  thresholds: {
    pre_2010_share_cap_c: 0.5,
    tier_a_markets: 5,
    tier_a_observed_share: 0.8,
    tier_b_observed_share: 0.5,
  },
  totals: {
    by_evidence: { stock: 200, registrations: 600, interpolated: 100, forecast: 100 },
    by_outcome: { outside: 0, shared: 70, single: 1530, unassignable: 0, unreviewed: 0 },
    fleet: 1000,
    no_data: { first_year_from_2000: 1, not_reviewed: 1, reviewed: 1 },
    tiers: {
      A: { codes: 1, certain_p50: 100 },
      B: { codes: 1, certain_p50: 1500 },
      C: { codes: 1, certain_p50: 0 },
      D: { codes: 2, certain_p50: 0 },
    },
  },
  year: 2026,
}

/** A report with `count` extra single-code rows, for pagination. */
export function bigReport(count: number): Report {
  const extra = Array.from({ length: count }, (_, index) => code({
    code: `Q9${String(index).padStart(2, '0')}`,
    label: `Velora Works V3 Current ${index}`,
    fleet: {
      certain_2028: 1, certain_p10: 1, certain_p50: 10 + index, certain_p90: 20 + index,
      shared_2028: 0, shared_p50: 0,
    },
  }))
  return { ...report, codes: [...report.codes, ...extra] }
}

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', 'X-Correlation-ID': 'corr-coverage' },
  })
}

export function fetcherFor(body: unknown, status = 200) {
  return vi.fn<typeof fetch>().mockImplementation(() => Promise.resolve(json(body, status)))
}

export function renderCoverage(
  fetcher: typeof fetch,
  options: {
    search?: DataCoverageSearch
    onSearchChange?: (next: DataCoverageSearch) => void
    clientRelease?: boolean
  } = {},
) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <AppProviders queryClient={queryClient}>
      <DataCoverageWorkbench
        apiClient={new PlannerApiClient(fetcher)}
        clientRelease={options.clientRelease ?? false}
        onSearchChange={options.onSearchChange}
        search={options.search}
      />
    </AppProviders>,
  )
}
