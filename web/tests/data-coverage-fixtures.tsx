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
type Country = components['schemas']['EurocodeDataCountryGapResponse']
type Band = components['schemas']['EurocodeDataBandResponse']
type Estimate = components['schemas']['EurocodeDataEstimateResponse']

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
    lower_bound_p50: 0,
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

export function estimate(overrides: Partial<Estimate> = {}): Estimate {
  return { above_official: 0, before_sources: 0, missing: 0, other: 0, unnamed: 0, ...overrides }
}

export function band(name: string, official: number, modelled: number, overrides: Partial<Band> = {}): Band {
  return {
    band: name, official, modelled, modelled_alternative: null, status: null, estimated: estimate(),
    ...overrides,
  }
}

/** A market whose office publishes a total only, unless the overrides say otherwise. */
export function country(overrides: Partial<Country> & Pick<Country, 'geography'>): Country {
  return {
    publisher_geography: overrides.geography,
    year: 2026,
    official: 1000,
    official_status: null,
    modelled: 900,
    modelled_share: 0.9,
    estimated: estimate({ missing: 100, before_sources: 60, unnamed: 10, other: 30 }),
    band_year: null,
    band_official_total: null,
    convention: null,
    open_ended: false,
    bands: [],
    under_20: null,
    registrations_include_used_imports: [],
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
      flags: ['new', 'shared-only', 'one-market', 'pre-2010', 'lower-bound'],
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
    // Total only, flagged imputed.
    country({ geography: 'FR', official_status: 'i' }),
    // Age bands of an earlier year, no 20+ band, both conventions shown, a provisional total.
    country({
      geography: 'GB',
      publisher_geography: 'UK',
      official: 5000,
      official_status: 'p',
      modelled: 2000,
      modelled_share: 0.4,
      estimated: estimate({ missing: 3000, before_sources: 1200, unnamed: 300, other: 1500 }),
      band_year: 2025,
      band_official_total: 4800,
      convention: 'unclear',
      open_ended: true,
      bands: [
        band('Y_LT2', 1000, 900, {
          modelled_alternative: 1100,
          status: 'b',
          estimated: estimate({ missing: 100, before_sources: 40, unnamed: 10, other: 50 }),
        }),
        band('Y2-5', 1000, 500, {
          modelled_alternative: 400,
          estimated: estimate({ missing: 500, before_sources: 200, unnamed: 100, other: 200 }),
        }),
        band('Y5-10', 1500, 400, {
          modelled_alternative: 300,
          estimated: estimate({ missing: 1100, before_sources: 500, unnamed: 100, other: 500 }),
        }),
        band('Y10-20', 1300, 200, {
          modelled_alternative: 200,
          estimated: estimate({ missing: 1100, before_sources: 400, unnamed: 100, other: 600 }),
        }),
      ],
    }),
    // Full age bands under the 3-year convention, a band the model overshoots.
    country({
      geography: 'NL',
      official: 3000,
      modelled: 2600,
      modelled_share: 0.8667,
      estimated: estimate({ missing: 400, before_sources: 150, unnamed: 50, other: 200 }),
      band_year: 2026,
      band_official_total: 3000,
      convention: '3y',
      bands: [
        band('Y_LT2', 300, 350, { estimated: estimate({ above_official: 50 }) }),
        band('Y2-5', 600, 600),
        band('Y5-10', 900, 800, { estimated: estimate({ missing: 100, other: 100 }) }),
        band('Y10-20', 900, 700, { estimated: estimate({ missing: 200, before_sources: 150, unnamed: 50 }) }),
        band('Y_GT20', 300, 150, { estimated: estimate({ missing: 150, other: 150 }) }),
      ],
      under_20: {
        official: 2700,
        modelled: 2450,
        modelled_share: 0.9074,
        estimated: estimate({ missing: 300, before_sources: 150, unnamed: 50, other: 100, above_official: 50 }),
      },
    }),
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
