import type { components } from '../../lib/api/schema'
import type { DataCoverageSearch, SortDirection, SortKey } from '../../lib/data-coverage-search'


export type DataReport = components['schemas']['EurocodeDataReportResponse']
type DataCode = components['schemas']['EurocodeDataCodeResponse']
type NoDataCode = components['schemas']['EurocodeNoDataResponse']
export type CountryGap = components['schemas']['EurocodeDataCountryGapResponse']

export const PAGE_SIZE = 50

export interface CoverageRow {
  code: string
  label: string
  drive: string | null
  tier: string
  firstYear: number | null
  lastYear: number | null
  cars: number | null
  carsP10: number | null
  carsP90: number | null
  shared: number | null
  observedShare: number | null
  observedMarkets: string[]
  sources: number | null
  flags: string[]
  hasCars: boolean
}

const FLAG_TEXT: Record<string, string> = {
  'pre-2010': 'Before 2010',
  'shared-only': 'Only shared',
  'one-market': 'One market',
  new: 'New',
}

export function flagText(flag: string): string {
  return FLAG_TEXT[flag] ?? flag
}

function noDataLabel(item: NoDataCode): string {
  return item.label ?? [item.makes.join(', '), item.descriptions[0]].filter(Boolean).join(' · ')
}

export function buildRows(report: DataReport): CoverageRow[] {
  const codes = report.codes.map((item: DataCode): CoverageRow => ({
    code: item.code,
    label: item.label,
    drive: item.drive,
    tier: item.tier,
    firstYear: item.first_year,
    lastYear: item.last_year,
    cars: item.fleet.certain_p50,
    carsP10: item.fleet.certain_p10,
    carsP90: item.fleet.certain_p90,
    shared: item.fleet.shared_p50,
    observedShare: item.observed_share,
    observedMarkets: item.observed_markets,
    sources: item.sources.length,
    flags: item.flags.map(flagText),
    hasCars: true,
  }))
  const noData = report.no_data.map((item: NoDataCode): CoverageRow => ({
    code: item.code,
    label: noDataLabel(item),
    drive: null,
    tier: item.tier,
    firstYear: item.first_year,
    lastYear: item.last_year,
    cars: null,
    carsP10: null,
    carsP90: null,
    shared: null,
    observedShare: null,
    observedMarkets: [],
    sources: null,
    flags: [item.reviewed ? 'Reviewed, no cars' : 'Not reviewed'],
    hasCars: false,
  }))
  return [...codes, ...noData]
}

export function filterRows(
  rows: CoverageRow[],
  search: Pick<DataCoverageSearch, 'q' | 'tier'>,
): CoverageRow[] {
  const needle = search.q?.trim().toLowerCase()
  return rows.filter((row) =>
    (!search.tier || row.tier === search.tier)
    && (!needle || row.code.toLowerCase().includes(needle) || row.label.toLowerCase().includes(needle)),
  )
}

function sortValue(row: CoverageRow, key: SortKey): number | string | null {
  switch (key) {
    case 'code': return row.code
    case 'label': return row.label.toLowerCase()
    case 'tier': return row.tier
    case 'years': return row.firstYear
    case 'cars': return row.cars
    case 'shared': return row.shared
    case 'observed': return row.observedShare
    case 'markets': return row.hasCars ? row.observedMarkets.length : null
    case 'sources': return row.sources
  }
}

/** Rows without a value for the column go last in either direction; ties fall back to the code. */
export function sortRows(rows: CoverageRow[], key: SortKey, direction: SortDirection): CoverageRow[] {
  const sign = direction === 'asc' ? 1 : -1
  return [...rows].sort((left, right) => {
    const a = sortValue(left, key)
    const b = sortValue(right, key)
    if (a === null && b !== null) return 1
    if (b === null && a !== null) return -1
    if (a !== null && b !== null && a !== b) return (a < b ? -1 : 1) * sign
    return left.code < right.code ? -1 : left.code > right.code ? 1 : 0
  })
}

export const BAND_LABELS: readonly (readonly [string, string])[] = [
  ['Y_LT2', '<2 y'],
  ['Y2-5', '2–4 y'],
  ['Y5-10', '5–9 y'],
  ['Y10-20', '10–19 y'],
  ['Y_GT20', '20+ y'],
]

/** The markets whose official count the model misses most come first. */
export function sortCountries(countries: CountryGap[]): CountryGap[] {
  return [...countries].sort((left, right) =>
    (right.official - right.modelled) - (left.official - left.modelled)
    || left.geography.localeCompare(right.geography))
}
