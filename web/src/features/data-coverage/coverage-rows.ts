import type { components } from '../../lib/api/schema'
import type { DataCoverageSearch, SortDirection, SortKey } from '../../lib/data-coverage-search'
import { format } from '../eurocodes/eurocode-text'


export type DataReport = components['schemas']['EurocodeDataReportResponse']
type DataCode = components['schemas']['EurocodeDataCodeResponse']
type NoDataCode = components['schemas']['EurocodeNoDataResponse']
export type CountryGap = components['schemas']['EurocodeDataCountryGapResponse']
export type CountryEstimate = components['schemas']['EurocodeDataEstimateResponse']

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
  'lower-bound': 'At least',
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

/** Eurostat's flags on an official figure, as the legend names them. */
const STATUS_TEXT: Record<string, string> = {
  i: 'Imputed',
  p: 'Provisional',
  b: 'Break in series',
}

export const STATUS_LEGEND = Object.entries(STATUS_TEXT)

export function statusText(status: string): string {
  return STATUS_TEXT[status] ?? status
}

/** How a country's estimated gap splits; a model above the official count has nothing to split. */
export const ESTIMATE_PARTS: readonly (readonly [keyof CountryEstimate, string])[] = [
  ['before_sources', 'Registered before model-level sources'],
  ['unnamed', 'Registrations without a model name'],
  ['other', 'Used imports and other'],
]

export const NO_AGE_SPLIT = 'No age split published'
export const NO_OVER_20_BAND = 'Office has no 20+ band'

/** Band names differ for an office whose "Y10-20" counts every car 10 years and older. */
export function bandLabel(band: string, openEnded: boolean): string {
  if (openEnded && band === 'Y10-20') return '10+ y'
  return BAND_LABELS.find(([name]) => name === band)?.[1] ?? band
}

const CONVENTION_TEXT: Record<string, string> = {
  '2y': '2 registration years count as "less than 2 years"',
  '3y': '3 registration years count as "less than 2 years", so every band edge moves up a year',
  '2y-4y': '2 registration years count as "less than 2 years" and 4 as "2–5 years", so "5–10 years" holds 4 (ages 6–9)',
  unclear: 'convention unclear: the model is shown under both readings (2 and 3 registration years)',
}

/** What a reader needs to interpret one country's row; nothing for a plain total. */
export function countryNotes(gap: CountryGap): string[] {
  const notes: string[] = []
  if (gap.band_year !== null && gap.band_year !== gap.year) {
    notes.push(
      `age bands from ${gap.band_year}`
      + (gap.band_official_total === null ? '' : ` (office total that year ${format(gap.band_official_total)})`),
    )
  }
  if (gap.convention !== null && gap.bands.length > 0) {
    notes.push(CONVENTION_TEXT[gap.convention] ?? `convention ${gap.convention}`)
  }
  if (gap.open_ended) {
    notes.push('no 20+ band: the 10–19 y figure counts every car 10 years and older')
  }
  if (gap.registrations_include_used_imports.length > 0) {
    notes.push(
      `official new-car counts for registration years ${gap.registrations_include_used_imports.join(', ')} include used imports`,
    )
  }
  return notes
}

/** The Eurostat flags any figure of the countries carries, so the legend appears only when needed. */
export function hasStatus(countries: CountryGap[]): boolean {
  return countries.some((gap) => gap.official_status !== null || gap.bands.some((band) => band.status !== null))
}

/** The markets whose official count the model misses most come first. */
export function sortCountries(countries: CountryGap[]): CountryGap[] {
  return [...countries].sort((left, right) =>
    (right.official - right.modelled) - (left.official - left.modelled)
    || left.geography.localeCompare(right.geography))
}
