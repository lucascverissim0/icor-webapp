import type { components } from './api/schema'


type OpportunityGroupBy = components['schemas']['OpportunityGroupBy']
export type OpportunitySort = 'score' | 'demand' | 'vehicle'

const SORTS: readonly OpportunitySort[] = ['score', 'demand', 'vehicle']

// Matches the server's own cap. A longer value is truncated rather than
// rejected, because a URL someone pasted should still open the ranking.
const MAX_TEXT_LENGTH = 64

export interface OpportunitySearch {
  groupBy: OpportunityGroupBy
  market?: string[]
  horizon?: number[]
  q?: string
  order?: OpportunitySort
  /** Only vehicles ICOR has (yes) or has not (no) worked on; absent shows all. */
  worked?: 'yes' | 'no'
  page: number
}

export interface OpportunityRouteSearch extends OpportunitySearch {
  invalidKeys?: string[]
}

export interface ParsedOpportunitySearch {
  value: OpportunitySearch
  invalidKeys: string[]
}

const GROUPINGS: readonly OpportunityGroupBy[] = ['eurocode', 'model_year', 'model', 'brand']

/** Eurocodes lead the ranking; the verified client release serves model years only. */
export function defaultGrouping(clientRelease: boolean): OpportunityGroupBy {
  return clientRelease ? 'model_year' : 'eurocode'
}

const DEFAULT_GROUPING = defaultGrouping(import.meta.env.VITE_ICOR_CLIENT_RELEASE === 'verified')

function rawValues(value: unknown): unknown[] {
  if (value === undefined || value === null) return []
  return Array.isArray(value) ? value : [value]
}

export function parseOpportunitySearch(
  raw: Record<string, unknown>,
  fallbackGroupBy: OpportunityGroupBy = DEFAULT_GROUPING,
): ParsedOpportunitySearch {
  const invalid = new Set<string>()
  const groupBy = GROUPINGS.includes(raw.groupBy as OpportunityGroupBy)
    ? (raw.groupBy as OpportunityGroupBy)
    : fallbackGroupBy
  if (raw.groupBy !== undefined && groupBy !== raw.groupBy) invalid.add('groupBy')

  const rawMarkets = rawValues(raw.market)
  const validMarkets = rawMarkets.filter(
    (value): value is string => typeof value === 'string' && value.length > 0,
  )
  const markets = [
    ...new Set(validMarkets),
  ]
  if (validMarkets.length !== rawMarkets.length) invalid.add('market')

  const rawHorizons = rawValues(raw.horizon)
  const horizons = [
    ...new Set(
      rawHorizons
        .map((value) =>
          typeof value === 'number'
            ? value
            : typeof value === 'string'
              ? Number(value)
              : Number.NaN,
        )
        .filter(Number.isInteger),
    ),
  ]
  const allHorizonsValid = rawHorizons.every((value) => Number.isInteger(Number(value)))
  if (!allHorizonsValid) invalid.add('horizon')
  const rawText = raw.q
  let q: string | undefined
  if (typeof rawText === 'string' && rawText.trim().length > 0) {
    q = rawText.trim().slice(0, MAX_TEXT_LENGTH)
    if (q !== rawText) invalid.add('q')
  } else if (rawText !== undefined && rawText !== null && rawText !== '') {
    invalid.add('q')
  }

  const order = SORTS.includes(raw.order as OpportunitySort)
    ? (raw.order as OpportunitySort)
    : 'score'
  if (raw.order !== undefined && order !== raw.order) invalid.add('order')

  const worked = raw.worked === 'yes' || raw.worked === 'no' ? raw.worked : undefined
  if (raw.worked !== undefined && worked === undefined) invalid.add('worked')

  const parsedPage = Number(raw.page ?? 1)
  const page = Number.isInteger(parsedPage) && parsedPage >= 1 ? parsedPage : 1
  if (raw.page !== undefined && page !== parsedPage) invalid.add('page')

  return {
    value: {
      groupBy,
      page,
      ...(markets.length > 0 && { market: markets }),
      ...(horizons.length > 0 && { horizon: horizons }),
      ...(q !== undefined && { q }),
      ...(order !== 'score' && { order }),
      ...(worked !== undefined && { worked }),
    },
    invalidKeys: [...invalid].sort(),
  }
}

export function serializeOpportunitySearch<T extends OpportunityRouteSearch>(
  search: T,
): OpportunitySearch {
  return {
    groupBy: search.groupBy,
    page: search.page,
    ...(search.market && { market: search.market }),
    ...(search.horizon && { horizon: search.horizon }),
    ...(search.q && { q: search.q }),
    ...(search.order && search.order !== 'score' && { order: search.order }),
    ...(search.worked && { worked: search.worked }),
  }
}
