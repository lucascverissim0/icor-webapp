import type { components } from './api/schema'


type OpportunityGroupBy = components['schemas']['OpportunityGroupBy']

export interface OpportunitySearch {
  groupBy: OpportunityGroupBy
  market?: string[]
  horizon?: number[]
}

export interface OpportunityRouteSearch extends OpportunitySearch {
  invalidKeys?: string[]
}

export interface ParsedOpportunitySearch {
  value: OpportunitySearch
  invalidKeys: string[]
}

const GROUPINGS: readonly OpportunityGroupBy[] = ['brand', 'model', 'model_year']

function rawValues(value: unknown): unknown[] {
  if (value === undefined || value === null) return []
  return Array.isArray(value) ? value : [value]
}

export function parseOpportunitySearch(
  raw: Record<string, unknown>,
): ParsedOpportunitySearch {
  const invalid = new Set<string>()
  const groupBy = GROUPINGS.includes(raw.groupBy as OpportunityGroupBy)
    ? (raw.groupBy as OpportunityGroupBy)
    : 'brand'
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

  return {
    value: {
      groupBy,
      ...(markets.length > 0 && { market: markets }),
      ...(horizons.length > 0 && { horizon: horizons }),
    },
    invalidKeys: [...invalid].sort(),
  }
}

export function serializeOpportunitySearch<T extends OpportunityRouteSearch>(
  search: T,
): OpportunitySearch {
  return {
    groupBy: search.groupBy,
    ...(search.market && { market: search.market }),
    ...(search.horizon && { horizon: search.horizon }),
  }
}
