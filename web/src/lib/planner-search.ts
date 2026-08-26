import type { components } from './api/schema'


type EvidenceStatus = components['schemas']['EvidenceStatus']
type SortField = components['schemas']['SortField']
type SortDirection = components['schemas']['SortDirection']

export interface PlannerSearchOptions {
  markets: readonly string[]
  horizons: readonly number[]
  brands: readonly string[]
  models: readonly string[]
  evidenceStatuses: readonly EvidenceStatus[]
}

export interface PlannerSearch {
  market?: string[]
  horizon?: number[]
  brand?: string[]
  model?: string[]
  evidence?: EvidenceStatus[]
  sort: SortField
  direction: SortDirection
  page: number
}

export interface PlannerRouteSearch extends PlannerSearch {
  invalidKeys?: string[]
}

export interface ParsedPlannerSearch {
  value: PlannerSearch
  invalidKeys: string[]
}

const SORT_FIELDS: readonly SortField[] = [
  'base_demand',
  'downside_demand',
  'upside_demand',
  'brand',
  'model',
  'identity_confidence',
  'data_quality_confidence',
]
const SORT_DIRECTIONS: readonly SortDirection[] = ['asc', 'desc']
const EVIDENCE_STATUSES: readonly EvidenceStatus[] = [
  'demonstration',
  'prototype',
  'validated',
]

function rawValues(value: unknown): unknown[] {
  if (value === undefined || value === null) return []
  return Array.isArray(value) ? value : [value]
}

function unique<T>(values: readonly T[]): T[] {
  return [...new Set(values)]
}

function stringFilter(
  raw: unknown,
  allowed: readonly string[] | undefined,
): { values: string[]; invalid: boolean } {
  const candidates = rawValues(raw)
  const values = candidates.filter(
    (value): value is string =>
      typeof value === 'string' &&
      value.length > 0 &&
      (allowed === undefined || allowed.includes(value)),
  )
  return { values: unique(values), invalid: values.length !== candidates.length }
}

function horizonFilter(
  raw: unknown,
  allowed: readonly number[] | undefined,
): { values: number[]; invalid: boolean } {
  const candidates = rawValues(raw)
  const values = candidates
    .map((value) =>
      typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN,
    )
    .filter(
      (value) =>
        Number.isInteger(value) && (allowed === undefined || allowed.includes(value)),
    )
  return { values: unique(values), invalid: values.length !== candidates.length }
}

function enumValue<T extends string>(
  raw: unknown,
  allowed: readonly T[],
  fallback: T,
): { value: T; invalid: boolean } {
  if (raw === undefined) return { value: fallback, invalid: false }
  if (typeof raw === 'string' && allowed.includes(raw as T)) {
    return { value: raw as T, invalid: false }
  }
  return { value: fallback, invalid: true }
}

export function parsePlannerSearch(
  raw: Record<string, unknown>,
  options?: PlannerSearchOptions,
): ParsedPlannerSearch {
  const invalid = new Set<string>()
  const market = stringFilter(raw.market, options?.markets)
  const horizon = horizonFilter(raw.horizon, options?.horizons)
  const brand = stringFilter(raw.brand, options?.brands)
  const model = stringFilter(raw.model, options?.models)
  const evidence = stringFilter(
    raw.evidence,
    options?.evidenceStatuses ?? EVIDENCE_STATUSES,
  )
  const sort = enumValue(raw.sort, SORT_FIELDS, 'base_demand')
  const direction = enumValue(raw.direction, SORT_DIRECTIONS, 'desc')

  const rawPage = typeof raw.page === 'string' ? Number(raw.page) : raw.page
  const page = Number.isInteger(rawPage) && Number(rawPage) >= 1 ? Number(rawPage) : 1

  for (const [key, result] of Object.entries({ market, horizon, brand, model, evidence })) {
    if (raw[key] !== undefined && result.invalid) invalid.add(key)
  }
  if (sort.invalid) invalid.add('sort')
  if (direction.invalid) invalid.add('direction')
  if (raw.page !== undefined && page === 1 && rawPage !== 1) invalid.add('page')

  return {
    value: {
      ...(market.values.length > 0 && { market: market.values }),
      ...(horizon.values.length > 0 && { horizon: horizon.values }),
      ...(brand.values.length > 0 && { brand: brand.values }),
      ...(model.values.length > 0 && { model: model.values }),
      ...(evidence.values.length > 0 && {
        evidence: evidence.values as EvidenceStatus[],
      }),
      page,
      sort: sort.value,
      direction: direction.value,
    },
    invalidKeys: [...invalid].sort(),
  }
}

export function serializePlannerSearch<T extends PlannerRouteSearch>(search: T): PlannerSearch {
  return {
    ...(search.market && { market: search.market }),
    ...(search.horizon && { horizon: search.horizon }),
    ...(search.brand && { brand: search.brand }),
    ...(search.model && { model: search.model }),
    ...(search.evidence && { evidence: search.evidence }),
    page: search.page,
    sort: search.sort,
    direction: search.direction,
  }
}
