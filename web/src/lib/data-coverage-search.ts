export const TIERS = ['A', 'B', 'C', 'D'] as const
export type Tier = (typeof TIERS)[number]

export const SORT_KEYS = [
  'code', 'label', 'tier', 'years', 'cars', 'shared', 'observed', 'markets', 'sources',
] as const
export type SortKey = (typeof SORT_KEYS)[number]
export type SortDirection = 'asc' | 'desc'

export const DEFAULT_SORT: SortKey = 'cars'
export const DEFAULT_DIRECTION: SortDirection = 'desc'
const MAX_TEXT_LENGTH = 64

export interface DataCoverageSearch {
  q?: string
  tier?: Tier
  by: SortKey
  dir: SortDirection
  page: number
}

/** A pasted or stale URL still opens the page: unknown values fall back to the defaults. */
export function parseDataCoverageSearch(raw: Record<string, unknown>): DataCoverageSearch {
  const q = typeof raw.q === 'string' ? raw.q.slice(0, MAX_TEXT_LENGTH) : ''
  const tier = TIERS.find((value) => value === raw.tier)
  const by = SORT_KEYS.find((value) => value === raw.by) ?? DEFAULT_SORT
  const dir = raw.dir === 'asc' || raw.dir === 'desc' ? raw.dir : DEFAULT_DIRECTION
  const page = Number(raw.page)
  return {
    ...(q.trim() && { q }),
    ...(tier && { tier }),
    by,
    dir,
    page: Number.isInteger(page) && page > 0 ? page : 1,
  }
}
