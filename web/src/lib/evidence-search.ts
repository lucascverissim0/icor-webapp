import type { EvidenceObservationsQuery } from './api/client'


export type EvidenceSearch = Omit<EvidenceObservationsQuery, 'pageSize'>

const MEASURES = ['new_registrations', 'active_fleet'] as const
const MAPPING_STATUSES = [
  'exact_identifier', 'curated_alias', 'normalized_label', 'reviewed_probable',
  'ambiguous', 'rejected', 'unresolved',
] as const

function bounded(value: unknown, maximum: number): string | undefined {
  return typeof value === 'string' && value.length > 0 && value.length <= maximum
    ? value
    : undefined
}

export function parseEvidenceSearch(raw: Record<string, unknown>): EvidenceSearch {
  const releaseId = bounded(raw.releaseId, 80)
  const geography = bounded(raw.geography, 80)
  const search = bounded(raw.search, 100)
  const measure = MEASURES.includes(raw.measure as (typeof MEASURES)[number])
    ? raw.measure as EvidenceSearch['measure']
    : undefined
  const mappingStatus = MAPPING_STATUSES.includes(
    raw.mappingStatus as (typeof MAPPING_STATUSES)[number],
  ) ? raw.mappingStatus as EvidenceSearch['mappingStatus'] : undefined
  const parsedPage = Number(raw.page)
  const page = Number.isInteger(parsedPage) && parsedPage > 0 ? parsedPage : 1
  return {
    ...(releaseId && { releaseId }),
    ...(geography && { geography }),
    ...(measure && { measure }),
    ...(mappingStatus && { mappingStatus }),
    ...(search && { search }),
    page,
  }
}
