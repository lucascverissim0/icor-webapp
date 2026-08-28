import type { RegistrationRankingQuery } from './api/client'


export type RegistrationSearch = Pick<RegistrationRankingQuery, 'geography' | 'year' | 'search' | 'page'>

export function parseRegistrationSearch(raw: Record<string, unknown>): RegistrationSearch {
  const search = typeof raw.search === 'string' && raw.search.length > 0 && raw.search.length <= 100
    ? raw.search
    : undefined
  const parsedPage = Number(raw.page)
  const page = Number.isInteger(parsedPage) && parsedPage > 0 ? parsedPage : 1
  const parsedYear = Number(raw.year)
  const year = Number.isInteger(parsedYear) && parsedYear >= 1900 && parsedYear <= 2200
    ? parsedYear
    : 2024
  const geography = typeof raw.geography === 'string' && raw.geography.length > 0
    ? raw.geography
    : 'EU27'
  return { geography, year, ...(search && { search }), page }
}
