import type { RegistrationRankingQuery } from './api/client'


export type RegistrationSearch = Pick<RegistrationRankingQuery, 'search' | 'page'>

export function parseRegistrationSearch(raw: Record<string, unknown>): RegistrationSearch {
  const search = typeof raw.search === 'string' && raw.search.length > 0 && raw.search.length <= 100
    ? raw.search
    : undefined
  const parsedPage = Number(raw.page)
  const page = Number.isInteger(parsedPage) && parsedPage > 0 ? parsedPage : 1
  return { ...(search && { search }), page }
}
