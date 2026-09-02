import { describe, expect, it } from 'vitest'

import {
  parseOpportunitySearch,
  serializeOpportunitySearch,
} from '../src/lib/opportunity-search'


describe('opportunity URL state', () => {
  it('normalizes an invalid grouping and reports the rejected key', () => {
    const parsed = parseOpportunitySearch({ groupBy: 'profit' })

    expect(parsed.value.groupBy).toBe('brand')
    expect(parsed.invalidKeys).toEqual(['groupBy'])
  })

  it('deduplicates canonical market and horizon filters', () => {
    const parsed = parseOpportunitySearch({
      groupBy: 'model_year',
      market: ['FR', 'FR'],
      horizon: ['2030', 2030],
    })

    expect(parsed.value).toEqual({
      groupBy: 'model_year',
      market: ['FR'],
      horizon: [2030],
      page: 1,
    })
    expect(parsed.invalidKeys).toEqual([])
  })

  it('serializes only API-relevant canonical route state', () => {
    expect(
      serializeOpportunitySearch({
        groupBy: 'model',
        market: ['DE'],
        page: 1,
        invalidKeys: ['horizon'],
      }),
    ).toEqual({ groupBy: 'model', market: ['DE'], page: 1 })
  })

  it('normalizes pagination and rejects invalid page values', () => {
    expect(parseOpportunitySearch({ groupBy: 'brand', page: '3' }).value.page).toBe(3)
    expect(parseOpportunitySearch({ groupBy: 'brand', page: '0' }).invalidKeys).toContain('page')
  })
})
