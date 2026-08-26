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
    })
    expect(parsed.invalidKeys).toEqual([])
  })

  it('serializes only API-relevant canonical route state', () => {
    expect(
      serializeOpportunitySearch({
        groupBy: 'model',
        market: ['DE'],
        invalidKeys: ['horizon'],
      }),
    ).toEqual({ groupBy: 'model', market: ['DE'] })
  })
})
