import { describe, expect, it } from 'vitest'

import {
  parseOpportunitySearch,
  serializeOpportunitySearch,
} from '../src/lib/opportunity-search'


describe('opportunity URL state', () => {
  it('normalizes an invalid grouping and reports the rejected key', () => {
    const parsed = parseOpportunitySearch({ groupBy: 'profit' })

    expect(parsed.value.groupBy).toBe('model_year')
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

describe('opportunity search text and ordering', () => {
  it('keeps a search term and a non-default order in route state', () => {
    const parsed = parseOpportunitySearch({
      groupBy: 'model_year',
      q: 'Golf',
      order: 'demand',
    })

    expect(parsed.value.q).toBe('Golf')
    expect(parsed.value.order).toBe('demand')
    expect(parsed.invalidKeys).toEqual([])
  })

  it('omits the default order so a plain URL stays plain', () => {
    const parsed = parseOpportunitySearch({ groupBy: 'model_year', order: 'score' })

    expect(parsed.value.order).toBeUndefined()
    expect(serializeOpportunitySearch(parsed.value)).not.toHaveProperty('order')
  })

  it('falls back to the score order and reports an unknown one', () => {
    const parsed = parseOpportunitySearch({ groupBy: 'model_year', order: 'cheapest' })

    expect(parsed.value.order).toBeUndefined()
    expect(parsed.invalidKeys).toEqual(['order'])
  })

  it('drops a blank search term rather than sending an empty filter', () => {
    const parsed = parseOpportunitySearch({ groupBy: 'model_year', q: '   ' })

    expect(parsed.value.q).toBeUndefined()
  })

  it('truncates an overlong term to the length the API accepts', () => {
    const parsed = parseOpportunitySearch({ groupBy: 'model_year', q: 'a'.repeat(100) })

    expect(parsed.value.q).toHaveLength(64)
    expect(parsed.invalidKeys).toEqual(['q'])
  })

  it('round-trips the search term through serialization', () => {
    const parsed = parseOpportunitySearch({ groupBy: 'model', q: 'Polo', order: 'vehicle' })

    expect(serializeOpportunitySearch(parsed.value)).toMatchObject({
      groupBy: 'model',
      q: 'Polo',
      order: 'vehicle',
    })
  })
})
