import { describe, expect, it } from 'vitest'

import {
  defaultGrouping,
  parseOpportunitySearch,
  serializeOpportunitySearch,
} from '../src/lib/opportunity-search'


describe('opportunity URL state', () => {
  it('normalizes an invalid grouping and reports the rejected key', () => {
    const parsed = parseOpportunitySearch({ groupBy: 'profit' }, 'model_year')

    expect(parsed.value.groupBy).toBe('model_year')
    expect(parsed.invalidKeys).toEqual(['groupBy'])
  })

  it('defaults to Eurocodes, and to model years in the verified client release', () => {
    expect(defaultGrouping(false)).toBe('eurocode')
    expect(defaultGrouping(true)).toBe('model_year')
    const plain = parseOpportunitySearch({})
    expect(plain.value.groupBy).toBe('eurocode')
    expect(plain.invalidKeys).toEqual([])
    expect(parseOpportunitySearch({}, defaultGrouping(true)).value.groupBy).toBe('model_year')
  })

  it('accepts eurocode as a grouping and keeps an explicit model-year one', () => {
    expect(parseOpportunitySearch({ groupBy: 'eurocode' }, 'model_year').value.groupBy).toBe('eurocode')
    expect(parseOpportunitySearch({ groupBy: 'model_year' }).value.groupBy).toBe('model_year')
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

describe('ICOR worked filter', () => {
  it('keeps yes and no in route state and drops anything else', () => {
    expect(parseOpportunitySearch({ groupBy: 'model_year', worked: 'yes' }).value.worked).toBe('yes')
    expect(parseOpportunitySearch({ groupBy: 'model_year', worked: 'no' }).value.worked).toBe('no')
    const invalid = parseOpportunitySearch({ groupBy: 'model_year', worked: 'maybe' })
    expect(invalid.value.worked).toBeUndefined()
    expect(invalid.invalidKeys).toEqual(['worked'])
    expect(serializeOpportunitySearch({ groupBy: 'model_year', page: 1, worked: 'no' })).toEqual({
      groupBy: 'model_year',
      page: 1,
      worked: 'no',
    })
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
