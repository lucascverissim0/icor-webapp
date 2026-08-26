// @vitest-environment node

import { describe, expect, it } from 'vitest'

import {
  parsePlannerSearch,
  serializePlannerSearch,
  type PlannerSearchOptions,
} from '../src/lib/planner-search'


const options: PlannerSearchOptions = {
  markets: ['DE', 'FR'],
  horizons: [2028, 2030],
  brands: ['Aurora Mobility', 'Velora Works'],
  models: ['A1 Horizon', 'V3 Current'],
  evidenceStatuses: ['demonstration'],
}

describe('planner URL state', () => {
  it('reports and removes obsolete filter values', () => {
    expect(parsePlannerSearch({ market: 'XX', page: '-4' }, options)).toEqual({
      value: { page: 1, sort: 'base_demand', direction: 'desc' },
      invalidKeys: ['market', 'page'],
    })
  })

  it('deduplicates canonical arrays and parses numeric horizons', () => {
    expect(
      parsePlannerSearch(
        {
          market: ['FR', 'FR', 'DE'],
          horizon: ['2030', 2028],
          brand: 'Aurora Mobility',
          model: ['A1 Horizon'],
          evidence: 'demonstration',
          sort: 'brand',
          direction: 'asc',
          page: '2',
        },
        options,
      ),
    ).toEqual({
      value: {
        market: ['FR', 'DE'],
        horizon: [2030, 2028],
        brand: ['Aurora Mobility'],
        model: ['A1 Horizon'],
        evidence: ['demonstration'],
        page: 2,
        sort: 'brand',
        direction: 'asc',
      },
      invalidKeys: [],
    })
  })

  it('reports a partially invalid filter while preserving valid intent', () => {
    expect(
      parsePlannerSearch({ market: ['FR', 'obsolete'], horizon: [2030, 2035] }, options),
    ).toEqual({
      value: {
        market: ['FR'],
        horizon: [2030],
        page: 1,
        sort: 'base_demand',
        direction: 'desc',
      },
      invalidKeys: ['horizon', 'market'],
    })
  })

  it('serializes only canonical URL fields and omits route diagnostics', () => {
    const serialized = serializePlannerSearch({
      market: ['FR'],
      page: 1,
      sort: 'base_demand',
      direction: 'desc',
      invalidKeys: ['market'],
      legacyFilter: 'obsolete',
    })

    expect(serialized).toEqual({
      market: ['FR'],
      page: 1,
      sort: 'base_demand',
      direction: 'desc',
    })
  })
})
