import { describe, expect, it } from 'vitest'

import { evidenceShares, evidenceText } from '../src/features/eurocodes/eurocode-text'

const none = { stock: 0, registrations: 0, interpolated: 0, forecast: 0, unknown: 0 }

function percents(mix: typeof none) {
  return evidenceShares(mix).map((share) => [share.key, share.percent])
}

describe('evidenceShares', () => {
  it('rounds to whole percentages that add up to 100, largest remainder first', () => {
    expect(percents({ ...none, stock: 1, registrations: 1, forecast: 1 })).toEqual([
      ['stock', '34%'], ['registrations', '33%'], ['forecast', '33%'],
    ])
    expect(percents({ ...none, registrations: 2, interpolated: 3, forecast: 5 })).toEqual([
      ['registrations', '20%'], ['interpolated', '30%'], ['forecast', '50%'],
    ])
  })

  it('never shows a kind with units as 0%, and leaves out kinds without units', () => {
    expect(percents({ ...none, stock: 1, registrations: 999 })).toEqual([
      ['stock', '<1%'], ['registrations', '100%'],
    ])
  })

  it('names unrecorded units beside recorded ones, and says nothing when none are recorded', () => {
    expect(percents({ ...none, registrations: 3, unknown: 1 })).toEqual([
      ['registrations', '75%'], ['unknown', '25%'],
    ])
    expect(evidenceShares({ ...none, unknown: 5 })).toEqual([])
    expect(evidenceShares(none)).toEqual([])
  })

  it('reads as one line', () => {
    expect(evidenceText(evidenceShares({ ...none, stock: 1, forecast: 3 }))).toBe(
      'Rests on: 25% national registers · 75% forecast registrations',
    )
  })
})
