import { describe, expect, it } from 'vitest'

import { ordinal } from '../src/lib/ordinal'

describe('ordinal', () => {
  it('reads like English, teens included', () => {
    expect([0, 1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 92, 100, 101, 111, 112].map(ordinal)).toEqual([
      '0th', '1st', '2nd', '3rd', '4th', '11th', '12th', '13th', '21st', '22nd', '23rd', '92nd',
      '100th', '101st', '111th', '112th',
    ])
  })
})
