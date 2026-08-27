import { describe, expect, it } from 'vitest'

import { parseRegistrationSearch } from '../src/lib/registration-search'


describe('parseRegistrationSearch', () => {
  it('keeps bounded official registration state and normalizes the page', () => {
    expect(parseRegistrationSearch({ search: 'Alfa Romeo', page: '2' })).toEqual({
      search: 'Alfa Romeo', page: 2,
    })
  })

  it('drops overlong search and non-positive pages', () => {
    expect(parseRegistrationSearch({ search: 'x'.repeat(101), page: -3 })).toEqual({ page: 1 })
  })
})
