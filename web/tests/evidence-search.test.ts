import { describe, expect, it } from 'vitest'

import { parseEvidenceSearch } from '../src/lib/evidence-search'


describe('parseEvidenceSearch', () => {
  it('keeps bounded review filters and normalizes the page', () => {
    expect(parseEvidenceSearch({
      releaseId: 'uk-release', geography: 'United Kingdom', measure: 'new_registrations',
      mappingStatus: 'unresolved', search: 'acme', page: '2',
    })).toEqual({
      releaseId: 'uk-release', geography: 'United Kingdom', measure: 'new_registrations',
      mappingStatus: 'unresolved', search: 'acme', page: 2,
    })
  })

  it('drops invalid, overlong, and non-positive values', () => {
    expect(parseEvidenceSearch({
      releaseId: 'x'.repeat(81), measure: 'forecast', mappingStatus: 'canonical',
      search: 'x'.repeat(101), page: -3,
    })).toEqual({ page: 1 })
  })
})
