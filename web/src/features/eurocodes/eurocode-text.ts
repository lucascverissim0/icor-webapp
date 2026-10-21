import type { components } from '../../lib/api/schema'


export type EurocodeBreakdownData = components['schemas']['EurocodeBreakdownResponse']
export type EurocodeShare = components['schemas']['EurocodeShareResponse']

export const CHIP_LIMIT = 3

export function format(value: number): string {
  return new Intl.NumberFormat('en-US').format(value)
}

export function eurocodeYears(share: Pick<EurocodeShare, 'first_year' | 'last_year'>): string {
  return `${share.first_year}–${share.last_year ?? ''}`
}

/** The codes (or uncoded vehicle) a code's shared units may belong to instead. */
export function sharedWithText(share: EurocodeShare): string {
  const codes = share.shared_with.join(', ')
  if (!share.shared_with_uncoded) return codes
  return codes ? `${codes} or a vehicle without a code` : 'a vehicle without a code'
}

const UNASSIGNED_REASONS = [
  ['outside_units', 'outside', 'no code covers this registration year'],
  ['unassignable_units', 'unassignable', 'the label names no single catalogued vehicle'],
  ['unreviewed_units', 'unreviewed', 'not reviewed yet'],
] as const

export function unassignedParts(breakdown: EurocodeBreakdownData) {
  const parts = UNASSIGNED_REASONS.map(([field, key, reason]) => ({
    key, reason, units: breakdown[field],
  }))
  return {
    total: parts.reduce((sum, part) => sum + part.units, 0),
    reasons: parts.filter((part) => part.units > 0),
  }
}
