import type { components } from '../../lib/api/schema'


export type EurocodeBreakdownData = components['schemas']['EurocodeBreakdownResponse']
export type EurocodeShare = components['schemas']['EurocodeShareResponse']
export type EurocodeReading = components['schemas']['EurocodeSearchResponse']

export const CHIP_LIMIT = 3
export const VEHICLE_LIMIT = 3

export function format(value: number): string {
  return new Intl.NumberFormat('en-US').format(value)
}

export function eurocodeYears(share: Pick<EurocodeShare, 'first_year' | 'last_year'>): string {
  return `${share.first_year}–${share.last_year ?? ''}`
}

/** The codes (or uncoded vehicle) a code's shared units may belong to instead. */
export function sharedWithText(share: Pick<EurocodeShare, 'shared_with' | 'shared_with_uncoded'>): string {
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

/** "Eurocode Q201: Aurora Mobility A1 Horizon I, 2020–2025", or the bare code if no vehicle carries it. */
export function eurocodeReadingText(reading: EurocodeReading): string {
  if (reading.label == null || reading.first_year == null) return `Eurocode ${reading.code}`
  return `Eurocode ${reading.code}: ${reading.label}, ${eurocodeYears({ first_year: reading.first_year, last_year: reading.last_year ?? null })}`
}

export type EurocodeGroup = components['schemas']['EurocodeGroupResponse']

const DRIVE_TEXT: Record<string, string> = {
  LHD: 'Left-hand drive',
  RHD: 'Right-hand drive',
  both: 'Left- and right-hand drive',
}

/** The catalogue's drive side (LHD, RHD or both) in words. */
export function driveText(drive: string): string {
  return DRIVE_TEXT[drive] ?? drive
}

/** "Q201 · Aurora Mobility A1 Horizon I · 2020–2025": the heading of a Eurocode row. */
export function eurocodeGroupTitle(group: EurocodeGroup): string {
  return `${group.code} · ${group.label} · ${eurocodeYears(group)}`
}
