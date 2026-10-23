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

export type EvidenceMix = components['schemas']['EvidenceMixResponse']

/** What a code's certain units can rest on, strongest evidence first. */
export const EVIDENCE_KINDS = [
  {
    key: 'stock',
    label: 'national registers',
    meaning: 'Counted on the road by a national vehicle register, then aged to the forecast year with the survival curve.',
  },
  {
    key: 'registrations',
    label: 'new-car registrations × survival',
    meaning: 'Published new-car registrations, times the share of each year’s cars the survival curve keeps on the road.',
  },
  {
    key: 'interpolated',
    label: 'interpolated registrations',
    meaning: 'New-car registrations estimated between two published years, times the survival curve.',
  },
  {
    key: 'forecast',
    label: 'forecast registrations',
    meaning: 'Registrations of years not published yet, forecast from earlier years, times the survival curve.',
  },
  {
    key: 'unknown',
    label: 'not recorded',
    meaning: 'The source does not record what these units rest on.',
  },
] as const

export type EvidenceShare = (typeof EVIDENCE_KINDS)[number] & { units: number; percent: string }

/**
 * Each kind's share of the certain units, as whole percentages that add up to 100
 * (largest remainder). A kind with units but too few to reach 1% reads "<1%"; a
 * kind with none is left out. Nothing is returned when no kind is recorded.
 */
export function evidenceShares(mix: EvidenceMix): EvidenceShare[] {
  const total = EVIDENCE_KINDS.reduce((sum, kind) => sum + mix[kind.key], 0)
  if (total === 0 || mix.unknown === total) return []
  const exact = EVIDENCE_KINDS.map((kind) => (mix[kind.key] * 100) / total)
  const whole = exact.map(Math.floor)
  let left = 100 - whole.reduce((sum, value) => sum + value, 0)
  const byRemainder = exact
    .map((value, index) => ({ index, remainder: value - whole[index] }))
    .sort((first, second) => second.remainder - first.remainder || first.index - second.index)
  for (const { index } of byRemainder) {
    if (left === 0) break
    whole[index] += 1
    left -= 1
  }
  return EVIDENCE_KINDS.flatMap((kind, index) => mix[kind.key] > 0
    ? [{ ...kind, units: mix[kind.key], percent: whole[index] === 0 ? '<1%' : `${whole[index]}%` }]
    : [])
}

/** "Rests on: 9% national registers · 60% new-car registrations × survival · 31% forecast registrations". */
export function evidenceText(shares: EvidenceShare[]): string {
  return `Rests on: ${shares.map((share) => `${share.percent} ${share.label}`).join(' · ')}`
}
