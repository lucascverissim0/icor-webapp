import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import {
  EurocodeCompact,
  EurocodeNotLoaded,
  EurocodeSummary,
  EurocodeTable,
} from '../src/features/eurocodes/EurocodeBreakdown'
import type { EurocodeBreakdownData } from '../src/features/eurocodes/eurocode-text'


function share(code: string, extra: Partial<EurocodeBreakdownData['codes'][number]> = {}) {
  return {
    code, label: `Aurora Mobility A1 ${code}`, first_year: 2020, last_year: null, drive: 'LHD',
    certain_units: 400, shared_units: 0, shared_with: [], shared_with_uncoded: false, ...extra,
  }
}

const breakdown: EurocodeBreakdownData = {
  codes: [
    share('Q201', { last_year: 2025, shared_units: 100, shared_with: ['Q202'] }),
    share('Q202', { first_year: 2025, certain_units: 0, shared_units: 100, shared_with: ['Q201'], shared_with_uncoded: true }),
  ],
  single_units: 400, shared_units: 100, outside_units: 30, unassignable_units: 20, unreviewed_units: 0,
}

describe('EurocodeBreakdown', () => {
  it('shows certain and shared units without splitting them, and the unassigned units', () => {
    render(<EurocodeSummary breakdown={breakdown} />)

    expect(screen.getByText('Q201')).toBeVisible()
    expect(screen.getByText(/Aurora Mobility A1 Q201 · 2020–2025/)).toBeVisible()
    expect(screen.getByText(/Aurora Mobility A1 Q202 · 2025–$/)).toBeVisible()
    expect(screen.getByText('certain 400')).toBeVisible()
    expect(screen.getByText(/up to \+100 shared with Q202$/)).toBeVisible()
    expect(screen.getByText('up to 100, shared with Q201 or a vehicle without a code')).toBeVisible()
    expect(screen.queryByText('certain 0')).not.toBeInTheDocument()
    expect(screen.getByText('Not assigned to a Eurocode: 50')).toBeVisible()
    expect(screen.getByText('(no code covers this registration year: 30; the label names no single catalogued vehicle: 20)')).toBeVisible()
    expect(screen.queryByText(/unreviewed|not reviewed yet/)).not.toBeInTheDocument()
  })

  it('limits chips to three and counts the rest', () => {
    const many = { ...breakdown, codes: ['Q201', 'Q202', 'Q203', 'Q204', 'Q205'].map((code) => share(code)) }
    render(<EurocodeSummary breakdown={many} />)

    expect(screen.getAllByRole('listitem')).toHaveLength(3)
    expect(screen.getByText('+2 more codes')).toBeVisible()
  })

  it('says so when a vehicle has no code, and when only an uncoded vehicle is shared', () => {
    const none: EurocodeBreakdownData = { ...breakdown, codes: [], outside_units: 0, unassignable_units: 0, unreviewed_units: 12 }
    const { unmount } = render(<EurocodeSummary breakdown={none} />)
    expect(screen.getByText('No Eurocode assigned')).toBeVisible()
    expect(screen.getByText('Not assigned to a Eurocode: 12')).toBeVisible()
    unmount()

    render(<EurocodeCompact breakdown={{ ...breakdown, codes: [share('Q301', { shared_units: 5, shared_with_uncoded: true })] }} />)
    expect(screen.getByText(/up to \+5 shared with a vehicle without a code/)).toBeVisible()
  })

  it('renders the detail table with unassigned rows by reason', () => {
    render(<EurocodeTable breakdown={breakdown} />)

    const table = screen.getByRole('table')
    expect(within(table).getAllByRole('columnheader').map((cell) => cell.textContent)).toEqual([
      'Code', 'Vehicle', 'Registration years', 'Certain', 'Up to (shared)', 'Shared with',
    ])
    expect(within(table).getByRole('row', { name: /Q201.*2020–2025.*400.*\+100.*Q202/ })).toBeVisible()
    expect(within(table).getByRole('row', { name: /Not assigned: no code covers this registration year.*30/ })).toBeVisible()
    expect(within(table).getByRole('row', { name: /Not assigned: the label names no single catalogued vehicle.*20/ })).toBeVisible()
  })

  it('shows a calm not-loaded notice with the status warnings', () => {
    render(<EurocodeNotLoaded warnings={['Decisions snapshot differs.']} />)

    expect(screen.getByRole('note')).toHaveTextContent('Eurocodes are not loaded')
    expect(screen.getByText('Decisions snapshot differs.')).toBeVisible()
  })
})
