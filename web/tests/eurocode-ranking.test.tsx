import { QueryClient } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { AppProviders } from '../src/app/providers'
import { OpportunitiesWorkbench } from '../src/features/opportunities/OpportunitiesPage'
import { OpportunityDetailView } from '../src/features/opportunities/OpportunityDetailPage'
import { PlannerApiClient } from '../src/lib/api/client'
import { contributions, eurocodeStatus, json, opportunities, successFetcher } from './opportunity-fixtures'


const group = {
  code: 'Q201', label: 'Aurora Mobility A1 Horizon I', make: 'Aurora Mobility',
  first_year: 2020, last_year: 2025, drive: 'LHD',
  shared_units: 400, shared_with: ['Q202'], shared_with_uncoded: false,
  vehicles: [
    { brand: 'Aurora Mobility', model: 'A1 Horizon', certain_units: 1500, shared_units: 300 },
    { brand: 'Aurora Mobility', model: 'A1 Sport', certain_units: 200, shared_units: 100 },
    { brand: 'Meridian Motors', model: 'M5 Vector', certain_units: 50, shared_units: 0 },
    { brand: 'Velora Works', model: 'V3 Current', certain_units: 10, shared_units: 0 },
  ],
  evidence: { stock: 135, registrations: 900, interpolated: 0, forecast: 465, unknown: 0 },
} as const

const eurocodeRow = {
  ...opportunities.items[0],
  group_id: 'eurocode-Q201',
  group_by: 'eurocode',
  model: group.label,
  model_year: null,
  generation_name: null,
  eurocodes: null,
  demand: { downside_units: 1300, base_units: 1500, upside_units: 1700 },
  eurocode: group,
}

const eurocodePage = {
  ...opportunities,
  items: [eurocodeRow],
  eurocode_totals: {
    codes: [], single_units: 1500, shared_units: 400,
    outside_units: 100, unassignable_units: 0, unreviewed_units: 150,
  },
}

function fetcherFor(page: unknown, detail: unknown = eurocodeRow) {
  const base = successFetcher()
  const baseImpl = base.getMockImplementation()!
  base.mockImplementation((input, init) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    if (url.includes('/eurocodes/status')) return Promise.resolve(json(eurocodeStatus))
    if (url.includes('/opportunities/eurocode-Q201/fleet')) return Promise.resolve(json([]))
    if (url.includes('/opportunities/eurocode-Q201/contributions')) {
      return Promise.resolve(json(contributions.map((item) => ({
        ...item, generation: group.label, body_style: 'Every body the code fits',
      }))))
    }
    if (url.includes('/opportunities/eurocode-Q201')) return Promise.resolve(json(detail))
    if (url.includes('/opportunities?')) return Promise.resolve(json(page))
    return baseImpl(input, init)
  })
  return base
}

function renderEurocodes(fetcher: typeof fetch) {
  const onSearchChange = vi.fn()
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <AppProviders queryClient={queryClient}>
      <OpportunitiesWorkbench
        apiClient={new PlannerApiClient(fetcher)}
        onOpenDetails={vi.fn()}
        onSearchChange={onSearchChange}
        search={{ groupBy: 'eurocode', page: 1 }}
      />
    </AppProviders>,
  )
  return { onSearchChange }
}

describe('Eurocode ranking', () => {
  it('leads each row with its code, certain units, shared units and vehicles', async () => {
    renderEurocodes(fetcherFor(eurocodePage))

    const heading = await screen.findByRole('heading', { level: 3, name: 'Q201 · Aurora Mobility A1 Horizon I · 2020–2025' })
    expect(heading.querySelector('code')).toHaveTextContent('Q201')
    const card = heading.closest('li')!
    expect(within(card).getByText('1,500 certain replacements')).toBeVisible()
    expect(within(card).getByText('up to +400 shared with Q202')).toBeVisible()
    const vehicles = within(card).getByRole('group', { name: 'Vehicles that may carry this Eurocode' })
    expect(vehicles).toHaveTextContent('Aurora Mobility A1 Horizon · certain 1,500 · +300 shared')
    expect(vehicles).toHaveTextContent('Meridian Motors M5 Vector · certain 50')
    expect(vehicles).not.toHaveTextContent('Velora Works V3 Current')
    expect(within(card).getByText('+1 more vehicle')).toBeVisible()
    expect(within(card).queryByText('Generation review pending')).not.toBeInTheDocument()
    expect(within(card).queryByRole('group', { name: 'Eurocodes' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'View Q201 · Aurora Mobility A1 Horizon I · 2020–2025 details' })).toBeVisible()
    // What the certain units rest on sits with them, beside the assumption range.
    expect(within(card).getByText('1,300–1,700 assumption range')).toBeVisible()
    expect(within(card).getByText(
      'Rests on: 9% national registers · 60% new-car registrations × survival · 31% forecast registrations',
    )).toBeVisible()
    expect(screen.getByText(/assumed replacement rate of 4.3% a year ±20%.*not measured errors/)).toBeVisible()
  })

  it('says nothing about evidence a source does not record', async () => {
    const row = { ...eurocodeRow, eurocode: { ...group, evidence: { ...group.evidence, stock: 0, registrations: 0, forecast: 0, unknown: 1500 } } }
    renderEurocodes(fetcherFor({ ...eurocodePage, items: [row] }))

    expect(await screen.findByText('1,500 certain replacements')).toBeVisible()
    expect(screen.queryByText(/Rests on/)).not.toBeInTheDocument()
  })

  it('says a shared-only partner without a code, and shows the totals strip', async () => {
    const row = { ...eurocodeRow, eurocode: { ...group, shared_with: [], shared_with_uncoded: true } }
    renderEurocodes(fetcherFor({ ...eurocodePage, items: [row] }))

    expect(await screen.findByText('up to +400 shared with a vehicle without a code')).toBeVisible()
    const totals = screen.getByRole('group', { name: 'Eurocode totals' })
    expect(totals).toHaveTextContent('Certain on the codes listed: 1,500')
    expect(totals).toHaveTextContent('Shared between codes, counted once: 400')
    expect(totals).toHaveTextContent('Not assigned to a Eurocode: 250')
    expect(totals).toHaveTextContent('no code covers this registration year: 100; not reviewed yet: 150')
    expect(screen.getByText(/rank on the units the code certainly carries/)).toBeVisible()
    expect(screen.getByText(/compares each Eurocode with/)).toBeVisible()
    expect(screen.getByText('Certain replacements')).toBeVisible()
    expect(screen.queryByText('Forecast replacements')).not.toBeInTheDocument()
    expect(screen.getByPlaceholderText('Eurocode, brand or model, for example Golf')).toBeVisible()
  })

  it('offers model years when the service has no Eurocodes', async () => {
    const base = successFetcher()
    const baseImpl = base.getMockImplementation()!
    base.mockImplementation((input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
      if (url.includes('/opportunities?')) {
        return Promise.resolve(new Response(JSON.stringify({
          code: 'eurocodes_not_loaded', message: 'No Eurocode map is loaded in this service.',
          correlation_id: 'corr-1',
        }), { status: 422, headers: { 'Content-Type': 'application/problem+json' } }))
      }
      return baseImpl(input, init)
    })
    const { onSearchChange } = renderEurocodes(base)

    expect(await screen.findByRole('heading', { name: 'Eurocodes are not loaded in this service' })).toBeVisible()
    expect(screen.queryByText('We could not load this ranking')).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Show model years' }))
    expect(onSearchChange).toHaveBeenCalledWith({ groupBy: 'model_year', page: 1 })
  })

  it('orders the grouping control Eurocodes first', async () => {
    renderEurocodes(fetcherFor(eurocodePage))
    const control = await screen.findByRole('group', { name: 'Opportunity grouping' })
    expect(within(control).getAllByRole('button').map((button) => button.textContent))
      .toEqual(['Eurocodes', 'Model years', 'Models', 'Brands'])
    expect(within(control).getByRole('button', { name: 'Eurocodes' })).toHaveAttribute('aria-pressed', 'true')
  })
})

describe('Eurocode detail', () => {
  it('tabulates the vehicles, drops the fleet and words contributions as certain units', async () => {
    const fetcher = fetcherFor(eurocodePage)
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <AppProviders queryClient={queryClient}>
        <OpportunityDetailView apiClient={new PlannerApiClient(fetcher)} groupId="eurocode-Q201" search={{ groupBy: 'eurocode', page: 1 }} />
      </AppProviders>,
    )

    expect(await screen.findByRole('heading', { level: 2, name: 'Q201 · Aurora Mobility A1 Horizon I · 2020–2025' })).toBeVisible()
    expect(screen.getByText('Aurora Mobility · Left-hand drive')).toBeVisible()
    expect(screen.getByText(/The score compares this Eurocode with the whole market/)).toBeVisible()
    expect(screen.getByText(/ranked Eurocodes in the market — rank #/)).toBeVisible()
    expect(screen.getByRole('heading', { name: 'Vehicles that may carry Q201' })).toBeVisible()
    expect(screen.getByText(/Shared units are never split/)).toBeVisible()
    expect(screen.getByRole('row', { name: /Aurora Mobility A1 Horizon.*1,500.*\+300/ })).toBeVisible()
    expect(screen.getByRole('row', { name: /Meridian Motors M5 Vector.*50.*—/ })).toBeVisible()
    expect(screen.getByText('Certain units (base forecast)')).toBeVisible()
    expect(screen.getByText('Assumption low')).toBeVisible()
    expect(screen.getByText('Assumption high')).toBeVisible()
    expect(screen.getByText(/Ranges show only the survival-curve band.*backtests show real errors are larger/)).toBeVisible()
    expect(screen.getByRole('heading', { name: 'What the 1,500 certain units rest on' })).toBeVisible()
    const evidence = screen.getByRole('region', { name: 'What the certain units of Q201 rest on' })
    expect(within(evidence).getByRole('row', { name: /National registers 135 9% Counted on the road by a national vehicle register/ })).toBeVisible()
    expect(within(evidence).getByRole('row', { name: /New-car registrations × survival 900 60%/ })).toBeVisible()
    expect(within(evidence).getByRole('row', { name: /Forecast registrations 465 31%/ })).toBeVisible()
    expect(within(evidence).queryByText(/Interpolated|Not recorded/)).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Estimated active fleet' })).not.toBeInTheDocument()
    expect(screen.queryByText('Total estimated fleet')).not.toBeInTheDocument()
    expect(screen.getByText('The surviving fleet is not broken down by Eurocode yet.')).toBeVisible()
    expect(screen.queryByRole('heading', { name: 'Eurocodes' })).not.toBeInTheDocument()
    expect(await screen.findByText('Certain units of Q201')).toBeVisible()
    expect(screen.queryByText(/Every body the code fits/)).not.toBeInTheDocument()
  })
})
