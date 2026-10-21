import { QueryClient } from '@tanstack/react-query'
import axe from 'axe-core'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { AppProviders } from '../src/app/providers'
import { OpportunitiesWorkbench } from '../src/features/opportunities/OpportunitiesPage'
import { OpportunityDetailView } from '../src/features/opportunities/OpportunityDetailPage'
import { PlannerApiClient } from '../src/lib/api/client'
import {
  eurocodeStatus,
  json,
  opportunities,
  renderOpportunities,
  successFetcher,
} from './opportunity-fixtures'


describe('OpportunitiesWorkbench', () => {
  it('removes grouping and coverage mutations from the verified client release', async () => {
    const client = new PlannerApiClient(successFetcher())
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <AppProviders queryClient={queryClient}>
        <OpportunitiesWorkbench
          apiClient={client}
          clientRelease
          onOpenDetails={vi.fn()}
          onSearchChange={vi.fn()}
          search={{ groupBy: 'model_year', page: 1 }}
        />
      </AppProviders>,
    )

    expect(await screen.findByText('2,150 replacements')).toBeVisible()
    expect(screen.getByText('Official vehicle-year evidence')).toBeVisible()
    expect(screen.getByText('Generation A')).toBeVisible()
    expect(screen.queryByText('Summarize ranking by')).not.toBeInTheDocument()
    expect(screen.queryByText('Manage ICOR worked-model coverage')).not.toBeInTheDocument()
  })

  it('leads with model, year, generation, forecast demand, and transparent scoring', async () => {
    renderOpportunities(successFetcher())

    expect(await screen.findByText('2,150 replacements')).toHaveClass('opportunity-demand__base')
    expect(screen.getByText('Score 100.0')).toHaveAccessibleDescription(
      /100th demand percentile/i,
    )
    expect(screen.getByRole('heading', { name: /Aurora Mobility.*A1 Horizon.*2025/i })).toBeVisible()
    expect(screen.getByText('Generation A')).toBeVisible()
    expect(screen.getByText('ICOR worked', { selector: '.icor-experience' })).toBeVisible()
    expect(screen.getByText(/100th demand percentile/i)).toBeVisible()
    expect(screen.getByRole('heading', { name: 'How the opportunity score is calculated' })).toBeVisible()
    expect(screen.getByText(/Score = demand percentile × 100/i)).toBeVisible()
    expect(screen.queryByText(/readiness/i)).not.toBeInTheDocument()
    expect(screen.getByText(/Registration evidence through 2025.*provisional/i)).toBeVisible()
    expect(screen.getByText('Validated snapshot')).toBeVisible()
    const availableData = screen.getByLabelText('Available opportunity data')
    expect(availableData).toHaveTextContent('2 ranked records available')
    expect(availableData).toHaveTextContent('100 official model labels')
    expect(availableData).toHaveTextContent('10,800,000 registrations represented')
  })

  it('leads each card with its Eurocode and reports the status coverage', async () => {
    renderOpportunities(successFetcher())

    expect(await screen.findByText('Q201')).toBeVisible()
    expect(screen.getByText(/up to \+400 shared with Q202/)).toBeVisible()
    expect(screen.getByText('Not assigned to a Eurocode: 250')).toBeVisible()
    expect(await screen.findByText(/95\.0% of forecast demand assigned/)).toHaveTextContent('(75.0% to a single code)')
    expect(screen.queryByText(/Eurocodes are not loaded/)).not.toBeInTheDocument()
  })

  it('shows one calm notice and no Eurocode lines when Eurocodes are not loaded', async () => {
    const fetcher = successFetcher()
    const base = fetcher.getMockImplementation()!
    fetcher.mockImplementation((input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
      if (url.includes('/eurocodes/status')) return Promise.resolve(json({ ...eurocodeStatus, loaded: false, coverage: [], warnings: ['Map missing.'] }))
      if (url.includes('/opportunities?')) {
        return Promise.resolve(json({ ...opportunities, eurocodes_loaded: false, items: [{ ...opportunities.items[0], eurocodes: null }] }))
      }
      return base(input, init)
    })
    renderOpportunities(fetcher)

    expect(await screen.findByText('2,150 replacements')).toBeVisible()
    expect(await screen.findByText('Map missing.')).toBeVisible()
    expect(screen.getAllByText(/Eurocodes are not loaded/)).toHaveLength(1)
    expect(screen.queryByText('Q201')).not.toBeInTheDocument()
  })

  it('reads a search that is exactly a Eurocode both ways and leads with the searched chip', async () => {
    const fetcher = successFetcher()
    const base = fetcher.getMockImplementation()!
    const [first] = opportunities.items
    const carried = {
      ...opportunities,
      eurocode_search: { code: 'Q202', label: 'Aurora Mobility A1 Horizon II', first_year: 2025, last_year: null },
      items: [{
        ...first,
        eurocodes: {
          ...first.eurocodes,
          codes: [
            ...first.eurocodes.codes,
            ...['Q203', 'Q204'].map((code) => ({ ...first.eurocodes.codes[0], code, shared_units: 0, shared_with: [] })),
            { ...first.eurocodes.codes[0], code: 'Q202', certain_units: 0, shared_with: ['Q201'] },
          ],
        },
      }],
    }
    fetcher.mockImplementation((input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
      if (url.includes('/opportunities?')) return Promise.resolve(json(carried))
      return base(input, init)
    })
    const client = new PlannerApiClient(fetcher)
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <AppProviders queryClient={queryClient}>
        <OpportunitiesWorkbench apiClient={client} onOpenDetails={vi.fn()} onSearchChange={vi.fn()} search={{ groupBy: 'model_year', page: 1, q: 'q202' }} />
      </AppProviders>,
    )

    expect(await screen.findByText(/“q202” is read two ways: as Eurocode Q202: Aurora Mobility A1 Horizon II, 2025–, and as text in vehicle names/)).toBeVisible()
    // The searched code is the fourth code, yet it leads the three chips shown.
    const chips = within(screen.getByRole('group', { name: 'Eurocodes' })).getAllByRole('listitem')
    expect(chips).toHaveLength(3)
    expect(chips[0]).toHaveTextContent(/^Q202Searched code/)
    expect(chips[0]).toHaveClass('eurocode-chip--searched')
    expect(screen.getByText('+1 more code')).toBeVisible()
  })

  it('says when a searched Eurocode is carried by no reviewed vehicle', async () => {
    const fetcher = successFetcher()
    const base = fetcher.getMockImplementation()!
    fetcher.mockImplementation((input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
      if (url.includes('/opportunities?')) {
        return Promise.resolve(json({ ...opportunities, eurocode_search: { code: 'Q999', label: null, first_year: null, last_year: null } }))
      }
      return base(input, init)
    })
    const client = new PlannerApiClient(fetcher)
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <AppProviders queryClient={queryClient}>
        <OpportunitiesWorkbench apiClient={client} onOpenDetails={vi.fn()} onSearchChange={vi.fn()} search={{ groupBy: 'model_year', page: 1, q: 'Q999' }} />
      </AppProviders>,
    )

    expect(await screen.findByText(/“Q999” is Eurocode Q999 in ICOR's catalogue, but no reviewed vehicle carries it/)).toBeVisible()
    expect(screen.queryByText('Searched code')).not.toBeInTheDocument()
  })

  it('stays quiet when the status endpoint fails', async () => {
    const fetcher = successFetcher()
    const base = fetcher.getMockImplementation()!
    fetcher.mockImplementation((input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
      if (url.includes('/eurocodes/status')) return Promise.resolve(json({ detail: 'boom' }, 500))
      return base(input, init)
    })
    renderOpportunities(fetcher)

    expect(await screen.findByText('2,150 replacements')).toBeVisible()
    expect(screen.queryByText(/of forecast demand assigned/)).not.toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('switches grouping without discarding market intent', async () => {
    const user = userEvent.setup()
    const fetcher = successFetcher()
    const client = new PlannerApiClient(fetcher)
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const onSearchChange = vi.fn()
    render(
      <AppProviders queryClient={queryClient}>
        <OpportunitiesWorkbench apiClient={client} onOpenDetails={vi.fn()} onSearchChange={onSearchChange} search={{ groupBy: 'brand', market: ['FR'], page: 1 }} />
      </AppProviders>,
    )

    await user.click(await screen.findByRole('button', { name: 'Model years' }))

    expect(onSearchChange).toHaveBeenCalledWith({ groupBy: 'model_year', market: ['FR'], page: 1 })
  })

  it('navigates ranking pages and keeps filters', async () => {
    const user = userEvent.setup()
    const fetcher = successFetcher()
    const client = new PlannerApiClient(fetcher)
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const onSearchChange = vi.fn()
    render(<AppProviders queryClient={queryClient}><OpportunitiesWorkbench apiClient={client} onOpenDetails={vi.fn()} onSearchChange={onSearchChange} search={{ groupBy: 'model_year', market: ['FR'], page: 1 }} /></AppProviders>)

    await screen.findByRole('button', { name: 'Next page' })
    expect(fetcher.mock.calls.some(([input]) => {
      const url = typeof input === 'string'
        ? input
        : input instanceof URL ? input.href : input.url
      return url.includes('page_size=100')
    })).toBe(true)

    await user.click(screen.getByRole('button', { name: 'Next page' }))
    expect(onSearchChange).toHaveBeenCalledWith({ groupBy: 'model_year', market: ['FR'], page: 2 })

    await user.click(screen.getByRole('button', { name: 'Last page' }))
    expect(onSearchChange).toHaveBeenCalledWith({ groupBy: 'model_year', market: ['FR'], page: 2 })
  })

  it('opens a dedicated detail page for a ranking', async () => {
    const user = userEvent.setup()
    const client = new PlannerApiClient(successFetcher())
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const onOpenDetails = vi.fn()
    render(<AppProviders queryClient={queryClient}><OpportunitiesWorkbench apiClient={client} onOpenDetails={onOpenDetails} onSearchChange={vi.fn()} search={{ groupBy: 'model_year', page: 1 }} /></AppProviders>)

    await user.click(await screen.findByRole('button', { name: /View Aurora Mobility.*details/i }))
    expect(onOpenDetails).toHaveBeenCalledWith('brand-aurora')
    expect(queryClient.getQueryData([
      'opportunities',
      'brand-aurora',
      { groupBy: 'model_year', markets: undefined, horizons: undefined },
    ])).toMatchObject({ group_id: 'brand-aurora', brand: 'Aurora Mobility' })
  })

  it('has no automated accessibility violations in the ranked state', async () => {
    const { container } = renderOpportunities(successFetcher())
    await screen.findByText('2,150 replacements')

    expect((await axe.run(container)).violations).toEqual([])
  })
})

describe('OpportunityDetailView', () => {
  it('explains the selected rank and its contributing forecasts on a dedicated page', async () => {
    const client = new PlannerApiClient(successFetcher())
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <AppProviders queryClient={queryClient}>
        <OpportunityDetailView
          apiClient={client}
          groupId="brand-aurora"
          search={{ groupBy: 'model_year', market: ['FR'], horizon: [2030], page: 1 }}
        />
      </AppProviders>,
    )

    expect(await screen.findByRole('heading', { name: /Aurora Mobility.*A1 Horizon.*2025/i })).toBeVisible()
    expect(screen.getByText('Generation A')).toBeVisible()
    expect(screen.getByText('100.0')).toBeVisible()
    expect(screen.getByText(/2,150/)).toBeVisible()
    expect(await screen.findByRole('heading', { name: 'Estimated active fleet' })).toBeVisible()
    expect(screen.getByText('62,000 vehicles')).toBeVisible()
    expect(screen.getByText('53,000 vehicles')).toBeVisible()
    expect(screen.getAllByText('North America')).toHaveLength(2)
    expect(screen.getByRole('progressbar', { name: 'Demand score' })).toHaveAttribute('value', '100')
    expect(await screen.findByRole('heading', { name: 'FR · 2030' })).toBeVisible()
    expect(screen.getByRole('link', { name: /Back to opportunity ranking/i })).toHaveAttribute(
      'href',
      expect.stringContaining('market=FR'),
    )
    expect(screen.getByRole('heading', { name: 'Eurocodes' })).toBeVisible()
    expect(screen.getByText(/Shared units are never split/)).toBeVisible()
    expect(screen.getByRole('row', { name: /Q201.*2020–2025.*1,500.*\+400.*Q202/ })).toBeVisible()
    expect(screen.getByRole('row', { name: /Not assigned: not reviewed yet.*150/ })).toBeVisible()
  })

  it('omits the Eurocodes section when none are loaded', async () => {
    const fetcher = successFetcher()
    const base = fetcher.getMockImplementation()!
    fetcher.mockImplementation((input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
      if (url.includes('/opportunities/brand-aurora') && !url.includes('/fleet') && !url.includes('/contributions')) {
        return Promise.resolve(json({ ...opportunities.items[0], eurocodes: null }))
      }
      return base(input, init)
    })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <AppProviders queryClient={queryClient}>
        <OpportunityDetailView apiClient={new PlannerApiClient(fetcher)} groupId="brand-aurora" search={{ groupBy: 'model_year', page: 1 }} />
      </AppProviders>,
    )

    expect(await screen.findByRole('heading', { name: /Aurora Mobility.*A1 Horizon.*2025/i })).toBeVisible()
    expect(screen.queryByRole('heading', { name: 'Eurocodes' })).not.toBeInTheDocument()
  })
})

describe('OpportunitiesWorkbench filters', () => {
  it('offers only the markets the snapshot actually holds', async () => {
    const { findByRole } = renderOpportunities(successFetcher())

    const markets = await findByRole('group', { name: /markets/i })
    for (const market of ['DE', 'FR', 'GB']) {
      expect(within(markets).getByRole('checkbox', { name: market })).toBeTruthy()
    }
    expect(within(markets).queryByRole('checkbox', { name: 'ZZ' })).toBeNull()
  })

  it('offers only the forecast years the snapshot actually holds', async () => {
    const { findByLabelText } = renderOpportunities(successFetcher())

    // The filters render before the ranking resolves, so the option list is
    // empty until the snapshot's facets arrive.
    await screen.findByRole('option', { name: '2028' })

    const select = await findByLabelText('Forecast year')
    const years = within(select as HTMLSelectElement)
      .getAllByRole('option')
      .map((option) => option.textContent)
    expect(years).toEqual(['All forecast years', '2028', '2031'])
  })

  it('selecting a market returns to the first page', async () => {
    const { findByRole, onSearchChange } = renderOpportunities(successFetcher())
    const markets = await findByRole('group', { name: /markets/i })

    await userEvent.click(within(markets).getByRole('checkbox', { name: 'FR' }))

    expect(onSearchChange).toHaveBeenCalledWith(
      expect.objectContaining({ market: ['FR'], page: 1 }),
    )
  })

  it('reordering the ranking returns to the first page', async () => {
    const { findByLabelText, onSearchChange } = renderOpportunities(successFetcher())
    const select = await findByLabelText('Order by')

    await userEvent.selectOptions(select, 'demand')

    expect(onSearchChange).toHaveBeenCalledWith(
      expect.objectContaining({ order: 'demand', page: 1 }),
    )
  })

  it('does not request a page per keystroke', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const { findByLabelText, onSearchChange } = renderOpportunities(successFetcher())
      const box = await findByLabelText('Search vehicle')

      await userEvent.type(box, 'Golf')
      expect(onSearchChange).not.toHaveBeenCalled()

      await vi.advanceTimersByTimeAsync(400)
      expect(onSearchChange).toHaveBeenCalledTimes(1)
      expect(onSearchChange).toHaveBeenCalledWith(
        expect.objectContaining({ q: 'Golf', page: 1 }),
      )
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('score basis disclosure', () => {
  it('names the whole market as what a score is measured against', async () => {
    renderOpportunities(successFetcher())

    const note = await screen.findByRole('note')
    expect(note).toHaveTextContent(/whole-market/i)
    // The population arrives with the ranking, so the claim is only complete
    // once the request has answered.
    await waitFor(() => expect(note).toHaveTextContent(/4,872 ranked model years/))
    expect(note).not.toHaveTextContent(/relative to the rows currently shown/i)
  })

  it('says the same thing when the ranking is narrowed', async () => {
    const client = new PlannerApiClient(successFetcher())
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <AppProviders queryClient={queryClient}>
        <OpportunitiesWorkbench
          apiClient={client}
          onOpenDetails={vi.fn()}
          onSearchChange={vi.fn()}
          search={{ groupBy: 'model_year', page: 1, q: 'Golf' }}
        />
      </AppProviders>,
    )

    const note = await screen.findByRole('note')
    await waitFor(() => expect(note).toHaveTextContent(/4,872 ranked model years/))
    expect(note).toHaveTextContent(/never the score/i)
  })
})
