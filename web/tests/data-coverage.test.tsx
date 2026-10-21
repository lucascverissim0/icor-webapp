// @vitest-environment happy-dom

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { AppShell } from '../src/app/AppShell'
import { parseDataCoverageSearch } from '../src/lib/data-coverage-search'
import {
  bigReport, fetcherFor, renderCoverage, report,
} from './data-coverage-fixtures'


const available = { available: true, message: null, report }

function codeColumn(): string[] {
  const table = screen.getByRole('table', { name: /Eurocodes with cars/ })
  return within(table).getAllByRole('row').slice(1).map(
    (row) => within(row).getAllByRole('rowheader')[0].textContent ?? '',
  )
}

describe('DataCoverageWorkbench', () => {
  it('shows the year, snapshot, key figures and tiers from the report', async () => {
    renderCoverage(fetcherFor(available))

    expect(await screen.findByRole('heading', { name: 'Cars on the road at end of 2026' })).toBeVisible()
    expect(screen.getByText('Snapshot demo-snapshot')).toBeVisible()
    const kpis = screen.getByRole('region', { name: 'Key figures' })
    expect(within(kpis).getByText('1,000')).toBeVisible()
    // (stock 200 + registrations 600) / fleet 1000
    expect(within(kpis).getByText('80.0%')).toBeVisible()
    expect(within(kpis).getByText('1,500 certain cars')).toBeVisible()
    expect(within(kpis).getByText('1 not reviewed · 1 reviewed')).toBeVisible()
    expect(within(kpis).getByText('2 codes')).toBeVisible()
  })

  it('builds the tier legend from the response thresholds', async () => {
    renderCoverage(fetcherFor({
      ...available,
      report: { ...report, thresholds: { ...report.thresholds, tier_a_markets: 7, tier_a_observed_share: 0.9 } },
    }))

    const legend = await screen.findByText(/cars observed in/)
    expect(legend).toHaveTextContent('≥90% of cars observed in ≥7 markets')
    expect(legend).toHaveTextContent('≥50%')
    expect(legend).toHaveTextContent('50% of its life is before 2010')
  })

  it('shows the message when no report exists', async () => {
    renderCoverage(fetcherFor({ available: false, message: 'Run scripts/report_eurocode_data.py first.', report: null }))

    expect(await screen.findByRole('heading', { name: 'No data report yet' })).toBeVisible()
    expect(screen.getByText('Run scripts/report_eurocode_data.py first.')).toBeVisible()
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('shows a calm state when Eurocodes are not loaded', async () => {
    renderCoverage(fetcherFor({
      code: 'eurocodes_not_loaded', message: 'Eurocodes are not loaded in this service.',
      correlation_id: 'corr-1',
    }, 422))

    expect(await screen.findByRole('heading', { name: 'Eurocodes are not loaded in this service' })).toBeVisible()
    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument()
  })

  it('offers a retry for other failures', async () => {
    renderCoverage(fetcherFor({ code: 'internal', message: 'Boom.', correlation_id: 'corr-2' }, 500))

    expect(await screen.findByRole('heading', { name: 'Data coverage could not be loaded' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeVisible()
  })

  it('does not request or show the report in the verified client release', () => {
    const fetcher = fetcherFor(available)
    renderCoverage(fetcher, { clientRelease: true })

    expect(screen.getByRole('heading', { name: 'This view is not part of the verified release' })).toBeVisible()
    expect(fetcher).not.toHaveBeenCalled()
  })

  it('lists countries by missing cars with the publisher territory and empty bands', async () => {
    renderCoverage(fetcherFor(available))

    const table = await screen.findByRole('table', { name: /Official cars on the road against the model/ })
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(2)
    expect(within(rows[0]).getByRole('rowheader')).toHaveTextContent('GB (UK)')
    expect(within(rows[1]).getByRole('rowheader')).toHaveTextContent('FR')
    expect(within(rows[0]).getByText('5,000')).toBeVisible()
    expect(within(rows[0]).getByText('40.0%')).toBeVisible()
    expect(within(rows[0]).getByText('— / 500')).toBeVisible()
    expect(within(rows[0]).getByText('1,000 / 900')).toBeVisible()
    expect(within(table).getAllByRole('columnheader').map((cell) => cell.textContent)).toEqual([
      'Market', 'Official year', 'Official cars', 'Model cars', 'Model share',
      '<2 y', '2–4 y', '5–9 y', '10–19 y', '20+ y',
    ])
  })

  it('leaves the countries section out when there are no official counts', async () => {
    renderCoverage(fetcherFor({ ...available, report: { ...report, country_gap: [] } }))

    await screen.findByRole('table', { name: /Eurocodes with cars/ })
    expect(screen.queryByRole('heading', { name: /Countries/ })).not.toBeInTheDocument()
  })

  it('puts codes by cars first, with no-data codes last, and describes flags in words', async () => {
    renderCoverage(fetcherFor(available))

    await screen.findByRole('table', { name: /Eurocodes with cars/ })
    expect(codeColumn()).toEqual(['Q301', 'Q201', 'Q202', 'Q401', 'Q501'])

    const sharedOnly = screen.getByRole('row', { name: /Q202/ })
    expect(within(sharedOnly).getByText('New, Only shared, One market, Before 2010, At least')).toBeVisible()
    expect(within(screen.getByRole('row', { name: /Q401/ })).getByText('Reviewed, no cars')).toBeVisible()
    expect(within(screen.getByRole('row', { name: /Q501/ })).getByText('Not reviewed')).toBeVisible()
    // Years are open-ended with a dash; the unlabelled code falls back to the catalogue text.
    expect(within(sharedOnly).getByText('2025–')).toBeVisible()
    expect(within(screen.getByRole('row', { name: /Q501/ })).getByText('VELORA WORKS · V3 CURRENT WAGON')).toBeVisible()
  })

  it('carries the range and observed markets with the figures and links code rows to the ranking', async () => {
    renderCoverage(fetcherFor(available))

    const row = (await screen.findByRole('row', { name: /Q301/ }))
    expect(within(row).getByTitle('Range 1,100–1,800 (P10–P90)')).toHaveTextContent('1,500')
    expect(within(row).getByTitle('DE, FR')).toHaveTextContent('2')
    expect(within(row).getByRole('link', { name: 'Q301' })).toHaveAttribute(
      'href', '/opportunities?groupBy=eurocode&q=Q301',
    )
    expect(within(screen.getByRole('row', { name: /Q401/ })).queryByRole('link')).not.toBeInTheDocument()
  })

  it('searches code and vehicle text without regard to case', async () => {
    const user = userEvent.setup()
    renderCoverage(fetcherFor(available))

    await screen.findByRole('table', { name: /Eurocodes with cars/ })
    await user.type(screen.getByRole('searchbox', { name: 'Search code or vehicle' }), 'q2')
    expect(codeColumn()).toEqual(['Q201', 'Q202'])
    await user.clear(screen.getByRole('searchbox', { name: 'Search code or vehicle' }))
    await user.type(screen.getByRole('searchbox', { name: 'Search code or vehicle' }), 'MERIDIAN')
    expect(codeColumn()).toEqual(['Q301'])
    await user.type(screen.getByRole('searchbox', { name: 'Search code or vehicle' }), ' nothing')
    expect(screen.getByText('No Eurocode matches this search.')).toBeVisible()
  })

  it('filters by tier, including tier D', async () => {
    const user = userEvent.setup()
    renderCoverage(fetcherFor(available))

    await screen.findByRole('table', { name: /Eurocodes with cars/ })
    await user.selectOptions(screen.getByRole('combobox', { name: 'Tier' }), 'D')
    expect(codeColumn()).toEqual(['Q401', 'Q501'])
    await user.selectOptions(screen.getByRole('combobox', { name: 'Tier' }), 'A')
    expect(codeColumn()).toEqual(['Q201'])
    await user.selectOptions(screen.getByRole('combobox', { name: 'Tier' }), '')
    expect(codeColumn()).toHaveLength(5)
  })

  it('sorts from keyboard-usable header buttons and reverses on a second press', async () => {
    const user = userEvent.setup()
    renderCoverage(fetcherFor(available))

    await screen.findByRole('table', { name: /Eurocodes with cars/ })
    const cars = screen.getByRole('columnheader', { name: /Cars certain/ })
    expect(cars).toHaveAttribute('aria-sort', 'descending')

    await user.click(within(screen.getByRole('table', { name: /Eurocodes with cars/ })).getByRole('button', { name: 'Code' }))
    expect(codeColumn()).toEqual(['Q201', 'Q202', 'Q301', 'Q401', 'Q501'])
    expect(screen.getByRole('columnheader', { name: /Code/ })).toHaveAttribute('aria-sort', 'ascending')
    await user.click(within(screen.getByRole('table', { name: /Eurocodes with cars/ })).getByRole('button', { name: 'Code' }))
    expect(codeColumn()).toEqual(['Q501', 'Q401', 'Q301', 'Q202', 'Q201'])

    // A column without a value for a row puts that row last, whichever the direction.
    await user.click(within(screen.getByRole('table', { name: /Eurocodes with cars/ })).getByRole('button', { name: 'Observed %' }))
    expect(codeColumn().slice(-2)).toEqual(['Q401', 'Q501'])
  })

  it('pages the codes 50 at a time', async () => {
    const user = userEvent.setup()
    renderCoverage(fetcherFor({ ...available, report: bigReport(60) }))

    await screen.findByRole('table', { name: /Eurocodes with cars/ })
    expect(codeColumn()).toHaveLength(50)
    expect(screen.getByText('Page 1 of 2')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Next' }))
    expect(screen.getByText('Page 2 of 2')).toBeVisible()
    expect(codeColumn()).toHaveLength(15)
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled()
  })

  it('reports each change of search so the URL can follow', async () => {
    const user = userEvent.setup()
    const seen: unknown[] = []
    renderCoverage(fetcherFor(available), {
      search: { by: 'cars', dir: 'desc', page: 1 },
      onSearchChange: (next) => seen.push(next),
    })

    await screen.findByRole('table', { name: /Eurocodes with cars/ })
    await user.selectOptions(screen.getByRole('combobox', { name: 'Tier' }), 'B')
    expect(seen.at(-1)).toEqual({ by: 'cars', dir: 'desc', page: 1, tier: 'B' })
  })
})

describe('data coverage navigation and search', () => {
  it('links the page from the main navigation, but not in the verified client release', () => {
    const { unmount } = render(<AppShell clientRelease={false}>content</AppShell>)
    for (const link of screen.getAllByRole('link', { name: 'Data coverage' })) {
      expect(link).toHaveAttribute('href', '/data-coverage')
    }
    unmount()
    render(<AppShell clientRelease>content</AppShell>)
    expect(screen.queryByRole('link', { name: 'Data coverage' })).not.toBeInTheDocument()
  })

  it('reads a URL and falls back to the defaults for anything unknown', () => {
    expect(parseDataCoverageSearch({})).toEqual({ by: 'cars', dir: 'desc', page: 1 })
    expect(parseDataCoverageSearch({ q: 'q2', tier: 'C', by: 'years', dir: 'asc', page: '3' })).toEqual({
      q: 'q2', tier: 'C', by: 'years', dir: 'asc', page: 3,
    })
    expect(parseDataCoverageSearch({ tier: 'Z', by: 'nope', dir: 'up', page: '-1', q: '  ' })).toEqual({
      by: 'cars', dir: 'desc', page: 1,
    })
  })
})
