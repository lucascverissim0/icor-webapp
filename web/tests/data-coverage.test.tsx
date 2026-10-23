// @vitest-environment happy-dom

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { AppShell } from '../src/app/AppShell'
import { parseDataCoverageSearch } from '../src/lib/data-coverage-search'
import {
  band, bigReport, country, estimate, fetcherFor, renderCoverage, report,
} from './data-coverage-fixtures'


const available = { available: true, message: null, report }

function withCountries(...countries: ReturnType<typeof country>[]) {
  return { ...available, report: { ...report, country_gap: countries } }
}

function countryRow(name: string): HTMLElement {
  const table = screen.getByRole('table', { name: /Official cars on the road against the model/ })
  const row = within(table).getAllByRole('row').slice(1).find(
    (candidate) => within(candidate).getByRole('rowheader').textContent === name,
  )
  if (!row) throw new Error(`No country row for ${name}`)
  return row
}

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

  it('lists countries by largest gap with the publisher territory, under-20 figures and bands', async () => {
    renderCoverage(fetcherFor(available))

    const table = await screen.findByRole('table', { name: /Official cars on the road against the model/ })
    const rows = within(table).getAllByRole('row').slice(1)
    // Gaps: GB 3,000, NL 400, FR 100.
    expect(rows.map((row) => within(row).getByRole('rowheader').textContent)).toEqual(['GB (UK)', 'NL', 'FR'])
    expect(within(table).getAllByRole('columnheader').map((cell) => cell.textContent)).toEqual([
      'Market', 'Official year', 'Official cars', 'Model cars', 'Model share',
      'Estimated missing (country total)',
      'Under 20 years: official', 'Under 20 years: model', 'Under 20 years: share',
      '<2 y', '2–4 y', '5–9 y', '10–19 y', '20+ y',
    ])
    expect(within(rows[0]).getByText('5,000')).toBeVisible()
    expect(within(rows[0]).getByText('40.0%')).toBeVisible()
    expect(within(rows[0]).getByText('1,000 / 900')).toBeVisible()
    expect(within(rows[0]).getByText('1,300 / 200')).toBeVisible()

    // The 2,700 official cars under 20 years of NL against 2,450 in the model.
    expect(within(rows[1]).getByText('2,700')).toBeVisible()
    expect(within(rows[1]).getByText('2,450')).toBeVisible()
    expect(within(rows[1]).getByText('90.7%')).toBeVisible()
    expect(within(rows[1]).getByText('300 / 150')).toBeVisible()
  })

  it('keeps the largest gap first even when the model overshoots a market', async () => {
    renderCoverage(fetcherFor(withCountries(
      country({ geography: 'AA', official: 100, modelled: 500, estimated: estimate({ above_official: 400 }) }),
      country({ geography: 'BB', official: 900, modelled: 800 }),
      country({ geography: 'CC', official: 900, modelled: 800 }),
      country({ geography: 'DD', official: 2000, modelled: 100 }),
    )))

    const table = await screen.findByRole('table', { name: /Official cars on the road against the model/ })
    expect(within(table).getAllByRole('row').slice(1).map(
      (row) => within(row).getByRole('rowheader').textContent,
    )).toEqual(['DD', 'BB', 'CC', 'AA'])
  })

  it('shows the estimated missing cars of each country total and splits them on request', async () => {
    const user = userEvent.setup()
    renderCoverage(fetcherFor(available))

    await screen.findByRole('table', { name: /Official cars on the road against the model/ })
    const row = countryRow('NL')
    expect(within(row).getByText('400')).toBeVisible()
    const toggle = within(row).getByText('Split')
    const details = toggle.closest('details')
    expect(details).not.toHaveAttribute('open')

    await user.click(toggle)
    expect(details).toHaveAttribute('open')
    const items = within(row).getAllByRole('listitem').map((item) => item.textContent)
    expect(items).toEqual([
      'Registered before model-level sources150',
      'Registrations without a model name50',
      'Used imports and other200',
    ])
  })

  it('names the model above the official count instead of a split', async () => {
    renderCoverage(fetcherFor(withCountries(
      country({
        geography: 'NL', official: 500, modelled: 750, modelled_share: 1.5,
        estimated: estimate({ above_official: 250 }),
      }),
    )))

    await screen.findByRole('table', { name: /Official cars on the road against the model/ })
    const row = countryRow('NL')
    expect(within(row).getByText('Model above official: 250')).toBeVisible()
    expect(within(row).queryByText('Split')).not.toBeInTheDocument()
    expect(within(row).queryByText('Registered before model-level sources')).not.toBeInTheDocument()
  })

  it('says the estimates are country totals only, never assigned to a model or Eurocode', async () => {
    renderCoverage(fetcherFor(available))

    const note = await screen.findByText(/Estimated missing cars are country totals only/)
    expect(note).toBeVisible()
    expect(note).toHaveTextContent('never assigned to a model or a Eurocode')
  })

  it('shows a total-only market without age figures and with its Eurostat flag', async () => {
    renderCoverage(fetcherFor(withCountries(country({ geography: 'FR', official_status: 'i' }))))

    await screen.findByRole('table', { name: /Official cars on the road against the model/ })
    const row = countryRow('FR')
    expect(within(row).getByText('1,000')).toBeVisible()
    expect(within(row).getByText('i')).toBeVisible()
    const empty = within(row).getByText(/No age split published/)
    expect(empty).toHaveAttribute('colspan', '8')
    expect(within(row).queryByText('2,700')).not.toBeInTheDocument()
    // Without age bands there is nothing to open beyond the estimate's own split.
    expect(screen.queryByText(/age bands and the estimate split/)).not.toBeInTheDocument()
    expect(screen.queryByRole('list', { name: 'Notes on the countries' })).not.toBeInTheDocument()
    expect(screen.getByText(/Eurostat flags on a figure/)).toHaveTextContent('i imputed · p provisional · b break in series')
  })

  it('marks an open-ended market that has no 20+ band and counts its 10+ cars together', async () => {
    const user = userEvent.setup()
    renderCoverage(fetcherFor(available))

    await screen.findByRole('table', { name: /Official cars on the road against the model/ })
    const row = countryRow('GB (UK)')
    // The under-20 columns and the 20+ band both carry the reason.
    expect(within(row).getAllByTitle('Office has no 20+ band')).toHaveLength(2)
    expect(within(row).getByText('10+ y')).toBeVisible()
    expect(within(row).queryByText('90.7%')).not.toBeInTheDocument()
    const notes = screen.getByRole('list', { name: 'Notes on the countries' })
    expect(within(notes).getByText(/no 20\+ band: the 10–19 y figure counts every car 10 years and older/)).toBeVisible()

    await user.click(screen.getByText('GB (UK): age bands and the estimate split'))
    const bands = screen.getByRole('table', { name: /Age bands for GB \(UK\)/ })
    expect(within(bands).getAllByRole('rowheader').map((cell) => cell.textContent)).toEqual([
      '<2 y', '2–4 y', '5–9 y', '10+ y',
    ])
  })

  it('shows the model under the other reading when the convention is unclear', async () => {
    renderCoverage(fetcherFor(available))

    await screen.findByRole('table', { name: /Official cars on the road against the model/ })
    const row = countryRow('GB (UK)')
    expect(within(row).getByText('(other convention: 1,100)')).toBeVisible()
    expect(within(row).getByText('(other convention: 400)')).toBeVisible()
    // No alternative where the convention is known.
    expect(within(countryRow('NL')).queryByText(/other convention/)).not.toBeInTheDocument()
    const notes = screen.getByRole('list', { name: 'Notes on the countries' })
    expect(within(notes).getByText(/convention unclear: the model is shown under both readings/)).toBeVisible()
  })

  it('footnotes the band year, the convention and used imports', async () => {
    renderCoverage(fetcherFor(withCountries(
      country({
        geography: 'AA', official: 1000, band_year: 2024, band_official_total: 950, convention: '2y',
        bands: [band('Y_LT2', 950, 900)], registrations_include_used_imports: [2023, 2024],
      }),
      country({
        geography: 'BB', band_year: 2026, band_official_total: 1000, convention: '3y',
        bands: [band('Y_LT2', 1000, 900)],
      }),
      country({
        geography: 'CC', band_year: 2026, band_official_total: 1000, convention: '2y-4y',
        bands: [band('Y_LT2', 1000, 900)],
      }),
    )))

    const notes = await screen.findByRole('list', { name: 'Notes on the countries' })
    const items = within(notes).getAllByRole('listitem').map((item) => item.textContent)
    expect(items).toEqual([
      'AA: age bands from 2024 (office total that year 950)',
      'AA: 2 registration years count as "less than 2 years"',
      'AA: official new-car counts for registration years 2023, 2024 include used imports',
      'BB: 3 registration years count as "less than 2 years", so every band edge moves up a year',
      'CC: 2 registration years count as "less than 2 years" and 4 as "2–5 years", so "5–10 years" holds 4 (ages 6–9)',
    ])
  })

  it('explains the Eurostat flags once, and only when a figure carries one', async () => {
    const { unmount } = renderCoverage(fetcherFor(available))

    await screen.findByRole('table', { name: /Official cars on the road against the model/ })
    expect(within(countryRow('GB (UK)')).getByText('p')).toBeVisible()
    expect(within(countryRow('FR')).getByText('i')).toBeVisible()
    expect(screen.getByText(/Eurostat flags on a figure/)).toBeVisible()
    unmount()

    renderCoverage(fetcherFor(withCountries(country({ geography: 'FR' }))))
    await screen.findByRole('table', { name: /Official cars on the road against the model/ })
    expect(screen.queryByText(/Eurostat flags on a figure/)).not.toBeInTheDocument()
  })

  it('opens each band with its official and model cars and the split of its gap', async () => {
    const user = userEvent.setup()
    renderCoverage(fetcherFor(available))

    await screen.findByRole('table', { name: /Official cars on the road against the model/ })
    await user.click(screen.getByText('NL: age bands and the estimate split'))
    const bands = screen.getByRole('table', { name: /Age bands for NL/ })
    expect(within(bands).getAllByRole('columnheader').map((cell) => cell.textContent)).toEqual([
      'Age band', 'Official', 'Model', 'Estimated missing',
      'Registered before model-level sources', 'Registrations without a model name',
      'Used imports and other', 'Model above official',
    ])
    const cells = (name: string) => within(within(bands).getByRole('row', { name: new RegExp(`^${name}`) }))
      .getAllByRole('cell').map((cell) => cell.textContent)
    expect(cells('<2 y')).toEqual(['300', '350', '0', '0', '0', '0', '50'])
    expect(cells('10–19 y')).toEqual(['900', '700', '200', '150', '50', '0', '—'])
    expect(cells('20\\+ y')).toEqual(['300', '150', '150', '0', '0', '150', '—'])
  })

  it('flags a band figure and keeps the other-convention model beside it', async () => {
    const user = userEvent.setup()
    renderCoverage(fetcherFor(available))

    await screen.findByRole('table', { name: /Official cars on the road against the model/ })
    await user.click(screen.getByText('GB (UK): age bands and the estimate split'))
    const bands = screen.getByRole('table', { name: /Age bands for GB \(UK\)/ })
    const row = within(bands).getByRole('row', { name: /^<2 y/ })
    expect(within(row).getByText('b')).toBeVisible()
    expect(within(row).getByText('(other convention: 1,100)')).toBeVisible()
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
    expect(within(row).getByTitle('Survival-curve assumption range 1,100–1,800')).toHaveTextContent('1,500')
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
