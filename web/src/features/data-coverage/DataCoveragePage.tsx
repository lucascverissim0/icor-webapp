import { useQuery } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { ArrowDown, ArrowUp, ChevronsUpDown, Search } from 'lucide-react'
import { useMemo, useState } from 'react'

import { queryKeys } from '../../app/query-client'
import { dataCoverageRoute } from '../../app/router'
import { ApiProblem, PlannerApiClient, plannerApi } from '../../lib/api/client'
import {
  TIERS,
  type DataCoverageSearch,
  type SortKey,
} from '../../lib/data-coverage-search'
import { driveText, format } from '../eurocodes/eurocode-text'
import {
  BAND_LABELS,
  ESTIMATE_PARTS,
  NO_AGE_SPLIT,
  NO_OVER_20_BAND,
  PAGE_SIZE,
  STATUS_LEGEND,
  bandLabel,
  buildRows,
  countryNotes,
  filterRows,
  hasStatus,
  sortCountries,
  sortRows,
  statusText,
  type CountryGap,
  type CoverageRow,
  type DataReport,
} from './coverage-rows'


interface DataCoverageWorkbenchProps {
  apiClient?: PlannerApiClient
  clientRelease?: boolean
  onSearchChange?: (search: DataCoverageSearch) => void
  search?: DataCoverageSearch
}

const DEFAULT_SEARCH: DataCoverageSearch = { by: 'cars', dir: 'desc', page: 1 }

function percent(share: number): string {
  return `${(share * 100).toFixed(1)}%`
}

function wholePercent(share: number): string {
  return `${Math.round(share * 100)}%`
}

function yearsText(first: number | null, last: number | null): string {
  return first === null ? '—' : `${first}–${last ?? ''}`
}

function TierBadge({ tier }: { tier: string }) {
  return <span className={`coverage-tier coverage-tier--${tier.toLowerCase()}`}>Tier {tier}</span>
}

function StateMessage({ eyebrow, title, children }: {
  eyebrow: string
  title: string
  children?: React.ReactNode
}) {
  return (
    <section className="route-state" role="status">
      <p className="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      {children}
    </section>
  )
}

function Kpis({ report }: { report: DataReport }) {
  const { totals } = report
  const observed = totals.by_evidence.stock + totals.by_evidence.registrations
  return (
    <section aria-label="Key figures" className="coverage-kpis">
      <div className="coverage-kpi">
        <span>Cars in the model</span>
        <strong>{format(totals.fleet)}</strong>
      </div>
      <div className="coverage-kpi">
        <span>Observed</span>
        <strong>{totals.fleet > 0 ? percent(observed / totals.fleet) : '—'}</strong>
        <small>registered or counted, not estimated</small>
      </div>
      {(['A', 'B', 'C'] as const).map((tier) => (
        <div className="coverage-kpi" key={tier}>
          <span><TierBadge tier={tier} /></span>
          <strong>{format(totals.tiers[tier]?.codes ?? 0)} codes</strong>
          <small>{format(totals.tiers[tier]?.certain_p50 ?? 0)} certain cars</small>
        </div>
      ))}
      <div className="coverage-kpi">
        <span><TierBadge tier="D" /></span>
        <strong>{format(totals.tiers.D?.codes ?? 0)} codes</strong>
        <small>
          {format(totals.no_data.not_reviewed)} not reviewed · {format(totals.no_data.reviewed)} reviewed
        </small>
      </div>
    </section>
  )
}

function TierLegend({ report }: { report: DataReport }) {
  const { thresholds } = report
  return (
    <p className="coverage-legend">
      <strong>A</strong> ≥{wholePercent(thresholds.tier_a_observed_share)} of cars observed in ≥{thresholds.tier_a_markets} markets ·{' '}
      <strong>B</strong> ≥{wholePercent(thresholds.tier_b_observed_share)} · <strong>C</strong> less · production before 2010 caps a code at B,
      or at C when {wholePercent(thresholds.pre_2010_share_cap_c)} of its life is before 2010 · <strong>D</strong> no cars in the model
    </p>
  )
}

function marketName(gap: CountryGap): string {
  return gap.publisher_geography === gap.geography
    ? gap.geography
    : `${gap.geography} (${gap.publisher_geography})`
}

/** Eurostat's flag on a figure, small and named for screen readers. */
function StatusFlag({ status }: { status: string | null }) {
  if (status === null) return null
  return (
    <small className="coverage-flag" title={statusText(status)}>
      {status}
      <span className="sr-only"> ({statusText(status).toLowerCase()})</span>
    </small>
  )
}

function BandCell({ gap, band }: { gap: CountryGap; band: string }) {
  const item = gap.bands.find((candidate) => candidate.band === band)
  if (!item) {
    const reason = gap.open_ended && band === 'Y_GT20' ? NO_OVER_20_BAND : undefined
    return <td className="coverage-number" title={reason}>—</td>
  }
  return (
    <td className="coverage-number">
      <span>{format(item.official)} / {format(item.modelled)}</span>
      <StatusFlag status={item.status} />
      {item.modelled_alternative !== null && (
        <small className="coverage-aside">(other convention: {format(item.modelled_alternative)})</small>
      )}
      {gap.open_ended && band === 'Y10-20' && <small className="coverage-aside">{bandLabel(band, true)}</small>}
    </td>
  )
}

function EstimateCell({ gap }: { gap: CountryGap }) {
  const { estimated } = gap
  return (
    <td className="coverage-number coverage-estimate">
      <span>{format(estimated.missing)}</span>
      {estimated.above_official > 0
        ? <small className="coverage-aside">Model above official: {format(estimated.above_official)}</small>
        : (
          <details className="coverage-split">
            <summary>Split<span className="sr-only"> of the estimate for {marketName(gap)}</span></summary>
            <ul>
              {ESTIMATE_PARTS.map(([key, label]) => (
                <li key={key}><span>{label}</span><span>{format(estimated[key])}</span></li>
              ))}
            </ul>
          </details>
        )}
    </td>
  )
}

function Under20Cells({ gap }: { gap: CountryGap }) {
  const { under_20: under20 } = gap
  if (under20 === null) {
    const reason = gap.open_ended ? NO_OVER_20_BAND : NO_AGE_SPLIT
    return (
      <td className="coverage-number" colSpan={3} title={reason}>
        —<span className="sr-only"> {reason}</span>
      </td>
    )
  }
  return (
    <>
      <td className="coverage-number">{format(under20.official)}</td>
      <td className="coverage-number">{format(under20.modelled)}</td>
      <td className="coverage-number">{under20.modelled_share === null ? '—' : percent(under20.modelled_share)}</td>
    </>
  )
}

function CountryRow({ gap }: { gap: CountryGap }) {
  return (
    <tr>
      <th scope="row">{marketName(gap)}</th>
      <td>{gap.year}</td>
      <td className="coverage-number">
        {format(gap.official)}
        <StatusFlag status={gap.official_status} />
      </td>
      <td className="coverage-number">{format(gap.modelled)}</td>
      <td className="coverage-number">{gap.modelled_share === null ? '—' : percent(gap.modelled_share)}</td>
      <EstimateCell gap={gap} />
      {gap.bands.length === 0
        ? <td className="coverage-muted" colSpan={3 + BAND_LABELS.length}>— {NO_AGE_SPLIT}</td>
        : (
          <>
            <Under20Cells gap={gap} />
            {BAND_LABELS.map(([band]) => <BandCell band={band} gap={gap} key={band} />)}
          </>
        )}
    </tr>
  )
}

/** One country's age bands with the split of each band's estimated gap. */
function BandDetail({ gap }: { gap: CountryGap }) {
  const market = marketName(gap)
  return (
    <details className="coverage-detail">
      <summary>{market}: age bands and the estimate split</summary>
      <div aria-label={`Age bands for ${market}`} className="coverage-scroll" role="region" tabIndex={0}>
        <table className="coverage-table coverage-table--bands">
          <caption className="sr-only">Age bands for {market}: official cars, model cars and the estimated split of each gap</caption>
          <thead>
            <tr>
              <th scope="col">Age band</th>
              <th scope="col">Official</th>
              <th scope="col">Model</th>
              <th scope="col">Estimated missing</th>
              {ESTIMATE_PARTS.map(([key, label]) => <th key={key} scope="col">{label}</th>)}
              <th scope="col">Model above official</th>
            </tr>
          </thead>
          <tbody>
            {gap.bands.map((item) => (
              <tr key={item.band}>
                <th scope="row">{bandLabel(item.band, gap.open_ended)}</th>
                <td className="coverage-number">
                  {format(item.official)}
                  <StatusFlag status={item.status} />
                </td>
                <td className="coverage-number">
                  {format(item.modelled)}
                  {item.modelled_alternative !== null && (
                    <small className="coverage-aside">(other convention: {format(item.modelled_alternative)})</small>
                  )}
                </td>
                <td className="coverage-number">{format(item.estimated.missing)}</td>
                {ESTIMATE_PARTS.map(([key]) => (
                  <td className="coverage-number" key={key}>{format(item.estimated[key])}</td>
                ))}
                <td className="coverage-number">
                  {item.estimated.above_official > 0 ? format(item.estimated.above_official) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  )
}

function CountryNotes({ countries }: { countries: CountryGap[] }) {
  const notes = countries.flatMap((gap) => countryNotes(gap).map((note) => ({ gap, note })))
  const flags = hasStatus(countries)
  if (notes.length === 0 && !flags) return null
  return (
    <div className="coverage-footnotes">
      {notes.length > 0 && (
        <ul aria-label="Notes on the countries">
          {notes.map(({ gap, note }) => (
            <li key={`${gap.geography}-${note}`}><strong>{marketName(gap)}</strong>: {note}</li>
          ))}
        </ul>
      )}
      {flags && (
        <p className="coverage-note">
          Eurostat flags on a figure:{' '}
          {STATUS_LEGEND.map(([flag, text], index) => (
            <span key={flag}>{index > 0 && ' · '}<strong>{flag}</strong> {text.toLowerCase()}</span>
          ))}
        </p>
      )}
    </div>
  )
}

function CountriesTable({ countries }: { countries: CountryGap[] }) {
  if (countries.length === 0) return null
  const sorted = sortCountries(countries)
  const withBands = sorted.filter((gap) => gap.bands.length > 0)
  return (
    <section aria-labelledby="coverage-countries-title" className="coverage-panel">
      <h3 id="coverage-countries-title">Countries: official cars against the model</h3>
      <p className="coverage-note">Largest gap first. Age bands read official / model.</p>
      <p className="coverage-note">
        Estimated missing cars are country totals only. They are never assigned to a model or a Eurocode.
      </p>
      <div aria-label="Countries table" className="coverage-scroll" role="region" tabIndex={0}>
        <table className="coverage-table coverage-table--countries">
          <caption className="sr-only">Official cars on the road against the model, by market and age band</caption>
          <thead>
            <tr>
              <th scope="col">Market</th>
              <th scope="col">Official year</th>
              <th scope="col">Official cars</th>
              <th scope="col">Model cars</th>
              <th scope="col">Model share</th>
              <th scope="col">Estimated missing (country total)</th>
              <th scope="col">Under 20 years: official</th>
              <th scope="col">Under 20 years: model</th>
              <th scope="col">Under 20 years: share</th>
              {BAND_LABELS.map(([band, label]) => <th key={band} scope="col">{label}</th>)}
            </tr>
          </thead>
          <tbody>
            {sorted.map((gap) => <CountryRow gap={gap} key={gap.geography} />)}
          </tbody>
        </table>
      </div>
      <CountryNotes countries={sorted} />
      {withBands.length > 0 && (
        <div className="coverage-details">
          {withBands.map((gap) => <BandDetail gap={gap} key={gap.geography} />)}
        </div>
      )}
    </section>
  )
}

const COLUMNS: readonly { key: SortKey; label: string; first: 'asc' | 'desc' }[] = [
  { key: 'code', label: 'Code', first: 'asc' },
  { key: 'label', label: 'Vehicles', first: 'asc' },
  { key: 'tier', label: 'Tier', first: 'asc' },
  { key: 'years', label: 'Years', first: 'asc' },
  { key: 'cars', label: 'Cars certain', first: 'desc' },
  { key: 'shared', label: 'Shared', first: 'desc' },
  { key: 'observed', label: 'Observed %', first: 'desc' },
  { key: 'markets', label: 'Observed markets', first: 'desc' },
  { key: 'sources', label: 'Sources', first: 'desc' },
]

function SortHeader({ column, search, onSort }: {
  column: (typeof COLUMNS)[number]
  search: DataCoverageSearch
  onSort: (key: SortKey) => void
}) {
  const active = search.by === column.key
  const Icon = !active ? ChevronsUpDown : search.dir === 'asc' ? ArrowUp : ArrowDown
  return (
    <th aria-sort={active ? (search.dir === 'asc' ? 'ascending' : 'descending') : 'none'} scope="col">
      <button className="coverage-sort" onClick={() => onSort(column.key)} type="button">
        {column.label}
        <Icon aria-hidden="true" size={14} />
      </button>
    </th>
  )
}

function CodeRow({ row }: { row: CoverageRow }) {
  const range = row.carsP10 !== null && row.carsP90 !== null
    ? `Survival-curve assumption range ${format(row.carsP10)}–${format(row.carsP90)}`
    : undefined
  const markets = row.observedMarkets.join(', ')
  return (
    <tr>
      <th scope="row">
        {row.hasCars
          ? <a href={`/opportunities?groupBy=eurocode&q=${encodeURIComponent(row.code)}`}><code>{row.code}</code></a>
          : <code>{row.code}</code>}
      </th>
      <td>
        {row.label || '—'}
        {row.drive && row.drive !== 'both' && <small className="coverage-muted"> · {driveText(row.drive)}</small>}
      </td>
      <td><TierBadge tier={row.tier} /></td>
      <td>{yearsText(row.firstYear, row.lastYear)}</td>
      <td className="coverage-number">
        {row.cars === null ? '—' : <span title={range}>{format(row.cars)}</span>}
        {range && <span className="sr-only">, {range}</span>}
      </td>
      <td className="coverage-number">{row.shared === null ? '—' : format(row.shared)}</td>
      <td className="coverage-number">{row.observedShare === null ? '—' : percent(row.observedShare)}</td>
      <td className="coverage-number">
        {row.hasCars ? <span title={markets || undefined}>{row.observedMarkets.length}</span> : '—'}
        {markets && <span className="sr-only">: {markets}</span>}
      </td>
      <td className="coverage-number">{row.sources === null ? '—' : row.sources}</td>
      <td>{row.flags.length > 0 ? row.flags.join(', ') : '—'}</td>
    </tr>
  )
}

function CodesTable({ rows, search, onChange }: {
  rows: CoverageRow[]
  search: DataCoverageSearch
  onChange: (next: DataCoverageSearch) => void
}) {
  const [draft, setDraft] = useState(search.q ?? '')
  const filtered = useMemo(() => filterRows(rows, search), [rows, search])
  const sorted = useMemo(() => sortRows(filtered, search.by, search.dir), [filtered, search.by, search.dir])
  const pages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE))
  const page = Math.min(search.page, pages)
  const visible = sorted.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  function onSort(key: SortKey) {
    const column = COLUMNS.find((item) => item.key === key)
    const dir = search.by === key
      ? (search.dir === 'asc' ? 'desc' : 'asc')
      : (column?.first ?? 'desc')
    onChange({ ...search, by: key, dir, page: 1 })
  }

  return (
    <section aria-labelledby="coverage-codes-title" className="coverage-panel">
      <h3 id="coverage-codes-title">Eurocodes</h3>
      <div className="coverage-controls">
        <label className="coverage-search">
          <span>Search code or vehicle</span>
          <span className="coverage-search__field">
            <Search aria-hidden="true" size={16} />
            <input
              maxLength={64}
              onChange={(event) => {
                setDraft(event.target.value)
                onChange({ ...search, q: event.target.value || undefined, page: 1 })
              }}
              type="search"
              value={draft}
            />
          </span>
        </label>
        <label>
          <span>Tier</span>
          <select
            onChange={(event) => {
              const tier = TIERS.find((value) => value === event.target.value)
              onChange({ ...search, tier, page: 1 })
            }}
            value={search.tier ?? ''}
          >
            <option value="">All</option>
            {TIERS.map((tier) => <option key={tier} value={tier}>{tier}</option>)}
          </select>
        </label>
        <p aria-live="polite" className="coverage-note">{format(sorted.length)} of {format(rows.length)} codes</p>
      </div>
      {sorted.length === 0
        ? <p className="coverage-empty">No Eurocode matches this search.</p>
        : (
          <div aria-label="Eurocodes table" className="coverage-scroll" role="region" tabIndex={0}>
            <table className="coverage-table">
              <caption className="sr-only">Eurocodes with cars on the road and the evidence behind them</caption>
              <thead>
                <tr>
                  {COLUMNS.map((column) => (
                    <SortHeader column={column} key={column.key} onSort={onSort} search={search} />
                  ))}
                  <th scope="col">Flags</th>
                </tr>
              </thead>
              <tbody>{visible.map((row) => <CodeRow key={row.code} row={row} />)}</tbody>
            </table>
          </div>
        )}
      {pages > 1 && (
        <nav aria-label="Eurocode pages" className="pagination">
          <button disabled={page <= 1} onClick={() => onChange({ ...search, page: page - 1 })} type="button">Previous</button>
          <span>Page {page} of {pages}</span>
          <button disabled={page >= pages} onClick={() => onChange({ ...search, page: page + 1 })} type="button">Next</button>
        </nav>
      )}
    </section>
  )
}

export function DataCoverageWorkbench({
  apiClient = plannerApi,
  clientRelease = import.meta.env.VITE_ICOR_CLIENT_RELEASE === 'verified',
  onSearchChange,
  search,
}: DataCoverageWorkbenchProps) {
  const [localSearch, setLocalSearch] = useState<DataCoverageSearch>(DEFAULT_SEARCH)
  const current = search ?? localSearch
  const change = onSearchChange ?? setLocalSearch
  const coverage = useQuery({
    queryKey: queryKeys.eurocodeDataCoverage,
    queryFn: () => apiClient.eurocodeDataCoverage(),
    enabled: !clientRelease,
  })
  const report = coverage.data?.report ?? null
  const rows = useMemo(() => (report ? buildRows(report) : []), [report])

  if (clientRelease) {
    return (
      <StateMessage eyebrow="Not available" title="This view is not part of the verified release">
        <a className="primary-action" href="/opportunities">Return to opportunities</a>
      </StateMessage>
    )
  }
  if (coverage.isPending) {
    return <section aria-busy="true" className="route-state"><h2>Loading data coverage…</h2></section>
  }
  if (coverage.isError) {
    if (coverage.error instanceof ApiProblem && coverage.error.code === 'eurocodes_not_loaded') {
      return (
        <StateMessage eyebrow="Eurocodes unavailable" title="Eurocodes are not loaded in this service">
          <p>{coverage.error.message}</p>
        </StateMessage>
      )
    }
    return (
      <section className="route-state" role="alert">
        <p className="eyebrow">Data coverage</p>
        <h2>Data coverage could not be loaded</h2>
        <p>{coverage.error.message}</p>
        <button className="primary-action" onClick={() => void coverage.refetch()} type="button">Retry</button>
      </section>
    )
  }
  if (!coverage.data.available || !report) {
    return (
      <StateMessage eyebrow="Data coverage" title="No data report yet">
        <p>{coverage.data.message ?? 'No data report is available for this snapshot.'}</p>
      </StateMessage>
    )
  }

  return (
    <div className="data-coverage-page">
      <header className="coverage-header">
        <h2>Cars on the road at end of {report.year}</h2>
        <small className="coverage-muted">Snapshot {report.snapshot_id}</small>
      </header>
      <Kpis report={report} />
      <TierLegend report={report} />
      <CountriesTable countries={report.country_gap} />
      <CodesTable onChange={change} rows={rows} search={current} />
    </div>
  )
}

export function DataCoveragePage() {
  const navigate = useNavigate()
  const search = dataCoverageRoute.useSearch()
  return (
    <DataCoverageWorkbench
      onSearchChange={(next) => void navigate({ to: '/data-coverage', search: next, replace: true })}
      search={search}
    />
  )
}
