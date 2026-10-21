import { useEffect, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { BadgeCheck, ChartNoAxesCombined, TriangleAlert } from 'lucide-react'

import { opportunitiesRoute } from '../../app/router'
import { queryKeys } from '../../app/query-client'
import { ApiProblem, PlannerApiClient, plannerApi, type OpportunitiesQuery } from '../../lib/api/client'
import {
  serializeOpportunitySearch,
  type OpportunitySearch,
  type OpportunitySort,
} from '../../lib/opportunity-search'
import { EurocodeNotLoaded } from '../eurocodes/EurocodeBreakdown'
import {
  eurocodeReadingText,
  format as formatUnits,
  unassignedParts,
  type EurocodeBreakdownData,
  type EurocodeReading,
} from '../eurocodes/eurocode-text'
import { CoverageManager } from './CoverageManager'
import { OpportunityRanking } from './OpportunityRanking'


interface OpportunitiesWorkbenchProps {
  apiClient?: PlannerApiClient
  clientRelease?: boolean
  invalidKeys?: string[]
  onOpenDetails: (groupId: string) => void
  onSearchChange: (search: OpportunitySearch) => void
  search: OpportunitySearch
  selectedGroup?: string | null
}

function queryFromSearch(search: OpportunitySearch): OpportunitiesQuery {
  return {
    groupBy: search.groupBy,
    markets: search.market,
    horizons: search.horizon,
    text: search.q,
    sort: search.order,
    icorWorked: search.worked === undefined ? undefined : search.worked === 'yes',
    page: search.page,
    pageSize: 100,
  }
}

const SORT_LABELS: ReadonlyArray<readonly [OpportunitySort, string]> = [
  ['score', 'Opportunity score'],
  ['demand', 'Forecast demand'],
  ['vehicle', 'Vehicle name'],
]

interface RankingFiltersProps {
  availableHorizons: number[]
  availableMarkets: string[]
  onSearchChange: (search: OpportunitySearch) => void
  search: OpportunitySearch
}

function RankingFilters({
  availableHorizons,
  availableMarkets,
  onSearchChange,
  search,
}: RankingFiltersProps) {
  // The box is typed into locally and pushed into the URL after a pause, so a
  // request is not sent for every keystroke and the browser history does not
  // gain one entry per letter.
  const [draft, setDraft] = useState(search.q ?? '')
  // Adjusting state during render, rather than in an effect, is React's own
  // answer to "reset local state when a prop changes": it avoids the extra
  // render pass an effect would cost.
  const [lastPushed, setLastPushed] = useState(search.q ?? '')
  if ((search.q ?? '') !== lastPushed) {
    setLastPushed(search.q ?? '')
    setDraft(search.q ?? '')
  }
  useEffect(() => {
    const trimmed = draft.trim()
    if (trimmed === (search.q ?? '')) return
    const timer = globalThis.setTimeout(() => {
      onSearchChange({
        ...search,
        q: trimmed === '' ? undefined : trimmed,
        page: 1,
      })
    }, 300)
    return () => { globalThis.clearTimeout(timer) }
  }, [draft, onSearchChange, search])

  const selectedMarkets = search.market ?? []
  const toggleMarket = (market: string) => {
    const next = selectedMarkets.includes(market)
      ? selectedMarkets.filter((value) => value !== market)
      : [...selectedMarkets, market]
    onSearchChange({
      ...search,
      market: next.length > 0 ? next : undefined,
      page: 1,
    })
  }

  const active =
    selectedMarkets.length > 0 ||
    (search.horizon?.length ?? 0) > 0 ||
    Boolean(search.q) ||
    search.worked !== undefined ||
    (search.order ?? 'score') !== 'score'

  return (
    <section aria-label="Narrow the ranking" className="ranking-filters">
      <div className="ranking-filters__row">
        <label className="ranking-filters__field" htmlFor="opportunity-search">
          <span>Search vehicle</span>
          <input
            autoComplete="off"
            id="opportunity-search"
            maxLength={64}
            onChange={(event) => { setDraft(event.target.value) }}
            placeholder={search.groupBy === 'eurocode' ? 'Eurocode, brand or model, for example Golf' : 'Brand, model or Eurocode, for example Golf'}
            type="search"
            value={draft}
          />
        </label>

        <label className="ranking-filters__field" htmlFor="opportunity-horizon">
          <span>Forecast year</span>
          <select
            id="opportunity-horizon"
            onChange={(event) => {
              const value = event.target.value
              onSearchChange({
                ...search,
                horizon: value === '' ? undefined : [Number(value)],
                page: 1,
              })
            }}
            value={search.horizon?.[0] ?? ''}
          >
            <option value="">All forecast years</option>
            {availableHorizons.map((horizon) => (
              <option key={horizon} value={horizon}>{horizon}</option>
            ))}
          </select>
        </label>

        <label className="ranking-filters__field" htmlFor="opportunity-worked">
          <span>ICOR worked</span>
          <select
            id="opportunity-worked"
            onChange={(event) => {
              const value = event.target.value
              onSearchChange({
                ...search,
                worked: value === 'yes' || value === 'no' ? value : undefined,
                page: 1,
              })
            }}
            value={search.worked ?? ''}
          >
            <option value="">All vehicles</option>
            <option value="yes">Worked on</option>
            <option value="no">Not worked on</option>
          </select>
        </label>

        <label className="ranking-filters__field" htmlFor="opportunity-order">
          <span>Order by</span>
          <select
            id="opportunity-order"
            onChange={(event) => {
              const value = event.target.value as OpportunitySort
              onSearchChange({
                ...search,
                order: value === 'score' ? undefined : value,
                page: 1,
              })
            }}
            value={search.order ?? 'score'}
          >
            {SORT_LABELS.map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </label>

        {active && (
          <button
            className="ranking-filters__clear"
            onClick={() => {
              onSearchChange({ groupBy: search.groupBy, page: 1 })
            }}
            type="button"
          >
            Clear filters
          </button>
        )}
      </div>

      {availableMarkets.length > 0 && (
        <fieldset className="ranking-filters__markets">
          <legend>
            Markets
            {selectedMarkets.length === 0
              ? ' — every market in this snapshot'
              : ` — ${selectedMarkets.length.toString()} selected`}
          </legend>
          <div className="ranking-filters__market-list">
            {availableMarkets.map((market) => (
              <label key={market}>
                <input
                  checked={selectedMarkets.includes(market)}
                  onChange={() => { toggleMarket(market) }}
                  type="checkbox"
                  value={market}
                />
                {market}
              </label>
            ))}
          </div>
        </fieldset>
      )}
    </section>
  )
}

// What a score is compared against, named so the disclosure can say it.
const RANKED_NOUN: Record<OpportunitySearch['groupBy'], string> = {
  brand: 'brands',
  model: 'models',
  model_year: 'model years',
  eurocode: 'Eurocodes',
}

function formatCount(value: string | number): string {
  return Number(value).toLocaleString('en-US')
}

function percent(part: number, whole: number): string {
  return `${(part / whole * 100).toFixed(1)}%`
}

/** How a search that is exactly a Eurocode was read: never silently one way. */
function EurocodeSearchNotice({ reading, text }: { reading: EurocodeReading; text: string }) {
  return (
    <p className="eurocode-notice eurocode-search-notice" role="status">
      {reading.label == null
        ? `“${text}” is ${eurocodeReadingText(reading)} in ICOR's catalogue, but no reviewed vehicle carries it. Only vehicles whose name contains “${text}” are listed.`
        : `“${text}” is read two ways: as ${eurocodeReadingText(reading)}, and as text in vehicle names. Both are listed; rows that may carry ${reading.code} show it first, marked “Searched code”.`}
    </p>
  )
}

function ProblemState({ error, onRetry }: { error: Error; onRetry: () => void }) {
  const correlation = error instanceof ApiProblem ? error.correlationId : null
  return (
    <section className="opportunity-state" role="alert">
      <p className="eyebrow">Opportunities unavailable</p>
      <h2>We could not load this ranking</h2>
      <p>{error.message}</p>
      {correlation && <p className="correlation">Reference: {correlation}</p>}
      <button className="primary-action" onClick={onRetry} type="button">Retry</button>
    </section>
  )
}

/** The Eurocode view's units by outcome: certain on the rows, shared once, and not assigned. */
function EurocodeTotals({ totals }: { totals: EurocodeBreakdownData }) {
  const { total, reasons } = unassignedParts(totals)
  return (
    <p aria-label="Eurocode totals" className="eurocode-totals" role="group">
      <span>Certain on the codes listed: <strong>{formatUnits(totals.single_units)}</strong></span>
      <span>Shared between codes, counted once: <strong>{formatUnits(totals.shared_units)}</strong></span>
      <span>
        Not assigned to a Eurocode: <strong>{formatUnits(total)}</strong>
        {reasons.length > 0 && ` (${reasons.map((part) => `${part.reason}: ${formatUnits(part.units)}`).join('; ')})`}
      </span>
    </p>
  )
}

/** Shown instead of the generic error when the service has no Eurocode map. */
function EurocodesNotLoadedState({ message, onShowModelYears }: { message: string; onShowModelYears: () => void }) {
  return (
    <section className="opportunity-state" role="status">
      <p className="eyebrow">Eurocodes unavailable</p>
      <h2>Eurocodes are not loaded in this service</h2>
      <p>{message}</p>
      <button className="primary-action" onClick={onShowModelYears} type="button">Show model years</button>
    </section>
  )
}

export function OpportunitiesWorkbench({
  apiClient = plannerApi,
  clientRelease = import.meta.env.VITE_ICOR_CLIENT_RELEASE === 'verified',
  invalidKeys = [],
  onOpenDetails,
  onSearchChange,
  search,
  selectedGroup = null,
}: OpportunitiesWorkbenchProps) {
  const opportunityQuery = queryFromSearch(search)
  // What one ranked row is, in the score explanation's words.
  const rankedItem = search.groupBy === 'eurocode' ? 'Eurocode' : 'vehicle'
  const detailQuery: OpportunitiesQuery = {
    groupBy: search.groupBy,
    markets: search.market,
    horizons: search.horizon,
  }
  const queryClient = useQueryClient()
  const ranking = useQuery({
    queryKey: queryKeys.opportunities(opportunityQuery),
    queryFn: ({ signal }) => apiClient.opportunities(opportunityQuery, signal),
  })
  const registrationSummary = useQuery({
    queryKey: ['registrations', 'summary'],
    queryFn: () => apiClient.registrationSummary(),
  })

  // The status is a quiet extra: if it cannot be read, the page shows nothing for it.
  const eurocodeStatus = useQuery({
    queryKey: queryKeys.eurocodeStatus,
    queryFn: () => apiClient.eurocodeStatus(),
    retry: false,
  })
  const eurocodeCoverage = eurocodeStatus.data?.loaded
    ? [...eurocodeStatus.data.coverage].sort((left, right) => right.horizon - left.horizon)[0]
    : undefined

  return (
    <div className="opportunities-page">
      <header className="opportunities-hero">
        <div>
          <p className="eyebrow">Windshield replacement forecast</p>
          <h2>{clientRelease ? 'Prioritized vehicle-year opportunities' : 'Prioritized model and generation opportunities'}</h2>
          <p>{clientRelease
            ? 'This client release covers every forecastable official-source make/model label and registration cohort year. Ranked generation names use reviewed manufacturer production windows; forecasts remain planning estimates.'
            : 'Start with the vehicle opportunities forecast for upcoming years. The score is forecast demand alone; ICOR experience is shown beside it.'}</p>
        </div>
        <span className="status-pill">{clientRelease ? 'Official vehicle-year evidence' : 'Validated snapshot'}</span>
      </header>

      {registrationSummary.data && (() => {
        const latest = [...registrationSummary.data.availability]
          .filter(({ geography }) => geography === 'EU27')
          .sort((left, right) => right.year - left.year)[0]
        return latest ? (
          <aside className="forecast-freshness" aria-label="Forecast evidence freshness">
            <strong>Registration evidence through {latest.year} ({latest.status})</strong>
            <span>Official model-level registrations are released annually, not as a live daily feed. Forecast horizons continue beyond the latest observed release.</span>
          </aside>
        ) : null
      })()}

      <section className="score-method" aria-labelledby="score-method-title">
        <div><p className="eyebrow">Transparent ranking</p><h2 id="score-method-title">How the opportunity score is calculated</h2></div>
        <div className="score-method__formula">
          <p><strong>Score = demand percentile × 100</strong><span>The highest forecast replacement demand scores 100. Whether ICOR has worked on a vehicle is shown beside the score and never changes it.</span></p>
        </div>
        <p className="score-method__scope" role="note">
          Scores are whole-market. The demand percentile compares each {rankedItem}{' '}
          with{' '}
          {ranking.data?.demand_population
            ? `all ${formatCount(ranking.data.demand_population)} ranked ${RANKED_NOUN[search.groupBy]}`
            : 'every ranked vehicle'}{' '}
          in this snapshot — every market and both forecast horizons — so a{' '}
          {rankedItem} scores the same here as it does anywhere else in the app.
          Filtering or searching changes which rows appear and the units they
          carry, never the score.
          {search.groupBy === 'eurocode' && ' Eurocode rows rank on the units the code certainly carries; units shared between codes are shown beside them and never split.'}
        </p>
      </section>

      {invalidKeys.length > 0 && (
        <p className="url-notice" role="status">Adjusted URL filters: {invalidKeys.join(', ')}</p>
      )}

      {!clientRelease && <div>
        <p className="grouping-label">Summarize ranking by</p>
        <div aria-label="Opportunity grouping" className="grouping-control" role="group">
        {([
          ['eurocode', 'Eurocodes'],
          ['model_year', 'Model years'],
          ['model', 'Models'],
          ['brand', 'Brands'],
        ] as const).map(([value, label]) => (
          <button
            aria-pressed={search.groupBy === value}
            key={value}
            onClick={() => {
              onSearchChange({ ...search, groupBy: value, page: 1 })
            }}
            type="button"
          >
            {label}
          </button>
        ))}
        </div>
      </div>}

      <RankingFilters
        availableHorizons={[...(ranking.data?.available_horizons ?? [])]}
        availableMarkets={[...(ranking.data?.available_markets ?? [])]}
        onSearchChange={onSearchChange}
        search={search}
      />

      {ranking.isPending && <section aria-busy="true" className="opportunity-state"><h2>Loading opportunity ranking…</h2></section>}
      {ranking.isError && (ranking.error instanceof ApiProblem && ranking.error.code === 'eurocodes_not_loaded' && search.groupBy === 'eurocode'
        ? <EurocodesNotLoadedState
            message={ranking.error.message}
            onShowModelYears={() => { onSearchChange({ ...search, groupBy: 'model_year', page: 1 }) }}
          />
        : <ProblemState error={ranking.error} onRetry={() => void ranking.refetch()} />)}
      {ranking.data && (
        <>
          <aside className="dataset-coverage" aria-label="Available opportunity data">
            <div><strong>{formatCount(ranking.data.total)}</strong>{' '}<span>ranked records available</span></div>
            {registrationSummary.data && <div><strong>{formatCount(registrationSummary.data.model_count)}</strong>{' '}<span>official model labels</span></div>}
            {registrationSummary.data && <div><strong>{formatCount(registrationSummary.data.total_registrations)}</strong>{' '}<span>registrations represented</span></div>}
            {eurocodeCoverage && eurocodeCoverage.total_units > 0 && (
              <div>
                <strong>Eurocodes:</strong>{' '}
                <span>
                  {percent(eurocodeCoverage.single_units + eurocodeCoverage.shared_units, eurocodeCoverage.total_units)} of forecast demand assigned
                  ({percent(eurocodeCoverage.single_units, eurocodeCoverage.total_units)} to a single code) · {eurocodeCoverage.horizon}
                </span>
              </div>
            )}
          </aside>
          {!ranking.data.eurocodes_loaded && <EurocodeNotLoaded warnings={eurocodeStatus.data?.warnings} />}
          {ranking.data.eurocodes_loaded && eurocodeStatus.data && eurocodeStatus.data.warnings.length > 0 && (
            <div className="eurocode-notice" role="note">
              {eurocodeStatus.data.warnings.map((warning) => <p key={warning}>{warning}</p>)}
            </div>
          )}
          <dl aria-label="Opportunity summary" className="opportunity-summary">
            <div><span aria-hidden="true" className="opportunity-summary__icon"><ChartNoAxesCombined size={21} /></span><dt>{search.groupBy === 'eurocode' ? 'Certain replacements' : 'Forecast replacements'}</dt><dd>{ranking.data.summary.base_units.toLocaleString('en-US')}</dd></div>
            <div><span aria-hidden="true" className="opportunity-summary__icon"><BadgeCheck size={21} /></span><dt>Exact ICOR coverage</dt><dd>{ranking.data.summary.exact_covered_base_units.toLocaleString('en-US')}</dd></div>
            <div><span aria-hidden="true" className="opportunity-summary__icon"><TriangleAlert size={21} /></span><dt>High-demand gap</dt><dd>{ranking.data.summary.high_demand_uncovered_base_units.toLocaleString('en-US')}</dd></div>
          </dl>
          {search.groupBy === 'eurocode' && ranking.data.eurocode_totals && <EurocodeTotals totals={ranking.data.eurocode_totals} />}
          {ranking.data.integrity_warnings.map((warning) => <p className="integrity-warning" key={warning} role="alert">{warning}</p>)}
          {ranking.data.eurocode_search && search.q && <EurocodeSearchNotice reading={ranking.data.eurocode_search} text={search.q} />}
          {ranking.data.items.length === 0 ? (
            <section className="opportunity-state">
              <h2>No forecast candidates match this view</h2>
              <p>{search.q
                ? ranking.data.eurocode_search?.label
                  ? `No vehicle in this view has “${search.q}” in its name or may carry Eurocode ${ranking.data.eurocode_search.code}. Its registration years may lie outside the forecast, or the market and year filters may exclude it.`
                  : `No vehicle matches “${search.q}” in this snapshot. It may be spelled differently by the registration source, or it may not be forecastable yet.`
                : 'Change the market or horizon filters to restore candidates.'}</p>
            </section>
          ) : (
            <OpportunityRanking
              onSelect={(groupId) => {
                const selected = ranking.data.items.find((row) => row.group_id === groupId)
                if (selected) {
                  queryClient.setQueryData(
                    queryKeys.opportunity(groupId, detailQuery),
                    selected,
                  )
                }
                onOpenDetails(groupId)
              }}
              rows={ranking.data.items}
              searchedCode={ranking.data.eurocode_search?.label ? ranking.data.eurocode_search.code : null}
              selectedGroup={selectedGroup}
            />
          )}
          {ranking.data.pages > 1 && (
            <nav aria-busy={ranking.isFetching} aria-label="Opportunity pages" className="pagination">
              <button disabled={ranking.isFetching || search.page <= 1} onClick={() => onSearchChange({ ...search, page: 1 })} type="button">First page</button>
              <button disabled={ranking.isFetching || search.page <= 1} onClick={() => onSearchChange({ ...search, page: search.page - 1 })} type="button">Previous page</button>
              <span aria-live="polite">{ranking.isFetching
                ? 'Loading page ' + search.page.toLocaleString('en-US') + '…'
                : <>Showing {((ranking.data.page - 1) * ranking.data.page_size + 1).toLocaleString('en-US')}–{Math.min(ranking.data.page * ranking.data.page_size, ranking.data.total).toLocaleString('en-US')} of {ranking.data.total.toLocaleString('en-US')} · Page {ranking.data.page} of {ranking.data.pages}</>}</span>
              <button disabled={ranking.isFetching || search.page >= ranking.data.pages} onClick={() => onSearchChange({ ...search, page: search.page + 1 })} type="button">Next page</button>
              <button disabled={ranking.isFetching || search.page >= ranking.data.pages} onClick={() => onSearchChange({ ...search, page: ranking.data.pages })} type="button">Last page</button>
            </nav>
          )}
        </>
      )}

      {!clientRelease && <details className="coverage-disclosure">
        <summary>Manage ICOR worked-model coverage</summary>
        <p>Records of vehicles ICOR has worked on. They are displayed and filterable; they do not change the score.</p>
        <CoverageManager apiClient={apiClient} opportunityQuery={opportunityQuery} />
      </details>}

    </div>
  )
}

export function OpportunitiesPage() {
  const navigate = useNavigate()
  const routeSearch = opportunitiesRoute.useSearch()
  const search = serializeOpportunitySearch(routeSearch)
  return (
    <OpportunitiesWorkbench
      invalidKeys={routeSearch.invalidKeys}
      onOpenDetails={(groupId) => void navigate({
        to: '/opportunities/$groupId',
        params: { groupId },
        search,
      })}
      onSearchChange={(nextSearch) => void navigate({ to: '/opportunities', search: nextSearch })}
      search={search}
    />
  )
}
