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
            placeholder="Brand or model, for example Golf"
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
}

function formatCount(value: string | number): string {
  return Number(value).toLocaleString('en-US')
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

  return (
    <div className="opportunities-page">
      <header className="opportunities-hero">
        <div>
          <p className="eyebrow">Windshield replacement forecast</p>
          <h2>{clientRelease ? 'Prioritized vehicle-year opportunities' : 'Prioritized model and generation opportunities'}</h2>
          <p>{clientRelease
            ? 'This client release covers every forecastable official-source make/model label and registration cohort year. Ranked generation names use reviewed manufacturer production windows; forecasts remain planning estimates.'
            : 'Start with the vehicle opportunities forecast for upcoming years. Demand drives up to 80 points; verified ICOR experience adds up to 20 readiness points.'}</p>
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
          <p><strong>Demand points = demand percentile × 80</strong><span>The highest forecast demand approaches 80 points; this does not change the replacement forecast.</span></p>
          <p><strong>Readiness points = (exact units + 0.5 × fallback units) ÷ total units × 20</strong><span>Exact ICOR configuration coverage gets full weight. Vehicle-year and legacy worked-model matches get half weight. Uncovered units get zero.</span></p>
          <p><strong>Total score = demand points + readiness points</strong><span>Maximum 100 points: 80 for market demand and 20 for ICOR readiness.</span></p>
        </div>
        <p className="score-method__scope" role="note">
          Scores are whole-market. The demand percentile compares each vehicle
          with{' '}
          {ranking.data?.demand_population
            ? `all ${formatCount(ranking.data.demand_population)} ranked ${RANKED_NOUN[search.groupBy]}`
            : 'every ranked vehicle'}{' '}
          in this snapshot — every market and both forecast horizons — so a
          vehicle scores the same here as it does anywhere else in the app.
          Filtering or searching changes which rows appear and the units they
          carry, never the score.
        </p>
      </section>

      {invalidKeys.length > 0 && (
        <p className="url-notice" role="status">Adjusted URL filters: {invalidKeys.join(', ')}</p>
      )}

      {!clientRelease && <div>
        <p className="grouping-label">Summarize ranking by</p>
        <div aria-label="Opportunity grouping" className="grouping-control" role="group">
        {([
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
      {ranking.isError && <ProblemState error={ranking.error} onRetry={() => void ranking.refetch()} />}
      {ranking.data && (
        <>
          <aside className="dataset-coverage" aria-label="Available opportunity data">
            <div><strong>{formatCount(ranking.data.total)}</strong>{' '}<span>ranked records available</span></div>
            {registrationSummary.data && <div><strong>{formatCount(registrationSummary.data.model_count)}</strong>{' '}<span>official model labels</span></div>}
            {registrationSummary.data && <div><strong>{formatCount(registrationSummary.data.total_registrations)}</strong>{' '}<span>registrations represented</span></div>}
          </aside>
          <dl aria-label="Opportunity summary" className="opportunity-summary">
            <div><span aria-hidden="true" className="opportunity-summary__icon"><ChartNoAxesCombined size={21} /></span><dt>Forecast replacements</dt><dd>{ranking.data.summary.base_units.toLocaleString('en-US')}</dd></div>
            <div><span aria-hidden="true" className="opportunity-summary__icon"><BadgeCheck size={21} /></span><dt>Exact ICOR coverage</dt><dd>{ranking.data.summary.exact_covered_base_units.toLocaleString('en-US')}</dd></div>
            <div><span aria-hidden="true" className="opportunity-summary__icon"><TriangleAlert size={21} /></span><dt>High-demand gap</dt><dd>{ranking.data.summary.high_demand_uncovered_base_units.toLocaleString('en-US')}</dd></div>
          </dl>
          {ranking.data.integrity_warnings.map((warning) => <p className="integrity-warning" key={warning} role="alert">{warning}</p>)}
          {ranking.data.items.length === 0 ? (
            <section className="opportunity-state">
              <h2>No forecast candidates match this view</h2>
              <p>{search.q
                ? `No vehicle matches “${search.q}” in this snapshot. It may be spelled differently by the registration source, or it may not be forecastable yet.`
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
        <p>Use this only to maintain the experience records that affect the readiness portion of the score.</p>
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
