import { useQuery } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { useState } from 'react'

import { opportunitiesRoute } from '../../app/router'
import { queryKeys } from '../../app/query-client'
import { ApiProblem, PlannerApiClient, plannerApi, type OpportunitiesQuery } from '../../lib/api/client'
import { serializeOpportunitySearch, type OpportunitySearch } from '../../lib/opportunity-search'
import { CoverageManager } from './CoverageManager'
import { OpportunityDrillDown } from './OpportunityDrillDown'
import { OpportunityRanking } from './OpportunityRanking'


interface OpportunitiesWorkbenchProps {
  apiClient?: PlannerApiClient
  clientRelease?: boolean
  invalidKeys?: string[]
  onSearchChange: (search: OpportunitySearch) => void
  search: OpportunitySearch
}

function queryFromSearch(search: OpportunitySearch): OpportunitiesQuery {
  return {
    groupBy: search.groupBy,
    markets: search.market,
    horizons: search.horizon,
    page: search.page,
    pageSize: 25,
  }
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
  onSearchChange,
  search,
}: OpportunitiesWorkbenchProps) {
  const [selectedGroup, setSelectedGroup] = useState<string | null>(null)
  const opportunityQuery = queryFromSearch(search)
  const ranking = useQuery({
    queryKey: queryKeys.opportunities(opportunityQuery),
    queryFn: ({ signal }) => apiClient.opportunities(opportunityQuery, signal),
  })
  const registrationSummary = useQuery({
    queryKey: ['registrations', 'summary'],
    queryFn: () => apiClient.registrationSummary(),
  })
  const drillDown = useQuery({
    queryKey: queryKeys.opportunityConfigurations(selectedGroup ?? '', opportunityQuery),
    queryFn: () => apiClient.opportunityConfigurations(selectedGroup ?? '', opportunityQuery),
    enabled: selectedGroup !== null,
  })

  return (
    <div className="opportunities-page">
      <header className="opportunities-hero">
        <div>
          <p className="eyebrow">Windshield replacement forecast</p>
          <h2>Prioritized model and generation opportunities</h2>
          <p>{clientRelease
            ? 'This first client release shows only model-years with an unambiguous manufacturer-reviewed generation. Forecasts remain planning estimates.'
            : 'Start with the vehicle opportunities forecast for upcoming years. Demand drives up to 80 points; verified ICOR experience adds up to 20 readiness points.'}</p>
        </div>
        <span className="status-pill">{clientRelease ? 'Verified identity catalog' : 'Validated snapshot'}</span>
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
              setSelectedGroup(null)
              onSearchChange({ ...search, groupBy: value, page: 1 })
            }}
            type="button"
          >
            {label}
          </button>
        ))}
        </div>
      </div>}

      {ranking.isPending && <section aria-busy="true" className="opportunity-state"><h2>Loading opportunity ranking…</h2></section>}
      {ranking.isError && <ProblemState error={ranking.error} onRetry={() => void ranking.refetch()} />}
      {ranking.data && (
        <>
          <dl aria-label="Opportunity summary" className="opportunity-summary">
            <div><dt>Forecast replacements</dt><dd>{ranking.data.summary.base_units.toLocaleString('en-US')}</dd></div>
            <div><dt>Exact ICOR coverage</dt><dd>{ranking.data.summary.exact_covered_base_units.toLocaleString('en-US')}</dd></div>
            <div><dt>High-demand gap</dt><dd>{ranking.data.summary.high_demand_uncovered_base_units.toLocaleString('en-US')}</dd></div>
          </dl>
          {ranking.data.integrity_warnings.map((warning) => <p className="integrity-warning" key={warning} role="alert">{warning}</p>)}
          {ranking.data.items.length === 0 ? (
            <section className="opportunity-state"><h2>No forecast candidates match this view</h2><p>Change the market or horizon filters to restore candidates.</p></section>
          ) : (
            <OpportunityRanking
              onSelect={setSelectedGroup}
              rows={ranking.data.items}
              selectedGroup={selectedGroup}
            />
          )}
          {ranking.data.pages > 1 && (
            <nav aria-label="Opportunity pages" className="pagination">
              <button disabled={ranking.data.page <= 1} onClick={() => onSearchChange({ ...search, page: ranking.data.page - 1 })} type="button">Previous page</button>
              <span>Page {ranking.data.page} of {ranking.data.pages}</span>
              <button disabled={ranking.data.page >= ranking.data.pages} onClick={() => onSearchChange({ ...search, page: ranking.data.page + 1 })} type="button">Next page</button>
            </nav>
          )}
        </>
      )}

      {selectedGroup && (
        <OpportunityDrillDown
          error={drillDown.error}
          isPending={drillDown.isPending}
          onClose={() => setSelectedGroup(null)}
          onRetry={() => void drillDown.refetch()}
          rows={drillDown.data ?? []}
        />
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
      onSearchChange={(nextSearch) => void navigate({ to: '/opportunities', search: nextSearch })}
      search={search}
    />
  )
}
