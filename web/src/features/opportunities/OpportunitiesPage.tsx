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
  invalidKeys?: string[]
  onSearchChange: (search: OpportunitySearch) => void
  search: OpportunitySearch
}

function queryFromSearch(search: OpportunitySearch): OpportunitiesQuery {
  return {
    groupBy: search.groupBy,
    markets: search.market,
    horizons: search.horizon,
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
  invalidKeys = [],
  onSearchChange,
  search,
}: OpportunitiesWorkbenchProps) {
  const [selectedGroup, setSelectedGroup] = useState<string | null>(null)
  const opportunityQuery = queryFromSearch(search)
  const ranking = useQuery({
    queryKey: queryKeys.opportunities(opportunityQuery),
    queryFn: () => apiClient.opportunities(opportunityQuery),
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
          <p className="eyebrow">Production opportunity planning</p>
          <h2>Where demand and readiness meet</h2>
          <p>Rank generation-level replacement opportunities without changing the baseline, then inspect the separate advantage from existing ICOR production.</p>
        </div>
        <span className="status-pill">Validated snapshot</span>
      </header>

      {invalidKeys.length > 0 && (
        <p className="url-notice" role="status">Adjusted URL filters: {invalidKeys.join(', ')}</p>
      )}

      <div aria-label="Opportunity grouping" className="grouping-control" role="group">
        {([
          ['brand', 'Brands'],
          ['model', 'Models'],
          ['model_year', 'Model years'],
        ] as const).map(([value, label]) => (
          <button
            aria-pressed={search.groupBy === value}
            key={value}
            onClick={() => {
              setSelectedGroup(null)
              onSearchChange({ ...search, groupBy: value })
            }}
            type="button"
          >
            {label}
          </button>
        ))}
      </div>

      {ranking.isPending && <section aria-busy="true" className="opportunity-state"><h2>Loading opportunity ranking…</h2></section>}
      {ranking.isError && <ProblemState error={ranking.error} onRetry={() => void ranking.refetch()} />}
      {ranking.data && (
        <>
          <dl aria-label="Opportunity summary" className="opportunity-summary">
            <div><dt>Base replacements</dt><dd>{ranking.data.summary.base_units.toLocaleString('en-US')}</dd></div>
            <div><dt>Exact-covered base</dt><dd>{ranking.data.summary.exact_covered_base_units.toLocaleString('en-US')}</dd></div>
            <div><dt>Uncovered high demand</dt><dd>{ranking.data.summary.high_demand_uncovered_base_units.toLocaleString('en-US')}</dd></div>
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

      <CoverageManager apiClient={apiClient} opportunityQuery={opportunityQuery} />

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
