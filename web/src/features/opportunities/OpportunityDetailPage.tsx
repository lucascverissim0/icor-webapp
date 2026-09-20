import { useQuery } from '@tanstack/react-query'

import { useNavigate } from '@tanstack/react-router'

import { opportunityDetailRoute } from '../../app/router'
import { queryKeys } from '../../app/query-client'
import { ApiProblem, PlannerApiClient, plannerApi, type OpportunitiesQuery } from '../../lib/api/client'
import { serializeOpportunitySearch, type OpportunitySearch } from '../../lib/opportunity-search'
import type { components } from '../../lib/api/schema'
import { OpportunitiesWorkbench } from './OpportunitiesPage'


function format(value: number): string {
  return new Intl.NumberFormat('en-US').format(value)
}

function queryFromSearch(search: OpportunitySearch): OpportunitiesQuery {
  return {
    groupBy: search.groupBy,
    markets: search.market,
    horizons: search.horizon,
  }
}

function backHref(search: OpportunitySearch): string {
  const parameters = new URLSearchParams({ groupBy: search.groupBy, page: String(search.page) })
  for (const market of search.market ?? []) parameters.append('market', market)
  for (const horizon of search.horizon ?? []) parameters.append('horizon', String(horizon))
  return `/opportunities?${parameters.toString()}`
}

type FleetEstimate = components['schemas']['OpportunityFleetEstimateResponse']

function FleetSummary({ estimates }: { estimates: FleetEstimate[] }) {
  const horizons = [...new Set(estimates.map((row) => row.forecast_horizon))].sort()
  return (
    <section aria-labelledby="fleet-title" className="detail-section">
      <div className="detail-section__heading"><p className="eyebrow">Surviving vehicles</p><h2 id="fleet-title">Estimated active fleet</h2></div>
      <p className="detail-note">Fleet is shown separately for each forecast horizon so the same vehicles are never added together across years.</p>
      <div className="fleet-horizon-list">
        {horizons.map((horizon) => {
          const regions = estimates.filter((row) => row.forecast_horizon === horizon)
          const total = regions.reduce((sum, row) => sum + row.estimated_fleet_units, 0)
          return (
            <article className="fleet-horizon" key={horizon}>
              <div className="fleet-horizon__total"><div><p className="eyebrow">Forecast horizon {horizon}</p><h3>Total estimated fleet</h3></div><strong>{format(total)} vehicles</strong></div>
              <dl aria-label={`${horizon} estimated fleet by world region`} className="fleet-regions">
                {regions.map((row) => <div key={row.world_region}><dt>{row.world_region}</dt><dd>{format(row.estimated_fleet_units)} vehicles</dd></div>)}
              </dl>
            </article>
          )
        })}
      </div>
    </section>
  )
}

function ProblemState({ error, retry, back }: { error: Error; retry: () => void; back: string }) {
  return (
    <section className="opportunity-state" role="alert">
      <p className="eyebrow">Ranking details unavailable</p>
      <h2>We could not load this opportunity</h2>
      <p>{error.message}</p>
      {error instanceof ApiProblem && error.correlationId && <p>Reference: {error.correlationId}</p>}
      <div className="detail-actions"><button className="primary-action" onClick={retry} type="button">Retry</button><a href={back}>Back to ranking</a></div>
    </section>
  )
}

export function OpportunityDetailView({
  apiClient = plannerApi,
  groupId,
  search,
}: {
  apiClient?: PlannerApiClient
  groupId: string
  search: OpportunitySearch
}) {
  const query = queryFromSearch(search)
  const opportunity = useQuery({
    queryKey: queryKeys.opportunity(groupId, query),
    queryFn: ({ signal }) => apiClient.opportunity(groupId, query, signal),
  })
  const contributions = useQuery({
    queryKey: queryKeys.opportunityContributions(groupId, query),
    queryFn: ({ signal }) => apiClient.opportunityContributions(groupId, query, signal),
  })
  const fleet = useQuery({
    queryKey: queryKeys.opportunityFleet(groupId, query),
    queryFn: ({ signal }) => apiClient.opportunityFleet(groupId, query, signal),
  })
  const back = backHref(search)

  if (opportunity.isPending) return <section aria-busy="true" className="opportunity-state"><h2>Loading ranking details...</h2></section>
  if (opportunity.isError) return <ProblemState back={back} error={opportunity.error} retry={() => void opportunity.refetch()} />

  const row = opportunity.data
  const title = [row.brand, row.model, row.model_year].filter(Boolean).join(' · ')
  return (
    <div className="opportunity-detail-page">
      <a className="back-link" href={back}>← Back to opportunity ranking</a>
      <header className="opportunity-detail-hero">
        <div><p className="eyebrow">Ranking detail</p><h2>{title}</h2><p className="generation-name">{row.generation_name ?? 'Generation review pending'}</p>{row.generation_source_url && <><a className="generation-source" href={row.generation_source_url} rel="noreferrer" target="_blank">Manufacturer generation source ↗</a><p className="detail-note">Mapped from the manufacturer production window. A registration-year cohort can include transition-year stock.</p></>}</div>
        <div className="detail-score"><span>Opportunity score</span><strong>{row.score.total_points.toFixed(1)}</strong><small>out of 100</small></div>
      </header>

      <section aria-labelledby="demand-title" className="detail-section">
        <div className="detail-section__heading"><p className="eyebrow">Forecast outcome</p><h2 id="demand-title">Replacement demand behind the rank</h2></div>
        <dl className="detail-metrics">
          <div><dt>Downside</dt><dd>{format(row.demand.downside_units)}</dd></div>
          <div className="detail-metrics__primary"><dt>Base forecast</dt><dd>{format(row.demand.base_units)}</dd></div>
          <div><dt>Upside</dt><dd>{format(row.demand.upside_units)}</dd></div>
          <div><dt>Contributing forecasts</dt><dd>{format(row.contributing_configuration_count)}</dd></div>
        </dl>
        <p className="detail-note">Replacement estimates use the validated snapshot and remain planning estimates, not observed future claims.</p>
      </section>

      <section aria-labelledby="score-title" className="detail-section detail-score-breakdown">
        <div className="detail-section__heading"><p className="eyebrow">Transparent calculation</p><h2 id="score-title">Why this opportunity ranks here</h2></div>
        <div className="score-bars">
          <div><div><strong>Demand</strong><span>{row.score.demand_points.toFixed(1)} / 80</span></div><progress aria-label="Demand score" max="80" value={row.score.demand_points} /><p>{Math.round(row.score.demand_percentile * 100)}th percentile among the current filtered ranking.</p></div>
          <div><div><strong>ICOR readiness</strong><span>{row.score.readiness_points.toFixed(1)} / 20</span></div><progress aria-label="Readiness score" max="20" value={row.score.readiness_points} /><p>Exact coverage receives full weight; vehicle-year or worked-model fallback receives half weight.</p></div>
        </div>
      </section>

      {fleet.isPending && <section aria-busy="true" className="detail-section"><h2>Loading estimated fleet...</h2></section>}
      {fleet.isError && <section className="detail-section" role="alert"><h2>Estimated fleet unavailable</h2><p>{fleet.error.message}</p><button onClick={() => void fleet.refetch()} type="button">Retry fleet estimate</button></section>}
      {fleet.data && <FleetSummary estimates={fleet.data} />}

      <section aria-labelledby="coverage-title" className="detail-section">
        <div className="detail-section__heading"><p className="eyebrow">Production position</p><h2 id="coverage-title">Coverage of base demand</h2></div>
        <dl className="detail-metrics detail-metrics--three">
          <div><dt>Exact configuration</dt><dd>{format(row.exact_covered_base_units)}</dd></div>
          <div><dt>Vehicle-year fallback</dt><dd>{format(row.fallback_covered_base_units)}</dd></div>
          <div><dt>Uncovered</dt><dd>{format(row.uncovered_base_units)}</dd></div>
        </dl>
      </section>

      <section aria-labelledby="contributions-title" className="detail-section">
        <div className="detail-section__heading"><p className="eyebrow">Underlying forecast rows</p><h2 id="contributions-title">Market and horizon contributions</h2></div>
        {contributions.isPending && <p aria-busy="true">Loading contributions...</p>}
        {contributions.isError && <div role="alert"><p>{contributions.error.message}</p><button onClick={() => void contributions.refetch()} type="button">Retry contributions</button></div>}
        {contributions.data && <div className="detail-contribution-list">{contributions.data.map((item) => (
          <article className="detail-contribution" key={item.configuration_id}>
            <div><h3>{item.market} · {item.forecast_horizon}</h3><p>{item.generation} · {item.body_style}</p></div>
            <dl><div><dt>Base demand</dt><dd>{format(item.demand.base_units)}</dd></div><div><dt>Range</dt><dd>{format(item.demand.downside_units)}–{format(item.demand.upside_units)}</dd></div></dl>
          </article>
        ))}</div>}
      </section>
    </div>
  )
}

export function OpportunityDetailPage() {
  const navigate = useNavigate()
  const { groupId } = opportunityDetailRoute.useParams()
  const routeSearch = opportunityDetailRoute.useSearch()
  const search = serializeOpportunitySearch(routeSearch)
  return (
    <div className="opportunity-detail-route">
      <div className="opportunity-detail-context">
        <OpportunitiesWorkbench
          onOpenDetails={(nextGroupId) => void navigate({
            to: '/opportunities/$groupId',
            params: { groupId: nextGroupId },
            search,
          })}
          onSearchChange={(nextSearch) => void navigate({ to: '/opportunities', search: nextSearch })}
          search={search}
          selectedGroup={groupId}
        />
      </div>
      <aside aria-label="Selected opportunity" className="opportunity-detail-panel">
        <OpportunityDetailView groupId={groupId} search={search} />
      </aside>
    </div>
  )
}
