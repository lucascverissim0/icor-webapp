import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'

import { configurationRoute, plannerRoute } from '../../app/router'
import { ApiProblem, PlannerApiClient, plannerApi } from '../../lib/api/client'
import type { components } from '../../lib/api/schema'
import { parsePlannerSearch, serializePlannerSearch, type PlannerSearch } from '../../lib/planner-search'

type Configuration = components['schemas']['PlanningConfigurationResponse']
type PlannerOptions = components['schemas']['PlannerOptionsResponse']

interface PlannerWorkbenchProps {
  apiClient?: PlannerApiClient
  invalidKeys?: string[]
  onSearchChange: (search: PlannerSearch) => void
  onSelect: (configurationId: string) => void
  search: PlannerSearch
}

const MARKET_NAMES: Record<string, string> = { DE: 'Germany', FR: 'France' }
const DEFAULT_SEARCH: PlannerSearch = { page: 1, sort: 'base_demand', direction: 'desc' }

function formatUnits(value: number): string {
  return `${new Intl.NumberFormat('en-US').format(value)} units`
}

function titleCase(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1)
}

function activeConstraintNames(search: PlannerSearch): string[] {
  return [
    ...(search.market ?? []).map((value) => MARKET_NAMES[value] ?? value),
    ...(search.horizon ?? []).map(String),
    ...(search.brand ?? []),
    ...(search.model ?? []),
    ...(search.evidence ?? []).map(titleCase),
  ]
}

function FilterGroup({ legend, values, selected, labelFor = String, onToggle }: {
  legend: string
  values: readonly (string | number)[]
  selected: readonly (string | number)[]
  labelFor?: (value: string | number) => string
  onToggle: (value: string | number) => void
}) {
  return (
    <fieldset className="filter-group">
      <legend>{legend}</legend>
      {values.map((value) => (
        <label key={value} className="filter-option">
          <input checked={selected.includes(value)} onChange={() => onToggle(value)} type="checkbox" />
          <span>{labelFor(value)}</span>
        </label>
      ))}
    </fieldset>
  )
}

function Filters({ options, search, onApply }: {
  options: PlannerOptions
  search: PlannerSearch
  onApply: (search: PlannerSearch) => void
}) {
  const [draft, setDraft] = useState(search)

  function toggle(key: 'market' | 'horizon' | 'brand' | 'model' | 'evidence', value: string | number) {
    const current = (draft[key] ?? []) as (string | number)[]
    const next = current.includes(value) ? current.filter((candidate) => candidate !== value) : [...current, value]
    setDraft({ ...draft, [key]: next.length > 0 ? next : undefined, page: 1 })
  }

  return (
    <form className="planner-filters" onSubmit={(event) => {
      event.preventDefault()
      onApply(parsePlannerSearch(draft as unknown as Record<string, unknown>).value)
    }}>
      <div className="filter-heading">
        <div><p className="eyebrow">Narrow the scenario</p><h2>Filters</h2></div>
        <span>{activeConstraintNames(draft).length} active</span>
      </div>
      <FilterGroup labelFor={(value) => MARKET_NAMES[String(value)] ?? String(value)} legend="Market" onToggle={(value) => toggle('market', value)} selected={draft.market ?? []} values={options.markets} />
      <FilterGroup legend="Forecast horizon" onToggle={(value) => toggle('horizon', value)} selected={draft.horizon ?? []} values={options.horizons} />
      <FilterGroup legend="Brand" onToggle={(value) => toggle('brand', value)} selected={draft.brand ?? []} values={options.brands} />
      <FilterGroup legend="Model" onToggle={(value) => toggle('model', value)} selected={draft.model ?? []} values={options.models} />
      <FilterGroup labelFor={(value) => titleCase(String(value))} legend="Evidence" onToggle={(value) => toggle('evidence', value)} selected={draft.evidence ?? []} values={options.evidence_statuses} />
      <button className="primary-action filter-apply" type="submit">Apply filters</button>
    </form>
  )
}

function Summary({ summary }: { summary: components['schemas']['PlannerSummaryResponse'] }) {
  const metrics = [
    ['Candidates', String(summary.candidate_count)],
    ['Downside', formatUnits(summary.downside_units)],
    ['Base demand', formatUnits(summary.base_units)],
    ['Upside', formatUnits(summary.upside_units)],
  ]
  return <dl className="summary-strip" aria-label="Planner summary">{metrics.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
}

function ConfigurationRow({ configuration, onSelect }: { configuration: Configuration; onSelect: (id: string) => void }) {
  return (
    <div className="configuration-row" role="row">
      <div className="configuration-identity" role="cell">
        <p className="configuration-title">{configuration.brand} {configuration.model}</p>
        <p>{configuration.generation}</p>
        <p>{configuration.market} · {configuration.forecast_horizon} · {configuration.body_style}</p>
      </div>
      <div role="cell">
        <span className="cell-label">P50 opportunity</span>
        <strong>{formatUnits(configuration.demand.base_units)}</strong>
        <span>{new Intl.NumberFormat('en-US').format(configuration.demand.downside_units)}–{formatUnits(configuration.demand.upside_units)}</span>
      </div>
      <div role="cell">
        <span className="cell-label">Evidence</span>
        <span className="status-pill">{titleCase(configuration.evidence_status)}</span>
        <span>Identity confidence: {titleCase(configuration.identity_confidence.level)}</span>
      </div>
      <div className="configuration-action" role="cell">
        <button type="button" onClick={() => onSelect(configuration.configuration_id)}>View details for {configuration.brand} {configuration.model}</button>
      </div>
    </div>
  )
}

function ProblemState({ error, onRetry }: { error: Error; onRetry: () => void }) {
  const correlation = error instanceof ApiProblem ? error.correlationId : null
  return (
    <section className="planner-state" role="alert">
      <p className="eyebrow">Planner unavailable</p><h2>We could not load these generation opportunities</h2>
      <p>{error.message}</p>{correlation && <p className="correlation">Reference: {correlation}</p>}
      <button className="primary-action" onClick={onRetry} type="button">Retry</button>
    </section>
  )
}

export function PlannerWorkbench({ apiClient = plannerApi, invalidKeys = [], onSearchChange, onSelect, search }: PlannerWorkbenchProps) {
  const optionsQuery = useQuery({ queryKey: ['planner', 'options'], queryFn: () => apiClient.options() })
  const parsedSearch = useMemo(() => {
    if (!optionsQuery.data) return { value: search, invalidKeys: [] }
    const parsed = parsePlannerSearch(search as unknown as Record<string, unknown>, {
      markets: optionsQuery.data.markets,
      horizons: optionsQuery.data.horizons,
      brands: optionsQuery.data.brands,
      models: optionsQuery.data.models,
      evidenceStatuses: optionsQuery.data.evidence_statuses,
    })
    return parsed.invalidKeys.length > 0
      ? { ...parsed, value: { ...parsed.value, page: 1 } }
      : parsed
  }, [optionsQuery.data, search])
  const canonicalSearch = parsedSearch.value
  useEffect(() => {
    if (parsedSearch.invalidKeys.length > 0) onSearchChange(canonicalSearch)
  }, [canonicalSearch, onSearchChange, parsedSearch.invalidKeys.length])
  const configurationsQuery = useQuery({
    queryKey: ['planner', 'configurations', canonicalSearch],
    queryFn: () => apiClient.configurations({
      markets: canonicalSearch.market, horizons: canonicalSearch.horizon, brands: canonicalSearch.brand,
      models: canonicalSearch.model, evidence: canonicalSearch.evidence, sort: canonicalSearch.sort,
      direction: canonicalSearch.direction, page: canonicalSearch.page,
    }),
    enabled: optionsQuery.isSuccess,
  })
  const constraints = activeConstraintNames(canonicalSearch)
  const normalizedKeys = [...new Set([...invalidKeys, ...parsedSearch.invalidKeys])].sort()

  if (optionsQuery.isPending) return <section className="planner-state" aria-busy="true"><h2>Loading planner controls…</h2></section>
  if (optionsQuery.isError) return <ProblemState error={optionsQuery.error} onRetry={() => void optionsQuery.refetch()} />

  return (
    <div className="planner-layout">
      <Filters key={JSON.stringify(canonicalSearch)} options={optionsQuery.data} search={canonicalSearch} onApply={onSearchChange} />
      <section className="planner-results" aria-labelledby="results-title">
        <div className="results-heading"><div><p className="eyebrow">Generation-level outlook</p><h2 id="results-title">Replacement opportunity planner</h2><p>Official registration history with explicit survival, hazard, and forecast assumptions. Values are opportunity ranges, not exact fitment demand.</p></div><span className="status-pill">Validated snapshot</span></div>
        {normalizedKeys.length > 0 && <p className="url-notice" role="status">Adjusted URL filters: {normalizedKeys.join(', ')}</p>}
        {constraints.length > 0 && (
          <div className="active-constraints">
            <p><strong>{constraints.length} active {constraints.length === 1 ? 'filter' : 'filters'}</strong></p>
            <button type="button" onClick={() => onSearchChange(DEFAULT_SEARCH)}>Reset filters</button>
          </div>
        )}
        {configurationsQuery.isPending && <section className="planner-state" aria-busy="true"><h2>Loading configurations…</h2></section>}
        {configurationsQuery.isError && <ProblemState error={configurationsQuery.error} onRetry={() => void configurationsQuery.refetch()} />}
        {configurationsQuery.data && <Summary summary={configurationsQuery.data.summary} />}
        <div className="result-controls">
          <label>
            <span>Sort results</span>
            <select
              aria-label="Sort results"
              onChange={(event) => onSearchChange({ ...canonicalSearch, page: 1, sort: event.target.value as PlannerSearch['sort'] })}
              value={canonicalSearch.sort}
            >
              <option value="base_demand">Base demand</option>
              <option value="downside_demand">Downside demand</option>
              <option value="upside_demand">Upside demand</option>
              <option value="brand">Brand</option>
              <option value="model">Model</option>
              <option value="identity_confidence">Identity confidence</option>
              <option value="data_quality_confidence">Data quality confidence</option>
            </select>
          </label>
          <button
            aria-label={`Sort ${canonicalSearch.direction === 'asc' ? 'descending' : 'ascending'}`}
            onClick={() => onSearchChange({ ...canonicalSearch, page: 1, direction: canonicalSearch.direction === 'asc' ? 'desc' : 'asc' })}
            type="button"
          >
            {canonicalSearch.direction === 'asc' ? 'Ascending' : 'Descending'}
          </button>
        </div>
        {configurationsQuery.data?.items.length === 0 && (
          <section className="planner-state"><p className="eyebrow">No matching generations</p><h2>No results for the selected filters</h2><p>Reset the active constraints to return to the complete forecastable set.</p></section>
        )}
        {configurationsQuery.data && configurationsQuery.data.items.length > 0 && (
          <div className="configuration-grid" role="table" aria-label="Generation opportunities">
            <div className="configuration-header" role="row"><span role="columnheader">Generation</span><span role="columnheader">Opportunity</span><span role="columnheader">Evidence</span><span role="columnheader">Action</span></div>
            {configurationsQuery.data.items.map((configuration) => <ConfigurationRow key={configuration.configuration_id} configuration={configuration} onSelect={onSelect} />)}
          </div>
        )}
        {configurationsQuery.data && configurationsQuery.data.pages > 1 && (
          <nav aria-label="Results pages" className="pagination">
            <button disabled={canonicalSearch.page <= 1} onClick={() => onSearchChange({ ...canonicalSearch, page: canonicalSearch.page - 1 })} type="button">Previous page</button>
            <span>Page {configurationsQuery.data.page} of {configurationsQuery.data.pages}</span>
            <button disabled={canonicalSearch.page >= configurationsQuery.data.pages} onClick={() => onSearchChange({ ...canonicalSearch, page: canonicalSearch.page + 1 })} type="button">Next page</button>
          </nav>
        )}
      </section>
    </div>
  )
}

export function PlannerPage() {
  const navigate = useNavigate()
  const routeSearch = plannerRoute.useSearch()
  const search = serializePlannerSearch(routeSearch)
  return <PlannerWorkbench invalidKeys={routeSearch.invalidKeys} onSearchChange={(nextSearch) => void navigate({ to: '/planner', search: nextSearch })} onSelect={(configurationId) => void navigate({ to: configurationRoute.to, params: { configurationId }, search })} search={search} />
}
