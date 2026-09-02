import { useQuery } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { Database, Search, ShieldCheck } from 'lucide-react'
import { useState, type FormEvent } from 'react'

import { registrationsRoute } from '../../app/router'
import {
  ApiProblem,
  PlannerApiClient,
  plannerApi,
  type RegistrationRankingQuery,
} from '../../lib/api/client'
import type { RegistrationSearch } from '../../lib/registration-search'


interface RegistrationsWorkbenchProps {
  apiClient?: PlannerApiClient
  onSearchChange?: (search: RegistrationSearch) => void
  search?: RegistrationSearch
}

function number(value: number | string): string {
  return Number(value).toLocaleString('en-US')
}

function layerPriority(layer: { status: string; evidence_kind: string }): number {
  const publication = layer.status === 'final' ? 0 : layer.status === 'provisional' ? 2 : 4
  const evidence = layer.evidence_kind === 'observed' ? 0 : 1
  return publication + evidence
}

function Unavailable({ error, retry }: { error: Error; retry: () => void }) {
  const correlation = error instanceof ApiProblem ? error.correlationId : null
  return (
    <section className="registration-state" role="alert">
      <p className="eyebrow">Verified snapshot required</p>
      <h2>Official registration data is unavailable</h2>
      <p>{error.message}</p>
      {correlation && <p className="correlation">Reference: {correlation}</p>}
      <button className="primary-action" onClick={retry} type="button">Retry</button>
    </section>
  )
}

export function RegistrationsWorkbench({
  apiClient = plannerApi,
  onSearchChange,
  search,
}: RegistrationsWorkbenchProps) {
  const [localSearch, setLocalSearch] = useState<RegistrationSearch>({ geography: 'EU27', year: 2024, page: 1 })
  const routeSearch = search ?? localSearch
  const [draftSearch, setDraftSearch] = useState(routeSearch.search ?? '')
  const query: RegistrationRankingQuery = {
    geography: routeSearch.geography ?? 'EU27',
    year: routeSearch.year ?? 2024,
    search: routeSearch.search,
    page: routeSearch.page ?? 1,
    pageSize: 25,
  }
  const updateSearch = (next: RegistrationSearch) => {
    if (onSearchChange) onSearchChange(next)
    else setLocalSearch(next)
  }
  const summary = useQuery({
    queryKey: ['registrations', 'summary'],
    queryFn: () => apiClient.registrationSummary(),
  })
  const selectedLayer = summary.data?.availability
    .filter((item) => item.geography === query.geography && item.year === query.year)
    .sort((left, right) => layerPriority(left) - layerPriority(right))[0]
  const scopeAvailable = selectedLayer !== undefined
  const ranking = useQuery({
    queryKey: ['registrations', 'ranking', query],
    queryFn: () => apiClient.registrationRanking(query),
    enabled: summary.isSuccess && scopeAvailable,
  })

  if (summary.isError) {
    return <Unavailable error={summary.error} retry={() => void summary.refetch()} />
  }
  if (ranking.isError) {
    return <Unavailable error={ranking.error} retry={() => void ranking.refetch()} />
  }

  function applySearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    updateSearch({ ...routeSearch, search: draftSearch.trim() || undefined, page: 1 })
  }

  return (
    <div className="registrations-page">
      <header className="registrations-hero">
        <div>
          <p className="eyebrow">European passenger-car evidence</p>
          <h2>{selectedLayer?.evidence_kind === 'estimated' ? 'Estimated' : 'Official'} {query.year} registrations</h2>
          <p>Ranked make and model families with the evidence layer and source lineage kept visible.</p>
        </div>
        <span className="official-pill"><ShieldCheck aria-hidden="true" size={17} /> {selectedLayer ? `${selectedLayer.status} ${selectedLayer.evidence_kind}` : 'Scope unavailable'}</span>
      </header>

      <section className="registration-boundary" aria-label="Registration interpretation boundary">
        <Database aria-hidden="true" size={20} />
        <div>
          <strong>Registration year is not model year</strong>
          <p>Model year is unavailable. Windshield fitment and replacement forecasts are not inferred from this dataset.</p>
        </div>
      </section>

      {summary.isPending && <section aria-busy="true" className="registration-state"><h2>Verifying official snapshot…</h2></section>}
      {summary.data && (
        <section className="registration-summary" aria-label="Official registration summary">
          <div><span>Latest EU27 registrations</span><strong>{number(summary.data.total_registrations)}</strong></div>
          <div><span>Canonical model families</span><strong>{number(summary.data.model_count)}</strong></div>
          <div><span>History</span><strong>{summary.data.years[0]}–{summary.data.years.at(-1)}</strong></div>
          <div><span>Snapshot</span><code>{summary.data.snapshot_id.replace('snapshot-', '')}</code></div>
        </section>
      )}

      <section className="registration-panel">
        <div className="registration-heading">
          <div><p className="eyebrow">Observed market evidence</p><h3>EU27 make and model ranking</h3></div>
          <a href="/evidence">Inspect source evidence</a>
        </div>
        <form className="registration-search" onSubmit={applySearch}>
          {summary.data && <div className="registration-scope">
            <label>Geography<select value={query.geography} onChange={(event) => updateSearch({ ...routeSearch, geography: event.target.value, page: 1 })}>{summary.data.geographies.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
            <label>Registration year<select value={query.year} onChange={(event) => updateSearch({ ...routeSearch, year: Number(event.target.value), page: 1 })}>{summary.data.years.map((value) => {
              const available = summary.data.availability.some(
                (item) => item.geography === query.geography && item.year === value,
              )
              return <option key={value} value={value}>{value}{available ? '' : ` — unavailable for ${query.geography}`}</option>
            })}</select></label>
          </div>}
          <label htmlFor="registration-search">Search make or model</label>
          <div>
            <Search aria-hidden="true" size={18} />
            <input
              aria-label="Search make or model"
              id="registration-search"
              maxLength={100}
              onChange={(event) => setDraftSearch(event.target.value)}
              placeholder="e.g. Alfa Romeo"
              type="search"
              value={draftSearch}
            />
            <button className="primary-action" type="submit">Search registrations</button>
          </div>
        </form>

        {summary.data && !scopeAvailable && (
          <div className="registration-state">
            <h3>No official model-family data for this scope</h3>
            <p>{query.geography} {query.year} is unavailable, not zero. Choose a year labelled as available.</p>
          </div>
        )}
        {scopeAvailable && ranking.isPending && <div aria-busy="true" className="registration-state">Loading official ranking…</div>}
        {ranking.data && ranking.data.items.length === 0 && (
          <div className="registration-state"><h3>No matching make or model</h3><p>Clear the search to return to the complete ranking.</p></div>
        )}
        {ranking.data && ranking.data.items.length > 0 && (
          <>
            <div className="registration-table-wrap">
              <table className="registration-table">
                <caption>{number(ranking.data.total)} canonical model families</caption>
                <thead><tr><th>Rank</th><th>Make and model</th><th>{query.year} registrations</th><th>Evidence</th></tr></thead>
                <tbody>{ranking.data.items.map((row) => (
                  <tr key={row.vehicle_id}>
                    <td data-label="Rank"><strong>#{row.rank}</strong></td>
                    <td data-label="Make and model">
                      <strong>{row.make}</strong><span>{row.model}</span><small>Model year unavailable</small>
                      {row.label_breakdown.length > 0 && <details>
                        <summary>Official source labels</summary>
                        <ul>{row.label_breakdown.map((label) => (
                          <li key={`${label.source_make}:${label.source_model}`}>
                            <span>{label.source_make} {label.source_model}</span>
                            <strong>{number(label.registrations)} registrations</strong>
                          </li>
                        ))}</ul>
                      </details>}
                    </td>
                    <td data-label={`${query.year} registrations`}><strong>{number(row.registrations)}</strong><span>{row.publication_status[0].toUpperCase() + row.publication_status.slice(1)} {row.evidence_kind}</span></td>
                    <td data-label="Evidence"><strong>{row.evidence_confidence}/100</strong><span>{number(row.input_observation_count)} source groups</span></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
            <nav aria-label="Registration pages" className="pagination">
              <button
                disabled={ranking.data.page <= 1}
                onClick={() => updateSearch({ ...routeSearch, page: (routeSearch.page ?? 1) - 1 })}
                type="button"
              >Previous</button>
              <span>Page {ranking.data.page} of {ranking.data.pages}</span>
              <button
                disabled={ranking.data.page >= ranking.data.pages}
                onClick={() => updateSearch({ ...routeSearch, page: (routeSearch.page ?? 1) + 1 })}
                type="button"
              >Next</button>
            </nav>
          </>
        )}
      </section>
    </div>
  )
}

export function RegistrationsPage() {
  const navigate = useNavigate()
  const search = registrationsRoute.useSearch()
  return (
    <RegistrationsWorkbench
      onSearchChange={(next) => void navigate({ to: '/registrations', search: next })}
      search={search}
    />
  )
}
