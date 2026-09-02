import { useQuery } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { ExternalLink, ShieldCheck } from 'lucide-react'
import { useState, type FormEvent } from 'react'

import {
  ApiProblem,
  PlannerApiClient,
  plannerApi,
  type EvidenceObservationsQuery,
} from '../../lib/api/client'
import { evidenceRoute } from '../../app/router'
import type { EvidenceSearch } from '../../lib/evidence-search'


interface EvidenceWorkbenchProps {
  apiClient?: PlannerApiClient
  onSearchChange?: (search: EvidenceSearch) => void
  search?: EvidenceSearch
}

function number(value: number | string): string {
  return Number(value).toLocaleString('en-US')
}

function title(value: string): string {
  return value.replaceAll('_', ' ').replace(/^./, (letter) => letter.toUpperCase())
}

function Unavailable({ error, retry }: { error: Error; retry: () => void }) {
  const correlation = error instanceof ApiProblem ? error.correlationId : null
  return (
    <section className="evidence-state" role="alert">
      <p className="eyebrow">Validated candidate required</p>
      <h2>Source evidence is unavailable</h2>
      <p>{error.message}</p>
      {correlation && <p className="correlation">Reference: {correlation}</p>}
      <button className="primary-action" onClick={retry} type="button">Retry</button>
    </section>
  )
}

export function EvidenceWorkbench({
  apiClient = plannerApi,
  onSearchChange,
  search,
}: EvidenceWorkbenchProps) {
  const [draftSearch, setDraftSearch] = useState('')
  const [showObservations, setShowObservations] = useState(false)
  const [localQuery, setLocalQuery] = useState<EvidenceSearch>({ page: 1 })
  const routeQuery = search ?? localQuery
  const query: EvidenceObservationsQuery = { ...routeQuery, pageSize: 25 }
  const updateQuery = (update: (current: EvidenceSearch) => EvidenceSearch) => {
    const next = update(routeQuery)
    if (onSearchChange) onSearchChange(next)
    else setLocalQuery(next)
  }
  const summary = useQuery({
    queryKey: ['evidence', 'summary'],
    queryFn: () => apiClient.evidenceSummary(),
  })
  const observations = useQuery({
    queryKey: ['evidence', 'observations', query],
    queryFn: () => apiClient.evidenceObservations(query),
    enabled: summary.isSuccess && showObservations,
  })

  if (summary.isError) return <Unavailable error={summary.error} retry={() => void summary.refetch()} />

  function applyFilters(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    updateQuery((current) => ({ ...current, search: draftSearch.trim() || undefined, page: 1 }))
  }

  return (
    <div className="evidence-page">
      <header className="evidence-hero">
        <div>
          <p className="eyebrow">Official source review</p>
          <h2>Source evidence</h2>
          <p>Inspect validated public-source records and exact-normalized model-family mappings before reconciliation, estimation, or forecasting.</p>
        </div>
        <span className="candidate-pill"><ShieldCheck aria-hidden="true" size={17} /> Candidate only</span>
      </header>

      <section className="evidence-boundary" aria-label="Evidence interpretation boundary">
        <strong>Exact normalized model-family identity; model year unavailable</strong>
        <p>Registration year is not model year. This candidate does not feed forecasts, and zero values are published from it.</p>
      </section>

      {summary.isPending && <section aria-busy="true" className="evidence-state"><h2>Validating source evidence…</h2></section>}
      {summary.data && (
        <>
          <dl className="evidence-metrics" aria-label="Candidate evidence summary">
            <div><dt>Observations</dt><dd>{number(summary.data.observation_count)}</dd></div>
            <div><dt>Official releases</dt><dd>{number(summary.data.releases.length)}</dd></div>
            <div><dt>Published values</dt><dd>{number(summary.data.published_value_count)}</dd></div>
            <div><dt>Validation warnings</dt><dd>{number(summary.data.warning_count)}</dd></div>
          </dl>

          <section className="evidence-panel">
            <div className="evidence-heading">
              <div><p className="eyebrow">Release ledger</p><h3>Publisher provenance</h3></div>
              <code title={summary.data.database_sha256}>Snapshot {summary.data.snapshot_id.replace('snapshot-', '')}</code>
            </div>
            <div className="release-grid">
              {summary.data.releases.filter((release, _, releases) => !releases.some(
                (candidate) => candidate.source_id === release.source_id &&
                  (candidate.published_at > release.published_at ||
                    (candidate.published_at === release.published_at &&
                      candidate.release_id > release.release_id)),
              )).map((release) => (
                <article className="release-card" key={release.release_id}>
                  <div><span>{title(release.measure)}</span><span>{title(release.publication_status)}</span></div>
                  <h4>{release.publisher}</h4>
                  <p><strong>Latest release:</strong> {release.release_id} · published {release.published_at.slice(0, 10)}</p>
                  <p>{release.geography} · {release.coverage_start} to {release.coverage_end}</p>
                  <p><strong>What it proves:</strong> {release.what_it_proves}</p>
                  <p><strong>Limits:</strong> {release.limitations}</p>
                  <dl>
                    <div><dt>Raw</dt><dd>{number(release.raw_record_count)}</dd></div>
                    <div><dt>Accepted</dt><dd>{number(release.accepted_record_count)}</dd></div>
                    <div><dt>Rejected</dt><dd>{number(release.rejected_record_count)}</dd></div>
                    <div><dt>Quarantined</dt><dd>{number(release.quarantined_record_count)}</dd></div>
                    <div><dt>Warnings</dt><dd>{number(release.validation_warning_count)}</dd></div>
                  </dl>
                  <nav aria-label={`${release.publisher} evidence links`}>
                    <a href={release.source_url} rel="noopener noreferrer" target="_blank">Open source <ExternalLink aria-hidden="true" size={14} /></a>
                    <a href={release.terms_url} rel="noopener noreferrer" target="_blank">Usage terms</a>
                  </nav>
                </article>
              ))}
            </div>
          </section>

          <section className="evidence-panel">
            <div className="evidence-heading">
              <div><p className="eyebrow">Observation browser</p><h3>Publisher-supplied rows</h3></div>
              <span>{number(summary.data.mapping_status_counts.normalized_label ?? 0)} exact-normalized · {number(summary.data.mapping_status_counts.rejected ?? 0)} rejected</span>
            </div>
            {!showObservations && (
              <div className="evidence-state">
                <p>Raw publisher rows are available as a bounded, paginated drill-down.</p>
                <button className="primary-action" onClick={() => setShowObservations(true)} type="button">
                  Browse raw observations
                </button>
              </div>
            )}
            {showObservations && <>
            <p className="evidence-context">
              Observation year says when a count was observed. First registration, manufacture,
              and model year describe the vehicles inside that observation and are never relabelled
              as annual registrations.
            </p>
            <form className="evidence-filters" onSubmit={applyFilters}>
              <label>Search source labels<input aria-label="Search source labels" maxLength={100} onChange={(event) => setDraftSearch(event.target.value)} type="search" value={draftSearch} /></label>
              <label>Release<select onChange={(event) => updateQuery((current) => ({ ...current, releaseId: event.target.value || undefined, page: 1 }))} value={query.releaseId ?? ''}><option value="">All releases</option>{summary.data.releases.map((release) => <option key={release.release_id} value={release.release_id}>{release.release_id}</option>)}</select></label>
              <label>Geography<select onChange={(event) => updateQuery((current) => ({ ...current, geography: event.target.value || undefined, page: 1 }))} value={query.geography ?? ''}><option value="">All geographies</option>{summary.data.geographies.map((geography) => <option key={geography}>{geography}</option>)}</select></label>
              <label>Measure<select onChange={(event) => updateQuery((current) => ({ ...current, measure: (event.target.value || undefined) as EvidenceObservationsQuery['measure'], page: 1 }))} value={query.measure ?? ''}><option value="">All measures</option>{summary.data.measures.map((measure) => <option key={measure} value={measure}>{title(measure)}</option>)}</select></label>
              <label>Observation year<input aria-label="Observation year" min={1900} max={2200} onChange={(event) => updateQuery((current) => ({ ...current, observationYear: event.target.value ? Number(event.target.value) : undefined, page: 1 }))} type="number" value={query.observationYear ?? ''} /></label>
              <label>Year field<select aria-label="Year field" onChange={(event) => updateQuery((current) => ({ ...current, yearSemantics: (event.target.value || undefined) as EvidenceObservationsQuery['yearSemantics'], page: 1 }))} value={query.yearSemantics ?? ''}>
                <option value="">Any year field</option>
                <option value="observation_year">Observation year</option>
                <option value="registration_cohort_year">First registration</option>
                <option value="manufacture_year">Manufacture year</option>
                <option value="model_year">Model year</option>
              </select></label>
              <button className="primary-action" type="submit">Apply filters</button>
            </form>

            {observations.isError && <Unavailable error={observations.error} retry={() => void observations.refetch()} />}
            {observations.isPending && <div aria-busy="true" className="evidence-state">Loading observations…</div>}
            {observations.data && (
              <>
                <div
                  aria-label="Scrollable source observations"
                  className="evidence-table-wrap"
                  role="region"
                  tabIndex={0}
                >
                  <table className="evidence-table">
                    <caption>{number(observations.data.total)} source observations</caption>
                    <thead><tr><th>Reported identity</th><th>Year semantics</th><th>Measure</th><th>Mapping</th><th>Provenance</th></tr></thead>
                    <tbody>{observations.data.items.map((row) => (
                      <tr key={row.observation_id}>
                        <td><strong>{row.original_make}</strong><span>{row.original_model}</span>{row.original_type && <small>{row.original_type}</small>}</td>
                        <td>
                          <strong>{row.geography}</strong>
                          <span>Observation year <b>{row.observation_year}</b></span>
                          {row.registration_cohort_year !== null && <span>First registration <b>{row.registration_cohort_year}</b></span>}
                          {row.manufacture_year !== null && <span>Manufacture year <b>{row.manufacture_year}</b></span>}
                          {row.model_year !== null && <span>Model year <b>{row.model_year}</b></span>}
                        </td>
                        <td><strong>{number(row.value)} {row.unit}</strong><span>{title(row.measure)}</span></td>
                        <td><span className="mapping-pill">{title(row.mapping_status)}</span><small>Confidence {row.confidence_total}/100</small></td>
                        <td>
                          <strong>{row.release_id}</strong><span>{row.original_row_locator}</span>
                          {row.transformation_notes.map((note) => <small key={note}>{note}</small>)}
                          {row.validation_flags.map((flag) => <small key={flag}>Validation: {title(flag)}</small>)}
                        </td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
                <nav aria-label="Observation pages" className="pagination">
                  <button disabled={observations.data.page <= 1} onClick={() => updateQuery((current) => ({ ...current, page: (current.page ?? 1) - 1 }))} type="button">Previous</button>
                  <span>Page {observations.data.page} of {observations.data.pages}</span>
                  <button disabled={observations.data.page >= observations.data.pages} onClick={() => updateQuery((current) => ({ ...current, page: (current.page ?? 1) + 1 }))} type="button">Next</button>
                </nav>
              </>
            )}
            </>}
          </section>
        </>
      )}
    </div>
  )
}

export function EvidencePage() {
  const navigate = useNavigate()
  const search = evidenceRoute.useSearch()
  return <EvidenceWorkbench
    onSearchChange={(next) => void navigate({ to: '/evidence', search: next })}
    search={search}
  />
}
