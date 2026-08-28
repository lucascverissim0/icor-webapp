import { useQuery } from '@tanstack/react-query'

import { plannerApi } from '../../lib/api/client'

function number(value: number): string {
  return new Intl.NumberFormat('en-US').format(value)
}

export function CompletenessPage() {
  const report = useQuery({
    queryKey: ['completeness'],
    queryFn: () => plannerApi.completeness(),
  })

  if (report.isPending) return <section className="route-state" aria-busy="true"><h2>Loading completeness…</h2></section>
  if (report.isError) return <section className="route-state" role="alert"><p className="eyebrow">Completeness unavailable</p><h2>The active snapshot report could not be loaded</h2><p>{report.error.message}</p></section>

  const totals = report.data.items.reduce((result, row) => ({
    observations: result.observations + row.observation_count,
    assigned: result.assigned + row.assigned_observation_count,
    forecastable: result.forecastable + row.forecastable_count,
    evidenceOnly: result.evidenceOnly + row.evidence_only_count,
  }), { observations: 0, assigned: 0, forecastable: 0, evidenceOnly: 0 })

  return (
    <div className="completeness-page">
      <header className="registrations-hero">
        <div><p className="eyebrow">Audit surface</p><h2>Snapshot completeness</h2><p>Exact observed, assigned, forecastable, and evidence-only counts by geography and year.</p></div>
        <span className="status-pill">Snapshot {report.data.snapshot_id}</span>
      </header>
      <dl className="summary-strip" aria-label="Completeness summary">
        <div><dt>Observations</dt><dd>{number(totals.observations)}</dd></div>
        <div><dt>Generation assigned</dt><dd>{number(totals.assigned)}</dd></div>
        <div><dt>Forecastable</dt><dd>{number(totals.forecastable)}</dd></div>
        <div><dt>Evidence only</dt><dd>{number(totals.evidenceOnly)}</dd></div>
      </dl>
      <section className="opportunity-ranking" aria-labelledby="completeness-table-title">
        <h2 id="completeness-table-title">Coverage by year</h2>
        <div className="table-scroll"><table><thead><tr><th>Geography</th><th>Year</th><th>Observed</th><th>Assigned</th><th>Forecastable</th><th>Evidence only</th><th>Rejected</th></tr></thead><tbody>
          {report.data.items.map((row) => <tr key={row.completeness_id}><td>{row.geography}</td><td>{row.year}</td><td>{number(row.observation_count)}</td><td>{number(row.assigned_observation_count)}</td><td>{number(row.forecastable_count)}</td><td>{number(row.evidence_only_count)}</td><td>{number(row.rejected_record_count)}</td></tr>)}
        </tbody></table></div>
      </section>
    </div>
  )
}
