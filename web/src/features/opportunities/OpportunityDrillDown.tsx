import { ApiProblem } from '../../lib/api/client'
import type { components } from '../../lib/api/schema'


type DrillDownRow = components['schemas']['OpportunityDrillDownResponse']

export function OpportunityDrillDown({ rows, isPending, error, onRetry, onClose }: {
  rows: DrillDownRow[]
  isPending: boolean
  error: Error | null
  onRetry: () => void
  onClose: () => void
}) {
  return (
    <section aria-labelledby="drilldown-title" className="opportunity-drilldown">
      <div className="drilldown-heading">
        <div><p className="eyebrow">Configuration contribution</p><h2 id="drilldown-title">Demand behind this opportunity</h2></div>
        <button onClick={onClose} type="button">Close details</button>
      </div>
      {isPending && <p aria-busy="true">Loading contributing configurations…</p>}
      {error && (
        <div role="alert"><p>{error.message}</p>{error instanceof ApiProblem && error.correlationId && <p>Reference: {error.correlationId}</p>}<button onClick={onRetry} type="button">Retry details</button></div>
      )}
      {rows.map((row) => (
        <article className="drilldown-row" key={`${row.configuration.configuration_id}-${row.model_year_demand.model_year}`}>
          <div><h3>{row.configuration.brand} {row.configuration.model}</h3><p>{row.configuration.sku ?? 'SKU unknown'} · {row.configuration.generation}</p></div>
          <dl><div><dt>Model year</dt><dd>{row.model_year_demand.model_year}</dd></div><div><dt>Base demand</dt><dd>{row.model_year_demand.demand.base_units.toLocaleString('en-US')} replacements</dd></div><div><dt>Coverage</dt><dd>{row.coverage_status.replaceAll('_', ' ')}</dd></div></dl>
        </article>
      ))}
    </section>
  )
}
