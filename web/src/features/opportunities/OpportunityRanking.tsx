import type { components } from '../../lib/api/schema'


type OpportunityRow = components['schemas']['OpportunityRowResponse']

function format(value: number): string {
  return new Intl.NumberFormat('en-US').format(value)
}

function label(row: OpportunityRow): string {
  return [row.brand, row.model, row.model_year].filter(Boolean).join(' · ')
}

export function OpportunityRanking({ rows, selectedGroup, onSelect }: {
  rows: OpportunityRow[]
  selectedGroup: string | null
  onSelect: (groupId: string) => void
}) {
  return (
    <section aria-labelledby="ranking-title" className="opportunity-ranking">
      <div className="ranking-heading">
        <div><p className="eyebrow">Ranked decision set</p><h2 id="ranking-title">Replacement opportunities</h2></div>
        <p>Score = 80% relative demand + 20% production readiness</p>
      </div>
      <ol className="opportunity-list">
        {rows.map((row, index) => {
          const descriptionId = `score-${row.group_id}`
          return (
            <li className={selectedGroup === row.group_id ? 'opportunity-card opportunity-card--selected' : 'opportunity-card'} key={row.group_id}>
              <span className="opportunity-rank" aria-label={`Rank ${index + 1}`}>{index + 1}</span>
              <div className="opportunity-card__body">
                <div className="opportunity-card__heading">
                  <div><h3>{label(row)}</h3><p>{row.contributing_configuration_count} contributing configurations</p></div>
                  <span className={`coverage-status coverage-status--${row.coverage_status}`}>{row.coverage_status.replaceAll('_', ' ')}</span>
                </div>
                <div className="opportunity-demand">
                  <strong className="opportunity-demand__base">{format(row.demand.base_units)} replacements</strong>
                  <span>{format(row.demand.downside_units)}–{format(row.demand.upside_units)} range</span>
                </div>
                <div className="coverage-composition" aria-label="Base demand coverage composition">
                  <span>Exact {format(row.exact_covered_base_units)}</span>
                  <span>Fallback {format(row.fallback_covered_base_units)}</span>
                  <span>Uncovered {format(row.uncovered_base_units)}</span>
                </div>
                <div className="opportunity-score">
                  <strong aria-describedby={descriptionId}>Score {row.score.total_points.toFixed(1)}</strong>
                  <span id={descriptionId}>{row.score.demand_points.toFixed(0)} points from relative demand and {row.score.readiness_points.toFixed(1)} points from production readiness.</span>
                </div>
                <button aria-expanded={selectedGroup === row.group_id} onClick={() => onSelect(row.group_id)} type="button">View {label(row)} details</button>
              </div>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
