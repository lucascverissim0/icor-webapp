import { ArrowUpRight } from 'lucide-react'

import { EurocodeSummary } from '../eurocodes/EurocodeBreakdown'
import { eurocodeGroupTitle, eurocodeYears, sharedWithText, VEHICLE_LIMIT, type EurocodeGroup } from '../eurocodes/eurocode-text'
import type { components } from '../../lib/api/schema'
import { ordinal } from '../../lib/ordinal'


type OpportunityRow = components['schemas']['OpportunityRowResponse']

function format(value: number): string {
  return new Intl.NumberFormat('en-US').format(value)
}

function label(row: OpportunityRow): string {
  if (row.eurocode) return eurocodeGroupTitle(row.eurocode)
  return [row.brand, row.model, row.model_year].filter(Boolean).join(' · ')
}

/** What a Eurocode row's units are: certain ones, and the ones shared with other codes. */
function EurocodeDemand({ group, baseUnits }: { group: EurocodeGroup; baseUnits: number }) {
  return (
    <>
      <strong className="opportunity-demand__base">{format(baseUnits)} certain replacements</strong>
      {group.shared_units > 0 && (
        <span>up to +{format(group.shared_units)} shared with {sharedWithText(group)}</span>
      )}
    </>
  )
}

function EurocodeVehicles({ group }: { group: EurocodeGroup }) {
  const shown = group.vehicles.slice(0, VEHICLE_LIMIT)
  const hidden = group.vehicles.length - shown.length
  return (
    <div aria-label="Vehicles that may carry this Eurocode" className="eurocode-row-vehicles" role="group">
      <ul>
        {shown.map((vehicle) => (
          <li key={`${vehicle.brand}|${vehicle.model}`}>
            <strong>{vehicle.brand} {vehicle.model}</strong>
            <span> · certain {format(vehicle.certain_units)}{vehicle.shared_units > 0 && ` · +${format(vehicle.shared_units)} shared`}</span>
          </li>
        ))}
      </ul>
      {hidden > 0 && <p className="eurocode-more">+{hidden} more {hidden === 1 ? 'vehicle' : 'vehicles'}</p>}
    </div>
  )
}

/** ICOR's own position on a row: shown for information, never part of the score. */
function IcorPosition({ row }: { row: OpportunityRow }) {
  return (
    <div aria-label="ICOR position, not part of the score" className="icor-position" role="group">
      <p className="icor-position__label">ICOR · not scored</p>
      <div className="coverage-composition" aria-label="Base demand coverage composition">
        <span className={`coverage-status coverage-status--${row.coverage_status}`}>{row.coverage_status.replaceAll('_', ' ')}</span>
        <span>Exact {format(row.exact_covered_base_units)}</span>
        <span>Fallback {format(row.fallback_covered_base_units)}</span>
        <span>Uncovered {format(row.uncovered_base_units)}</span>
      </div>
      {row.icor_worked && <p className="icor-experience">ICOR worked</p>}
    </div>
  )
}

export function OpportunityRanking({ rows, selectedGroup, onSelect, searchedCode = null, firstRank = 1 }: {
  rows: OpportunityRow[]
  selectedGroup: string | null
  onSelect: (groupId: string) => void
  searchedCode?: string | null
  /** The position of the first row in the whole ranking, so later pages keep counting. */
  firstRank?: number
}) {
  return (
    <section aria-labelledby="ranking-title" className="opportunity-ranking">
      <div className="ranking-heading">
        <div><p className="eyebrow">Ranked decision set</p><h2 id="ranking-title">Replacement opportunities</h2></div>
        <p>Score = whole-market demand percentile × 100</p>
      </div>
      <ol className="opportunity-list" start={firstRank}>
        {rows.map((row, index) => {
          const descriptionId = `score-${row.group_id}`
          return (
            <li className={selectedGroup === row.group_id ? 'opportunity-card opportunity-card--selected' : 'opportunity-card'} key={row.group_id}>
              <span className="opportunity-rank" aria-label={`Rank ${firstRank + index}`}>{firstRank + index}</span>
              <div className="opportunity-card__body">
                <div className="opportunity-card__heading">
                  <div>
                    {row.eurocodes && !row.eurocode && <EurocodeSummary breakdown={row.eurocodes} searchedCode={searchedCode} />}
                    {row.eurocode
                      ? <h3><code>{row.eurocode.code}</code> · {row.eurocode.label} · {eurocodeYears(row.eurocode)}</h3>
                      : <h3>{label(row)}</h3>}
                    {!row.eurocode && <p className="generation-name">{row.generation_name ?? 'Generation review pending'}</p>}
                    <p>{row.contributing_configuration_count} contributing configurations</p>
                    {row.eurocode && <EurocodeVehicles group={row.eurocode} />}
                  </div>
                </div>
                <div className="opportunity-demand">
                  {row.eurocode
                    ? <EurocodeDemand baseUnits={row.demand.base_units} group={row.eurocode} />
                    : <strong className="opportunity-demand__base">{format(row.demand.base_units)} replacements</strong>}
                  <span>{format(row.demand.downside_units)}–{format(row.demand.upside_units)} range</span>
                </div>
                <div className="opportunity-score">
                  <strong aria-describedby={descriptionId}>Score {row.score.total_points.toFixed(1)}</strong>
                  <span id={descriptionId}>
                    {ordinal(Math.round(row.score.demand_percentile * 100))} demand percentile
                    {row.score.demand_rank !== null && row.score.demand_population > 0
                      ? ` · #${format(row.score.demand_rank)} of ${format(row.score.demand_population)} in the market`
                      : ' · no forecast demand to rank'}
                  </span>
                </div>
                <IcorPosition row={row} />
                <button
                  aria-expanded={selectedGroup === row.group_id}
                  aria-label={`View ${label(row)} details`}
                  className="opportunity-card__details"
                  onClick={() => onSelect(row.group_id)}
                  type="button"
                >
                  <span>View details</span>
                  <ArrowUpRight aria-hidden="true" size={16} />
                </button>
              </div>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
