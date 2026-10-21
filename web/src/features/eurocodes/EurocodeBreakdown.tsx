import {
  CHIP_LIMIT,
  eurocodeYears,
  format,
  sharedWithText,
  unassignedParts,
  type EurocodeBreakdownData,
  type EurocodeShare,
} from './eurocode-text'

/** "Not assigned to a Eurocode: Z", with each reason that applies. */
export function EurocodeUnassigned({ breakdown }: { breakdown: EurocodeBreakdownData }) {
  const { total, reasons } = unassignedParts(breakdown)
  if (total === 0) return null
  return (
    <p className="eurocode-unassigned">
      <strong>Not assigned to a Eurocode: {format(total)}</strong>
      <span> ({reasons.map((part) => `${part.reason}: ${format(part.units)}`).join('; ')})</span>
    </p>
  )
}

function ShareText({ share }: { share: EurocodeShare }) {
  return (
    <span className="eurocode-share">
      {share.certain_units > 0 || share.shared_units === 0
        ? <>
            <span>certain {format(share.certain_units)}</span>
            {share.shared_units > 0 && (
              <span> · up to +{format(share.shared_units)} shared with {sharedWithText(share)}</span>
            )}
          </>
        : <span>up to {format(share.shared_units)}, shared with {sharedWithText(share)}</span>}
    </span>
  )
}

/** Compact chips for a ranking card: the first codes, then the units no code explains.
 *
 * A searched code leads, so a row listed because it may carry the code always shows it.
 */
export function EurocodeSummary({ breakdown, searchedCode = null }: {
  breakdown: EurocodeBreakdownData
  searchedCode?: string | null
}) {
  const ordered = [
    ...breakdown.codes.filter((share) => share.code === searchedCode),
    ...breakdown.codes.filter((share) => share.code !== searchedCode),
  ]
  const shown = ordered.slice(0, CHIP_LIMIT)
  const hidden = breakdown.codes.length - shown.length
  return (
    <div aria-label="Eurocodes" className="eurocode-summary" role="group">
      {breakdown.codes.length === 0 && <p className="eurocode-none">No Eurocode assigned</p>}
      {shown.length > 0 && (
        <ul className="eurocode-chips">
          {shown.map((share) => (
            <li className={share.code === searchedCode ? 'eurocode-chip eurocode-chip--searched' : 'eurocode-chip'} key={share.code}>
              <code>{share.code}</code>
              {share.code === searchedCode && <span className="eurocode-chip__badge">Searched code</span>}
              <span className="eurocode-chip__label">{share.label} · {eurocodeYears(share)}</span>
              <ShareText share={share} />
            </li>
          ))}
        </ul>
      )}
      {hidden > 0 && <p className="eurocode-more">+{hidden} more {hidden === 1 ? 'code' : 'codes'}</p>}
      <EurocodeUnassigned breakdown={breakdown} />
    </div>
  )
}

/** One line per code for a forecast market, counting registrations of the selection. */
export function EurocodeCompact({ breakdown }: { breakdown: EurocodeBreakdownData }) {
  return (
    <div className="eurocode-compact">
      {breakdown.codes.length === 0 && <p className="eurocode-none">No Eurocode assigned</p>}
      {breakdown.codes.length > 0 && (
        <ul>
          {breakdown.codes.map((share) => (
            <li key={share.code}><code>{share.code}</code> <ShareText share={share} /></li>
          ))}
        </ul>
      )}
      <EurocodeUnassigned breakdown={breakdown} />
    </div>
  )
}

/** The full table for a detail page. */
export function EurocodeTable({ breakdown }: { breakdown: EurocodeBreakdownData }) {
  const { reasons } = unassignedParts(breakdown)
  return (
    <>
      {breakdown.codes.length === 0 && <p className="eurocode-none">No Eurocode assigned</p>}
      {(breakdown.codes.length > 0 || reasons.length > 0) && (
        <div aria-label="Eurocodes of this vehicle" className="eurocode-table-wrap" role="region" tabIndex={0}>
          <table className="eurocode-table">
            <thead>
              <tr>
                <th scope="col">Code</th><th scope="col">Vehicle</th><th scope="col">Registration years</th>
                <th scope="col">Certain</th><th scope="col">Up to (shared)</th><th scope="col">Shared with</th>
              </tr>
            </thead>
            <tbody>
              {breakdown.codes.map((share) => (
                <tr key={share.code}>
                  <th scope="row"><code>{share.code}</code></th>
                  <td>{share.label}</td>
                  <td>{eurocodeYears(share)}</td>
                  <td>{format(share.certain_units)}</td>
                  <td>{share.shared_units > 0 ? `+${format(share.shared_units)}` : '—'}</td>
                  <td>{share.shared_units > 0 ? sharedWithText(share) : '—'}</td>
                </tr>
              ))}
              {reasons.map((part) => (
                <tr className="eurocode-table__unassigned" key={part.key}>
                  <th colSpan={3} scope="row">Not assigned: {part.reason}</th>
                  <td>{format(part.units)}</td>
                  <td colSpan={2}>—</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

/** The single calm notice shown when no Eurocode map is loaded. */
export function EurocodeNotLoaded({ warnings = [] }: { warnings?: string[] }) {
  return (
    <div className="eurocode-notice" role="note">
      <p>Eurocodes are not loaded, so vehicles are shown without them.</p>
      {warnings.map((warning) => <p key={warning}>{warning}</p>)}
    </div>
  )
}
