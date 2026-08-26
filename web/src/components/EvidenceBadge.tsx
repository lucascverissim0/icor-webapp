interface EvidenceBadgeProps {
  label?: string
}

export function EvidenceBadge({ label = 'Demonstration data' }: EvidenceBadgeProps) {
  return (
    <span className="evidence-badge">
      <span aria-hidden="true" className="evidence-badge__dot" />
      {label}
    </span>
  )
}
