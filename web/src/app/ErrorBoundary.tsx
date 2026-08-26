import type { ErrorComponentProps } from '@tanstack/react-router'

export function RouteErrorFallback({ reset }: ErrorComponentProps) {
  return (
    <section aria-labelledby="route-error-title" className="route-state">
      <p className="eyebrow">Planner unavailable</p>
      <h2 id="route-error-title">This view could not be opened</h2>
      <p>Your filters are still in the address bar. Retry the view when you are ready.</p>
      <button className="primary-action" onClick={reset} type="button">
        Retry view
      </button>
    </section>
  )
}
