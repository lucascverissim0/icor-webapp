import { useQuery } from '@tanstack/react-query'
import { useRouter } from '@tanstack/react-router'

import { configurationRoute } from '../../app/router'
import { ApiProblem, PlannerApiClient, plannerApi } from '../../lib/api/client'
import type { components } from '../../lib/api/schema'
import { serializePlannerSearch } from '../../lib/planner-search'

type Configuration = components['schemas']['PlanningConfigurationResponse']

interface ConfigurationDetailBaseProps {
  apiClient?: PlannerApiClient
  configurationId: string
}

type ConfigurationDetailProps = ConfigurationDetailBaseProps & (
  | { backHref: string; onBack?: never }
  | { backHref?: never; onBack: () => void }
)

function formatUnits(value: number): string {
  return `${new Intl.NumberFormat('en-US').format(value)} units`
}

function titleCase(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1)
}

function factValue(value: boolean | null): string {
  if (value === null) return 'Unknown'
  return value ? 'Yes' : 'Not fitted'
}

function ReturnAction({ backHref, label, onBack, primary = false }: {
  backHref?: string
  label: string
  onBack?: () => void
  primary?: boolean
}) {
  const className = primary ? 'primary-action' : 'back-link'
  return backHref
    ? <a className={className} href={backHref}>{label}</a>
    : <button className={className} onClick={onBack} type="button">{label}</button>
}

function EquipmentFacts({ equipment }: { equipment: Configuration['equipment'] }) {
  const facts = [
    ['Camera / ADAS', equipment.camera_adas],
    ['Head-up display', equipment.hud],
    ['Heated glass', equipment.heated],
    ['Acoustic glass', equipment.acoustic],
    ['Rain / light sensor', equipment.rain_light_sensor],
  ] as const
  return (
    <section className="detail-section" aria-labelledby="equipment-heading">
      <h2 id="equipment-heading">Equipment compatibility</h2>
      <dl className="fact-grid">
        {facts.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{factValue(value)}{value === false && <span className="equipment-alias">No</span>}</dd></div>)}
      </dl>
    </section>
  )
}

function ConfidenceCard({ label, confidence }: {
  label: string
  confidence: Configuration['identity_confidence']
}) {
  return (
    <div className="confidence-card">
      <dt>{label}</dt>
      <dd><span className="status-pill">{titleCase(confidence.level)}</span><p>{confidence.reason}</p></dd>
    </div>
  )
}

function DetailContent({ configuration, backHref, onBack }: { configuration: Configuration; backHref?: string; onBack?: () => void }) {
  const updated = new Intl.DateTimeFormat('en-GB', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'UTC' }).format(new Date(configuration.updated_at))
  return (
    <article className="configuration-detail">
      <ReturnAction backHref={backHref} label="Back to planner" onBack={onBack} />
      <header className="detail-hero">
        <div>
          <p className="eyebrow">Configuration detail</p>
          <h1>{configuration.brand} {configuration.model}</h1>
          <p>{configuration.market} · Model years {configuration.model_year_start}–{configuration.model_year_end} · {configuration.generation}</p>
        </div>
        <span className="status-pill">{titleCase(configuration.evidence_status)} evidence</span>
      </header>

      <dl className="identity-strip">
        <div><dt>SKU</dt><dd>{configuration.sku ?? 'Unknown'}</dd></div>
        <div><dt>Part family</dt><dd>{configuration.part_family ?? 'Unknown'}</dd></div>
        <div><dt>Body style</dt><dd>{configuration.body_style}</dd></div>
        <div><dt>Drive side</dt><dd>{configuration.drive_side ? titleCase(configuration.drive_side) : 'Unknown'}</dd></div>
        <div><dt>Facelift</dt><dd>{configuration.facelift ?? 'Not specified'}</dd></div>
      </dl>

      <section className="detail-section" aria-labelledby="demand-heading">
        <h2 id="demand-heading">Demand composition</h2>
        <div className="demand-highlight">
          <div><span>Base demand</span><strong>{formatUnits(configuration.demand.base_units)}</strong></div>
          <div><span>Downside–upside</span><strong>{new Intl.NumberFormat('en-US').format(configuration.demand.downside_units)}–{formatUnits(configuration.demand.upside_units)}</strong></div>
        </div>
        <dl className="fact-grid assumptions">
          <div><dt>Vehicle exposure</dt><dd><span>{new Intl.NumberFormat('en-US').format(configuration.vehicle_exposure_units)} vehicles</span><span className="exposure-alias">{new Intl.NumberFormat('en-US').format(configuration.vehicle_exposure_units)} units</span></dd></div>
          <div><dt>Replacement-rate assumption</dt><dd>{new Intl.NumberFormat('en-US', { style: 'percent', maximumFractionDigits: 2 }).format(configuration.replacement_rate)}</dd></div>
          <div><dt>Forecast horizon</dt><dd>{configuration.forecast_horizon}</dd></div>
        </dl>
        <p className="detail-note">These values are synthetic demonstration assumptions and are not calibrated production forecasts.</p>
      </section>

      <EquipmentFacts equipment={configuration.equipment} />

      <section className="detail-section" aria-labelledby="confidence-heading">
        <h2 id="confidence-heading">Confidence and evidence</h2>
        <dl className="confidence-grid">
          <ConfidenceCard label="Identity confidence" confidence={configuration.identity_confidence} />
          <ConfidenceCard label="Data-quality confidence" confidence={configuration.data_quality_confidence} />
        </dl>
      </section>

      <section className="detail-section" aria-labelledby="sources-heading">
        <h2 id="sources-heading">Synthetic demonstration sources</h2>
        <div className="source-list">
          {configuration.sources.map((source) => <article key={`${source.name}-${source.description}`}><h3>{source.name}</h3><p>{source.description}</p><span>Synthetic demonstration evidence</span></article>)}
        </div>
      </section>

      <footer className="detail-metadata">
        <span>Data version <strong>{configuration.data_version}</strong></span>
        <span>Updated {updated} UTC</span>
      </footer>
    </article>
  )
}

export function ConfigurationDetail({ apiClient = plannerApi, backHref, configurationId, onBack }: ConfigurationDetailProps) {
  const detailQuery = useQuery({
    queryKey: ['planner', 'configuration', configurationId],
    queryFn: () => apiClient.configuration(configurationId),
  })

  if (detailQuery.isPending) return <section className="planner-state" aria-busy="true"><h1>Loading configuration detail…</h1></section>
  if (detailQuery.isError) {
    const notFound = detailQuery.error instanceof ApiProblem && detailQuery.error.status === 404
    if (notFound) return <section className="planner-state"><p className="eyebrow">No matching configuration</p><h1>Configuration not found</h1><p>The requested identifier is unavailable in this demonstration dataset.</p><ReturnAction backHref={backHref} label="Return to planner" onBack={onBack} primary /></section>
    return <section className="planner-state" role="alert"><p className="eyebrow">Detail unavailable</p><h1>This configuration could not be loaded</h1><p>{detailQuery.error.message}</p><button className="primary-action" onClick={() => void detailQuery.refetch()} type="button">Retry</button><ReturnAction backHref={backHref} label="Return to planner" onBack={onBack} /></section>
  }
  return <DetailContent backHref={backHref} configuration={detailQuery.data} onBack={onBack} />
}

export function ConfigurationDetailPage() {
  const router = useRouter()
  const { configurationId } = configurationRoute.useParams()
  const search = serializePlannerSearch(configurationRoute.useSearch())
  const backHref = router.buildLocation({ to: '/planner', search }).href
  return <ConfigurationDetail backHref={backHref} configurationId={configurationId} />
}
