import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { queryKeys } from '../../app/query-client'
import { ApiProblem, PlannerApiClient, plannerApi, type OpportunitiesQuery } from '../../lib/api/client'
import type { components } from '../../lib/api/schema'


type Coverage = components['schemas']['ProductionCoverageResponse']
type CoveragePayload = components['schemas']['ProductionCoverageRequest']

interface CoverageManagerProps {
  apiClient?: PlannerApiClient
  opportunityQuery: OpportunitiesQuery
}

function years(start: number, end: number): number[] {
  return Array.from({ length: end - start + 1 }, (_value, index) => start + index)
}

export function CoverageManager({ apiClient = plannerApi, opportunityQuery }: CoverageManagerProps) {
  const queryClient = useQueryClient()
  const [brand, setBrand] = useState('')
  const [model, setModel] = useState('')
  const [modelYear, setModelYear] = useState('')
  const [configurationId, setConfigurationId] = useState('')
  const [fallback, setFallback] = useState(false)
  const [fallbackConfirmed, setFallbackConfirmed] = useState(false)
  const [note, setNote] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [deleteId, setDeleteId] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)

  const coverageQuery = useQuery({ queryKey: queryKeys.coverage, queryFn: () => apiClient.coverage() })
  const configurationsQuery = useQuery({
    queryKey: ['planner', 'configurations', 'coverage-options'],
    queryFn: () => apiClient.configurations({ pageSize: 100 }),
  })
  const configurations = useMemo(
    () => configurationsQuery.data?.items ?? [],
    [configurationsQuery.data?.items],
  )
  const brands = useMemo(() => [...new Set(configurations.map((row) => row.brand))].sort(), [configurations])
  const models = useMemo(() => [...new Set(configurations.filter((row) => row.brand === brand).map((row) => row.model))].sort(), [brand, configurations])
  const modelYears = useMemo(() => [...new Set(configurations.filter((row) => row.brand === brand && row.model === model).flatMap((row) => years(row.model_year_start, row.model_year_end)))].sort(), [brand, configurations, model])
  const exactConfigurations = configurations.filter((row) => row.brand === brand && row.model === model && modelYear !== '' && row.model_year_start <= Number(modelYear) && row.model_year_end >= Number(modelYear))

  function clearForm() {
    setBrand(''); setModel(''); setModelYear(''); setConfigurationId('')
    setFallback(false); setFallbackConfirmed(false); setNote(''); setEditingId(null)
  }

  async function refreshConfirmedState() {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: queryKeys.coverage }),
      queryClient.invalidateQueries({ queryKey: queryKeys.opportunities(opportunityQuery) }),
    ])
  }

  const save = useMutation({
    mutationFn: (payload: CoveragePayload) => editingId
      ? apiClient.updateCoverage(editingId, payload)
      : apiClient.createCoverage(payload),
    onSuccess: async () => {
      await refreshConfirmedState()
      setStatus(editingId ? 'Production coverage updated.' : 'Production coverage saved.')
      clearForm()
    },
  })
  const remove = useMutation({
    mutationFn: (coverageId: string) => apiClient.deleteCoverage(coverageId),
    onSuccess: async () => {
      await refreshConfirmedState()
      setStatus('Production coverage deleted.')
      setDeleteId(null)
    },
  })

  function submit(event: React.FormEvent) {
    event.preventDefault()
    setStatus(null)
    const payload: CoveragePayload = fallback
      ? { match_type: 'vehicle_year_fallback', configuration_id: null, brand, model, model_year: Number(modelYear), note: note || null }
      : { match_type: 'exact_configuration', configuration_id: configurationId, brand: null, model: null, model_year: Number(modelYear), note: note || null }
    save.mutate(payload)
  }

  function edit(row: Coverage) {
    setEditingId(row.coverage_id); setBrand(row.brand); setModel(row.model)
    setModelYear(String(row.model_year)); setConfigurationId(row.configuration_id ?? '')
    setFallback(row.match_type === 'vehicle_year_fallback')
    setFallbackConfirmed(row.match_type === 'vehicle_year_fallback')
    setNote(row.note ?? ''); setStatus(null)
  }

  const canSubmit = Boolean(brand && model && modelYear && (fallback ? fallbackConfirmed : configurationId))
  const mutationError = save.error ?? remove.error

  return (
    <section aria-labelledby="coverage-title" className="coverage-manager">
      <div className="coverage-manager__heading"><div><p className="eyebrow">Shared local prototype state</p><h2 id="coverage-title">Manage ICOR production</h2><p>Record only non-secret production coverage. This local database has no user accounts or audit attribution.</p></div></div>
      <div className="coverage-manager__layout">
        <form onSubmit={submit}>
          <label>Brand<select aria-label="Brand" onChange={(event) => { setBrand(event.target.value); setModel(''); setModelYear(''); setConfigurationId('') }} value={brand}><option value="">Select brand</option>{brands.map((value) => <option key={value}>{value}</option>)}</select></label>
          <label>Model<select aria-label="Model" disabled={!brand} onChange={(event) => { setModel(event.target.value); setModelYear(''); setConfigurationId('') }} value={model}><option value="">Select model</option>{models.map((value) => <option key={value}>{value}</option>)}</select></label>
          <label>Model year<select aria-label="Model year" disabled={!model} onChange={(event) => { setModelYear(event.target.value); setConfigurationId('') }} value={modelYear}><option value="">Select model year</option>{modelYears.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
          <label className="fallback-toggle"><input checked={fallback} onChange={(event) => { setFallback(event.target.checked); setFallbackConfirmed(false); setConfigurationId('') }} type="checkbox" />Exact configuration unknown</label>
          {!fallback && <label>Exact configuration / SKU<select aria-label="Exact configuration / SKU" disabled={!modelYear} onChange={(event) => setConfigurationId(event.target.value)} value={configurationId}><option value="">Select configuration</option>{exactConfigurations.map((row) => <option key={row.configuration_id} value={row.configuration_id}>{row.sku ?? 'SKU unknown'} · {row.generation}</option>)}</select></label>}
          {fallback && <div className="fallback-warning"><p><strong>Lower-precision match.</strong> This covers every matching configuration for the selected vehicle year at half readiness weight.</p><label><input checked={fallbackConfirmed} onChange={(event) => setFallbackConfirmed(event.target.checked)} type="checkbox" />I understand this is lower precision than exact configuration coverage.</label></div>}
          <label>Planner note<textarea maxLength={500} onChange={(event) => setNote(event.target.value)} value={note} /></label>
          <div className="coverage-form__actions"><button className="primary-action" disabled={!canSubmit || save.isPending} type="submit">{editingId ? 'Update coverage' : fallback ? 'Save fallback coverage' : 'Save exact coverage'}</button>{editingId && <button onClick={clearForm} type="button">Cancel edit</button>}</div>
          {mutationError && <p className="mutation-error" role="alert">{mutationError.message}{mutationError instanceof ApiProblem && mutationError.correlationId ? ` Reference: ${mutationError.correlationId}` : ''}</p>}
          {status && <p className="mutation-success" role="status">{status}</p>}
        </form>

        <div className="coverage-list" aria-live="polite">
          <h3>Recorded coverage</h3>
          {(coverageQuery.isPending || configurationsQuery.isPending) && <p aria-busy="true">Loading production coverage…</p>}
          {coverageQuery.isError && <p role="alert">{coverageQuery.error.message}</p>}
          {coverageQuery.data?.length === 0 && <p>No production coverage has been recorded yet.</p>}
          {coverageQuery.data?.map((row) => (
            <article key={row.coverage_id}>
              <div><strong>{row.brand} {row.model} · {row.model_year}</strong><span>{row.sku ?? 'Vehicle-year fallback'} · {row.match_type.replaceAll('_', ' ')}</span></div>
              {row.note && <p>{row.note}</p>}
              <div className="coverage-list__actions"><button onClick={() => edit(row)} type="button">Edit {row.brand} {row.model}</button><button onClick={() => setDeleteId(row.coverage_id)} type="button">Delete {row.brand} {row.model}</button></div>
              {deleteId === row.coverage_id && <div className="delete-confirmation" role="alert"><p>Delete this coverage record? The ranking refreshes only after deletion is confirmed.</p><button disabled={remove.isPending} onClick={() => remove.mutate(row.coverage_id)} type="button">Confirm delete {row.brand} {row.model}</button><button onClick={() => setDeleteId(null)} type="button">Keep coverage</button></div>}
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}
