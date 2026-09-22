import { useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import { ApiProblem, PlannerApiClient, plannerApi } from '../../lib/api/client'
import type { components } from '../../lib/api/schema'

type VehicleForecast = components['schemas']['VehicleForecastResponse']
type VehicleOption = components['schemas']['VehicleOptionResponse']

function normalized(value: string): string {
  return value.trim().toLocaleLowerCase().replace(/\s+/g, ' ')
}

function directMatch(vehicles: VehicleOption[], search: string): VehicleOption | undefined {
  const query = normalized(search)
  if (!query) return undefined
  return vehicles.find((item) => normalized(`${item.brand} ${item.model}`) === query)
    ?? vehicles.find((item) => normalized(item.model) === query)
}

function units(value: number): string {
  return new Intl.NumberFormat('en-US').format(value)
}

function ForecastResults({ result, clientRelease }: { result: VehicleForecast; clientRelease: boolean }) {
  return (
    <section className="vehicle-forecast-results" aria-labelledby="vehicle-forecast-title">
      <header className="results-heading">
        <div>
          <p className="eyebrow">{result.horizon} windshield replacement forecast</p>
          <h2 id="vehicle-forecast-title">{clientRelease
            ? `${result.brand} ${result.model} · registration year ${result.selected_year}`
            : `${result.brand} ${result.model} · ${result.generation_name}`}</h2>
          <p>{clientRelease
            ? `The official-source ${result.selected_year} registration cohort is included. Fleet decay has already been applied before the windshield replacement hazard.`
            : `Cohorts ${result.included_cohort_years.join(', ')} are included. Fleet decay has already been applied before the windshield replacement hazard.`}</p>
        </div>
        <span className="status-pill">{clientRelease ? 'Official registration cohort' : `${result.generation_confidence} generation confidence`}</span>
      </header>
      <p className="forecast-boundary"><strong>Only registration cohorts through the latest observed year are counted.</strong> Historical gap estimates remain explicit; future sales cohorts {result.excluded_forecast_cohort_years.join(', ') || 'none'} are not added to the circulating fleet.</p>
      <div className="forecast-method-note">
        <strong>Calculation chain</strong>
        <span>Registration cohorts → surviving fleet at {result.horizon} → age/geography hazard → P10/P50/P90 replacement forecast.</span>
        <span>This remains assumption-led until proprietary fitment and replacement-history calibration are available.</span>
        <span>Methods used for these figures: survival <code>{result.survival_method}</code>, hazard <code>{result.hazard_method}</code>, uncertainty <code>{result.uncertainty_method}</code>.</span>
      </div>
      {result.survival_method.startsWith('mixed:') && <p className="integrity-warning">These cohorts were not all built with the same survival curve ({result.survival_method}). Treat the fleet figures as provisional.</p>}
      {result.excluded_ambiguous_years.length > 0 && <p className="integrity-warning">Excluded ambiguous transition years: {result.excluded_ambiguous_years.join(', ')}.</p>}
      <div aria-label="Forecast results by market" className="vehicle-forecast-table-wrap" role="region" tabIndex={0}>
        <table className="vehicle-forecast-table">
          <thead><tr><th>Market</th><th>Registration cohorts</th><th>Surviving fleet P50</th><th>Replacements P50</th><th>Replacement range P10–P90</th></tr></thead>
          <tbody>{result.markets.map((row) => (
            <tr key={row.code}>
              <th scope="row">{row.name}</th>
              {row.availability === 'available' && row.active_fleet && row.replacements ? <>
                <td>{units(row.registration_cohort_units ?? 0)}</td>
                <td>{units(row.active_fleet.base_units)}</td>
                <td><strong>{units(row.replacements.base_units)}</strong></td>
                <td>{units(row.replacements.downside_units)}–{units(row.replacements.upside_units)}</td>
              </> : <td colSpan={4}>Unavailable — not zero</td>}
            </tr>
          ))}</tbody>
        </table>
      </div>
    </section>
  )
}

export function VehicleForecastSearch({
  apiClient = plannerApi,
  clientRelease = import.meta.env.VITE_ICOR_CLIENT_RELEASE === 'verified',
}: {
  apiClient?: PlannerApiClient
  clientRelease?: boolean
}) {
  const [input, setInput] = useState('')
  const [search, setSearch] = useState('')
  const [brandInput, setBrandInput] = useState<string | null>(null)
  const [modelInput, setModelInput] = useState<string | null>(null)
  const [selectionMode, setSelectionMode] = useState<'year' | 'generation'>('year')
  const [year, setYear] = useState('')
  const [generation, setGeneration] = useState('')
  const [horizon, setHorizon] = useState('')
  const options = useQuery({
    queryKey: ['vehicle-forecasts', 'options', search],
    queryFn: () => apiClient.vehicleForecastOptions({ search }),
  })
  const matchedVehicle = useMemo(
    () => directMatch(options.data?.vehicles ?? [], search),
    [options.data, search],
  )
  const brand = brandInput ?? matchedVehicle?.brand ?? ''
  const model = modelInput ?? (brandInput === null ? matchedVehicle?.model : '') ?? ''
  const modelOptions = useQuery({
    queryKey: ['vehicle-forecasts', 'models', brand],
    queryFn: () => apiClient.vehicleForecastOptions({ brand }),
    enabled: Boolean(brand),
  })
  const selectionOptions = useQuery({
    queryKey: ['vehicle-forecasts', 'selection', brand, model],
    queryFn: () => apiClient.vehicleForecastOptions({ brand, model }),
    enabled: Boolean(brand && model),
  })
  const forecast = useMutation({
    mutationFn: () => apiClient.vehicleForecast({
      brand, model, horizon: Number(horizon || selectionOptions.data?.horizons[0]),
      ...(selectionMode === 'year' ? { year: Number(year) } : { generation }),
    }),
  })
  const brands = useMemo(
    () => options.data?.brands?.length ? options.data.brands : [...new Set((options.data?.vehicles ?? []).map((item) => item.brand))],
    [options.data],
  )
  const models = useMemo(
    () => [...new Set((modelOptions.data?.vehicles ?? []).map((item) => item.model))],
    [modelOptions.data],
  )

  const selectedHorizon = horizon || selectionOptions.data?.horizons[0]?.toString() || ''
  const canCalculate = Boolean(brand && model && selectedHorizon && (selectionMode === 'year' ? year : generation))
  const error = forecast.error instanceof ApiProblem ? forecast.error.message : forecast.error?.message

  return (
    <div className="vehicle-search-page">
      <header className="opportunities-hero">
        <div><p className="eyebrow">Vehicle forecast search</p><h2>{clientRelease ? 'Search by vehicle and registration year' : 'Search by model year or generation'}</h2><p>{clientRelease ? 'Choose an official-source make/model label and registration year, then forecast that cohort across the requested European markets.' : 'Choose one vehicle identity, then forecast every surviving cohort assigned to that generation across the requested European markets.'}</p></div>
        <span className="status-pill">Cohort-based</span>
      </header>
      <section className="vehicle-search-panel" aria-labelledby="vehicle-search-heading">
        <h2 id="vehicle-search-heading">1. Find a vehicle</h2>
        <form className="vehicle-search-bar" onSubmit={(event) => {
          event.preventDefault()
          const value = input.trim()
          setBrandInput(null); setModelInput(null); setYear(''); setGeneration(''); setHorizon(''); forecast.reset(); setSearch(value)
        }}>
          <label htmlFor="vehicle-search">Search brand or model</label>
          <div><input id="vehicle-search" type="search" value={input} onChange={(event) => setInput(event.target.value)} /><button className="primary-action" type="submit">Search vehicles</button></div>
        </form>
        {options.isError && <p role="alert">Could not load vehicle matches.</p>}
        {modelOptions.isError && <p role="alert">Could not load models for the selected brand.</p>}
        {search && options.data && <p className="vehicle-search-feedback" role="status">{options.data.vehicles.length.toLocaleString('en-US')} forecastable matches. {directMatch(options.data.vehicles, search) ? 'Exact model selected.' : 'Type or choose the exact brand and model below.'}</p>}
        <div className="vehicle-select-grid">
          <label>Brand<select aria-label="Brand" disabled={options.isFetching} value={brand} onChange={(event) => { setBrandInput(event.target.value); setModelInput(''); setYear(''); setGeneration(''); setHorizon(''); forecast.reset() }}><option value="">{options.isFetching ? 'Loading brands…' : 'Select brand'}</option>{brands.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
          <label>Model<select aria-label="Model" disabled={!brand || modelOptions.isFetching} value={model} onChange={(event) => { setModelInput(event.target.value); setYear(''); setGeneration(''); setHorizon(''); forecast.reset() }}><option value="">{modelOptions.isFetching ? 'Loading models…' : 'Select model'}</option>{models.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
        </div>
      </section>
      {brand && model && <section className="vehicle-search-panel" aria-labelledby="forecast-selection-heading">
        <h2 id="forecast-selection-heading">{clientRelease ? '2. Choose registration year' : '2. Choose year or generation'}</h2>
        {!clientRelease && <fieldset className="selection-mode"><legend>Selection method</legend><label><input checked={selectionMode === 'year'} name="selection-mode" onChange={() => { setSelectionMode('year'); setGeneration('') }} type="radio" /> Model year</label><label><input checked={selectionMode === 'generation'} name="selection-mode" onChange={() => { setSelectionMode('generation'); setYear('') }} type="radio" /> Generation directly</label></fieldset>}
        <div className="vehicle-select-grid">
          {selectionMode === 'year' ? <label>{clientRelease ? 'Registration year' : 'Model year'}<select aria-label={clientRelease ? 'Registration year' : 'Model year'} value={year} onChange={(event) => setYear(event.target.value)}><option value="">Select year</option>{selectionOptions.data?.years.map((item) => <option key={item} value={item}>{item}</option>)}</select></label> : <label>Generation<select aria-label="Generation" value={generation} onChange={(event) => setGeneration(event.target.value)}><option value="">Select generation</option>{selectionOptions.data?.generations.map((item) => <option key={item.key} value={item.key}>{item.name} · {item.confidence} confidence</option>)}</select></label>}
          <label>Forecast year<select aria-label="Forecast year" value={selectedHorizon} onChange={(event) => setHorizon(event.target.value)}>{selectionOptions.data?.horizons.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
        </div>
        {!selectionOptions.isFetching && selectionOptions.data && selectionOptions.data.horizons.length === 0 && <p className="mutation-error" role="alert">No forecastable vehicle matches this exact brand and model. Check the spelling or choose a suggestion.</p>}
        <button className="primary-action" disabled={!canCalculate || forecast.isPending} onClick={() => forecast.mutate()} type="button">{forecast.isPending ? 'Calculating…' : 'Calculate forecast'}</button>
        {error && <p className="mutation-error" role="alert">{error}</p>}
      </section>}
      {forecast.data && <ForecastResults clientRelease={clientRelease} result={forecast.data} />}
    </div>
  )
}
