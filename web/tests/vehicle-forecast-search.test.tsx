import { QueryClient } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { AppProviders } from '../src/app/providers'
import { VehicleForecastSearch } from '../src/features/planner/VehicleForecastSearch'
import { PlannerApiClient } from '../src/lib/api/client'


function json(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
}

const searchOptions = {
  vehicles: [
    { brand: 'Volkswagen', model: 'Golf' },
    { brand: 'Volkswagen', model: 'Golf Plus' },
  ],
  years: [], generations: [], horizons: [2028, 2031],
}
const selectedOptions = {
  vehicles: searchOptions.vehicles,
  years: [2018, 2020, 2021],
  generations: [{
    key: 'volkswagen-golf-mk8-europe', name: 'Golf Mk8', start_year: 2019,
    end_year: null, basis: 'manufacturer_generation_window', confidence: 'high',
  }],
  horizons: [2028, 2031],
}
const forecast = {
  brand: 'Volkswagen', model: 'Golf', selected_year: 2020,
  generation_key: 'volkswagen-golf-mk8-europe', generation_name: 'Golf Mk8',
  generation_basis: 'manufacturer_generation_window', generation_confidence: 'high',
  generation_source_url: 'https://example.test/golf', generation_start_year: 2019,
  generation_end_year: null, horizon: 2028, included_cohort_years: [2020, 2021],
  excluded_ambiguous_years: [2019],
  excluded_forecast_cohort_years: [2026, 2027, 2028],
  markets: [
    { code: 'EU27', name: 'Europe (EU27)', availability: 'available', registration_cohort_units: 1000, cohort_count: 2, active_fleet: { downside_units: 700, base_units: 800, upside_units: 900 }, replacements: { downside_units: 24, base_units: 32, upside_units: 41 } },
    { code: 'GB', name: 'United Kingdom (GB; England is not separable)', availability: 'unavailable', registration_cohort_units: null, cohort_count: 0, active_fleet: null, replacements: null },
  ],
  survival_method: 'constant-annual-retention-v1', hazard_method: 'age-band-geography-hazard-v1',
  uncertainty_method: 'seeded-triangular-propagation-v1',
  calibration_status: 'assumption_led_without_proprietary_fitment_or_hazard_calibration',
  data_version: 'snapshot-test',
}

describe('VehicleForecastSearch', () => {
  it('limits the client release to truthful source model-year selection', async () => {
    const user = userEvent.setup()
    const modelYearForecast = {
      ...forecast,
      generation_key: 'source-registration-year:Volkswagen:Golf:2020',
      generation_name: 'Volkswagen Golf — 2020 registration cohort',
      generation_basis: 'official_source_registration_cohort',
      generation_confidence: 'source-reported',
      generation_source_url: null,
      generation_start_year: 2020,
      generation_end_year: 2020,
      included_cohort_years: [2020],
      excluded_ambiguous_years: [],
    }
    const modelYearOptions = { ...selectedOptions, generations: [] }
    const fetcher = vi.fn<typeof fetch>().mockImplementation((input) => {
      const url = typeof input === 'string'
        ? input
        : input instanceof URL ? input.href : input.url
      if (url.includes('/vehicle-forecasts?')) return Promise.resolve(json(modelYearForecast))
      if (url.includes('brand=Volkswagen') && url.includes('model=Golf')) return Promise.resolve(json(modelYearOptions))
      return Promise.resolve(json(searchOptions))
    })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<AppProviders queryClient={queryClient}><VehicleForecastSearch apiClient={new PlannerApiClient(fetcher)} clientRelease /></AppProviders>)

    await user.type(screen.getByRole('searchbox', { name: 'Search brand or model' }), 'Golf')
    await user.click(screen.getByRole('button', { name: 'Search vehicles' }))
    await user.selectOptions(await screen.findByRole('combobox', { name: 'Registration year' }), '2020')
    await user.click(screen.getByRole('button', { name: 'Calculate forecast' }))

    expect(screen.queryByRole('radio', { name: 'Generation directly' })).not.toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: /Volkswagen Golf · registration year 2020/i })).toBeVisible()
    expect(screen.getByText('Official registration cohort')).toBeVisible()
  })

  it('searches then selects brand, model, year, and forecasts surviving fleet by market', async () => {
    const user = userEvent.setup()
    const fetcher = vi.fn<typeof fetch>().mockImplementation((input) => {
      const url = typeof input === 'string'
        ? input
        : input instanceof URL ? input.href : input.url
      if (url.includes('/vehicle-forecasts?')) return Promise.resolve(json(forecast))
      if (url.includes('brand=Volkswagen') && url.includes('model=Golf')) return Promise.resolve(json(selectedOptions))
      return Promise.resolve(json(searchOptions))
    })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<AppProviders queryClient={queryClient}><VehicleForecastSearch apiClient={new PlannerApiClient(fetcher)} /></AppProviders>)

    await user.type(screen.getByRole('searchbox', { name: 'Search brand or model' }), 'Golf')
    await user.click(screen.getByRole('button', { name: 'Search vehicles' }))
    expect(await screen.findByRole('combobox', { name: 'Brand' })).toHaveValue('Volkswagen')
    expect(screen.getByRole('combobox', { name: 'Model' })).toHaveValue('Golf')
    await user.selectOptions(await screen.findByRole('combobox', { name: 'Model year' }), '2020')
    await user.click(screen.getByRole('button', { name: 'Calculate forecast' }))

    expect(await screen.findByRole('heading', { name: /Volkswagen Golf · Golf Mk8/i })).toBeVisible()
    expect(screen.getByText('800')).toBeVisible()
    expect(screen.getByText('32')).toBeVisible()
    expect(screen.getByText(/England is not separable/i)).toBeVisible()
    expect(screen.getByText(/fleet decay has already been applied/i)).toBeVisible()
    expect(screen.getByText(/Future sales cohorts 2026, 2027, 2028 are not added/i)).toBeVisible()
  })

  it('accepts a manually typed brand and model outside the suggestion list', async () => {
    const user = userEvent.setup()
    const fordOptions = {
      vehicles: [], years: [2018, 2019, 2020],
      generations: [{
        key: 'estimated-ford-focus', name: 'estimated Ford Focus', start_year: 2010,
        end_year: 2025, basis: 'estimated_generation_window', confidence: 'low',
      }],
      horizons: [2028],
    }
    const fordForecast = {
      ...forecast,
      brand: 'Ford', model: 'Focus', generation_key: 'estimated-ford-focus',
      generation_name: 'estimated Ford Focus', generation_basis: 'estimated_generation_window',
      generation_confidence: 'low', generation_source_url: null, generation_start_year: 2010,
    }
    const fetcher = vi.fn<typeof fetch>().mockImplementation((input) => {
      const url = typeof input === 'string'
        ? input
        : input instanceof URL ? input.href : input.url
      if (url.includes('/vehicle-forecasts?')) return Promise.resolve(json(fordForecast))
      if (url.includes('brand=Ford') && url.includes('model=Focus')) return Promise.resolve(json(fordOptions))
      return Promise.resolve(json(searchOptions))
    })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<AppProviders queryClient={queryClient}><VehicleForecastSearch apiClient={new PlannerApiClient(fetcher)} /></AppProviders>)

    const brand = await screen.findByRole('combobox', { name: 'Brand' })
    await user.type(brand, 'Ford')
    await user.type(screen.getByRole('combobox', { name: 'Model' }), 'Focus')
    await user.selectOptions(await screen.findByRole('combobox', { name: 'Model year' }), '2020')
    await user.click(screen.getByRole('button', { name: 'Calculate forecast' }))

    expect(await screen.findByRole('heading', { name: /Ford Focus · estimated Ford Focus/i })).toBeVisible()
  })
})
