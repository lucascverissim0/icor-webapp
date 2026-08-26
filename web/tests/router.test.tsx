// @vitest-environment happy-dom

import { createMemoryHistory } from '@tanstack/react-router'
import { describe, expect, it } from 'vitest'

import {
  configurationRoute,
  createPlannerRouter,
  opportunitiesRoute,
  plannerRoute,
} from '../src/app/router'
import { RouteErrorFallback } from '../src/app/ErrorBoundary'


describe('planner router', () => {
  it('preserves canonical search through detail navigation and browser history', async () => {
    const history = createMemoryHistory({
      initialEntries: ['/planner?market=FR&page=2&sort=brand&direction=asc'],
    })
    const router = createPlannerRouter(history)
    await router.load()

    expect(router.state.location.search).toMatchObject({
      market: ['FR'],
      page: 2,
      sort: 'brand',
      direction: 'asc',
    })

    await router.navigate({
      to: '/planner/configurations/$configurationId',
      params: { configurationId: 'demo-configuration' },
      search: (previous) => ({
        ...previous,
        page: previous.page ?? 1,
        sort: previous.sort ?? 'base_demand',
        direction: previous.direction ?? 'desc',
      }),
    })
    expect(router.state.location.pathname).toBe(
      '/planner/configurations/demo-configuration',
    )
    expect(router.state.location.search).toMatchObject({ market: ['FR'], page: 2 })

    router.history.back()
    await router.load()
    expect(router.state.location.pathname).toBe('/planner')
    expect(router.state.location.search).toMatchObject({ market: ['FR'], page: 2 })
  })

  it('contains planner and detail errors inside the application shell', () => {
    expect(plannerRoute.options.errorComponent).toBe(RouteErrorFallback)
    expect(configurationRoute.options.errorComponent).toBe(RouteErrorFallback)
    expect(opportunitiesRoute.options.errorComponent).toBe(RouteErrorFallback)
  })

  it('validates opportunity grouping independently from planner state', async () => {
    const history = createMemoryHistory({
      initialEntries: ['/opportunities?groupBy=model_year&market=FR'],
    })
    const router = createPlannerRouter(history)
    await router.load()

    expect(router.state.location.pathname).toBe('/opportunities')
    expect(router.state.location.search).toMatchObject({
      groupBy: 'model_year',
      market: ['FR'],
    })
  })
})
