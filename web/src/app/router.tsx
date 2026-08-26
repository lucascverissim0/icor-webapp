import {
  createRootRoute,
  createRoute,
  createRouter,
  redirect,
} from '@tanstack/react-router'

import { App } from './App'
import { AppShell } from './AppShell'
import { RouteErrorFallback } from './ErrorBoundary'
import { parsePlannerSearch } from '../lib/planner-search'


function validatePlannerSearch(raw: Record<string, unknown>) {
  const parsed = parsePlannerSearch(raw)
  return {
    ...parsed.value,
    ...(parsed.invalidKeys.length > 0 && { invalidKeys: parsed.invalidKeys }),
  }
}

const rootRoute = createRootRoute({
  component: AppShell,
  errorComponent: RouteErrorFallback,
  notFoundComponent: () => (
    <section className="route-state">
      <p className="eyebrow">Not found</p>
      <h2>This planner view does not exist</h2>
      <a className="primary-action" href="/planner">
        Return to planner
      </a>
    </section>
  ),
})

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  beforeLoad: () =>
    redirect({
      to: '/planner',
      search: { page: 1, sort: 'base_demand', direction: 'desc' },
      throw: true,
    }),
})

export const plannerRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/planner',
  validateSearch: validatePlannerSearch,
  component: App,
})

export const configurationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/planner/configurations/$configurationId',
  validateSearch: validatePlannerSearch,
  component: () => <App detailPlaceholder />,
})

const routeTree = rootRoute.addChildren([indexRoute, plannerRoute, configurationRoute])

export const router = createRouter({
  routeTree,
  defaultPreload: 'intent',
})

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
