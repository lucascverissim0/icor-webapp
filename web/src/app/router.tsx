import {
  createRootRoute,
  createRoute,
  createRouter,
  defaultParseSearch,
  redirect,
  type RouterHistory,
} from '@tanstack/react-router'

import { App } from './App'
import { AppShell } from './AppShell'
import { RouteErrorFallback } from './ErrorBoundary'
import { PlannerPage } from '../features/planner/PlannerPage'
import { parsePlannerSearch, type PlannerRouteSearch } from '../lib/planner-search'


function validatePlannerSearch(raw: PlannerRouteSearch): PlannerRouteSearch {
  const parsed = parsePlannerSearch(raw as unknown as Record<string, unknown>)
  return {
    ...parsed.value,
    ...(parsed.invalidKeys.length > 0 && { invalidKeys: parsed.invalidKeys }),
  }
}

const rootRoute = createRootRoute({
  validateSearch: validatePlannerSearch,
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
  component: PlannerPage,
  errorComponent: RouteErrorFallback,
})

export const configurationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/planner/configurations/$configurationId',
  component: () => <App detailPlaceholder />,
  errorComponent: RouteErrorFallback,
})

const routeTree = rootRoute.addChildren([indexRoute, plannerRoute, configurationRoute])

export function createPlannerRouter(history?: RouterHistory) {
  const plannerRouter = createRouter({
    routeTree,
    defaultPreload: 'intent',
    parseSearch: (search) =>
      validatePlannerSearch(defaultParseSearch(search) as PlannerRouteSearch),
    ...(history && { history }),
  })
  if (history) history.subscribe(() => void plannerRouter.load())
  return plannerRouter
}

export const router = createPlannerRouter()

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
