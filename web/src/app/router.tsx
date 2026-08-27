import {
  createRootRoute,
  createRoute,
  createRouter,
  defaultParseSearch,
  redirect,
  type RouterHistory,
} from '@tanstack/react-router'

import { AppShell } from './AppShell'
import { RouteErrorFallback } from './ErrorBoundary'
import { OpportunitiesPage } from '../features/opportunities/OpportunitiesPage'
import { EvidencePage } from '../features/evidence/EvidencePage'
import { ConfigurationDetailPage } from '../features/planner/ConfigurationDetail'
import { PlannerPage } from '../features/planner/PlannerPage'
import { parsePlannerSearch, type PlannerRouteSearch } from '../lib/planner-search'
import {
  parseOpportunitySearch,
  type OpportunityRouteSearch,
} from '../lib/opportunity-search'


function validatePlannerSearch(raw: PlannerRouteSearch): PlannerRouteSearch {
  const parsed = parsePlannerSearch(raw as unknown as Record<string, unknown>)
  return {
    ...parsed.value,
    ...(parsed.invalidKeys.length > 0 && { invalidKeys: parsed.invalidKeys }),
  }
}

function validateOpportunitySearch(raw: OpportunityRouteSearch): OpportunityRouteSearch {
  const parsed = parseOpportunitySearch(raw as unknown as Record<string, unknown>)
  return {
    ...parsed.value,
    ...(parsed.invalidKeys.length > 0 && { invalidKeys: parsed.invalidKeys }),
  }
}

function parseSharedSearch(search: string): Record<string, unknown> {
  const raw = defaultParseSearch(search) as Record<string, unknown>
  const market = raw.market === undefined
    ? undefined
    : Array.isArray(raw.market) ? raw.market : [raw.market]
  const horizonValues = raw.horizon === undefined
    ? undefined
    : Array.isArray(raw.horizon) ? raw.horizon : [raw.horizon]
  const horizon = horizonValues?.map((value) =>
    typeof value === 'number' ? value : Number(value),
  )
  return {
    ...raw,
    ...(market && { market }),
    ...(horizon && { horizon }),
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
  component: PlannerPage,
  errorComponent: RouteErrorFallback,
})

export const configurationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/planner/configurations/$configurationId',
  validateSearch: validatePlannerSearch,
  component: ConfigurationDetailPage,
  errorComponent: RouteErrorFallback,
})

export const opportunitiesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/opportunities',
  validateSearch: validateOpportunitySearch,
  component: OpportunitiesPage,
  errorComponent: RouteErrorFallback,
})

export const evidenceRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/evidence',
  component: EvidencePage,
  errorComponent: RouteErrorFallback,
})

const routeTree = rootRoute.addChildren([
  indexRoute,
  plannerRoute,
  configurationRoute,
  opportunitiesRoute,
  evidenceRoute,
])

export function createPlannerRouter(history?: RouterHistory) {
  const plannerRouter = createRouter({
    routeTree,
    defaultPreload: 'intent',
    parseSearch: parseSharedSearch,
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
