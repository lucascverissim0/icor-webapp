import { keepPreviousData, QueryClient } from '@tanstack/react-query'

import { ApiProblem } from '../lib/api/client'
import type { OpportunitiesQuery } from '../lib/api/client'


export const queryKeys = {
  opportunities: (query: OpportunitiesQuery) => ['opportunities', query] as const,
  opportunity: (groupId: string, query: OpportunitiesQuery) =>
    ['opportunities', groupId, query] as const,
  opportunityConfigurations: (groupId: string, query: OpportunitiesQuery) =>
    ['opportunities', groupId, 'configurations', query] as const,
  opportunityContributions: (groupId: string, query: OpportunitiesQuery) =>
    ['opportunities', groupId, 'contributions', query] as const,
  opportunityFleet: (groupId: string, query: OpportunitiesQuery) =>
    ['opportunities', groupId, 'fleet', query] as const,
  coverage: ['production-coverage'] as const,
}


export function createPlannerQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        placeholderData: keepPreviousData,
        retry: (failureCount, error) => {
          if (error instanceof ApiProblem && error.status < 500) return false
          return failureCount < 1
        },
        staleTime: 30_000,
      },
    },
  })
}

export const defaultQueryClient = createPlannerQueryClient()
