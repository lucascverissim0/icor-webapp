import { keepPreviousData, QueryClient } from '@tanstack/react-query'

import { ApiProblem } from '../lib/api/client'


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
