// @vitest-environment node

import { QueryObserver } from '@tanstack/react-query'
import { describe, expect, it } from 'vitest'

import { createPlannerQueryClient } from '../src/app/query-client'
import { ApiProblem } from '../src/lib/api/client'


describe('planner query defaults', () => {
  it('does not retry validation or not-found API problems', async () => {
    for (const status of [404, 422]) {
      const client = createPlannerQueryClient()
      let attempts = 0

      const request = client.fetchQuery({
        queryKey: ['problem', status],
        queryFn: () => {
          attempts += 1
          throw new ApiProblem('request_problem', 'Request failed.', null, [], status)
        },
        retryDelay: 0,
      })

      await expect(request).rejects.toMatchObject({ status })
      expect(attempts).toBe(1)
      client.clear()
    }
  })

  it('retries a server failure exactly once', async () => {
    const client = createPlannerQueryClient()
    let attempts = 0

    const request = client.fetchQuery({
      queryKey: ['server-problem'],
      queryFn: () => {
        attempts += 1
        throw new ApiProblem('server_error', 'Service failed.', null, [], 500)
      },
      retryDelay: 0,
    })

    await expect(request).rejects.toMatchObject({ status: 500 })
    expect(attempts).toBe(2)
    client.clear()
  })

  it('retains the prior result while a changed query is pending', () => {
    const client = createPlannerQueryClient()
    client.setQueryData(['planner', 1], { page: 1 })
    const observer = new QueryObserver(client, {
      queryKey: ['planner', 1],
      queryFn: () => Promise.resolve({ page: 1 }),
    })
    const unsubscribe = observer.subscribe(() => undefined)

    observer.setOptions({
      queryKey: ['planner', 2],
      queryFn: () => new Promise<{ page: number }>(() => undefined),
    })

    const pending = observer.getCurrentResult()
    expect(pending.data).toEqual({ page: 1 })
    expect(pending.isPlaceholderData).toBe(true)

    unsubscribe()
    client.clear()
  })
})
