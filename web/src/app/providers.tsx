import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { PropsWithChildren } from 'react'

import { defaultQueryClient } from './query-client'

interface AppProvidersProps extends PropsWithChildren {
  queryClient?: QueryClient
}

export function AppProviders({
  children,
  queryClient = defaultQueryClient,
}: AppProvidersProps) {
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
}
