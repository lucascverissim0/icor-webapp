import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider } from '@tanstack/react-router'

import { AppProviders } from './app/providers'
import { router } from './app/router'
import './app/styles.css'


const root = document.getElementById('root')
if (root === null) throw new Error('Application root element is missing')

createRoot(root).render(
  <StrictMode>
    <AppProviders>
      <RouterProvider router={router} />
    </AppProviders>
  </StrictMode>,
)
