import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { AppShell } from '../src/app/AppShell'


describe('AppShell', () => {
  it('always labels demonstration evidence and primary navigation', () => {
    render(
      <AppShell>
        <h2>Planner content</h2>
      </AppShell>,
    )

    expect(screen.getByText('Demonstration data')).toBeVisible()
    expect(screen.getByRole('heading', { name: 'Windshield demand planner' })).toBeVisible()
    expect(screen.getAllByRole('navigation', { name: 'Primary' })).toHaveLength(2)
    expect(screen.getByRole('button', { name: 'Open navigation' })).toBeVisible()
    expect(screen.getByRole('main')).toContainElement(
      screen.getByRole('heading', { name: 'Planner content' }),
    )
  })

  it('provides a skip link and does not rely on hover for planner access', () => {
    render(<AppShell>Planner content</AppShell>)

    expect(screen.getByRole('link', { name: 'Skip to planner content' })).toHaveAttribute(
      'href',
      '#planner-content',
    )
    for (const link of screen.getAllByRole('link', { name: 'Planner workbench' })) {
      expect(link).toHaveAttribute('href', '/planner')
    }
  })
})
