import axe from 'axe-core'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { AppShell } from '../src/app/AppShell'
import { RouteErrorFallback } from '../src/app/ErrorBoundary'


describe('AppShell', () => {
  it('leads with official evidence and separates prototype navigation', () => {
    render(
      <AppShell>
        <h2>Planner content</h2>
      </AppShell>,
    )

    expect(screen.getByText('Official evidence')).toBeVisible()
    expect(screen.getByRole('heading', { name: 'Vehicle evidence and planning' })).toBeVisible()
    expect(screen.getAllByText('Prototype').length).toBeGreaterThan(0)
    expect(screen.getByRole('navigation', { name: 'Primary' })).toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Mobile primary' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Open navigation' })).toBeVisible()
    expect(screen.getByRole('main')).toContainElement(
      screen.getByRole('heading', { name: 'Planner content' }),
    )
  })

  it('provides a skip link and does not rely on hover for planner access', () => {
    render(<AppShell>Planner content</AppShell>)

    expect(screen.getByRole('link', { name: 'Skip to main content' })).toHaveAttribute(
      'href',
      '#app-content',
    )
    for (const link of screen.getAllByRole('link', { name: 'Demand planner (prototype)' })) {
      expect(link).toHaveAttribute('href', '/planner')
    }
    for (const link of screen.getAllByRole('link', { name: 'Opportunities (prototype)' })) {
      expect(link).toHaveAttribute('href', '/opportunities')
    }
    for (const link of screen.getAllByRole('link', { name: 'Official registrations' })) {
      expect(link).toHaveAttribute('href', '/registrations')
    }
    for (const link of screen.getAllByRole('link', { name: 'Source evidence' })) {
      expect(link).toHaveAttribute('href', '/evidence')
    }
  })

  it('labels the mobile disclosure according to its current action', async () => {
    const user = userEvent.setup()
    render(<AppShell>Planner content</AppShell>)

    const trigger = screen.getByRole('button', { name: 'Open navigation' })
    await user.click(trigger)

    expect(screen.getByRole('button', { name: 'Close navigation' })).toHaveAttribute(
      'aria-expanded',
      'true',
    )
  })

  it('keeps the shell and evidence label around a safe child-route error', () => {
    render(
      <AppShell>
        <RouteErrorFallback error={new Error('private failure detail')} reset={() => undefined} />
      </AppShell>,
    )

    expect(screen.getByText('Official evidence')).toBeVisible()
    expect(screen.getByRole('navigation', { name: 'Primary' })).toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Mobile primary' })).toBeInTheDocument()
    expect(screen.getByText('This view could not be opened')).toBeVisible()
    expect(screen.queryByText('private failure detail')).not.toBeInTheDocument()
  })

  it('has no automated axe violations in the default shell state', async () => {
    const { container } = render(<AppShell>Planner content</AppShell>)

    const results = await axe.run(container)

    expect(results.violations).toEqual([])
  })
})
