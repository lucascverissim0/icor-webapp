import { Outlet } from '@tanstack/react-router'
import { Menu } from 'lucide-react'
import { useState, type PropsWithChildren } from 'react'

import { EvidenceBadge } from '../components/EvidenceBadge'


function PlannerLink() {
  return (
    <a className="navigation-link navigation-link--active" href="/planner">
      <span aria-hidden="true" className="navigation-link__marker" />
      Planner workbench
    </a>
  )
}

export function AppShell({ children }: PropsWithChildren) {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="app-shell">
      <a className="skip-link" href="#planner-content">
        Skip to planner content
      </a>

      <aside className="desktop-rail">
        <a aria-label="ICOR planner home" className="brand-mark" href="/planner">
          <span className="brand-mark__symbol">I</span>
          <span>ICOR</span>
        </a>
        <nav aria-label="Primary" className="desktop-navigation">
          <PlannerLink />
        </nav>
        <p className="rail-caption">Configuration-level planning</p>
      </aside>

      <div className="shell-content">
        <header className="shell-header">
          <div>
            <p className="eyebrow">Planning scenario</p>
            <h1>Windshield demand planner</h1>
          </div>
          <EvidenceBadge />
        </header>

        <nav aria-label="Mobile primary" className="mobile-navigation">
          <button
            aria-controls="mobile-navigation-links"
            aria-expanded={mobileOpen}
            aria-label={mobileOpen ? 'Close navigation' : 'Open navigation'}
            className="mobile-navigation__trigger"
            onClick={() => setMobileOpen((open) => !open)}
            type="button"
          >
            <Menu aria-hidden="true" size={19} />
            Menu
          </button>
          <div hidden={!mobileOpen} id="mobile-navigation-links">
            <PlannerLink />
          </div>
        </nav>

        <main id="planner-content" tabIndex={-1}>
          {children ?? <Outlet />}
        </main>
      </div>
    </div>
  )
}
