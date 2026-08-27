import { Outlet } from '@tanstack/react-router'
import { Menu } from 'lucide-react'
import { useState, type PropsWithChildren } from 'react'

import { EvidenceBadge } from '../components/EvidenceBadge'


function NavigationLink({ href, label }: { href: string; label: string }) {
  const active = globalThis.location?.pathname.startsWith(href) ?? false
  return (
    <a
      aria-current={active ? 'page' : undefined}
      className={`navigation-link${active ? ' navigation-link--active' : ''}`}
      href={href}
    >
      <span aria-hidden="true" className="navigation-link__marker" />
      {label}
    </a>
  )
}

function PrimaryLinks() {
  return (
    <>
      <NavigationLink href="/registrations" label="Official registrations" />
      <NavigationLink href="/evidence" label="Source evidence" />
      <span className="navigation-section-label">Prototype</span>
      <NavigationLink href="/planner" label="Demand planner (prototype)" />
      <NavigationLink href="/opportunities" label="Opportunities (prototype)" />
    </>
  )
}

export function AppShell({ children }: PropsWithChildren) {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="app-shell">
      <a className="skip-link" href="#app-content">
        Skip to main content
      </a>

      <aside className="desktop-rail">
        <a aria-label="ICOR home" className="brand-mark" href="/registrations">
          <span className="brand-mark__symbol">I</span>
          <span>ICOR</span>
        </a>
        <nav aria-label="Primary" className="desktop-navigation">
          <PrimaryLinks />
        </nav>
        <p className="rail-caption">Official vehicle evidence with clearly separated prototypes</p>
      </aside>

      <div className="shell-content">
        <header className="shell-header">
          <div>
            <p className="eyebrow">Official data workspace</p>
            <h1>Vehicle evidence and planning</h1>
          </div>
          <EvidenceBadge label="Official evidence" />
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
            <PrimaryLinks />
          </div>
        </nav>

        <main id="app-content" tabIndex={-1}>
          {children ?? <Outlet />}
        </main>
      </div>
    </div>
  )
}
