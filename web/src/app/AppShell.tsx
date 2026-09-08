import { Outlet } from '@tanstack/react-router'
import { Menu } from 'lucide-react'
import { useState, type PropsWithChildren } from 'react'

import { EvidenceBadge } from '../components/EvidenceBadge'


const icorLogo = new URL('../assets/icor-logo-white.svg', import.meta.url).href


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

function PrimaryLinks({ clientRelease }: { clientRelease: boolean }) {
  return (
    <>
      <span className="navigation-section-label">Decision tools</span>
      <NavigationLink href="/opportunities" label="Opportunities" />
      <NavigationLink href="/planner" label="Model search" />
      {!clientRelease && <>
        <span className="navigation-section-label">Data &amp; audit</span>
        <NavigationLink href="/registrations" label="Official registrations" />
        <NavigationLink href="/evidence" label="Source evidence" />
        <NavigationLink href="/completeness" label="Completeness" />
        <span className="navigation-section-label">Administration</span>
        <NavigationLink href="/exports" label="ML export" />
      </>}
    </>
  )
}

export function AppShell({
  children,
  clientRelease = import.meta.env.VITE_ICOR_CLIENT_RELEASE === 'verified',
}: PropsWithChildren<{ clientRelease?: boolean }>) {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="app-shell">
      <a className="skip-link" href="#app-content">
        Skip to main content
      </a>

      <aside className="desktop-rail">
        <a aria-label="ICOR home" className="brand-mark" href="/opportunities">
          <img alt="ICOR — automatically perfect" src={icorLogo} />
        </a>
        <nav aria-label="Primary" className="desktop-navigation">
          <PrimaryLinks clientRelease={clientRelease} />
        </nav>
        <p className="rail-caption">Forecast windshield replacement demand by model and generation.</p>
      </aside>

      <div className="shell-content">
        <header className="shell-header">
          <a aria-label="ICOR home" className="brand-mark brand-mark--mobile" href="/opportunities">
            <img alt="ICOR — automatically perfect" src={icorLogo} />
          </a>
          <div className="shell-header__title">
            <p className="eyebrow">Decision workspace</p>
            <h1>Windshield replacement forecasts</h1>
          </div>
          <EvidenceBadge label={clientRelease ? 'Verified client preview' : 'Forecast workspace'} />
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
            <PrimaryLinks clientRelease={clientRelease} />
          </div>
        </nav>

        <main id="app-content" tabIndex={-1}>
          {children ?? <Outlet />}
        </main>
      </div>
    </div>
  )
}
