import { useState } from 'react'

import { plannerApi } from '../../lib/api/client'

export function ExportPage() {
  const [cutoff, setCutoff] = useState('2026-08-28')
  const [token, setToken] = useState('')
  const [status, setStatus] = useState<string | null>(null)

  async function download() {
    setStatus('Preparing cutoff-safe export…')
    try {
      const blob = await plannerApi.mlExport(cutoff, token)
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `icor-ml-${cutoff}.csv`
      link.click()
      URL.revokeObjectURL(url)
      setStatus('Export downloaded. The access token was not stored.')
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Export failed.')
    }
  }

  return (
    <div className="export-page">
      <header className="registrations-hero"><div><p className="eyebrow">Model-ready evidence</p><h2>Cutoff-safe ML export</h2><p>Download generation assignments known by a selected date. Later publications and later observation periods are excluded.</p></div><span className="status-pill">Protected local export</span></header>
      <section className="opportunity-ranking" aria-labelledby="export-title">
        <h2 id="export-title">Prepare export</h2>
        <p>The local API requires the 32+ character capability token configured as <code>ICOR_EXPORT_TOKEN</code>. It is used for this request only.</p>
        <div className="export-form">
          <label><span>Evidence cutoff</span><input type="date" value={cutoff} onChange={(event) => setCutoff(event.target.value)} /></label>
          <label><span>Local export token</span><input autoComplete="off" type="password" value={token} onChange={(event) => setToken(event.target.value)} /></label>
          <button className="primary-action" disabled={token.length < 32 || !cutoff} onClick={() => void download()} type="button">Download CSV</button>
        </div>
        {status && <p aria-live="polite" role="status">{status}</p>}
      </section>
    </div>
  )
}
