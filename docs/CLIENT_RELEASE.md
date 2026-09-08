# Verified client preview

The first client-shareable release is a deliberately narrow, authenticated preview.
It exposes only vehicle/model-year rows that resolve to one unambiguous,
manufacturer-reviewed generation. The current reviewed catalog contains Volkswagen
Golf Mk6, Mk7, and Mk8 for Europe. Publisher aliases are consolidated behind the
canonical Volkswagen Golf identity, and annual transition years are excluded when
month-level generation windows overlap.

This boundary is fail-closed. The client preview:

- exposes Opportunities and Model search only;
- permits read-only opportunity, vehicle-forecast, and evidence-freshness requests;
- blocks registrations, source audit, completeness, ML export, API documentation,
  coverage management, and all non-login mutations;
- requires Argon2id credentials, signed short-lived secure cookies, login throttling,
  HTTPS, and the existing preview security headers;
- labels forecasts as assumption-led planning estimates rather than validated ICOR
  demand or exact windshield/SKU fitment.

The internal development app remains unchanged when client-release mode is absent.

## Release gates

Before sharing a build, confirm all of the following:

1. The historically exposed OpenAI key has been revoked and rotated. Never place the
   replacement secret in Git, chat, a command shown to another person, or this runbook.
2. The active snapshot ID and release-check results match the durable handoff.
3. The frontend was built with VITE_ICOR_CLIENT_RELEASE=verified.
4. The authenticated server runs with ICOR_CLIENT_RELEASE_MODE=verified and serves
   the same compiled client bundle through ICOR_PREVIEW_ASSET_ROOT.
5. The forwarded URL uses HTTPS and is shared only with the intended named client
   reviewer. Do not expose the local Vite or API development ports.
6. The reviewer is told that this is a Volkswagen Golf pilot; generation catalog
   expansion, proprietary fitment, and replacement-history calibration are later
   release phases.

## Build the client bundle

From the development worktree:

    cd web
    $env:VITE_ICOR_CLIENT_RELEASE = "verified"
    npm ci
    npm run build -- --outDir ..\.local\client-release --emptyOutDir
    cd ..

The ignored .local/client-release directory prevents an internal bundle from being
mistaken for the client bundle.

## Configure authentication

Generate one named reviewer hash and an independent session secret. Enter passwords
only into the credential generator's local prompt or protected environment; do not
commit or paste them into documentation.

    uv run python scripts/generate_preview_credentials.py hash-user --username client-reviewer
    uv run python scripts/generate_preview_credentials.py session-secret

Store the resulting ICOR_PREVIEW_USERS and ICOR_PREVIEW_SESSION_SECRET values as
Codespaces secrets. Also configure a separate 32-plus-character ICOR_EXPORT_TOKEN
because the existing fail-closed preview runner requires it, even though the client
access policy blocks the export endpoint.

Set these non-secret runtime variables in the preview environment:

    $env:ICOR_CLIENT_RELEASE_MODE = "verified"
    $env:ICOR_PREVIEW_ASSET_ROOT = "$PWD\.local\client-release"

Use the existing Codespaces bootstrap and authenticated preview runner described in
docs/DEVELOPMENT.md. Keep the forwarded port private during owner smoke testing.
Changing its visibility or sharing an external URL is a separate release action that
requires explicit authorization.

## Client smoke test

After signing in over HTTPS:

1. Confirm the navigation contains only Opportunities and Model search.
2. Confirm Opportunities shows  Verified identity catalog,  exactly one canonical
   make/model (Volkswagen Golf), and a named Golf generation on every row.
3. Confirm no 2012 or 2019 transition-year opportunity is displayed.
4. Search for Volkswagen Golf, select a non-transition model year or a generation,
   and calculate 2028 and 2031 forecasts.
5. Confirm the eight configured market rows render and unavailable evidence is never
   displayed as zero.
6. Confirm /evidence, /registrations, /completeness, /exports, /docs, and
   /openapi.json return 404 after authentication.
7. Sign out and confirm protected pages return 401.

Stop or delete the preview when the review window ends. Never call this pilot a
production deployment or a validated windshield demand forecast.
