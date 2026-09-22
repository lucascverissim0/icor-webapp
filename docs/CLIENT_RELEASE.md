# Authenticated client preview

The client-shareable release exposes every forecastable official-source make/model
label and registration cohort year. These source-reported vehicle-year identities replace
invented fallback generation names. A manufacturer generation is shown only where an
independent manufacturer source verifies it; the application never presents a broad
estimated generation as fact.

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

1. Revoke the historically exposed OpenAI key and rotate every environment that may
   still reference it **before any internet launch**. Do not treat removing the key from
   the current tree as sufficient because it remains in Git history. Never place the old
   or replacement secret in Git, chat, a command shown to another person, or this
   runbook.
2. The active snapshot ID and release-check results match the durable handoff.
3. The frontend was built with VITE_ICOR_CLIENT_RELEASE=verified.
4. The authenticated server runs with ICOR_CLIENT_RELEASE_MODE=verified and serves
   the same compiled client bundle through ICOR_PREVIEW_ASSET_ROOT.
5. The forwarded URL uses HTTPS and is shared only with the intended named client
   reviewer. Do not expose the local Vite or API development ports.
6. The reviewer is told that source-reported vehicle-year coverage is complete for the
   forecastable public snapshot, while complete generation canonicalization,
   proprietary fitment, and replacement-history calibration require later ICOR data.
7. The forecast benchmark in docs/FORECAST_VALIDATION.md passes against the active
   snapshot, and the snapshot method version matches the application method version.

## Build and deploy the container

The client preview runs from an image that carries a pruned, client-scoped
snapshot. Derive that snapshot from the promoted full one -- deriving copies,
prunes and re-validates in minutes, where a second build would repeat the whole
replay:

    uv run python scripts/build_client_snapshot.py --root .local/evidence derive --active
    uv run python scripts/build_client_snapshot.py --root .local/evidence promote --snapshot <id>

Copy the promoted client snapshot to `.local/client-evidence`, which is the only
evidence path the Dockerfile copies. The image builds the bundle itself with
`VITE_ICOR_CLIENT_RELEASE=verified`, installs the `preview` extra only so the
`openai` and `streamlit` packages are absent from the runtime, and fails the
build if the baked snapshot is not client-scoped.

Deployment needs flyctl, an authenticated Fly account, and the three secrets set
through `fly secrets import` so no value ever reaches a command line, a shell
history or a transcript:

    fly deploy --remote-only

`--remote-only` builds on Fly's builder, so the image layer is uploaded once
rather than rebuilt and re-pushed from a home connection on every change.

Verify the deployed URL against the gates and the smoke test below:

    uv run python scripts/verify_client_release.py --url https://<host> --username client-reviewer

It reads the password from stdin and prints one JSON verdict; record that output
in `docs/CODEX_HANDOFF.md` as the evidence for gates 2, 4, 5 and 7.

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
2. Confirm Opportunities shows “Official vehicle-year evidence” and paginates across
   more than one make/model.
3. Confirm every opportunity is labelled by make, source-reported model, and
   registration year;
   no `estimated-generation-*` label is visible.
4. Search for at least Volkswagen Golf and Ford Focus, confirm their reviewed
   generation labels, select a registration year, and calculate 2028 and 2031
   forecasts.
5. Confirm the eight configured market rows render and unavailable evidence is never
   displayed as zero.
6. Confirm /evidence, /registrations, /completeness, /exports, /docs, and
   /openapi.json return 404 after authentication.
7. Sign out and confirm protected pages return 401.

Stop or delete the preview when the review window ends. Never call this preview a
production deployment, a complete generation/fitment catalogue, or a validated
windshield demand forecast.
