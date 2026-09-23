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

Then promote it into `.local/client-evidence`, **not** into `.local/evidence`:

    mkdir .local/client-evidence/candidates
    mv .local/evidence/candidates/<derived-id> .local/client-evidence/candidates/
    uv run python scripts/build_evidence_snapshot.py promote         --root .local/client-evidence --snapshot <derived-id>
    rm -r .local/client-evidence/candidates

Promotion rewrites `active.json` in whatever root it is given. Promoting a
client-scoped snapshot into the development root makes it the *active* snapshot
there, and the internal app then refuses to start, because outside client-release
mode it requires a full-scope snapshot. It also leaves `verified.json` naming a
snapshot that is no longer active, so `mark-verified` has to be re-run against a
9.25 GB database. Give the client artifact its own root and none of that happens.

`.local/client-evidence` is a whole **evidence root**, not a snapshot directory
-- `Dockerfile:58` copies it to `/srv/icor/evidence` and the store resolves
`<root>/snapshots/<snapshot_id>`. It must contain exactly:

    .local/client-evidence/active.json
    .local/client-evidence/snapshots/<derived-id>/evidence.sqlite3
    .local/client-evidence/snapshots/<derived-id>/snapshot.json
    .local/client-evidence/snapshots/<derived-id>/validation.json

A stray file in the snapshot directory fails the build, `candidates/` would put a
second two-gigabyte copy into the build context, and `verified.json` must not be
pre-seeded because the image writes it. Prove the whole thing before paying for
an upload:

    uv run python scripts/build_evidence_snapshot.py verify --root .local/client-evidence

The image builds the bundle itself with `VITE_ICOR_CLIENT_RELEASE=verified`, and
fails the build if the baked snapshot is not client-scoped.

`openai` and `streamlit` are absent from the runtime, but not because of the
`preview` extra: a PEP 621 extra is additive, so `--extra preview` installs the
whole of `[project.dependencies]` too. They are absent because they live in a
non-default `legacy` dependency group that `uv sync --no-dev` leaves out. Confirm
it on a built image rather than trusting this paragraph:

    ls /app/.venv/lib/python3.12/site-packages | grep -E 'openai|streamlit'

## Snapshot verification is done once, at build time

Opening the active snapshot normally hashes the whole database and re-runs the
validator. On the promoted full snapshot that is about five and a half minutes;
on the client-scoped one it is proportionally less but still far from free.

Paying it on every process start is reasonable on a host that keeps the machine
running. It is not reasonable on one that stops an idle container, because the
next visitor waits for it, and `fly.toml` only avoids that by refusing to scale
to zero. Render's free tier stops a service after fifteen minutes of inactivity
and gives it a tenth of a CPU, which turns that wait into minutes.

So the image does the verification once, while it is built, and records the
result in `verified.json` beside `active.json`. `ICOR_SNAPSHOT_TRUST_BAKED=1`,
set in the image, is what allows the runtime to rely on it. The marker is
written by `write_verified_marker`, which derives it from a real verification;
it is never written by hand.

A trusted start still resolves the active pointer, reads the manifest, hashes it
and requires both the pointer and the marker to agree with it. A swapped
`snapshot.json`, a marker naming another snapshot, a marker left behind by an
earlier image, and a missing marker are all refused. Promotion never consults
the marker: trust describes an artifact that was already verified, so it cannot
be what verifies one. Outside `container` host mode the flag is ignored, because
a working directory changes under the process and the expensive check is the
only thing that would notice.

**What this gives up:** detecting corruption of the database file after the
image was built. The container filesystem is immutable and the snapshot is
read-only within it, so the property that remains is "this container serves the
artifact that was verified when its image was built". On a host where that is
not true, unset the flag.

## Deploying to Render instead of Fly

Render builds from a Git clone, and the client snapshot is roughly two gigabytes
of ignored data that cannot live in Git. So Render is given a prebuilt image
rather than the repository: build locally, push to a registry, and create the
service with "Deploy an existing image". Render documents the free compute plan
as selectable for that, for a `linux/amd64` image under 10 GB compressed.

Render sets `PORT` and expects the service to bind it. This app reads
`ICOR_PREVIEW_PORT`, so set that to Render's port explicitly rather than relying
on `PORT`; it already binds `0.0.0.0`.

Set on the service, beyond what the image already carries:

    ICOR_PREVIEW_PORT=10000
    ICOR_PREVIEW_PUBLIC_ORIGIN=https://<service>.onrender.com
    ICOR_PREVIEW_TRUSTED_PROXIES=<see below>
    ICOR_EXPORT_TOKEN=<32 or more characters>
    ICOR_PREVIEW_USERS={"client-reviewer":"$argon2id$..."}
    ICOR_PREVIEW_SESSION_SECRET=<base64url of at least 32 bytes>

`ICOR_PREVIEW_PUBLIC_ORIGIN` must be a plain HTTPS origin with no port and no
path, or the runner refuses to start. `ICOR_PREVIEW_USERS` is a JSON object;
`generate_preview_credentials.py hash-user` prints the bare hash, so wrap it.

`ICOR_PREVIEW_TRUSTED_PROXIES` is the one value that needs a decision rather
than a copy. `*` is safe only where the container port cannot be reached
directly, because otherwise `X-Forwarded-For` becomes spoofable and login
throttling stops working. Render exposes services only through its own proxy,
which is the same posture `fly.toml` relies on, but Render publishes no stable
proxy CIDR to narrow it to.

Free-tier limits worth knowing before sharing the URL: the service sleeps after
fifteen minutes idle and takes about a minute to wake, the filesystem is
ephemeral, a new deploy has fifteen minutes to pass its health check, and the
workspace has five gigabytes of outbound bandwidth a month.

Deployment needs flyctl, an authenticated Fly account, and the three secrets set
through `fly secrets import` so no value ever reaches a command line, a shell
history or a transcript:

    fly deploy --remote-only

`--remote-only` builds on Fly's builder, so the image layer is uploaded once
rather than rebuilt and re-pushed from a home connection on every change.

Verify the deployed URL against the gates and the smoke test below:

    uv run python scripts/verify_client_release.py --url https://<host> --username client-reviewer

It reads the password from stdin and prints one JSON verdict; record that output
in `docs/CODEX_HANDOFF.md` as the evidence for gates 2, 4, 5 and 7. The verdict
must read `"verdict": "deployed-release-verified"`. No other value is release
evidence.

Before paying for a deploy, run the same checks in process against the real
client artifacts:

    uv run python scripts/verify_client_release.py --local --username client-reviewer

`--local` generates its own throwaway credential, so it reads nothing from stdin,
and it runs the container preflight before building the app. It drives the app
over an ASGI scope that declares https, which is the only way the authenticated
half can run locally at all: the session cookie is `Secure`, so it is never
returned over plain HTTP, and every check after sign-in would otherwise collapse.

That declaration is also its limit. HSTS is emitted because the scope says https,
not because TLS was negotiated, so `gate5:url-is-https` and `gate5:hsts-present`
are reported as `"evidence": "asserted"` and listed under `not_proven_locally`.
A passing local run prints `"verdict": "local-preflight-passed"`, which is a
pre-flight and never the release evidence. Smoke points 1, 4 and 5 are visual and
are not covered by either mode.

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
