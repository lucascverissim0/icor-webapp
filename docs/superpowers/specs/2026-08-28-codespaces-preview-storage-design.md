# ICOR GitHub Codespaces Preview and Remote Evidence Storage Design

**Date:** 2026-08-28  
**Status:** Approved in conversation; awaiting written-spec review  
**Scope:** Temporary authenticated ICOR preview, remote snapshot construction, and
remote evidence retention on a personal GitHub Codespace

## Context

The approved multi-year, generation-aware ICOR application is implemented on the
isolated `development/windshield-demand-platform` worktree, but its final 20-release
real-data snapshot is not yet complete or promoted. Repeated local snapshot builds
consumed several gigabytes of scratch space. The user requires the remaining build and
temporary application preview to move off the working computer, with no new software
installed locally.

The first preview audience is the user and the ICOR manager. Employee access may be
added later. The preview must therefore be authenticated even though GitHub Codespaces
uses a public forwarded URL for sharing outside the codespace owner. This is a preview
environment, not the production deployment.

## Goals

- Build the exact approved 20-release snapshot inside GitHub Codespaces rather than on
  the local working computer.
- Keep generated databases, source artifacts, credentials, and temporary files out of
  Git.
- Serve the compiled React client and FastAPI API from one authenticated origin.
- Give the user and ICOR manager separate preview credentials.
- Preserve the existing atomic candidate validation and promotion model and the strict
  prohibition on runtime demo-data fallback.
- Keep the protected production checkout and `main` branch unchanged.
- Retain a clear migration boundary from preview credentials to employee SSO.
- Require no installation of GitHub CLI or other software on the user's computer.

## Non-goals

- This design does not create a production hosting platform or production identity
  system.
- It does not make the Codespace continuously available; the preview may stop when
  idle and is subject to GitHub's retention policy.
- It does not publish evidence artifacts, databases, or credentials in a Git repository.
- It does not merge to `main`, modify the protected production checkout, or deploy the
  existing Streamlit application.
- It does not add proprietary fitment truth or expand the already approved analytical
  scope.

## Considered approaches

### 1. Authenticated, same-origin GitHub Codespaces preview — selected

Push only the isolated development branch, create a Codespace from it, acquire official
sources directly in the Codespace, construct and promote the snapshot there, and expose
one application port behind a preview authentication gate.

This is the fastest no-local-install route. A personal GitHub account has enough remote
machine disk for the source store, build scratch space, candidate, and promoted snapshot.
The trade-offs are idle shutdown, a temporary URL, and eventual Codespace deletion.

### 2. Organization-restricted Codespaces port

An organization-owned repository can restrict a forwarded port to organization members.
This would avoid an application preview gate, but it adds organization administration,
manager membership, policy configuration, and possibly plan constraints. It is not the
fastest fit for the current personal repository.

### 3. Oracle Cloud virtual machine

An Always Free VM offers substantially more persistent storage and a more conventional
deployment lifecycle. It is more durable, but account verification, VM hardening,
networking, TLS, and operating-system administration make it slower to establish. It
remains a future alternative if the preview becomes long-lived.

Free app hosts with ephemeral filesystems or one-gigabyte databases were rejected
because the validated SQLite snapshot is multi-gigabyte and must remain immutable and
available to the API.

## Architecture

### Repository and isolation

The existing private repository remains the source of code. Only
`development/windshield-demand-platform` is pushed. `main` and the production checkout
remain at their current commit. Evidence roots, generated databases, environment files,
access credentials, and build logs remain ignored.

The local machine uses its existing Git and Git Credential Manager. No GitHub CLI or
other tool is installed locally. Codespace creation is performed in the GitHub website
after the reviewed development branch is pushed.

### Codespace bootstrap

A repository-owned bootstrap command performs these operations idempotently inside the
Codespace:

1. Verify supported Python, Node, npm, uv, and available-disk requirements.
2. Install locked Python and frontend dependencies inside the Codespace only.
3. Acquire the 20 approved, checksum-pinned official releases directly from their
   publishers using the existing acquisition commands.
4. Verify manifest identities, exact byte sizes, checksums, and staged release
   invariants.
5. Build the generation-aware snapshot with the approved deterministic build timestamp
   and seed.
6. Run candidate validation and the completeness report.
7. Promote only a zero-error candidate whose inventory matches the approved releases.
8. Build the React production bundle and start the same-origin preview service.

The persistent evidence root lives below `/workspaces`, which survives Codespace stops
and starts. Temporary transfer files and failed scratch directories are removed only
after their exact paths are validated. Immutable staged releases and the promoted
snapshot remain retained for the preview lifecycle.

### Same-origin application service

FastAPI serves the compiled `web/dist` assets and the existing `/api` routes on one
port. Unknown non-API paths use an SPA fallback to `index.html`; API paths never fall
through to the SPA. Static serving rejects traversal, symlink escape, malformed paths,
and access outside the compiled asset root.

The deployment composition is separate from the local development runner. Local
development keeps its current Vite/FastAPI split and local-origin constraints. The
preview composition binds to the Codespace interface only when explicitly selected and
does not silently broaden the default local service.

### Preview authentication

All application HTML, assets, API routes, documentation, and export routes are protected
by a preview authentication boundary. A minimal data-free health endpoint may remain
unauthenticated for lifecycle checks.

The user and ICOR manager receive individually named credentials. Password verifiers and
the session-signing secret are supplied through Codespaces secrets and are never written
to Git, logs, URLs, manifests, frontend bundles, or snapshot metadata. Authentication
fails closed when configuration is missing, malformed, duplicated, or too weak.

Successful login creates a short-lived, signed, `HttpOnly`, `Secure`, `SameSite=Strict`
session cookie. Password checks use a memory-hard verifier and constant-time comparison.
Login attempts are throttled without logging submitted credentials. Logout invalidates
the browser session. Security headers prohibit framing and reduce content-sniffing and
referrer leakage.

The preview identity boundary is isolated behind a small interface. A future OIDC/SSO
adapter can replace credential verification and session issuance without changing the
planner, evidence repository, APIs, or frontend data contracts.

## Data flow

1. The operator creates a Codespace from the reviewed development branch.
2. The bootstrap downloads official source bytes directly to the remote environment.
3. Acquisition verifies each source before it enters the immutable release store.
4. The builder reads the release store, writes to a unique candidate scratch directory,
   computes generation/cohort/opportunity materializations, and validates lineage and
   completeness.
5. Atomic promotion copies the validated candidate to the immutable snapshot store and
   changes `active.json` only after revalidation.
6. FastAPI opens the promoted SQLite database read-only. Planner, registrations,
   evidence, opportunities, completeness, and ML export all resolve the same snapshot.
7. An authenticated browser loads the React bundle and calls same-origin `/api` routes.
8. Stopping the Codespace stops compute while the `/workspaces` evidence state remains
   available until the Codespace is deleted or reaches its retention limit.

No evidence data is uploaded from the working computer. The remote build remains
reproducible from the committed code, pinned manifests, deterministic parameters, and
official source endpoints.

## Failure and recovery behavior

- Insufficient disk, unsupported tools, or unavailable prerequisites fail before source
  acquisition or snapshot mutation.
- A network failure or source checksum mismatch stops acquisition and does not alter the
  active pointer.
- Each build uses a unique staging directory. Failure cleanup resolves and verifies the
  exact directory beneath the candidate root before removal.
- Validation errors preserve the last known-good active snapshot and candidate evidence
  needed for diagnosis; no fixture or demo data is substituted.
- Authentication configuration errors prevent preview startup. Unauthorized requests
  receive a generic response without disclosing account existence or application data.
- Missing frontend assets fail preview startup rather than exposing an API-only service
  accidentally.
- A stopped Codespace can be restarted with its persistent `/workspaces` data. If the
  Codespace is deleted, the snapshot is rebuilt from the private development branch and
  verified official sources.
- Secrets are re-entered through Codespaces settings after environment recreation; they
  are not recovered from repository files.

## Verification

### Security and deployment tests

- Authentication accepts each configured preview user and rejects absent, invalid,
  malformed, expired, and tampered sessions.
- Configuration fails closed for missing/weak secrets and duplicate users.
- Throttling, logout, cookie attributes, security headers, and secret-safe logging are
  covered.
- Static asset tests cover valid assets, SPA navigation, API non-fallback, traversal,
  encoding tricks, and symlink escape.
- The production bundle contains no credential, local path, demo repository reference,
  or development-only API origin.
- Unauthenticated and authenticated browser smoke tests run through the forwarded URL.

### Existing product gates

- Full maintained backend suite and focused snapshot integration suites.
- Frontend unit tests, typecheck, lint, production build, accessibility, and browser
  workflows.
- Python and npm dependency audits and maintained Ruff checks.
- Exact approved 20-release inventory, year/geography coverage, snapshot identity,
  manifest digest, and deterministic build parameters.
- Zero-error candidate validation and completeness reporting.
- Registration, generation assignment, cohort, opportunity, lineage, and export counts
  reconciled against the promoted database.
- Runtime proof that every planner/opportunity workflow reports the same promoted
  snapshot and that no demo fallback exists.
- A second deterministic build or equivalent identity replay proving reproducibility.
- Git evidence that `main` and the protected production checkout remain unchanged.

## Operational lifecycle

The preview URL is shared only with the user and ICOR manager. The forwarded port is
made public only while the application-level authentication gate is active. The
Codespace is stopped when unused to preserve free compute allowance. Before deletion,
the operator records the final snapshot identity, manifest digest, validation report,
and verification results in the durable handoff.

Adding a small number of preview users requires only adding separately named credential
verifiers to Codespaces secrets and restarting the preview. Broader employee rollout is
a separate production design: select an OIDC provider, define organization roles,
replace preview authentication, move to durable hosting, and conduct a production
security and operations review.

## Completion criteria

This design is implemented only when all of the following are true:

1. The reviewed development branch exists on GitHub without evidence data or secrets.
2. A Codespace can bootstrap from that branch without local-machine installation or
   evidence upload.
3. The approved 20-release snapshot builds, validates, and promotes remotely.
4. The compiled application and API run on one authenticated forwarded origin.
5. The user and ICOR manager can sign in with separate credentials; an unauthenticated
   browser cannot access application data.
6. All product, security, dependency, browser, and reproducibility gates pass.
7. The runtime contains no demo planner/opportunity fallback.
8. `main`, the protected production checkout, and unrelated user changes remain
   untouched.
9. The durable handoff records exact remote lifecycle and recovery instructions.

