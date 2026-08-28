# ICOR Codespaces Preview and Remote Evidence Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a reproducible GitHub Codespaces workflow that remotely builds and retains the approved 20-release evidence snapshot and serves the compiled ICOR React/FastAPI application behind individually named preview credentials.

**Architecture:** Preserve `icor.api.app:create_app` as the local API composition and add a separate preview composition that wraps it with fail-closed authentication, security headers, and a safe same-origin SPA asset fallback. A repository-owned Python bootstrap validates the Codespace, acquires the pinned releases directly into `/workspaces`, builds and validates a unique candidate, promotes it atomically, compiles the React client, and prints operator-safe next steps without starting an unauthenticated service.

**Tech Stack:** Python 3.12, FastAPI/Starlette, Argon2id via `argon2-cffi`, HMAC-SHA256 signed cookies, SQLite snapshot store, React 19, Vite 8, Node 24, npm, uv, GitHub Codespaces/devcontainers.

**Spec:** `docs/superpowers/specs/2026-08-28-codespaces-preview-storage-design.md`

## Global Constraints

- Work only in `C:\Users\LucasCravoVERISSIMO\icor-webapp-development` on `development/windshield-demand-platform`; never modify or merge `main` or the protected production checkout.
- Push only `development/windshield-demand-platform`, and only after the complete local application gates pass.
- Never commit evidence artifacts, generated databases, build logs, environment files, password verifiers, signing secrets, or other credentials.
- Preserve the paused local release store at `.local/evidence/releases.local-build-paused`; remote acquisition must not depend on or upload it.
- Remote evidence must live below `/workspaces`, and snapshot promotion must remain atomic and validation-gated.
- Acquire exactly the checksum-pinned EEA 2010–2024, KBA FZ10, and UK DfT VEH0160/VEH0120/VEH0124 AM/VEH0124 NZ releases (20 total).
- Use build timestamp `2026-08-27T12:00:00+00:00` and deterministic seed `20260827`.
- The preview must fail closed when its user verifiers, session secret, evidence root, or compiled frontend are missing or invalid.
- Protect application HTML, assets, `/api`, documentation, and exports; only a data-free `/healthz` and the login boundary may be anonymous.
- Keep local Vite/FastAPI development on loopback with its current origin checks; preview binding is explicit and separate.
- Do not create a Codespace, expose a port, write GitHub secrets, or push a branch until the corresponding local implementation and verification checkpoint passes.
- Do not stage the user's unrelated `AGENTS.md` modification.

---

### Task 1: Preserve the Existing Generation-Aware Product Slice

**Files:**
- Modify: `docs/CODEX_HANDOFF.md`
- Stage: the pre-existing application, API, repository, web schema, documentation, and test changes listed by `git status`, excluding `AGENTS.md`

**Interfaces:**
- Consumes: the uncommitted generation/cohort/opportunity implementation described in `docs/CODEX_HANDOFF.md`
- Produces: a verified Git commit that gives the preview work a durable product baseline

- [ ] **Step 1: Record the exact pre-checkpoint state**

Run `git status --short --branch`, `git diff --check`, and `git diff --name-only`. Confirm `.local/evidence/releases.local-build-paused` exists, `.local/evidence/releases` does not exist, and no `.build-*` directory exists below `.local/evidence/candidates`.

- [ ] **Step 2: Run the focused product tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/generations/test_registry.py tests/application/test_registrations.py tests/application/test_snapshot_completeness_report.py tests/api/test_planner_api.py tests/integration/test_clean_room_multiyear_snapshot.py -q
```

Expected: all selected tests pass; only an explicitly documented Windows symlink-privilege test may skip.

- [ ] **Step 3: Run maintained static checks on the changed product files**

Run `.venv\Scripts\ruff.exe check` over every changed Python implementation and test file reported by `git status`. Run `cd web; npm run typecheck; npm run lint; npm test -- --run; cd ..`.

Expected: every command exits zero.

- [ ] **Step 4: Update the durable handoff**

Append the exact test counts, skip count, branch head before commit, paused-release status, and statement that no remote or production state changed. Do not rewrite earlier handoff history.

- [ ] **Step 5: Commit the recovered product slice**

Stage only the product files, tests, `.gitignore`, README/development docs, completeness script, and handoff. Explicitly exclude `AGENTS.md` and this preview plan. Commit with `feat: complete generation snapshot runtime` and confirm `git status --short` shows only the unrelated `AGENTS.md` plus the plan.

### Task 2: Define and Validate Preview Security Configuration

**Files:**
- Create: `src/icor/preview/__init__.py`
- Create: `src/icor/preview/config.py`
- Test: `tests/preview/test_config.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: environment mapping containing `ICOR_PREVIEW_USERS`, `ICOR_PREVIEW_SESSION_SECRET`, and optional `ICOR_PREVIEW_SESSION_TTL_SECONDS`
- Produces: `PreviewSettings.from_environment(environment: Mapping[str, str]) -> PreviewSettings`; immutable `PreviewUser(username: str, password_hash: str)` records; `ConfigurationError` with secret-safe messages

- [ ] **Step 1: Write failing configuration tests**

Cover a valid JSON object such as `{"lucas":"$argon2id$...","manager":"$argon2id$..."}`, normalized unique usernames, an independently generated 32-byte base64url session secret, and a bounded session TTL. Parametrize rejection of missing variables, non-object JSON, empty/duplicate case-folded users, plaintext or non-Argon2id verifiers, weak/invalid signing secrets, and TTL values outside 300–43,200 seconds. Assert exceptions never contain input secrets or hashes.

- [ ] **Step 2: Run the tests and verify RED**

Run `.venv\Scripts\python.exe -m pytest tests/preview/test_config.py -q`.

Expected: collection fails because `icor.preview.config` does not exist.

- [ ] **Step 3: Add the minimal typed configuration implementation**

Implement frozen dataclasses and strict decoding. Decode the signing key with `base64.urlsafe_b64decode` after validated padding and require at least 32 decoded random bytes. Accept only `$argon2id$` encoded hashes. Store usernames for display while comparing their `casefold()` forms for uniqueness. Default the TTL to 3,600 seconds.

- [ ] **Step 4: Lock the memory-hard verifier dependency**

Add `argon2-cffi>=25.1,<26` to project dependencies and run `uv lock`. Do not add a second web framework or session package.

- [ ] **Step 5: Run focused tests and lint**

Run the configuration tests plus `.venv\Scripts\ruff.exe check src/icor/preview/config.py tests/preview/test_config.py`.

Expected: both exit zero.

- [ ] **Step 6: Commit the configuration boundary**

Commit the new package/config/tests and lockfiles with `feat: validate preview security configuration`.

### Task 3: Implement Signed Sessions, Login Throttling, and Security Headers

**Files:**
- Create: `src/icor/preview/auth.py`
- Create: `src/icor/preview/security.py`
- Test: `tests/preview/test_auth.py`

**Interfaces:**
- Consumes: `PreviewSettings`, request cookies, login form username/password, monotonic time
- Produces: `SessionCodec.issue(username: str, now: datetime) -> str`; `SessionCodec.verify(token: str, now: datetime) -> str | None`; `PreviewAuthenticator.verify(username: str, password: str) -> bool`; `LoginThrottle.allow(key: str, now: float) -> bool`; authentication/security ASGI middleware

- [ ] **Step 1: Write failing session and password tests**

Test valid, expired, malformed, and single-byte-tampered tokens; key rotation invalidation; valid credentials for each named user; generic rejection for missing users and wrong passwords; and log capture proving submitted credentials/hash strings are absent.

- [ ] **Step 2: Verify the session/password tests fail**

Run `.venv\Scripts\python.exe -m pytest tests/preview/test_auth.py -q`.

Expected: failure because the auth interfaces do not exist.

- [ ] **Step 3: Implement deterministic signed sessions and Argon2id checks**

Use canonical compact JSON containing version, username, issued-at, expiry, and a 128-bit random nonce. Sign the base64url payload with HMAC-SHA256 and compare signatures with `hmac.compare_digest`. Use `argon2.PasswordHasher.verify`; perform a dummy Argon2id verification for unknown users so account existence does not change the verification path.

- [ ] **Step 4: Write failing throttling and middleware tests**

Test the allowed-attempt window, generic HTTP 429 behavior after five failed attempts in 15 minutes, successful-login reset, stale bucket eviction, anonymous allowance only for `/healthz` and `/auth/login`, and authentication enforcement for `/`, assets, `/api`, `/docs`, `/openapi.json`, and exports.

- [ ] **Step 5: Implement the bounded in-memory throttle and middleware**

Key throttling by a keyed digest of normalized username plus client address so raw usernames are not retained. Cap buckets and evict expired entries. Attach authenticated username to `request.state.preview_username`; return generic HTML/JSON-safe unauthorized responses without account disclosure.

- [ ] **Step 6: Add and verify security headers**

Test and set `Content-Security-Policy: default-src 'self'; object-src 'none'; frame-ancestors 'none'`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`, and `Cache-Control: no-store` on authentication responses.

- [ ] **Step 7: Run focused tests and lint, then commit**

Run `.venv\Scripts\python.exe -m pytest tests/preview/test_auth.py -q` and Ruff on the new files. Commit with `feat: protect preview sessions`.

### Task 4: Add the Login/Logout Boundary and Same-Origin Preview Factory

**Files:**
- Create: `src/icor/preview/app.py`
- Create: `src/icor/preview/static.py`
- Test: `tests/preview/test_app.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: `create_app()` from `icor.api.app`, `PreviewSettings`, compiled `web/dist`, active snapshot root, coverage database, and export token
- Produces: `create_preview_app(settings: PreviewSettings | None = None, *, asset_root: Path | None = None, snapshot_root: Path | None = None) -> FastAPI`; `resolve_asset(asset_root: Path, request_path: str) -> Path | None`

- [ ] **Step 1: Isolate preview credentials in every test**

Extend the autouse fixture to delete all `ICOR_PREVIEW_*` variables so machine or Codespaces secrets cannot affect tests.

- [ ] **Step 2: Write failing startup and authentication-route tests**

Test fail-closed startup for missing configuration, missing active generation snapshot, and absent `web/dist/index.html`. Test login GET, generic invalid POST response, valid POST producing a `Secure; HttpOnly; SameSite=Strict; Path=/` cookie without password/hash leakage, authenticated navigation, POST logout cookie deletion, and invalidated post-logout access.

- [ ] **Step 3: Implement the preview factory and auth routes**

Compose the existing API rather than duplicating routers. Keep `/api/health` protected and add a data-free `/healthz` returning only `{"status":"ok"}`. Render a minimal server-owned login form, accept form data with a strict size limit, issue the signed cookie on success, and clear it on logout. Disable Swagger/ReDoc in preview or leave their routes protected; never expose schema before authentication.

- [ ] **Step 4: Write failing static-serving tests**

Cover a hashed JS asset, CSS/media content types, extensionless SPA navigation, authenticated API 404 remaining JSON 404, traversal using `..`, backslashes, NUL, mixed/percent-encoded separators, and an in-root symlink targeting a file outside the asset root. Assert no local absolute path appears in responses.

- [ ] **Step 5: Implement safe static resolution and SPA fallback**

Reject malformed segments before resolution. Resolve existing files strictly and require the final path to be relative to the strictly resolved asset root. Never follow a symlink outside that root. Return `index.html` only for non-API extensionless application paths; missing asset-like paths and all `/api/*` paths remain 404.

- [ ] **Step 6: Verify focused preview composition and local API regression**

Run `.venv\Scripts\python.exe -m pytest tests/preview/test_app.py tests/api/test_planner_api.py tests/api/test_snapshot_runtime_composition.py -q`, followed by Ruff.

Expected: all tests pass and the local CORS contract remains unchanged.

- [ ] **Step 7: Commit the preview composition**

Commit with `feat: serve authenticated same-origin preview`.

### Task 5: Build an Idempotent Codespaces Snapshot Bootstrap

**Files:**
- Create: `src/icor/preview/bootstrap.py`
- Create: `scripts/bootstrap_codespaces_preview.py`
- Test: `tests/preview/test_bootstrap.py`

**Interfaces:**
- Consumes: repository root, `/workspaces` root, official source registry, subprocess runner, available-disk provider
- Produces: `BootstrapPlan` with exact 20 source keys/release IDs; `validate_environment(...)`; `bootstrap(...) -> BootstrapResult`; CLI `--check`, `--acquire`, `--build`, `--promote`, and `--prepare` phases

- [ ] **Step 1: Write failing inventory and prerequisite tests**

Assert the source keys are `eea-2010-final` through `eea-2024-final`, `kba-fz10-2024`, `uk-veh0160-gb`, `uk-veh0120-gb`, `uk-veh0124-am`, and `uk-veh0124-nz`, with 20 unique expected release IDs. Test rejection outside GitHub Codespaces, evidence roots outside `/workspaces`, unsupported Python/Node/npm/uv versions, missing lockfiles, and insufficient free bytes.

- [ ] **Step 2: Verify RED and implement environment planning**

Run the focused tests, then implement immutable plan/result dataclasses and injected probes. Require Python 3.12, a Node version satisfying `web/package.json`, npm, uv 0.11.3, and a conservative free-space floor derived from retained releases plus candidate/promoted headroom. Never print environment values that may contain secrets.

- [ ] **Step 3: Write failing phase/idempotency tests**

With fake runners and stores, assert acquisition invokes `scripts/acquire_official_evidence.py` once for each absent release and verifies already-present releases; build passes all 20 unique `--release` arguments plus the approved timestamp/seed; completeness runs before promotion; a nonzero validation/completeness result leaves `active.json` untouched; an already-active matching snapshot skips rebuild; frontend preparation uses `npm ci` then `npm run build`.

- [ ] **Step 4: Implement bootstrap phases using existing public CLIs**

Use argument arrays, never shell interpolation. Persist evidence at `/workspaces/.icor/evidence` by default. Capture machine-readable command output, redact paths in operator summaries, and retain failed candidates for diagnosis. Resolve cleanup targets beneath the candidates root before deleting only bootstrap-owned transfer fragments; do not delete immutable releases, candidates, snapshots, or broad roots.

- [ ] **Step 5: Add safe CLI output and restart behavior**

`--check` performs no mutation. `--prepare` performs validated acquisition/build/completeness/promotion/frontend build and emits snapshot ID, manifest digest, counts, and the exact start command. A second run verifies and reuses staged releases and a matching promoted snapshot.

- [ ] **Step 6: Run bootstrap tests, source-registry tests, and lint**

Run `.venv\Scripts\python.exe -m pytest tests/preview/test_bootstrap.py tests/evidence/test_acquisition.py tests/evidence/test_source_registry.py -q` and Ruff.

- [ ] **Step 7: Commit the bootstrap**

Commit with `feat: bootstrap codespaces evidence preview`.

### Task 6: Add an Explicit Preview Runner and Harden the Devcontainer

**Files:**
- Create: `scripts/run_codespaces_preview.py`
- Test: `tests/preview/test_runner.py`
- Modify: `.devcontainer/devcontainer.json`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: completed bootstrap state, `ICOR_PREVIEW_*` secrets, `ICOR_EVIDENCE_ACTIVE_ROOT`, `ICOR_COVERAGE_DB`, and `ICOR_EXPORT_TOKEN`
- Produces: prerequisite-checking runner that executes `uvicorn icor.preview.app:create_preview_app --factory --host 0.0.0.0 --port 8000`; Codespaces port metadata that never makes the port public automatically

- [ ] **Step 1: Write failing runner tests**

Test `--check` rejection for non-Codespaces execution, missing/weak preview config, missing active snapshot, missing frontend bundle, or missing export token. Assert default host is not broadened by the local runner and the Codespaces runner refuses a non-explicit host override.

- [ ] **Step 2: Implement the runner**

Use direct subprocess argument arrays and preserve process exit codes. Bind `0.0.0.0` only after every preview prerequisite passes. Never print secrets, verifiers, cookies, or the entire environment.

- [ ] **Step 3: Replace the legacy devcontainer auto-start**

Pin the existing Python 3.12 image by its supported immutable reference if available in project policy, install uv 0.11.3 and locked Python/npm dependencies, open the new handoff/docs files, label port 8000 as `ICOR authenticated preview`, set `onAutoForward` to `silent`, and do not set public visibility or auto-start Streamlit.

- [ ] **Step 4: Verify configuration safety**

Add tests that parse `.devcontainer/devcontainer.json` and assert there is no password, verifier, signing key, public-port directive, Streamlit auto-start, CORS/XSRF weakening, or evidence path inside Git.

- [ ] **Step 5: Run runner/toolchain tests and commit**

Run `.venv\Scripts\python.exe -m pytest tests/preview/test_runner.py tests/test_toolchain.py tests/test_repository_security.py -q` plus Ruff. Commit with `chore: configure secure codespaces preview`.

### Task 7: Document Secret Generation, Remote Lifecycle, and Recovery

**Files:**
- Create: `scripts/generate_preview_credentials.py`
- Test: `tests/preview/test_credentials_cli.py`
- Modify: `README.md`
- Modify: `docs/DEVELOPMENT.md`
- Modify: `docs/CODEX_HANDOFF.md`
- Modify: `DEPLOYMENT.md`

**Interfaces:**
- Consumes: an interactively entered password and a public username label
- Produces: an Argon2id verifier or a random 32-byte base64url session key on stdout, never the password; operator instructions for GitHub Codespaces secrets and the preview lifecycle

- [ ] **Step 1: Write failing credential-helper tests**

Inject password readers and randomness. Assert hashes verify with Argon2id, signing keys decode to 32 bytes, plaintext passwords never appear in stdout/stderr, non-interactive password arguments are rejected, and invalid usernames fail without echoing input.

- [ ] **Step 2: Implement the narrow helper**

Provide `hash-user --username NAME` using hidden double-entry input and `session-secret` using `secrets.token_bytes(32)`. Print only copyable secret values and the corresponding Codespaces secret name; do not write files.

- [ ] **Step 3: Write exact operator documentation**

Document browser-based Codespace creation from the development branch; secret names and JSON shape; separate Lucas/manager credentials; `--check`, `--prepare`, runner, port-visibility, smoke-test, stop/restart, retention, rebuild, and deletion procedures. State that the port becomes public only after auth smoke tests and returns private immediately after review.

- [ ] **Step 4: Document recovery and evidence identity capture**

Include commands to verify `active.json`, snapshot/database digest, release inventory, completeness report, and deterministic replay. Explain that Codespace deletion loses remote evidence and requires direct official reacquisition; it never requires a local upload.

- [ ] **Step 5: Run helper tests, docs command checks, and commit**

Run focused tests and every documented `--help`/`--check` command that is safe locally. Commit with `docs: add codespaces preview runbook`.

### Task 8: Run Complete Local Security and Application Gates

**Files:**
- Modify: `docs/CODEX_HANDOFF.md`
- Modify only if generated: `requirements.txt`, `web/openapi.json`, `web/src/lib/api/schema.ts`

**Interfaces:**
- Consumes: all preview and pre-existing product commits
- Produces: auditable local verification evidence and a branch that is safe to push for Codespace creation

- [ ] **Step 1: Run repository integrity checks**

Run `git diff --check`, verify ignored paths with `git check-ignore`, scan tracked files for credential shapes/local database signatures, and confirm `git status` contains no evidence artifact, secret, generated frontend bundle, or unrelated staged file.

- [ ] **Step 2: Run the complete backend gates**

Run:

```powershell
uv lock --check
.venv\Scripts\ruff.exe check src tests scripts/audit_baseline.py scripts/build_evidence_snapshot.py scripts/bootstrap_codespaces_preview.py scripts/run_codespaces_preview.py scripts/generate_preview_credentials.py
.venv\Scripts\python.exe -m pytest
uv run pip-audit
```

Expected: all maintained tests/checks pass; only documented platform skips remain; dependency audit reports no known third-party vulnerabilities.

- [ ] **Step 3: Run the complete frontend gates**

Run:

```powershell
cd web
npm run openapi:check
npm test -- --run
npm run typecheck
npm run lint
npm run build
npm run e2e
cd ..
```

Expected: every command exits zero and the production bundle scan finds no credential, absolute local path, demo repository reference, or development API origin.

- [ ] **Step 4: Run authenticated composition smoke tests**

Create only disposable test credentials and a fixture snapshot under an ignored exact temporary root. Start the preview on loopback, verify anonymous `/healthz`, anonymous denial for app/API/assets/docs/export, both named logins, same-origin data access, logout, expired/tampered cookie rejection, and safe process shutdown. Do not expose a forwarded/public port locally.

- [ ] **Step 5: Prove protected state is unchanged**

Record the local `main`/production checkout commit and `origin/main` commit, and confirm they still equal the pre-work values. Confirm the only intended branch is `development/windshield-demand-platform`.

- [ ] **Step 6: Update handoff and commit verification evidence**

Append exact commands, test counts, audit result, bundle result, active local snapshot identity, paused-release status, and the next browser-only GitHub actions. Commit with `docs: checkpoint preview verification`.

### Task 9: Push the Reviewed Development Branch and Perform the Remote Preview Build

**Files:**
- Modify after remote verification: `docs/CODEX_HANDOFF.md`

**Interfaces:**
- Consumes: the verified local development branch and user-controlled GitHub/Codespaces UI
- Produces: a private remote development branch, an authenticated public-forwarded Codespaces preview, and recorded remote snapshot/reproducibility evidence

- [ ] **Step 1: Inspect the outgoing branch before remote mutation**

Run `git status --short --branch`, `git log --oneline origin/main..HEAD`, and a tracked-secret/evidence scan. Require a clean intended diff except the unrelated unstaged `AGENTS.md` change. Stop if any local evidence, secret, or generated database is tracked.

- [ ] **Step 2: Push only the development branch**

After the local gate passes, run `git push -u origin development/windshield-demand-platform`. Do not push or merge `main`, tags, or any other ref.

- [ ] **Step 3: Have the user create the Codespace and secrets in GitHub's website**

The user selects the private repository and development branch, creates the Codespace, then adds `ICOR_PREVIEW_USERS`, `ICOR_PREVIEW_SESSION_SECRET`, and `ICOR_EXPORT_TOKEN` as Codespaces secrets. Credentials are communicated to the manager outside Git, logs, and this conversation.

- [ ] **Step 4: Bootstrap and verify remotely before public forwarding**

Inside the Codespace run the documented `--check` and `--prepare` commands, then the complete backend/frontend gates. Start the preview while the port remains private and verify health plus authenticated access as the owner.

- [ ] **Step 5: Prove deterministic identity**

Run the documented identity replay/second deterministic build, compare snapshot identity, database digest, release inventory, and completeness results, and retain the exact machine-readable reports below the ignored `/workspaces/.icor` state.

- [ ] **Step 6: Temporarily share and smoke-test**

Only after authentication checks pass, explicitly change port 8000 visibility to public. In a logged-out browser verify application/API denial; then verify separate Lucas and manager logins, core planner/opportunity/evidence/completeness flows, export authorization, and logout. Immediately return the port to private if any gate fails.

- [ ] **Step 7: Record the remote checkpoint**

Record Codespace lifecycle/recovery steps, remote branch commit, snapshot identity, manifest/database digests, counts, verification results, and preview shutdown/private-port state in `docs/CODEX_HANDOFF.md`. Commit and push only that documentation checkpoint to the development branch.
