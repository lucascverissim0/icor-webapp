# Source Evidence Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the validated EEA/KBA/UK candidate reviewable in the local web app as source evidence without promoting it, resolving identities, or presenting forecasts.

**Architecture:** Add a read-only application service that opens one explicitly configured candidate directory, verifies its manifest/database checksum, and executes bounded SQLite queries for summary and observation pages. Expose it through separate `/api/v1/evidence` routes and a React `/evidence` workspace whose language and visual states distinguish source observations from canonical vehicles, published values, and forecasts. Existing demo planner/opportunity services remain unchanged.

**Tech Stack:** Python 3.12, FastAPI/Pydantic, SQLite read-only URI queries, React 19, TanStack Query/Router, TypeScript, Vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-08-26-real-data-evidence-forecasting-design.md`

## Global Constraints

- Work only in `C:\Users\LucasCravoVERISSIMO\icor-webapp-development`; do not push, merge, deploy, promote a snapshot, or change the protected production checkout.
- The workspace is local/internal source diagnostics, not the real-data opportunity ranking promised by later delivery stages.
- Read only a complete candidate whose canonical manifest identity and database SHA-256 match; missing or invalid configuration returns typed unavailable state and never falls back to fixtures.
- Bound query text to 100 characters, page size to 100, and all SQL parameters; source strings render as untrusted text.
- Preserve raw publisher labels, exact measures/geographies/periods, mapping status, confidence reasons, release provenance, and snapshot versions.
- Do not create canonical vehicles, identity mappings, reconciled values, estimates, active-fleet reconstructions, or windshield forecasts in this milestone.

---

### Task 1: Read-only Evidence Review Service

**Files:**
- Create: `src/icor/application/evidence_review.py`
- Test: `tests/application/test_evidence_review.py`

**Interfaces:**
- Produces: `EvidenceReviewService.from_candidate(path: Path) -> EvidenceReviewService`.
- Produces: `summary() -> EvidenceSummary` and `list_observations(query: EvidenceObservationQuery) -> EvidenceObservationPage`.
- Query fields: `release_id`, `geography`, `measure`, `mapping_status`, `search`, `page`, and `page_size`.

- [ ] **Step 1: Write failing tests** for exact candidate checksum verification, release/mapping summaries, deterministic filtered pagination, literal wildcard escaping, read-only operation, and invalid-candidate rejection.
- [ ] **Step 2: Run** `uv run pytest tests/application/test_evidence_review.py -v` and confirm the module is missing.
- [ ] **Step 3: Implement frozen response dataclasses and bound SQLite read-only queries.** Use `load_snapshot_manifest`, `sha256_file`, `SQLiteEvidenceRepository` schema validation, `mode=ro`, deterministic `ORDER BY release_id, geography, original_make, original_model, period_end, observation_id`, escaped `LIKE ... ESCAPE '\'`, and aggregate SQL rather than `list_observations()`.
- [ ] **Step 4: Re-run the focused tests and maintained Ruff gate.**
- [ ] **Step 5: Commit** `feat: query validated source evidence candidates`.

### Task 2: Evidence HTTP Contract

**Files:**
- Create: `src/icor/api/evidence.py`
- Modify: `src/icor/api/schemas.py`
- Modify: `src/icor/api/app.py`
- Test: `tests/api/test_evidence_api.py`

**Interfaces:**
- Produces: `GET /api/v1/evidence/summary`.
- Produces: `GET /api/v1/evidence/observations?release_id=&geography=&measure=&mapping_status=&search=&page=&page_size=`.
- `create_app(..., evidence_service: EvidenceReviewService | None = None)` supports explicit test injection; default composition reads `ICOR_EVIDENCE_CANDIDATE` and otherwise exposes typed `503 evidence_unavailable`.

- [ ] **Step 1: Write failing API tests** for summary/page serialization, filters, bounded invalid queries, missing candidate 503, safe errors, and unchanged demo health behavior.
- [ ] **Step 2: Run** `uv run pytest tests/api/test_evidence_api.py -v` and confirm route failures.
- [ ] **Step 3: Implement strict Pydantic schemas, the thin router, typed unavailable boundary, and explicit app composition.** Never include local filesystem paths or raw exception text in responses.
- [ ] **Step 4: Run API tests plus `tests/api/test_planner_api.py` and `tests/api/test_opportunity_api.py`; run Ruff.**
- [ ] **Step 5: Commit** `feat: expose source evidence review API`.

### Task 3: Source Evidence Web Workspace

**Files:**
- Create: `web/src/features/evidence/EvidencePage.tsx`
- Create: `web/tests/EvidencePage.test.tsx`
- Modify: `web/src/app/router.tsx`
- Modify: `web/src/app/AppShell.tsx`
- Modify: `web/src/app/styles.css`
- Modify: `web/src/lib/api/client.ts`
- Regenerate: `web/openapi.json`
- Regenerate: `web/src/lib/api/schema.ts`

**Interfaces:**
- Produces: `/evidence` route and primary navigation item `Source evidence`.
- Displays candidate status/snapshot, four release cards, record reconciliation, mapping-status notice, filters, and paginated source observations.
- Uses generated `EvidenceSummaryResponse` and `EvidenceObservationPageResponse` types.

- [ ] **Step 1: Regenerate OpenAPI types after Task 2 and write failing component tests** for real-source labels, zero-published warning, filter requests, pagination, unavailable state, keyboard-labelled controls, and no forecast/opportunity claim.
- [ ] **Step 2: Run** `npm test -- --run web/tests/EvidencePage.test.tsx` from `web` and confirm missing component failures.
- [ ] **Step 3: Implement API client calls and the responsive evidence workspace** using existing typography, badges, panels, loading/error patterns, and URL-backed filters. Show `Reported source labels—not canonical vehicle identities` prominently.
- [ ] **Step 4: Run Vitest, typecheck, lint, production build, and OpenAPI drift check.**
- [ ] **Step 5: Commit** `feat: add source evidence review workspace`.

### Task 4: Live Verification and Handoff

**Files:**
- Modify: `README.md`
- Modify: `docs/DEVELOPMENT.md`
- Modify: `docs/CODEX_HANDOFF.md`
- Create or modify: `web/e2e/evidence.spec.ts`

**Interfaces:**
- Launcher receives `ICOR_EVIDENCE_CANDIDATE=<absolute candidate directory>`.
- Browser opens `http://127.0.0.1:<web-port>/evidence` against the exact validated candidate.

- [ ] **Step 1: Add a browser journey** covering load, four releases, source disclaimer, search/filter, mobile reflow, and no serious/critical axe findings.
- [ ] **Step 2: Run the focused browser journey and inspect desktop/mobile screenshots.**
- [ ] **Step 3: Document the environment variable, review route, source limitations, and non-promotion/non-forecast status.**
- [ ] **Step 4: Run full Python tests, maintained Ruff, lock/audit, OpenAPI, frontend unit/type/lint/build, focused browser/accessibility, and `git diff --check`.**
- [ ] **Step 5: Start a fresh local review server on available ports, verify API/web HTTP 200 and candidate identity, open `/evidence` in the default browser, and record exact process/log state in the handoff.**
- [ ] **Step 6: Commit** `docs: document source evidence workspace` while preserving unrelated `AGENTS.md` and pre-existing handoff edits unstaged when necessary.

## Self-Review

- Spec coverage: this plan implements only the approved local/internal source-diagnostics portion of API/UI stages 5, 8, and 9; canonical identity governance, reconciliation, estimation, forecasting, and the real opportunity ranking remain separately testable later stages.
- Placeholder scan: no TBD/TODO, vague error handling, or unnamed interfaces remain.
- Type consistency: `EvidenceReviewService`, query/result types, response schemas, generated TypeScript types, and `/api/v1/evidence` routes use the same names across tasks.
