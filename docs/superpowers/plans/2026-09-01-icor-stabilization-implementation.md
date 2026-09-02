# ICOR Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver fast, understandable registration lookup, Source Evidence, Generation Planner, and Opportunity Ranking workflows over a scope-aware 2000-to-latest timeline.

**Architecture:** Build immutable query projections and indexes into a new snapshot schema, then query them through bounded SQL repository methods instead of scanning observations or materializing all planning records in Python. Keep observed, provisional, estimated, active-fleet, and annual-registration semantics explicit from source ingestion through API schemas and UI labels.

**Tech Stack:** Python 3.13, FastAPI, SQLite, Pydantic, pytest, React 19, TypeScript, TanStack Query/Router, Vitest, Testing Library, Playwright.

**Spec:** `docs/superpowers/specs/2026-09-01-icor-search-data-performance-stabilization-design.md`

## Global Constraints

- Work only on `development/windshield-demand-platform`; do not modify protected `main`.
- Preserve the unrelated local `AGENTS.md` modification.
- Use failing tests before every production behavior change.
- Never present 2000-2009 EU model estimates as observed official registrations.
- Preserve valid vintage evidence while separating observation, first-registration, manufacture, and model years.
- No interactive endpoint may deserialize an unbounded result set.
- Registration, Evidence, and Planner first pages target p95 <= 2 seconds; Opportunity targets p95 <= 3 seconds; Planner options target p95 <= 1 second.
- Build a new sealed candidate and promote atomically only after validation; never mutate the active snapshot in place.

---

### Task 1: Correct registration search semantics

**Files:**
- Modify: `web/src/features/registrations/RegistrationsPage.tsx`
- Modify: `web/src/features/registrations/RegistrationsPage.test.tsx`
- Modify: `src/icor/application/registrations.py`
- Modify: `tests/application/test_registrations.py`
- Modify: `tests/api/test_registration_api.py`

**Interfaces:**
- Consumes: existing `RegistrationQuery(search: str | None)` and URL-backed registration search state.
- Produces: `_search_tokens(search: str) -> tuple[str, ...]`; token predicates that match across canonical make plus model; route updates that preserve geography/year/page-size.

- [x] **Step 1: Write the failing frontend regression test**

Add a test that starts at `?geography=DE&year=2023&page=4`, enters `Volkswagen Golf`, submits, and asserts the state becomes `geography=DE&year=2023&search=Volkswagen+Golf&page=1`.

- [x] **Step 2: Run the frontend test and verify RED**

Run: `npm test -- RegistrationsPage.test.tsx --run`
Expected: FAIL because geography and year are replaced by the search-only object.

- [x] **Step 3: Preserve route-backed filters**

Change the submit update to spread the parsed route state before setting trimmed search and page 1. Keep the clear action subject to the same preservation rule.

- [x] **Step 4: Run the frontend test and verify GREEN**

Run: `npm test -- RegistrationsPage.test.tsx --run`
Expected: PASS.

- [x] **Step 5: Write failing backend multi-token tests**

Create a real temporary SQLite fixture containing canonical `volkswagen / golf`, `volkswagen / polo`, and `ford / golf` rows. Assert `Volkswagen Golf` returns only the Volkswagen Golf identity, while punctuation and repeated whitespace normalize to the same result and `%` remains literal.

- [x] **Step 6: Run backend tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/application/test_registrations.py tests/api/test_registration_api.py -q`
Expected: FAIL because the whole string is matched independently against make or model.

- [x] **Step 7: Implement tokenized cross-field matching**

Add `_search_tokens` using Unicode case folding, whitespace compression, and punctuation-to-space normalization. For every token, append a parenthesized `(LOWER(v.make) LIKE ? OR LOWER(v.model) LIKE ?)` predicate; combine token predicates with `AND` and retain `_escape_like` for literal wildcards.

- [x] **Step 8: Verify Task 1 and commit**

Run the two focused frontend/backend commands, Ruff on `registrations.py`, and `git diff --check`. Commit only Task 1 files with `fix: make registration search scope aware`.

### Task 2: Add snapshot query projections and indexes

**Files:**
- Modify: `src/icor/infrastructure/sqlite_evidence_repository.py`
- Create: `src/icor/application/snapshot_queries.py`
- Modify: `src/icor/application/snapshot_build.py`
- Modify: `tests/infrastructure/test_sqlite_evidence_repository.py`
- Modify: `tests/application/test_snapshot_build.py`

**Interfaces:**
- Produces: schema version 5; `SnapshotQueryProjectionService.apply(repository) -> SnapshotQueryProjectionResult`.
- Produces tables: `registration_family_aggregate`, `registration_label_aggregate`, `evidence_release_summary`, and `planner_option`.

- [x] **Step 1: Write failing schema tests**

Assert fresh repositories report schema 5 and contain composite indexes for observation scope/year/status, evidence filters, cohort opportunity joins, and canonical normalized make/model. Assert opening future schema 6 fails.

- [x] **Step 2: Run schema tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/infrastructure/test_sqlite_evidence_repository.py -q`
Expected: FAIL at schema version and missing table/index assertions.

- [x] **Step 3: Add schema 5**

Create the four query tables with primary keys that include scope, year, publication status, and canonical identity. Add indexes matching registration ordering/search, Evidence filters/sorts, Planner option queries, opportunity horizon/geography, and lineage joins. Migrate mutable test databases from version 4; sealed active snapshots remain immutable.

- [x] **Step 4: Verify schema GREEN**

Run the schema test command and expect PASS.

- [x] **Step 5: Write failing projection tests**

Build a small snapshot with Golf official-label variants, final and provisional releases, active-fleet vintage rows, and planner rows. Assert family totals, label breakdown, release summaries, and planner options are deterministic; assert active fleet never enters new-registration aggregates.

- [x] **Step 6: Implement projection build and verify GREEN**

Use `INSERT ... SELECT ... GROUP BY` inside one repository transaction. Preserve source/release lineage as stable JSON arrays and keep observed/provisional/estimated statuses in separate rows. Call projection creation after generation planning and before candidate sealing. Run focused snapshot-build tests and expect PASS.

- [x] **Step 7: Verify Task 2 and commit**

Run focused repository/build tests, Ruff, and `git diff --check`. Commit with `feat: build indexed snapshot query projections`.

### Task 3: Serve fast family registration lookup and timeline availability

**Files:**
- Modify: `src/icor/application/registrations.py`
- Modify: `src/icor/api/schemas.py`
- Modify: `web/src/lib/api/schema.ts`
- Modify: `web/src/features/registrations/RegistrationsPage.tsx`
- Modify: registration application/API/frontend tests.

**Interfaces:**
- Produces: `RegistrationAvailability(year, geography, status, evidence_kind)` and `RegistrationLabelBreakdown` on family rows.
- Registration summary exposes the inclusive 2000-to-latest selector range plus scope-aware availability; missing is distinct from zero.

- [x] **Step 1: Write failing behavior tests**

Assert a Golf-family row totals its official labels, exposes their breakdown, separates final/provisional/estimated values, and returns `scope_unavailable` for unsupported EU27 2000 rather than zero. Assert UK availability can differ from EU27.

- [x] **Step 2: Verify RED**

Run focused registration application/API/frontend tests and confirm failures are missing fields/behavior.

- [x] **Step 3: Query aggregate projections**

Replace observation-table ranking and summary scans with indexed aggregate-table count/page queries. Return only one requested page and its pre-aggregated totals. Add a label-breakdown detail query scoped to one family row.

- [x] **Step 4: Render family results and availability**

Show one family total, expandable official labels, Final/Provisional/Estimated badge, source lineage, and unavailable-year explanation. Populate years 2000 through latest while disabling or explaining unavailable scope combinations.

- [x] **Step 5: Verify Task 3 and commit**

Run focused Python and Vitest suites plus an `EXPLAIN QUERY PLAN` test that asserts aggregate indexes are used. Commit with `feat: serve family registration aggregates`.

### Task 4: Redesign and index Source Evidence

**Files:**
- Modify: `src/icor/application/evidence_review.py`
- Modify: `src/icor/api/evidence.py`
- Modify: `src/icor/api/schemas.py`
- Modify: `web/src/lib/evidence-search.ts`
- Modify: `web/src/features/evidence/EvidencePage.tsx`
- Modify: evidence application/API/frontend tests.

**Interfaces:**
- Produces: source-catalog summaries from `evidence_release_summary` and bounded `EvidenceObservationPage` filters including observation year and year semantics.

- [x] **Step 1: Write failing catalog and year-semantics tests**

Assert the default response explains each source's measure and coverage without scanning observations. Assert a 2024 stock observation with first-use 1914 renders observation year 2024, first-registration year 1914, and never the label `1914 registrations`.

- [x] **Step 2: Verify RED**

Run evidence application/API/frontend tests and confirm the current raw-table-first behavior fails.

- [x] **Step 3: Implement bounded evidence queries**

Read catalog metrics from the projection table. Add indexed filters and stable pagination for raw rows. Select only response columns and perform no unfiltered observation aggregate during a request.

- [x] **Step 4: Implement source-catalog UI**

Render plain-language source cards first. Put raw observations behind an explicit drill-down and spell out Observation year, First registration, Manufacture year, and Model year. Preserve vintage evidence and surface validation flags.

- [x] **Step 5: Verify Task 4 and commit**

Run focused Python/Vitest tests and query-plan checks; commit with `feat: make source evidence understandable and bounded`.

### Task 5: Replace unbounded Planner materialization

**Files:**
- Modify: `src/icor/application/planner.py`
- Modify: `src/icor/infrastructure/snapshot_planner_repository.py`
- Modify: `src/icor/api/planner.py`
- Modify: `tests/infrastructure/test_snapshot_planner_repository.py`
- Modify: `tests/application/test_planner_application.py`
- Modify: `tests/api/test_planner_api.py`
- Modify: `web/src/features/planner/PlannerPage.tsx` and tests.

**Interfaces:**
- Replace interactive `list_all()` calls with `options()`, `search(query)`, `get(id)`, and `list_model_year_demand(id, page, page_size)` repository methods.

- [x] **Step 1: Write failing repository/API tests**

Assert options use `planner_option`, search returns one SQL-paginated page, detail loads one configuration, and a guard connection raises if an unbounded projection query is attempted.

- [x] **Step 2: Verify RED**

Run focused Planner tests and observe failures from the current `list_all()` contract.

- [x] **Step 3: Implement dedicated SQL methods**

Push filters, sorting, count, pagination, model-year demand, and requested-record lineage into SQLite. Cache only immutable options/version metadata. Remove `list_all()` from interactive services.

- [x] **Step 4: Add bounded UI loading/error recovery**

Render controls from options independently, cancel stale page requests, show actionable retry errors, and prevent an indefinite loading shell.

- [x] **Step 5: Verify Task 5 and commit**

Run focused tests and representative query timing; commit with `perf: paginate generation planner in sqlite`.

### Task 6: Replace unbounded Opportunity aggregation

**Files:**
- Modify: `src/icor/application/opportunities.py`
- Create: `src/icor/infrastructure/snapshot_opportunity_repository.py`
- Modify: `src/icor/api/opportunities.py`
- Modify: `src/icor/api/schemas.py`
- Modify: `web/src/lib/opportunity-search.ts`
- Modify: `web/src/features/opportunities/OpportunityRanking.tsx`
- Modify: opportunity repository/application/API/frontend tests.

**Interfaces:**
- Produces: `OpportunityRepository.search(query) -> OpportunityPage` and `drill_down(group_id, query, page, page_size)` with SQL-side grouping and coverage resolution.

- [ ] **Step 1: Write failing bounded-query tests**

Assert deterministic brand/model/model-year ranking, summary totals, coverage states, pagination, and selected-group drill-down without calling Planner `list_all()`.

- [ ] **Step 2: Verify RED**

Run focused opportunity tests and confirm current all-atom aggregation fails the bounded repository contract.

- [ ] **Step 3: Implement SQL opportunity repository**

Aggregate requested group/market/horizon in SQLite, join exact/fallback coverage, compute deterministic scoring inputs, paginate before Python serialization, and query drill-down only for one group.

- [ ] **Step 4: Update API and UI**

Add page/sort parameters and render loading, error, retry, empty, and paginated results without blocking on the entire snapshot.

- [ ] **Step 5: Verify Task 6 and commit**

Run focused tests and production-sized timing; commit with `perf: rank opportunities with bounded sql`.

### Task 7: Add latest official data and comprehensive quality gates

**Files:**
- Modify: `src/icor/evidence/source_inventory.py`
- Modify: `src/icor/evidence/sources/eea.py`
- Modify: pinned release manifests under `data/source-releases/`.
- Modify: `src/icor/evidence/validation.py`
- Modify: source acquisition/EEA/validation/integration tests.

**Interfaces:**
- Adds pinned EEA 2025 provisional release and validation checks for year relationships, status leakage, overlaps, and count drift.

- [ ] **Step 1: Write failing source and validation tests**

Assert 2025 provisional records retain provisional status, are superseded by a final release when present, and never enter final-only queries. Assert impossible future years fail while plausible 1914 vintage stock remains valid with explicit semantics.

- [ ] **Step 2: Verify RED**

Run focused EEA acquisition/source/validation tests.

- [ ] **Step 3: Pin and ingest EEA 2025 provisional**

Add the official manifest URL, publication timestamp, checksum, coverage, status, and table mapping. Keep the artifact immutable and fail closed on checksum/schema drift.

- [ ] **Step 4: Implement quality gates**

Validate year relationships, source overlap/dependency groups, observed-versus-estimated separation, generic/missing labels, and accepted/rejected/quarantined count drift. Emit inspectable audit results into the snapshot.

- [ ] **Step 5: Verify Task 7 and commit**

Run focused and clean-room integration tests; commit with `feat: add latest provisional evidence and quality gates`.

### Task 8: Full verification, candidate rebuild, promotion, and live smoke

**Files:**
- Modify: `docs/CODEX_HANDOFF.md`
- Modify documentation only if commands or semantics changed.

**Interfaces:**
- Produces one verified candidate snapshot, atomic promotion record, measured endpoint report, and durable handoff.

- [ ] **Step 1: Run complete local gates**

Run lock check, maintained Ruff, full pytest, Vitest, TypeScript, ESLint, production Vite build, OpenAPI compatibility, dependency audits, Playwright, and `git diff --check`. Record exact counts and failures.

- [ ] **Step 2: Build and validate a candidate**

Use the documented clean-room build with all pinned releases. Validate schema 5, checksums, quality audit, release membership, Golf 2024 family total/breakdown, timeline availability, and planner/opportunity totals.

- [ ] **Step 3: Measure performance**

Measure cold readiness and warm p95 for representative Registration, Evidence, Planner, and Opportunity requests. Do not promote if any endpoint deserializes an unbounded set or misses its target without a documented, approved revision.

- [ ] **Step 4: Promote atomically and run live checks**

Promote only the verified candidate, restart the authenticated private Codespaces preview, exercise all four workflows and logout, verify anonymous 401 behavior, and reopen the app for Lucas.

- [ ] **Step 5: Update durable handoff and commit**

Record exact commit, snapshot ID/SHA, releases, test counts, timings, live URL/state, rollback snapshot, and remaining owner actions. Commit with `docs: hand off stabilized ICOR preview`.
