# ICOR Opportunity Ranking and Production Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local opportunity-ranking page that reconciles configuration/model-year demand, explains an 80/20 demand-readiness score, and manages exact or fallback ICOR production coverage in SQLite.

**Architecture:** Extend the existing modular monolith with immutable opportunity domain types, application services, and a versioned SQLite coverage adapter. FastAPI remains a thin transport layer, while the React client renders typed results and refetches rankings only after committed mutations.

**Tech Stack:** Python 3.12, dataclasses, SQLite, FastAPI, Pydantic 2, pytest, React 19, TypeScript, TanStack Router/Query, Vitest, Testing Library, Playwright, axe-core.

**Spec:** `docs/superpowers/specs/2026-08-26-opportunity-ranking-production-coverage-design.md`

## Global Constraints

- Work only in `C:\Users\LucasCravoVERISSIMO\icor-webapp-development` on `development/windshield-demand-platform`; preserve unrelated `AGENTS.md` and handoff changes.
- Do not push, merge, deploy, use production/customer data, or modify the protected `main` checkout or Streamlit application.
- Demonstration forecasts remain deterministic, synthetic, and visibly labelled; production coverage is shared local prototype state and must never contain secrets.
- Exact configuration coverage takes precedence over vehicle-year fallback, and the same demand unit is never counted twice.
- Raw downside/base/upside demand is immutable; `demand_readiness_v1` allocates 0–80 demand points and 0–20 readiness points.
- The backend owns canonical matching, aggregation, reconciliation, coverage resolution, and scoring; React owns interaction and presentation only.
- Writes use bound SQL parameters and atomic transactions; unsupported/corrupt schema versions stop writes without deleting or rebuilding the database.
- API mutations return committed state; the UI refetches coverage and opportunities after success and preserves form state after failure.
- Notes reject control characters, are length-bounded, and receive no HTML interpretation.
- Maintain Python `>=3.12,<3.13`, Node 22+, strict TypeScript, existing problem responses/correlation IDs, local-only origins, and all four strict legacy XFAILs.

---

### Task 1: Add reconciled model-year demand to the canonical fixture

**Files:**
- Modify: `src/icor/domain/planner.py`
- Modify: `src/icor/application/planner.py`
- Modify: `src/icor/infrastructure/demo_planner_repository.py`
- Modify: `data/demo/planner-v1.json`
- Modify: `tests/domain/test_planner.py`
- Modify: `tests/infrastructure/test_demo_planner_repository.py`

**Interfaces:**
- Consumes: existing `PlanningConfiguration`, `DemandRange`, and `DemoPlannerRepository`.
- Produces: `ModelYearDemand`; `PlanningConfiguration.model_year_demand: tuple[ModelYearDemand, ...]`; `PlannerRepository.list_model_year_demand()`.

- [ ] **Step 1: Write failing invariant and fixture tests**

```python
def test_model_year_demand_reconciles_with_configuration(sample_configuration):
    assert sum(row.base_units for row in sample_configuration.model_year_demand) == (
        sample_configuration.demand.base_units
    )

def test_fixture_rejects_non_reconciling_model_year_demand(tmp_path):
    path = write_fixture(tmp_path, model_year_base_delta=1)
    with pytest.raises(FixtureError, match="invalid"):
        DemoPlannerRepository.from_path(path)
```

- [ ] **Step 2: Run RED verification**

Run: `uv run pytest tests/domain/test_planner.py tests/infrastructure/test_demo_planner_repository.py -v`

Expected: FAIL because `ModelYearDemand` and fixture model-year rows do not exist.

- [ ] **Step 3: Implement the immutable type and reconciliation rule**

```python
@dataclass(frozen=True, slots=True)
class ModelYearDemand:
    configuration_id: str
    model_year: int
    forecast_horizon: int
    demand: DemandRange
    evidence_status: EvidenceStatus
    data_version: str
    sources: tuple[SourceSummary, ...]
```

Require every year to fall within the configuration range; require unique `(configuration_id, model_year, forecast_horizon)` identities; and require downside, base, and upside sums to equal configuration totals exactly.

- [ ] **Step 4: Add deterministic synthetic model-year rows to all eight fixture configurations**

Use explicit integer rows in `planner-v1.json`; label their source description as a synthetic allocation for workflow demonstration. Do not calculate or present an equal split as observed evidence.

- [ ] **Step 5: Run GREEN verification and commit**

Run: `uv run pytest tests/domain/test_planner.py tests/infrastructure/test_demo_planner_repository.py -v`

Expected: PASS.

Run: `uv run ruff check src tests`

Expected: `All checks passed!`

Commit: `feat: add reconciled model-year demand`

---

### Task 2: Define production coverage and the ranking policy

**Files:**
- Create: `src/icor/domain/opportunities.py`
- Create: `src/icor/application/ranking.py`
- Create: `tests/domain/test_opportunities.py`
- Create: `tests/application/test_ranking.py`

**Interfaces:**
- Consumes: `ModelYearDemand`, `DemandRange`, and canonical configuration identities.
- Produces: `CoverageMatchType`, `CoverageStatus`, `ProductionCoverage`, `OpportunityCandidate`, `OpportunityScore`, `RankingStrategy`, and `DemandReadinessV1.score(candidates)`.

- [ ] **Step 1: Write failing coverage invariant tests**

```python
def test_exact_coverage_requires_configuration_id():
    with pytest.raises(ValueError, match="configuration"):
        make_coverage(match_type=CoverageMatchType.EXACT_CONFIGURATION, configuration_id=None)

def test_note_rejects_control_characters():
    with pytest.raises(ValueError, match="control"):
        make_coverage(note="line\x00break")
```

- [ ] **Step 2: Write failing ranking examples**

```python
def test_exact_coverage_receives_full_readiness_weight():
    result = DemandReadinessV1().score((candidate(base=100, exact=100),))[0]
    assert result.demand_points == 80
    assert result.readiness_points == 20
    assert result.total_points == 100

def test_fallback_receives_half_readiness_weight():
    result = DemandReadinessV1().score((candidate(base=100, fallback=100),))[0]
    assert result.readiness_points == 10
```

Add tests for average-rank ties, single non-zero candidate, all-zero candidates, mixed coverage, and stable identity tie-breaking.

- [ ] **Step 3: Run RED verification**

Run: `uv run pytest tests/domain/test_opportunities.py tests/application/test_ranking.py -v`

Expected: FAIL because the modules do not exist.

- [ ] **Step 4: Implement validated coverage value objects**

```python
class CoverageMatchType(StrEnum):
    EXACT_CONFIGURATION = "exact_configuration"
    VEHICLE_YEAR_FALLBACK = "vehicle_year_fallback"

@dataclass(frozen=True, slots=True)
class ProductionCoverage:
    coverage_id: str
    match_type: CoverageMatchType
    configuration_id: str | None
    brand: str
    model: str
    model_year: int
    sku: str | None
    note: str | None
    created_at: datetime
    updated_at: datetime
```

Normalize no free-form identity values. Require UTC timestamps, a non-empty stable ID, note length `<= 500`, no C0/C1 controls, exact configuration for exact matches, and `configuration_id is None` for fallbacks.

- [ ] **Step 5: Implement `demand_readiness_v1` behind a protocol**

```python
class RankingStrategy(Protocol):
    name: str
    version: str
    def score(self, candidates: tuple[OpportunityCandidate, ...]) -> tuple[OpportunityScore, ...]: ...
```

Calculate tie-aware average positional percentiles. Compute `demand_points = percentile * 80` and `readiness_points = ((exact + fallback * 0.5) / base * 20)` with zero readiness when base is zero. Preserve full precision internally and round the displayed total to one decimal only in API serialization.

- [ ] **Step 6: Run GREEN verification and commit**

Run: `uv run pytest tests/domain/test_opportunities.py tests/application/test_ranking.py -v`

Expected: PASS.

Commit: `feat: define opportunity ranking policy`

---

### Task 3: Persist production coverage transactionally in SQLite

**Files:**
- Create: `src/icor/application/coverage.py`
- Create: `src/icor/infrastructure/sqlite_coverage_repository.py`
- Create: `tests/infrastructure/test_sqlite_coverage_repository.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `ProductionCoverage` and `CoverageMatchType`.
- Produces: `CoverageRepository` protocol; `DuplicateCoverageError`; `CoverageSchemaError`; `SQLiteCoverageRepository(path)` with `list_all`, `get`, `create`, `update`, and `delete`.

- [ ] **Step 1: Write failing migration and CRUD tests**

```python
def test_repository_migrates_empty_database(tmp_path):
    repository = SQLiteCoverageRepository(tmp_path / "coverage.sqlite3")
    assert repository.schema_version == 1

def test_duplicate_exact_identity_rolls_back(repository, exact_coverage):
    repository.create(exact_coverage)
    with pytest.raises(DuplicateCoverageError):
        repository.create(replace(exact_coverage, coverage_id="coverage-2"))
    assert repository.list_all() == (exact_coverage,)
```

Cover fallback uniqueness, update timestamps, missing IDs, transaction rollback, foreign-key enablement, and refusal of schema version `> 1`.

- [ ] **Step 2: Run RED verification**

Run: `uv run pytest tests/infrastructure/test_sqlite_coverage_repository.py -v`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement explicit schema version 1 and repository methods**

Create `schema_version(version INTEGER NOT NULL)` and `production_coverage` with a match-type check, immutable `coverage_id`, canonical fields, UTC ISO timestamps, and partial unique indexes for exact and fallback identities. Open connections with `PRAGMA foreign_keys = ON`; use `with connection:` transactions and `?` parameters only.

- [ ] **Step 4: Configure the ignored local database path**

Add `.local/*.sqlite3` to `.gitignore`. The app factory will later read `ICOR_COVERAGE_DB`, defaulting to `ROOT / ".local" / "production-coverage.sqlite3"`; do not create the default database during import.

- [ ] **Step 5: Run GREEN verification and commit**

Run: `uv run pytest tests/infrastructure/test_sqlite_coverage_repository.py -v`

Expected: PASS.

Commit: `feat: persist production coverage in sqlite`

---

### Task 4: Add canonical coverage and opportunity application services

**Files:**
- Create: `src/icor/application/opportunities.py`
- Modify: `src/icor/application/coverage.py`
- Create: `tests/application/test_opportunities.py`
- Create: `tests/application/test_coverage.py`

**Interfaces:**
- Consumes: planner repository, coverage repository, and `RankingStrategy`.
- Produces: `OpportunityGroupBy`, `OpportunityQuery`, `OpportunityResult`, `OpportunityService.list`, `OpportunityService.drill_down`, and `ProductionCoverageService` CRUD commands.

- [ ] **Step 1: Write failing aggregation and precedence tests**

```python
def test_exact_match_overrides_fallback_without_double_counting(service):
    row = service.list(OpportunityQuery(group_by=OpportunityGroupBy.MODEL_YEAR)).items[0]
    assert row.exact_covered_base_units + row.fallback_covered_base_units + row.uncovered_base_units == row.demand.base_units

def test_model_year_groups_reconcile_to_configuration_rows(service):
    result = service.list(OpportunityQuery(group_by=OpportunityGroupBy.MODEL_YEAR))
    assert sum(row.demand.base_units for row in result.items) == result.summary.base_units
```

Cover brand/model/model-year grouping, market/horizon filters, coverage statuses, top-quartile uncovered summary, stable sorting, and stale configuration integrity errors.

- [ ] **Step 2: Write failing canonical command tests**

```python
def test_exact_create_resolves_canonical_sku(service):
    saved = service.create(CreateCoverageCommand.exact("cfg-1", 2024, note=None))
    assert saved.sku == "DEMO-SKU-001"

def test_fallback_requires_explicit_match_type(service):
    with pytest.raises(CanonicalCoverageError):
        service.create(CreateCoverageCommand.fallback("Unknown", "Model", 2024, None))
```

- [ ] **Step 3: Run RED verification**

Run: `uv run pytest tests/application/test_opportunities.py tests/application/test_coverage.py -v`

Expected: FAIL because the services do not exist.

- [ ] **Step 4: Implement reconciliation, aggregation, and commands**

Resolve coverage per model-year demand row: exact `(configuration_id, model_year)` first, then fallback `(brand, model, model_year)`. Aggregate immutable demand and coverage buckets, derive `exact_covered`, `fallback_only`, `mixed`, or `uncovered`, invoke the ranking strategy on the complete filtered candidate set, and calculate the `>= 0.75` tie-aware high-demand summary.

- [ ] **Step 5: Run GREEN verification and commit**

Run: `uv run pytest tests/application/test_opportunities.py tests/application/test_coverage.py -v`

Expected: PASS.

Commit: `feat: add opportunity and coverage services`

---

### Task 5: Expose versioned opportunity and coverage APIs

**Files:**
- Create: `src/icor/api/opportunities.py`
- Modify: `src/icor/api/schemas.py`
- Modify: `src/icor/api/app.py`
- Modify: `tests/api/test_planner_api.py`
- Create: `tests/api/test_opportunity_api.py`
- Modify: `web/openapi.json`
- Modify: `web/src/lib/api/schema.ts`

**Interfaces:**
- Consumes: both application services and existing `ProblemResponse`.
- Produces: all six endpoints from the spec, typed request/response schemas, and regenerated OpenAPI types.

- [ ] **Step 1: Write failing endpoint contract tests**

```python
def test_create_exact_coverage_and_refresh_opportunity(client):
    created = client.post("/api/v1/production-coverage", json=exact_payload)
    assert created.status_code == 201
    ranked = client.get("/api/v1/opportunities?group_by=model_year")
    assert ranked.status_code == 200
    assert any(row["exact_covered_base_units"] > 0 for row in ranked.json()["items"])

def test_duplicate_coverage_returns_problem_409(client):
    client.post("/api/v1/production-coverage", json=exact_payload)
    response = client.post("/api/v1/production-coverage", json=exact_payload)
    assert response.status_code == 409
    assert response.json()["code"] == "duplicate_coverage"
```

Cover group validation, drill-down 404, canonical 422, update/delete 404, safe persistence 500, and correlation IDs.

- [ ] **Step 2: Run RED verification**

Run: `uv run pytest tests/api/test_opportunity_api.py -v`

Expected: FAIL with 404 routes.

- [ ] **Step 3: Add schemas, thin routes, and app wiring**

Define `ProductionCoverageCreateRequest`, `ProductionCoverageUpdateRequest`, `ProductionCoverageResponse`, `OpportunityRowResponse`, `OpportunityPageResponse`, and `DeleteCoverageResponse`. Inject an optional coverage repository into `create_app` for isolated tests. Permit `GET, POST, PUT, DELETE, OPTIONS` in local CORS.

- [ ] **Step 4: Regenerate and check the OpenAPI contract**

Run: `cd web; npm run openapi:generate`

Expected: `openapi.json` and `src/lib/api/schema.ts` include all six endpoints.

- [ ] **Step 5: Run GREEN verification and commit**

Run: `uv run pytest tests/api/test_planner_api.py tests/api/test_opportunity_api.py -v`

Expected: PASS.

Run: `cd web; npm run openapi:check`

Expected: no generated drift.

Commit: `feat: expose opportunity and coverage api`

---

### Task 6: Add typed client, query keys, navigation, and route state

**Files:**
- Modify: `web/src/lib/api/client.ts`
- Create: `web/src/lib/opportunity-search.ts`
- Modify: `web/src/app/query-client.ts`
- Modify: `web/src/app/router.tsx`
- Modify: `web/src/app/AppShell.tsx`
- Modify: `web/tests/api-client.test.ts`
- Create: `web/tests/opportunity-search.test.ts`
- Modify: `web/tests/router.test.tsx`
- Modify: `web/tests/app-shell.test.tsx`

**Interfaces:**
- Consumes: generated OpenAPI types.
- Produces: `opportunities`, `opportunityConfigurations`, `coverage`, `createCoverage`, `updateCoverage`, `deleteCoverage`; validated `groupBy`; and `/opportunities` route/navigation.

- [ ] **Step 1: Write failing request and route tests**

```typescript
it('sends explicit fallback coverage as JSON', async () => {
  await client.createCoverage({ match_type: 'vehicle_year_fallback', brand: 'A', model: 'B', model_year: 2024, configuration_id: null, note: null })
  expect(fetcher).toHaveBeenCalledWith('/api/v1/production-coverage', expect.objectContaining({ method: 'POST' }))
})

it('normalizes an invalid grouping to brand', () => {
  expect(parseOpportunitySearch({ groupBy: 'profit' }).value.groupBy).toBe('brand')
})
```

- [ ] **Step 2: Run RED verification**

Run: `cd web; npm test -- --run tests/api-client.test.ts tests/opportunity-search.test.ts tests/router.test.tsx tests/app-shell.test.tsx`

Expected: FAIL because client methods and route do not exist.

- [ ] **Step 3: Implement typed requests and mutation-safe query keys**

Extend the shared request helper to accept method/body, set `Content-Type: application/json` only when a body exists, and preserve `ApiProblem`. Centralize keys such as `['opportunities', normalizedQuery]` and `['production-coverage']`; do not mutate cached ranking rows optimistically.

- [ ] **Step 4: Add route and accessible active navigation**

Add `/opportunities` with `groupBy: 'brand' | 'model' | 'model_year'`. Refactor `PlannerLink` into a route-aware navigation item and expose both Planner workbench and Opportunities in desktop/mobile navigation.

- [ ] **Step 5: Run GREEN verification and commit**

Run: `cd web; npm test -- --run tests/api-client.test.ts tests/opportunity-search.test.ts tests/router.test.tsx tests/app-shell.test.tsx`

Expected: PASS.

Commit: `feat: connect opportunity web route`

---

### Task 7: Build ranking, drill-down, and production coverage management UI

**Files:**
- Create: `web/src/features/opportunities/OpportunitiesPage.tsx`
- Create: `web/src/features/opportunities/OpportunityRanking.tsx`
- Create: `web/src/features/opportunities/OpportunityDrillDown.tsx`
- Create: `web/src/features/opportunities/CoverageManager.tsx`
- Modify: `web/src/app/styles.css`
- Create: `web/tests/opportunities-page.test.tsx`
- Create: `web/tests/coverage-manager.test.tsx`

**Interfaces:**
- Consumes: typed client methods and TanStack Query keys from Task 6.
- Produces: polished responsive `/opportunities`, brand/model/model-year switch, ranked rows, configuration drill-down, and exact/fallback coverage CRUD.

- [ ] **Step 1: Write failing ranking presentation tests**

```typescript
it('keeps raw base demand more prominent than the score', async () => {
  renderOpportunityPage()
  expect(await screen.findByText('12,400 replacements')).toHaveClass('opportunity-demand__base')
  expect(screen.getByText('Score 86.5')).toHaveAccessibleDescription(/80% demand.*20% readiness/i)
})

it('switches grouping without losing filters', async () => {
  await user.click(await screen.findByRole('button', { name: 'Model years' }))
  expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ search: expect.objectContaining({ groupBy: 'model_year' }) }))
})
```

- [ ] **Step 2: Write failing coverage workflow tests**

```typescript
it('requires deliberate confirmation before fallback creation', async () => {
  renderCoverageManager()
  await user.click(screen.getByLabelText('Exact configuration unknown'))
  expect(screen.getByRole('button', { name: 'Save fallback coverage' })).toBeDisabled()
  await user.click(screen.getByLabelText(/I understand this is lower precision/i))
  expect(screen.getByRole('button', { name: 'Save fallback coverage' })).toBeEnabled()
})

it('retains form values and reports correlation id after failure', async () => {
  server.use(failingMutation('corr-123'))
  await submitExactCoverage()
  expect(screen.getByDisplayValue('2024')).toBeInTheDocument()
  expect(screen.getByText(/corr-123/)).toBeInTheDocument()
})
```

Also cover create/edit/delete confirmation, success announcements, refetch only after success, loading/empty/retry states, and canonical cascading selectors.

- [ ] **Step 3: Run RED verification**

Run: `cd web; npm test -- --run tests/opportunities-page.test.tsx tests/coverage-manager.test.tsx`

Expected: FAIL because the components do not exist.

- [ ] **Step 4: Implement the page in focused components**

Render summary cards for total base demand, exact-covered demand, and uncovered high-demand units. Use native segmented buttons, semantic list/table structures, visible textual coverage composition, score component explanation, and an `aria-live` mutation status. Exact selection is the default; reveal fallback only after the explicit unknown choice and confirmation.

- [ ] **Step 5: Add responsive and reduced-motion styles**

Reuse existing tokens. Allow ranking and manager columns only when both remain useful; stack in task order on tablet/mobile. Ensure controls wrap, rows do not create page-level horizontal overflow, focus is visible, and semantic state never relies on color alone.

- [ ] **Step 6: Run GREEN verification and commit**

Run: `cd web; npm test -- --run tests/opportunities-page.test.tsx tests/coverage-manager.test.tsx`

Expected: PASS.

Run: `cd web; npm run lint; npm run build`

Expected: PASS.

Commit: `feat: build opportunity ranking workbench`

---

### Task 8: Verify browser workflows, operations, and durable handoff

**Files:**
- Create: `web/e2e/opportunities.spec.ts`
- Modify: `scripts/run_planner_dev.py`
- Modify: `README.md`
- Modify: `docs/LOCAL_DEVELOPMENT.md`
- Modify: `docs/CODEX_HANDOFF.md`

**Interfaces:**
- Consumes: the complete API and web slice.
- Produces: isolated browser coverage, documented database behavior, fresh verification evidence, and a current durable checkpoint.

- [ ] **Step 1: Write browser tests for committed ranking changes**

```typescript
test('exact, fallback, edit, and delete reconcile ranking', async ({ page }) => {
  await page.goto('/opportunities')
  await addExactCoverage(page)
  await expect(page.getByText(/Exact-covered/)).toBeVisible()
  await editCoverage(page)
  await deleteCoverage(page)
  await addFallbackCoverage(page)
  await expect(page.getByText(/lower precision/i)).toBeVisible()
})
```

Use a temporary `ICOR_COVERAGE_DB` per Playwright run. Add desktop and mobile drill-down, keyboard navigation, axe, and no-horizontal-overflow checks.

- [ ] **Step 2: Run browser RED then GREEN**

Run: `cd web; npm run e2e -- opportunities.spec.ts`

Expected initially: FAIL before fixtures/helpers are wired; after implementation: PASS.

- [ ] **Step 3: Document local state and real-data boundary**

Document `ICOR_COVERAGE_DB`, shared-local/no-auth limitations, safe reset by moving the exact ignored database file only when the user requests it, and the fact that forecast/fitment data remains demonstration evidence until a separately validated ingestion contract is approved.

- [ ] **Step 4: Run the complete fresh quality gate**

Run: `uv lock --check`

Run: `uv run ruff check src tests scripts/audit_baseline.py scripts/run_planner_dev.py`

Run: `uv run pytest -p no:cacheprovider`

Run: `uv run pip-audit`

Run: `cd web; npm run openapi:check; npm test -- --run; npm run lint; npm run build; npm run e2e; npm audit --audit-level=high`

Run: `git diff --check`

Expected: all commands exit 0; pytest reports only the four documented strict XFAILs; both audits report no known vulnerabilities.

- [ ] **Step 5: Perform live desktop/mobile visual QA**

Start the local launcher on unused local ports with an ignored coverage database. Inspect `/opportunities` at desktop and mobile widths, brand/model/model-year switching, drill-down, exact CRUD, fallback warning, focus order, errors, empty states, and overflow. Record exact ports/PID and screenshot locations in the handoff without secrets.

- [ ] **Step 6: Update durable handoff and commit**

Record files changed, actual command outputs, unresolved real-data needs, production isolation, and server status in `docs/CODEX_HANDOFF.md` while preserving concurrent content.

Commit: `test: verify opportunity ranking workflows`

---

## Plan self-review

- Spec coverage: every domain, persistence, API, UI, error, operational, and acceptance requirement maps to Tasks 1–8.
- Placeholder scan: no unfinished markers or vague implementation instructions remain.
- Type consistency: `ModelYearDemand`, `ProductionCoverage`, `CoverageRepository`, `RankingStrategy`, application services, response schemas, generated client methods, and UI query keys flow in dependency order.
- Scope remains one coherent product slice; proprietary ingestion and calibrated forecasting remain a separate architectural subproject because their data contracts and validation evidence are not yet available.
