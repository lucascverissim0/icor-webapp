# ICOR Planner Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a responsive, navigable planner workbench backed by a versioned FastAPI contract and deterministic demonstration data for windshield-configuration demand planning.

**Architecture:** Add import-safe Python domain, application, infrastructure, and FastAPI adapters under `src/icor`, plus a separate Vite/React/TypeScript adapter under `web`. The backend owns business semantics, filtering, sorting, pagination, confidence, and summaries; the frontend owns accessible presentation and typed URL state while consuming a checked OpenAPI contract.

**Tech Stack:** Python 3.12, FastAPI, Pydantic 2, pytest, React 19, TypeScript strict mode, Vite, TanStack Router, TanStack Query, Tailwind CSS, Vitest, Testing Library, Playwright, axe-core.

**Spec:** `docs/superpowers/specs/2026-08-25-planner-webapp-design.md`

## Global Constraints

- Work only in `C:\Users\LucasCravoVERISSIMO\icor-webapp-development` on `development/windshield-demand-platform`.
- Treat `C:\Users\LucasCravoVERISSIMO\icor-webapp`, branch `main`, the deployed Streamlit app, and production data as read-only.
- Do not push, merge, deploy, access production services/data, or require secrets.
- Python remains `>=3.12,<3.13`; frontend tooling must run on Node 22 or newer.
- The fixture is deterministic, synthetic, visibly labelled `demonstration`, and contains no customer or proprietary catalog facts.
- Unknown fitment facts remain `null`/unknown and are never coerced to `false` or zero.
- The UI never computes demand, confidence, or compatibility business rules; API routes never read fixture or legacy script files directly.
- API errors use a safe problem shape with a correlation ID and never expose stack traces or environment values.
- Preserve the four strict legacy XFAILs and keep `ui/` and `scripts/` behaviorally unchanged.
- Tests prohibit unexpected network access; local API CORS permits only the configured local Vite origin.
- Every displayed demand value must be accompanied by evidence status; color is never the only semantic carrier.

---

### Task 1: Define the canonical planner domain

**Files:**
- Create: `src/icor/domain/__init__.py`
- Create: `src/icor/domain/planner.py`
- Create: `tests/domain/test_planner.py`

**Interfaces:**
- Consumes: no adapter or framework types.
- Produces: `EvidenceStatus`, `ConfidenceLevel`, `SortField`, `SortDirection`, `Equipment`, `DemandRange`, `Confidence`, `SourceSummary`, `PlanningConfiguration`, `PlannerQuery`, `PlannerPage`, and pure `filter_sort_paginate(records, query)`.

- [ ] **Step 1: Write failing value-semantics and query tests**

```python
def test_unknown_equipment_remains_unknown(sample_configuration):
    assert sample_configuration.equipment.hud is None

def test_demand_range_requires_ordered_non_negative_units():
    with pytest.raises(ValueError, match="downside <= base <= upside"):
        DemandRange(downside_units=20, base_units=10, upside_units=30)

def test_filter_sort_paginate_is_deterministic(sample_configurations):
    query = PlannerQuery(market="FR", brands=("Renault",), page=1, page_size=1)
    result = filter_sort_paginate(sample_configurations, query)
    assert result.total == 2
    assert result.items[0].demand.base_units >= result.items[-1].demand.base_units
```

- [ ] **Step 2: Run the domain tests to verify RED**

Run: `uv run pytest tests/domain/test_planner.py -v`

Expected: FAIL because `icor.domain.planner` does not exist.

- [ ] **Step 3: Implement frozen domain value objects and pure query logic**

Use frozen dataclasses and string enums. `DemandRange.__post_init__` validates integer units and `0 <= downside <= base <= upside`; year ranges and horizon validate ascending values; `PlannerQuery` validates `page >= 1` and `1 <= page_size <= 100`. Filtering is exact and case-sensitive over canonical values, default sorting is `(base_units DESC, configuration_id ASC)`, and pagination never mutates inputs.

```python
class EvidenceStatus(StrEnum):
    DEMONSTRATION = "demonstration"
    PROTOTYPE = "prototype"
    VALIDATED = "validated"

@dataclass(frozen=True, slots=True)
class Equipment:
    camera_adas: bool | None
    hud: bool | None
    heated: bool | None
    acoustic: bool | None
    rain_light_sensor: bool | None

@dataclass(frozen=True, slots=True)
class PlannerQuery:
    markets: tuple[str, ...] = ()
    horizons: tuple[int, ...] = ()
    brands: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    evidence: tuple[EvidenceStatus, ...] = ()
    sort: SortField = SortField.BASE_DEMAND
    direction: SortDirection = SortDirection.DESC
    page: int = 1
    page_size: int = 25
```

- [ ] **Step 4: Run focused and full Python verification**

Run: `uv run pytest tests/domain/test_planner.py -v`

Expected: PASS.

Run: `uv run ruff check src tests`

Expected: `All checks passed!`

- [ ] **Step 5: Commit the domain boundary**

```powershell
git add src/icor/domain tests/domain
git commit -m "feat: define planner domain contract"
```

---

### Task 2: Add the application service and repository protocol

**Files:**
- Create: `src/icor/application/__init__.py`
- Create: `src/icor/application/planner.py`
- Create: `tests/application/test_planner.py`

**Interfaces:**
- Consumes: Task 1 domain records and `PlannerQuery`.
- Produces: `PlannerRepository.list_all() -> tuple[PlanningConfiguration, ...]`, `PlannerRepository.get(configuration_id)`, `PlannerService.options()`, `PlannerService.search(query)`, and `PlannerService.detail(configuration_id)`.

- [ ] **Step 1: Write failing adapter-independent service tests**

```python
class FakeRepository:
    def __init__(self, records): self.records = tuple(records)
    def list_all(self): return self.records
    def get(self, configuration_id):
        return next((row for row in self.records if row.configuration_id == configuration_id), None)

def test_options_are_unique_sorted_and_include_scenario(sample_configurations):
    options = PlannerService(FakeRepository(sample_configurations)).options()
    assert options.markets == ("DE", "FR")
    assert options.evidence_status is EvidenceStatus.DEMONSTRATION

def test_missing_detail_returns_none(sample_configurations):
    assert PlannerService(FakeRepository(sample_configurations)).detail("missing") is None
```

- [ ] **Step 2: Run the service tests to verify RED**

Run: `uv run pytest tests/application/test_planner.py -v`

Expected: FAIL because the application service does not exist.

- [ ] **Step 3: Implement the protocol and thin use cases**

```python
class PlannerRepository(Protocol):
    def list_all(self) -> tuple[PlanningConfiguration, ...]: ...
    def get(self, configuration_id: str) -> PlanningConfiguration | None: ...

class PlannerService:
    def __init__(self, repository: PlannerRepository) -> None:
        self._repository = repository

    def search(self, query: PlannerQuery) -> PlannerPage:
        return filter_sort_paginate(self._repository.list_all(), query)
```

`options()` derives sorted canonical values and scenario metadata from repository records. `detail()` delegates identity lookup without knowledge of files or FastAPI.

- [ ] **Step 4: Run service tests and lint**

Run: `uv run pytest tests/application/test_planner.py -v`

Expected: PASS.

Run: `uv run ruff check src tests`

Expected: `All checks passed!`

- [ ] **Step 5: Commit the application boundary**

```powershell
git add src/icor/application tests/application
git commit -m "feat: add planner application service"
```

---

### Task 3: Load deterministic demonstration data through a read-only adapter

**Files:**
- Create: `data/demo/planner-v1.json`
- Create: `src/icor/infrastructure/__init__.py`
- Create: `src/icor/infrastructure/demo_planner_repository.py`
- Create: `tests/infrastructure/test_demo_planner_repository.py`

**Interfaces:**
- Consumes: Task 1 domain types and Task 2 `PlannerRepository` protocol.
- Produces: `DemoPlannerRepository.from_path(path: Path)`, immutable record tuples, `data_version == "demo-planner-v1"`, and typed `FixtureError` on invalid fixtures.

- [ ] **Step 1: Write failing fixture integrity and unknown-value tests**

```python
def test_fixture_is_deterministic_and_demonstration_only(demo_repository):
    rows = demo_repository.list_all()
    assert len(rows) >= 8
    assert len({row.configuration_id for row in rows}) == len(rows)
    assert {row.evidence_status for row in rows} == {EvidenceStatus.DEMONSTRATION}

def test_fixture_preserves_unknown_fitment(demo_repository):
    assert any(row.equipment.hud is None for row in demo_repository.list_all())

def test_repository_never_writes_fixture(demo_fixture_path, demo_repository):
    before = demo_fixture_path.read_bytes()
    demo_repository.list_all()
    assert demo_fixture_path.read_bytes() == before
```

- [ ] **Step 2: Run repository tests to verify RED**

Run: `uv run pytest tests/infrastructure/test_demo_planner_repository.py -v`

Expected: FAIL because the fixture adapter does not exist.

- [ ] **Step 3: Add the versioned synthetic fixture**

Create at least eight clearly fictional planning records spanning France and Germany, 2028 and 2030 horizons, and multiple brands/models/configurations. Each record includes ordered demand ranges, exposure units, a replacement-rate assumption, confidence reasons, source summaries labelled synthetic, ISO `updated_at`, and nullable equipment flags. Use identifiers prefixed `demo-`; do not copy legacy or proprietary fitment values.

- [ ] **Step 4: Implement strict read-only parsing**

`from_path()` reads once with UTF-8, validates fixture version and required keys, rejects duplicates, maps JSON `null` directly to Python `None`, and wraps JSON/type/value failures in `FixtureError("Demonstration planner fixture is invalid")` without echoing fixture content.

- [ ] **Step 5: Run focused tests, full tests, and lint**

Run: `uv run pytest tests/infrastructure/test_demo_planner_repository.py -v`

Expected: PASS.

Run: `uv run pytest -p no:cacheprovider`

Expected: all existing tests pass with exactly the four documented strict XFAILs.

Run: `uv run ruff check src tests`

Expected: `All checks passed!`

- [ ] **Step 6: Commit the data adapter**

```powershell
git add data/demo src/icor/infrastructure tests/infrastructure
git commit -m "feat: add deterministic planner demo repository"
```

---

### Task 4: Expose the versioned FastAPI contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `requirements.txt`
- Create: `src/icor/api/__init__.py`
- Create: `src/icor/api/schemas.py`
- Create: `src/icor/api/planner.py`
- Create: `src/icor/api/app.py`
- Create: `tests/api/test_planner_api.py`

**Interfaces:**
- Consumes: Task 2 `PlannerService` and Task 3 repository.
- Produces: `create_app(repository=None) -> FastAPI`; `GET /api/health`, `/api/v1/planner/options`, `/api/v1/planner/configurations`, and `/api/v1/planner/configurations/{configuration_id}`; OpenAPI 3.1 schema; `ProblemResponse`.

- [ ] **Step 1: Write failing API contract tests**

```python
def test_health_reports_fixture_readiness(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "fixture_ready": True, "data_version": "demo-planner-v1"}

def test_invalid_pagination_uses_safe_problem_shape(client):
    response = client.get("/api/v1/planner/configurations?page_size=101")
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_request"
    assert body["correlation_id"]
    assert "traceback" not in response.text.lower()

def test_missing_configuration_is_typed_404(client):
    response = client.get("/api/v1/planner/configurations/missing")
    assert response.status_code == 404
    assert response.json()["code"] == "configuration_not_found"
```

- [ ] **Step 2: Run API tests to verify RED**

Run: `uv run pytest tests/api/test_planner_api.py -v`

Expected: FAIL because FastAPI and the API package are absent.

- [ ] **Step 3: Add bounded API dependencies**

Add `fastapi>=0.116,<1` and `uvicorn[standard]>=0.35,<1` to project dependencies, plus `httpx>=0.28,<1` to the dev group. Run `uv lock`, `uv sync --locked`, and regenerate the exact `requirements.txt` using the same documented foundation export command.

- [ ] **Step 4: Implement schemas, routes, and app factory**

Pydantic models use `ConfigDict(extra="forbid")`, string enums, explicit `_units` fields, ISO datetimes, and nullable equipment booleans. Query parameters enforce pages and sort enums before service invocation. A request-ID middleware accepts no caller-provided ID, creates `uuid4().hex`, sets `X-Correlation-ID`, and exception handlers serialize only:

```python
class ProblemResponse(BaseModel):
    code: str
    message: str
    correlation_id: str
    field_errors: list[FieldError] = []
```

`create_app()` defaults to the repository-root `data/demo/planner-v1.json`, allows CORS only from `ICOR_WEB_ORIGIN` defaulting to `http://127.0.0.1:5173`, and creates no network clients.

- [ ] **Step 5: Verify API tests, OpenAPI, full Python suite, and lock**

Run: `uv run pytest tests/api/test_planner_api.py -v`

Expected: PASS.

Run: `uv run python -c "from icor.api.app import create_app; assert create_app().openapi()['openapi'].startswith('3.1.')"`

Expected: exit 0.

Run: `uv lock --check`

Expected: lock resolves without changes.

Run: `uv run pytest -p no:cacheprovider`

Expected: all tests pass with only the four documented XFAILs.

- [ ] **Step 6: Commit the API adapter**

```powershell
git add pyproject.toml uv.lock requirements.txt src/icor/api tests/api
git commit -m "feat: expose planner FastAPI contract"
```

---

### Task 5: Scaffold the strict React adapter and checked API client

**Files:**
- Create: `web/package.json`
- Create: `web/package-lock.json`
- Create: `web/tsconfig.json`
- Create: `web/vite.config.ts`
- Create: `web/vitest.config.ts`
- Create: `web/eslint.config.js`
- Create: `web/index.html`
- Create: `web/src/main.tsx`
- Create: `web/src/app/providers.tsx`
- Create: `web/src/lib/api/schema.ts`
- Create: `web/src/lib/api/client.ts`
- Create: `web/scripts/export-openapi.py`
- Create: `web/tests/api-client.test.ts`

**Interfaces:**
- Consumes: Task 4 OpenAPI endpoint/schema.
- Produces: `PlannerApiClient`, generated `schema.ts`, `QueryClientProvider`, Vite `/api` proxy, and commands `dev`, `build`, `typecheck`, `lint`, `test`, `openapi:check`.

- [ ] **Step 1: Write failing client and schema-drift tests**

```typescript
it('serializes repeatable filters without inventing values', async () => {
  const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify(successPage), { status: 200 }))
  await new PlannerApiClient(fetcher).configurations({ markets: ['FR'], brands: ['Renault'], page: 1 })
  expect(fetcher.mock.calls[0][0]).toContain('markets=FR')
  expect(fetcher.mock.calls[0][0]).toContain('brands=Renault')
})

it('throws a typed ApiProblem for non-success responses', async () => {
  const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify(problem), { status: 422 }))
  await expect(new PlannerApiClient(fetcher).configurations({ page: 0 })).rejects.toMatchObject({ code: 'invalid_request' })
})
```

- [ ] **Step 2: Create the Vite workspace and install exact lockfile dependencies**

Use npm with React, React DOM, TanStack Router, TanStack Query, Tailwind CSS, and Lucide React as runtime dependencies; add TypeScript, Vite, React plugin, Vitest, jsdom, Testing Library, user-event, ESLint, TypeScript ESLint, Playwright, axe-core, and an OpenAPI TypeScript generator as dev dependencies. Commit `package-lock.json`; do not use floating CDN assets.

- [ ] **Step 3: Generate and check the typed contract**

`export-openapi.py` imports `create_app()`, writes stable JSON with sorted keys to `web/openapi.json`, and requires no running server. `npm run openapi:generate` creates `src/lib/api/schema.ts`; `npm run openapi:check` regenerates both and fails on `git diff --exit-code -- web/openapi.json web/src/lib/api/schema.ts`.

- [ ] **Step 4: Implement the injectable API client and providers**

`PlannerApiClient` accepts a `typeof fetch`, builds `URLSearchParams` from normalized typed inputs, validates HTTP status, and throws `ApiProblem` without exposing response internals. Query defaults retain prior data during refetch, retry server failures once, and never retry 4xx validation/not-found responses.

- [ ] **Step 5: Verify frontend unit, type, lint, build, and schema checks**

Run: `npm test -- --run`

Expected: PASS.

Run: `npm run typecheck; npm run lint; npm run build; npm run openapi:check`

Expected: all exit 0 and production assets are emitted under ignored `web/dist/`.

- [ ] **Step 6: Commit the frontend foundation**

```powershell
git add web .gitignore
git commit -m "feat: scaffold typed planner web adapter"
```

---

### Task 6: Build typed URL state and the responsive application shell

**Files:**
- Create: `web/src/app/router.tsx`
- Create: `web/src/app/AppShell.tsx`
- Create: `web/src/app/ErrorBoundary.tsx`
- Create: `web/src/app/styles.css`
- Create: `web/src/lib/planner-search.ts`
- Create: `web/src/components/EvidenceBadge.tsx`
- Create: `web/tests/planner-search.test.ts`
- Create: `web/tests/app-shell.test.tsx`

**Interfaces:**
- Consumes: Task 5 providers and API schema enums.
- Produces: `PlannerSearch`, `parsePlannerSearch(raw)`, `serializePlannerSearch(value)`, `/planner` and `/planner/configurations/$configurationId` routes, persistent demonstration badge, desktop rail, and labelled mobile menu.

- [ ] **Step 1: Write failing URL normalization and shell accessibility tests**

```typescript
it('reports and removes obsolete URL filter values', () => {
  expect(parsePlannerSearch({ market: 'XX', page: '-4' }, validOptions)).toEqual({
    value: { page: 1, sort: 'base_demand', direction: 'desc' },
    invalidKeys: ['market', 'page'],
  })
})

it('always labels demonstration evidence in the shell', () => {
  render(<AppShell />)
  expect(screen.getByText('Demonstration data')).toBeVisible()
  expect(screen.getByRole('navigation', { name: 'Primary' })).toBeInTheDocument()
})
```

- [ ] **Step 2: Run focused tests to verify RED**

Run: `npm test -- --run web/tests/planner-search.test.ts web/tests/app-shell.test.tsx`

Expected: FAIL because the router and shell do not exist.

- [ ] **Step 3: Implement canonical URL parsing and routes**

Search parsing accepts only option values returned by the API, deduplicates arrays, normalizes invalid page/sort/direction to documented defaults, and returns `invalidKeys` for an inline recovery notice. Detail navigation retains the entire normalized search object; browser back/forward restores it.

- [ ] **Step 4: Implement the industrial shell and responsive tokens**

Use CSS custom properties for neutral surfaces, a single ICOR teal accent, readable semantic status colors, 44px minimum interactive targets on touch layouts, visible `:focus-visible`, and no hover-only action. The desktop rail collapses into a labelled disclosure menu below 768px. Render the evidence badge in the shell, not individual routes, so it cannot disappear.

- [ ] **Step 5: Verify focused tests, axe smoke test, typecheck, and build**

Run: `npm test -- --run; npm run typecheck; npm run build`

Expected: PASS.

- [ ] **Step 6: Commit shell and URL state**

```powershell
git add web/src web/tests
git commit -m "feat: add responsive planner shell and URL state"
```

---

### Task 7: Implement the planner workbench and all recoverable states

**Files:**
- Create: `web/src/features/planner/api.ts`
- Create: `web/src/features/planner/view-model.ts`
- Create: `web/src/features/planner/PlannerPage.tsx`
- Create: `web/src/features/planner/PlannerFilters.tsx`
- Create: `web/src/features/planner/SummaryMetrics.tsx`
- Create: `web/src/features/planner/ConfigurationTable.tsx`
- Create: `web/src/features/planner/ConfigurationCards.tsx`
- Create: `web/src/features/planner/PlannerStates.tsx`
- Create: `web/tests/planner-workbench.test.tsx`

**Interfaces:**
- Consumes: Task 5 client/schema and Task 6 typed search/navigation.
- Produces: query hooks, formatted comparison view models, filter reset/apply, sortable rows/cards, selection navigation, loading/empty/invalid-filter/API-error/retry states.

- [ ] **Step 1: Write failing interaction and state tests with mocked HTTP**

```typescript
it('shows demand range beside evidence and confidence', async () => {
  renderPlanner({ response: successPage })
  expect(await screen.findByText('1,240 units')).toBeVisible()
  expect(screen.getByText('980–1,510 units')).toBeVisible()
  expect(screen.getByText('Demonstration')).toBeVisible()
  expect(screen.getByText('Identity confidence: Medium')).toBeVisible()
})

it('keeps filters while retrying a server error', async () => {
  const user = userEvent.setup()
  renderPlanner({ responses: [serverProblem, successPage], search: { market: ['FR'] } })
  await user.click(await screen.findByRole('button', { name: 'Retry' }))
  expect(await screen.findByRole('checkbox', { name: 'France' })).toBeChecked()
})

it('empty results name active constraints and offer reset', async () => {
  renderPlanner({ response: emptyPage, search: { brand: ['Renault'] } })
  expect(await screen.findByText(/Renault/)).toBeVisible()
  expect(screen.getByRole('button', { name: 'Reset filters' })).toBeVisible()
})
```

- [ ] **Step 2: Run workbench tests to verify RED**

Run: `npm test -- --run web/tests/planner-workbench.test.tsx`

Expected: FAIL because planner components do not exist.

- [ ] **Step 3: Implement query hooks and presentation-only view models**

Query keys contain the full normalized URL query. View models format locale-safe integers with explicit `units`, map `null` equipment to `Unknown`, and combine labels without calculating demand or confidence.

- [ ] **Step 4: Implement wide table, narrow cards, filters, summaries, and states**

At 1024px and above render a sortable semantic table and bounded left filter panel. Below 1024px render accessible cards and a modal filter sheet with focus return. Rows/cards expose a labelled `View details` action; sorting writes the URL. Loading uses labelled skeletons; error retry retains URL state; empty reset clears only planner filters; invalid-link notice lists normalized keys.

- [ ] **Step 5: Verify interaction, accessibility, types, and build**

Run: `npm test -- --run web/tests/planner-workbench.test.tsx`

Expected: PASS with no axe violations in the workbench state fixtures.

Run: `npm run typecheck; npm run lint; npm run build`

Expected: all exit 0.

- [ ] **Step 6: Commit the workbench**

```powershell
git add web/src/features web/tests
git commit -m "feat: build responsive planner workbench"
```

---

### Task 8: Add configuration detail and traceability

**Files:**
- Create: `web/src/features/planner/ConfigurationDetail.tsx`
- Create: `web/src/features/planner/EquipmentFacts.tsx`
- Create: `web/src/features/planner/DemandTrace.tsx`
- Create: `web/tests/configuration-detail.test.tsx`

**Interfaces:**
- Consumes: Task 6 detail route/search and Task 7 API hooks/view models.
- Produces: wide-screen contextual detail panel, narrow-screen full route, compatibility facts, demand composition, assumptions, confidence reasons, sources, not-found recovery, and back navigation preserving filters.

- [ ] **Step 1: Write failing deep-link, unknown, traceability, and 404 tests**

```typescript
it('renders unknown equipment explicitly', async () => {
  renderDetail({ response: detailWithUnknownHud })
  expect(await screen.findByText('Head-up display')).toBeVisible()
  expect(screen.getByText('Unknown')).toBeVisible()
})

it('preserves planner query when returning from a deep link', async () => {
  const user = userEvent.setup()
  renderDetail({ search: { market: ['FR'], horizon: [2030] } })
  await user.click(await screen.findByRole('link', { name: 'Back to planner' }))
  expect(currentLocation().search).toMatchObject({ market: ['FR'], horizon: [2030] })
})

it('offers a safe return for a missing configuration', async () => {
  renderDetail({ response: notFoundProblem })
  expect(await screen.findByRole('heading', { name: 'Configuration not found' })).toBeVisible()
  expect(screen.getByRole('link', { name: 'Return to planner' })).toBeVisible()
})
```

- [ ] **Step 2: Run detail tests to verify RED**

Run: `npm test -- --run web/tests/configuration-detail.test.tsx`

Expected: FAIL because detail components do not exist.

- [ ] **Step 3: Implement traceable detail presentation**

Render canonical vehicle identity first, then demand range, exposure/replacement assumption, equipment facts, identity/data-quality confidence with reasons, evidence status, source summaries, update timestamp, and data version. Use definition lists and headings. Never hide unknowns, never infer compatibility, and label all fixture sources as synthetic demonstration evidence.

- [ ] **Step 4: Implement responsive route/panel behavior and recovery**

On wide planner routes, selection opens an `aside` while keeping comparison context. On narrow layouts and direct links, render the same detail component as the main route. A 404 retains the query and offers return; API errors expose retry.

- [ ] **Step 5: Verify detail tests, full frontend suite, and build**

Run: `npm test -- --run; npm run typecheck; npm run lint; npm run build`

Expected: all exit 0.

- [ ] **Step 6: Commit detail traceability**

```powershell
git add web/src/features web/tests
git commit -m "feat: add planner configuration traceability"
```

---

### Task 9: Add browser, accessibility, responsive, and contract gates

**Files:**
- Create: `web/playwright.config.ts`
- Create: `web/e2e/planner.spec.ts`
- Create: `web/e2e/accessibility.spec.ts`
- Create: `web/e2e/responsive.spec.ts`
- Create: `scripts/run_planner_dev.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: all prior API/frontend commands.
- Produces: one local launcher, deterministic browser journey, desktop/mobile overflow checks, keyboard/axe gate, OpenAPI drift gate, and CI frontend jobs.

- [ ] **Step 1: Write failing Playwright journeys**

```typescript
test('filters, selects, deep-links, and restores history', async ({ page }) => {
  await page.goto('/planner')
  await page.getByRole('checkbox', { name: 'France' }).check()
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await page.getByRole('link', { name: /View details/ }).first().click()
  await expect(page.getByRole('heading', { name: /configuration detail/i })).toBeVisible()
  await page.goBack()
  await expect(page.getByRole('checkbox', { name: 'France' })).toBeChecked()
})

for (const viewport of [{ width: 390, height: 844 }, { width: 1440, height: 900 }]) {
  test(`has no page overflow at ${viewport.width}`, async ({ page }) => {
    await page.setViewportSize(viewport)
    await page.goto('/planner')
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  })
}
```

- [ ] **Step 2: Run the browser suite to verify RED**

Run: `npm run e2e`

Expected: FAIL because the launcher/configuration and browser gates are absent.

- [ ] **Step 3: Implement the cross-platform local launcher**

`scripts/run_planner_dev.py` validates repository paths, starts `uv run uvicorn icor.api.app:create_app --factory --host 127.0.0.1 --port 8000` and `npm run dev -- --host 127.0.0.1 --port 5173` without a shell, forwards Ctrl+C, terminates both children, and never reads production secrets. A `--check` option verifies prerequisites without starting processes.

- [ ] **Step 4: Implement browser and accessibility gates**

Playwright starts the launcher, tests Chromium at 390x844 and 1440x900, injects axe, checks primary routes for serious/critical violations, exercises keyboard filters/detail/back-forward/retry/not-found, asserts visible focus, and asserts page-level `scrollWidth <= innerWidth`.

- [ ] **Step 5: Extend CI without changing deployment**

Add a Node 22 frontend job that runs `npm ci`, OpenAPI drift check, typecheck, lint, unit tests, production build, installs Playwright Chromium, and runs e2e against the local-only launcher. Keep existing Python Windows/Linux checks intact and add API dependencies to their locked sync naturally.

- [ ] **Step 6: Run the complete quality gate**

Run from repository root:

```powershell
uv lock --check
uv run ruff check src tests scripts
uv run pytest -p no:cacheprovider
uv run pip-audit
Set-Location web
npm ci
npm run openapi:check
npm test -- --run
npm run typecheck
npm run lint
npm run build
npm run e2e
Set-Location ..
git diff --check
```

Expected: all commands exit 0; pytest reports only the four documented strict XFAILs; pip-audit reports no known published third-party vulnerabilities.

- [ ] **Step 7: Commit the end-to-end gates**

```powershell
git add web scripts/run_planner_dev.py .github/workflows/ci.yml .gitignore
git commit -m "test: add planner browser and contract gates"
```

---

### Task 10: Document, run, and visually verify the local product slice

**Files:**
- Modify: `README.md`
- Modify: `docs/DEVELOPMENT.md`
- Modify: `docs/CODEX_HANDOFF.md`
- Modify: `docs/superpowers/specs/2026-08-25-planner-webapp-design.md`

**Interfaces:**
- Consumes: Task 9 launcher and all quality gates.
- Produces: exact setup/run/test instructions, approved spec status, durable verification evidence, local URLs, and desktop/mobile review artifacts under ignored `.local/review/`.

- [ ] **Step 1: Write a documentation contract test**

Add assertions to `tests/test_toolchain.py` that `README.md` and `docs/DEVELOPMENT.md` contain `scripts/run_planner_dev.py`, `http://127.0.0.1:5173`, `http://127.0.0.1:8000/docs`, `demonstration data`, and explicit language that no production secrets/customer data are required.

- [ ] **Step 2: Run the contract test to verify RED**

Run: `uv run pytest tests/test_toolchain.py -v`

Expected: FAIL on the missing planner documentation strings.

- [ ] **Step 3: Document local operation and product status**

Document Python/Node prerequisites, `uv sync --locked`, `npm ci`, launcher use, separate-process fallback commands, all test gates, synthetic fixture location, API docs URL, Streamlit coexistence, and the exact deferred production/proprietary-data scope. Change the design status to `approved` with the approval date.

- [ ] **Step 4: Start the app and verify live endpoints**

Run `uv run python scripts/run_planner_dev.py`, then verify `/api/health` returns HTTP 200/readiness and the planner route returns HTTP 200. Keep the server local-only.

- [ ] **Step 5: Capture and inspect desktop/mobile screenshots**

Use Playwright to capture `.local/review/planner-desktop.png`, `.local/review/planner-mobile.png`, and detail views. Inspect each for hierarchy, truncation, overflow, empty space, focus visibility, and evidence/confidence clarity; correct any visual defect and rerun affected tests.

- [ ] **Step 6: Run final fresh verification**

Repeat the complete Task 9 quality gate after all documentation/visual fixes. Verify the protected production checkout separately with read-only commands: it remains clean on `main` and `HEAD == origin/main == 1ba1d7c41a5fa8354134685b5c85509a0b8f6137`.

- [ ] **Step 7: Update durable handoff with evidence**

Record approval, files/commits, actual command outputs, live server PIDs/URLs, screenshot paths, unresolved risks, deferred scope, and exact next review actions. Do not record access keys, environment values, customer data, or fixture contents.

- [ ] **Step 8: Commit documentation and handoff**

```powershell
git add README.md docs tests/test_toolchain.py
git commit -m "docs: document planner web app workflow"
```

- [ ] **Step 9: Present the navigable local app for Lucas's review**

Report the local planner URL, concise feature outcome, exact verification results, production-isolation confirmation, known deferrals, and whether stopping the terminal would stop the two development processes. Do not push, merge, or deploy.
