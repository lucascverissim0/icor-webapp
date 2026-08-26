# ICOR Planner Web App Design

Date: 2026-08-25
Status: approved on 2026-08-25; implemented as a local demonstration product slice

## Purpose

Build the first navigable version of ICOR's future decision-planning product: a
responsive planner workbench that helps a windshield manufacturer move from market
signals to windshield-configuration and SKU decisions. This is a local-first product
slice on `development/windshield-demand-platform`; it does not replace, deploy to, or
modify the current Streamlit production application.

This slice establishes the web product shell, its API boundary, and a believable
planner workflow using deterministic demonstration data. It must not present the
current prototype forecast as validated production evidence, and it must not invent
proprietary fitment facts that have not yet been supplied.

## Product outcome

A planner can open the local app on desktop, tablet, or phone and:

1. understand the current planning scenario and the evidence status;
2. narrow demand by market, horizon, brand, model, and windshield configuration;
3. compare candidate windshield SKUs or part families in one workbench;
4. inspect base, downside, and upside demand together with identity and data-quality
   confidence;
5. open a configuration detail view that explains vehicle compatibility, demand
   composition, assumptions, and sources;
6. preserve filter and selection state in the URL so a view can be revisited; and
7. clearly distinguish demonstration/prototype values from validated business data.

The experience follows the planner-workbench visual direction selected by Lucas. It
is a dense professional tool, not an executive dashboard and not a step-by-step
wizard.

## Scope boundaries

### Included

- A new responsive React application with an ICOR product shell.
- A FastAPI backend mounted over import-safe Python application services.
- A deterministic demonstration dataset expressed in the future canonical contract.
- Planner filters, summary metrics, sortable comparison results, selection, and a
  detail panel or route.
- Explicit evidence-status and confidence presentation.
- Loading, empty, invalid-filter, API-error, and not-found states.
- Automated frontend, API-contract, accessibility, and responsive-layout checks.
- Local developer commands and documentation.

### Deferred

- Proprietary catalog or replacement-history ingestion.
- Selecting, training, or claiming validity for a production forecasting model.
- Fixing the legacy Streamlit forecast defects `ICOR-001`, `ICOR-006`, `ICOR-009`,
  and `ICOR-030` unless a shared extraction is required by this slice.
- Authentication, authorization, multi-tenant storage, saved scenarios, comments,
  collaboration, exports, background jobs, and production observability.
- Migration or retirement of the Streamlit application.
- Push, merge, deployment, or any production/customer-data operation.

## Architecture decision

Use a modular monolith with two independently testable adapters in one repository:

- **Web adapter:** React and TypeScript, built with Vite. The UI owns interaction,
  responsive layout, accessible presentation, and URL state. TanStack Router provides
  typed routes and search parameters; TanStack Query owns server-state fetching and
  cache behavior. Tailwind CSS supplies design tokens and responsive primitives;
  small accessible headless components may be used where native elements are not
  sufficient.
- **API adapter:** FastAPI under the existing Python 3.12 `icor` package. Pydantic
  request and response models define the contract. API routes call application
  services; routes never read fixtures or legacy script files directly.
- **Application and domain layers:** import-safe Python modules define planner queries,
  entities, confidence/evidence semantics, and repository protocols. They have no
  dependency on FastAPI, Streamlit, browser code, or concrete storage.
- **Data adapter:** a read-only in-memory/file-backed demonstration repository loads a
  versioned, non-proprietary fixture. It can later be replaced by validated catalog
  and forecast repositories without changing the UI contract or domain services.

During development Vite and FastAPI run as separate local processes, with Vite
proxying `/api` to FastAPI. A production build path may later serve static assets from
the Python service or another approved host, but deployment topology is deliberately
deferred.

### Why this architecture

The product's future differentiation—fitment identity, demand calculation, evidence,
and model evaluation—belongs in Python, where the current code and data tooling
already live. React provides the responsive, stateful workbench that Streamlit cannot
comfortably become. Keeping both adapters in one modular monolith preserves fast local
iteration and atomic contract changes without introducing service discovery, queues,
or distributed operations before they are justified.

The rejected alternatives are:

- **Next.js plus Python API:** useful for server rendering and public SEO, neither of
  which this authenticated planning tool currently needs; it adds a second server
  runtime and overlapping backend responsibilities.
- **Continue with Streamlit:** fastest for a disposable analytics prototype, but it
  would constrain responsive interaction, URL-addressable planner state, component
  testing, and the selected workbench experience.
- **Microservices:** premature for one product team and one bounded planning workflow;
  it would turn internal interfaces into network and deployment problems.

## Repository structure

The implementation will add focused boundaries rather than mix new UI code with the
legacy Streamlit pages:

```text
web/
  src/
    app/                 # router, providers, shell, global error boundary
    features/planner/    # filters, comparison, detail, view models
    components/          # reusable presentation primitives
    lib/                 # API client, formatting, URL-state helpers
  tests/
  package.json
  vite.config.ts
src/icor/
  domain/planner.py      # entities and value semantics
  application/planner.py # use cases and repository protocols
  infrastructure/demo_planner_repository.py
  api/
    app.py               # FastAPI factory and middleware
    planner.py           # thin planner routes
    schemas.py           # versioned request/response schemas
data/demo/
  planner-v1.json        # synthetic/non-proprietary deterministic fixture
tests/
  api/
  application/
```

Files may be split further when a unit would otherwise mix responsibilities. Legacy
`ui/` and `scripts/` remain behaviorally unchanged in this slice.

## Domain and API contract

### Canonical planning record

Each comparison row represents one candidate windshield configuration or part family,
not merely a vehicle model. The initial contract contains:

- stable `configuration_id` and optional demonstration `sku`/`part_family`;
- market, brand, model, model year range, generation/facelift, and body style;
- distinguishing equipment flags such as camera/ADAS, HUD, heating, acoustic glazing,
  rain/light sensor, and drive side when known;
- forecast horizon and base/downside/upside replacement-demand units;
- vehicle exposure and replacement-rate assumption summaries;
- identity-confidence and data-quality-confidence levels with short reasons;
- evidence status (`demonstration`, `prototype`, or `validated`) and source summaries;
- `updated_at` plus a data-version identifier.

Unknown facts remain explicitly unknown; they are never converted to `false` or zero.
Demonstration identifiers and values are visibly labelled and cannot use customer or
proprietary data.

### Endpoints

- `GET /api/health` returns service and fixture readiness without secrets.
- `GET /api/v1/planner/options` returns filter choices and scenario metadata.
- `GET /api/v1/planner/configurations` accepts validated filter, sort, and pagination
  parameters and returns summary totals plus comparison rows.
- `GET /api/v1/planner/configurations/{configuration_id}` returns the traceable detail
  record or a typed not-found response.

The JSON contract uses explicit units and ISO dates. Pydantic rejects invalid enums,
ranges, pagination, and unsupported sort fields. The frontend client is generated or
checked from the OpenAPI schema so incompatible changes fail verification.

## User experience

### App shell

The shell contains a compact ICOR identity, page title, scenario/evidence badge, and
responsive navigation. Desktop uses a persistent left rail or compact top/side hybrid;
small screens use a labelled disclosure menu. A visible “Demonstration data” status is
present on every planner screen.

### Planner workbench

On wide screens, filters occupy a bounded left panel, the central comparison region
uses the remaining width, and selection opens a contextual detail panel where space
permits. On narrow screens, summary metrics stack, filters move into a modal sheet,
comparison rows become accessible cards or a horizontally safe compact list, and
detail becomes a full route/screen. No essential action depends on hover.

The primary hierarchy is:

1. scenario, market, and forecast horizon;
2. demand range and number of candidate configurations;
3. comparison of configuration/SKU opportunities;
4. traceable evidence and assumptions for the selected candidate.

The default sort is base demand descending, with confidence visible beside demand so
volume is never interpreted without evidence quality. Filters apply predictably and
are reflected in typed URL search parameters. Invalid or obsolete URL values are
reported and safely normalized rather than crashing or silently selecting a different
business value.

### Visual system

Use a restrained industrial planning aesthetic: neutral surfaces, high-contrast text,
one primary ICOR accent, and semantic colors reserved for evidence/confidence states.
Typography and spacing favor scanability at high information density. Number formats
include units and locale-safe separators. Color is never the only carrier of meaning.

## Data flow

1. The router validates URL search parameters and creates a planner query.
2. TanStack Query calls the typed API client; a request key contains the normalized
   query.
3. FastAPI validates the request and invokes the planner application service.
4. The service queries the demonstration repository and calculates response summaries
   using pure domain functions.
5. Pydantic serializes the versioned response; the frontend maps it into focused view
   models and renders the workbench.
6. Selecting a row changes the configuration route while preserving the current
   query, allowing browser back/forward and shareable local URLs.

The UI does not calculate business demand, confidence, or compatibility rules. The
backend does not encode layout or presentation decisions.

## Error handling and safety

- FastAPI uses a consistent problem response containing a safe code, human-readable
  message, optional field errors, and request correlation ID; stack traces never enter
  API responses.
- The React root has a route-aware error boundary. Query failures show an inline retry
  state without discarding filters or previously selected intent.
- Empty results explain which filters constrained the view and offer a clear reset.
- Fixture/schema failures prevent readiness and produce a diagnostic safe for local
  development without exposing environment values.
- API access is local-only by documented default. CORS allows only the configured
  local Vite origin. No secrets are required for the demonstration slice.
- The demonstration repository is read-only. No endpoint mutates canonical files.
- Source text, fixture content, logs, tests, and screenshots must contain no secrets,
  customer information, or proprietary catalog values.

## Testing strategy

Implementation follows test-driven development at each boundary.

### Python

- Domain tests cover ranges, unknown values, confidence/evidence semantics, sorting,
  filtering, summaries, and deterministic pagination.
- Application tests use a repository fake and prove that business rules are adapter
  independent.
- FastAPI tests cover success, invalid queries, empty results, not found, response
  schema, health/readiness, and safe error shapes.
- Existing Streamlit and foundation tests remain green, including strict XFAILs until
  their separate remediation.

### Frontend

- Unit/component tests cover filter serialization, metric and confidence formatting,
  workbench states, row/card selection, and accessible labels.
- Mock-service tests cover loading, success, empty, validation error, server error,
  retry, and deep-linked detail.
- Browser tests cover the core planner journey at representative desktop and mobile
  viewports, keyboard navigation, browser history, and absence of horizontal page
  overflow.
- Automated accessibility checks target WCAG 2.2 AA fundamentals; manual verification
  covers focus order, visible focus, zoom/reflow, and screen-reader names for primary
  controls.

### Contract and quality gates

- OpenAPI generation or schema comparison must detect frontend/backend drift.
- TypeScript strict mode, frontend linting, production build, Ruff, pytest, dependency
  audit, secret scan, and `git diff --check` must pass.
- The final local review includes rendered desktop and mobile screenshots and a live
  navigable app. Visual review checks content hierarchy, truncation, overflow, empty
  space, and semantic-state clarity.

## Delivery sequence

1. Establish the FastAPI factory, domain contract, and deterministic demo repository
   behind passing Python tests.
2. Scaffold the Vite/React shell and typed API client with strict frontend quality
   gates.
3. Build the responsive planner workbench states and URL-driven filters using component
   tests.
4. Add configuration detail and traceability views.
5. Add browser, accessibility, contract, and responsive visual verification.
6. Document one-command local startup (or a small cross-platform launcher), update the
   durable handoff, and present the navigable local app for Lucas's review.

Each stage remains reviewable and keeps the Streamlit reference operational.

## Acceptance criteria

This slice is complete when all of the following are freshly demonstrated:

1. The React planner and FastAPI service start locally from the development worktree
   without production secrets or customer data.
2. A planner can filter, sort, select, deep-link, navigate back/forward, and inspect a
   windshield-configuration detail on desktop and mobile.
3. Every displayed demand value includes its evidence status; base/downside/upside and
   confidence meanings are unambiguous.
4. Unknown fitment facts are shown as unknown, never silently coerced to negative or
   zero values.
5. The API contract is validated, versioned, documented through OpenAPI, and protected
   against invalid ranges, sorts, and pagination.
6. Loading, empty, invalid-link, not-found, and server-error paths are usable and retain
   recoverable user state.
7. Keyboard and automated accessibility checks pass, and the app has no page-level
   horizontal overflow at the verified viewports.
8. Frontend tests/build/type checks and existing Python lint/tests/audit pass with no
   new expected failures or network access.
9. The demonstration fixture is deterministic, visibly labelled, non-proprietary, and
   isolated behind a replaceable repository adapter.
10. The protected `main` checkout and current deployed application remain unchanged;
    nothing is pushed, merged, or deployed without Lucas's later explicit approval.
