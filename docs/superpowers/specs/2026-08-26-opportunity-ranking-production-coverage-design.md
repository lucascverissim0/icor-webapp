# ICOR Opportunity Ranking and Production Coverage Design

Date: 2026-08-26
Status: approved in conversation on 2026-08-26; awaiting written-spec review

## Purpose

Add a decision page that shows which vehicle brands, models, model years, and exact
windshield configurations are most interesting for ICOR based on forecast windshield
replacement demand. Let an ICOR user record what the company already produces so the
ranking can recognize that increasing an existing product is generally easier than
starting a new one.

This is a local-first product slice on `development/windshield-demand-platform`. It
extends the React/FastAPI modular monolith and does not deploy, merge, push, use
customer data, or modify the protected production checkout.

## Product outcomes

A planner can:

1. compare ranked brands, models, and model years by forecast windshield
   replacements;
2. see the unmodified downside, base, and upside replacement demand behind every
   ranking;
3. understand separately how existing ICOR production affected an opportunity score;
4. drill from an aggregate result to the exact windshield configurations that
   contribute to it;
5. record, edit, and delete exact configuration or SKU coverage when known;
6. deliberately record a broader brand/model/model-year fallback when an exact
   configuration is not known;
7. see rankings refresh after confirmed coverage changes; and
8. distinguish demonstration forecasts and shared local prototype coverage from
   validated production evidence.

The page supports production prioritization; it does not claim that the opportunity
score is expected profit, manufacturing feasibility, or a validated commercial
recommendation.

## Scope boundaries

### Included

- A responsive `/opportunities` route linked from the existing product shell.
- Brand, model, and model-year ranking views with configuration drill-down.
- Versioned demonstration demand at exact configuration and vehicle model-year level.
- A shared local SQLite database for production-coverage records.
- Exact configuration/SKU coverage and deliberately broader fallback coverage.
- A versioned, explainable ranking strategy based on demand and production readiness.
- Create, read, update, and delete API operations for production coverage.
- Validation, transactional persistence, explicit error states, and automated tests.
- A ranking interface that can accept cost-basis inputs and strategies in a later
  product phase.

### Deferred

- Proprietary fitment, production, shipment, cost, or replacement-history ingestion.
- A cost-basis score, expected-margin score, or optimization model.
- Authentication, authorization, user attribution, tenants, approval workflows, and
  audit-grade change history.
- Production database selection, backup, migration operations, or disaster recovery.
- Forecast calibration, backtesting, or remediation of the legacy strict XFAILs.
- Push, merge, deployment, or modification of the active multi-user application.

## Domain model

### Model-year demand

The current demonstration fixture supplies one demand range for an exact windshield
configuration spanning a range of model years. It cannot support a truthful model-year
ranking. Extend the demonstration contract with immutable model-year demand records:

- `configuration_id`: existing canonical windshield configuration identity;
- `model_year`: integer vehicle model year within the configuration's applicability;
- `forecast_horizon`: the forecast year;
- `downside_units`, `base_units`, and `upside_units`: non-negative replacement counts;
- `evidence_status`, `data_version`, and source metadata.

For each configuration and horizon, the sum of its model-year demand records must
reconcile exactly with the existing configuration-level demand range. Demonstration
values remain deterministic, synthetic, and visibly labelled. No equal split or other
allocation may be presented as observed evidence without an explicit source label.

### Production coverage

`ProductionCoverage` is independent from forecast data and contains:

- an immutable coverage identifier;
- `match_type`, either `exact_configuration` or `vehicle_year_fallback`;
- `configuration_id` for an exact match, otherwise null;
- canonical brand, model, and one model year;
- the resolved SKU when the selected configuration has one, otherwise null;
- optional non-secret planner note;
- created and updated timestamps in UTC.

An exact entry must resolve to a known configuration whose brand, model, model-year
range, and SKU agree with the submitted canonical values. A fallback must resolve to a
known brand/model/model-year combination and must be explicitly requested. Free-form
vehicle identities are not accepted in this slice.

Duplicate active coverage for the same exact configuration and model year, or for the
same fallback identity, is rejected. An exact match takes precedence over a fallback
and the readiness contribution is never counted twice.

### Opportunity result

An opportunity row includes:

- grouping level and stable group identity;
- display brand, optional model, and optional model year;
- reconciled downside, base, and upside replacement totals;
- contributing configuration count;
- exact-covered, fallback-covered, and uncovered base units;
- coverage status;
- demand score, readiness score, and total opportunity score;
- ranking strategy name and version;
- a concise explanation of the score; and
- evidence and data-version metadata.

Coverage status is one of `exact_covered`, `fallback_only`, `mixed`, or `uncovered`.
`exact_covered` requires every base-demand unit in the group to have an exact match;
`fallback_only` has covered units but no exact match; `mixed` contains more than one
of exact, fallback, and uncovered demand; and `uncovered` has no matching coverage.

## Ranking strategy

The initial strategy is named `demand_readiness_v1`. It is a replaceable application
policy, not a column persisted on forecast or coverage records.

Within the complete filtered candidate set for the selected grouping level:

1. Rank base replacement demand from lowest to highest. For two or more candidates,
   divide each candidate's average zero-based positional rank by `candidate_count - 1`;
   ties therefore receive the same average percentile. A single non-zero candidate
   receives percentile 1; an all-zero candidate set receives percentile 0.
2. Scale the demand percentile to 0–80 points.
3. Calculate a readiness ratio from base units: exact-covered units count at 100%,
   fallback-covered units count at 50%, and uncovered units count at 0%.
4. Scale the readiness ratio to 0–20 points.
5. Add the two components and round only the displayed total to one decimal place.

This makes existing exact production a moderate advantage while allowing a
substantially larger unmet market to outrank it. Raw demand is never changed by the
readiness calculation. Downside and upside remain visible context but do not affect
`v1` ranking.

The API returns the strategy name, version, component values, and explanation. A
future cost-basis strategy will implement the same ranking interface and may consume
separately modelled setup cost, variable cost, capacity, price, and margin evidence.
Those facts must not be inferred from production coverage or added as nullable fields
without a validated data contract.

## Architecture

### Domain and application layers

- Add immutable model-year demand, production coverage, opportunity grouping, and
  score-component types.
- Define separate read protocols for forecast data and write-capable protocols for
  production coverage.
- Add an `OpportunityService` for reconciliation, aggregation, coverage resolution,
  scoring, sorting, and drill-down.
- Add a `ProductionCoverageService` for canonical validation and CRUD operations.
- Define a `RankingStrategy` protocol and implement `demand_readiness_v1`.

The services have no dependency on FastAPI, React, SQLite, Streamlit, or legacy
scripts. The existing `PlannerService` continues to own planner searches and details.

### Persistence adapter

Use Python's SQLite support through a focused repository adapter. The database path is
local configuration, defaults to an ignored development-data location, and must never
point into the demonstration fixture or tracked production data.

Initialize the schema with an explicit version table and idempotent migration runner.
Enable foreign keys and execute each write transaction atomically. Repository methods
use bound parameters and domain objects; API handlers do not issue SQL. Tests use an
isolated temporary database.

### API adapter

Add versioned endpoints:

- `GET /api/v1/opportunities?group_by=brand|model|model_year`
- `GET /api/v1/opportunities/{group_id}/configurations`
- `GET /api/v1/production-coverage`
- `POST /api/v1/production-coverage`
- `PUT /api/v1/production-coverage/{coverage_id}`
- `DELETE /api/v1/production-coverage/{coverage_id}`

Opportunity queries may reuse the planner's market and horizon filters. Sorting
defaults to total opportunity score descending with stable identity tie-breaking.
Mutation responses return the committed record or a successful deletion result;
ranking refresh remains a separate read so a mutation does not hide query failures.

Use the existing problem-response envelope and correlation IDs. Invalid canonical
matches return 422, duplicates return 409, missing records return 404, and unexpected
persistence failures return a non-sensitive 500 response.

### Web adapter

Add a lazy route at `/opportunities`, a primary navigation link, typed API client
methods, TanStack Query keys, and focused components for ranking, drill-down, and
coverage management. Do not place ranking or matching business logic in React.

After a successful mutation, invalidate opportunity and coverage queries. Do not
optimistically alter a score before the backend commits and recomputes it.

## Page and interaction design

The page header explains that interest combines forecast replacement demand with
ICOR production readiness and links to a short scoring explanation.

Summary cards show:

- total base forecast replacements in the current view;
- base demand covered by exact production; and
- uncovered base-demand units among rows in the highest demand quartile. The quartile
  uses the same tie-aware percentile calculation as the score and includes rows at or
  above percentile 0.75.

A segmented control switches between Brands, Models, and Model years. Each row shows
the name, demand range, exact/fallback/uncovered coverage composition, readiness
effect, total score, and concise explanation. Raw base replacements remain the most
prominent numeric fact. Selecting a row reveals its contributing exact windshield
configurations and their model-year demand without losing the aggregate context.

The “Manage ICOR production” panel provides cascading canonical selectors:

1. brand;
2. model;
3. model year; and
4. exact configuration/SKU.

Exact configuration is the normal path. “Exact configuration unknown” reveals the
fallback action, explains its lower precision and smaller ranking contribution, and
requires deliberate confirmation. Existing entries can be reviewed, edited, and
deleted. Destructive deletion requires confirmation.

Desktop layout may place management beside rankings when space permits. Tablet and
mobile stack content in task order. Native controls, visible focus, labelled status
messages, keyboard-operable disclosure, reduced-motion support, and appropriate table
or list semantics are required. The existing ICOR visual language and evidence badge
remain authoritative.

## Data flow

1. The page loads opportunity results and current production coverage from separate
   queries.
2. The API obtains canonical model-year demand and coverage through repository
   protocols.
3. `OpportunityService` reconciles totals, resolves exact-before-fallback coverage,
   aggregates at the requested level, and invokes the configured ranking strategy.
4. The UI renders raw demand, coverage composition, and score components returned by
   the API.
5. A coverage mutation is validated against canonical forecast identities and written
   in one SQLite transaction.
6. After commit, the UI refetches both coverage and opportunity queries and announces
   success. Failure preserves the user's form and the previously confirmed ranking.

## Error and empty states

- An unavailable opportunity query shows a retry action without hiding the page's
  explanation.
- A failed coverage mutation retains submitted selections, shows the correlation ID,
  and does not claim success.
- No forecast candidates, no saved coverage, no matches after filters, and all-zero
  demand have distinct messages.
- A stale/deleted configuration in stored coverage is treated as a data-integrity
  error, excluded from scoring, and surfaced for correction; it is not silently
  remapped.
- Corrupt or unsupported SQLite schema versions prevent writes and return a safe
  service error rather than rebuilding or deleting the database automatically.

## Verification

### Python

- Domain invariants for model-year demand and coverage identities.
- Exact reconciliation between model-year and configuration totals.
- Brand, model, and model-year aggregation.
- Percentile ties, a single candidate, all-zero demand, and stable tie-breaking.
- Exact match precedence, fallback weighting, mixed coverage, and no double counting.
- Canonical validation, duplicates, transaction rollback, migrations, and CRUD.
- API success and 404/409/422/500 problem contracts.
- OpenAPI export drift.

### React

- Brand, model, and model-year switching and rendering.
- Demand prominence, score explanation, coverage composition, and drill-down.
- Exact configuration selection and deliberate fallback confirmation.
- Create, edit, delete, retained-form failure, loading, empty, and retry states.
- Query invalidation only after confirmed mutations.
- Keyboard behavior, accessible names/statuses, and responsive navigation.

### Browser and quality gates

- Add exact coverage and observe a refetched readiness/ranking change.
- Edit and delete coverage and observe reconciled results.
- Add a fallback and verify its lower contribution and visible warning.
- Inspect desktop and mobile layouts and the aggregate-to-configuration drill-down.
- Run fresh Ruff, pytest, frontend unit tests, ESLint, TypeScript/build, Playwright,
  OpenAPI drift, lock checks, and dependency audits before completion claims.

## Security and operational notes

The SQLite database is shared local prototype state because the app does not yet have
authentication or authorization. The UI must say so, and the server must bind locally
under the existing launcher defaults. Notes reject control characters and use bounded
lengths; no secrets or customer/private data belong in coverage records.

This slice is not deployable as a multi-user write service until identity, roles,
authorization, audit requirements, concurrency expectations, production persistence,
backup, and privacy controls have their own approved design.

## Acceptance criteria

The slice is complete when:

1. the local app exposes a polished `/opportunities` page for brands, models, and
   model years;
2. every aggregate reconciles to exact configuration/model-year demonstration demand;
3. a user can manage exact or deliberately broader production coverage in SQLite;
4. exact coverage overrides fallback coverage without double counting;
5. `demand_readiness_v1` produces auditable 80/20 component scores while leaving raw
   demand unchanged;
6. cost-basis ranking can be added behind the ranking interface without migrating
   coverage records;
7. errors never discard uncommitted form input or claim an unconfirmed ranking change;
8. automated and visual checks pass with fresh evidence; and
9. the protected production checkout, deployed app, proprietary data, and unrelated
   concurrent changes remain untouched.
