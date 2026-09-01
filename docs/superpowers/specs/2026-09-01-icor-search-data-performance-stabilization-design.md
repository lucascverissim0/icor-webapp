# ICOR Search, Data Coverage, and Performance Stabilization Design

**Date:** 2026-09-01
**Status:** Approved
**Branch:** `development/windshield-demand-platform`

## Objective

Make ICOR's registration lookup, Source Evidence, Generation Planner, and Opportunity
Ranking usable against the full production-sized evidence snapshot. Extend the visible
timeline from 2000 through the latest official release without presenting estimates as
observed facts, and make year semantics unambiguous.

## Confirmed baseline

The active snapshot is 10.75 GB and contains 1,529,210 observations, 81,122 canonical
vehicles, 80,885 generation entries, 1,350,858 generation assignments, 1,736,619
cohort estimates, and 106,210 opportunity estimates.

Official evidence presently loaded is:

- EEA model-level new registrations, final releases for 2010-2024.
- KBA FZ10 new registrations for 2024.
- UK DfT VEH0160 new registrations for 2001-2025.
- UK DfT VEH0120 active fleet for 1994-2025.
- UK DfT VEH0124 detailed active-fleet/cohort evidence for observation years 2014-2025.

The official EEA data hub also exposes a 2025 provisional model-level release published
on 2026-06-25. It is not in the active snapshot. EEA model-level monitoring does not
provide 2000-2009 registrations; aggregate indicators covering that interval do not
justify fabricating model-level observed values.

## Product semantics

### Latest data

“Today” means the latest official release available as of the snapshot build date, not
real-time calendar-year registrations. Final and provisional observations may both be
shown, but provisional values must carry a visible status badge and source release.
Replacing a provisional year with a final release must be deterministic and auditable.

### 2000-to-latest timeline

The year control spans 2000 through the latest available official year. Availability is
scope-aware:

- Observed EU27 model-level values are shown only where supporting official evidence
  exists.
- Observed national values may extend further back or forward than EU27 values.
- A missing official model-level value is shown as unavailable, not zero.
- Any 2000-2009 EU model-level reconstruction is explicitly labelled `estimated`, kept
  separate from observed values, carries low confidence and a visible methodology, and
  exposes its supporting national or aggregate evidence.
- Observed and estimated values are never silently summed into one fact.

### Year fields

The UI and API distinguish:

- Observation year: when the stock or registration measure was observed.
- First-registration year: the vehicle cohort's first-use/registration year.
- Manufacture year: when the vehicle was manufactured.
- Model year: a product identity attribute when supplied by a source.

The audited 1914 rows are valid UK vintage active-fleet observations, not 1914 annual
sales. They remain in raw evidence with clear labels and do not enter annual-new-
registration views. Validation rejects impossible future relationships while preserving
plausible antique vehicles.

## Registration lookup

### Search behavior

Submitting a search preserves geography, year, page size, and all other route-backed
filters. Search input is normalized by case-folding, whitespace compression, and safe
punctuation handling. Multi-token input is matched across canonical make and model, so
`Volkswagen Golf` matches make `Volkswagen` plus model `Golf`. Literal wildcard
characters remain escaped.

The primary result is a canonical model-family total. Official labels and variants are
available in an expandable breakdown with their individual counts, source releases, and
geographies. This keeps a Golf lookup understandable while retaining source fidelity.
Search supports keyboard submission, an explicit clear action, URL restoration, empty
results, and loading/error states.

### Query model

Registration ranking moves from repeated observation scans to a build-time materialized
registration aggregate keyed by:

- geography and aggregate geography scope;
- year;
- publication status;
- canonical family identity;
- official source label identity;
- source and release lineage.

The table stores observed values separately from estimates. Supporting indexes cover
scope/year/status ordering and normalized make/model-family lookup. The API performs
database-side filtering, ranking, totals, and pagination. It never reads all matching
observations merely to return one page.

## Source Evidence redesign

The default page is a source catalog rather than a raw observation table. Each source
card shows publisher, measure, geography, coverage, latest release, final/provisional
status, accepted/rejected/quarantined counts, warnings, and a plain-language explanation
of what the source can and cannot prove.

Raw observations are a secondary drill-down. Filters include source, geography, measure,
observation year, year-field semantics, mapping status, and search. Column labels spell
out year meaning. A contextual explanation appears before raw rows.

Evidence summary values are materialized during snapshot creation or queried through
indexed summary tables. Observation pagination uses indexed filters and a stable cursor
or indexed offset appropriate to the chosen sort. The endpoint does not perform multiple
unindexed full-table aggregates per page visit.

## Planner repository

`SnapshotPlannerRepository.list_all()` is removed from interactive request paths. The
repository exposes dedicated operations for:

- small immutable option metadata;
- filtered configuration count and page queries;
- one configuration detail query;
- paginated model-year demand;
- source lineage for only the requested records.

Filtering, sorting, and pagination occur in SQLite. Query projections return only fields
needed by the endpoint. Repeated source tuples are not duplicated across every in-memory
configuration. Options and version metadata may be cached because a sealed snapshot is
immutable; page data is read on demand.

## Opportunity Ranking repository

Opportunity aggregation, coverage resolution, scoring inputs, and pagination move to
database-side queries or build-time materialized ranking inputs. The endpoint accepts
group, market, horizon, sort, page, and page-size parameters and returns only one page
plus summary totals.

Production-coverage joins preserve exact, fallback, mixed, and uncovered semantics.
Ranking remains deterministic and versioned. Drill-down queries only the selected group.
No request constructs all planning configurations, all cohort atoms, or all groups in
Python.

## Snapshot schema and build

The snapshot schema receives an explicit version bump. Build-time migrations are not
applied to sealed active snapshots in place; a new candidate is built and verified.
New structures include:

- materialized registration-family and official-label aggregates;
- materialized source/release quality summaries;
- planner option metadata;
- query indexes for evidence, planner, opportunity, and lineage access;
- quality-audit results and publication-status fields required by the UI.

Every materialized value retains deterministic lineage to source releases and, where
needed for drill-down, contributing observation identifiers.

## Data-quality audit

Snapshot validation reports and gates:

- year values outside supported bounds;
- cohort/manufacture years later than observation year;
- new-registration cohort years inconsistent with their observation period;
- invalid or missing year semantics;
- duplicate canonical/source identities;
- missing model labels and generic aggregates;
- incompatible units or measures;
- observed/estimated status leakage;
- release overlap and dependency-group double counting;
- unexpected changes in accepted, rejected, quarantined, and mapped counts.

Plausible historical vehicles are warnings or valid evidence, not rejected solely because
they predate 1950. The Source Evidence UI makes all flags and dispositions inspectable.

## API and compatibility

Existing route names remain stable where practical. Additive response fields carry data
status, latest-release meaning, family/variant hierarchy, year semantics, pagination,
and quality metadata. Any required breaking change receives an explicit versioned route
or coordinated frontend update. OpenAPI compatibility remains a required gate.

Errors distinguish unavailable scope, no matching result, snapshot unavailable, and
internal failure. The frontend renders an actionable state for each and never remains in
an indefinite loading state. Requests receive bounded server timeouts and correlation
identifiers remain available for diagnosis.

## Performance targets

Measured on the current production-sized snapshot after warm-up in the preview
Codespace:

- Registration lookup first page: p95 at or below 2 seconds.
- Source catalog: p95 at or below 2 seconds.
- Raw evidence first page with common filters: p95 at or below 2 seconds.
- Planner options: p95 at or below 1 second.
- Planner first page: p95 at or below 2 seconds.
- Opportunity first page: p95 at or below 3 seconds.
- No interactive endpoint may deserialize an unbounded result set.
- Initial application readiness target: under 60 seconds, with checksum/integrity work
  moved to promotion or an explicit verification command rather than every boot.

Cold-start results are recorded separately from warm-query latency. Automated query-plan
checks ensure intended indexes are used for representative requests.

## Test-first delivery

Implementation begins with failing tests for:

- preserving route scope during search;
- multi-token make/model matching and Golf family aggregation;
- observed/provisional/estimated separation;
- scope-aware 2000-to-latest availability;
- year-semantic display and antique evidence handling;
- evidence source catalog and paginated drill-down;
- SQL-backed planner pagination and detail;
- SQL-backed opportunity grouping, ranking, pagination, and drill-down;
- anomaly validation and snapshot schema/version checks;
- bounded query counts and production-sized performance budgets;
- loading, empty, error, and recovery states in browser tests.

Verification includes focused tests during development, full Python and frontend suites,
Ruff, type checking, lint, production build, OpenAPI compatibility, dependency audits,
Playwright, snapshot validation, live authenticated smoke tests, and timed representative
queries.

## Rollout

Build and validate a new candidate snapshot without changing the active one. Compare
source counts, registrations, quality results, Golf 2024 output, planner totals, and
opportunity totals against documented expectations. Promote atomically only after every
gate passes. Restart the private Codespaces preview, confirm authentication and all four
affected workflows, then open the application for Lucas. Rollback remains the previous
sealed snapshot and previous development commit; protected `main` is unchanged unless
separately authorized.

## Non-goals

- Claiming real-time 2026 registrations when no official 2026 release exists.
- Presenting 2000-2009 EU model estimates as observed official registrations.
- Inferring exact windshield fitment or calibrated replacement demand from registration
  data alone.
- Deleting valid vintage vehicle evidence merely because its cohort year is old.
- Merging or deploying to protected production without separate authorization.
