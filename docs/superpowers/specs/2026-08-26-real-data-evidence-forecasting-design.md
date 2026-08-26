# ICOR Real-Data Evidence and Forecasting Design

Date: 2026-08-26
Status: approved for implementation

## Purpose

Replace the fixture-led planner data path with a reproducible, multi-source evidence
system that supports EU passenger-car opportunity analysis by make, model, and model
year. The product must preserve source truth, reconcile overlaps without hiding
conflicts, estimate incomplete history explicitly, reconstruct active vehicle fleets,
and produce uncertainty-aware opportunity forecasts for 2028 and 2031.

This is a local-first redesign on `development/windshield-demand-platform`. It does
not modify the protected `main` checkout, production deployment, production secrets,
or customer data. It does not claim that public registrations alone validate
windshield replacement rates or windshield fitment.

## Product outcome

An authenticated planner can:

1. open directly to a complete EU opportunity ranking;
2. compare replacement-opportunity downside, base, and upside for 2028 or 2031;
3. see active fleet and fleet growth separately from replacement opportunity;
4. distinguish observed, reconciled, estimated, and forecast values;
5. filter by geography, make, model, model year, evidence confidence, forecast
   confidence, and inclusion of estimated history;
6. inspect every result's sources, identity mappings, conflicts, assumptions, method
   versions, and freshness;
7. search for a specific make, model, and model year on a dedicated page; and
8. understand when evidence is insufficient rather than receiving a fabricated value.

The first geographic product is the EU aggregate with available national drill-down.
Geography remains versioned and replaceable so worldwide coverage can be added later.

## Scope

### Included

- Immutable public-source releases with provenance, checksums, licensing metadata,
  retrieval time, and final/provisional/superseded status.
- Source-specific parsers behind one observation contract.
- Normalized but source-preserving vehicle labels and measures.
- A governed canonical make/model identity registry with reviewable mappings.
- Dependency-aware reconciliation and conflict detection.
- Explainable evidence-confidence components and categorical bands.
- Separate historical estimates where direct EU model evidence is incomplete.
- Cohort-based active-fleet reconstruction.
- Backtested future-registration models and uncertainty propagation.
- Assumption-led windshield replacement opportunity with P10/P50/P90 outputs.
- Atomic dataset snapshots, last-known-good serving, diagnostics, and reproducibility.
- The EU opportunity landing page, evidence workspace, and vehicle-search page.
- Automated parser, data-quality, domain, API, frontend, browser, accessibility,
  security, and clean-room rebuild verification.

### Deferred

- Proprietary windshield fitment/SKU catalog ingestion.
- Validation or calibration from company replacement, production, claim, shipment, or
  installation history.
- Any claim that a public-data replacement hazard is production validated.
- A commercial worldwide dataset contract or redistribution decision.
- Worldwide product launch.
- Production deployment, push, merge, production authentication migration, customer
  data, or changes to the existing multi-user application.
- Automated fuzzy identity merges without review.
- Machine-learning complexity that does not improve rolling out-of-sample results.

## Source strategy

The architecture is source neutral. Each source is an adapter with its own terms,
schema, geography, measure, revisions, and dependency group.

### Initial official sources

1. **European Environment Agency passenger-car CO2 monitoring data**
   - Coverage: EU member-state new passenger-car registrations, 2010-2024 in the
     currently reviewed release family.
   - Useful fields include reporting country, manufacturer/make, commercial name,
     type approval, type, variant, version, fuel, and registration count/record
     representation as documented by the release.
   - 2023 is final in the reviewed metadata; 2024 is provisional.
   - Role: EU-wide registration core and canonical overlap spine.

2. **UK Department for Transport/DVLA vehicle licensing files**
   - `df_VEH0160_GB`: first registrations by make/generic model/model from 2001.
   - `df_VEH0120_GB`: licensed/SORN stock by make/generic model/model from 1994 Q4.
   - Role: older model-level national evidence, stock/registration cohort validation,
     and overlap comparison. UK observations retain their historical geography and are
     never relabelled as EU observations.

3. **German Kraftfahrt-Bundesamt model-series registration tables**
   - FZ 10/FZ 11 releases provide new registrations by brand/model series for
     available years and months.
   - Role: high-authority EU-member national overlap, model identity evidence, and
     reconciliation checks against EEA-derived German totals.

Further official national or licensed sources use the same adapter contract. A source
cannot enter a published snapshot until its redistribution/use terms, field semantics,
revision policy, and expected totals are recorded.

### Source dependence

Evidence that originates from the same administrative register is correlated even if
published by two organizations. Each source release declares a `dependency_group`.
Agreement within one dependency group validates extraction and transformation but does
not receive the same confidence gain as independent agreement. The system never counts
the EEA publication and the corresponding national register as two independent samples
of the same vehicles.

## Architecture

Retain the existing modular monolith and replace the demonstration repository through
focused boundaries:

```text
external release
  -> immutable source artifact + manifest
  -> source parser
  -> normalized source observation
  -> canonical identity mapping
  -> reconciliation and confidence
  -> historical estimate / cohort reconstruction
  -> forecast and uncertainty
  -> versioned published snapshot
  -> FastAPI application services
  -> React opportunity and vehicle-search views
```

### Units

- **Source registry:** source definition, publisher, authority class, dependency group,
  measure semantics, geography, licence/terms link, cadence, and revision policy.
- **Release store:** immutable downloaded artifacts and manifests. Published source
  files are not silently replaced in place.
- **Parsers:** source-specific pure or isolated transformations into the common
  observation schema.
- **Canonical identity registry:** stable makes/models and reviewable source mappings.
- **Evidence service:** equivalence grouping, precedence, conflict classification, and
  evidence scoring.
- **Estimation service:** versioned historical estimation strategies that never mutate
  observations.
- **Fleet service:** cohort survival and active-fleet reconstruction.
- **Forecast service:** backtest, model selection, simulation, and forecast scoring.
- **Snapshot publisher:** validates and atomically promotes one complete queryable data
  version.
- **Application services:** ranking, detail, search, source evidence, and diagnostics.

Application services depend on protocols, not SQLite/file details. Source parsers do
not call UI or API code. The frontend never calculates business forecasts or
confidence.

## Core data contracts

### Source release

A release manifest includes:

- stable source and release identifiers;
- publisher and source URL;
- retrieval timestamp and stated publication timestamp;
- temporal and geographic coverage;
- measure definition and unit;
- final, provisional, corrected, or superseded status;
- dependency group;
- licence/terms reference and permitted local use;
- artifact path, byte size, and SHA-256 checksum;
- parser name/version and expected schema;
- raw, accepted, rejected, and quarantined record counts; and
- aggregate validation results and promotion status.

### Observation

One observation represents one source's claim and includes:

- source/release identity and original row locator;
- source geography and geography-definition version;
- period and period precision;
- measure (`new_registrations`, `active_fleet`, or another explicitly defined metric);
- value, unit, rounding/suppression metadata, and publication status;
- original make/model/model-year/type labels and identifiers;
- normalized labels that do not replace the originals;
- canonical identity mapping ID and mapping status;
- transformation notes and validation flags; and
- evidence-confidence components.

Values with different measures, geography definitions, or periods are not equivalent
and cannot be reconciled into one observed value.

### Canonical identity

The registry models makes, model families, aliases, market-specific names, and model
years separately from future windshield configurations. A mapping has one status:

- `exact_identifier`;
- `curated_alias`;
- `normalized_label`;
- `reviewed_probable`;
- `ambiguous`;
- `rejected`; or
- `unresolved`.

Ambiguous and unresolved mappings do not contribute to model-level published totals.
They remain visible in diagnostics and in source-level aggregate reconciliation. No
fuzzy score alone may publish a merge.

### Published value

Every user-visible numeric value has:

- value status: `observed`, `reconciled`, `estimated`, or `forecast`;
- metric, unit, geography, period, vehicle identity, and snapshot version;
- input observation or estimate identifiers;
- method and parameter version;
- evidence confidence and reasons;
- forecast confidence and reasons where applicable;
- P10/P50/P90 where applicable; and
- freshness, warnings, and reproducibility metadata.

## Reconciliation

### Equivalence and precedence

Only semantically equivalent observations enter one reconciliation group. The
deterministic primary selection order is:

1. final over provisional;
2. direct administrative publication over a derived republication;
3. complete over explicitly partial/suppressed coverage;
4. corrected/current release over superseded release; and
5. more precise identifiers and periods over coarser equivalents.

The selected reconciled value references every candidate. Original observations never
change. If the selected primary later changes, a new snapshot explains why.

### Agreement and conflict

For positive equivalent values `a` and `b`, relative difference is:

`abs(a - b) / max(abs(a), abs(b), 1)`

Initial versioned thresholds are:

- at most 2%: `concordant`;
- greater than 2% and at most 10%: `review_required`;
- greater than 10%: `conflict`.

Exact source-specific rounding and suppression tolerances override these defaults when
documented. Thresholds are recalibrated only through a new method version after actual
overlap distributions are reviewed.

Conflicting values are never averaged automatically. The selected primary remains
visible beside alternatives, residuals, dependency information, and selection reasons.

### Aggregate checks

Where a publisher supplies totals, accepted detail must reconcile within documented
rounding/suppression tolerance. Checks include total mass, duplicates, impossible
negative values, year/geography leakage, missing identifiers, unknown code frequency,
and unexpected distribution shifts. A mandatory failure quarantines the release.

## Evidence confidence

Evidence confidence is not forecast uncertainty. It is a 0-100 explainable score:

- source authority: 0-25;
- publication/revision status: 0-10;
- geographic and record coverage: 0-25;
- canonical identity quality: 0-20; and
- independent agreement: 0-20.

Bands are:

- High: 80-100;
- Medium: 60-79;
- Low: 40-59; and
- Very low: 0-39.

Every score exposes its components and reasons. No-overlap evidence receives a neutral,
documented agreement component rather than being treated as either confirmed or
conflicting. Correlated agreement has a capped contribution. Provisional evidence,
reviewed-probable identity, partial coverage, and historical estimation apply explicit
versioned caps. Ambiguous/unresolved identity is excluded rather than assigned a
misleading model-level score.

The first implementation must encode the rubric as data/configuration plus pure tested
functions. Component point assignments and caps become final only after source profiling
shows the real completeness and overlap distributions; changing them creates a new
confidence-method version.

## Historical estimation

Observed national evidence and estimated EU values are separate datasets.

For incomplete model-year history, the estimator may use:

- observed national model shares;
- official national and EU total registrations;
- market-size weights;
- neighbouring-year continuity;
- brand, segment, geography, and model-family partial pooling; and
- stock-cohort evidence where registration history is unavailable.

The estimator produces distributions, not disguised observations. It records input
countries, market coverage, assumptions, method version, and P10/P50/P90.

Publication guardrails are:

- no estimate from unresolved identity;
- no extrapolation from stock to annual registrations without an explicit survival and
  migration/scrappage model;
- no EU label on a national total;
- no silent interpolation across a model launch/discontinuation boundary;
- no point estimate when the method cannot produce a defensible interval; and
- an `insufficient_evidence` result remains a valid product outcome.

The 1995-2009 period may contain low- or very-low-confidence EU estimates. The UI must
show the observed country evidence beside them and automatically widen uncertainty as
country/market coverage weakens.

## Forecasting

### Registration history

Observed/reconciled registrations remain the historical base. Historical estimates
are inputs with larger observation uncertainty; source confidence controls uncertainty
and diagnostics, not arbitrary multiplication of counts.

### Active-fleet reconstruction

For model cohort `m`, registration year `y`, and target year `t`:

`active_fleet[m,y,t] = registrations[m,y] * survival_probability[geography,segment,t-y]`

Survival curves are estimated and validated from official stock-by-age/cohort evidence
where available. Geography and segment fallbacks are explicit. Imports, exports,
re-registration, scrappage, suppression, and boundary changes enter the uncertainty or
method limitations. One full year of attrition applies to a one-year-old cohort.

### Future registrations

Candidate annual models initially include:

- last-observation/seasonal-naive where applicable;
- robust damped trend; and
- hierarchical brand/model trend with partial pooling.

Rolling-origin backtests select the simplest reliable candidate per series using
versioned error and stability criteria. Sparse series use an explicit parent fallback.
Structural breaks and model launch/discontinuation boundaries are flagged. Additional
models are accepted only when they improve out-of-sample accuracy and calibration.

### Windshield replacement opportunity

For active cohort exposure:

`replacement_opportunity = active_fleet * replacement_hazard(age, geography, vehicle)`

The hazard is a distribution, not the legacy fixed 2.1% fact. Until defensible public
evidence or proprietary replacement history is integrated, it remains an explicit,
versioned assumption with wide uncertainty. The product labels results
`assumption-led opportunity estimate`; it does not label them validated replacements.

Future proprietary fitment maps canonical vehicle cohorts to windshield configurations
or SKUs behind a separate mapping boundary. Until then, the product ranks vehicle
cohorts and never invents windshield compatibility.

### Uncertainty and forecast confidence

Simulation propagates uncertainty from historical missingness, identity, country
coverage, survival, future registrations, replacement hazard, and forecast horizon.
Downside/base/upside are P10/P50/P90, not arbitrary percentage adjustments.

Rolling backtests record point error, bias, stability, and empirical interval coverage.
Forecast confidence depends on those results, history depth, input evidence, structural
stability, fallback depth, and horizon. A model without sufficient backtest evidence is
labelled `experimental`. Intervals widen automatically for sparse inputs and longer
horizons such as 2031.

## Snapshot publication and failure handling

A build creates a candidate snapshot in isolation:

1. acquire or locate approved release artifacts;
2. verify checksums and manifests;
3. parse and quarantine invalid records;
4. normalize and map identities;
5. run source and aggregate quality gates;
6. reconcile evidence and calculate confidence;
7. estimate history, reconstruct fleets, and forecast;
8. run snapshot invariants and reproducibility checks; and
9. atomically promote the complete snapshot.

No partial snapshot becomes queryable. A failed source or build leaves the last
known-good snapshot active and records a safe diagnostic. The UI shows freshness and
failure warnings. If no valid snapshot exists, the API returns a typed unavailable
response; it never substitutes fixture values.

Builds are idempotent for identical artifacts, configuration, code, and seeds.
Randomized estimation uses recorded deterministic seeds. Snapshot identifiers include
or reference source, mapping, reconciliation, confidence, estimation, survival,
hazard, and forecast versions.

## API

The existing versioned FastAPI boundary evolves without exposing storage details.
Required capabilities are:

- health and readiness with active snapshot/freshness status;
- opportunity options and complete paginated EU rankings;
- opportunity detail with history, fleet, forecast, confidence, and evidence;
- vehicle make/model/model-year search and detail;
- source comparisons, warnings, and reproducibility metadata; and
- local/internal snapshot diagnostics separated from the primary product workflow.

All ranking and detail requests resolve against one snapshot ID to prevent mixed-version
pages. Responses use explicit units, ISO dates, status enums, confidence components,
and typed problem details. The generated/checked TypeScript client must fail CI on
OpenAPI drift.

## User experience

### Authenticated EU opportunity landing page

The default route shows the complete EU opportunity ranking immediately. Controls
include horizon (2028/2031), EU/available country geography, make/model search,
confidence filters, and an estimated-history inclusion control.

Rows show:

- rank, make, model, and model year/cohort;
- active fleet and fleet growth as separate values;
- replacement-opportunity P10/P50/P90;
- observed/reconciled/estimated/forecast status;
- evidence and forecast confidence;
- freshness and material warnings; and
- a path to evidence detail.

The default ranking uses base (P50) replacement opportunity. Confidence never changes
the raw opportunity value or hides a high-volume/low-confidence row; users may filter
confidence explicitly.

### Evidence workspace

Selecting a row exposes:

- observed and estimated registration history without visually connecting missing
  observations;
- active-fleet reconstruction and survival assumptions;
- forecast distribution and backtest evidence;
- source values, residuals, dependence groups, and selection reason;
- confidence components and caps;
- canonical identity mapping and unresolved exclusions;
- limitations, conflicts, and missing coverage; and
- all source/snapshot/method versions required to reproduce the result.

### Vehicle search

A separate page supports make, model, and model-year lookup against the same canonical
snapshot. It shows cohort history, country coverage, active-fleet projection,
replacement opportunity, provenance, and confidence. Search does not query a separate
or stale dataset.

### Presentation rules

- Observed and estimated values have distinct labels and visual treatment.
- Confidence includes text/reasons and never relies on color alone.
- Provisional, stale, conflicting, experimental, and assumption-led states are
  prominent.
- Missing values remain missing; zero is used only for a real measured zero.
- URL state preserves filters and selection.
- Mobile retains full provenance access; no essential action requires hover.
- Keyboard, screen reader, contrast, zoom/reflow, focus, loading, empty, and error
  behavior meet the existing WCAG 2.2 AA-oriented quality bar.

## Security and privacy

- Public-source ingestion requires no production secrets in source control.
- Download credentials for a future licensed source use validated local secret
  configuration and never enter manifests, logs, snapshots, tests, or chat.
- Authenticated product routes enforce authorization server-side; UI hiding is not an
  authorization boundary.
- Internal diagnostics require a separate authorization capability before any
  production use.
- API inputs are bounded and validated; expensive queries are paginated/rate-limited.
- Source strings render as untrusted text. Logs avoid raw private/customer records.
- Production customer/proprietary data and deployment remain out of scope.

## Testing and quality gates

### Parsers and evidence

- Contract tests use small, legally retainable representative release samples.
- Parser tests cover schema changes, encoding, suppression, duplicates, corrections,
  invalid rows, and aggregate totals.
- Identity tests cover exact identifiers, curated aliases, ambiguity, rejection, and
  the prohibition on automatic fuzzy publication.
- Reconciliation tests cover semantic equivalence, precedence, dependence, thresholds,
  conflicts, revisions, and immutable inputs.
- Confidence tests cover every component, cap, band boundary, missing overlap, and
  correlated agreement.

### Estimation and forecasting

- Historical-estimation tests prove observations are not overwritten and guardrails
  return `insufficient_evidence` when required.
- Property tests enforce non-negative counts, ordered P10 <= P50 <= P90, conserved
  published totals within documented tolerance, stable identity, and deterministic
  rebuilds.
- Fleet tests cover cohort aging, the one-year-old attrition convention, fallbacks, and
  survival bounds.
- Rolling backtests cover no-leakage folds, naive baselines, selection, sparse fallbacks,
  bias/error metrics, and interval coverage.
- Forecast tests prove longer/sparser cases carry the expected uncertainty inputs and
  that insufficient validation produces `experimental` status.

### API and frontend

- API tests cover snapshot consistency, ranking, detail, search, pagination, invalid
  filters, unavailable/stale snapshots, safe errors, and authorization.
- OpenAPI drift, strict TypeScript, lint, production build, Python type/lint/tests,
  dependency audit, and secret scan are mandatory.
- Component tests cover status/confidence formatting, URL state, warnings, charts with
  gaps, loading, empty, conflict, unavailable, and retry behavior.
- Browser tests cover the ranking-to-evidence and vehicle-search journeys, 2028/2031,
  observed/estimated filtering, back/forward/deep links, keyboard operation, serious/
  critical accessibility findings, and mobile/desktop overflow.
- Final visual review uses live real-source-derived local snapshots and verifies that
  provisional/estimated/experimental states cannot be mistaken for observed facts.

### Clean-room rebuild

One integration gate rebuilds a published snapshot from approved local raw artifacts
and an empty derived-data directory. It verifies checksums, deterministic output,
aggregate reconciliation, recorded exclusions, snapshot promotion, API results, and
UI-visible version/freshness metadata.

## Delivery sequence

1. Define source/release/observation/snapshot contracts and local storage migrations.
2. Implement atomic release manifests, candidate builds, validation, and promotion.
3. Ingest/profile EEA and publish a source-level EU registration snapshot.
4. Add KBA overlap, dependency-aware reconciliation, conflicts, and evidence scoring.
5. Add UK registration/stock evidence and canonical identity review workflow.
6. Implement historical EU estimation with guardrails and observed/estimated separation.
7. Implement cohort survival, candidate forecasts, rolling backtests, and simulation.
8. Replace the fixture-led API repository with snapshot-backed application services.
9. Rebuild the landing page, evidence workspace, and vehicle search over real snapshots.
10. Complete clean-room, security, contract, browser, accessibility, responsive, and
    visual verification; update documentation and present the local app for review.

Each stage is test-driven and independently reviewable. Downloading public releases is
an explicit acquisition step; no production deployment, push, merge, or proprietary
data operation is implied.

## Acceptance criteria

The real-data EU slice is complete only when freshly demonstrated that:

1. Approved EEA, KBA, and UK source releases are manifest-backed, checksummed,
   reproducible, and legally documented for local use.
2. The clean-room pipeline produces the same published snapshot for identical inputs,
   versions, configuration, and seeds.
3. Every UI value resolves to immutable observations or versioned estimates/forecasts.
4. Correlated sources are not counted as independent confidence evidence.
5. Conflicts remain visible and are never silently averaged.
6. Observed, reconciled, estimated, and forecast values cannot be confused in API or
   UI contracts.
7. Low-confidence 1995-2009 estimates expose observed country inputs, limitations, and
   wider uncertainty; unsupported cases return `insufficient_evidence`.
8. Active fleet uses cohort survival rather than a hidden constant attrition shortcut.
9. Replacement opportunity uses explicit hazard assumptions and remains labelled
   assumption-led until calibrated against defensible outcome history.
10. 2028 and 2031 P10/P50/P90 forecasts record backtests, method versions, input
    evidence, and experimental status when validation is insufficient.
11. The authenticated landing page shows the complete EU ranking, while vehicle search
    and evidence detail use the same snapshot.
12. Source/build failure preserves the last known-good snapshot and displays freshness;
    no fixture fallback enters the real-data product.
13. All parser, data-quality, domain, forecast, API, frontend, contract, security,
    accessibility, responsive, dependency, and clean-room gates pass.
14. Desktop and mobile visual review confirms that evidence states, confidence,
    uncertainty, warnings, and provenance are clear and usable.
15. The protected production checkout/deployment remains unchanged, and nothing is
    pushed, merged, or deployed without explicit authorization.
