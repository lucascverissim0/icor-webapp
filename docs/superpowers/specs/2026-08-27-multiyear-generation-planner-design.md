# Multi-year generation-aware ICOR planner design

Date: 2026-08-27

Status: approved in conversation; pending written-spec review

## Objective

Replace the demonstration planner and opportunity data with one auditable,
multi-year, EU-first evidence snapshot. The product must map every usable canonical
vehicle observation to one deterministic vehicle-generation approximation, expose the
quality of that approximation, reconstruct active cohorts, and calculate real-data-
backed generation-level windshield replacement opportunities.

The system is designed to maximize useful generation coverage for later machine
learning without silently turning approximations into manufacturer-confirmed facts.
It retains the approved long-term geography-neutral architecture so additional world
markets can be added after the EU-first product is complete and reviewed.

## User decisions

- Use first-registration year as credible proxy evidence for generation assignment.
- Produce one concrete generation ID per usable canonical vehicle observation.
- Prefer broad mapping coverage, while retaining confidence, method, alternatives,
  and provenance for every approximation.
- Use government registration and type-approval sources first, manufacturer archives
  second, and reputable licensed open registries when independently corroborated.
- Do not accept forums, dealer listings, or unsupported AI-generated mappings as
  evidence or training truth.
- During unresolved generation overlaps, apply one deterministic tie-break rather
  than leaving probabilistic output as the product label.
- Ingest every legally reusable historical release that can be validated; “all data”
  does not include untraceable scraped values.
- Remove demonstration records from the planner and opportunities once the complete
  real-data snapshot passes the defined gates. Never fall back to demo data.

## Scope

### Included

- A complete inventory of discovered, legally reusable EU and supporting European
  historical registration, fleet, vehicle-identity, and generation evidence.
- Finalized EEA passenger-car annual releases available from 2010 onward.
- Separately labelled provisional EEA releases when retained for review.
- Existing UK DfT/DVLA first-registration history from 2001 through 2025 and active-
  fleet history from 1994 through 2025.
- Available KBA annual/model-series releases and additional licensed national sources
  that pass source review and parser validation.
- A versioned, market-aware canonical vehicle and generation registry.
- Deterministic generation resolution for every usable canonical observation.
- Registration-cohort history, active-fleet reconstruction, and generation-level
  replacement-opportunity estimates with uncertainty.
- Snapshot-backed registrations, planner, opportunities, evidence, and ML-export
  workflows.
- Completeness reporting, reproducible acquisition/build/promotion, and full product
  verification.

### Excluded

- Claiming that first-registration year is a manufacturer-confirmed model year.
- Claiming exact windshield fitment or SKU identity without the proprietary ICOR
  fitment catalogue or equivalent authoritative evidence.
- Training or presenting an opaque ML forecast before deterministic baselines,
  temporal backtests, and leakage controls exist.
- Using LLM output as source evidence, a generation label, a forecast input, or a
  training target.
- Production deployment, pushing, merging to the protected main branch, or modifying
  the protected production checkout without explicit authorization.
- Declaring worldwide coverage complete as part of this EU-first milestone.

## Terminology and year semantics

- **Registration cohort year:** year in which the source reports a vehicle first
  registered. This is the primary broad-coverage proxy for generation assignment.
- **Manufacture year:** source-reported year of manufacture. It remains separate from
  first registration.
- **Model year:** a manufacturer-defined model-year value, stored only when a source
  explicitly provides that meaning.
- **Generation:** a sourced or estimated interval for a canonical make/model family,
  market, and applicable body/facelift context.
- **Confirmed generation:** supported by manufacturer or exact type-approval/
  type/variant/version evidence.
- **Estimated generation:** a stable chronological class inferred from supported
  evidence when no authoritative named generation is available.
- **Usable observation:** a valid passenger-vehicle observation with sufficient make,
  model-family, geography, and period identity to enter the canonical registry.

Registration cohort, manufacture year, and model year are separate fields in storage,
API contracts, exports, and UI labels. A derived generation never changes the raw
publisher fields.

## Source policy and acquisition

### Source hierarchy

1. Government registration, licensing, and type-approval data.
2. Manufacturer generation histories, press archives, catalogues, and homologation
   evidence.
3. Reputable licensed open registries whose relevant mapping is corroborated by an
   independent source.

Each source must have a recorded publisher, URL, retrieval and publication times,
coverage, measure semantics, dependency group, revision state, licence/terms,
artifact checksum, parser/version, expected schema, accepted/rejected counts, and
aggregate validations. A source without clear permitted use remains in the discovery
inventory but cannot enter a published snapshot.

### Release inventory

The acquisition workflow records every discovered candidate release as one of:

- acquired and validated;
- acquired and quarantined;
- unavailable or superseded;
- excluded for incompatible semantics;
- excluded for licence/redistribution constraints; or
- pending manual source review.

This inventory is the basis of the final “all data found” report. Absence is explicit;
the system never implies that undiscovered or inaccessible data was acquired.

### Immutability and safety

Artifacts are checksum-pinned and stored immutably. Parser failure, schema drift,
aggregate mismatch, invalid licence metadata, or an incomplete release blocks the
candidate build. The active snapshot is changed only through validated atomic
promotion; the last known-good snapshot remains available on failure.

Final and provisional evidence never silently merge. Dependency groups prevent a
national publication and its EEA republication from being counted as independent
vehicles or independent confidence evidence.

## Architecture

The existing modular monolith remains. The expanded flow is:

```text
source discovery inventory
  -> immutable release artifacts and manifests
  -> source-specific parsers
  -> normalized observations with separate year semantics
  -> canonical make/model registry
  -> versioned generation evidence registry
  -> deterministic generation resolver
  -> reconciliation and confidence
  -> cohort history and active-fleet reconstruction
  -> replacement-opportunity estimation and uncertainty
  -> versioned candidate snapshot
  -> validation and atomic promotion
  -> FastAPI services
  -> React registrations/planner/opportunities/evidence views
  -> versioned ML export
```

The units communicate through explicit domain/application protocols. Source parsers
do not invoke UI code. The frontend performs no business forecasting, generation
assignment, reconciliation, or confidence calculation.

## Data contracts

### Canonical vehicle family

- stable canonical make and model-family IDs;
- display and normalized labels;
- market/geography applicability;
- reviewed source aliases;
- body styles and technical descriptors when available; and
- lifecycle and version metadata.

### Generation registry entry

- stable generation ID;
- canonical make/model-family ID;
- sourced generation name or stable estimated-generation label;
- market and geography applicability;
- start/end month and their precision;
- body style, facelift, platform, and technical descriptors when available;
- predecessor/successor relationships;
- evidence references and dependency groups;
- whether the identity is manufacturer-confirmed, registry-corroborated, or
  algorithmically estimated;
- confidence band and reasons; and
- registry/resolver versions.

An estimated generation never receives an invented official designation. It uses a
stable label such as `estimated-generation-3 (2014-2019)` until stronger reviewed
evidence supports a named identity.

### Generation assignment

- source observation and canonical vehicle IDs;
- selected generation ID;
- registration cohort, manufacture year, and model year as separate nullable fields;
- all viable alternative generation IDs considered;
- assignment method and evidence references;
- confidence band, reason codes, and ML training weight;
- resolver and registry versions; and
- review state and timestamp.

### ML export row

- snapshot and schema versions;
- geography and source-period semantics;
- canonical make/model/generation identity;
- raw and normalized year fields;
- technical descriptors available at the prediction date;
- observation/reconciliation status;
- generation method, confidence, alternatives, and training weight;
- historical cohort/fleet features;
- forecast/target definition and method versions when present; and
- complete source-lineage identifiers.

## Deterministic generation resolution

The resolver assigns one generation to every usable canonical observation in this
order:

1. Match exact reviewed manufacturer or type-approval/type/variant/version evidence.
2. Use model variant, body, fuel, trim, platform, dimensions, powertrain, and other
   source descriptors to narrow candidates.
3. Match first-registration date to market-specific generation and facelift windows.
4. If exactly one generation is active, select it.
5. If generations overlap, prefer the candidate supported by detailed descriptors.
6. Otherwise choose the generation active for the greatest number of months in the
   registration year.
7. If still tied and the successor had officially launched in that market, select the
   newer generation.

The tie-break is deterministic and covered by fixtures for launches, run-outs,
facelifts, concurrent body styles, and boundary years. The assignment retains every
candidate and why it lost, even though the product and ML export receive one selected
generation.

### Estimated generation creation

When no sourced generation window exists, the pipeline creates chronological
estimated generations from corroborated evidence. Signals include:

- first-registration continuity and discontinuities;
- appearance/disappearance of type-approval and source model identifiers;
- body/platform/dimension changes;
- powertrain and model-name structure changes;
- manufacturer or open-registry lifecycle boundaries; and
- independent national-source overlap.

The estimator must not split or join generations from one unexplained statistical
change. Each boundary records its signals and evidence. Where evidence is sparse, a
single broad estimated generation is preferable to fabricated precision. Re-running
unchanged inputs and versions produces identical IDs and boundaries.

### Confidence and training use

Confidence is explainable rather than cosmetic:

- confirmed: exact authoritative identity evidence;
- high: one generation uniquely covers the registration period with credible window
  evidence;
- medium: descriptor-supported overlap or corroborated estimated boundary;
- low: deterministic transition or sparse-evidence estimated assignment.

Every usable observation remains mapped. Low-confidence records are visible and
exported, but ML consumers can filter or down-weight them using the stored training
weight. Confidence and weight formulas are versioned and must be calibrated from
actual coverage/error review rather than silently hard-coded as universal truth.

## Reconciliation and historical coverage

Only semantically equivalent values reconcile. Registration totals, active-fleet
stock, manufacture-year cohorts, and model-year values remain distinct measures.

Equivalent source observations use deterministic precedence: final over provisional,
direct administrative evidence over derivative publication, complete over partial,
current correction over superseded, and precise identity/period over coarse. Conflicts
remain visible and are not averaged automatically. National and EEA publications from
the same register cannot double-count registrations.

The published product exposes observed, reconciled, estimated, and forecast values as
different states. Missing historical EU model coverage may be estimated from national
evidence only through versioned methods with explicit geography, coverage, and
uncertainty; a national value is never relabelled as an observed EU total.

## Active-fleet and replacement opportunity

Registration cohorts feed market/segment survival curves to reconstruct active fleet
by geography, make, model, assigned generation, and cohort year. Observed fleet
evidence validates and calibrates the reconstruction where available.

Generation-level windshield replacement opportunity applies an explicit,
versioned age/geography/vehicle hazard distribution to reconstructed fleets.
Uncertainty from source coverage, generation assignment, historical estimation,
survival, hazard, and horizon propagates to P10/P50/P90 output.

Until proprietary fitment truth is integrated:

- output stops at generation/body/facelift resolution;
- the UI states that multiple incompatible windshields may remain within a generation;
- no exact ICOR SKU or fitment match is inferred; and
- opportunity remains assumption-led and is not described as a calibrated production
  forecast.

## Product behavior

### Runtime composition

The promoted multi-year snapshot replaces `demo-planner-v1` for registrations,
planner, and opportunities. Runtime composition has no demo fallback. If the snapshot
is missing, invalid, stale beyond policy, or incomplete, affected pages show a typed
unavailable/stale state while provenance remains inspectable.

### Registrations

The official-data view supports geography, registration year, make/model,
generation, assignment confidence, and observed/estimated status. It shows source
coverage and exact year semantics.

### Planner

The planner shows historical registrations, generation assignment and evidence,
cohort-derived active fleet, assumptions, confidence, P10/P50/P90 opportunity, and
forecast horizons. Users can drill from a generation into its source observations and
mapping decision.

### Opportunities

Opportunities ranks real generation-level replacement demand. Raw demand remains
separate from ICOR production readiness. Coverage management cannot claim exact
configuration/SKU precision until proprietary fitment identity is integrated.

### Evidence and completeness

The evidence workspace exposes release inventory, source status, raw observations,
generation mappings, alternatives, confidence reasons, exclusions, and version
metadata. A completeness view reports, by geography and year:

- acquired and excluded releases;
- registrations and active-fleet observations;
- canonical make/model families;
- generation assignments by confidence;
- sourced versus estimated generations;
- rejected/invalid records and reasons; and
- forecastable versus evidence-only coverage.

### ML export

The export uses the same promoted snapshot and is immutable/versioned. It never
contains UI-derived values or LLM labels. Temporal feature materialization excludes
evidence published after the prediction cutoff to prevent look-ahead leakage.

## Failure handling

- Network/source failure cannot modify an existing stored artifact or active snapshot.
- Schema drift quarantines the release and identifies the failed contract.
- Invalid generation windows, cycles, impossible dates, or unstable IDs block build.
- Ambiguous raw labels that cannot form a canonical model remain rejected with reasons;
  they do not receive fabricated vehicle identities.
- Every usable canonical vehicle must receive a sourced or estimated generation ID;
  otherwise candidate promotion fails.
- API errors are typed and do not expose filesystem paths, tracebacks, credentials, or
  private data.
- No failure path falls back to fixture/demo planner data.

## Verification

### Data and domain tests

- parser contracts for every release schema and revision;
- checksums, byte ceilings, licensing metadata, and aggregate reconciliation;
- make/model alias normalization and stable canonical IDs;
- exact, unique-window, overlap, run-out, facelift, concurrent-body, and fallback
  generation resolution;
- stable estimated-generation boundaries and IDs;
- 100% generation assignment for usable canonical observations;
- duplicate/dependency-group protection and conflict visibility;
- observed/reconciled/estimated/forecast separation;
- cohort conservation, survival, hazard, and P10/P50/P90 invariants; and
- temporal-cutoff/leakage tests for ML exports and backtests.

### Application and UI tests

- snapshot-backed API contracts and OpenAPI/frontend type agreement;
- no runtime reference or fallback to demo planner records;
- URL-addressable filters and drill-down from opportunity to evidence;
- loading, empty, unavailable, stale, conflict, and retry behavior;
- confidence and approximation language that cannot be confused with confirmed
  fitment;
- keyboard, screen-reader, mobile/desktop, and serious/critical accessibility checks;
  and
- browser journeys using the real candidate snapshot.

### Clean-room and operational gates

An empty derived-data directory is rebuilt twice from the approved immutable local
artifacts and versioned configuration. Snapshot identity, database digest, aggregate
counts, generation assignments, and published output must match exactly. Full Python,
lint/type, dependency/security, frontend tests, production build, OpenAPI drift,
browser, accessibility, responsive, and visual-review gates must pass before local
promotion and completion reporting.

## Completion criteria

The milestone is complete only when fresh evidence shows:

1. The source inventory records every discovered candidate release and its outcome.
2. Every legally usable accepted release included in scope is checksum-pinned,
   manifest-backed, parsed, and validated.
3. Every usable canonical observation has one deterministic generation ID with method,
   alternatives, evidence, confidence, and resolver version.
4. Exact publisher totals reconcile within documented tolerances and correlated
   sources do not double-count vehicles.
5. The final report gives exact years, geographies, releases, vehicle/observation
   counts, mapping confidence distribution, estimated-generation counts, exclusions,
   and known limitations.
6. Registrations, planner, opportunities, evidence, and ML export use the same promoted
   multi-year snapshot.
7. No demonstration vehicle or `demo-planner-v1` fallback is reachable from planner
   or opportunities runtime paths.
8. Planner and opportunities show real generation-level cohort/fleet evidence and
   assumption-led P10/P50/P90 replacement opportunity.
9. No UI or API implies exact windshield/SKU fitment without authoritative fitment
   evidence.
10. Clean-room reproducibility and all verification gates pass with fresh output.
11. The local app is restarted on the reviewed snapshot and every major route is
    live-verified before delivery.
12. The protected production checkout/deployment remains unchanged and nothing is
    pushed, merged, or deployed without Lucas's explicit authorization.

## Delivery sequence

1. Expand source discovery/acquisition and generalize versioned annual parsers.
2. Extend year semantics, canonical vehicle, generation evidence, assignment, and
   snapshot schemas through forward-only migrations.
3. Build the market-aware generation registry and deterministic resolver.
4. Rebuild/reconcile the multi-year evidence candidate and publish completeness
   diagnostics.
5. Implement cohort reconstruction and versioned replacement-opportunity baselines.
6. Replace planner/opportunity demo composition with snapshot-backed services.
7. Update registrations, evidence, planner, opportunities, and ML export together.
8. Run clean-room, full automated, browser/accessibility, and visual verification.
9. Promote only the validated local snapshot, restart the local app, and provide the
   exact completion report required above.

Each stage is test-driven, independently reviewable, and preserves the active known-
good snapshot until its replacement passes the full applicable gates.
