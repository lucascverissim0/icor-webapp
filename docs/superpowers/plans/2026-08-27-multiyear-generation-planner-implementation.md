# Multi-year Generation-aware ICOR Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the demonstration planner and opportunities with one deterministic, auditable, locally promoted multi-year EU evidence snapshot that assigns every usable canonical observation to a generation approximation and exposes cohort-based P10/P50/P90 windshield replacement opportunity.

**Architecture:** Extend the existing modular monolith through forward-only SQLite migrations and focused domain/application protocols. Official source adapters populate immutable observations; deterministic registry, generation, cohort, and opportunity transforms publish derived records into the same snapshot consumed by registrations, evidence, planner, opportunities, completeness, and ML export APIs. The React client remains presentation-only and every runtime path fails closed if the active snapshot is absent or invalid.

**Tech Stack:** Python 3.12+, dataclasses/Decimal/SQLite/FastAPI/Pydantic/pytest/Ruff; React 19, TypeScript, TanStack Router/Query, Vitest, Playwright, axe; uv and npm locked dependencies.

**Spec:** `docs/superpowers/specs/2026-08-27-multiyear-generation-planner-design.md`

## Global Constraints

- Work only in `C:\Users\LucasCravoVERISSIMO\icor-webapp-development`; preserve unrelated changes to `AGENTS.md` and `docs/CODEX_HANDOFF.md`.
- Do not push, merge, deploy, modify the protected production checkout, or use production/customer secrets or data.
- Keep registration cohort year, manufacture year, and manufacturer-defined model year separate in storage, API, export, and UI language.
- Use LLM output for neither evidence nor generation labels, calculation inputs, forecasts, or training targets.
- Every usable canonical observation receives one deterministic generation ID with method, alternatives, provenance, confidence, resolver version, and training weight; rejected non-canonical observations remain rejected.
- Never infer exact windshield configuration or SKU without authoritative fitment evidence; generation-level output remains explicitly assumption-led.
- Runtime composition has no `demo-planner-v1` fallback. Invalid, missing, stale, or incomplete snapshots return typed unavailable/stale states.
- Preserve immutable release artifacts and atomic last-known-good snapshot promotion. A failed build cannot change the active pointer.
- Derived identifiers and outputs are byte-for-byte deterministic for identical artifacts, versions, `build_as_of`, and seed.

---

### Task 1: Multi-year Release Inventory and Source Contracts

**Files:**
- Create: `src/icor/domain/source_inventory.py`
- Create: `src/icor/evidence/source_inventory.py`
- Modify: `src/icor/evidence/acquisition.py`
- Modify: `src/icor/evidence/sources/eea.py`
- Modify: `src/icor/evidence/sources/kba.py`
- Modify: `src/icor/evidence/sources/uk_dft.py`
- Modify: `scripts/acquire_official_evidence.py`
- Test: `tests/domain/test_source_inventory.py`
- Test: `tests/evidence/test_source_inventory.py`
- Test: `tests/evidence/sources/test_eea.py`
- Test: `tests/evidence/sources/test_kba.py`
- Test: `tests/evidence/sources/test_uk_dft.py`

**Interfaces:**
- Produces: `SourceInventoryEntry(source_key, period_start, period_end, status, release_id, revision_state, licence_status, reason_codes)` and `SourceInventory.entries() -> tuple[SourceInventoryEntry, ...]`.
- Produces: versioned EEA annual, KBA annual/model-series, UK first-registration, and UK active-fleet loaders that preserve separate year semantics.

- [ ] **Step 1: Write failing source-inventory and annual-parser tests.**

```python
def test_inventory_records_every_discovered_release_outcome():
    entries = inventory.entries()
    assert {entry.status for entry in entries} >= {
        InventoryStatus.ACQUIRED_VALIDATED,
        InventoryStatus.EXCLUDED_LICENCE,
        InventoryStatus.UNAVAILABLE,
    }
    assert all(entry.reason_codes for entry in entries if entry.release_id is None)

def test_first_registration_does_not_become_model_year(parsed_observation):
    assert parsed_observation.registration_cohort_year == 2020
    assert parsed_observation.manufacture_year is None
    assert parsed_observation.model_year is None
```

- [ ] **Step 2: Run `uv run pytest tests/domain/test_source_inventory.py tests/evidence/test_source_inventory.py tests/evidence/sources -v` and confirm the missing contracts and annual parsing behavior fail.**
- [ ] **Step 3: Implement immutable inventory contracts, canonical JSON persistence, explicit discovery outcomes, and parser support for validated annual releases without weakening existing 2024/UK aggregate checks.**
- [ ] **Step 4: Acquire only reviewed, legally reusable releases through the allowlisted acquisition boundary; checksum-pin artifacts and quarantine schema/terms failures.**
- [ ] **Step 5: Re-run focused tests, inspect source totals against publisher metadata, and commit `feat: inventory multiyear official evidence`.**

### Task 2: Forward-only Snapshot Schema for Generations and Derived Products

**Files:**
- Modify: `src/icor/domain/evidence.py`
- Create: `src/icor/domain/generations.py`
- Create: `src/icor/domain/cohorts.py`
- Modify: `src/icor/domain/snapshots.py`
- Modify: `src/icor/infrastructure/sqlite_evidence_repository.py`
- Modify: `src/icor/application/snapshot_build.py`
- Modify: `src/icor/evidence/validation.py`
- Test: `tests/domain/test_generations.py`
- Test: `tests/domain/test_cohorts.py`
- Modify: `tests/infrastructure/test_sqlite_evidence_repository.py`
- Modify: `tests/evidence/test_validation.py`

**Interfaces:**
- Produces: `GenerationEntry`, `GenerationAlternative`, `GenerationAssignment`, `CohortEstimate`, `OpportunityEstimate`, and `CompletenessRecord` frozen domain contracts.
- Repository adds immutable batch writes and deterministic list/query methods for generation evidence, assignments, cohorts, opportunity intervals, completeness, and export lineage.

- [ ] **Step 1: Write failing domain and schema-migration tests.**

```python
def test_assignment_retains_one_selection_and_ranked_alternatives():
    assignment = generation_assignment(selected="gen-golf-8", alternatives=("gen-golf-7",))
    assert assignment.selected_generation_id not in assignment.alternative_generation_ids
    assert assignment.training_weight == Decimal("0.55")

def test_opportunity_interval_is_ordered_and_non_negative():
    estimate = opportunity_estimate(p10="8", p50="10", p90="14")
    assert estimate.p10 <= estimate.p50 <= estimate.p90
```

- [ ] **Step 2: Run the focused domain/repository tests and confirm RED for missing contracts and schema tables.**
- [ ] **Step 3: Add a forward-only SQLite migration with exact schema fingerprints, foreign keys, immutable inserts, stable indexes, and separate registration/manufacture/model-year columns.**
- [ ] **Step 4: Extend snapshot identity versions and validation so missing assignments, orphan alternatives, invalid windows/cycles, unstable IDs, bad confidence weights, and unordered/negative opportunity intervals block promotion.**
- [ ] **Step 5: Re-run focused tests and commit `feat: extend snapshot for generation planning`.**

### Task 3: Versioned Market-aware Generation Registry and Resolver

**Files:**
- Create: `src/icor/generations/registry.py`
- Create: `src/icor/generations/resolver.py`
- Create: `src/icor/generations/estimator.py`
- Create: `src/icor/generations/confidence.py`
- Create: `src/icor/application/generation_mapping.py`
- Test: `tests/generations/test_registry.py`
- Test: `tests/generations/test_resolver.py`
- Test: `tests/generations/test_estimator.py`
- Test: `tests/application/test_generation_mapping.py`

**Interfaces:**
- Produces: `GenerationRegistry.candidates(vehicle_id, geography, registration_date) -> tuple[GenerationEntry, ...]`.
- Produces: `GenerationResolver.resolve(observation, candidates) -> GenerationAssignment` with methods `exact_identifier`, `descriptor_overlap`, `unique_window`, `active_month_coverage`, `newer_launch_tiebreak`, and `estimated_generation`.
- Produces: `EstimatedGenerationBuilder.build(vehicle_history) -> tuple[GenerationEntry, ...]` with stable chronological IDs.

- [ ] **Step 1: Write failing fixtures for exact identifiers, unique windows, transition overlaps, run-outs, facelifts, concurrent bodies, newer-launch ties, sparse histories, and deterministic estimated IDs.**

```python
def test_transition_year_uses_active_month_coverage_before_newer_tie_break():
    result = resolver.resolve(observation(cohort_year=2020), (old_generation(end="2020-09"), new_generation(start="2020-10")))
    assert result.selected_generation_id == "generation-old"
    assert result.method is AssignmentMethod.ACTIVE_MONTH_COVERAGE

def test_every_usable_observation_gets_one_assignment(mapping_result):
    assert mapping_result.usable_count == mapping_result.assigned_count
    assert mapping_result.unassigned_ids == ()
```

- [ ] **Step 2: Run `uv run pytest tests/generations tests/application/test_generation_mapping.py -v` and confirm RED.**
- [ ] **Step 3: Implement registry validation, deterministic precedence, loss reasons for alternatives, confidence reason codes, versioned training weights, and stable estimated-generation boundaries without invented official names.**
- [ ] **Step 4: Add batch transformation to canonical observations and make 100% assignment of usable observations a candidate-build invariant.**
- [ ] **Step 5: Re-run focused tests, mutation-check the overlap precedence, and commit `feat: resolve deterministic vehicle generations`.**

### Task 4: Reconciliation, Cohort Reconstruction, and Opportunity Baseline

**Files:**
- Create: `src/icor/forecasting/reconciliation.py`
- Create: `src/icor/forecasting/survival.py`
- Create: `src/icor/forecasting/registration_forecast.py`
- Create: `src/icor/forecasting/replacement_hazard.py`
- Create: `src/icor/forecasting/uncertainty.py`
- Create: `src/icor/application/generation_planning.py`
- Test: `tests/forecasting/test_reconciliation.py`
- Test: `tests/forecasting/test_survival.py`
- Test: `tests/forecasting/test_registration_forecast.py`
- Test: `tests/forecasting/test_replacement_hazard.py`
- Test: `tests/forecasting/test_uncertainty.py`
- Test: `tests/application/test_generation_planning.py`

**Interfaces:**
- Produces: deterministic dependency-aware reconciled registration history with observed/reconciled/estimated/forecast status retained.
- Produces: `CohortReconstructor.reconstruct(...) -> tuple[CohortEstimate, ...]` and `OpportunityModel.estimate(..., horizon: Literal[2028, 2031]) -> OpportunityEstimate`.

- [ ] **Step 1: Write failing conservation, dependency-group, temporal backtest, survival, hazard, and interval tests.**

```python
def test_one_year_old_cohort_receives_one_full_year_of_attrition():
    assert survival.remaining(Decimal("1000"), age_years=1) == Decimal("944.4")

def test_dependency_group_is_not_double_counted(reconciler):
    result = reconciler.reconcile((eea_value("100"), national_republication("100")))
    assert result.value == Decimal("100")
    assert result.independent_evidence_count == 1

def test_uncertainty_is_seeded_ordered_and_reproducible(model):
    first = model.estimate(inputs, horizon=2028, seed=20260827)
    assert first == model.estimate(inputs, horizon=2028, seed=20260827)
    assert first.p10 <= first.p50 <= first.p90
```

- [ ] **Step 2: Run focused forecasting/application tests and confirm RED.**
- [ ] **Step 3: Implement deterministic precedence, explicit estimates for coverage gaps, geography/segment survival curves, simple forecast candidates selected by rolling-origin error, age/geography hazard assumptions, and seeded uncertainty propagation.**
- [ ] **Step 4: Materialize generation-level history, active fleet, P10/P50/P90 2028/2031 opportunity, confidence, assumptions, and reproducibility versions into the candidate repository.**
- [ ] **Step 5: Run focused tests and commit `feat: calculate generation opportunity baseline`.**

### Task 5: Snapshot-backed APIs, Completeness, and ML Export

**Files:**
- Create: `src/icor/application/completeness.py`
- Create: `src/icor/application/ml_export.py`
- Create: `src/icor/infrastructure/snapshot_planner_repository.py`
- Modify: `src/icor/application/planner.py`
- Modify: `src/icor/application/opportunities.py`
- Modify: `src/icor/application/registrations.py`
- Modify: `src/icor/application/evidence_review.py`
- Modify: `src/icor/api/schemas.py`
- Modify: `src/icor/api/app.py`
- Modify: `src/icor/api/planner.py`
- Modify: `src/icor/api/opportunities.py`
- Create: `src/icor/api/completeness.py`
- Create: `src/icor/api/exports.py`
- Test: `tests/infrastructure/test_snapshot_planner_repository.py`
- Modify: `tests/api/test_planner_api.py`
- Modify: `tests/api/test_opportunity_api.py`
- Create: `tests/api/test_completeness_api.py`
- Create: `tests/api/test_ml_export_api.py`

**Interfaces:**
- Runtime composition opens exactly one active `SnapshotStore`; all product services report its snapshot/data/method versions.
- Adds `GET /api/completeness` and `GET /api/exports/ml.csv?cutoff=YYYY-MM-DD`; planner/opportunity contracts expose generation identity, year semantics, history, cohort fleet, intervals, assumptions, confidence, and evidence drill-down IDs.

- [ ] **Step 1: Replace demo API expectations with failing real-snapshot contract tests and add a repository search proving no runtime import/reference to `DemoPlannerRepository` or `demo-planner-v1`.**
- [ ] **Step 2: Run API/repository tests and confirm RED.**
- [ ] **Step 3: Implement read-only snapshot queries, bounded filters/pagination, typed unavailable/stale/incomplete errors, completeness aggregation, deterministic CSV export, and temporal-cutoff exclusion of post-cutoff evidence.**
- [ ] **Step 4: Regenerate OpenAPI and TypeScript types; assert every route exposes the same snapshot/version set and no demo fallback is reachable.**
- [ ] **Step 5: Run focused API, export, OpenAPI drift, and runtime-composition tests; commit `feat: serve generation planning snapshot`.**

### Task 6: React Product Workflows

**Files:**
- Modify: `web/src/app/AppShell.tsx`
- Modify: `web/src/app/router.tsx`
- Modify: `web/src/app/styles.css`
- Modify: `web/src/lib/api/client.ts`
- Modify: `web/src/features/registrations/RegistrationsPage.tsx`
- Modify: `web/src/features/evidence/EvidencePage.tsx`
- Modify: `web/src/features/planner/PlannerPage.tsx`
- Modify: `web/src/features/planner/ConfigurationDetail.tsx`
- Modify: `web/src/features/opportunities/OpportunitiesPage.tsx`
- Modify: `web/src/features/opportunities/OpportunityRanking.tsx`
- Create: `web/src/features/completeness/CompletenessPage.tsx`
- Create: `web/src/features/exports/ExportPage.tsx`
- Modify: `web/tests/*.test.tsx`
- Modify: `web/e2e/*.spec.ts`

**Interfaces:**
- All filters are URL-addressable; opportunity rows link to generation evidence; mobile retains provenance; observed/estimated/forecast states include text labels and never rely on color alone.

- [ ] **Step 1: Write failing component/router tests for snapshot-backed copy, generation/year/confidence filters, evidence drill-down, completeness, export, and unavailable/stale/conflict states.**
- [ ] **Step 2: Run `npm test -- --run` in `web` and confirm RED.**
- [ ] **Step 3: Implement the planner workbench visual hierarchy with concrete real-data content, responsive tables/cards, keyboard-operable controls, accessible names/live regions, approximation warnings, and generation—not SKU—claims.**
- [ ] **Step 4: Update Playwright journeys for 390px and 1440px routes, keyboard reachability, URL reload, zero serious/critical axe findings, and opportunity-to-evidence navigation.**
- [ ] **Step 5: Run Vitest, TypeScript, ESLint, production build, and focused Chromium journeys; commit `feat: deliver generation planning workbench`.**

### Task 7: Deterministic Real-data Build and Local Promotion

**Files:**
- Modify: `scripts/build_evidence_snapshot.py`
- Create: `scripts/report_snapshot_completeness.py`
- Modify: `docs/DEVELOPMENT.md`
- Modify: `README.md`
- Test: `tests/integration/test_multiyear_generation_snapshot.py`
- Test: `tests/integration/test_clean_room_multiyear_snapshot.py`

**Interfaces:**
- The build pipeline composes official loaders plus canonical identity, generation mapping, reconciliation, cohort, and opportunity transformers behind versioned registry names.
- The report emits canonical JSON with releases, exact years/geographies/counts, confidence distribution, sourced/estimated generations, exclusions, limitations, snapshot ID, and database digest.

- [ ] **Step 1: Write failing clean-room tests requiring two byte-identical builds, 100% usable generation assignment, coherent published products, and exact completeness reporting.**
- [ ] **Step 2: Run the integration tests and confirm RED.**
- [ ] **Step 3: Build the candidate twice from approved immutable local artifacts with fixed `build_as_of` and seed; compare snapshot identity, database SHA-256, aggregate counts, assignments, and published values.**
- [ ] **Step 4: Run full candidate validation, inspect exclusions/conflicts and publisher totals, then atomically promote only if every required gate passes.**
- [ ] **Step 5: Verify the active pointer and read-only repository match the candidate exactly; commit `feat: finalize multiyear generation snapshot` without ignored data artifacts.**

### Task 8: Whole-product Verification and Durable Handoff

**Files:**
- Modify: `docs/CODEX_HANDOFF.md`
- Modify: `README.md`
- Modify: `docs/DEVELOPMENT.md`

**Interfaces:**
- Final handoff records exact release inventory, years/geographies, counts, confidence distribution, limitations, verification commands/results, active snapshot, local process/URL state, and production-isolation evidence.

- [ ] **Step 1: Run `uv lock --check`, maintained Ruff, full pytest, dependency audit, OpenAPI drift, all frontend unit tests, TypeScript, ESLint, npm audit, and production build.**
- [ ] **Step 2: Run clean-room rebuild verification and the complete real-snapshot Playwright/accessibility/responsive suite serially.**
- [ ] **Step 3: Search runtime/source bundles for forbidden demo fallback and unsupported exact-fitment claims; verify ignored raw data remains untracked and `git diff --check` exits zero.**
- [ ] **Step 4: Restart the local app on the promoted snapshot, live-check health plus registrations, planner, opportunities, evidence, completeness, and export routes, and keep the review server running.**
- [ ] **Step 5: Verify the protected production checkout and deployment are unchanged, update the durable handoff, review it for secrets/placeholders, and mark the goal complete only after fresh evidence supports every completion criterion.**

## Self-Review

- Spec coverage: Tasks 1-2 cover source inventory, immutable multi-year/year-semantic storage, and forward-only schemas; Task 3 covers generation registry/resolution and 100% usable coverage; Task 4 covers reconciliation, cohorts, forecasting baseline, hazard, and uncertainty; Tasks 5-6 cover the shared snapshot APIs, all product workflows, completeness, and leakage-safe ML export; Tasks 7-8 cover deterministic builds, promotion, verification, local restart, reporting, and production isolation.
- Placeholder scan: no TBD/TODO, “similar to,” or unspecified error-handling/test steps remain.
- Type consistency: generation, cohort, opportunity, completeness, and export contracts originate in Tasks 2-4 and are consumed unchanged by Tasks 5-7; every runtime service consumes one active snapshot identity.
