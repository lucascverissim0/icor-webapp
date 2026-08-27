# Canonical Registration Product Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the web app's default demonstration landing experience with a fail-closed, snapshot-backed ranking of real 2024 EEA passenger-car registrations by canonical make and model, without inventing model years, fitment, or replacement forecasts.

**Architecture:** Keep the immutable source ledger and candidate/promotion machinery intact. Add a deterministic identity-attribution decorator that resolves only non-generic exact normalized make/model labels to canonical model families whose model year is explicitly unknown, then query those mapped observations through a read-only registration service with a versioned EEA-to-EU27 aggregation rule. Add a typed FastAPI boundary and a React landing page; the existing synthetic planner remains available only as a clearly separated prototype and is never used as fallback data.

**Tech Stack:** Python 3.12, dataclasses, SQLite, FastAPI/Pydantic, React 19, TypeScript 5.9, TanStack Router/Query, Vitest, Playwright, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-26-real-data-evidence-forecasting-design.md`

## Global Constraints

- Work only in `C:\Users\LucasCravoVERISSIMO\icor-webapp-development` on `development/windshield-demand-platform`.
- Do not modify the protected `main` checkout, push, merge, deploy, or use production/customer/proprietary data.
- Preserve original source labels and rows; identity attribution is additive and deterministic.
- Only exact normalized non-generic make/model labels may be published in this slice; fuzzy matching is prohibited.
- Registration period is not model year. Canonical model year remains `None` and the UI must say it is unavailable.
- EU27 totals use only the finalized EEA 2024 release and its EU27 member rows; KBA and UK records must not be silently double-counted into that total.
- Missing/invalid real snapshot configuration returns typed unavailable state; no demo fixture fallback is permitted on real-data routes.
- Every production behavior is developed RED-GREEN-REFACTOR and every commit preserves the baseline gates.

---

### Task 1: Represent Canonical Model Families Without Invented Model Years

**Files:**
- Modify: `src/icor/domain/evidence.py`
- Modify: `src/icor/infrastructure/sqlite_evidence_repository.py`
- Test: `tests/domain/test_evidence.py`
- Test: `tests/infrastructure/test_sqlite_evidence_repository.py`

**Interfaces:**
- Consumes: existing `CanonicalVehicle` and schema-v1 evidence repository.
- Produces: `CanonicalVehicle.model_year: int | None` and schema v2 with nullable `canonical_vehicle.model_year`, plus strict read-only validation compatibility for immutable schema-v1 snapshots and rejection of non-integer non-null years.

- [ ] **Step 1: Write failing domain and repository tests**

```python
def test_canonical_model_family_can_record_unknown_model_year():
    vehicle = CanonicalVehicle("vehicle-example-alpha", "Example", "Alpha", None, "EU")
    assert vehicle.model_year is None


def test_repository_round_trips_unknown_model_year(repository):
    vehicle = CanonicalVehicle("vehicle-example-alpha", "Example", "Alpha", None, "EU")
    repository.add_vehicle(vehicle)
    assert repository.get_vehicle(vehicle.vehicle_id) == vehicle
```

Also assert strings/floats remain invalid, new databases initialize at schema v2, and immutable schema-v1 databases remain readable without in-place mutation.

- [ ] **Step 2: Run RED verification**

Run: `.venv\Scripts\python.exe -m pytest tests/domain/test_evidence.py tests/infrastructure/test_sqlite_evidence_repository.py -q`

Expected: FAIL because `None` is rejected and schema v1 requires a non-null integer.

- [ ] **Step 3: Implement the nullable model-year contract and migration**

Change the dataclass field to `model_year: int | None`, validate only non-null values as integers, advance `_SCHEMA_VERSION` to `2`, and make new ledgers use nullable `model_year`. Retain the exact schema-v1 fingerprint for read-only compatibility; never rewrite an immutable old snapshot in place. Update the schema-v2 fingerprint and row decoder.

- [ ] **Step 4: Run GREEN verification**

Run: `.venv\Scripts\python.exe -m pytest tests/domain/test_evidence.py tests/infrastructure/test_sqlite_evidence_repository.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/icor/domain/evidence.py src/icor/infrastructure/sqlite_evidence_repository.py tests/domain/test_evidence.py tests/infrastructure/test_sqlite_evidence_repository.py
git commit -m "feat: represent canonical model families honestly"
```

---

### Task 2: Add Conservative Deterministic Identity Attribution

**Files:**
- Create: `src/icor/evidence/identity.py`
- Modify: `src/icor/evidence/source_registry.py`
- Modify: `src/icor/application/snapshot_build.py`
- Test: `tests/evidence/test_identity.py`
- Test: `tests/evidence/test_source_registry.py`
- Test: `tests/application/test_snapshot_build.py`

**Interfaces:**
- Consumes: `Observation`, `CanonicalVehicle`, `IdentityMapping`, `SQLiteEvidenceRepository.add_observations()`.
- Produces: `ExactNormalizedIdentityResolver.resolve(observation: Observation, *, reviewed_at: datetime) -> ResolvedIdentity` and `IdentityAttributingRepository.add_observations(observations: Sequence[Observation]) -> None`.

- [ ] **Step 1: Write failing resolver tests**

```python
def test_exact_normalized_labels_map_to_stable_model_family(observation):
    result = resolver.resolve(observation, reviewed_at=BUILD_AS_OF)
    assert result.vehicle.make == observation.original_make.strip()
    assert result.vehicle.model == observation.original_model.strip()
    assert result.vehicle.model_year is None
    assert result.observation.mapping_status is MappingStatus.NORMALIZED_LABEL
    assert result.mapping.status is MappingStatus.NORMALIZED_LABEL


@pytest.mark.parametrize("label", ["SONSTIGE", "OTHER", "UNKNOWN", "(not reported)"])
def test_generic_model_labels_never_publish(observation, label):
    result = resolver.resolve(replace(observation, original_model=label, normalized_model=label.casefold()), reviewed_at=BUILD_AS_OF)
    assert result.vehicle is None
    assert result.observation.mapping_status in {MappingStatus.REJECTED, MappingStatus.UNRESOLVED}
```

Cover stable IDs, whitespace/case normalization, absent normalized labels, preservation of originals, identity confidence/reasons, and no fuzzy/punctuation alias merge.

- [ ] **Step 2: Run RED verification**

Run: `.venv\Scripts\python.exe -m pytest tests/evidence/test_identity.py -q`

Expected: FAIL because the identity resolver does not exist.

- [ ] **Step 3: Implement resolver and repository decorator**

Use `stable_evidence_id("vehicle-model", normalized_make, normalized_model, "europe")` and `stable_evidence_id("mapping", observation.observation_id, "exact-normalized-v1")`. Use one deterministic display label from the first source observation only for presentation, never as an additional merge key. The decorator must insert each missing vehicle once, insert attributed observations in the existing batch, then insert exactly one selected mapping per resolved observation. Rejected/unresolved rows remain immutable and receive no publishable mapping.

- [ ] **Step 4: Write failing composition tests**

Assert `OFFICIAL_SOURCE_VERSIONS.identity_registry == "exact-normalized-model-family-v1"`, snapshot identity changes from the unresolved version, and official loaders receive the attribution decorator while fictional integration loaders remain unchanged.

- [ ] **Step 5: Implement production composition**

Add an optional repository transformer to `SnapshotBuilder`; production CLI composition supplies `IdentityAttributingRepository`, while tests and non-production registries can opt out explicitly. Set build review time from `SnapshotBuildRequest.build_as_of`, never wall-clock time.

- [ ] **Step 6: Run GREEN verification and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/evidence/test_identity.py tests/evidence/test_source_registry.py tests/application/test_snapshot_build.py -q`

Expected: PASS.

```powershell
git add src/icor/evidence/identity.py src/icor/evidence/source_registry.py src/icor/application/snapshot_build.py tests/evidence/test_identity.py tests/evidence/test_source_registry.py tests/application/test_snapshot_build.py
git commit -m "feat: map exact official vehicle identities"
```

---

### Task 3: Query Snapshot-Backed Official Registration Rankings

**Files:**
- Create: `src/icor/application/registrations.py`
- Test: `tests/application/test_registrations.py`

**Interfaces:**
- Consumes: a sealed candidate or active snapshot database with mapped EEA observations.
- Produces: `RegistrationService.from_candidate(path: Path)`, `RegistrationService.from_active(root: Path)`, `summary() -> RegistrationSummary`, and `ranking(query: RegistrationQuery) -> RegistrationPage`.

- [ ] **Step 1: Write failing service tests**

```python
def test_eu27_ranking_sums_only_final_eea_member_observations(mapped_candidate):
    page = RegistrationService.from_candidate(mapped_candidate).ranking(
        RegistrationQuery(geography="EU27", year=2024, search=None, page=1, page_size=25)
    )
    assert page.items[0].make == "Example Motors"
    assert page.items[0].registrations == Decimal("15")
    assert page.items[0].status == "derived_observed"
    assert page.items[0].model_year is None
    assert page.items[0].source_ids == ("eea-co2-cars",)


def test_eu27_ranking_excludes_kba_and_non_members(mapped_candidate):
    page = service.ranking(RegistrationQuery(geography="EU27", year=2024))
    assert page.total_registrations == Decimal("15")
```

Cover exact snapshot checksum/schema verification, EU27 member allow-list, source/release allow-list, search escaping, deterministic rank ties, page bounds, confidence aggregation, source/input counts, `None` model year, unsupported year/geography, and typed unavailable state.

- [ ] **Step 2: Run RED verification**

Run: `.venv\Scripts\python.exe -m pytest tests/application/test_registrations.py -q`

Expected: FAIL because the service does not exist.

- [ ] **Step 3: Implement the read-only service**

Open SQLite with `mode=ro`, `PRAGMA query_only = ON`, and the same manifest/database/release verification used by evidence review. For EU27/2024, select only `source_id = 'eea-co2-cars'`, final `new_registrations`, publishable normalized mappings, and the versioned EU27 ISO-alpha-2 set. Group by canonical vehicle, sum raw observed values, compute a weighted evidence-confidence total capped to valid ranges, order by registrations descending then make/model/ID, and expose input observation count plus release IDs. Do not query `DemoPlannerRepository`.

- [ ] **Step 4: Run GREEN verification and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/application/test_registrations.py -q`

Expected: PASS.

```powershell
git add src/icor/application/registrations.py tests/application/test_registrations.py
git commit -m "feat: query official registration rankings"
```

---

### Task 4: Expose a Typed Fail-Closed Registration API

**Files:**
- Create: `src/icor/api/registrations.py`
- Modify: `src/icor/api/schemas.py`
- Modify: `src/icor/api/app.py`
- Test: `tests/api/test_registration_api.py`
- Modify: `web/openapi.json`
- Modify: `web/src/lib/api/schema.ts`
- Modify: `web/src/lib/api/client.ts`
- Test: `web/tests/api-client.test.ts`

**Interfaces:**
- Consumes: `RegistrationService` configured by `ICOR_EVIDENCE_ACTIVE_ROOT` or `ICOR_EVIDENCE_CANDIDATE`.
- Produces: `GET /api/v1/registrations/summary` and `GET /api/v1/registrations/ranking?geography=EU27&year=2024&search=&page=1&page_size=25`.

- [ ] **Step 1: Write failing API tests**

Assert typed serialization, pagination/filter validation, snapshot/version/freshness metadata, model year `null`, source IDs, and `503 registration_data_unavailable` when no verified real snapshot is configured. Assert the demo planner remains independent and is never invoked by these routes.

- [ ] **Step 2: Run RED verification**

Run: `.venv\Scripts\python.exe -m pytest tests/api/test_registration_api.py -q`

Expected: FAIL because the routes and schemas do not exist.

- [ ] **Step 3: Implement schemas, router, and app composition**

Bound search to 100 characters and page size to 100. Prefer an active snapshot root when configured; allow the exact candidate path only for local review. Configuration or verification errors must be logged by exception type only and exposed as the typed 503 problem without filesystem paths or raw exceptions. Update the API description from demonstration-only to the local evidence-led product contract.

- [ ] **Step 4: Generate the TypeScript contract and implement client methods**

Run: `Push-Location web; npm run openapi:generate; Pop-Location`

Add `getRegistrationSummary()` and `getRegistrationRanking(query)` using the generated types and existing safe problem parsing.

- [ ] **Step 5: Run GREEN verification and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/api/test_registration_api.py -q`

Run: `Push-Location web; npm test -- --run tests/api-client.test.ts; npm run openapi:check; Pop-Location`

Expected: PASS and no OpenAPI drift.

```powershell
git add src/icor/api/registrations.py src/icor/api/schemas.py src/icor/api/app.py tests/api/test_registration_api.py web/openapi.json web/src/lib/api/schema.ts web/src/lib/api/client.ts web/tests/api-client.test.ts
git commit -m "feat: expose official registration API"
```

---

### Task 5: Make Real Registrations the Web App Landing Experience

**Files:**
- Create: `web/src/features/registrations/RegistrationsPage.tsx`
- Create: `web/src/lib/registration-search.ts`
- Modify: `web/src/app/router.tsx`
- Modify: `web/src/app/AppShell.tsx`
- Modify: `web/src/app/styles.css`
- Test: `web/tests/registrations-page.test.tsx`
- Test: `web/tests/registration-search.test.ts`
- Test: `web/tests/router.test.tsx`
- Test: `web/tests/app-shell.test.tsx`

**Interfaces:**
- Consumes: typed registration summary/ranking client methods.
- Produces: default `/registrations` route with URL-backed `search`, `page`, and geography/year state; `/` redirects there; prototype planner/opportunity links are grouped and labelled `Prototype`.

- [ ] **Step 1: Write failing component and URL-state tests**

Assert the real-data page renders `Official 2024 registrations`, source attribution, snapshot/freshness, EU27 total, ranked make/model rows, confidence text, `Model year unavailable`, pagination, loading/empty/unavailable/retry states, and no demonstration forecast language. Assert `%`, `_`, Unicode, back/forward, and deep-link behavior.

- [ ] **Step 2: Run RED verification**

Run: `Push-Location web; npm test -- --run tests/registrations-page.test.tsx tests/registration-search.test.ts tests/router.test.tsx tests/app-shell.test.tsx; Pop-Location`

Expected: FAIL because the route and components do not exist.

- [ ] **Step 3: Implement the landing page and navigation hierarchy**

Use semantic table markup with a compact mobile card reflow, visible rank/make/model/registrations/confidence, text-first warnings, keyboard-reachable pagination and evidence links, and an explanatory notice: registration year is not model year; windshield fitment and replacement forecasts are not yet inferred from this dataset. Preserve `/evidence` as the provenance workspace. Keep prototype routes functional but visually secondary.

- [ ] **Step 4: Run GREEN verification and commit**

Run: `Push-Location web; npm test -- --run; npm run typecheck; npm run lint; npm run build; Pop-Location`

Expected: all frontend gates pass.

```powershell
git add web/src/features/registrations/RegistrationsPage.tsx web/src/lib/registration-search.ts web/src/app/router.tsx web/src/app/AppShell.tsx web/src/app/styles.css web/tests/registrations-page.test.tsx web/tests/registration-search.test.ts web/tests/router.test.tsx web/tests/app-shell.test.tsx
git commit -m "feat: make official registrations the landing page"
```

---

### Task 6: Build, Validate, and Promote the First Canonical Real Snapshot

**Files:**
- Modify: `scripts/build_evidence_snapshot.py`
- Modify: `tests/integration/test_clean_room_evidence_snapshot.py`
- Runtime only: ignored `.local/evidence/**`

**Interfaces:**
- Consumes: four already staged checksum-pinned official releases and `OFFICIAL_SOURCE_VERSIONS`.
- Produces: a deterministic mapped candidate, successful validation report, and local active pointer; promotion never deletes the last-known-good snapshot.

- [ ] **Step 1: Extend the clean-room integration test before live build**

Build the fictional two-row release through the production identity composition and assert one canonical model family, two normalized mappings, no invented model year, identical snapshot/database hashes across two clean roots, successful promotion, and real registration query result 15.

- [ ] **Step 2: Run RED then GREEN integration verification**

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_clean_room_evidence_snapshot.py -q`

Expected RED before production composition; PASS after implementation.

- [ ] **Step 3: Build the official candidate from pinned releases**

Use the existing explicit local root, the four exact staged release IDs, deterministic seed `20260827`, and build-as-of `2026-08-27T12:00:00Z`. Capture only sanitized IDs/counts/checksums; never print raw rows or credentials.

- [ ] **Step 4: Independently verify candidate invariants**

Run CLI `verify`, repository validation, database SHA-256 recomputation, release membership/count reconciliation, mapped/unresolved/rejected counts, canonical-model count, selected-mapping count, EU27 registration total, top rows, and a second deterministic build comparison. Promotion is forbidden if any invariant fails.

- [ ] **Step 5: Promote locally and verify last-known-good serving**

Run CLI `promote` for the validated candidate, then `status` and `verify`. Assert the active pointer names the exact candidate and the read-only registration service returns real EEA results. This is local derived data only; do not push, merge, deploy, or alter the protected checkout.

- [ ] **Step 6: Commit integration coverage**

```powershell
git add scripts/build_evidence_snapshot.py tests/integration/test_clean_room_evidence_snapshot.py
git commit -m "test: verify canonical real snapshot promotion"
```

---

### Task 7: Run Browser Review, Full Gates, and Record the Checkpoint

**Files:**
- Create: `web/e2e/registrations-real-data.spec.ts`
- Modify: `README.md`
- Modify: `docs/DEVELOPMENT.md`
- Modify: `docs/CODEX_HANDOFF.md`

**Interfaces:**
- Consumes: promoted real snapshot and completed API/UI.
- Produces: real-data browser journey, operator instructions, exact verification evidence, and a local review URL.

- [ ] **Step 1: Write and run the real-data browser journey**

Cover default redirect, EU27 summary/ranking, search/deep link/back-forward, source evidence navigation, explicit unavailable model year, prototype separation, keyboard reachability, 390px and 1440px reflow, no serious/critical axe findings, and no horizontal page overflow.

Run: `Push-Location web; npx playwright test e2e/registrations-real-data.spec.ts --workers=1; Pop-Location`

Expected: PASS against the live mapped snapshot.

- [ ] **Step 2: Run final Python and frontend gates**

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests scripts/audit_baseline.py scripts/build_evidence_snapshot.py
Push-Location web
npm test -- --run
npm run openapi:check
npm run typecheck
npm run lint
npm run build
Pop-Location
uv lock --check
uv run pip-audit
git diff --check
```

Expected: all non-XFAIL tests pass; only documented Windows symlink skips and strict legacy XFAILs remain; no OpenAPI drift, lint/type/build error, known dependency vulnerability, or whitespace error.

- [ ] **Step 3: Start a fresh local review instance**

Launch API and Vite on unused localhost ports with the active real snapshot configured. Verify health, summary, ranking, and page HTTP 200 responses; confirm the landing route shows official real data and the real routes cannot fall back to `demo-planner-v1`.

- [ ] **Step 4: Update documentation and durable handoff**

Record exact snapshot/release IDs, counts, checksums, mapping policy, EU27 total, verification commands/results, local ports/PIDs, limitations, unchanged protected checkout/deployment, and next milestone: cross-source alias review/reconciliation followed by historical estimation and forecasting. Do not claim windshield demand validation.

- [ ] **Step 5: Commit documentation**

```powershell
git add web/e2e/registrations-real-data.spec.ts README.md docs/DEVELOPMENT.md docs/CODEX_HANDOFF.md
git commit -m "docs: publish canonical registration checkpoint"
```
