# Official Source Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest approved finalized EEA, KBA, and UK government passenger-car evidence into immutable, validated local candidate snapshots without substituting it for the demonstration planner or inventing forecasts.

**Architecture:** Add three source-specific streaming adapters behind the existing `EvidenceLoader` boundary, a conservative shared normalization layer, and an explicit application composition registry. Raw official files remain ignored immutable runtime artifacts; legally retainable miniature contract samples and manifests exercise clean-room builds in tests. An operator acquisition command downloads only allowlisted official HTTPS resources, records checksums and metadata, stages them through `ReleaseStore`, and never promotes a snapshot automatically.

**Tech Stack:** Python 3.13, standard-library CSV/ZIP/HTTP, openpyxl for KBA XLSX, pytest, SQLite evidence ledger, existing snapshot CLI.

**Spec:** `docs/superpowers/specs/2026-08-26-real-data-evidence-forecasting-design.md`

## Global Constraints

- Work only in `C:\Users\LucasCravoVERISSIMO\icor-webapp-development`; do not deploy, push, merge, or modify the protected production checkout.
- Use 2024 final EEA evidence, KBA FZ10 December 2024 annual cumulative totals, and finalized UK/DVLA data through 2024 for the first comparable release family.
- Preserve raw labels and source row locators; normalization never proves cross-source identity.
- Publish no model-level value from ambiguous, unresolved, suppressed, unavailable, or non-car records.
- Keep EEA/German national observations dependency-aware; correlated publications do not count as independent agreement.
- Do not replace the current demo API repository in this milestone and do not calculate windshield forecasts.
- Raw downloaded artifacts and built snapshots remain ignored local runtime data; only small contract samples may be committed.

---

### Task 1: Shared Parser Contracts and Conservative Normalization

**Files:**
- Create: `src/icor/evidence/source_records.py`
- Create: `src/icor/evidence/normalization.py`
- Test: `tests/evidence/test_source_records.py`
- Test: `tests/evidence/test_normalization.py`

**Interfaces:**
- Produces: `ParsedRelease(records, raw_count, accepted_count, rejected_count, quarantined_count, warnings)` and `normalize_vehicle_label(value: str) -> str | None`.
- Produces: `stable_evidence_id(prefix: str, *parts: str) -> str` using canonical UTF-8 SHA-256 input and an identifier-safe truncated digest.

- [ ] **Step 1: Write failing contract and normalization tests**

```python
def test_normalization_preserves_meaning_without_guessing_aliases():
    assert normalize_vehicle_label("  ŠKODA  OCTAVIA  ") == "škoda octavia"
    assert normalize_vehicle_label("[x]") is None
    assert normalize_vehicle_label("VW") == "vw"

def test_stable_evidence_id_is_order_sensitive_and_repeatable():
    assert stable_evidence_id("obs", "DE", "VW", "GOLF") == stable_evidence_id("obs", "DE", "VW", "GOLF")
    assert stable_evidence_id("obs", "DE", "VW", "GOLF") != stable_evidence_id("obs", "DE", "GOLF", "VW")
```

- [ ] **Step 2: Run `uv run pytest tests/evidence/test_source_records.py tests/evidence/test_normalization.py -v` and confirm missing-module failures.**
- [ ] **Step 3: Implement immutable parse-summary contracts, Unicode whitespace/case normalization, explicit DfT marker handling, and deterministic IDs without fuzzy matching.**
- [ ] **Step 4: Re-run the focused tests and confirm they pass.**
- [ ] **Step 5: Commit `feat: add source parsing contracts` with only Task 1 files.**

### Task 2: EEA 2024 Final Passenger-Car Adapter

**Files:**
- Create: `src/icor/evidence/sources/eea.py`
- Create: `src/icor/evidence/sources/__init__.py`
- Create: `tests/fixtures/sources/eea-2024-final-sample.csv`
- Create: `tests/evidence/sources/test_eea.py`

**Interfaces:**
- Consumes: `StoredRelease`, `SQLiteEvidenceRepository`, shared normalization and stable IDs.
- Produces: `EEAPassengerCarLoader.load(releases: tuple[StoredRelease, ...], repository: SQLiteEvidenceRepository) -> None` registered as `eea_co2_cars_csv_v1`.

- [ ] **Step 1: Write a failing parser contract using actual 2024-final column names and representative quoted/non-ASCII labels.**

```python
def test_eea_groups_vehicle_rows_without_inventing_model_year(eea_release, repository):
    EEAPassengerCarLoader().load((eea_release,), repository)
    rows = repository.list_observations()
    assert {(r.geography, r.original_make, r.original_model, r.value) for r in rows} == {
        ("DE", "VOLKSWAGEN", "GOLF", Decimal("2")),
        ("FR", "RENAULT", "CLIO", Decimal("1")),
    }
    assert all(r.mapping_status is MappingStatus.UNRESOLVED for r in rows)
    assert all(r.normalized_model_year is None for r in rows)
```

- [ ] **Step 2: Run `uv run pytest tests/evidence/sources/test_eea.py -v` and confirm the loader is missing.**
- [ ] **Step 3: Implement streaming semicolon/comma dialect detection, exact schema validation, passenger-car country/make/commercial-name grouping, unresolved mappings, source-row provenance ranges, and aggregate checks.**
- [ ] **Step 4: Add rejection tests for wrong status/year/schema, missing country/make/model, negative counts, and mixed provisional rows; run the focused suite.**
- [ ] **Step 5: Commit `feat: parse final EEA passenger car evidence`.**

### Task 3: KBA FZ10 2024 Adapter

**Files:**
- Create: `src/icor/evidence/sources/kba.py`
- Create: `tests/fixtures/sources/kba-fz10-2024-12-sample.xlsx`
- Create: `tests/evidence/sources/test_kba.py`

**Interfaces:**
- Produces: `KBAFZ10Loader.load(...) -> None`, parser name `kba_fz10_xlsx_v1`.
- Uses only the December annual cumulative total for passenger-car brand/model-series rows; fuel subtotals remain source metadata, not independent registrations.

- [ ] **Step 1: Write a failing workbook-layout contract that proves title/header discovery, German number parsing, total-row exclusion, and one observation per brand/model series.**
- [ ] **Step 2: Run `uv run pytest tests/evidence/sources/test_kba.py -v` and confirm failure because the loader is absent.**
- [ ] **Step 3: Implement read-only XLSX parsing with bounded worksheet/header search, formula rejection, merged-cell-safe labels, final annual-period semantics, and unresolved normalized mappings.**
- [ ] **Step 4: Add tests for workbook drift, duplicate model rows, suppressed cells, malformed totals, and non-December releases; run the focused suite.**
- [ ] **Step 5: Commit `feat: parse KBA FZ10 registration evidence`.**

### Task 4: UK DfT/DVLA Registration and Fleet Adapters

**Files:**
- Create: `src/icor/evidence/sources/uk_dft.py`
- Create: `tests/fixtures/sources/df_VEH0160_GB-sample.csv`
- Create: `tests/fixtures/sources/df_VEH0120_GB-sample.csv`
- Create: `tests/evidence/sources/test_uk_dft.py`

**Interfaces:**
- Produces: `UKFirstRegistrationLoader` (`uk_dft_veh0160_csv_v1`) and `UKActiveFleetLoader` (`uk_dft_veh0120_csv_v1`).
- VEH0160 outputs quarterly `NEW_REGISTRATIONS`; VEH0120 outputs quarter-end `ACTIVE_FLEET` and keeps Licensed separate from SORN.

- [ ] **Step 1: Write failing wide-CSV tests covering Cars-only filtering, quarterly column parsing, `[c]` quarantine, `[x]`/`[z]` rejection, zero preservation, and GB geography.**
- [ ] **Step 2: Run `uv run pytest tests/evidence/sources/test_uk_dft.py -v` and confirm missing-loader failures.**
- [ ] **Step 3: Implement streaming wide-to-long transforms, exact DfT marker semantics, generic-model preference with detailed-model provenance, and no UK/EU relabelling.**
- [ ] **Step 4: Add tests for schema drift, duplicate identity/quarter rows, non-car exclusion, SORN separation, and annual 2024 cut-off; run the focused suite.**
- [ ] **Step 5: Commit `feat: parse UK registration and fleet evidence`.**

### Task 5: Production Loader Registry and Multi-Source Snapshot Build

**Files:**
- Create: `src/icor/evidence/source_registry.py`
- Modify: `scripts/build_evidence_snapshot.py`
- Modify: `src/icor/domain/snapshots.py`
- Test: `tests/evidence/test_source_registry.py`
- Test: `tests/integration/test_official_source_snapshot.py`

**Interfaces:**
- Produces: `official_loader_registry() -> Mapping[str, EvidenceLoader]` and source-specific `SnapshotVersions` values.
- CLI production composition uses only explicit official loader names; injected test registries remain supported.

- [ ] **Step 1: Write failing tests proving all four parser names are registered, unknown names fail closed, mixed-source build order is deterministic, and identical artifacts rebuild the same candidate.**
- [ ] **Step 2: Run the focused registry/integration tests and confirm RED.**
- [ ] **Step 3: Wire the official registry into CLI composition, replace foundation-only version constants with `official-sources-v1` versions, and retain explicit loader injection for tests.**
- [ ] **Step 4: Build a clean-room candidate from all contract samples; assert release/observation counts, source geographies/measures, no published model values, and no fixture fallback.**
- [ ] **Step 5: Run `uv run pytest tests/evidence tests/integration/test_clean_room_evidence_snapshot.py tests/integration/test_official_source_snapshot.py -v`.**
- [ ] **Step 6: Commit `feat: compose official evidence snapshot loaders`.**

### Task 6: Allowlisted Acquisition, Manifest Generation, and Runtime Profiling

**Files:**
- Create: `scripts/acquire_official_evidence.py`
- Create: `src/icor/evidence/acquisition.py`
- Create: `tests/evidence/test_acquisition.py`
- Create: `tests/integration/test_official_acquisition.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `acquire --source {eea-2024-final,kba-fz10-2024,uk-veh0160-gb,uk-veh0120-gb} --root PATH`.
- Downloads only pinned HTTPS hosts/paths, rejects redirects outside the allowlist, limits bytes, fsyncs a temporary artifact, computes SHA-256, profiles through the matching parser, writes a canonical manifest, then stages through `ReleaseStore`.

- [ ] **Step 1: Write failing tests for host/path allowlists, redirect rejection, byte limits, content-type/signature checks, atomic temp cleanup, checksum/manifests, and idempotent staging.**
- [ ] **Step 2: Run the acquisition tests and confirm RED.**
- [ ] **Step 3: Implement the downloader and manifest builder with CC BY 4.0 EEA attribution, official KBA/GovData terms link, and UK Open Government Licence v3.0 attribution.**
- [ ] **Step 4: Run mocked acquisition integration tests and confirm no network is required for CI.**
- [ ] **Step 5: Acquire the four approved live artifacts into `.local/evidence`, record actual checksums/sizes/counts, build one candidate with a deterministic timestamp/seed, and do not promote automatically.**
- [ ] **Step 6: Commit `feat: acquire approved official evidence releases` without raw artifacts or runtime manifests.**

### Task 7: Validation, Documentation, and Handoff

**Files:**
- Modify: `docs/DEVELOPMENT.md`
- Modify: `README.md`
- Modify: `docs/CODEX_HANDOFF.md`
- Modify: `tests/test_toolchain.py`

**Interfaces:**
- Documents exact acquisition/build/verify commands, licences, known source limitations, candidate snapshot identity/counts, and the fact that the current planner still serves demo data until snapshot-backed APIs are implemented.

- [ ] **Step 1: Add failing documentation/toolchain assertions for all official sources, terms, runtime-ignore rules, and no automatic promotion.**
- [ ] **Step 2: Update operator documentation and the durable handoff with source URLs, publication status, checksums, parser results, exclusions, candidate ID, and unresolved risks.**
- [ ] **Step 3: Run source/parser/integration tests, then `uv run pytest -q`, maintained Ruff, lock check, pip-audit, and `git diff --check`.**
- [ ] **Step 4: Inspect the live candidate database read-only and reconcile KBA and UK accepted totals against official published totals/tables; document any explainable difference or quarantine the release.**
- [ ] **Step 5: Confirm no raw artifacts are tracked, no active pointer changed, the local planner process state, and the protected checkout/remote were untouched.**
- [ ] **Step 6: Commit `docs: document official evidence ingestion` only if unrelated user edits can remain excluded safely.**

## Self-Review

- Spec coverage: this plan implements source acquisition, manifests, parsers, normalization, source-level validation, deterministic multi-source candidate builds, and operator documentation for the three approved source families. Reconciliation, canonical curation, estimates, forecasts, API replacement, and UI replacement remain later approved milestones.
- Placeholder scan: no TBD/TODO or unspecified error-handling steps remain.
- Type consistency: every loader implements the existing `EvidenceLoader.load(tuple[StoredRelease, ...], SQLiteEvidenceRepository) -> None`; all production parser names are unique and registered through one mapping.
