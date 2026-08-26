# ICOR Evidence and Snapshot Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the immutable evidence ledger and atomic, last-known-good snapshot lifecycle that all real-source ingestion, reconciliation, forecasting, API, and UI work will use.

**Architecture:** Extend the modular monolith with immutable evidence-domain contracts, a versioned SQLite evidence store, a filesystem release store, and an application-owned snapshot publication service. Candidate builds live in isolated directories and become queryable only through an atomic active-snapshot pointer after manifest, checksum, and invariant validation succeeds.

**Tech Stack:** Python 3.12, frozen dataclasses, enums, protocols, SQLite, JSON manifests, SHA-256, pathlib, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-26-real-data-evidence-forecasting-design.md`

## Global Constraints

- Work only in `C:\Users\LucasCravoVERISSIMO\icor-webapp-development` on `development/windshield-demand-platform`; preserve unrelated `AGENTS.md` and handoff changes.
- Do not push, merge, deploy, use production/customer data, or modify the protected `main` checkout or Streamlit application.
- Keep source artifacts immutable and checksummed; never silently replace a release or mutate an accepted observation.
- Different measures, units, geography definitions, or periods are never equivalent observations.
- Ambiguous and unresolved identities remain visible but cannot contribute to model-level published totals.
- Candidate builds are isolated; only a completely validated snapshot is promoted, and failure preserves the previous active snapshot.
- Never fall back to demonstration fixtures when a real-data snapshot is unavailable.
- Identical artifacts, configuration, code versions, and deterministic seeds must produce the same snapshot identifier and published database bytes.
- Store no credentials, secrets, private/customer records, or unapproved copyrighted source extracts in Git.
- Maintain Python `>=3.12,<3.13`, bound SQLite parameters, UTC timestamps, Ruff, strict pytest behavior, and all four existing strict legacy XFAILs.

---

### Task 1: Define immutable evidence and snapshot domain contracts

**Files:**
- Create: `src/icor/domain/evidence.py`
- Create: `src/icor/domain/snapshots.py`
- Create: `tests/domain/test_evidence.py`
- Create: `tests/domain/test_snapshots.py`

**Interfaces:**
- Consumes: Python standard-library `date`, `datetime`, `Decimal`, `Enum`, and frozen dataclasses.
- Produces: `Measure`, `PublicationStatus`, `PeriodPrecision`, `MappingStatus`, `ValueStatus`, `ReleaseManifest`, `Observation`, `CanonicalVehicle`, `IdentityMapping`, `EvidenceConfidence`, `PublishedValue`, `SnapshotVersions`, `SnapshotManifest`, and `SnapshotStatus`.

- [ ] **Step 1: Write failing release and observation invariant tests**

```python
def test_release_manifest_requires_utc_retrieval_and_sha256():
    with pytest.raises(ValueError, match="UTC"):
        make_release(retrieved_at=datetime(2026, 8, 26))
    with pytest.raises(ValueError, match="SHA-256"):
        make_release(sha256="abc")


def test_observation_rejects_negative_counts_and_missing_source_label():
    with pytest.raises(ValueError, match="non-negative"):
        make_observation(value=Decimal("-1"))
    with pytest.raises(ValueError, match="original"):
        make_observation(original_make="")
```

Cover invalid identifiers, blank publishers/URLs/terms, reversed coverage dates, non-integer count values, missing units, naive timestamps, and unsupported publication statuses.

- [ ] **Step 2: Write failing identity, confidence, and published-value tests**

```python
def test_unresolved_identity_cannot_publish_model_value():
    with pytest.raises(ValueError, match="unresolved"):
        make_published_value(mapping_status=MappingStatus.UNRESOLVED)


@pytest.mark.parametrize(
    ("score", "band"),
    [(0, "very_low"), (39, "very_low"), (40, "low"), (59, "low"),
     (60, "medium"), (79, "medium"), (80, "high"), (100, "high")],
)
def test_confidence_band_boundaries(score, band):
    assert make_confidence(total=score).band.value == band
```

Also assert exactly five named confidence components, component sum equality, ordered `p10 <= p50 <= p90`, required input IDs, and forecast-confidence presence only for forecast values.

- [ ] **Step 3: Run RED verification**

Run: `uv run pytest tests/domain/test_evidence.py tests/domain/test_snapshots.py -v`

Expected: FAIL because the evidence and snapshot modules do not exist.

- [ ] **Step 4: Implement explicit enums and frozen contracts**

```python
class Measure(StrEnum):
    NEW_REGISTRATIONS = "new_registrations"
    ACTIVE_FLEET = "active_fleet"


class MappingStatus(StrEnum):
    EXACT_IDENTIFIER = "exact_identifier"
    CURATED_ALIAS = "curated_alias"
    NORMALIZED_LABEL = "normalized_label"
    REVIEWED_PROBABLE = "reviewed_probable"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class EvidenceConfidence:
    authority: int
    publication_status: int
    coverage: int
    identity: int
    independent_agreement: int
    reasons: tuple[str, ...]
    applied_cap: int | None = None

    @property
    def total(self) -> int:
        raw = self.authority + self.publication_status + self.coverage + self.identity + self.independent_agreement
        return min(raw, self.applied_cap) if self.applied_cap is not None else raw
```

Use fixed maxima 25/10/25/20/20, derive bands at 40/60/80, retain original labels beside normalized labels, and reject model-level publication for `AMBIGUOUS`, `REJECTED`, or `UNRESOLVED` mappings.

- [ ] **Step 5: Implement snapshot identity and version contracts**

```python
@dataclass(frozen=True, slots=True)
class SnapshotVersions:
    source_registry: str
    identity_registry: str
    reconciliation_method: str
    confidence_method: str
    estimation_method: str
    survival_method: str
    hazard_method: str
    forecast_method: str


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    snapshot_id: str
    status: SnapshotStatus
    built_at: datetime
    deterministic_seed: int
    release_ids: tuple[str, ...]
    versions: SnapshotVersions
    database_sha256: str
    observation_count: int
    published_value_count: int
    warnings: tuple[str, ...]
```

Require sorted unique release IDs, non-negative counts and seed, UTC timestamps, lowercase 64-character hashes, and a stable snapshot ID supplied by the builder in Task 6.

- [ ] **Step 6: Run GREEN verification and commit**

Run: `uv run pytest tests/domain/test_evidence.py tests/domain/test_snapshots.py -v`

Expected: PASS.

Run: `uv run ruff check src tests`

Expected: `All checks passed!`

Commit: `feat: define evidence and snapshot contracts`

---

### Task 2: Add deterministic serialization and release-manifest validation

**Files:**
- Create: `src/icor/evidence/serialization.py`
- Create: `src/icor/evidence/release_manifests.py`
- Create: `src/icor/evidence/__init__.py`
- Create: `tests/evidence/test_serialization.py`
- Create: `tests/evidence/test_release_manifests.py`

**Interfaces:**
- Consumes: `ReleaseManifest`, `SnapshotManifest`, dataclasses, enums, UTC datetimes, decimals.
- Produces: `canonical_json_bytes(value) -> bytes`, `sha256_file(path: Path) -> str`, `load_release_manifest(path: Path) -> ReleaseManifest`, and `write_release_manifest(path: Path, manifest: ReleaseManifest) -> None`.

- [ ] **Step 1: Write failing deterministic serialization tests**

```python
def test_canonical_json_is_stable_across_mapping_order():
    left = canonical_json_bytes({"b": 2, "a": 1})
    right = canonical_json_bytes({"a": 1, "b": 2})
    assert left == right == b'{"a":1,"b":2}\n'


def test_sha256_file_matches_known_digest(tmp_path):
    artifact = tmp_path / "release.csv"
    artifact.write_bytes(b"make,model,count\nA,B,1\n")
    assert sha256_file(artifact) == "76581d541a61bac3f24c3b020b3a871245b926d017537253567ff9c3c766925a"
```

Calculate and lock the expected digest from the exact bytes during the RED test if the shown digest differs; do not weaken the equality assertion.

- [ ] **Step 2: Write failing manifest round-trip and rejection tests**

```python
def test_release_manifest_round_trips_without_information_loss(tmp_path, release_manifest):
    path = tmp_path / "manifest.json"
    write_release_manifest(path, release_manifest)
    assert load_release_manifest(path) == release_manifest


def test_manifest_rejects_unknown_fields(tmp_path):
    path = write_json(tmp_path, valid_manifest_dict() | {"credential": "forbidden"})
    with pytest.raises(ManifestError, match="unknown"):
        load_release_manifest(path)
```

Cover malformed JSON, missing required fields, invalid enums/dates/hashes, absolute artifact paths, `..` traversal, duplicate release IDs in snapshot manifests, and non-UTF-8 manifests.

- [ ] **Step 3: Run RED verification**

Run: `uv run pytest tests/evidence/test_serialization.py tests/evidence/test_release_manifests.py -v`

Expected: FAIL because serialization and manifest loaders do not exist.

- [ ] **Step 4: Implement canonical JSON and strict schema conversion**

Use UTF-8, `sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`, and one trailing newline. Convert dataclasses, enums, dates, datetimes, decimals, tuples, and paths explicitly; reject floats so platform-dependent numeric formatting cannot enter snapshot identity.

`load_release_manifest` must compare the exact key set with the contract, parse every enum and ISO value, then construct the domain object so domain validation remains authoritative. `write_release_manifest` must use a same-directory temporary file, flush and `os.fsync`, then `os.replace`.

- [ ] **Step 5: Run GREEN verification and commit**

Run: `uv run pytest tests/evidence/test_serialization.py tests/evidence/test_release_manifests.py -v`

Expected: PASS.

Commit: `feat: validate deterministic release manifests`

---

### Task 3: Store immutable release artifacts safely

**Files:**
- Create: `src/icor/infrastructure/release_store.py`
- Create: `tests/infrastructure/test_release_store.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: source artifact `Path`, `ReleaseManifest`, `sha256_file`, and manifest serialization.
- Produces: `ReleaseStore(root: Path)`, `StoredRelease`, `ReleaseAlreadyExistsError`, `ReleaseIntegrityError`; methods `stage`, `get`, `verify`, and `list_releases`.

- [ ] **Step 1: Write failing immutable-store tests**

```python
def test_stage_copies_artifact_and_manifest_under_release_id(tmp_path, release_manifest):
    store = ReleaseStore(tmp_path / "raw")
    stored = store.stage(source_artifact(tmp_path), release_manifest)
    assert stored.artifact_path.read_bytes() == b"make,model,count\nA,B,1\n"
    assert store.verify(release_manifest.release_id) == stored


def test_existing_release_cannot_be_replaced(store, release_manifest, tmp_path):
    store.stage(source_artifact(tmp_path, b"one"), release_manifest_for(b"one"))
    with pytest.raises(ReleaseAlreadyExistsError):
        store.stage(source_artifact(tmp_path, b"two"), release_manifest_for(b"two"))
```

Also cover source/hash mismatch, post-stage corruption, incomplete release directories, traversal IDs, same-content idempotency, and stable sorted listing.

- [ ] **Step 2: Run RED verification**

Run: `uv run pytest tests/infrastructure/test_release_store.py -v`

Expected: FAIL because `ReleaseStore` does not exist.

- [ ] **Step 3: Implement same-volume atomic staging**

Layout:

```text
.local/evidence/raw/<source_id>/<release_id>/
  artifact.<source extension>
  manifest.json
```

Validate source/release IDs against `^[a-z0-9][a-z0-9._-]{0,79}$`. Copy into `.staging/<uuid>/`, hash the copied bytes, write the manifest atomically, then rename the complete directory. If the destination exists, verify and return it only when bytes and manifest are identical; otherwise raise without modifying either copy.

- [ ] **Step 4: Ignore runtime evidence while retaining legal samples**

Add `.local/evidence/` to `.gitignore`. Contract-test samples will later live under `tests/fixtures/sources/` and remain deliberately small and legally retainable.

- [ ] **Step 5: Run GREEN verification and commit**

Run: `uv run pytest tests/infrastructure/test_release_store.py -v`

Expected: PASS.

Commit: `feat: store immutable source releases`

---

### Task 4: Add the versioned SQLite evidence ledger

**Files:**
- Create: `src/icor/application/evidence.py`
- Create: `src/icor/infrastructure/sqlite_evidence_repository.py`
- Create: `tests/infrastructure/test_sqlite_evidence_repository.py`

**Interfaces:**
- Consumes: `ReleaseManifest`, `Observation`, `CanonicalVehicle`, `IdentityMapping`, `PublishedValue`, and `SnapshotManifest`.
- Produces: `EvidenceRepository` protocol and `SQLiteEvidenceRepository(path: Path, writable: bool = False)` with `add_release`, `add_observations`, `add_vehicle`, `add_mapping`, `add_published_values`, `add_snapshot`, `get_*`, and `list_*` methods; `EvidenceSchemaError`, `DuplicateEvidenceError`, and `ImmutableEvidenceError`.

- [ ] **Step 1: Write failing schema and immutability tests**

```python
def test_empty_database_migrates_to_schema_v1(tmp_path):
    repository = SQLiteEvidenceRepository(tmp_path / "evidence.sqlite3", writable=True)
    assert repository.schema_version == 1


def test_observation_identity_is_immutable(repository, observation):
    repository.add_observations((observation,))
    with pytest.raises(DuplicateEvidenceError):
        repository.add_observations((replace(observation, value=Decimal("2")),))
    assert repository.get_observation(observation.observation_id) == observation
```

Cover future schema refusal, corrupt/missing version table, read-only write rejection, transaction rollback on one invalid duplicate, foreign-key enforcement, and deterministic ordering.

- [ ] **Step 2: Write failing semantic-boundary tests**

```python
def test_mapping_requires_existing_vehicle_and_observation(repository, mapping):
    with pytest.raises(ImmutableEvidenceError, match="reference"):
        repository.add_mapping(mapping)


def test_published_value_retains_every_input(repository, published_value):
    seed_dependencies(repository, published_value)
    repository.add_published_values((published_value,))
    assert repository.get_published_value(published_value.value_id).input_ids == (
        "observation-eea-de-2024-1",
    )
```

Cover unique source-row locators, explicit geography/version/measure/period keys, mapping status retention, and prohibition of ambiguous/unresolved inputs in model-level published values.

- [ ] **Step 3: Run RED verification**

Run: `uv run pytest tests/infrastructure/test_sqlite_evidence_repository.py -v`

Expected: FAIL because the repository does not exist.

- [ ] **Step 4: Implement explicit migration 1**

Create normalized tables `schema_version`, `source_release`, `observation`, `canonical_vehicle`, `identity_mapping`, `published_value`, `published_value_input`, and `snapshot`. Use foreign keys, `CHECK` constraints mirroring enums, UTC ISO text, decimal strings, JSON only for ordered reasons/warnings/version bundles, and unique indexes for `(release_id, original_row_locator)` and canonical vehicle identity. Do not add update/delete methods for evidence records.

Open writable connections with `PRAGMA foreign_keys=ON`, `PRAGMA journal_mode=WAL`, and `PRAGMA synchronous=FULL`. Open published snapshots in SQLite read-only URI mode. All writes use explicit transactions and bound parameters.

- [ ] **Step 5: Run GREEN verification and commit**

Run: `uv run pytest tests/infrastructure/test_sqlite_evidence_repository.py -v`

Expected: PASS.

Run: `uv run ruff check src tests`

Expected: `All checks passed!`

Commit: `feat: add immutable sqlite evidence ledger`

---

### Task 5: Implement release and snapshot validation gates

**Files:**
- Create: `src/icor/evidence/validation.py`
- Create: `tests/evidence/test_validation.py`

**Interfaces:**
- Consumes: stored releases, observations, identity mappings, published values, and snapshot manifests.
- Produces: `Severity`, `ValidationFinding`, `ValidationReport`, `ReleaseValidator.validate(stored_release)`, and `SnapshotValidator.validate(repository, manifest)`.

- [ ] **Step 1: Write failing release-gate tests**

```python
def test_checksum_failure_is_mandatory_and_blocks_release(stored_release):
    stored_release.artifact_path.write_bytes(b"corrupted")
    report = ReleaseValidator().validate(stored_release)
    assert report.can_promote is False
    assert report.findings[0].code == "release.checksum_mismatch"


def test_record_counts_must_reconcile(release_with_counts):
    report = ReleaseValidator().validate(release_with_counts(raw=10, accepted=8, rejected=1, quarantined=0))
    assert any(row.code == "release.record_count_mismatch" for row in report.findings)
```

Cover missing terms metadata, missing artifact, byte-size mismatch, count conservation, reversed coverage, and valid releases.

- [ ] **Step 2: Write failing snapshot-invariant tests**

```python
def test_snapshot_rejects_orphan_published_inputs(repository, snapshot_manifest):
    insert_orphan_input_with_foreign_keys_disabled(repository)
    report = SnapshotValidator().validate(repository, snapshot_manifest)
    assert report.can_promote is False
    assert "snapshot.orphan_input" in {row.code for row in report.findings}


def test_snapshot_rejects_unordered_intervals(repository, snapshot_manifest):
    insert_invalid_interval_for_validation(repository, p10="20", p50="10", p90="30")
    report = SnapshotValidator().validate(repository, snapshot_manifest)
    assert "snapshot.interval_order" in {row.code for row in report.findings}
```

Also cover negative values, observation/manifest count mismatch, unresolved model publication, absent releases, database hash mismatch, and a clean report.

- [ ] **Step 3: Run RED verification**

Run: `uv run pytest tests/evidence/test_validation.py -v`

Expected: FAIL because validation gates do not exist.

- [ ] **Step 4: Implement stable findings and promotion policy**

```python
@dataclass(frozen=True, slots=True)
class ValidationFinding:
    code: str
    severity: Severity
    message: str
    record_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    findings: tuple[ValidationFinding, ...]

    @property
    def can_promote(self) -> bool:
        return not any(row.severity is Severity.ERROR for row in self.findings)
```

Sort findings by `(severity, code, record_id or "")`; never include credentials, raw record content, or stack traces. Mandatory errors block promotion; warnings remain attached to the snapshot manifest.

- [ ] **Step 5: Run GREEN verification and commit**

Run: `uv run pytest tests/evidence/test_validation.py -v`

Expected: PASS.

Commit: `feat: enforce evidence snapshot quality gates`

---

### Task 6: Build deterministic candidate snapshots

**Files:**
- Create: `src/icor/application/snapshot_build.py`
- Create: `src/icor/infrastructure/snapshot_store.py`
- Create: `tests/application/test_snapshot_build.py`
- Create: `tests/infrastructure/test_snapshot_store.py`

**Interfaces:**
- Consumes: `ReleaseStore`, `SQLiteEvidenceRepository`, validators, `SnapshotVersions`, sorted release IDs, deterministic seed, and an `EvidenceLoader` protocol.
- Produces: `SnapshotBuildRequest`, `SnapshotBuildResult`, `EvidenceLoader.load(releases, repository) -> None`, `SnapshotBuilder.build(request)`, and `SnapshotStore` operations `candidate_path`, `promote`, `active_manifest`, and `open_active_repository`.

- [ ] **Step 1: Write failing deterministic-build tests**

```python
def test_identical_inputs_produce_identical_snapshot(tmp_path, build_request):
    first = make_builder(tmp_path / "first").build(build_request)
    second = make_builder(tmp_path / "second").build(build_request)
    assert first.manifest.snapshot_id == second.manifest.snapshot_id
    assert first.manifest.database_sha256 == second.manifest.database_sha256
    assert first.database_path.read_bytes() == second.database_path.read_bytes()


def test_snapshot_id_changes_when_method_version_changes(builder, build_request):
    first = builder.build(build_request)
    changed = replace(build_request, versions=replace(build_request.versions, confidence_method="confidence-v2"))
    assert builder.build(changed).manifest.snapshot_id != first.manifest.snapshot_id
```

Use a test loader that inserts two observations in deliberately reversed input order to prove canonical sorting.

- [ ] **Step 2: Write failing last-known-good promotion tests**

```python
def test_failed_candidate_leaves_active_snapshot_unchanged(snapshot_store, valid_candidate, invalid_candidate):
    snapshot_store.promote(valid_candidate)
    active_before = snapshot_store.active_manifest()
    with pytest.raises(SnapshotPromotionError):
        snapshot_store.promote(invalid_candidate)
    assert snapshot_store.active_manifest() == active_before


def test_no_active_snapshot_is_typed_unavailable(snapshot_store):
    with pytest.raises(SnapshotUnavailableError):
        snapshot_store.open_active_repository()
```

Cover interrupted pointer writes, missing candidate files, hash changes after validation, read-only active repository, and idempotent repeat promotion.

- [ ] **Step 3: Run RED verification**

Run: `uv run pytest tests/application/test_snapshot_build.py tests/infrastructure/test_snapshot_store.py -v`

Expected: FAIL because the builder and snapshot store do not exist.

- [ ] **Step 4: Implement content-derived build identity**

Compute `snapshot_id = "snapshot-" + sha256(canonical_json_bytes(identity_payload))[:20]`, where `identity_payload` contains sorted release IDs with artifact hashes, every `SnapshotVersions` field, and the deterministic seed. Insert records in stable primary-key order, checkpoint WAL into the database, run `VACUUM`, close all handles, hash the final database, and validate before returning `SnapshotBuildResult`.

Candidate layout:

```text
.local/evidence/candidates/<snapshot_id>/
  evidence.sqlite3
  snapshot.json
  validation.json
```

- [ ] **Step 5: Implement atomic active-snapshot promotion**

Promote the immutable candidate directory to `.local/evidence/snapshots/<snapshot_id>/`, then atomically replace `.local/evidence/active.json` with a tiny pointer containing `snapshot_id`, manifest hash, and promotion time. Re-open and verify the target before pointer replacement. Never delete the previous snapshot during promotion.

- [ ] **Step 6: Run GREEN verification and commit**

Run: `uv run pytest tests/application/test_snapshot_build.py tests/infrastructure/test_snapshot_store.py -v`

Expected: PASS.

Commit: `feat: build and promote atomic evidence snapshots`

---

### Task 7: Add a clean-room foundation CLI and representative sample

**Files:**
- Create: `scripts/build_evidence_snapshot.py`
- Create: `tests/fixtures/sources/sample-registration.csv`
- Create: `tests/fixtures/sources/sample-registration.manifest.json`
- Create: `tests/integration/test_clean_room_evidence_snapshot.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: release-manifest loader, `ReleaseStore`, `SnapshotBuilder`, `SnapshotStore`, and an explicit local evidence root.
- Produces: `main(argv: Sequence[str] | None = None) -> int`; CLI commands `stage-release`, `build`, `promote`, `status`, and `verify`.

- [ ] **Step 1: Write a failing end-to-end clean-room test**

```python
def test_clean_room_build_is_reproducible_and_promotable(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = run_cli_build(first_root, SAMPLE_MANIFEST)
    second = run_cli_build(second_root, SAMPLE_MANIFEST)
    assert first.snapshot_id == second.snapshot_id
    assert first.database_sha256 == second.database_sha256
    assert run_cli(["promote", "--root", str(first_root), "--snapshot", first.snapshot_id]) == 0
    assert read_status(first_root)["active_snapshot_id"] == first.snapshot_id
```

Also assert no network socket is used, no output is written outside `--root`, `status` returns typed unavailable state before promotion, validation failure exits non-zero, and CLI JSON contains IDs/counts but no raw rows.

- [ ] **Step 2: Run RED verification**

Run: `uv run pytest tests/integration/test_clean_room_evidence_snapshot.py -v`

Expected: FAIL because the CLI and representative source adapter do not exist.

- [ ] **Step 3: Add the minimal contract-test source loader**

The retained CSV contains exactly these columns and two fictional rows:

```csv
reporting_country,registration_year,make,model,new_registrations
DE,2024,Example Motors,Alpha,10
FR,2024,Example Motors,Alpha,5
```

Implement the sample loader inside the integration fixture, not production parser code. It creates immutable observations with `normalized_label` mappings and no model-level published values. This proves the platform boundary without pretending to ingest EEA data before the dedicated EEA plan.

- [ ] **Step 4: Implement safe JSON CLI output**

Require explicit `--root`; reject roots resolving outside the workspace unless the caller supplies `--allow-external-root`. Use subcommand-specific required arguments, return exit `0` on success, `2` on invalid input, `3` on failed validation, and `4` when no active snapshot exists. Print one canonical JSON object to stdout and concise sanitized errors to stderr.

- [ ] **Step 5: Run GREEN verification and commit**

Run: `uv run pytest tests/integration/test_clean_room_evidence_snapshot.py -v`

Expected: PASS.

Commit: `feat: add clean-room evidence snapshot cli`

---

### Task 8: Document and verify the foundation checkpoint

**Files:**
- Modify: `README.md`
- Modify: `docs/DEVELOPMENT.md`
- Modify: `docs/CODEX_HANDOFF.md`
- Test: `tests/test_repository_security.py`

**Interfaces:**
- Consumes: the completed evidence foundation and existing repository quality gates.
- Produces: local evidence commands, storage/recovery documentation, data-handling warnings, and a recorded verification checkpoint for the next EEA-ingestion plan.

- [ ] **Step 1: Extend repository-security assertions before documentation**

```python
def test_runtime_evidence_and_candidate_snapshots_are_ignored(repository_root):
    ignored = git_check_ignore(repository_root, ".local/evidence/candidates/example/evidence.sqlite3")
    assert ignored


def test_representative_source_fixture_contains_no_real_people_or_credentials(repository_root):
    sample = (repository_root / "tests/fixtures/sources/sample-registration.csv").read_text("utf-8")
    assert "password" not in sample.casefold()
    assert "token" not in sample.casefold()
```

- [ ] **Step 2: Run the focused security test**

Run: `uv run pytest tests/test_repository_security.py -v`

Expected: PASS after the ignore rule from Task 3 and the fictional sample from Task 7 are present.

- [ ] **Step 3: Document the exact local workflow and failure behavior**

Add commands for manifest validation, staging, candidate build, validation, promotion, status, and verification. State that the foundation contains no real EEA/KBA/UK parser, no forecast, no API replacement, and no fixture fallback. Explain immutable source storage, active-pointer recovery, local-data deletion implications, and the requirement to review source terms before acquisition.

- [ ] **Step 4: Run the full foundation verification**

Run:

```powershell
uv lock --check
uv run ruff check src tests scripts
uv run pytest -p no:cacheprovider
uv run pip-audit
git diff --check
git status --short
```

Expected: lock and Ruff pass; all non-XFAIL tests pass; only the four documented strict legacy XFAILs remain; pip-audit reports no known vulnerabilities; whitespace check passes; status contains only intended plan/foundation/handoff changes plus the pre-existing unrelated `AGENTS.md` change.

- [ ] **Step 5: Record the checkpoint and commit documentation**

Update `docs/CODEX_HANDOFF.md` with exact commits, commands, pass counts, remaining risks, whether any process is running, and the next plan: EEA release acquisition/profile/parser/source-level snapshot. Do not claim the full real-data application is complete.

Commit: `docs: document evidence snapshot foundation`

