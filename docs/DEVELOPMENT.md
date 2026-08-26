# Local development

## Repository isolation

All development work belongs in
`C:\Users\LucasCravoVERISSIMO\icor-webapp-development` on the
`development/windshield-demand-platform` branch. Treat the `icor-webapp` checkout
on `main` as read-only. Do not merge, push, deploy, or change production data or
secrets without explicit authorization.

## Runtime setup

The supported runtime is Python 3.12 and the dependency source of truth is
`uv.lock`. Install uv, enter the development worktree, and synchronize the exact
runtime and development dependencies:

```powershell
cd C:\Users\LucasCravoVERISSIMO\icor-webapp-development
uv python install 3.12
uv sync --locked --all-groups
```

Do not use the machine's global Python environment for this project.

Install the locked web dependencies with a supported Node release (24.15.0 is the
current development runtime):

```powershell
cd web
npm ci
cd ..
```

## Local secrets

Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` only when a
local integration is needed. The destination is ignored by Git. Keep its values
blank for offline review, and never copy production credentials or customer data
into the development worktree. Do not commit or paste any secret.

The dashboard can start without integration keys, although OpenAI and search-backed
operations will remain unavailable. The example user is intentionally disabled by
an empty password.

## Verification

Run the deterministic, read-only baseline audit:

```powershell
uv run python scripts/audit_baseline.py
```

Run static checks and the complete test suite:

```powershell
uv lock --check
uv run ruff check src tests scripts/audit_baseline.py scripts/build_evidence_snapshot.py
uv run pytest
uv run pip-audit
```

Tests disable network sockets and clear integration credential environment
variables. They must not overwrite `data/passenger_car_data.xlsx` or canonical
source data.

## Evidence snapshot foundation

The evidence CLI stores all runtime state below an explicit, ignored local root. Before
acquiring any source, review and record the source's terms, permitted local use,
redistribution restrictions, attribution, retention/deletion requirements, and any
personal-data restrictions. Do not download or stage a source until that review supports
the intended use. Never put credentials, customer data, unapproved copyrighted extracts,
or real source files in Git.

The following commands show the exact local command boundary. The fixture paths are
fictional contract-test data only; use a reviewed real manifest and artifact only in a
later approved source-acquisition plan.

```powershell
$evidenceRoot = '.local/evidence'
$manifest = 'tests/fixtures/sources/sample-registration.manifest.json'
$artifact = 'tests/fixtures/sources/sample-registration.csv'
$candidateSnapshotId = 'candidate-snapshot-id-from-a-successful-build'

uv run python -c "from pathlib import Path; from icor.evidence.release_manifests import load_release_manifest; print(load_release_manifest(Path('tests/fixtures/sources/sample-registration.manifest.json')).release_id)"
uv run python scripts/build_evidence_snapshot.py stage-release --root $evidenceRoot --manifest $manifest --artifact $artifact
uv run python scripts/build_evidence_snapshot.py build --root $evidenceRoot --release sample-registration-2024 --build-as-of 2026-08-27T00:00:00+00:00 --deterministic-seed 0
uv run python scripts/build_evidence_snapshot.py promote --root $evidenceRoot --snapshot $candidateSnapshotId
uv run python scripts/build_evidence_snapshot.py status --root $evidenceRoot
uv run python scripts/build_evidence_snapshot.py verify --root $evidenceRoot
```

The manifest command is the pre-stage validation boundary. `stage-release` verifies
the declared byte count and SHA-256 before immutably storing the artifact and manifest.
`build` verifies the staged release and candidate ledger, then emits a candidate only
when validation permits promotion; `promote` revalidates it before atomically changing
the active pointer; `status` reports the active pointer state; and `verify` opens the
active SQLite repository read-only and validates the pointed snapshot.

At this checkpoint the public CLI deliberately has no registered parser. The example
`build` therefore returns the typed `unsupported_parser` result (exit 2), and `promote`
has no candidate ID to use. If it is invoked with `$candidateSnapshotId` anyway,
promotion rejects the missing candidate (exit 3). `status` and `verify` then each emit
`{"active_snapshot_id": null, "state": "unavailable"}` and exit 4 because no active
snapshot exists. That is intentional: there is no real EEA, KBA, or UK parser, no
forecast, no API replacement, and no fixture fallback. A future approved source plan
must add a reviewed parser through application composition before a candidate can be
built; it must not substitute the test fixture.

All source-release directories and promoted snapshot directories are immutable. A failed
build or promotion leaves the current active snapshot unchanged, and promotion never
deletes an earlier snapshot. If `active.json` is missing or corrupt, do not hand-edit it:
re-promote a retained known-good candidate by its snapshot ID, or rebuild from verified
staged releases when no valid candidate remains. An unavailable or invalid active state
must remain unavailable; the application must not fall back to fixtures.

Local evidence state is ignored by Git but can still be consequential. Deleting the
exact evidence root removes locally staged source copies, candidates, snapshots, and any
active-pointer recovery path; it is not reversible through Git. Stop local processes,
preserve any required retention/terms evidence, and back up the exact root if recovery is
needed before deleting it. Never recursively delete `.local/` or a path resolved from an
unverified environment variable.

Verify the planner contract and frontend:

```powershell
cd web
npm run openapi:check
npm test -- --run
npm run typecheck
npm run lint
npm run build
npm run e2e
cd ..
```

## Start the app

From the development worktree, run:

```powershell
uv run streamlit run ui/app.py
```

Open `http://localhost:8501` if Streamlit does not open it automatically. Stop the
server with `Ctrl+C` in its terminal.

## Start the planner web app

The planner uses deterministic fixture records from `data/demo/planner-v1.json`.
They are clearly labelled as **demonstration data** and are not production forecasts.
The local workflow requires no production secrets and no customer data.

From the development worktree, validate or start both local-only processes:

```powershell
uv run python scripts/run_planner_dev.py --check
uv run python scripts/run_planner_dev.py
```

Open `http://127.0.0.1:5173/planner` for configuration planning,
`http://127.0.0.1:5173/opportunities` for opportunity ranking and local production
coverage, and `http://127.0.0.1:8000/docs` for the FastAPI contract. Stop both
processes with `Ctrl+C`. If separate terminals are preferable, run the API from the
repository root:

```powershell
uv run uvicorn icor.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Then run the web client from `web`:

```powershell
npm run dev -- --host 127.0.0.1 --port 5173
```

### Local production-coverage state

By default the opportunity page stores coverage records at the ignored path
`.local/production-coverage.sqlite3`. Set `ICOR_COVERAGE_DB` before starting the API
to use a different local database:

```powershell
$env:ICOR_COVERAGE_DB = 'C:\path\to\local\production-coverage.sqlite3'
uv run python scripts/run_planner_dev.py
```

The database is shared by every browser using that local API and currently has no
authentication, authorization, user attribution, backup, or audit-grade history.
Store no secrets, customer data, or proprietary catalog exports in notes. The app
refuses unsupported or corrupt schema versions instead of deleting or rebuilding the
file automatically.

If a disposable local database must be reset, stop the API first and move the exact
configured SQLite file to a backup location. Do not recursively delete `.local/` or
remove an unresolved environment-variable path.

Forecast and fitment values remain synthetic demonstration evidence. Connecting the
company catalog and tracked replacement history requires a separately approved,
validated ingestion contract; SQLite coverage records alone do not make the forecast
production-valid.

The planner and the legacy Streamlit app can coexist on their separate ports. No
production deployment, proprietary fitment integration, or customer-data migration
is part of this local product slice.

## Forecast status

The existing replacement-rate and LLM-assisted forecasts are retained only as a
behavioral baseline during this foundation phase. They do not establish a
production statistical or machine-learning choice. Later model work must define
the forecast target and canonical windshield identity, then benchmark deterministic
baselines and candidate statistical/ML models with backtests, calibrated uncertainty,
and traceable model/data versions.
