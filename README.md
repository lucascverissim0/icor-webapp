# ICOR windshield-demand platform

ICOR is an auditable, local-first planning application that turns official European
vehicle evidence into generation-level replacement-opportunity baselines for 2028
and 2031. It preserves registration-year semantics, reports P10/P50/P90 intervals,
and keeps observed, estimated, forecastable, and evidence-only records distinct.

The baseline does not claim an exact windshield, trim, ADAS, configuration, or part
fitment. Survival, replacement-hazard, registration forecast, and uncertainty
methods are explicit versioned assumptions intended for planning and later
out-of-sample comparison.

Development is local-first and isolated from the currently deployed application.
See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for setup, audit, test, and local-app
commands.

## Local web app

The React/FastAPI app serves one verified active snapshot across registrations,
source evidence, generation planning, opportunity ranking, completeness reporting,
and protected cutoff-safe ML export. Registration year is explicitly not treated as
model year, and no runtime demonstration-data fallback is available.
Local operation uses no production secrets and no customer data.

```powershell
uv sync --locked --all-groups
cd web
npm ci
cd ..
$env:ICOR_EVIDENCE_ACTIVE_ROOT = "$PWD\.local\evidence"
$env:ICOR_EXPORT_TOKEN = '<locally-generated-32-plus-character-capability-token>'
uv run python scripts/run_planner_dev.py
```

Open registrations at `http://127.0.0.1:5173/`, planning at `/planner`, ranked
opportunities at `/opportunities`, source evidence at `/evidence`, completeness at
`/completeness`, protected export at `/exports`, and API documentation at
`http://127.0.0.1:8000/docs`.

The opportunity ranking uses the same active generation snapshot plus separate local
production-coverage state in an ignored SQLite database. Coverage state is not a
multi-user or production catalog. The existing Streamlit application remains
available separately and is not modified or deployed by this workflow.

## Official evidence snapshots

The repository acquires checksum-pinned public releases and builds immutable local
snapshots from finalized EEA annual registrations for 2010–2024, KBA FZ10 2024
registrations, and UK DfT/DVLA registrations and licensed-fleet data through
finalized 2025 Q4. See
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#official-evidence-acquisition) for exact commands,
licences, and limitations. The active local snapshot uses an exact-normalized
make/model-family registry, retains ambiguous labels as rejected, assigns every
usable year-bearing observation to a sourced or explicitly estimated generation,
and excludes sparse series from forecasts rather than inventing precision.
