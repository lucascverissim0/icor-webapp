# ICOR windshield-demand platform

ICOR is evolving into an auditable planning application that forecasts windshield
replacement demand by market, vehicle brand, model, model year, and exact
windshield-compatible configuration or part family.

The current forecasting workflows are prototypes. Their constant-rate and
LLM-assisted outputs are not yet validated production forecasts. Statistical and
machine-learning approaches will be selected later through deterministic baselines,
backtesting, uncertainty reporting, and out-of-sample comparison.

Development is local-first and isolated from the currently deployed application.
See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for setup, audit, test, and local-app
commands.

## Local web app

The default React/FastAPI landing page serves a verified **official-data** product
slice: finalized 2024 EU-27 passenger-car registrations ranked by exact-normalized
make and model family. Registration year is explicitly not treated as model year.
The separate planner and opportunity workflows remain clearly labelled prototypes
using deterministic demonstration forecasts.
Local operation uses no production secrets and no customer data.

```powershell
uv sync --locked --all-groups
cd web
npm ci
cd ..
$env:ICOR_EVIDENCE_ACTIVE_ROOT = "$PWD\.local\evidence"
$env:ICOR_EVIDENCE_CANDIDATE = "$PWD\.local\evidence\candidates\snapshot-2f13ba3f0cd083c7eea8"
uv run python scripts/run_planner_dev.py
```

Open official registrations at `http://127.0.0.1:5173/`, source evidence at
`http://127.0.0.1:5173/evidence`, and API documentation at
`http://127.0.0.1:8000/docs`. Prototype routes remain at `/planner` and
`/opportunities`.

The opportunity ranking uses the same synthetic demonstration forecast plus shared
local production-coverage state in an ignored SQLite database. It supports exact
configuration/SKU coverage and an explicitly lower-precision vehicle-year fallback;
it is not a multi-user or production data store. The existing Streamlit application
remains available separately and is not modified or deployed by this workflow.

## Official evidence snapshots

The repository can now acquire checksum-pinned public releases and build immutable local
candidate snapshots from EEA 2024-final registrations, KBA FZ10 2024 registrations, and
UK DfT/DVLA registrations and licensed fleet data through finalized 2025 Q4. See
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#official-evidence-acquisition) for exact commands,
licences, and limitations. The current active local snapshot uses an exact-normalized
make/model-family registry, retains ambiguous labels as rejected, and exposes the
verified EU-27 aggregation at `/registrations`. It does not infer model year,
windshield fitment, replacement demand, or forecasts. `/evidence` remains the
read-only provenance workspace, while prototype forecasts continue to use clearly
labelled demonstration data.
