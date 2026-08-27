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

## Local planner preview

The React/FastAPI product slice runs entirely on deterministic **demonstration data**.
It requires no production secrets and no customer data.

```powershell
uv sync --locked --all-groups
cd web
npm ci
cd ..
uv run python scripts/run_planner_dev.py
```

Open the configuration planner at `http://127.0.0.1:5173/planner`, the opportunity
ranking at `http://127.0.0.1:5173/opportunities`, and the API documentation at
`http://127.0.0.1:8000/docs`.

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
licences, and limitations. Source labels remain unresolved and candidates are never
promoted automatically, so the running planner continues to use clearly labelled demo
data until reviewed identity reconciliation and a snapshot-backed API are implemented.
