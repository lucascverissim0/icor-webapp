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

Open the planner at `http://127.0.0.1:5173/planner` and the API documentation at
`http://127.0.0.1:8000/docs`. The existing Streamlit application remains available
separately and is not modified or deployed by this workflow.
