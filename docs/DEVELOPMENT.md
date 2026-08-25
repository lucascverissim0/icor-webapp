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
uv run ruff check src tests scripts/audit_baseline.py
uv run pytest
uv run pip-audit
```

Tests disable network sockets and clear integration credential environment
variables. They must not overwrite `data/passenger_car_data.xlsx` or canonical
source data.

## Start the app

From the development worktree, run:

```powershell
uv run streamlit run ui/app.py
```

Open `http://localhost:8501` if Streamlit does not open it automatically. Stop the
server with `Ctrl+C` in its terminal.

## Forecast status

The existing replacement-rate and LLM-assisted forecasts are retained only as a
behavioral baseline during this foundation phase. They do not establish a
production statistical or machine-learning choice. Later model work must define
the forecast target and canonical windshield identity, then benchmark deterministic
baselines and candidate statistical/ML models with backtests, calibrated uncertainty,
and traceable model/data versions.
