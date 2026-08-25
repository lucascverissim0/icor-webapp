# ICOR Development Foundation Design

Date: 2026-08-25
Status: approved by Lucas on 2026-08-25

## Purpose

Create a secure and reproducible local development foundation for the ICOR windshield-demand application. This is the first subproject in the approved delivery sequence. It makes future changes testable and auditable without claiming that the existing forecasting equations are correct.

The company's proprietary fitment and replacement data is explicitly out of scope for this subproject. It has only one year of reliable history and will be integrated later as fitment truth and as limited calibration/validation evidence, not treated as a sufficient long-run training set.

## Decisions

### Runtime and dependency management

- Python 3.12 is the sole supported runtime for local development, CI, the devcontainer, and deployment configuration.
- `pyproject.toml` is the human-edited dependency and tool configuration.
- `uv.lock` is the authoritative cross-platform dependency lock.
- `requirements.txt` remains as an exact production export for Streamlit Community Cloud compatibility; it is generated from the lock rather than edited independently.
- `uv sync --locked` creates the local `.venv`. Tests and application commands run through `uv run`.
- Runtime libraries receive exact resolved versions in the lock. Direct dependency constraints in `pyproject.toml` must have lower and upper bounds so upgrades are deliberate.

### Isolation and secrets

- Implementation occurs in an isolated Git worktree after Lucas consents to creating it.
- No production customer data, production output, or production secrets may be used in local development or automated tests.
- `.streamlit/secrets.toml`, `.env`, generated outputs, caches, and logs remain ignored.
- A committed secrets example may contain key names and obviously non-secret placeholders only.
- Tests must clear OpenAI, SerpAPI, PostHog, and authentication secrets and must fail if an unexpected external HTTP request is attempted.
- No task in this subproject rotates deployment secrets, rewrites Git history, deploys the app, or changes production state.

### Application security baseline

- Streamlit CORS and XSRF protections remain enabled. The devcontainer must not override either protection.
- The devcontainer uses a pinned Python 3.12 image family and installs dependencies from the lock without upgrading the operating system during ordinary setup.
- Backend logs and analytics are not required for foundation verification.
- The historically exposed OpenAI key remains an owner action: revoke and rotate first, then plan history cleanup separately.

## Architecture

The foundation adds four narrowly scoped layers while preserving current application behavior:

1. **Toolchain configuration** defines Python, dependencies, linting, and test behavior in `pyproject.toml`, `.python-version`, `uv.lock`, and the production requirements export.
2. **Safe runtime configuration** introduces a small import-safe module under `src/icor/` that reads non-secret defaults and environment settings without importing Streamlit or executing network/filesystem pipelines.
3. **Characterization and safety tests** record current deterministic behavior and known defects before fixes. Tests operate on temporary copies or pure functions and never overwrite `data/passenger_car_data.xlsx`.
4. **Continuous integration** runs locked dependency verification, linting, tests, and dependency auditing on Linux and Windows with Python 3.12.

This subproject does not create the final forecasting architecture. It creates the reliable execution boundary needed to build that architecture in later subprojects.

## Characterization coverage

The initial suite must cover behaviors that can be tested deterministically without external services:

- all 22 market files parse as UTF-8 JSON;
- record counts and missing-unit counts are reported, including the known incomplete years;
- duplicate business keys are detected rather than silently ignored;
- the current ICOR source format and Script 1 parser mismatch is reproduced;
- NaN selection behavior is reproduced in a focused test;
- unsafe model text such as `Nissan X-Trail / Rogue` is recognized as unsafe for filenames;
- Script 1 and Script 2 fleet-survival conventions are shown to disagree for the same cohort;
- imports of the new `icor` package perform no network calls and create no files;
- no tracked current-source file contains a credential-shaped OpenAI secret.

Known-bug tests must be marked with stable issue identifiers and an explicit expected-failure reason when the desired behavior is not implemented in this subproject. A passing test must never assert that an incorrect forecast is scientifically valid; it may only characterize the current result.

## Baseline report

A deterministic, read-only audit command will produce a JSON-compatible report in memory and optionally write it to a caller-selected path outside canonical data. It reports:

- source file inventory and record counts;
- missing required fields and null units;
- duplicate `(region, year, normalized model, normalized generation)` keys;
- ICOR catalog parse/match counts;
- configured runtime and forecast constants;
- repository-relative paths only.

The audit must not modify source files, call external services, include usernames, inspect secrets, or emit absolute user-machine paths in committed artifacts.

## Error handling

- Invalid configuration raises a typed configuration error with the setting name and a safe remediation message.
- Invalid source records are returned as structured audit findings containing file, record index, field, and reason.
- Missing optional credentials disable the corresponding external integration; they must not be converted into factual negative answers.
- CI treats test failures, lint failures, stale lock/export files, and vulnerable direct dependencies as failures.
- Known source-data defects remain visible as audit findings until the data-ingestion subproject defines and enforces the replacement policy.

## Verification and acceptance criteria

The subproject is complete only when all of the following are freshly verified:

1. A clean Python 3.12 environment can be created with `uv sync --locked` on Windows and Linux CI.
2. `uv lock --check` confirms that dependency metadata and the lock agree.
3. The production requirements export is reproducible from the lock.
4. Ruff passes on the files placed under its configured scope.
5. Pytest passes, with known defects reported as explicit expected failures rather than silently skipped tests.
6. The baseline audit produces the same normalized findings on repeated runs.
7. Tests make no external network calls and do not modify tracked data.
8. The devcontainer starts Streamlit without disabling CORS or XSRF protection.
9. Documentation gives exact local setup, test, audit, and app-start commands.
10. `git diff --check` reports no whitespace errors and the final worktree contains no secret or generated local output.

## Files and responsibilities

- `pyproject.toml`: project metadata, bounded direct dependencies, development dependencies, pytest and Ruff configuration.
- `uv.lock`: authoritative resolved dependency graph.
- `requirements.txt`: generated exact production dependency export.
- `.python-version`: Python 3.12 selection.
- `.devcontainer/devcontainer.json`: reproducible secure container setup.
- `.github/workflows/ci.yml`: Windows/Linux quality and security gates.
- `src/icor/__init__.py`: package boundary and version metadata.
- `src/icor/config.py`: import-safe validated non-secret runtime settings.
- `src/icor/audit.py`: pure read-only market/catalog audit functions.
- `scripts/audit_baseline.py`: thin command-line adapter for the audit functions.
- `tests/conftest.py`: network and secret isolation fixtures.
- `tests/test_config.py`: runtime configuration tests.
- `tests/test_audit.py`: deterministic source audit tests.
- `tests/characterization/`: focused current-behavior and known-defect tests.
- `docs/DEVELOPMENT.md`: exact local workflow and safety rules.
- `README.md`: concise project purpose and links to development and architecture documentation.

## Deferred from this foundation subproject

The following items are not excluded from the product program. Lucas explicitly requires the final product review to examine and, where evidence supports it, modify every one of them—including the statistical and machine-learning model choices. They are deferred only so the development foundation can be completed and verified before domain behavior changes:

- Fixing forecast formulas, ICOR matching, duplicate aggregation, or generation identity.
- Defining the vehicle-configuration and windshield-SKU schema.
- Importing the company's proprietary one-year dataset.
- Training or selecting a statistical or machine-learning forecasting model.
- Changing authentication, roles, output isolation, or the production deployment.
- Running OpenAI, search, Wikipedia, PostHog, or other network integrations during tests.

Those concerns are mandatory later subprojects and will receive their own approved specifications, model/data comparisons, acceptance criteria, and implementation plans. No existing algorithm or model is presumed correct merely because it is outside this first subproject.
