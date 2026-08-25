# ICOR Development Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python 3.12 development environment, safe import boundary, deterministic source-data audit, characterization suite, and cross-platform quality gates without changing current forecasting behavior.

**Architecture:** `pyproject.toml` and `uv.lock` become the dependency source of truth, with an exact generated `requirements.txt` retained for Streamlit deployment. A new import-safe `icor` package contains validated non-secret settings and pure audit logic; all external integrations remain outside this boundary. Pytest records deterministic repository behavior and known defects, while CI verifies the same locked environment on Windows and Linux.

**Tech Stack:** Python 3.12, uv 0.11.3, pytest, Ruff, pip-audit, GitHub Actions, Streamlit.

**Spec:** `docs/superpowers/specs/2026-08-25-foundation-design.md`

## Global Constraints

- Work only in `C:\Users\LucasCravoVERISSIMO\icor-webapp-development` on `development/windshield-demand-platform`.
- Treat `C:\Users\LucasCravoVERISSIMO\icor-webapp` and branch `main` as read-only.
- Do not push, merge, deploy, rewrite Git history, rotate secrets, or access production services/data.
- Python 3.12 is the sole supported project runtime.
- Tests must clear integration credentials and prohibit unexpected external network access.
- Never overwrite `data/passenger_car_data.xlsx` or canonical source files during tests.
- Preserve current application output behavior; known defects are characterized, not repaired, in this subproject.
- The proprietary company dataset is not imported in this subproject.
- Forecasting algorithms and ML choices remain mandatory later review work; this plan does not endorse them.

---

### Task 1: Establish the locked Python 3.12 toolchain

**Files:**
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `uv.lock`
- Modify: `requirements.txt`
- Create: `tests/test_toolchain.py`

**Interfaces:**
- Consumes: the direct runtime dependencies currently declared in `requirements.txt`.
- Produces: Python constraint `>=3.12,<3.13`, uv dependency groups, deterministic `uv.lock`, and an exact production requirements export.

- [ ] **Step 1: Write the toolchain contract test**

```python
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_project_requires_only_python_312() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["requires-python"] == ">=3.12,<3.13"
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_production_requirements_are_exact_pins() -> None:
    lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines
    requirements = [line.split(";", 1)[0].strip() for line in lines]
    assert all(re.match(r"^[A-Za-z0-9_.-]+==[^=; ]+$", line) for line in requirements)
```

- [ ] **Step 2: Run the test and verify the missing project metadata fails**

Run: `python -m pytest tests/test_toolchain.py -v`

Expected: FAIL because `pyproject.toml` and `.python-version` do not exist.

- [ ] **Step 3: Create the project metadata and Python selector**

Create `.python-version` containing exactly:

```text
3.12
```

Create `pyproject.toml` with:

```toml
[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[project]
name = "icor-windshield-demand"
version = "0.1.0"
description = "Auditable windshield demand forecasting application"
requires-python = ">=3.12,<3.13"
dependencies = [
  "bcrypt>=4.1,<5",
  "beautifulsoup4>=4.12,<5",
  "numpy>=1.26,<3",
  "openai>=1.10,<2",
  "openpyxl>=3.1,<4",
  "pandas>=2.2,<3",
  "posthog>=3.6,<4",
  "requests>=2.32.4,<3",
  "streamlit>=1.54,<2",
  "streamlit-authenticator>=0.4,<0.5",
  "tabulate>=0.9,<1",
]

[dependency-groups]
dev = [
  "pip-audit>=2.9,<3",
  "pytest>=8.3,<9",
  "pytest-socket>=0.7,<1",
  "ruff>=0.12,<1",
]

[tool.hatch.build.targets.wheel]
packages = ["src/icor"]

[tool.pytest.ini_options]
addopts = "-ra --strict-config --strict-markers"
testpaths = ["tests"]
xfail_strict = true

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

- [ ] **Step 4: Install the project runtime and resolve the lock**

Run: `uv python install 3.12`

Expected: a managed CPython 3.12 runtime is available to uv.

Run: `uv lock`

Expected: `uv.lock` is created with no resolution errors.

Run: `uv sync --locked --all-groups`

Expected: `.venv` is created and all runtime/development dependencies install successfully.

- [ ] **Step 5: Export exact production requirements**

Run: `uv export --locked --no-dev --no-hashes --no-emit-project --format requirements.txt --output-file requirements.txt`

Expected: every non-comment requirement uses `==` and the export contains no development-only packages.

- [ ] **Step 6: Run the toolchain contract and lock checks**

Run: `uv run pytest tests/test_toolchain.py -v`

Expected: PASS.

Run: `uv lock --check`

Expected: exit 0 with no stale-lock warning.

- [ ] **Step 7: Commit the toolchain foundation**

```powershell
git add .python-version pyproject.toml uv.lock requirements.txt tests/test_toolchain.py
git commit -m "build: lock Python 3.12 development environment"
```

### Task 2: Add import-safe validated runtime settings

**Files:**
- Create: `src/icor/__init__.py`
- Create: `src/icor/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: a repository root `Path` and optional `Mapping[str, str]` environment.
- Produces: `Settings`, `ConfigurationError`, and `load_settings(root: Path, environ: Mapping[str, str] | None = None) -> Settings`.

- [ ] **Step 1: Write failing settings tests**

```python
from pathlib import Path

import pytest

from icor.config import ConfigurationError, load_settings


def test_load_settings_uses_repository_relative_defaults(tmp_path: Path) -> None:
    settings = load_settings(tmp_path, {})
    assert settings.environment == "local"
    assert settings.data_dir == tmp_path / "data"
    assert settings.output_dir == tmp_path / ".local" / "outputs"
    assert settings.external_network_enabled is False


def test_load_settings_rejects_unknown_environment(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="ICOR_ENVIRONMENT"):
        load_settings(tmp_path, {"ICOR_ENVIRONMENT": "mystery"})


def test_network_requires_explicit_true(tmp_path: Path) -> None:
    settings = load_settings(tmp_path, {"ICOR_EXTERNAL_NETWORK": "true"})
    assert settings.external_network_enabled is True
```

- [ ] **Step 2: Run the tests and verify the missing package fails**

Run: `uv run pytest tests/test_config.py -v`

Expected: collection ERROR with `ModuleNotFoundError: No module named 'icor'`.

- [ ] **Step 3: Implement the minimal configuration boundary**

Create `src/icor/__init__.py`:

```python
"""Import-safe core for the ICOR windshield-demand application."""

__version__ = "0.1.0"
```

Create `src/icor/config.py`:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path


class ConfigurationError(ValueError):
    """Raised when a non-secret runtime setting is invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    environment: str
    data_dir: Path
    output_dir: Path
    external_network_enabled: bool


def load_settings(root: Path, environ: Mapping[str, str] | None = None) -> Settings:
    values = os.environ if environ is None else environ
    environment = values.get("ICOR_ENVIRONMENT", "local").strip().lower()
    if environment not in {"local", "test", "production"}:
        raise ConfigurationError(
            "ICOR_ENVIRONMENT must be one of: local, test, production"
        )
    network_value = values.get("ICOR_EXTERNAL_NETWORK", "false").strip().lower()
    if network_value not in {"true", "false"}:
        raise ConfigurationError("ICOR_EXTERNAL_NETWORK must be true or false")
    resolved_root = root.resolve()
    return Settings(
        environment=environment,
        data_dir=resolved_root / "data",
        output_dir=resolved_root / ".local" / "outputs",
        external_network_enabled=network_value == "true",
    )
```

- [ ] **Step 4: Run the configuration tests**

Run: `uv run pytest tests/test_config.py -v`

Expected: 3 passed.

- [ ] **Step 5: Verify import has no filesystem side effects**

Run: `uv run python -c "from icor.config import load_settings; print(load_settings.__name__)"`

Expected: prints `load_settings`; `git status --short` shows no generated application data.

- [ ] **Step 6: Commit the configuration boundary**

```powershell
git add src/icor/__init__.py src/icor/config.py tests/test_config.py
git commit -m "feat: add safe runtime configuration boundary"
```

### Task 3: Build the deterministic read-only data audit

**Files:**
- Create: `src/icor/audit.py`
- Create: `tests/test_audit.py`

**Interfaces:**
- Consumes: `audit_repository(root: Path) -> AuditReport`.
- Produces: immutable `AuditFinding`, immutable `AuditReport`, and `AuditReport.to_dict() -> dict[str, object]` containing only repository-relative paths, configured runtime data, catalog counts, and source-extracted forecast constants.

- [ ] **Step 1: Write failing audit tests with a minimal repository fixture**

```python
import json
from pathlib import Path

from icor.audit import audit_repository


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_audit_reports_duplicate_identity_and_null_units(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    write_json(
        data / "Top100_2025.txt",
        [
            {"model": "Example One", "generation": "I", "units_sold": 100},
            {"model": "Example One", "generation": "I", "units_sold": None},
        ],
    )
    (data / "icor_supported_models.txt").write_text('{"example one": {2025: "G1"}}')

    report = audit_repository(tmp_path)

    assert report.file_count == 1
    assert report.record_count == 2
    assert report.catalog_model_count == 1
    assert report.catalog_match_count == 1
    assert [finding.code for finding in report.findings] == [
        "duplicate_identity",
        "incomplete_top100",
        "null_units",
        "unsupported_icor_catalog_format",
    ]


def test_audit_dictionary_is_deterministic_and_relative(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    write_json(
        data / "Top100_World_2025.txt",
        [{"model": "Example Two", "generation": "II", "units_sold": 20}],
    )
    (data / "icor_supported_models.txt").write_text("example two\t2025")

    first = audit_repository(tmp_path).to_dict()
    second = audit_repository(tmp_path).to_dict()

    assert first == second
    assert str(tmp_path) not in json.dumps(first)
```

- [ ] **Step 2: Run the audit tests and verify the missing module fails**

Run: `uv run pytest tests/test_audit.py -v`

Expected: collection ERROR because `icor.audit` does not exist.

- [ ] **Step 3: Implement immutable findings and filename classification**

Implement these public types in `src/icor/audit.py`:

```python
@dataclass(frozen=True, slots=True, order=True)
class AuditFinding:
    code: str
    path: str
    record_index: int | None
    field: str | None
    message: str


@dataclass(frozen=True, slots=True)
class AuditReport:
    file_count: int
    record_count: int
    catalog_model_count: int
    catalog_match_count: int
    runtime_python: str
    forecast_constants: tuple[tuple[str, int | float | str], ...]
    findings: tuple[AuditFinding, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "file_count": self.file_count,
            "record_count": self.record_count,
            "catalog_model_count": self.catalog_model_count,
            "catalog_match_count": self.catalog_match_count,
            "runtime_python": self.runtime_python,
            "forecast_constants": dict(self.forecast_constants),
            "findings": [asdict(finding) for finding in self.findings],
        }
```

Use strict patterns `Top100_YYYY.txt` for EU and `Top100_World_YYYY.txt` for World. Normalize duplicate keys with Unicode `casefold()` and collapsed whitespace while preserving punctuation. Sort input paths and findings before returning.

- [ ] **Step 4: Implement audit rules**

Implement `audit_repository(root)` to:

1. Parse every matching market file as UTF-8 JSON.
2. Report `invalid_json`, `non_list_document`, `non_object_record`, and `missing_field` safely.
3. Accept `units_sold`, falling back to `projected_units_2025`; report `null_units` when neither contains a number.
4. Report `incomplete_top100` when a file contains other than 100 records.
5. Report `duplicate_identity` for repeated `(region, year, model, generation)` keys.
6. Report `unsupported_icor_catalog_format` when the catalog is neither valid JSON nor a tab-delimited catalog.
7. Count quoted top-level catalog model keys without executing the custom mapping, and report how many normalized catalog names intersect with normalized market model names.
8. Parse `DECAY_RATE`, `REPL_RATE_MEAN`, `REPAIR_RATE`, `YEARS_TO_PROJECT`, `SELECTED_YEAR`, and `DEADLINE_YEAR` from Script 1/2 syntax trees without importing either script; store sorted `(qualified_name, value)` pairs.
9. Record the configured runtime as `major.minor` from `sys.version_info`.
10. Emit repository-relative POSIX-style paths only.

- [ ] **Step 5: Run focused and repository audit tests**

Run: `uv run pytest tests/test_audit.py -v`

Expected: PASS.

Add an integration test asserting the real repository has 22 market files, 2,190 records, and 50 null-unit records. Run it and confirm PASS.

- [ ] **Step 6: Commit the read-only audit core**

```powershell
git add src/icor/audit.py tests/test_audit.py
git commit -m "feat: add deterministic source data audit"
```

### Task 4: Add a safe baseline-audit command

**Files:**
- Create: `scripts/audit_baseline.py`
- Create: `tests/test_audit_cli.py`

**Interfaces:**
- Consumes: `python scripts/audit_baseline.py [--root PATH] [--output PATH]`.
- Produces: deterministic UTF-8 JSON on stdout or at the explicitly supplied output path; returns 0 for a completed audit even when findings exist and 2 for invocation/configuration errors.

- [ ] **Step 1: Write the failing CLI test**

```python
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_audit_cli_is_read_only_and_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    before = (ROOT / "data" / "passenger_car_data.xlsx").read_bytes()
    command = [
        sys.executable,
        str(ROOT / "scripts" / "audit_baseline.py"),
        "--root",
        str(ROOT),
        "--output",
        str(output),
    ]

    first = subprocess.run(command, check=False, capture_output=True, text=True)
    first_bytes = output.read_bytes()
    second = subprocess.run(command, check=False, capture_output=True, text=True)

    assert first.returncode == second.returncode == 0
    assert first_bytes == output.read_bytes()
    assert json.loads(first_bytes)["file_count"] == 22
    assert (ROOT / "data" / "passenger_car_data.xlsx").read_bytes() == before
```

- [ ] **Step 2: Run the CLI test and verify the missing script fails**

Run: `uv run pytest tests/test_audit_cli.py -v`

Expected: FAIL because `scripts/audit_baseline.py` does not exist.

- [ ] **Step 3: Implement the thin CLI adapter**

Use `argparse`, call only `icor.audit.audit_repository`, serialize with `json.dumps(..., indent=2, sort_keys=True, ensure_ascii=False) + "\n"`, and write only when `--output` is explicitly provided. Resolve the default root from the script's parent repository directory.

- [ ] **Step 4: Run CLI tests and a manual audit**

Run: `uv run pytest tests/test_audit_cli.py -v`

Expected: PASS.

Run: `uv run python scripts/audit_baseline.py`

Expected: valid JSON on stdout, no network activity, and no tracked file changes.

- [ ] **Step 5: Commit the audit command**

```powershell
git add scripts/audit_baseline.py tests/test_audit_cli.py
git commit -m "feat: expose read-only baseline audit command"
```

### Task 5: Characterize known deterministic defects and enforce test isolation

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/characterization/source_loader.py`
- Create: `tests/characterization/test_known_defects.py`
- Create: `tests/test_repository_security.py`

**Interfaces:**
- Consumes: current tracked source files and market/catalog fixtures.
- Produces: stable `ICOR-001`, `ICOR-006`, `ICOR-009`, and `ICOR-030` expected-failure tests plus network/secret isolation gates.

- [ ] **Step 1: Add the network and credential isolation fixture**

Create an autouse fixture that deletes `OPENAI_API_KEY`, `SERPAPI_KEY`, `POSTHOG_API_KEY`, and `ICOR_EXTERNAL_NETWORK` with `monkeypatch.delenv`, and enable pytest-socket's network prohibition through `--disable-socket` in pytest `addopts`.

- [ ] **Step 2: Write the credential-shape scan test**

Use `git ls-files -z` to inspect current tracked text files only. Reject patterns matching `sk-[A-Za-z0-9_-]{20,}` while skipping known binary extensions. The test must never print a matching value; on failure it reports only the repository-relative path.

- [ ] **Step 3: Create a safe AST function loader**

Implement `load_function(path: Path, function_name: str, globals_dict: dict[str, object])` by parsing the source with `ast.parse`, compiling only the named `FunctionDef`, executing it in the supplied minimal globals, and returning the resulting callable. Never import `scripts/script1.py`, because it executes its pipeline at module scope.

- [ ] **Step 4: Write strict expected-failure tests for known defects**

```python
@pytest.mark.xfail(strict=True, reason="ICOR-001: Script 1 cannot parse the catalog mapping")
def test_icor_001_supported_audi_a3_is_recognized() -> None:
    parser = load_function(SCRIPT1, "parse_icor_models", {"re": re})
    catalog = (ROOT / "data" / "icor_supported_models.txt").read_text(encoding="utf-8")
    assert "Audi A3" in parser(catalog)


@pytest.mark.xfail(strict=True, reason="ICOR-006: generated filenames retain path separators")
def test_icor_006_model_filename_has_no_separator() -> None:
    generated = "Nissan X-Trail / Rogue".replace(" ", "_")
    assert "/" not in generated and "\\" not in generated


@pytest.mark.xfail(strict=True, reason="ICOR-009: NaN wins over a valid regional value")
def test_icor_009_valid_year_count_wins_over_nan() -> None:
    assert max(float("nan") or 0, 5 or 0) == 5


@pytest.mark.xfail(strict=True, reason="ICOR-030: fleet survival conventions disagree")
def test_icor_030_one_year_old_cohort_has_one_survival_value() -> None:
    decay = 0.0556
    script1_survival = (1 - decay) ** 1
    script2_survival = 1.0
    assert script1_survival == script2_survival
```

- [ ] **Step 5: Verify the characterization suite**

Run: `uv run pytest tests/characterization tests/test_repository_security.py -v`

Expected: all safety tests pass and exactly 4 tests are reported as XFAIL with the named issue identifiers.

- [ ] **Step 6: Commit characterization and safety tests**

```powershell
git add pyproject.toml tests/conftest.py tests/characterization tests/test_repository_security.py
git commit -m "test: characterize known deterministic defects"
```

### Task 6: Secure the devcontainer and add cross-platform CI

**Files:**
- Modify: `.devcontainer/devcontainer.json`
- Create: `.github/workflows/ci.yml`
- Extend: `tests/test_toolchain.py`

**Interfaces:**
- Consumes: locked Python 3.12 toolchain and the complete test suite.
- Produces: secure devcontainer startup and Windows/Linux CI gates.

- [ ] **Step 1: Write failing configuration assertions**

Extend `tests/test_toolchain.py` to assert the devcontainer contains `3.12-bookworm`, uses `uv sync --locked --all-groups`, and contains neither `enableCORS false`, `enableXsrfProtection false`, `apt upgrade`, nor an unpinned `pip install streamlit` command.

- [ ] **Step 2: Run the assertions and verify they fail against the current container**

Run: `uv run pytest tests/test_toolchain.py -v`

Expected: FAIL on the current Python 3.11 image and insecure Streamlit flags.

- [ ] **Step 3: Replace the devcontainer configuration**

Use image `mcr.microsoft.com/devcontainers/python:1-3.12-bookworm`, install uv 0.11.3 as a container feature or explicit pinned setup command, run `uv sync --locked --all-groups`, and start the app with `uv run streamlit run ui/app.py`. Do not pass CORS or XSRF overrides.

- [ ] **Step 4: Add the CI workflow**

Create `.github/workflows/ci.yml` with pull-request and push triggers for the development branch, a matrix of `ubuntu-latest` and `windows-latest`, `actions/checkout`, `astral-sh/setup-uv` pinned to uv `0.11.3`, Python 3.12 installation, `uv sync --locked --all-groups`, `uv lock --check`, `uv run ruff check src tests scripts/audit_baseline.py`, and `uv run pytest`.

Add a separate Linux audit job running `uv run pip-audit`. Do not give the workflow write permissions or repository secrets.

- [ ] **Step 5: Run local configuration, lint, and test checks**

Run: `uv run pytest tests/test_toolchain.py -v`

Expected: PASS.

Run: `uv run ruff check src tests scripts/audit_baseline.py`

Expected: PASS with no warnings.

- [ ] **Step 6: Commit secure environment gates**

```powershell
git add .devcontainer/devcontainer.json .github/workflows/ci.yml tests/test_toolchain.py
git commit -m "ci: add secure cross-platform quality gates"
```

### Task 7: Document the reproducible local workflow

**Files:**
- Modify: `README.md`
- Create: `docs/DEVELOPMENT.md`
- Create: `.streamlit/secrets.example.toml`
- Extend: `tests/test_repository_security.py`

**Interfaces:**
- Consumes: the commands established in Tasks 1-6.
- Produces: exact setup, audit, test, and local-app commands plus non-secret configuration examples.

- [ ] **Step 1: Write documentation safety assertions**

Add tests asserting that `.streamlit/secrets.example.toml` contains no `sk-` token, that `.streamlit/secrets.toml` remains ignored by `git check-ignore`, and that `docs/DEVELOPMENT.md` names the development worktree, `uv sync --locked --all-groups`, `uv run pytest`, `uv run python scripts/audit_baseline.py`, and `uv run streamlit run ui/app.py`.

- [ ] **Step 2: Run the assertions and verify missing documentation fails**

Run: `uv run pytest tests/test_repository_security.py -v`

Expected: FAIL because the example and development guide do not exist.

- [ ] **Step 3: Write concise project and development documentation**

Document:

- the windshield-demand business objective;
- protected `main` versus the development worktree rule;
- Python 3.12/uv installation and locked sync;
- blank local secret placeholders and the prohibition on production data/secrets;
- audit, lint, test, and local Streamlit commands;
- the existing forecast's prototype status;
- the fact that statistical/ML approaches will be benchmarked later and are not selected by this foundation.

The secrets example may contain only empty values under `[openai]`, `[serpapi]`, `[posthog]`, and a clearly local-only `[users.example]` record with no usable password.

- [ ] **Step 4: Run documentation safety tests**

Run: `uv run pytest tests/test_repository_security.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the documentation**

```powershell
git add README.md docs/DEVELOPMENT.md .streamlit/secrets.example.toml tests/test_repository_security.py
git commit -m "docs: add safe reproducible development workflow"
```

### Task 8: Perform full foundation verification and update continuity

**Files:**
- Modify: `docs/CODEX_HANDOFF.md`

**Interfaces:**
- Consumes: all foundation deliverables.
- Produces: fresh verification evidence and an exact next-subproject checkpoint.

- [ ] **Step 1: Verify the locked dependency graph and export**

Run: `uv lock --check`

Expected: exit 0.

Run: `uv export --locked --no-dev --no-hashes --no-emit-project --format requirements.txt --output-file requirements.txt`

Expected: exit 0 followed by `git diff --exit-code -- requirements.txt`.

- [ ] **Step 2: Run all static and automated checks**

Run: `uv run ruff check src tests scripts/audit_baseline.py`

Expected: exit 0.

Run: `uv run pytest`

Expected: all non-xfail tests pass and exactly 4 known-defect tests are XFAIL.

Run: `uv run pip-audit`

Expected: no known vulnerabilities in the resolved environment. If findings exist, stop and resolve or document the blocking dependency before claiming completion.

- [ ] **Step 3: Verify deterministic audit and tracked-data integrity**

Run the baseline audit twice to two temporary files, compare their SHA-256 hashes, and verify the hashes match. Compare `git hash-object data/passenger_car_data.xlsx` with its value at the start of implementation; they must be identical.

- [ ] **Step 4: Verify production isolation**

From `C:\Users\LucasCravoVERISSIMO\icor-webapp`, run `git status --short --branch`, compare `HEAD` with `origin/main`, and run `git diff --exit-code origin/main --`.

Expected: clean `main`, identical commit IDs, and no tracked diff.

- [ ] **Step 5: Update the durable handoff**

Record exact commands, pass/fail counts, dependency-audit results, commits, environment state, no-server/server status, unresolved expected failures, and the next recommended subproject. Do not record machine-absolute temporary paths, secrets, or proprietary data.

- [ ] **Step 6: Verify repository hygiene and commit the checkpoint**

Run: `git diff --check`

Expected: no output and exit 0.

Run: `git status --short`

Expected before the checkpoint commit: only `docs/CODEX_HANDOFF.md` is modified.

```powershell
git add docs/CODEX_HANDOFF.md
git commit -m "docs: record verified foundation checkpoint"
```
