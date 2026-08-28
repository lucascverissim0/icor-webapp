import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_requires_only_python_312() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["requires-python"] == ">=3.12,<3.13"
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_production_requirements_are_exact_pins() -> None:
    lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines
    requirements = [line.split(";", 1)[0].strip() for line in lines]
    assert all(re.match(r"^[A-Za-z0-9_.-]+==[^=; ]+$", line) for line in requirements)


def test_core_package_imports_outside_pytest() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import icor; print(icor.__version__)"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0"


def test_devcontainer_uses_locked_secure_runtime() -> None:
    configuration = (ROOT / ".devcontainer" / "devcontainer.json").read_text(encoding="utf-8")
    assert "3.12-bookworm" in configuration
    assert "uv sync --locked --all-groups" in configuration
    assert "enableCORS false" not in configuration
    assert "enableXsrfProtection false" not in configuration
    assert "apt upgrade" not in configuration
    assert not re.search(r"pip(?:3)? install(?: --user)? streamlit", configuration)


def test_planner_local_operation_is_documented() -> None:
    required_phrases = (
        "scripts/run_planner_dev.py",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000/docs",
        "no runtime demonstration-data fallback",
        "no production secrets",
        "no customer data",
    )
    for relative_path in ("README.md", "docs/DEVELOPMENT.md"):
        contents = (ROOT / relative_path).read_text(encoding="utf-8").lower()
        for phrase in required_phrases:
            assert phrase.lower() in contents, f"{relative_path} must document {phrase!r}"
