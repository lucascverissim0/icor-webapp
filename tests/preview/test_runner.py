from __future__ import annotations

import json
from pathlib import Path

import pytest

from icor.preview.config import ConfigurationError


def _valid_environment() -> dict[str, str]:
    return {
        "CODESPACES": "true",
        "ICOR_PREVIEW_USERS": (
            '{"Lucas":"$argon2id$v=19$m=65536,t=3,p=4$'
            'c2FsdHNhbHRzYWx0c2FsdA$YWJjZA"}'
        ),
        "ICOR_PREVIEW_SESSION_SECRET": "c3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3M",
        "ICOR_EXPORT_TOKEN": "e" * 32,
    }


def test_runner_plan_requires_codespaces_configuration_snapshot_assets_and_export(
    tmp_path: Path,
) -> None:
    from icor.preview.runner import RunnerError, validate_runner

    assets = tmp_path / "dist"
    assets.mkdir()
    (assets / "index.html").write_text("app", encoding="utf-8")
    snapshot = tmp_path / "evidence"
    snapshot.mkdir()
    (snapshot / "active.json").write_text("{}", encoding="utf-8")
    coverage = tmp_path / "coverage.sqlite3"
    coverage.touch()

    environment = _valid_environment()
    validate_runner(environment, asset_root=assets, snapshot_root=snapshot, coverage_db=coverage)

    cases = (
        ({**environment, "CODESPACES": "false"}, assets, snapshot, coverage),
        (
            {key: value for key, value in environment.items() if key != "ICOR_EXPORT_TOKEN"},
            assets,
            snapshot,
            coverage,
        ),
        (environment, tmp_path / "missing-assets", snapshot, coverage),
        (environment, assets, tmp_path / "missing-snapshot", coverage),
        (environment, assets, snapshot, tmp_path / "missing-coverage"),
    )
    for selected, selected_assets, selected_snapshot, selected_coverage in cases:
        with pytest.raises((RunnerError, ConfigurationError)):
            validate_runner(
                selected,
                asset_root=selected_assets,
                snapshot_root=selected_snapshot,
                coverage_db=selected_coverage,
            )


def test_codespaces_command_is_explicit_and_fixed() -> None:
    from icor.preview.runner import server_command

    assert server_command() == (
        "uv",
        "run",
        "uvicorn",
        "icor.preview.app:create_preview_app",
        "--factory",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    )


def test_devcontainer_never_autostarts_or_publishes_preview() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / ".devcontainer" / "devcontainer.json").read_text("utf-8"))
    serialized = json.dumps(payload).casefold()

    assert "streamlit" not in serialized
    assert "postattachcommand" not in {key.casefold() for key in payload}
    assert payload["portsAttributes"]["8000"] == {
        "label": "ICOR authenticated preview",
        "onAutoForward": "silent",
    }
    assert payload["forwardPorts"] == [8000]
    assert payload["features"]["ghcr.io/devcontainers/features/sshd:1.1.0"] == {}
    lock = json.loads((root / ".devcontainer" / "devcontainer-lock.json").read_text("utf-8"))
    sshd_lock = lock["features"]["ghcr.io/devcontainers/features/sshd:1.1.0"]
    assert sshd_lock == {
        "version": "1.1.0",
        "resolved": (
            "ghcr.io/devcontainers/features/sshd@sha256:"
            "f5251b8e4325f68f7280973c6cd65daff414449c66f240621502d4e8e74eb7ee"
        ),
        "integrity": (
            "sha256:"
            "f5251b8e4325f68f7280973c6cd65daff414449c66f240621502d4e8e74eb7ee"
        ),
    }
    assert "public" not in serialized
    for forbidden in ("password", "argon2", "session_secret", "icor_preview_users", "cors", "xsrf"):
        assert forbidden not in serialized


def test_local_runner_remains_loopback_only() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts" / "run_planner_dev.py").read_text("utf-8")
    assert '"127.0.0.1"' in source
    assert '"0.0.0.0"' not in source
