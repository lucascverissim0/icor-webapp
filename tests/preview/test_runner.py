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
    assert "image" not in payload
    assert payload["build"] == {"dockerfile": "Dockerfile"}
    assert payload["features"]["ghcr.io/devcontainers/features/sshd:1.1.0"] == {}
    assert payload["overrideFeatureInstallOrder"] == [
        "ghcr.io/devcontainers/features/sshd",
        "ghcr.io/devcontainers/features/node",
    ]
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


def _container_environment() -> dict[str, str]:
    environment = _valid_environment()
    del environment["CODESPACES"]
    environment.update(
        {
            "ICOR_PREVIEW_HOST_MODE": "container",
            "ICOR_PREVIEW_PUBLIC_ORIGIN": "https://icor-client-preview.example",
            "ICOR_PREVIEW_TRUSTED_PROXIES": "*",
            "ICOR_CLIENT_RELEASE_MODE": "verified",
        }
    )
    return environment


@pytest.mark.parametrize(
    "declared",
    ("", "   ", "fly", "kubernetes", "CODESPACES-ish"),
)
def test_runner_refuses_an_absent_or_unknown_host_mode(declared: str) -> None:
    """Absent, empty or misspelled must all fail closed, never fall back."""

    from icor.preview.runner import RunnerError, host_mode

    environment = {"ICOR_PREVIEW_HOST_MODE": declared}
    with pytest.raises(RunnerError, match="explicit supported host mode"):
        host_mode(environment)


def test_runner_refuses_an_empty_environment() -> None:
    from icor.preview.runner import RunnerError, host_mode

    with pytest.raises(RunnerError, match="explicit supported host mode"):
        host_mode({})


def test_codespaces_declares_itself_and_is_accepted() -> None:
    """The platform sets this variable, so it is a declaration, not a guess."""

    from icor.preview.runner import CODESPACES_MODE, host_mode

    assert host_mode({"CODESPACES": "true"}) == CODESPACES_MODE


def test_codespaces_mode_still_requires_the_codespaces_environment() -> None:
    from icor.preview.runner import RunnerError, host_mode

    with pytest.raises(RunnerError, match="requires GitHub Codespaces"):
        host_mode({"ICOR_PREVIEW_HOST_MODE": "codespaces", "CODESPACES": "false"})


@pytest.mark.parametrize(
    "origin",
    (
        "",
        "http://icor.example",
        "https://",
        "https://icor.example:8000",
        "https://user@icor.example",
        "https://icor.example/path",
        "https://icor.example?query=1",
        "https://icor.example#fragment",
    ),
)
def test_container_mode_requires_a_clean_https_public_origin(origin: str) -> None:
    from icor.preview.runner import RunnerError, host_mode

    environment = _container_environment()
    environment["ICOR_PREVIEW_PUBLIC_ORIGIN"] = origin
    with pytest.raises(RunnerError, match="HTTPS public origin"):
        host_mode(environment)


def test_container_mode_requires_declared_trusted_proxies() -> None:
    from icor.preview.runner import RunnerError, host_mode

    environment = _container_environment()
    environment["ICOR_PREVIEW_TRUSTED_PROXIES"] = "   "
    with pytest.raises(RunnerError, match="trusted proxy set"):
        host_mode(environment)


def test_container_mode_serves_only_the_verified_client_release() -> None:
    from icor.preview.runner import RunnerError, host_mode

    environment = _container_environment()
    environment["ICOR_CLIENT_RELEASE_MODE"] = ""
    with pytest.raises(RunnerError, match="verified client release"):
        host_mode(environment)


def test_container_mode_is_accepted_when_fully_declared() -> None:
    from icor.preview.runner import CONTAINER_MODE, host_mode

    assert host_mode(_container_environment()) == CONTAINER_MODE


def test_container_command_forwards_proxy_headers_to_a_declared_set() -> None:
    from icor.preview.runner import CONTAINER_MODE, server_command

    command = server_command(
        CONTAINER_MODE, host="0.0.0.0", port=8080, trusted_proxies="10.0.0.0/8"
    )

    assert "--proxy-headers" in command
    assert command[command.index("--forwarded-allow-ips") + 1] == "10.0.0.0/8"
    assert command[command.index("--port") + 1] == "8080"


def test_container_command_never_enables_reload_or_debug_logging() -> None:
    from icor.preview.runner import CONTAINER_MODE, server_command

    command = server_command(CONTAINER_MODE, trusted_proxies="*")

    assert "--reload" not in command
    assert "debug" not in command


def test_container_command_refuses_an_undeclared_proxy_set() -> None:
    from icor.preview.runner import CONTAINER_MODE, RunnerError, server_command

    with pytest.raises(RunnerError, match="supported host mode"):
        server_command(CONTAINER_MODE, trusted_proxies=None)


@pytest.mark.parametrize("raw", ("0", "65536", "-1", "http"))
def test_container_port_rejects_values_outside_the_port_range(raw: str) -> None:
    from icor.preview.runner import RunnerError, container_port

    with pytest.raises(RunnerError, match="port is invalid"):
        container_port({"ICOR_PREVIEW_PORT": raw})


def test_container_port_defaults_only_when_unset() -> None:
    from icor.preview.runner import DEFAULT_CONTAINER_PORT, container_port

    assert container_port({}) == DEFAULT_CONTAINER_PORT
    assert container_port({"ICOR_PREVIEW_PORT": "9000"}) == 9000
