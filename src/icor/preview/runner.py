"""Fail-closed prerequisite checks for the explicit Codespaces server runner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from icor.preview.config import PreviewSettings


class RunnerError(RuntimeError):
    """The authenticated preview is not safe to start."""


def validate_runner(
    environment: Mapping[str, str],
    *,
    asset_root: Path,
    snapshot_root: Path,
    coverage_db: Path,
) -> PreviewSettings:
    if environment.get("CODESPACES", "").casefold() != "true":
        raise RunnerError("preview runner requires GitHub Codespaces")
    settings = PreviewSettings.from_environment(environment)
    export_token = environment.get("ICOR_EXPORT_TOKEN", "")
    if len(export_token) < 32:
        raise RunnerError("preview export authorization is unavailable")
    if not (asset_root / "index.html").is_file():
        raise RunnerError("compiled preview frontend is unavailable")
    if not (snapshot_root / "active.json").is_file():
        raise RunnerError("active preview snapshot is unavailable")
    if not coverage_db.is_file():
        raise RunnerError("coverage database is unavailable")
    return settings


def server_command() -> tuple[str, ...]:
    return (
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