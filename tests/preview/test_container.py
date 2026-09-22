"""The container artifacts have to keep two promises this test enforces."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
FLY_TOML = ROOT / "fly.toml"


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_the_build_context_never_copies_the_repository_wholesale() -> None:
    """`COPY . .` would put the leaked key in the image, via Git history.

    The key is still reachable in four commits, so an allowlist is the only
    construction that can be checked.
    """

    for line in _dockerfile().splitlines():
        stripped = line.strip()
        if not stripped.startswith("COPY"):
            continue
        assert not re.match(r"^COPY\s+\.\s", stripped), stripped
        assert not re.match(r"^COPY\s+(--\S+\s+)*\.\s+\S+$", stripped), stripped


def test_the_dockerignore_denies_history_secrets_and_the_legacy_app() -> None:
    patterns = {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    for denied in (".git/", ".env", "ui/", ".venv/", "node_modules/"):
        assert denied in patterns, denied


def test_the_runtime_image_does_not_install_the_openai_client() -> None:
    """The part of release gate 1 that code can actually satisfy."""

    dockerfile = _dockerfile()

    assert "--extra preview" in dockerfile
    assert "--all-extras" not in dockerfile

    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    preview = manifest["project"]["optional-dependencies"]["preview"]
    names = {re.split(r"[\[><=!~]", item, maxsplit=1)[0].strip() for item in preview}

    assert "openai" not in names
    assert "streamlit" not in names
    assert {"fastapi", "uvicorn", "argon2-cffi"} <= names


def test_the_image_refuses_to_build_around_a_full_scope_snapshot() -> None:
    """A bad bake must be a red build, not a crash-looping machine."""

    dockerfile = _dockerfile()

    assert "open_active_snapshot" in dockerfile
    assert "manifest.scope == 'client-release'" in dockerfile


def test_fly_configuration_declares_no_secret_values() -> None:
    configuration = tomllib.loads(FLY_TOML.read_text(encoding="utf-8"))
    environment = configuration.get("env", {})

    for secret in (
        "ICOR_PREVIEW_USERS",
        "ICOR_PREVIEW_SESSION_SECRET",
        "ICOR_EXPORT_TOKEN",
    ):
        assert secret not in environment, secret
    assert "$argon2" not in FLY_TOML.read_text(encoding="utf-8")


def test_fly_serves_https_only_and_does_not_scale_to_zero() -> None:
    configuration = tomllib.loads(FLY_TOML.read_text(encoding="utf-8"))
    service = configuration["http_service"]

    assert service["force_https"] is True
    # Verifying the baked snapshot takes about forty seconds; scaling to zero
    # would charge that to the reviewer.
    assert service["auto_stop_machines"] is False
    assert service["min_machines_running"] >= 1
    assert configuration["env"]["ICOR_PREVIEW_PUBLIC_ORIGIN"].startswith("https://")


def test_the_container_entrypoint_validates_before_serving() -> None:
    entrypoint = (ROOT / "scripts" / "run_container_preview.py").read_text(
        encoding="utf-8"
    )

    assert "validate_runner" in entrypoint
    assert "os.execv" in entrypoint


@pytest.mark.parametrize(
    "variable",
    (
        "ICOR_PREVIEW_HOST_MODE",
        "ICOR_CLIENT_RELEASE_MODE",
        "ICOR_PREVIEW_ASSET_ROOT",
        "ICOR_EVIDENCE_ACTIVE_ROOT",
        "ICOR_COVERAGE_DB",
    ),
)
def test_the_image_declares_every_runtime_variable_the_runner_requires(
    variable: str,
) -> None:
    assert variable in _dockerfile()
