"""The container artifacts have to keep two promises this test enforces."""

from __future__ import annotations

import importlib
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
FLY_TOML = ROOT / "fly.toml"

# The entrypoint is a script, not a package module.
sys.path.insert(0, str(ROOT / "scripts"))


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

    # The evidence rules were once a list of the paths that happened to be
    # large, and missed `.local/evidence/snapshots/` -- 26 GB, which a build
    # context would have uploaded. Deny everything and re-admit exactly one
    # path, so a new directory under .local cannot join a context silently.
    assert ".local/**" in patterns
    assert "!.local/client-evidence/" in patterns
    assert "!.local/client-evidence/**" in patterns
    assert not any(
        pattern.startswith(".local/") and pattern != ".local/**"
        for pattern in patterns
    ), patterns
    # Local agent state carries a live loopback session key.
    assert ".superpowers/" in patterns


def _requirement_names(items: list[str]) -> set[str]:
    return {re.split(r"[\[><=!~]", item, maxsplit=1)[0].strip() for item in items}


def test_the_runtime_image_does_not_install_the_openai_client() -> None:
    """The part of release gate 1 that code can actually satisfy.

    This test used to assert only that the `preview` extra left openai and
    streamlit out, and passed for months while the image would have carried
    both: a PEP 621 extra is *additive*, so `--extra preview` installs the whole
    of `[project.dependencies]` first. The base list is what decides, which is
    what is asserted now.
    """

    dockerfile = _dockerfile()

    assert "--extra preview" in dockerfile
    assert "--all-extras" not in dockerfile
    assert "--no-dev" in dockerfile
    # A group is only installed when it is asked for by name or is a default.
    assert "--group" not in dockerfile
    assert "--all-groups" not in dockerfile

    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = _requirement_names(manifest["project"]["dependencies"])
    preview = _requirement_names(
        manifest["project"]["optional-dependencies"]["preview"]
    )
    legacy = _requirement_names(manifest["dependency-groups"]["legacy"])

    for forbidden in ("openai", "streamlit", "streamlit-authenticator"):
        assert forbidden not in base, forbidden
        assert forbidden not in preview, forbidden
    assert {"openai", "streamlit"} <= legacy
    assert {"fastapi", "uvicorn", "argon2-cffi"} <= preview
    # `legacy` must not be a default group, or `--no-dev` would keep it.
    defaults = manifest.get("tool", {}).get("uv", {}).get("default-groups", ["dev"])
    assert "legacy" not in defaults


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


def test_the_entrypoint_finds_the_server_on_this_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The container's PID 1 must be runnable where it is developed.

    It resolved the bare name only, so on Windows it looked for `uvicorn` beside
    the interpreter where the file is `uvicorn.exe`. `os.execv` then raised
    `FileNotFoundError`, which the entrypoint does not catch, so the one piece of
    code that decides whether the container serves at all could not be exercised
    outside an image that had never been built.
    """

    module = importlib.import_module("run_container_preview")

    interpreter = tmp_path / "python.exe"
    interpreter.write_bytes(b"")
    monkeypatch.setattr(module.sys, "executable", str(interpreter))

    with pytest.raises(FileNotFoundError):
        module._server_executable("uvicorn")

    (tmp_path / "uvicorn.exe").write_bytes(b"")
    assert module._server_executable("uvicorn") == tmp_path / "uvicorn.exe"

    (tmp_path / "uvicorn").write_bytes(b"")
    assert module._server_executable("uvicorn") == tmp_path / "uvicorn"
