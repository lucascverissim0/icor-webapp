"""Fail-closed prerequisite checks for the explicitly named preview host.

The preview used to refuse to start anywhere but GitHub Codespaces. That was a
safe default while Codespaces was the only target, but refusing every other host
is not itself the security property: what matters is that the host is
*declared*, never guessed.

So the host mode is explicit. `ICOR_PREVIEW_HOST_MODE` names it. Codespaces sets
`CODESPACES=true` itself, which is an unambiguous declaration by the platform,
so that is accepted as naming codespaces mode. Anything else -- an empty,
misspelled or absent value with no Codespaces environment -- raises.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

from icor.preview.config import PreviewSettings

CODESPACES_MODE = "codespaces"
CONTAINER_MODE = "container"
HOST_MODE_VARIABLE = "ICOR_PREVIEW_HOST_MODE"
PUBLIC_ORIGIN_VARIABLE = "ICOR_PREVIEW_PUBLIC_ORIGIN"
TRUSTED_PROXIES_VARIABLE = "ICOR_PREVIEW_TRUSTED_PROXIES"
PORT_VARIABLE = "ICOR_PREVIEW_PORT"
DEFAULT_CONTAINER_PORT = 8080
_SUPPORTED_HOST_MODES = frozenset({CODESPACES_MODE, CONTAINER_MODE})


class RunnerError(RuntimeError):
    """The authenticated preview is not safe to start."""


def _require_https_origin(value: str) -> None:
    """A public origin must be plain, exact HTTPS and nothing else.

    Session cookies are issued `secure`, so on a non-TLS origin every login
    silently fails: the browser discards the cookie and the reviewer loops back
    to the form with no error to report.
    """

    parsed = urlparse(value.strip())
    try:
        port = parsed.port
    except ValueError as error:
        raise RunnerError("container host mode requires an HTTPS public origin") from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.path not in ("", "/")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise RunnerError("container host mode requires an HTTPS public origin")


def host_mode(environment: Mapping[str, str]) -> str:
    declared = environment.get(HOST_MODE_VARIABLE, "").strip().casefold()
    if not declared:
        if environment.get("CODESPACES", "").casefold() == "true":
            return CODESPACES_MODE
        raise RunnerError("preview runner requires an explicit supported host mode")
    if declared not in _SUPPORTED_HOST_MODES:
        raise RunnerError("preview runner requires an explicit supported host mode")
    if declared == CODESPACES_MODE:
        if environment.get("CODESPACES", "").casefold() != "true":
            raise RunnerError("codespaces host mode requires GitHub Codespaces")
        return CODESPACES_MODE
    _require_https_origin(environment.get(PUBLIC_ORIGIN_VARIABLE, ""))
    if not environment.get(TRUSTED_PROXIES_VARIABLE, "").strip():
        # Demanding an explicit answer is the point. `--proxy-headers` with no
        # trusted set makes `X-Forwarded-For` spoofable and throttling useless,
        # while omitting it collapses every reviewer into one throttle bucket.
        raise RunnerError("container host mode requires a declared trusted proxy set")
    if environment.get("ICOR_CLIENT_RELEASE_MODE", "").strip().casefold() != "verified":
        raise RunnerError("container host mode serves the verified client release only")
    return CONTAINER_MODE


def container_port(environment: Mapping[str, str]) -> int:
    raw = environment.get(PORT_VARIABLE, "").strip()
    if not raw:
        return DEFAULT_CONTAINER_PORT
    try:
        port = int(raw)
    except ValueError as error:
        raise RunnerError("preview port is invalid") from error
    if not 1 <= port <= 65_535:
        raise RunnerError("preview port is invalid")
    return port


def validate_runner(
    environment: Mapping[str, str],
    *,
    asset_root: Path,
    snapshot_root: Path,
    coverage_db: Path,
) -> PreviewSettings:
    host_mode(environment)
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


def server_command(
    mode: str = CODESPACES_MODE,
    *,
    host: str = "0.0.0.0",
    port: int = DEFAULT_CONTAINER_PORT,
    trusted_proxies: str | None = None,
) -> tuple[str, ...]:
    if mode == CODESPACES_MODE:
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
    if mode != CONTAINER_MODE or not trusted_proxies:
        raise RunnerError("preview server command requires a supported host mode")
    # `--proxy-headers` is not cosmetic. The login throttle keys on the client
    # address, so behind a proxy without it every reviewer shares one bucket and
    # five bad guesses lock the named reviewer out globally.
    return (
        "uvicorn",
        "icor.preview.app:create_preview_app",
        "--factory",
        "--host",
        host,
        "--port",
        str(port),
        "--proxy-headers",
        "--forwarded-allow-ips",
        trusted_proxies,
        "--timeout-keep-alive",
        "75",
    )
