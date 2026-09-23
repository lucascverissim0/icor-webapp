#!/usr/bin/env python3
"""Container entrypoint for the authenticated client preview.

Validates the declared host and every prerequisite, then replaces this process
with uvicorn. `execv` rather than a subprocess, so the server is PID 1 and
receives the stop signal directly instead of through a shim that might not
forward it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from icor.api.app import DEFAULT_EVIDENCE_ROOT
from icor.preview.runner import (
    CONTAINER_MODE,
    RunnerError,
    container_port,
    host_mode,
    server_command,
    validate_runner,
)

ASSET_ROOT_VARIABLE = "ICOR_PREVIEW_ASSET_ROOT"
SNAPSHOT_ROOT_VARIABLE = "ICOR_EVIDENCE_ACTIVE_ROOT"
COVERAGE_DB_VARIABLE = "ICOR_COVERAGE_DB"


def main(argv: list[str] | None = None) -> int:
    del argv
    environment = os.environ
    try:
        mode = host_mode(environment)
        if mode != CONTAINER_MODE:
            raise RunnerError("this entrypoint serves container host mode only")
        asset_root = Path(environment[ASSET_ROOT_VARIABLE])
        snapshot_root = Path(
            environment.get(SNAPSHOT_ROOT_VARIABLE, str(DEFAULT_EVIDENCE_ROOT))
        )
        coverage_db = Path(environment[COVERAGE_DB_VARIABLE])
        validate_runner(
            environment,
            asset_root=asset_root,
            snapshot_root=snapshot_root,
            coverage_db=coverage_db,
        )
        command = server_command(
            CONTAINER_MODE,
            host="0.0.0.0",
            port=container_port(environment),
            trusted_proxies=environment["ICOR_PREVIEW_TRUSTED_PROXIES"],
        )
    except (RunnerError, KeyError, ValueError) as error:
        # Never echo the environment: it carries the session secret and the
        # reviewer password hash.
        print(f"preview refused to start: {type(error).__name__}", file=sys.stderr)
        return 2

    try:
        executable = _server_executable(command[0])
    except FileNotFoundError as error:
        print(f"preview refused to start: {error}", file=sys.stderr)
        return 2
    os.execv(str(executable), list(command))


def _server_executable(name: str) -> Path:
    """Find the server beside this interpreter, whatever the platform calls it.

    This resolved the bare name only, so on Windows it looked for `uvicorn`
    where the file is `uvicorn.exe`, `os.execv` raised `FileNotFoundError`, and
    that is not one of the exceptions caught above — a bare traceback instead of
    a diagnosis. The image runs on Linux, where the bare name is right, so the
    defect was invisible there and made this file impossible to exercise
    anywhere else. It is the container's PID 1; it should be runnable locally.
    """

    directory = Path(sys.executable).parent
    for candidate in (directory / name, directory / f"{name}.exe"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{name} is not installed beside {sys.executable}")


if __name__ == "__main__":
    sys.exit(main())
