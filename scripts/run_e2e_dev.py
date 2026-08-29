"""Run the explicit browser-test API and Vite client together."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path

if __package__:
    from scripts.e2e_fixture import prepare_e2e_fixture
else:
    from e2e_fixture import prepare_e2e_fixture

ROOT = Path(__file__).resolve().parents[1]


def fixture_root_for(api_port: int, web_port: int) -> Path:
    return ROOT / ".local" / f"e2e-fixture-{api_port}-{web_port}"


def prepare_environment(
    environment: Mapping[str, str],
    *,
    fixture_root: Path = ROOT / ".local" / "e2e-fixture",
) -> dict[str, str]:
    prepared = dict(environment)
    evidence = prepared.get("ICOR_E2E_EVIDENCE_CANDIDATE")
    generation = prepared.get("ICOR_E2E_GENERATION_CANDIDATE")
    if bool(evidence) != bool(generation):
        raise ValueError("E2E evidence and generation candidates must both be configured")
    if not evidence:
        candidate = str(prepare_e2e_fixture(fixture_root))
        prepared["ICOR_E2E_EVIDENCE_CANDIDATE"] = candidate
        prepared["ICOR_E2E_GENERATION_CANDIDATE"] = candidate
    return prepared


def _stop(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)


def run(api_port: int, web_port: int) -> int:
    uv = shutil.which("uv")
    npm = shutil.which("npm")
    if uv is None or npm is None:
        return 2
    process_environment = prepare_environment(
        os.environ,
        fixture_root=fixture_root_for(api_port, web_port),
    )
    web_environment = process_environment.copy()
    web_environment["ICOR_API_ORIGIN"] = f"http://127.0.0.1:{api_port}"
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    processes: list[subprocess.Popen[bytes]] = []
    try:
        processes.append(
            subprocess.Popen(
                [
                    uv,
                    "run",
                    "uvicorn",
                    "scripts.e2e_app:create_e2e_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(api_port),
                ],
                cwd=ROOT,
                env=process_environment,
                creationflags=flags,
                start_new_session=os.name != "nt",
            )
        )
        processes.append(
            subprocess.Popen(
                [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(web_port)],
                cwd=ROOT / "web",
                env=web_environment,
                creationflags=flags,
                start_new_session=os.name != "nt",
            )
        )
        while True:
            for process in processes:
                if (code := process.poll()) is not None:
                    return code or 1
            time.sleep(0.2)
    except KeyboardInterrupt:
        return 0
    finally:
        for process in reversed(processes):
            _stop(process)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", required=True, type=int)
    parser.add_argument("--web-port", required=True, type=int)
    args = parser.parse_args()
    if not all(1 <= port <= 65535 for port in (args.api_port, args.web_port)):
        return 2
    return run(args.api_port, args.web_port)


if __name__ == "__main__":
    raise SystemExit(main())
