"""Run the explicit browser-test API and Vite client together."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
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


def _pump(process: subprocess.Popen[bytes], label: str) -> threading.Thread:
    """Relay a child stream without letting the child hold our own stdout.

    Playwright terminates its webServer by killing that process group, which
    does not reach these children because they start their own session. When
    they inherited our stdout they kept the write end of Playwright pipe open
    after we exited, so the Node `close` event never fired and teardown hung
    with every test already green. Reading the output ourselves means the only
    holder of that pipe is this process.
    """

    def relay() -> None:
        stream = process.stdout
        if stream is None:
            return
        with suppress(ValueError, OSError):
            for line in iter(stream.readline, b""):
                sys.stdout.write(f"[{label}] {line.decode(errors='replace')}")
                sys.stdout.flush()

    thread = threading.Thread(target=relay, name=f"relay-{label}", daemon=True)
    thread.start()
    return thread


def _stop(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    else:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)


def _install_termination_handlers() -> None:
    """Make a signal reach the `finally` that reaps the children.

    Without this the default SIGTERM disposition ends the process outright and
    the children are orphaned.
    """

    def raise_interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, raise_interrupt)
    signal.signal(signal.SIGTERM, raise_interrupt)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, raise_interrupt)


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
    _install_termination_handlers()
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
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        )
        _pump(processes[-1], "api")
        processes.append(
            subprocess.Popen(
                [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(web_port)],
                cwd=ROOT / "web",
                env=web_environment,
                creationflags=flags,
                start_new_session=os.name != "nt",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        )
        _pump(processes[-1], "web")
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
