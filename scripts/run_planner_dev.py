"""Run the local-only ICOR planner API and web client together."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"


def _executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Required executable is not available: {name}")
    return executable


def check_prerequisites() -> None:
    _executable("uv")
    _executable("npm")
    required = (ROOT / "pyproject.toml", WEB_ROOT / "package.json", WEB_ROOT / "node_modules")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Planner prerequisites are missing: {', '.join(missing)}")


def _stop(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        if process.poll() is not None:
            return
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=4)


def run(api_port: int = 8000, web_port: int = 5173) -> int:
    check_prerequisites()
    uv = _executable("uv")
    npm = _executable("npm")
    web_environment = os.environ.copy()
    web_environment["ICOR_API_ORIGIN"] = f"http://127.0.0.1:{api_port}"
    processes: list[subprocess.Popen[bytes]] = []
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    try:
        processes.append(
            subprocess.Popen(
                [
                    uv,
                    "run",
                    "uvicorn",
                    "icor.api.app:create_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(api_port),
                ],
                cwd=ROOT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
        )
        processes.append(
            subprocess.Popen(
                [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(web_port)],
                cwd=WEB_ROOT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
                env=web_environment,
            )
        )
        while not stopping:
            for process in processes:
                code = process.poll()
                if code is not None:
                    return code or 1
            time.sleep(0.2)
        return 0
    except OSError as error:
        raise RuntimeError(f"Could not start the planner processes: {error}") from error
    finally:
        for process in reversed(processes):
            _stop(process)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="validate prerequisites without starting servers"
    )
    parser.add_argument("--api-port", default=8000, type=int)
    parser.add_argument("--web-port", default=5173, type=int)
    args = parser.parse_args()
    try:
        check_prerequisites()
        if args.check:
            print("Planner prerequisites are ready.")
            return 0
        if not all(1 <= port <= 65535 for port in (args.api_port, args.web_port)):
            raise RuntimeError("Planner ports must be between 1 and 65535.")
        return run(api_port=args.api_port, web_port=args.web_port)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
