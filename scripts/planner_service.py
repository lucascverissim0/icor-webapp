"""Run the local ICOR planner API and web client as a detached local service.

`run_planner_dev.py` keeps both children attached to the terminal it was started
from: it polls in the parent and reaps them in a `finally`, so closing or
clearing that terminal takes the app down with it.

This starts the same two processes detached, with their output going to log
files and their pids recorded, so the app keeps serving after the terminal that
launched it is gone and any later terminal can inspect or stop it.

It is a local development convenience and binds to 127.0.0.1 only. Nothing here
deploys, and it is not a process supervisor: if a child exits on its own it is
not restarted, which `status` reports rather than hides.

    uv run python scripts/planner_service.py start
    uv run python scripts/planner_service.py status
    uv run python scripts/planner_service.py logs --follow
    uv run python scripts/planner_service.py stop
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

if __package__ in (None, ""):  # invoked as `python scripts/planner_service.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_planner_dev import ROOT, WEB_ROOT, _executable, check_prerequisites  # noqa: E402

STATE_PATH = ROOT / ".local" / "planner-service.json"
LOG_DIRECTORY = ROOT / ".local"
#: Set before starting, or the API silently resolves a different snapshot root.
REQUIRED_ENVIRONMENT = ("ICOR_EVIDENCE_ACTIVE_ROOT",)


class ServiceError(RuntimeError):
    """The service cannot be started, inspected or stopped as asked."""


@dataclass(frozen=True, slots=True)
class ServiceState:
    api_pid: int
    web_pid: int
    api_port: int
    web_port: int
    started_at: str
    api_log: str
    web_log: str


def read_state(state_path: Path = STATE_PATH) -> ServiceState | None:
    """The recorded service, or `None` when there is none to read.

    A truncated file means a previous run died mid-write, which is the same
    situation as no service at all and must not wedge the next `start`.
    """

    try:
        document = json.loads(state_path.read_text(encoding="utf-8"))
        return ServiceState(**document)
    except (OSError, ValueError, TypeError):
        return None


def write_state(state_path: Path, state: ServiceState) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(asdict(state), indent=2) + "\n", encoding="utf-8")


def is_running(pid: int) -> bool:
    """Whether a recorded pid is still alive, without signalling it."""

    if pid <= 0:
        return False
    if os.name == "nt":
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            check=False,
            text=True,
        )
        return str(pid) in completed.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    import signal

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # No SO_REUSEADDR: we want the same verdict uvicorn and Vite will get.
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _require_free_ports(api_port: int, web_port: int) -> None:
    """Refuse to start onto a taken port.

    Spawning anyway left the pair half dead: Vite came up, uvicorn exited on
    `[Errno 10048]` inside a log nobody was watching, and the app looked
    started. Another developer's server on the same port is also not ours to
    displace, so this reports and stops rather than reusing or killing it.
    """

    taken = [
        f"{name} port {port}"
        for name, port in (("API", api_port), ("web", web_port))
        if not _port_is_free(port)
    ]
    if taken:
        raise ServiceError(
            f"Already in use: {', '.join(taken)}. Another server is listening "
            "there, so nothing was started. Stop it, or choose free ports with "
            "`--api-port` / `--web-port`."
        )


def _require_environment() -> None:
    missing = [name for name in REQUIRED_ENVIRONMENT if not os.environ.get(name)]
    if missing:
        raise ServiceError(
            "Set these before starting the planner service so it serves the "
            f"snapshot you expect: {', '.join(missing)}"
        )


def _spawn_detached(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
    """Start one child with no handle on this terminal, and return its pid.

    The child must not inherit our stdio: if it did, closing the terminal would
    close the pipes underneath it. Its output goes to a log file instead, and on
    Windows DETACHED_PROCESS also keeps it out of this console's process group
    so a Ctrl+C or a window close does not reach it.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    try:
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
                | subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            )
        process = subprocess.Popen(
            command,
            cwd=ROOT if "uvicorn" in command else WEB_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
            env=environment,
        )
    finally:
        handle.close()
    return process.pid


def start(
    *,
    state_path: Path = STATE_PATH,
    log_directory: Path = LOG_DIRECTORY,
    api_port: int = 8000,
    web_port: int = 5173,
) -> ServiceState:
    """Start both processes detached and record them."""

    existing = read_state(state_path)
    if existing is not None and (is_running(existing.api_pid) or is_running(existing.web_pid)):
        raise ServiceError(
            f"The planner service is already running (api pid {existing.api_pid}, "
            f"web pid {existing.web_pid}). Use `stop`, or `start --restart`."
        )
    check_prerequisites()
    _require_environment()
    if not all(1 <= port <= 65535 for port in (api_port, web_port)):
        raise ServiceError("Planner ports must be between 1 and 65535.")
    _require_free_ports(api_port, web_port)

    environment = os.environ.copy()
    environment["ICOR_API_ORIGIN"] = f"http://127.0.0.1:{api_port}"
    api_log = log_directory / "planner-api.log"
    web_log = log_directory / "planner-web.log"
    api_pid = _spawn_detached(
        [
            _executable("uv"),
            "run",
            "uvicorn",
            "icor.api.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(api_port),
        ],
        api_log,
        environment,
    )
    web_pid = _spawn_detached(
        [
            _executable("npm"),
            "run",
            "dev",
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            str(web_port),
        ],
        web_log,
        environment,
    )
    state = ServiceState(
        api_pid=api_pid,
        web_pid=web_pid,
        api_port=api_port,
        web_port=web_port,
        started_at=datetime.now(UTC).isoformat(),
        api_log=str(api_log),
        web_log=str(web_log),
    )
    write_state(state_path, state)
    return state


def stop(*, state_path: Path = STATE_PATH) -> bool:
    """Stop both processes and clear the record. False when none was running."""

    state = read_state(state_path)
    if state is None:
        return False
    for pid in (state.web_pid, state.api_pid):
        if is_running(pid):
            _terminate(pid)
    state_path.unlink(missing_ok=True)
    return True


def _responds(url: str) -> bool:
    """Whether something is serving HTTP here.

    An error status still proves the server is up and answering, so only a
    transport failure counts as not responding. Treating an `HTTPError` as
    silence reported a healthy API as dead when the probe path was wrong.
    """

    try:
        with urlopen(url, timeout=2) as response:  # noqa: S310 - fixed localhost URL
            return 200 <= response.status < 600
    except HTTPError:
        return True
    except (URLError, OSError, ValueError):
        return False


def status(*, state_path: Path = STATE_PATH, probe: bool = False) -> dict[str, object]:
    """Report each process separately, so a half-dead pair is visible."""

    state = read_state(state_path)
    if state is None:
        return {"running": False, "healthy": False}
    api_alive = is_running(state.api_pid)
    web_alive = is_running(state.web_pid)
    report: dict[str, object] = {
        "running": api_alive or web_alive,
        "healthy": api_alive and web_alive,
        "started_at": state.started_at,
        "api": {
            "pid": state.api_pid,
            "running": api_alive,
            "url": f"http://127.0.0.1:{state.api_port}",
            "log": state.api_log,
        },
        "web": {
            "pid": state.web_pid,
            "running": web_alive,
            "url": f"http://127.0.0.1:{state.web_port}",
            "log": state.web_log,
        },
    }
    if probe:
        report["api_responding"] = _responds(
            f"http://127.0.0.1:{state.api_port}/api/health"
        )
        report["web_responding"] = _responds(f"http://127.0.0.1:{state.web_port}/")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=False)
    start_parser = commands.add_parser("start", help="start both processes detached")
    start_parser.add_argument("--api-port", default=8000, type=int)
    start_parser.add_argument("--web-port", default=5173, type=int)
    start_parser.add_argument(
        "--restart", action="store_true", help="stop a running service first"
    )
    commands.add_parser("stop", help="stop the recorded processes")
    status_parser = commands.add_parser("status", help="report what is running")
    status_parser.add_argument(
        "--probe", action="store_true", help="also request the API and web ports"
    )
    logs_parser = commands.add_parser("logs", help="print the recorded log paths")
    logs_parser.add_argument("--lines", default=40, type=int)
    parser.set_defaults(command="status")
    return parser


def _print_status(report: dict[str, object]) -> None:
    if not report["running"]:
        print("The planner service is not running.")
        return
    for name in ("api", "web"):
        detail = report[name]
        assert isinstance(detail, dict)
        mark = "running" if detail["running"] else "STOPPED"
        print(f"{name:>4}  {mark:<8} pid {detail['pid']:<8} {detail['url']}")
        if f"{name}_responding" in report:
            print(f"      responding: {report[f'{name}_responding']}")
    print(f"started at {report['started_at']}")
    if not report["healthy"]:
        print("One process is gone. Run `stop` then `start` to bring the pair back.")


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "start":
            if arguments.restart:
                stop()
            state = start(api_port=arguments.api_port, web_port=arguments.web_port)
            print(
                "Planner service started and detached from this terminal.\n"
                f"  web  http://127.0.0.1:{state.web_port}  (pid {state.web_pid})\n"
                f"  api  http://127.0.0.1:{state.api_port}  (pid {state.api_pid})\n"
                f"  logs {state.web_log}\n"
                f"       {state.api_log}\n"
                "Vite needs a few seconds for its first build. It keeps running "
                "when this terminal is closed; stop it with "
                "`uv run python scripts/planner_service.py stop`."
            )
            return 0
        if arguments.command == "stop":
            print("Planner service stopped." if stop() else "No planner service was running.")
            return 0
        if arguments.command == "logs":
            state = read_state()
            if state is None:
                print("No planner service was running.")
                return 0
            for path in (Path(state.api_log), Path(state.web_log)):
                print(f"=== {path} ===")
                if path.exists():
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                    print("\n".join(lines[-arguments.lines :]))
            return 0
        _print_status(status(probe=getattr(arguments, "probe", False)))
        return 0
    except (ServiceError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
