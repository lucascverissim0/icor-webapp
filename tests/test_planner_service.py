"""The detached local planner service.

`run_planner_dev.py` polls in the parent and kills both children in a `finally`,
so the app it starts dies with the terminal that launched it. This service
starts the same two processes detached, records them, and can report on or stop
them from any later terminal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_state_round_trips_through_the_recorded_file(tmp_path: Path) -> None:
    from scripts.planner_service import ServiceState, read_state, write_state

    state_path = tmp_path / "planner-service.json"
    state = ServiceState(
        api_pid=111,
        web_pid=222,
        api_port=8000,
        web_port=5173,
        started_at="2026-09-22T18:00:00+00:00",
        api_log=str(tmp_path / "api.log"),
        web_log=str(tmp_path / "web.log"),
    )

    write_state(state_path, state)

    assert read_state(state_path) == state
    assert json.loads(state_path.read_text(encoding="utf-8"))["api_pid"] == 111


def test_a_missing_or_corrupt_state_file_reads_as_no_service(tmp_path: Path) -> None:
    """A half-written file must not wedge `start`; it means nothing is running."""

    from scripts.planner_service import read_state

    assert read_state(tmp_path / "absent.json") is None
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert read_state(corrupt) is None


def test_start_refuses_while_a_recorded_process_is_still_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Starting twice would orphan the first pair with nothing recording it."""

    from scripts import planner_service

    state_path = tmp_path / "planner-service.json"
    planner_service.write_state(
        state_path,
        planner_service.ServiceState(1, 2, 8000, 5173, "now", "api.log", "web.log"),
    )
    monkeypatch.setattr(planner_service, "is_running", lambda pid: True)

    with pytest.raises(planner_service.ServiceError, match="already running"):
        planner_service.start(state_path=state_path, log_directory=tmp_path)


def test_start_replaces_state_left_behind_by_dead_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine that was rebooted leaves a stale file; it must not block start."""

    from scripts import planner_service

    state_path = tmp_path / "planner-service.json"
    planner_service.write_state(
        state_path,
        planner_service.ServiceState(1, 2, 8000, 5173, "now", "api.log", "web.log"),
    )
    monkeypatch.setattr(planner_service, "is_running", lambda pid: False)
    monkeypatch.setattr(planner_service, "check_prerequisites", lambda: None)
    monkeypatch.setattr(planner_service, "_require_environment", lambda: None)
    monkeypatch.setattr(planner_service, "_require_free_ports", lambda *ports: None)
    spawned: list[list[str]] = []

    def fake_spawn(command, log_path, environment):  # type: ignore[no-untyped-def]
        spawned.append(list(command))
        return 4000 + len(spawned)

    monkeypatch.setattr(planner_service, "_spawn_detached", fake_spawn)

    state = planner_service.start(state_path=state_path, log_directory=tmp_path)

    assert (state.api_pid, state.web_pid) == (4001, 4002)
    assert planner_service.read_state(state_path) == state
    assert len(spawned) == 2
    assert any("uvicorn" in part for part in spawned[0])
    # Vite runs through node, not npm: on Windows npm is a batch file, and the
    # cmd.exe wrapping it takes console control events that detachment should
    # have prevented, killing the dev server with a prompt nobody can answer.
    assert "node" in spawned[1][0]
    assert any(part.endswith("vite.js") for part in spawned[1])
    assert not any("npm" in part for part in spawned[1])


def test_start_names_the_missing_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API silently serves the wrong snapshot root when this is unset."""

    from scripts import planner_service

    monkeypatch.delenv("ICOR_EVIDENCE_ACTIVE_ROOT", raising=False)
    monkeypatch.setattr(planner_service, "check_prerequisites", lambda: None)

    with pytest.raises(planner_service.ServiceError, match="ICOR_EVIDENCE_ACTIVE_ROOT"):
        planner_service.start(
            state_path=tmp_path / "planner-service.json", log_directory=tmp_path
        )


def test_start_refuses_a_snapshot_root_that_does_not_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Set but wrong is worse than unset.

    The API starts, fails to open the active snapshot and answers 503 for every
    vehicle, while the pair looks healthy. A Git Bash `/c/Users/...` path on
    Windows is the easy way to land there.
    """

    from scripts import planner_service

    monkeypatch.setattr(planner_service, "check_prerequisites", lambda: None)
    monkeypatch.setenv("ICOR_EVIDENCE_ACTIVE_ROOT", "/c/Users/someone/evidence")

    with pytest.raises(planner_service.ServiceError, match="not a directory"):
        planner_service.start(
            state_path=tmp_path / "planner-service.json", log_directory=tmp_path
        )


def test_start_refuses_a_snapshot_root_with_nothing_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import planner_service

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    monkeypatch.setattr(planner_service, "check_prerequisites", lambda: None)
    monkeypatch.setenv("ICOR_EVIDENCE_ACTIVE_ROOT", str(evidence))

    with pytest.raises(planner_service.ServiceError, match="No active.json"):
        planner_service.start(
            state_path=tmp_path / "planner-service.json", log_directory=tmp_path
        )


def test_status_reports_each_process_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-dead pair must be visible rather than reported as healthy."""

    from scripts import planner_service

    state_path = tmp_path / "planner-service.json"
    planner_service.write_state(
        state_path,
        planner_service.ServiceState(11, 22, 8000, 5173, "now", "api.log", "web.log"),
    )
    monkeypatch.setattr(planner_service, "is_running", lambda pid: pid == 11)

    report = planner_service.status(state_path=state_path)

    assert report["api"]["running"] is True
    assert report["web"]["running"] is False
    assert report["healthy"] is False


def test_status_without_a_service_is_not_an_error(tmp_path: Path) -> None:
    from scripts import planner_service

    report = planner_service.status(state_path=tmp_path / "absent.json")

    assert report["running"] is False
    assert report["healthy"] is False


def test_stop_clears_the_state_file_after_stopping_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import planner_service

    state_path = tmp_path / "planner-service.json"
    planner_service.write_state(
        state_path,
        planner_service.ServiceState(11, 22, 8000, 5173, "now", "api.log", "web.log"),
    )
    stopped: list[int] = []
    monkeypatch.setattr(planner_service, "is_running", lambda pid: True)
    monkeypatch.setattr(planner_service, "_terminate", lambda pid: stopped.append(pid))

    assert planner_service.stop(state_path=state_path) is True
    assert sorted(stopped) == [11, 22]
    assert not state_path.exists()


def test_stop_without_a_service_reports_nothing_to_do(tmp_path: Path) -> None:
    from scripts import planner_service

    assert planner_service.stop(state_path=tmp_path / "absent.json") is False


def test_ports_are_validated_before_anything_is_spawned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import planner_service

    monkeypatch.setattr(planner_service, "check_prerequisites", lambda: None)
    monkeypatch.setattr(planner_service, "_require_environment", lambda: None)
    monkeypatch.setattr(planner_service, "_require_free_ports", lambda *ports: None)
    monkeypatch.setattr(
        planner_service,
        "_spawn_detached",
        lambda *args: pytest.fail("a bad port must not reach a spawn"),
    )

    with pytest.raises(planner_service.ServiceError, match="port"):
        planner_service.start(
            state_path=tmp_path / "planner-service.json",
            log_directory=tmp_path,
            api_port=0,
        )


def test_the_cli_exposes_the_four_verbs() -> None:
    from scripts.planner_service import build_parser

    parser = build_parser()

    for verb in ("start", "stop", "status", "logs"):
        assert parser.parse_args([verb]).command == verb


def test_start_refuses_a_port_another_server_already_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spawning onto a taken port left Vite up and uvicorn dead in a log.

    The pair then looked started. Another developer's server is also not ours
    to displace, so this reports and stops instead of reusing or killing it.
    """

    from scripts import planner_service

    monkeypatch.setattr(planner_service, "check_prerequisites", lambda: None)
    monkeypatch.setattr(planner_service, "_require_environment", lambda: None)
    monkeypatch.setattr(planner_service, "_port_is_free", lambda port: port != 8000)
    monkeypatch.setattr(
        planner_service,
        "_spawn_detached",
        lambda *args: pytest.fail("a taken port must not reach a spawn"),
    )

    with pytest.raises(planner_service.ServiceError, match="API port 8000"):
        planner_service.start(
            state_path=tmp_path / "planner-service.json", log_directory=tmp_path
        )
