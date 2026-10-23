from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest


def test_browser_fixture_builds_a_reusable_sealed_candidate(tmp_path: Path) -> None:
    from icor.application.evidence_review import EvidenceReviewService
    from icor.application.registrations import RegistrationQuery, RegistrationService
    from scripts.e2e_fixture import prepare_e2e_fixture

    first = prepare_e2e_fixture(tmp_path)
    second = prepare_e2e_fixture(tmp_path)

    assert first == second
    evidence = EvidenceReviewService.from_candidate(first)
    registrations = RegistrationService.from_candidate(first)
    assert evidence.summary().observation_count == 3
    summary = registrations.summary()
    assert summary.model_count == 3
    assert summary.total_registrations == Decimal("600")
    tesla = registrations.ranking(RegistrationQuery(search="Tesla"))
    assert [(row.make, row.model, row.registrations) for row in tesla.items] == [
        ("TESLA", "MODEL Y", Decimal("200"))
    ]


def test_browser_runner_prepares_both_candidates_or_rejects_partial_configuration(
    tmp_path: Path,
) -> None:
    from scripts.run_e2e_dev import fixture_root_for, prepare_environment

    prepared = prepare_environment({}, fixture_root=tmp_path)

    evidence = Path(prepared["ICOR_E2E_EVIDENCE_CANDIDATE"])
    generation = Path(prepared["ICOR_E2E_GENERATION_CANDIDATE"])
    assert evidence == generation
    assert (evidence / "snapshot.json").is_file()
    for partial in (
        {"ICOR_E2E_EVIDENCE_CANDIDATE": str(evidence)},
        {"ICOR_E2E_GENERATION_CANDIDATE": str(generation)},
    ):
        with pytest.raises(ValueError, match="both be configured"):
            prepare_environment(partial, fixture_root=tmp_path)

    explicit = {
        "ICOR_E2E_EVIDENCE_CANDIDATE": "evidence-candidate",
        "ICOR_E2E_GENERATION_CANDIDATE": "generation-candidate",
    }
    assert prepare_environment(explicit, fixture_root=tmp_path) == explicit
    empty = prepare_environment(
        {
            "ICOR_E2E_EVIDENCE_CANDIDATE": "",
            "ICOR_E2E_GENERATION_CANDIDATE": "",
        },
        fixture_root=tmp_path,
    )
    assert Path(empty["ICOR_E2E_EVIDENCE_CANDIDATE"]) == evidence
    assert fixture_root_for(18001, 19001) != fixture_root_for(18002, 19002)


def test_browser_runner_never_shares_its_own_stdio_with_the_children(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The property that actually prevents the CI teardown hang.

    Playwright kills the webServer process group, which does not reach children
    that started their own session. If those children hold the write end of the
    pipe Playwright is reading, its `close` event never fires and teardown hangs
    forever with every test already green.
    """

    import subprocess

    from scripts import run_e2e_dev

    recorded: list[dict[str, object]] = []

    class _FakeProcess:
        pid = 4321

        def __init__(self) -> None:
            self.stdout = None

        def poll(self) -> int | None:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            return None

    def fake_popen(command, **kwargs):  # type: ignore[no-untyped-def]
        recorded.append(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(run_e2e_dev.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(run_e2e_dev.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        run_e2e_dev, "prepare_environment", lambda environment, **_: dict(environment)
    )
    monkeypatch.setattr(run_e2e_dev, "_install_termination_handlers", lambda: None)
    monkeypatch.setattr(run_e2e_dev, "_pump", lambda process, label: None)
    monkeypatch.setattr(run_e2e_dev, "_guard", lambda process: None)
    # The fake pid is not ours; never signal whatever group really has it.
    monkeypatch.setattr(run_e2e_dev.os, "killpg", lambda pid, sig: None)
    # Nor change this test process's own signal handling.
    monkeypatch.setattr(run_e2e_dev, "_ignore_termination", lambda: None)

    run_e2e_dev.run(api_port=18999, web_port=19999)

    assert len(recorded) == 2
    for kwargs in recorded:
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.STDOUT


def test_browser_runner_escalates_when_a_child_ignores_the_first_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that will not stop must not be left running after teardown."""

    import subprocess

    from scripts import run_e2e_dev

    # The fake pid is not ours; never signal whatever group really has it.
    monkeypatch.setattr(run_e2e_dev.os, "killpg", lambda pid, sig: None)

    class _StubbornProcess:
        pid = 777

        def __init__(self) -> None:
            self.killed = False
            self._waits = 0

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            self._waits += 1
            if self._waits == 1:
                raise subprocess.TimeoutExpired(cmd="child", timeout=timeout or 0)
            return 0

        def kill(self) -> None:
            self.killed = True

    process = _StubbornProcess()
    run_e2e_dev._stop(process)  # type: ignore[arg-type]

    assert process.killed or process._waits >= 2


@pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX process groups")
def test_browser_runner_children_die_when_the_runner_is_killed_outright(
    tmp_path: Path,
) -> None:
    """A SIGKILLed runner never reaches its `finally`; its children must still stop.

    Playwright escalates to SIGKILL when teardown outlasts its grace period, and
    an interrupted test command kills the runner the same way. Before the reaper
    every such run left a browser-test API listening for days.
    """

    import os
    import signal
    import subprocess
    import sys
    import time

    root = Path(__file__).resolve().parents[1]
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import subprocess, sys, time\n"
        "from scripts import run_e2e_dev\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
        " start_new_session=True, stdout=subprocess.DEVNULL)\n"
        "run_e2e_dev._guard(child)\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    supervisor = subprocess.Popen(
        [sys.executable, str(runner)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        stdout=subprocess.PIPE,
        text=True,
    )
    assert supervisor.stdout is not None
    child = int(supervisor.stdout.readline())
    try:
        supervisor.send_signal(signal.SIGKILL)
        supervisor.wait(timeout=5)
        deadline = time.monotonic() + 10
        alive = True
        while alive and time.monotonic() < deadline:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                alive = False
            else:
                time.sleep(0.2)
        assert not alive, "the child outlived its killed runner"
    finally:
        with __import__("contextlib").suppress(ProcessLookupError):
            os.killpg(child, signal.SIGKILL)


def test_browser_runner_teardown_is_not_cut_short_by_a_second_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Playwright can signal twice; a second interrupt inside `finally` skipped the reaping."""

    import signal

    from scripts import run_e2e_dev

    events: list[str] = []

    class _FakeProcess:
        pid = 4321
        stdout = None

        def poll(self) -> int | None:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            events.append("wait")
            return 0

        def kill(self) -> None:
            return None

    handlers: dict[int, object] = {}
    monkeypatch.setattr(run_e2e_dev.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(run_e2e_dev.subprocess, "Popen", lambda command, **_: _FakeProcess())
    monkeypatch.setattr(
        run_e2e_dev, "prepare_environment", lambda environment, **_: dict(environment)
    )
    monkeypatch.setattr(run_e2e_dev, "_install_termination_handlers", lambda: None)
    monkeypatch.setattr(run_e2e_dev, "_pump", lambda process, label: None)
    monkeypatch.setattr(run_e2e_dev, "_guard", lambda process: None)
    monkeypatch.setattr(
        run_e2e_dev.signal, "signal", lambda signum, handler: handlers.__setitem__(signum, handler)
    )

    def killpg(pid: int, sig: int) -> None:
        events.append("killpg")
        # A second signal now would hit whatever handler is installed.
        assert handlers.get(signal.SIGTERM) is signal.SIG_IGN
        assert handlers.get(signal.SIGINT) is signal.SIG_IGN

    monkeypatch.setattr(run_e2e_dev.os, "killpg", killpg)

    run_e2e_dev.run(api_port=18999, web_port=19999)

    assert events[:2] == ["killpg", "killpg"]
    assert "wait" in events


def test_browser_runner_interrupts_once_however_many_signals_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import signal

    from scripts import run_e2e_dev

    handlers: dict[int, object] = {}
    monkeypatch.setattr(
        run_e2e_dev.signal, "signal", lambda signum, handler: handlers.__setitem__(signum, handler)
    )
    run_e2e_dev._install_termination_handlers()
    handler = handlers[signal.SIGTERM]

    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)  # type: ignore[operator]
    handler(signal.SIGINT, None)  # type: ignore[operator]
    handlers[signal.SIGINT](signal.SIGINT, None)  # type: ignore[operator]
