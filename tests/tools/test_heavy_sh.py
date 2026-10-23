"""tools/heavy.sh: one heavy job at a time on the whole Mac, behind the lock Friday shares."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_HEAVY = Path(__file__).resolve().parents[2] / "tools" / "heavy.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="heavy.sh is a POSIX shell script")


def _env(lock: Path) -> dict[str, str]:
    # Never the real Mac-wide lock, and never a wait on the real load or heat.
    return {
        **os.environ,
        "HEAVY_TEST_LOCK": str(lock),
        "HEAVY_TEST_LOAD_WAIT_MAX": "0",
        "HEAVY_TEST_THERMAL": "0",
    }


def _heavy(lock: Path, *command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_HEAVY), *command],
        env=_env(lock),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_the_job_exit_code_is_returned_and_the_lock_released(tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"

    assert _heavy(lock, "true").returncode == 0
    assert not lock.exists()
    ran = _heavy(lock, "sh", "-c", "exit 7")
    assert ran.returncode == 7
    assert "heavy: done rc=7" in ran.stderr
    assert not lock.exists()


def test_the_lock_names_the_running_command_and_its_holder(tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"

    ran = _heavy(lock, "sh", "-c", f'cat "{lock}/cmd"; echo; cat "{lock}/pid"')

    assert ran.returncode == 0
    command, pid = ran.stdout.splitlines()
    assert command.strip() == f'sh -c cat "{lock}/cmd"; echo; cat "{lock}/pid"'
    assert int(pid) > 1


def test_two_jobs_never_run_at_the_same_time(tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"
    log = tmp_path / "spans.log"
    job = f'echo start >> "{log}"; sleep 2; echo end >> "{log}"'
    # A busy wait polls every few seconds, so the second job waits at least that long.
    runs = [
        subprocess.Popen(
            ["bash", str(_HEAVY), "sh", "-c", job],
            env=_env(lock),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    for run in runs:
        assert run.wait(timeout=60) == 0

    events = [line.split()[0] for line in log.read_text().splitlines()]
    assert events == ["start", "end", "start", "end"]
    assert any("waiting" in (run.stderr.read() if run.stderr else "") for run in runs)
    assert not lock.exists()


def test_a_lock_left_by_a_dead_holder_is_reclaimed(tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"
    dead = subprocess.run(["sh", "-c", "echo $$"], capture_output=True, text=True, check=True)
    lock.mkdir()
    (lock / "pid").write_text(dead.stdout)
    (lock / "cmd").write_text("a job killed long ago")

    started = time.monotonic()
    ran = _heavy(lock, "true")

    assert ran.returncode == 0
    assert time.monotonic() - started < 30
    assert not lock.exists()


def test_without_a_command_it_prints_usage(tmp_path: Path) -> None:
    ran = _heavy(tmp_path / "heavy.lock")

    assert ran.returncode == 2
    assert "USAGE" in ran.stdout
