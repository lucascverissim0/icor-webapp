"""The PreToolUse hook that keeps heavy jobs behind tools/heavy.sh (MAC-COOL)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.hooks import mac_load_guard as guard

_HOOK = Path(__file__).resolve().parents[2] / "tools" / "hooks" / "mac_load_guard.py"
_CWD = "/Users/someone/ICOR"

HEAVY = [
    # full pytest, every spelling
    "uv run pytest",
    "uv run pytest -q",
    "uv run pytest -q tests",
    "uv run pytest tests/ -x",
    "uv run --group dev pytest -q",
    "uv run python -m pytest",
    ".venv/bin/pytest -q",
    ".venv/bin/python -m pytest .",
    "pytest -m 'not slow'",
    "timeout 900 uv run pytest -q 2>&1 | tail -5",
    "ICOR_EVIDENCE_ACTIVE_ROOT=/x uv run pytest",
    # web
    "npm test",
    "npm run test",
    "npm --prefix web test",
    "npm --prefix web run test -- --run",
    "cd web && npm test",
    "(cd web && npx vitest run)",
    "npx vitest",
    "npm run build",
    "npm --prefix web run build",
    "npx vite build",
    "npm run e2e",
    "npm --prefix web run e2e -- e2e/planner.spec.ts",
    "npx playwright test",
    "npx playwright test e2e/planner.spec.ts --workers 1",
    # long data jobs
    "uv run python scripts/build_evidence_snapshot.py build --root .local/evidence",
    "uv run python scripts/build_evidence_snapshot.py promote --snapshot s",
    "uv run scripts/rebuild_active_snapshot.py",
    ".venv/bin/python scripts/backtest_code_ranges.py",
    "uv run --project .. python ../scripts/benchmark_registration_forecasts.py",
    "python3 scripts/calibrate_age_anchor.py",
    "uv run python scripts/measure_register_gates.py",
    "uv run python scripts/verify_client_release.py --local",
    "uv run python scripts/compare_snapshot_cohorts.py",
    "uv run python scripts/report_snapshot_completeness.py",
    "uv run python scripts/build_client_snapshot.py",
    # shapes Friday's parser missed
    "npm test &",
    "uv run pytest & echo started",
    "bash -c 'uv run pytest -q'",
    'zsh -c "cd web && npm run build"',
    "nohup uv run python scripts/build_local_generations_db.py > out.log 2>&1 &",
    "caffeinate -i nice -n 5 uv run pytest",
]

LIGHT = [
    "uv run pytest tests/tools/test_mac_load_guard.py -q",
    "uv run pytest tests/forecasting",
    "uv run pytest -k range -q",
    "uv run pytest --lf",
    "uv run python -m pytest tests/test_audit.py::test_x",
    "npx vitest run src/lib/format.test.ts",
    "npm test -- src/lib/format.test.ts",
    "npm --prefix web test -- tests/planner.test.tsx",
    "npm run lint",
    "npm --prefix web run typecheck",
    "npx tsc --noEmit",
    "npm ci",
    "uv sync --all-groups",
    "uv run ruff check src tests",
    "uv run python scripts/acquire_cepik_fleet.py crawl --workers 2",
    "uv run python scripts/planner_service.py start",
    "uv run python scripts/build_evidence_snapshot.py --help",
    "git status && git diff --stat",
    "echo 'npm test'",
    "grep -rn pytest docs",
    "cat tests/conftest.py",
    "",
    # wrapped heavy jobs
    "tools/heavy.sh uv run pytest -q",
    "./tools/heavy.sh npm --prefix web run e2e",
    "bash tools/heavy.sh uv run python scripts/build_evidence_snapshot.py build",
    "cd ~/ICOR && tools/heavy.sh npx playwright test",
    # worktrees in places Spotlight never indexes
    "git worktree add .claude/worktrees/fix-x -b fix/x",
    "git worktree add /tmp/icor-check HEAD",
    "git worktree add ~/ICOR-wt.noindex/fix-y",
    "git worktree list",
]


@pytest.mark.parametrize("command", HEAVY)
def test_a_heavy_job_outside_heavy_sh_is_refused(command: str) -> None:
    message = guard.check(command, _CWD)

    assert message is not None
    assert "tools/heavy.sh" in message


@pytest.mark.parametrize("command", LIGHT)
def test_targeted_light_and_wrapped_commands_pass(command: str) -> None:
    assert guard.check(command, _CWD) is None


@pytest.mark.parametrize(
    "command",
    [
        "git worktree add ../icor-fix -b fix/x",
        "git worktree add ~/icor-fix",
        "git -C ~/ICOR worktree add worktrees/fix",
    ],
)
def test_a_worktree_spotlight_would_index_is_refused(command: str) -> None:
    message = guard.check(command, _CWD)

    assert message is not None
    assert ".noindex" in message


def _run_hook(stdin: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_HOOK)], input=stdin, capture_output=True, text=True, check=False
    )


def test_the_hook_blocks_with_exit_2_and_the_reason_on_stderr() -> None:
    payload = {"tool_name": "Bash", "tool_input": {"command": "uv run pytest"}, "cwd": _CWD}

    ran = _run_hook(json.dumps(payload))

    assert ran.returncode == 2
    assert "tools/heavy.sh" in ran.stderr


def test_the_hook_allows_a_wrapped_job() -> None:
    payload = {"tool_name": "Bash", "tool_input": {"command": "tools/heavy.sh uv run pytest"}}

    assert _run_hook(json.dumps(payload)).returncode == 0


def test_the_hook_ignores_other_tools() -> None:
    payload = {"tool_name": "Write", "tool_input": {"command": "uv run pytest"}}

    assert _run_hook(json.dumps(payload)).returncode == 0


def test_the_hook_fails_open_on_its_own_errors() -> None:
    assert _run_hook("not json").returncode == 0
    assert _run_hook(json.dumps({"tool_name": "Bash", "tool_input": None})).returncode == 0


def test_mentioning_heavy_sh_does_not_exempt_another_heavy_job() -> None:
    assert guard.check("echo tools/heavy.sh; uv run pytest", _CWD) is not None
    assert guard.check("tools/heavy.sh true && npm run build", _CWD) is not None
