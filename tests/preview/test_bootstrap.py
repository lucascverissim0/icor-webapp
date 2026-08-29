from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from icor.preview.bootstrap import (
    BUILD_AS_OF,
    DETERMINISTIC_SEED,
    BootstrapCoordinator,
    BootstrapError,
    CommandResult,
    EnvironmentReport,
    default_plan,
    probe_environment,
    validate_environment,
)

EXPECTED_SOURCE_KEYS = (
    *(f"eea-{year}-final" for year in range(2010, 2025)),
    "kba-fz10-2024",
    "uk-veh0160-gb",
    "uk-veh0120-gb",
    "uk-veh0124-am",
    "uk-veh0124-nz",
)


def test_default_plan_has_exactly_the_approved_twenty_releases() -> None:
    plan = default_plan()

    assert tuple(source.source_key for source in plan.sources) == EXPECTED_SOURCE_KEYS
    assert len(plan.release_ids) == 20
    assert len(set(plan.release_ids)) == 20
    assert plan.release_ids[0] == "eea-co2cars-2010-final-v2"
    assert plan.release_ids[-1] == "uk-dft-veh0160-gb-2025-final-20260713"
    assert plan.build_as_of == BUILD_AS_OF == "2026-08-27T12:00:00+00:00"
    assert plan.deterministic_seed == DETERMINISTIC_SEED == 20260827


def _environment(**changes) -> EnvironmentReport:
    values = {
        "codespaces": True,
        "python_version": (3, 12, 13),
        "node_version": "24.15.0",
        "npm_version": "11.8.0",
        "uv_version": "0.11.3",
        "free_bytes": 24 * 1024**3,
    }
    values.update(changes)
    return EnvironmentReport(**values)


def test_probe_environment_normalizes_uv_platform_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import icor.preview.bootstrap as bootstrap

    versions = {
        ("node", "--version"): "v24.15.0",
        ("npm", "--version"): "11.12.1",
        ("uv", "--version"): "uv 0.11.3 (x86_64-unknown-linux-gnu)",
    }
    monkeypatch.setenv("CODESPACES", "true")
    monkeypatch.setattr(bootstrap, "_tool_version", versions.__getitem__)
    monkeypatch.setattr(
        bootstrap.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=24 * 1024**3),
    )

    report = probe_environment(tmp_path)

    assert report.uv_version == "0.11.3"

def test_environment_accepts_supported_codespace_and_workspaces_root(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "workspaces" / "icor"
    repository.mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]", encoding="utf-8")
    (repository / "uv.lock").write_text("version = 1", encoding="utf-8")
    (repository / "web").mkdir()
    (repository / "web" / "package-lock.json").write_text("{}", encoding="utf-8")
    evidence = tmp_path / "workspaces" / ".icor" / "evidence"

    validate_environment(
        _environment(),
        repository_root=repository,
        workspaces_root=tmp_path / "workspaces",
        evidence_root=evidence,
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"codespaces": False},
        {"python_version": (3, 11, 9)},
        {"node_version": "22.21.0"},
        {"node_version": "23.9.0"},
        {"npm_version": ""},
        {"uv_version": "0.11.2"},
        {"free_bytes": 4 * 1024**3},
    ),
)
def test_environment_rejects_unsupported_or_underprovisioned_runtime(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    repository = tmp_path / "workspaces" / "icor"
    repository.mkdir(parents=True)
    for relative in ("pyproject.toml", "uv.lock", "web/package-lock.json"):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("locked", encoding="utf-8")

    with pytest.raises(BootstrapError):
        validate_environment(
            _environment(**changes),
            repository_root=repository,
            workspaces_root=tmp_path / "workspaces",
            evidence_root=tmp_path / "workspaces" / ".icor" / "evidence",
        )


def test_environment_rejects_evidence_outside_workspaces(tmp_path: Path) -> None:
    repository = tmp_path / "workspaces" / "icor"
    repository.mkdir(parents=True)
    for relative in ("pyproject.toml", "uv.lock", "web/package-lock.json"):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("locked", encoding="utf-8")

    with pytest.raises(BootstrapError, match="evidence root"):
        validate_environment(
            _environment(),
            repository_root=repository,
            workspaces_root=tmp_path / "workspaces",
            evidence_root=tmp_path / "outside" / "evidence",
        )


class RecordingRunner:
    def __init__(self, *, completeness_code: int = 0) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.completeness_code = completeness_code

    def run(self, command: tuple[str, ...], *, cwd: Path) -> CommandResult:
        del cwd
        self.commands.append(command)
        joined = " ".join(command)
        if " build " in f" {joined} ":
            output = json.dumps(
                {"snapshot_id": "snapshot-approved", "state": "candidate"}
            )
            return CommandResult(0, output, "")
        if "report_snapshot_completeness.py" in joined:
            payload = {
                "snapshot_id": "snapshot-approved",
                "release_ids": list(default_plan().release_ids),
                "warnings": [],
            }
            return CommandResult(self.completeness_code, json.dumps(payload), "")
        return CommandResult(0, "{}", "")


def test_acquisition_reuses_verified_releases_and_downloads_only_absent(
    tmp_path: Path,
) -> None:
    runner = RecordingRunner()
    plan = default_plan()
    absent = {plan.sources[0].release_id, plan.sources[-1].release_id}
    verified: list[str] = []
    coordinator = BootstrapCoordinator(
        repository_root=tmp_path,
        evidence_root=tmp_path / "evidence",
        runner=runner,
        release_is_valid=lambda release_id: verified.append(release_id) or release_id not in absent,
        active_matches=lambda _: False,
        python_command=("python",),
        npm_command=("npm",),
    )

    coordinator.acquire(plan)

    acquisition_commands = [
        command
        for command in runner.commands
        if "acquire_official_evidence.py" in " ".join(command)
    ]
    assert len(acquisition_commands) == 2
    assert acquisition_commands[0][-4:] == (
        "--source",
        plan.sources[0].source_key,
        "--root",
        str(tmp_path / "evidence" / "releases"),
    )
    assert verified == [source.release_id for source in plan.sources]


def test_prepare_builds_reports_promotes_then_compiles_frontend(tmp_path: Path) -> None:
    runner = RecordingRunner()
    coordinator = BootstrapCoordinator(
        repository_root=tmp_path,
        evidence_root=tmp_path / "evidence",
        runner=runner,
        release_is_valid=lambda _: True,
        active_matches=lambda _: False,
        python_command=("python",),
        npm_command=("npm",),
    )

    result = coordinator.prepare(default_plan())

    assert result.snapshot_id == "snapshot-approved"
    joined = [" ".join(command) for command in runner.commands]
    build_index = next(index for index, value in enumerate(joined) if " build " in f" {value} ")
    report_index = next(
        index
        for index, value in enumerate(joined)
        if "report_snapshot_completeness.py" in value
    )
    promote_index = next(index for index, value in enumerate(joined) if " promote " in f" {value} ")
    assert build_index < report_index < promote_index
    build = runner.commands[build_index]
    assert build.count("--release") == 20
    assert build[build.index("--build-as-of") + 1] == BUILD_AS_OF
    assert build[build.index("--deterministic-seed") + 1] == str(DETERMINISTIC_SEED)
    assert runner.commands[-2:] == [("npm", "ci"), ("npm", "run", "build")]


def test_prepare_never_promotes_failed_completeness(tmp_path: Path) -> None:
    runner = RecordingRunner(completeness_code=2)
    coordinator = BootstrapCoordinator(
        repository_root=tmp_path,
        evidence_root=tmp_path / "evidence",
        runner=runner,
        release_is_valid=lambda _: True,
        active_matches=lambda _: False,
        python_command=("python",),
        npm_command=("npm",),
    )

    with pytest.raises(BootstrapError, match="completeness"):
        coordinator.prepare(default_plan())

    assert not any(" promote " in f" {' '.join(command)} " for command in runner.commands)


def test_prepare_reuses_matching_active_snapshot(tmp_path: Path) -> None:
    runner = RecordingRunner()
    coordinator = BootstrapCoordinator(
        repository_root=tmp_path,
        evidence_root=tmp_path / "evidence",
        runner=runner,
        release_is_valid=lambda _: True,
        active_matches=lambda _: True,
        python_command=("python",),
        npm_command=("npm",),
    )

    result = coordinator.prepare(default_plan())

    assert result.reused_active
    assert not any(
        "build_evidence_snapshot.py" in " ".join(command) and "build" in command
        for command in runner.commands
    )
    assert runner.commands[-2:] == [("npm", "ci"), ("npm", "run", "build")]
