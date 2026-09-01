"""Idempotent orchestration for the remote Codespaces evidence preview."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from icor.evidence.acquisition import OFFICIAL_SOURCES
from icor.infrastructure.release_store import ReleaseIntegrityError, ReleaseStore
from icor.infrastructure.snapshot_store import SnapshotStore, SnapshotUnavailableError

BUILD_AS_OF = "2026-08-27T12:00:00+00:00"
DETERMINISTIC_SEED = 20260827
MINIMUM_FREE_BYTES = 20 * 1024**3
REQUIRED_UV_VERSION = "0.11.3"
SOURCE_KEYS = (
    *(f"eea-{year}-final" for year in range(2010, 2025)),
    "kba-fz10-2024",
    "uk-veh0160-gb",
    "uk-veh0120-gb",
    "uk-veh0124-am",
    "uk-veh0124-nz",
)


class BootstrapError(RuntimeError):
    """The remote preview cannot advance without violating its safety contract."""


@dataclass(frozen=True, slots=True)
class PlannedSource:
    source_key: str
    release_id: str


@dataclass(frozen=True, slots=True)
class BootstrapPlan:
    sources: tuple[PlannedSource, ...]
    build_as_of: str = BUILD_AS_OF
    deterministic_seed: int = DETERMINISTIC_SEED

    @property
    def release_ids(self) -> tuple[str, ...]:
        return tuple(sorted(source.release_id for source in self.sources))


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    codespaces: bool
    python_version: tuple[int, int, int]
    node_version: str
    npm_version: str
    uv_version: str
    free_bytes: int


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def run(self, command: tuple[str, ...], *, cwd: Path) -> CommandResult: ...


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    snapshot_id: str
    reused_active: bool


def default_plan() -> BootstrapPlan:
    return BootstrapPlan(
        tuple(
            PlannedSource(source_key, OFFICIAL_SOURCES[source_key].release_id)
            for source_key in SOURCE_KEYS
        )
    )


def validate_environment(
    report: EnvironmentReport,
    *,
    repository_root: Path,
    workspaces_root: Path,
    evidence_root: Path,
    require_build_capacity: bool = True,
) -> None:
    if not report.codespaces:
        raise BootstrapError("bootstrap requires GitHub Codespaces")
    if report.python_version[:2] != (3, 12):
        raise BootstrapError("Python 3.12 is required")
    if not _supported_node(report.node_version):
        raise BootstrapError("Node runtime is unsupported")
    if not _version_tuple(report.npm_version):
        raise BootstrapError("npm is unavailable")
    if report.uv_version != REQUIRED_UV_VERSION:
        raise BootstrapError("uv version is unsupported")
    if require_build_capacity and report.free_bytes < MINIMUM_FREE_BYTES:
        raise BootstrapError("available disk is insufficient")
    try:
        workspace = workspaces_root.resolve(strict=True)
        repository = repository_root.resolve(strict=True)
        evidence = evidence_root.resolve(strict=False)
    except OSError as error:
        raise BootstrapError("workspace paths are unavailable") from error
    if not repository.is_relative_to(workspace):
        raise BootstrapError("repository is outside the workspaces root")
    if not evidence.is_relative_to(workspace):
        raise BootstrapError("evidence root is outside the workspaces root")
    for required in (
        repository / "pyproject.toml",
        repository / "uv.lock",
        repository / "web" / "package-lock.json",
    ):
        if not required.is_file():
            raise BootstrapError("locked dependency inputs are unavailable")


class BootstrapCoordinator:
    """Run reviewed public CLIs in a validation-gated order."""

    def __init__(
        self,
        *,
        repository_root: Path,
        evidence_root: Path,
        runner: CommandRunner,
        release_is_valid: Callable[[str], bool],
        active_matches: Callable[[BootstrapPlan], bool | str],
        python_command: tuple[str, ...] = (sys.executable,),
        npm_command: tuple[str, ...] = ("npm",),
    ) -> None:
        self.repository_root = repository_root
        self.evidence_root = evidence_root
        self.runner = runner
        self.release_is_valid = release_is_valid
        self.active_matches = active_matches
        self.python_command = python_command
        self.npm_command = npm_command

    def acquire(self, plan: BootstrapPlan) -> None:
        validity = {
            source.release_id: self.release_is_valid(source.release_id)
            for source in plan.sources
        }
        if all(validity.values()):
            return
        downloads_root = self._validated_downloads_root()
        for source in plan.sources:
            if validity[source.release_id]:
                continue
            official = OFFICIAL_SOURCES[source.source_key]
            artifact: Path | None = None
            if not official.direct_download:
                artifact = (
                    downloads_root
                    / f"{source.source_key}{official.suffix}"
                )
                self._checked(
                    (
                        *self.python_command,
                        str(self.repository_root / "scripts" / "acquire_eea_history.py"),
                        "--destination",
                        str(artifact),
                        "--year",
                        str(official.coverage_start.year),
                    ),
                    label="official EEA history acquisition",
                )
            artifact_args = () if artifact is None else ("--artifact", str(artifact))
            self._checked(
                (
                    *self.python_command,
                    str(self.repository_root / "scripts" / "acquire_official_evidence.py"),
                    "--source",
                    source.source_key,
                    "--root",
                    str(self.evidence_root),
                    *artifact_args,
                ),
                label="official source acquisition",
            )

    def _validated_downloads_root(self) -> Path:
        downloads = self.evidence_root / "downloads"
        try:
            self.evidence_root.mkdir(parents=True, exist_ok=True)
            evidence = self.evidence_root.resolve(strict=True)
            if downloads.is_symlink():
                raise BootstrapError("downloads directory cannot be a symlink")
            downloads.mkdir(exist_ok=True)
            resolved = downloads.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise BootstrapError("downloads directory is unavailable") from error
        if not resolved.is_dir() or not resolved.is_relative_to(evidence):
            raise BootstrapError("downloads directory is outside the evidence root")
        return resolved

    def build(self, plan: BootstrapPlan) -> str:
        releases = tuple(
            part for release_id in plan.release_ids for part in ("--release", release_id)
        )
        result = self._checked(
            (
                *self.python_command,
                str(self.repository_root / "scripts" / "build_evidence_snapshot.py"),
                "build",
                "--root",
                str(self.evidence_root),
                "--allow-external-root",
                *releases,
                "--build-as-of",
                plan.build_as_of,
                "--deterministic-seed",
                str(plan.deterministic_seed),
            ),
            label="snapshot build",
        )
        payload = _json_object(result.stdout, "snapshot build")
        snapshot_id = payload.get("snapshot_id")
        if payload.get("state") != "candidate" or not isinstance(snapshot_id, str):
            raise BootstrapError("snapshot build returned an invalid result")
        return snapshot_id

    def validate_candidate(self, plan: BootstrapPlan, snapshot_id: str) -> None:
        result = self._checked(
            (
                *self.python_command,
                str(self.repository_root / "scripts" / "report_snapshot_completeness.py"),
                "--candidate",
                str(self.evidence_root / "candidates" / snapshot_id),
            ),
            label="snapshot completeness",
        )
        payload = _json_object(result.stdout, "snapshot completeness")
        if (
            payload.get("snapshot_id") != snapshot_id
            or payload.get("release_ids") != list(plan.release_ids)
            or payload.get("warnings") != []
        ):
            raise BootstrapError("snapshot completeness result is invalid")

    def promote(self, snapshot_id: str) -> None:
        self._checked(
            (
                *self.python_command,
                str(self.repository_root / "scripts" / "build_evidence_snapshot.py"),
                "promote",
                "--root",
                str(self.evidence_root),
                "--allow-external-root",
                "--snapshot",
                snapshot_id,
            ),
            label="snapshot promotion",
        )

    def compile_frontend(self) -> None:
        web = self.repository_root / "web"
        self._checked((*self.npm_command, "ci"), label="frontend dependency install", cwd=web)
        self._checked((*self.npm_command, "run", "build"), label="frontend build", cwd=web)

    def reusable_active(self, plan: BootstrapPlan) -> bool | str:
        if not all(self.release_is_valid(release_id) for release_id in plan.release_ids):
            return False
        return self.active_matches(plan)

    def prepare(
        self,
        plan: BootstrapPlan,
        *,
        reusable_active: bool | str | None = None,
    ) -> BootstrapResult:
        active = (
            self.reusable_active(plan)
            if reusable_active is None
            else reusable_active
        )
        if not active:
            self.acquire(plan)
            active = self.active_matches(plan)
        if active:
            self.compile_frontend()
            return BootstrapResult(
                snapshot_id=active if isinstance(active, str) else "active",
                reused_active=True,
            )
        snapshot_id = self.build(plan)
        self.validate_candidate(plan, snapshot_id)
        self.promote(snapshot_id)
        self.compile_frontend()
        return BootstrapResult(snapshot_id=snapshot_id, reused_active=False)

    def _checked(
        self,
        command: tuple[str, ...],
        *,
        label: str,
        cwd: Path | None = None,
    ) -> CommandResult:
        result = self.runner.run(command, cwd=cwd or self.repository_root)
        if result.returncode != 0:
            raise BootstrapError(f"{label} failed")
        return result


class SubprocessRunner:
    def run(self, command: tuple[str, ...], *, cwd: Path) -> CommandResult:
        completed = subprocess.run(
            command, cwd=cwd, check=False, capture_output=True, text=True
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def probe_environment(workspaces_root: Path) -> EnvironmentReport:
    return EnvironmentReport(
        codespaces=os.getenv("CODESPACES", "").casefold() == "true",
        python_version=sys.version_info[:3],
        node_version=_tool_version(("node", "--version")),
        npm_version=_tool_version(("npm", "--version")),
        uv_version=_tool_version(("uv", "--version")).removeprefix("uv ").partition(" ")[0],
        free_bytes=shutil.disk_usage(workspaces_root).free,
    )


def release_is_valid(evidence_root: Path, release_id: str) -> bool:
    try:
        ReleaseStore(evidence_root / "releases").verify(release_id)
    except FileNotFoundError:
        return False
    except (OSError, ReleaseIntegrityError) as error:
        raise BootstrapError("staged release failed integrity verification") from error
    return True


def matching_active(evidence_root: Path, plan: BootstrapPlan) -> str | bool:
    try:
        manifest = SnapshotStore(evidence_root).active_manifest()
    except SnapshotUnavailableError:
        return False
    if (
        manifest.release_ids == plan.release_ids
        and manifest.built_at.isoformat() == plan.build_as_of
        and manifest.deterministic_seed == plan.deterministic_seed
        and not manifest.versions.generation_registry.endswith("-v0")
        and not manifest.versions.generation_resolver.endswith("-v0")
    ):
        return manifest.snapshot_id
    return False


def _json_object(value: str, label: str) -> dict[str, object]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise BootstrapError(f"{label} returned an invalid result") from error
    if not isinstance(payload, dict):
        raise BootstrapError(f"{label} returned an invalid result")
    return payload


def _tool_version(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise BootstrapError("required tool is unavailable") from error
    return completed.stdout.strip().removeprefix("v")


def _version_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return ()


def _supported_node(value: str) -> bool:
    version = _version_tuple(value.removeprefix("v"))
    if len(version) < 3:
        return False
    major, minor, patch = version[:3]
    return (
        (major == 22 and (minor, patch) >= (22, 2))
        or (major == 24 and (minor, patch) >= (15, 0))
        or major >= 26
    )
