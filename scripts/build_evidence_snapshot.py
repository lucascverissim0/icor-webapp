#!/usr/bin/env python3
"""Build and promote local evidence snapshots through explicit safe roots."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path

from icor.application.snapshot_build import (
    EvidenceLoader,
    SnapshotBuilder,
    SnapshotBuildError,
    SnapshotBuildRequest,
)
from icor.domain.snapshots import SnapshotManifest, SnapshotVersions
from icor.evidence.release_manifests import ManifestError, load_release_manifest
from icor.evidence.serialization import canonical_json_bytes
from icor.infrastructure.release_store import (
    ReleaseAlreadyExistsError,
    ReleaseIntegrityError,
    ReleaseStore,
    StoredRelease,
)
from icor.infrastructure.snapshot_filesystem import SnapshotFilesystem, SnapshotPathError
from icor.infrastructure.snapshot_store import (
    SnapshotPromotionError,
    SnapshotStore,
    SnapshotUnavailableError,
)
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_VERSIONS = SnapshotVersions(*("foundation-v1",) * 8)


class CommandInputError(ValueError):
    """Command arguments do not satisfy the public CLI contract."""


class UnsupportedParserError(ValueError):
    """No application-composition loader is registered for a release parser."""


class SafeArgumentParser(argparse.ArgumentParser):
    """Avoid echoing raw user arguments and paths in parser failures."""

    def error(self, message: str) -> None:
        del message
        raise CommandInputError("invalid command input")


class RegistryEvidenceLoader:
    """Dispatch verified releases to application-supplied, source-neutral loaders."""

    def __init__(self, loaders: Mapping[str, EvidenceLoader]) -> None:
        self._loaders = dict(loaders)

    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        grouped: dict[str, list[StoredRelease]] = defaultdict(list)
        for release in releases:
            grouped[release.manifest.parser_name].append(release)
        missing = set(grouped).difference(self._loaders)
        if missing:
            raise UnsupportedParserError("release parser is not registered")
        for parser_name in sorted(grouped):
            self._loaders[parser_name].load(tuple(grouped[parser_name]), repository)


def _add_root_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-external-root", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(description="Build local immutable evidence snapshots.")
    commands = parser.add_subparsers(
        dest="command", required=True, parser_class=SafeArgumentParser
    )

    stage = commands.add_parser("stage-release")
    _add_root_arguments(stage)
    stage.add_argument("--manifest", type=Path, required=True)
    stage.add_argument("--artifact", type=Path, required=True)

    build = commands.add_parser("build")
    _add_root_arguments(build)
    build.add_argument("--release", action="append", required=True)
    build.add_argument("--build-as-of", required=True)
    build.add_argument("--deterministic-seed", type=int, default=0)

    promote = commands.add_parser("promote")
    _add_root_arguments(promote)
    promote.add_argument("--snapshot", required=True)

    status = commands.add_parser("status")
    _add_root_arguments(status)

    verify = commands.add_parser("verify")
    _add_root_arguments(verify)
    return parser


def _safe_root(path: Path, *, allow_external: bool) -> Path:
    try:
        workspace = WORKSPACE_ROOT.resolve(strict=True)
        absolute = Path(os.path.abspath(path))
        resolved = absolute.resolve(strict=False)
    except OSError as error:
        raise CommandInputError("invalid evidence root") from error
    if not allow_external and not resolved.is_relative_to(workspace):
        raise CommandInputError("external evidence root")
    return absolute


def _build_as_of(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise CommandInputError("invalid build timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise CommandInputError("build timestamp must be UTC")
    return parsed


def _release_store(root: Path) -> ReleaseStore:
    return ReleaseStore(root / "releases")


def _snapshot_payload(manifest: SnapshotManifest, *, state: str) -> dict[str, object]:
    return {
        "database_sha256": manifest.database_sha256,
        "observation_count": manifest.observation_count,
        "published_value_count": manifest.published_value_count,
        "release_ids": manifest.release_ids,
        "snapshot_id": manifest.snapshot_id,
        "state": state,
        "warning_count": len(manifest.warnings),
    }


def _emit(payload: Mapping[str, object]) -> None:
    sys.stdout.write(canonical_json_bytes(payload).decode("utf-8"))


def _reject(code: str, message: str, *, exit_code: int) -> int:
    _emit({"error": {"code": code}, "state": "rejected"})
    sys.stderr.write(f"{message}\n")
    return exit_code


def _stage_release(args: argparse.Namespace, root: Path) -> tuple[int, dict[str, object]]:
    manifest = load_release_manifest(args.manifest)
    stored = _release_store(root).stage(args.artifact, manifest)
    return 0, {
        "accepted_record_count": stored.manifest.accepted_record_count,
        "release_id": stored.release_id,
        "source_id": stored.source_id,
        "state": "staged",
    }


def _build_snapshot(
    args: argparse.Namespace,
    root: Path,
    loader_registry: Mapping[str, EvidenceLoader],
) -> tuple[int, dict[str, object]]:
    release_ids = tuple(sorted(args.release))
    if len(set(release_ids)) != len(release_ids):
        raise CommandInputError("release IDs must be unique")
    store = _release_store(root)
    try:
        releases = tuple(store.verify(release_id) for release_id in release_ids)
    except ReleaseIntegrityError as error:
        raise SnapshotBuildError("staged release failed validation") from error
    parser_names = {release.manifest.parser_name for release in releases}
    if parser_names.difference(loader_registry):
        raise UnsupportedParserError("release parser is not registered")
    request = SnapshotBuildRequest(
        release_ids=release_ids,
        versions=FOUNDATION_VERSIONS,
        deterministic_seed=args.deterministic_seed,
        build_as_of=_build_as_of(args.build_as_of),
    )
    result = SnapshotBuilder(
        root,
        store,
        RegistryEvidenceLoader(loader_registry),
    ).build(request)
    if not result.validation_report.can_promote:
        return 3, {
            "error": {"code": "snapshot_validation_failed"},
            "finding_codes": tuple(
                finding.code for finding in result.validation_report.findings
            ),
            "state": "validation_failed",
        }
    return 0, _snapshot_payload(result.manifest, state="candidate")


def _promote(args: argparse.Namespace, root: Path) -> tuple[int, dict[str, object]]:
    manifest = SnapshotStore(root).promote(args.snapshot)
    payload = _snapshot_payload(manifest, state="promoted")
    payload["active_snapshot_id"] = manifest.snapshot_id
    return 0, payload


def _status(root: Path) -> tuple[int, dict[str, object]]:
    manifest = SnapshotStore(root).active_manifest()
    payload = _snapshot_payload(manifest, state="active")
    payload["active_snapshot_id"] = manifest.snapshot_id
    return 0, payload


def _verify(root: Path) -> tuple[int, dict[str, object]]:
    store = SnapshotStore(root)
    manifest, repository = store.open_active_snapshot()
    payload = _snapshot_payload(manifest, state="verified")
    payload["repository_observation_count"] = len(repository.list_observations())
    payload["repository_published_value_count"] = len(repository.list_published_values())
    return 0, payload


def main(
    argv: Sequence[str] | None = None,
    *,
    loader_registry: Mapping[str, EvidenceLoader] | None = None,
) -> int:
    """Run one CLI command; production composition intentionally has no source loaders."""
    try:
        args = build_parser().parse_args(argv)
        root = _safe_root(args.root, allow_external=args.allow_external_root)
        registry = {} if loader_registry is None else loader_registry
        create_root = args.command in {"stage-release", "build", "promote"}
        if not create_root and not os.path.lexists(root):
            raise SnapshotUnavailableError("no active snapshot is available")
        with SnapshotFilesystem().pin_root(root, create=create_root) as pinned_root:
            if args.command == "stage-release":
                code, payload = _stage_release(args, pinned_root)
            elif args.command == "build":
                code, payload = _build_snapshot(args, pinned_root, registry)
            elif args.command == "promote":
                code, payload = _promote(args, pinned_root)
            elif args.command == "status":
                code, payload = _status(pinned_root)
            else:
                code, payload = _verify(pinned_root)
    except UnsupportedParserError:
        return _reject(
            "unsupported_parser",
            "The requested release parser is not supported.",
            exit_code=2,
        )
    except CommandInputError as error:
        code = "invalid_root" if "root" in str(error) else "invalid_input"
        message = (
            "Evidence root must be contained in the workspace."
            if code == "invalid_root"
            else "Invalid command input."
        )
        return _reject(code, message, exit_code=2)
    except SnapshotPathError:
        return _reject(
            "invalid_root",
            "Evidence root must be contained in the workspace.",
            exit_code=2,
        )
    except SnapshotUnavailableError:
        _emit({"active_snapshot_id": None, "state": "unavailable"})
        sys.stderr.write("No active snapshot is available.\n")
        return 4
    except (ManifestError, ReleaseAlreadyExistsError, ReleaseIntegrityError, OSError, ValueError):
        return _reject("invalid_input", "Release or command input is invalid.", exit_code=2)
    except (SnapshotBuildError, SnapshotPromotionError):
        return _reject("snapshot_validation_failed", "Snapshot validation failed.", exit_code=3)
    except Exception:
        return _reject(
            "operation_failed",
            "Evidence snapshot operation failed.",
            exit_code=3,
        )

    _emit(payload)
    if code == 3:
        sys.stderr.write("Snapshot validation failed.\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
