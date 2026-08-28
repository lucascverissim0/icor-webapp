#!/usr/bin/env python3
"""Validate or prepare the ICOR Codespaces evidence preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from icor.infrastructure.release_store import ReleaseIntegrityError, ReleaseStore
from icor.infrastructure.snapshot_store import SnapshotStore, SnapshotUnavailableError
from icor.preview.bootstrap import (
    BootstrapCoordinator,
    BootstrapError,
    SubprocessRunner,
    default_plan,
    probe_environment,
    validate_environment,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap the authenticated ICOR Codespaces preview."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Validate without mutation")
    mode.add_argument("--acquire", action="store_true", help="Acquire missing releases")
    mode.add_argument("--build", action="store_true", help="Build and validate a candidate")
    mode.add_argument("--promote", action="store_true", help="Promote a validated candidate")
    mode.add_argument("--prepare", action="store_true", help="Build the complete preview")
    parser.add_argument("--snapshot", help="Candidate snapshot ID for --promote")
    parser.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--workspaces-root", type=Path, default=Path("/workspaces"))
    parser.add_argument("--evidence-root", type=Path, default=Path("/workspaces/.icor/evidence"))
    args = parser.parse_args()
    try:
        validate_environment(
            probe_environment(args.workspaces_root),
            repository_root=args.repository_root,
            workspaces_root=args.workspaces_root,
            evidence_root=args.evidence_root,
        )
        plan = default_plan()
        if args.check:
            payload: dict[str, object] = {"release_count": 20, "state": "ready"}
        else:
            coordinator = _coordinator(args.repository_root, args.evidence_root)
            if args.acquire:
                coordinator.acquire(plan)
                payload = {"release_count": 20, "state": "acquired"}
            elif args.build:
                snapshot_id = coordinator.build(plan)
                coordinator.validate_candidate(plan, snapshot_id)
                payload = {"snapshot_id": snapshot_id, "state": "validated"}
            elif args.promote:
                if not args.snapshot:
                    raise BootstrapError("candidate snapshot ID is required")
                coordinator.promote(args.snapshot)
                payload = {"snapshot_id": args.snapshot, "state": "promoted"}
            else:
                result = coordinator.prepare(plan)
                payload = {
                    "release_count": len(plan.release_ids),
                    "reused": result.reused_active,
                    "snapshot_id": result.snapshot_id,
                    "start_command": "python scripts/run_codespaces_preview.py",
                    "state": "prepared",
                }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    except BootstrapError:
        print('{"state":"rejected"}')
        return 2
    return 0


def _coordinator(repository_root: Path, evidence_root: Path) -> BootstrapCoordinator:
    release_store = ReleaseStore(evidence_root / "releases")
    snapshot_store = SnapshotStore(evidence_root)

    def release_is_valid(release_id: str) -> bool:
        try:
            release_store.verify(release_id)
        except FileNotFoundError:
            return False
        except ReleaseIntegrityError as error:
            raise BootstrapError("stored release validation failed") from error
        return True

    def active_matches(plan) -> bool:  # type: ignore[no-untyped-def]
        try:
            manifest = snapshot_store.active_manifest()
        except SnapshotUnavailableError:
            return False
        return (
            manifest.release_ids == plan.release_ids
            and manifest.built_at.isoformat() == plan.build_as_of
            and manifest.deterministic_seed == plan.deterministic_seed
        )

    return BootstrapCoordinator(
        repository_root=repository_root,
        evidence_root=evidence_root,
        runner=SubprocessRunner(),
        release_is_valid=release_is_valid,
        active_matches=active_matches,
    )


if __name__ == "__main__":
    raise SystemExit(main())
