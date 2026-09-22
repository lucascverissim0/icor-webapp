#!/usr/bin/env python3
"""Derive and promote the pruned, client-scoped snapshot.

`derive` copies a built snapshot, empties the tables no client-release route
reads, vacuums, and re-issues the manifest and validation report through the
same writer and validator every other snapshot goes through. It does not
rebuild: a second canonical build would repeat the four-and-a-quarter-hour
replay to produce rows already known to be identical.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from icor.application.client_snapshot import ClientSnapshotError, derive_client_snapshot
from icor.infrastructure.snapshot_store import SnapshotStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    derive = commands.add_parser("derive")
    source = derive.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", help="a snapshot id under <root>/snapshots")
    source.add_argument("--active", action="store_true", help="the promoted snapshot")

    promote = commands.add_parser("promote")
    promote.add_argument("--snapshot", required=True)
    return parser


def _source_directory(root: Path, args: argparse.Namespace) -> Path:
    if args.active:
        manifest, _ = SnapshotStore(root).open_active_snapshot()
        return root / "snapshots" / manifest.snapshot_id
    for parent in ("snapshots", "candidates"):
        candidate = root / parent / args.snapshot
        if candidate.is_dir():
            return candidate
    raise ClientSnapshotError(f"snapshot is not present: {args.snapshot}")


def _derive(root: Path, args: argparse.Namespace) -> int:
    source = _source_directory(root, args)
    staging = root / "candidates" / ".client-derive"
    if staging.exists():
        shutil.rmtree(staging)
    derived = derive_client_snapshot(source, staging)
    target = root / "candidates" / derived.snapshot_id
    if target.exists():
        shutil.rmtree(staging)
        raise ClientSnapshotError(f"client snapshot already exists: {derived.snapshot_id}")
    staging.rename(target)

    database = target / "evidence.sqlite3"
    original = (source / "evidence.sqlite3").stat().st_size
    pruned = database.stat().st_size
    print(f"source      {source.name}")
    print(f"derived     {derived.snapshot_id}")
    print(f"scope       {derived.scope}")
    print(f"sha256      {derived.database_sha256}")
    print(f"size        {pruned:,} bytes (from {original:,}, {pruned / original:.1%})")
    print(f"observations {derived.observation_count}")
    print(f"\npromote with:\n  build_client_snapshot.py --root {root} "
          f"promote --snapshot {derived.snapshot_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "derive":
            return _derive(root, args)
        SnapshotStore(root).promote(args.snapshot)
        print(f"promoted {args.snapshot}")
        return 0
    except (ClientSnapshotError, OSError, RuntimeError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
