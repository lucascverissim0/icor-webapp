#!/usr/bin/env python3
"""Rebuild a candidate from exactly the releases the active snapshot was built from.

The release list is read from the active manifest rather than retyped, so a
rebuild cannot silently widen or narrow its own inputs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 3:
        print(
            "usage: rebuild_active_snapshot.py <evidence-root> <build-as-of> <seed>",
            file=sys.stderr,
        )
        return 2
    evidence_root, build_as_of, seed = arguments
    root = (ROOT / evidence_root).resolve()
    active = json.loads((root / "active.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (root / "snapshots" / active["snapshot_id"] / "snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    releases = tuple(manifest["release_ids"])
    print(f"rebuilding from {active['snapshot_id']} with {len(releases)} releases")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_evidence_snapshot.py"),
        "build",
        "--root",
        str(root),
        "--build-as-of",
        build_as_of,
        "--deterministic-seed",
        seed,
    ]
    for release_id in releases:
        command.extend(("--release", release_id))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
