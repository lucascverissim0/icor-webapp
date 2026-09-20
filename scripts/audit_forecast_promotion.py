#!/usr/bin/env python3
"""Report whether the active snapshot can support promotion-grade validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from icor.forecasting.snapshot_readiness import audit_snapshot_readiness
from icor.infrastructure.snapshot_store import SnapshotStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the active snapshot for leakage-safe forecast promotion."
    )
    parser.add_argument("--root", type=Path, default=Path(".local/evidence"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest, repository = SnapshotStore(args.root).open_active_snapshot()
    report = audit_snapshot_readiness(repository.path, manifest.snapshot_id)
    print(json.dumps(report.as_dict(), sort_keys=True))
    return 0 if report.ready else 3


if __name__ == "__main__":
    raise SystemExit(main())
