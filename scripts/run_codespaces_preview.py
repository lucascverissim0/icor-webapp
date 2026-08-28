#!/usr/bin/env python3
"""Start the authenticated preview only after Codespaces prerequisites pass."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from icor.preview.config import ConfigurationError
from icor.preview.runner import RunnerError, server_command, validate_runner


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the authenticated ICOR preview.")
    parser.add_argument("--check", action="store_true", help="Validate without starting")
    parser.add_argument("--asset-root", type=Path, default=Path("web/dist"))
    parser.add_argument(
        "--snapshot-root", type=Path, default=Path("/workspaces/.icor/evidence")
    )
    parser.add_argument(
        "--coverage-db",
        type=Path,
        default=Path("/workspaces/.icor/production-coverage.sqlite3"),
    )
    args = parser.parse_args()
    try:
        validate_runner(
            os.environ,
            asset_root=args.asset_root,
            snapshot_root=args.snapshot_root,
            coverage_db=args.coverage_db,
        )
    except (ConfigurationError, RunnerError):
        print('{"state":"rejected"}')
        return 2
    if args.check:
        print('{"state":"ready"}')
        return 0
    os.environ["ICOR_EVIDENCE_ACTIVE_ROOT"] = str(args.snapshot_root)
    os.environ["ICOR_COVERAGE_DB"] = str(args.coverage_db)
    result = subprocess.run(server_command(), check=False)
    if result.returncode:
        print(json.dumps({"state": "stopped"}, separators=(",", ":")))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())