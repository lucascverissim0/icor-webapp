#!/usr/bin/env python3
"""Acquire the checksum-reporting DGT passenger-car stock aggregate for Spain."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from icor.evidence.dgt_fleet_acquisition import acquire_fleet, download_archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--archive",
        type=Path,
        help="a previously downloaded pinned ZIP; it is kept, while a downloaded one is deleted",
    )
    args = parser.parse_args()
    archive = args.archive
    downloaded = archive is None
    if downloaded:
        archive = download_archive(args.destination.with_name(args.destination.name + ".zip"))
    try:
        result = acquire_fleet(args.destination, archive)
    finally:
        if downloaded and archive.exists():
            archive.unlink()
    payload = asdict(result)
    payload["path"] = str(payload["path"])
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
