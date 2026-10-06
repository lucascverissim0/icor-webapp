#!/usr/bin/env python3
"""Acquire the checksum-reporting RDW registered passenger-car fleet aggregate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from icor.evidence.rdw_fleet_acquisition import acquire_fleet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--page-size", type=int, default=50_000)
    args = parser.parse_args()
    result = acquire_fleet(args.destination, page_size=args.page_size)
    payload = asdict(result)
    payload["path"] = str(payload["path"])
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
