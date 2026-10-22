#!/usr/bin/env python3
"""Stream the Czech vehicle register into counts-only passenger-car aggregates."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from icor.evidence.cz_fleet_acquisition import acquire_fleet


def _report(rows: int, received: int) -> None:
    print(f"{rows:,} rows, {received / 1024**2:,.0f} MiB received", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        required=True,
        type=Path,
        help="the strict aggregate; {date} is replaced by the snapshot date (YYYYMMDD)",
    )
    args = parser.parse_args()
    result = acquire_fleet(args.destination, progress=_report)
    print(json.dumps(asdict(result), sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
