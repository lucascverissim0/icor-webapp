#!/usr/bin/env python3
"""Acquire checksum-reporting canonical EEA annual aggregate exports."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from icor.evidence.eea_history_acquisition import ANNUAL_RELEASES, acquire_year


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--year", required=True, type=int, choices=sorted(ANNUAL_RELEASES)
    )
    parser.add_argument("--page-size", type=int, default=100_000)
    args = parser.parse_args()
    result = acquire_year(
        ANNUAL_RELEASES[args.year],
        args.destination,
        page_size=args.page_size,
    )
    payload = asdict(result)
    payload["path"] = str(payload["path"])
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
