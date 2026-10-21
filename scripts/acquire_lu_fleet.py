#!/usr/bin/env python3
"""Acquire the counts-only Luxembourg passenger-car stock aggregate (SNCA, CC0)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from icor.evidence.lu_fleet_acquisition import acquire_fleet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--url",
        help="pin one monthly XML export on download.data.public.lu; default: the newest one",
    )
    args = parser.parse_args()
    result = acquire_fleet(args.destination, url=args.url)
    payload = asdict(result)
    payload["path"] = str(payload["path"])
    payload["profile_path"] = str(payload["profile_path"])
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
