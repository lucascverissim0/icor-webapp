#!/usr/bin/env python3
"""Acquire Estonia's registered passenger-car aggregate from Transpordiamet's open data."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from icor.evidence.ee_fleet_acquisition import (
    MAX_SOURCE_BYTES,
    SOURCE_URL,
    acquire_fleet,
    latest_distribution,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        required=True,
        type=Path,
        help="the strict aggregate (REGISTREERITUD); a counts-only profile is written next to it",
    )
    parser.add_argument(
        "--variant-destination",
        type=Path,
        help="the aggregate that also counts PEATATUD; default <destination>-with-suspended.csv",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="use the newest M1+M1G snapshot the portal lists instead of the pinned one",
    )
    parser.add_argument("--max-bytes", type=int, default=MAX_SOURCE_BYTES)
    parser.add_argument(
        "--with-suspended",
        action="store_true",
        help=(
            "write the aggregate that also counts PEATATUD, as Eurostat counts Estonia's "
            "fleet, to --destination (the pinned release) and the strict one beside it"
        ),
    )
    args = parser.parse_args()
    url = latest_distribution().url if args.latest else SOURCE_URL
    strict, variant = args.destination, args.variant_destination
    if args.with_suspended:
        strict = args.destination.with_name(f"{args.destination.stem}-strict.csv")
        variant = args.destination
    result = acquire_fleet(
        strict,
        variant_destination=variant,
        url=url,
        max_bytes=args.max_bytes,
    )
    payload = asdict(result)
    for name in ("path", "variant_path", "profile_path"):
        payload[name] = str(payload[name])
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
