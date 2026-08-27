#!/usr/bin/env python3
"""Acquire or stage one checksum-pinned official evidence release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from icor.evidence.acquisition import OFFICIAL_SOURCES, acquire


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=sorted(OFFICIAL_SOURCES))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--artifact", type=Path, help="Already downloaded exact official artifact")
    args = parser.parse_args()
    stored = acquire(OFFICIAL_SOURCES[args.source], args.root, artifact=args.artifact)
    print(
        json.dumps(
            {"release_id": stored.release_id, "source_id": stored.source_id, "state": "staged"},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
